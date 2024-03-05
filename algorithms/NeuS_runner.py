import os
import math
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
import tqdm
import comfy.utils
from pyhocon import ConfigFactory

from NeuS.models.dataset_mvdiff import Dataset
from NeuS.models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from NeuS.models.renderer import NeuSRenderer

from ..mesh_processer.mesh import Mesh


def ranking_loss(error, penalize_ratio=0.7, type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[: int(penalize_ratio * indices.shape[0])])
    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)

class NeuSParams:
    def __init__(
        self,
        end_iter = 1000, # longer time, better result. 1w will be ok for most cases
        batch_size = 512,
        learning_rate = 5e-4,
        learning_rate_alpha = 0.05,
        color_weight = 1.0,
        igr_weight = 0.1,
        mask_weight = 1.0,
        normal_weight = 1.0,
        sparse_weight = 0.1,
        warm_up_end = 500,
        anneal_end = 0,
        use_white_bkgd = True,
    ):
        self.end_iter = end_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_alpha = learning_rate_alpha
        self.color_weight = color_weight
        self.igr_weight = igr_weight
        self.mask_weight = mask_weight
        self.normal_weight = normal_weight
        self.sparse_weight = sparse_weight
        self.warm_up_end = warm_up_end
        self.anneal_end = anneal_end
        self.use_white_bkgd = use_white_bkgd
        

class NeuSRunner:
    def __init__(self, mv_images, mv_masks, mv_normals, conf_path, cam_pose_dir, num_views, NeuS_params=None, mode='train', is_continue=False, device='cuda'):
        self.device = torch.device(device)
        
        if NeuS_params is None:
            NeuS_params = NeuSParams()

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        
        self.base_exp_dir = self.conf['general.base_exp_dir']

        # Training parameters
        self.end_iter = NeuS_params.end_iter
        self.batch_size = NeuS_params.batch_size
        self.learning_rate = NeuS_params.learning_rate
        self.learning_rate_alpha = NeuS_params.learning_rate_alpha
        self.color_weight = NeuS_params.color_weight
        self.igr_weight = NeuS_params.igr_weight
        self.mask_weight = NeuS_params.mask_weight
        self.normal_weight = NeuS_params.normal_weight
        self.sparse_weight = NeuS_params.sparse_weight
        self.warm_up_end = NeuS_params.warm_up_end
        self.anneal_end = NeuS_params.anneal_end
        self.use_white_bkgd = NeuS_params.use_white_bkgd
        
        # Validation parameters
        self.validate_resolution_level = self.conf.get_int('validate.validate_resolution_level')
        self.save_freq = self.conf.get_int('validate.save_freq')
        self.report_freq = self.conf.get_int('validate.report_freq')
        self.val_freq = self.conf.get_int('validate.val_freq')
        self.val_mesh_freq = self.conf.get_int('validate.val_mesh_freq')
        
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []

        # Networks
        params_to_train_slow = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        # params_to_train += list(self.nerf_outside.parameters())
        params_to_train_slow += list(self.sdf_network.parameters())
        params_to_train_slow += list(self.deviation_network.parameters())
        # params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(
            [{'params': params_to_train_slow}, {'params': self.color_network.parameters(), 'lr': self.learning_rate * 2}], lr=self.learning_rate
        )

        self.renderer = NeuSRenderer(
            self.nerf_outside, self.sdf_network, self.deviation_network, self.color_network, **self.conf['model.neus_renderer']
        )
        
        # Prepare dataset
        self.dataset = Dataset(mv_images, mv_masks, mv_normals, self.conf['dataset'], cam_pose_dir, num_views)
        self.iter_step = 1

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            self.load_checkpoint(latest_model_name)

    def train(self):
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        
        comfy_pbar = comfy.utils.ProgressBar(self.end_iter)
        
        for epoch in tqdm.trange(res_step):
            data = self.dataset.get_batch_of_random_rays(self.batch_size).to(self.device)

            rays_o, rays_d, true_rgb, mask, true_normal, cosines = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:9],
                data[:, 9:10],
                data[:, 10:13],
                data[:, 13:],
            )
            
            # near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            near, far = self.dataset.get_near_far()

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            cosines[cosines > -0.1] = 0
            mask = ((mask > 0) & (cosines < -0.1)).to(torch.float32)
            #mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o, rays_d, near, far, background_rgb=background_rgb, cos_anneal_ratio=self.get_cos_anneal_ratio()
            )

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            #cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            #weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            # color_error = (color_fine - true_rgb) * mask
            # color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum

            color_errors = (color_fine - true_rgb).abs().sum(dim=1)
            color_fine_loss = ranking_loss(color_errors[mask[:, 0] > 0])

            #psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            # pdb.set_trace()
            mask_errors = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask, reduction='none')
            mask_loss = ranking_loss(mask_errors[:, 0], penalize_ratio=0.8)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            # calculate normal loss
            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            if feasible('inside_sphere'):
                normals = normals * render_out['inside_sphere'][..., None]
            normals = normals.sum(dim=1)

            # pdb.set_trace()
            normal_errors = 1 - F.cosine_similarity(normals, true_normal, dim=1)
            # normal_error = normal_error * mask[:, 0]
            # normal_loss = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
            normal_errors = normal_errors * torch.exp(cosines.abs()[:, 0]) / torch.exp(cosines.abs()).sum()
            normal_loss = ranking_loss(normal_errors[mask[:, 0] > 0], penalize_ratio=0.9, type='sum')

            sparse_loss = render_out['sparse_loss']

            loss = (
                color_fine_loss * self.color_weight
                + eikonal_loss * self.igr_weight
                + sparse_loss * self.sparse_weight
                + mask_loss * self.mask_weight
                + normal_loss * self.normal_weight
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.iter_step % self.report_freq == 0:
                print(
                    'iter:{:8>d} loss = {:4>f} color_ls = {:4>f} eik_ls = {:4>f} normal_ls = {:4>f} mask_ls = {:4>f} sparse_ls = {:4>f} lr={:5>f}'.format(
                        self.iter_step,
                        loss,
                        color_fine_loss,
                        eikonal_loss,
                        normal_loss,
                        mask_loss,
                        sparse_loss,
                        self.optimizer.param_groups[0]['lr'],
                    )
                )
                print('iter:{:8>d} s_val = {:4>f}'.format(self.iter_step, s_val.mean()))

            if self.iter_step % self.val_mesh_freq == 0:
                self.extract_mesh(resolution=256)

            self.update_learning_rate()

            self.iter_step += 1

            if self.iter_step % self.val_freq == 0:
                self.validate_image(idx=0)
                self.validate_image(idx=1)
                self.validate_image(idx=2)
                self.validate_image(idx=3)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                
            comfy_pbar.update_absolute(self.iter_step)           

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
            
    def extract_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles, vertex_colors = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        print(f"vertex_colors: {vertex_colors[123]}, {vertex_colors[213]}, {vertex_colors[321]}")

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
        mesh = Mesh.load_trimesh(given_mesh=mesh)
        
        # export as glb
        # os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        
        return mesh

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(idx, resolution_level=resolution_level),
                            self.dataset.mask_at(idx, resolution_level=resolution_level),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                    np.concatenate([normal_img[..., i], self.dataset.normal_cam_at(idx, resolution_level=resolution_level)])[:, :, ::-1],
                )
            if len(out_mask) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}_mask.png'.format(self.iter_step, i, idx)), mask_map[..., i])

    def save_maps(self, idx, img_idx, resolution_level=1):
        view_types = ['front', 'back', 'left', 'right']
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_mask = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('weight_sum'):
                out_mask.append(render_out['weight_sum'].detach().clip(0, 1).cpu().numpy())

            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        mask_map = None
        if len(out_mask) > 0:
            mask_map = (np.concatenate(out_mask, axis=0).reshape([H, W, 1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            world_normal_img = (normal_img.reshape([H, W, 3]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'coarse_maps'), exist_ok=True)
        img_rgba = np.concatenate([img_fine[:, :, ::-1], mask_map], axis=-1)
        normal_rgba = np.concatenate([world_normal_img[:, :, ::-1], mask_map], axis=-1)

        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_mlp_%03d_%s.png" % (img_idx, view_types[idx])), img_rgba)
        cv.imwrite(os.path.join(self.base_exp_dir, 'coarse_maps', "normals_grad_%03d_%s.png" % (img_idx, view_types[idx])), normal_rgba)

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.dataset.get_near_far()
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb
            )

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0, img_idx_1, np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir, '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)), fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()
