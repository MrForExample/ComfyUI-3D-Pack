import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nerfacc

import comfy.utils

from pytorch_msssim import SSIM, MS_SSIM

from kiui.op import safe_normalize
from kiui.cam import orbit_camera
from kiui.nn import MLP, trunc_exp

from shared_utils.image_utils import prepare_torch_img

class InstantNGP(nn.Module):
    def __init__(self, resolution=128, device="cuda"):
        super().__init__()
        from kiui.gridencoder import GridEncoder
        
        self.device = torch.device(device)
        self.ref_size_H = resolution
        self.ref_size_W = resolution
        
        self.render_step_size = 5e-3
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)
        self.estimator = nerfacc.OccGridEstimator(roi_aabb=self.aabb, resolution=64, levels=1)

        self.encoder_density = GridEncoder(num_levels=12) # VMEncoder(output_dim=16, mode='sum')
        self.encoder = GridEncoder(num_levels=12)
        self.mlp_density = MLP(self.encoder_density.output_dim, 1, 32, 2, bias=False)
        self.mlp = MLP(self.encoder.output_dim, 3, 32, 2, bias=False)
        
    def get_rays(self, pose, h, w, fovy, opengl=True):

        x, y = torch.meshgrid(
            torch.arange(w, device=pose.device),
            torch.arange(h, device=pose.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        cx = w * 0.5
        cy = h * 0.5

        focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cx + 0.5) / focal,
                    (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if opengl else 1.0),
        )  # [hw, 3]

        rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
        rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

        rays_o = rays_o.view(h, w, 3)
        rays_d = safe_normalize(rays_d).view(h, w, 3)

        return rays_o, rays_d
    
    def get_color(self, xs):
        return torch.sigmoid(self.mlp(self.encoder(xs.to(self.device))))
    
    def get_density(self, xs):
        # xs: [..., 3]
        xs = xs.to(self.device)
        prefix = xs.shape[:-1]
        xs = xs.view(-1, 3)
        feats = self.encoder_density(xs)
        density = trunc_exp(self.mlp_density(feats))
        density = density.view(*prefix, 1)
        return density
    
    def prepare_training(self, reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy):
        self.ref_imgs_num = len(reference_images)

        self.all_ref_cam_poses = reference_orbit_camera_poses
        self.ref_cam_fovy = reference_orbit_camera_fovy
        
        # prepare reference images and masks
        ref_imgs_torch_list = []
        ref_masks_torch_list = []
        for i in range(self.ref_imgs_num):
            ref_imgs_torch_list.append(prepare_torch_img(reference_images[i].unsqueeze(0), self.ref_size_H, self.ref_size_W, self.device))
            ref_masks_torch_list.append(prepare_torch_img(reference_masks[i].unsqueeze(2).unsqueeze(0), self.ref_size_H, self.ref_size_W, self.device))
            
        self.ref_imgs_torch = torch.cat(ref_imgs_torch_list, dim=0) # [N, 3, H, W]
        self.ref_masks_torch = torch.cat(ref_masks_torch_list, dim=0).squeeze(1) # [N, H, W]
        
    def render_nerf(self, pose, bg_color=1):
        
        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        
        # get rays
        rays_o, rays_d = self.get_rays(pose, self.ref_size_H, self.ref_size_W, self.ref_cam_fovy)
        hw = rays_o.shape[0] * rays_o.shape[1]
        rays_o = rays_o.view(hw, 3)
        rays_d = rays_d.view(hw, 3)
        
        # update occ grid
        if self.training:
            def occ_eval_fn(xs):
                sigmas = self.get_density(xs)
                return self.render_step_size * sigmas
            
            self.estimator.update_every_n_steps(self.render_step, occ_eval_fn=occ_eval_fn, occ_thre=0.01, n=8)
            self.render_step += 1

        # render
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            xs = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.get_density(xs)
            return sigmas.squeeze(-1)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=0.01,
                far_plane=100,
                render_step_size=self.render_step_size,
                stratified=self.training,
                cone_angle=0,
            )

        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        xs = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas = self.get_density(xs).squeeze(-1)
        rgbs = torch.sigmoid(self.mlp(self.encoder(xs)))

        n_rays=rays_o.shape[0]
        weights, t, alphas = nerfacc.render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=n_rays)
        color = nerfacc.accumulate_along_rays(weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays)
        alpha = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)

        color = color + (1.0 - alpha) * bg_color

        color = color.view(self.ref_size_H, self.ref_size_W, 3).clamp(0, 1).permute(2, 0, 1).contiguous()
        alpha = alpha.view(self.ref_size_H, self.ref_size_W).clamp(0, 1).contiguous()
        
        return color, alpha

    def fit_nerf(self, iters=512, bg_color=1):

        optimizer = torch.optim.Adam([
            {'params': self.encoder_density.parameters(), 'lr': 1e-2},
            {'params': self.encoder.parameters(), 'lr': 1e-2},
            {'params': self.mlp_density.parameters(), 'lr': 1e-3},
            {'params': self.mlp.parameters(), 'lr': 1e-3},
        ])

        print(f"[INFO] fitting nerf...")
        self.render_step = 0
        
        ref_imgs_num_minus_1 = self.ref_imgs_num-1
        
        comfy_pbar = comfy.utils.ProgressBar(iters)
        pbar = tqdm.trange(iters)
        for step in pbar:
            
            i = random.randint(0, ref_imgs_num_minus_1)

            radius, elevation, azimuth, center_X, center_Y, center_Z = self.all_ref_cam_poses[i]
            
            orbit_target = np.array([center_X, center_Y, center_Z], dtype=np.float32)
            pose = orbit_camera(elevation, azimuth, radius, target=orbit_target)
            
            image_gt = self.ref_imgs_torch[i]   # [3, H, W]
            alpha_gt = self.ref_masks_torch[i]  # [H, W]
            image_pred, alpha_pred = self.render_nerf(pose, bg_color)

            # if i % 200 == 0:
            #     kiui.vis.plot_image(image_gt, alpha_gt, image_pred, alpha_pred)
            
            loss_mse = F.mse_loss(image_pred, image_gt) + 0.1 * F.mse_loss(alpha_pred, alpha_gt)
            loss = loss_mse #+ 0.1 * self.encoder_density.tv_loss() #+ 0.0001 * self.encoder_density.density_loss()
            #loss += self.lambda_ssim * (1 - self.ms_ssim_loss(image_gt, image_pred))

            loss.backward()
            self.encoder_density.grad_total_variation(1e-8)
        
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"NeRF Fitting Loss = {loss_mse.item():.6f}")
            comfy_pbar.update_absolute(step + 1)
            
        torch.cuda.synchronize()
        
        print(f"[INFO] finished fitting nerf!")