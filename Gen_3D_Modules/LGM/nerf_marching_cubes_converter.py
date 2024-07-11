import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.utils

import nerfacc
import nvdiffrast.torch as dr

from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.mesh_utils import laplacian_smooth_loss, normal_consistency
from kiui.op import inverse_sigmoid
from kiui.cam import orbit_camera, get_perspective
from kiui.nn import MLP, trunc_exp

from mesh_processer.mesh import Mesh
from mesh_processer.mesh_utils import marching_cubes_density_to_mesh, color_func_to_albedo
from LGM.core.options import Options
from LGM.core.gs import GaussianRenderer
from LGM.core.utils import get_rays

# Triple renderer of gaussians, gaussian, and diso mesh.
# gaussian --> nerf --> mesh
class GSConverterNeRFMarchingCubes(nn.Module):
    def __init__(self, opt: Options, gs_ply):
        super().__init__()
        from kiui.gridencoder import GridEncoder

        self.opt = opt
        self.device = torch.device("cuda")

        # gs renderer
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

        self.gs_renderer = GaussianRenderer(opt)

        self.gaussians = self.gs_renderer.create_from_ply(gs_ply).to(self.device)

        # nerf renderer
        if not self.opt.force_cuda_rast:
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
        
        self.step = 0
        self.render_step_size = 5e-3
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)
        self.estimator = nerfacc.OccGridEstimator(roi_aabb=self.aabb, resolution=64, levels=1)

        self.encoder_density = GridEncoder(num_levels=12) # VMEncoder(output_dim=16, mode='sum')
        self.encoder = GridEncoder(num_levels=12)
        self.mlp_density = MLP(self.encoder_density.output_dim, 1, 32, 2, bias=False)
        self.mlp = MLP(self.encoder.output_dim, 3, 32, 2, bias=False)

        # mesh renderer
        self.proj = torch.from_numpy(get_perspective(self.opt.fovy)).float().to(self.device)
        self.v = self.f = None
        self.vt = self.ft = None
        self.deform = None
        self.albedo = None
        
       
    @torch.no_grad()
    def render_gs(self, pose):
    
        cam_poses = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        out = self.gs_renderer.render(self.gaussians.unsqueeze(0), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0))
        image = out['image'].squeeze(1).squeeze(0) # [C, H, W]
        alpha = out['alpha'].squeeze(2).squeeze(1).squeeze(0) # [H, W]

        return image, alpha

    def get_density(self, xs):
        # xs: [..., 3]
        xs = xs.to(self.device)
        prefix = xs.shape[:-1]
        xs = xs.view(-1, 3)
        feats = self.encoder_density(xs)
        density = trunc_exp(self.mlp_density(feats))
        density = density.view(*prefix, 1)
        return density
    
    def render_nerf(self, pose):
        
        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        
        # get rays
        resolution = self.opt.output_size
        rays_o, rays_d = get_rays(pose, resolution, resolution, self.opt.fovy)
        hw = rays_o.shape[0] * rays_o.shape[1]
        rays_o = rays_o.view(hw, 3)
        rays_d = rays_d.view(hw, 3)
        
        # update occ grid
        if self.training:
            def occ_eval_fn(xs):
                sigmas = self.get_density(xs)
                return self.render_step_size * sigmas
            
            self.estimator.update_every_n_steps(self.step, occ_eval_fn=occ_eval_fn, occ_thre=0.01, n=8)
            self.step += 1

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
        weights, trans, alphas = nerfacc.render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=n_rays)
        color = nerfacc.accumulate_along_rays(weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays)
        alpha = nerfacc.accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)

        color = color + 1 * (1.0 - alpha)

        color = color.view(resolution, resolution, 3).clamp(0, 1).permute(2, 0, 1).contiguous()
        alpha = alpha.view(resolution, resolution).clamp(0, 1).contiguous()
        
        return color, alpha

    def fit_nerf(self, iters=512, resolution=128):

        self.opt.output_size = resolution

        optimizer = torch.optim.Adam([
            {'params': self.encoder_density.parameters(), 'lr': 1e-2},
            {'params': self.encoder.parameters(), 'lr': 1e-2},
            {'params': self.mlp_density.parameters(), 'lr': 1e-3},
            {'params': self.mlp.parameters(), 'lr': 1e-3},
        ])

        print(f"[INFO] fitting nerf...")
        imgs, alphas = [], []
        
        comfy_pbar = comfy.utils.ProgressBar(iters)
        pbar = tqdm.trange(iters)
        for i in pbar:

            ver = np.random.randint(-45, 45)
            hor = np.random.randint(-180, 180)
            rad = np.random.uniform(1.5, 3.0)
            
            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            imgs.append(image_gt.permute(1, 2, 0).unsqueeze(0))
            alphas.append(alpha_gt.unsqueeze(0))
            image_pred, alpha_pred = self.render_nerf(pose)

            # if i % 200 == 0:
            #     kiui.vis.plot_image(image_gt, alpha_gt, image_pred, alpha_pred)
            
            loss_mse = F.mse_loss(image_pred, image_gt) + 0.1 * F.mse_loss(alpha_pred, alpha_gt)
            loss = loss_mse #+ 0.1 * self.encoder_density.tv_loss() #+ 0.0001 * self.encoder_density.density_loss()

            loss.backward()
            self.encoder_density.grad_total_variation(1e-8)
        
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            comfy_pbar.update_absolute(i + 1)
        
        print(f"[INFO] finished fitting nerf!")
        
        return torch.cat(imgs, dim=0), torch.cat(alphas, dim=0)
    
    def render_mesh(self, pose):

        h = w = self.opt.output_size

        v = self.v + self.deform
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ self.proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(-1).squeeze(0) # [H, W] important to enable gradients!
        
        if self.albedo is None:
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, H, W, 3]
            xyzs = xyzs.view(-1, 3)
            mask = (alpha > 0).view(-1)
            image = torch.zeros_like(xyzs, dtype=torch.float32)
            if mask.any():
                masked_albedo = torch.sigmoid(self.mlp(self.encoder(xyzs[mask].detach(), bound=1)))
                image[mask] = masked_albedo.float()
        else:
            texc, texc_db = dr.interpolate(self.vt.unsqueeze(0), rast, self.ft, rast_db=rast_db, diff_attrs='all')
            image = torch.sigmoid(dr.texture(self.albedo.unsqueeze(0), texc, uv_da=texc_db)) # [1, H, W, 3]

        image = image.view(1, h, w, 3)
        # image = dr.antialias(image, rast, v_clip, f).clamp(0, 1)
        image = image.squeeze(0).permute(2, 0, 1).contiguous() # [3, H, W]
        image = alpha * image + (1 - alpha)

        return image, alpha

    def fit_mesh(self, iters=2048, remesh_after_n_iteration=512, resolution=512, grid_size=256, S=128, density_thresh=10, decimate_target=5e4):

        self.opt.output_size = resolution
        
        vertices, triangles = marching_cubes_density_to_mesh(self.get_density, grid_size, S, density_thresh, decimate_target)
        
        self.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        self.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
        self.deform = nn.Parameter(torch.zeros_like(self.v)).to(self.device)
        
        # Here coarse mesh already resemble the target shape closely
        #mcube_mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)
        #mcube_mesh.auto_normal()
        #mcube_mesh.auto_uv()

        # fit mesh from gs
        lr_factor = 1
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-3 * lr_factor},
            {'params': self.mlp.parameters(), 'lr': 1e-3 * lr_factor},
            {'params': self.deform, 'lr': 1e-4},
        ])

        print(f"[INFO] fitting mesh...")
        comfy_pbar = comfy.utils.ProgressBar(iters)
        pbar = tqdm.trange(iters)
        for i in pbar:

            ver = np.random.randint(-10, 10)
            hor = np.random.randint(-180, 180)
            rad = self.opt.cam_radius # np.random.uniform(1, 2)

            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            image_pred, alpha_pred = self.render_mesh(pose)

            loss_mse = F.mse_loss(image_pred, image_gt) + 0.1 * F.mse_loss(alpha_pred, alpha_gt)
            loss_lap = laplacian_smooth_loss(self.v + self.deform, self.f)
            loss_normal = normal_consistency(self.v + self.deform, self.f)
            loss_offsets = (self.deform ** 2).sum(-1).mean()
            loss = loss_mse + 0.001 * loss_normal + 0.1 * loss_offsets * loss_lap * 0.01

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # remesh periodically
            if i > 0 and i % remesh_after_n_iteration == 0:
                vertices = (self.v + self.deform).detach().cpu().numpy()
                triangles = self.f.detach().cpu().numpy()
                vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)
                if triangles.shape[0] > decimate_target:
                    vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, optimalplacement=False)
                self.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
                self.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
                self.deform = nn.Parameter(torch.zeros_like(self.v)).to(self.device)
                lr_factor *= 0.5
                optimizer = torch.optim.Adam([
                    {'params': self.encoder.parameters(), 'lr': 1e-3 * lr_factor},
                    {'params': self.mlp.parameters(), 'lr': 1e-3 * lr_factor},
                    {'params': self.deform, 'lr': 1e-4},
                ])

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            comfy_pbar.update_absolute(i + 1)
        
        # last clean
        vertices = (self.v + self.deform).detach().cpu().numpy()
        triangles = self.f.detach().cpu().numpy()
        vertices, triangles = clean_mesh(vertices, triangles, remesh=False)
        self.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        self.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
        self.deform = nn.Parameter(torch.zeros_like(self.v).to(self.device))
        
        print(f"[INFO] finished fitting mesh!")

    def get_color(self, xs):
        return torch.sigmoid(self.mlp(self.encoder(xs.to(self.device))))
    
    # uv mesh refine
    def fit_mesh_uv(self, iters=512, resolution=512, texture_resolution=1024, padding=2):

        self.opt.output_size = resolution

        # unwrap uv
        print(f"[INFO] uv unwrapping...")
        mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)
        mesh.auto_normal()
        mesh.auto_uv()

        self.vt = mesh.vt
        self.ft = mesh.ft
        
        # get albedo map from NeRF
        albedo = color_func_to_albedo(mesh, self.get_color, texture_resolution, device=self.device, glctx=self.glctx)

        # optimize texture
        self.albedo = nn.Parameter(inverse_sigmoid(albedo)).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.albedo, 'lr': 1e-3},
        ])

        print(f"[INFO] fitting mesh texture...")
        comfy_pbar = comfy.utils.ProgressBar(iters)
        pbar = tqdm.trange(iters)
        for i in pbar:

            # shrink to front view as we care more about it...
            ver = np.random.randint(-5, 5)
            hor = np.random.randint(-15, 15)
            rad = self.opt.cam_radius # np.random.uniform(1, 2)
            
            pose = orbit_camera(ver, hor, rad)
            
            image_gt, alpha_gt = self.render_gs(pose)
            image_pred, alpha_pred = self.render_mesh(pose)

            loss_mse = F.mse_loss(image_pred, image_gt)
            loss = loss_mse

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"MSE = {loss_mse.item():.6f}")
            comfy_pbar.update_absolute(i + 1)
        
        print(f"[INFO] finished fitting mesh texture!")


    @torch.no_grad()
    def get_mesh(self):
        mesh = Mesh(v=self.v, f=self.f, vt=self.vt, ft=self.ft, albedo=torch.sigmoid(self.albedo), device=self.device)
        mesh.auto_normal()
        return mesh
