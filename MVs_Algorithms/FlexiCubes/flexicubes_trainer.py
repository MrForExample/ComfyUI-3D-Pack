import os
import imageio

import torch
import numpy as np

import tqdm
import comfy.utils

from .flexicubes_renderer import FlexiCubesRenderer
from .flexicubes import FlexiCubes
from .util import SimpleMesh
from .loss import sdf_reg_loss

from shared_utils.camera_utils import OrbitCamera
from mesh_processer.mesh import Mesh


def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs. 

class FlexiCubesTrainer:
    
    def __init__(
            self,
            training_iterations,
            batch_size,
            learning_rate,
            voxel_grids_resolution,
            depth_near=0.5,
            depth_far=5.5,
            mask_loss_weight=1.0,
            depth_loss_weight=100.0,
            normal_loss_weight=1.0,
            sdf_regularizer_weight=0.2,
            remove_floaters_weight=0.5,
            cube_stabilizer_weight=0.1,
            force_cuda_rast=False,
            device='cuda'
        ):
        self.device = torch.device(device)
        
        self.renderer = FlexiCubesRenderer(force_cuda_rast)

        #  Create and initialize FlexiCubes
        self.voxel_grid_res = voxel_grids_resolution
        self.fc = FlexiCubes(self.device)
        self.x_nx3, self.cube_fx8 = self.fc.construct_voxel_grid(self.voxel_grid_res)
        self.x_nx3 *= 2 # scale up the grid so that it's larger than the target object
        
        self.sdf = torch.rand_like(self.x_nx3[:,0]) - 0.1 # randomly init SDF
        self.sdf    = torch.nn.Parameter(self.sdf.clone().detach(), requires_grad=True)
        # set per-cube learnable weights to zeros
        self.weight = torch.zeros((self.cube_fx8.shape[0], 21), dtype=torch.float, device=self.device) 
        self.weight    = torch.nn.Parameter(self.weight.clone().detach(), requires_grad=True)
        self.deform = torch.nn.Parameter(torch.zeros_like(self.x_nx3), requires_grad=True)
        
        #  Retrieve all the edges of the voxel grid; these edges will be utilized to 
        #  compute the regularization loss in subsequent steps of the process.    
        all_edges = self.cube_fx8[:, self.fc.cube_edges].reshape(-1, 2)
        self.grid_edges = torch.unique(all_edges, dim=0)
        
        #  Setup optimizer
        self.optimizer = torch.optim.Adam([self.sdf, self.weight, self.deform], lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: lr_schedule(x)) 
        
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        
        self.depth_near = -depth_near
        self.depth_far = -depth_far
        
        self.mask_loss_weight = mask_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.normal_loss_weight = normal_loss_weight
        self.sdf_regularizer_weight = sdf_regularizer_weight
        self.remove_floaters_weight = remove_floaters_weight
        self.cube_stabilizer_weight = cube_stabilizer_weight
    
    def prepare_training(self, reference_depth_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy, reference_normals=None):
        self.ref_imgs_num = len(reference_depth_images)

        self.all_ref_cam_poses = reference_orbit_camera_poses
        self.ref_cam_fovy = reference_orbit_camera_fovy
    
        self.ref_size_H = reference_depth_images[0].shape[0]
        self.ref_size_W = reference_depth_images[0].shape[1]
        
        self.cam = OrbitCamera(self.ref_size_W, self.ref_size_H, fovy=reference_orbit_camera_fovy)
        
        # prepare reference images and masks
        self.ref_depth_imgs_torch = reference_depth_images[:, :, :, 0].unsqueeze(3).to(self.device) # (N, H, W, C) -> (N, H, W, 1)
        self.ref_masks_torch = reference_masks.unsqueeze(3).to(self.device)
        
        if reference_normals is not None:
            self.ref_normal_imgs_torch = (reference_normals * 2 - 1).to(self.device) # change value from [0, 1] -> [-1, 1]
        else:
            self.ref_normal_imgs_torch = None
        
        # prepare reference camera projection matrix for all camera poses
        mv_all = []
        mvp_all = []
        for pose in self.all_ref_cam_poses:
            mv, mvp = self.renderer.get_orbit_camera(pose[2], pose[1], cam_radius=pose[0], device=self.device)
            mv_all.append(mv)
            mvp_all.append(mvp)
        self.mv_all = torch.stack(mv_all).to(self.device)
        self.mvp_all = torch.stack(mvp_all).to(self.device)
        self.camposes_len = len(self.all_ref_cam_poses)
    
    def training(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        comfy_pbar = comfy.utils.ProgressBar(self.training_iterations)
        
        if self.ref_normal_imgs_torch is not None:
            return_types = ["mask", "depth", "normal"]
        else:
            return_types = ["mask", "depth"]
        
        for step in tqdm.trange(self.training_iterations):
            # sample random render & camera pose from multi-views
            batch_index = np.random.randint(0, self.camposes_len, size=self.batch_size)
            mv = self.mv_all[batch_index, :, :]
            mvp = self.mvp_all[batch_index, :, :]

            target_mask = self.ref_masks_torch[batch_index, :, :, :]
            target_depth = self.ref_depth_imgs_torch[batch_index, :, :, :]
            
            # extract and render FlexiCubes mesh
            grid_verts = self.x_nx3 + (2-1e-8) / (self.voxel_grid_res * 2) * torch.tanh(self.deform)
            vertices, faces, L_dev = self.fc(grid_verts, self.sdf, self.cube_fx8, self.voxel_grid_res, beta_fx12=self.weight[:,:12], alpha_fx8=self.weight[:,12:20],
                gamma_f=self.weight[:,20], training=True)
            flexicubes_mesh = SimpleMesh(vertices, faces)
            if self.ref_normal_imgs_torch is not None:
                flexicubes_mesh.auto_normals()
            buffers = self.renderer.render_mesh(flexicubes_mesh, mv, mvp, (self.ref_size_H, self.ref_size_W), self.depth_far, self.depth_near, return_types)
            
            t_iter = step / self.training_iterations
            # evaluate reconstruction loss
            # mask & depth shape: (N, H, W, 1)
            mask_loss = (buffers['mask'] - target_mask).abs().mean() * self.mask_loss_weight
            depth_loss = (((((buffers['depth'] - target_depth)* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.depth_loss_weight
            total_loss = mask_loss + depth_loss
            if self.ref_normal_imgs_torch is not None:
                target_normal = self.ref_normal_imgs_torch[batch_index, :, :, :]
                normal_loss = (((((buffers['normal'] - (target_normal))* target_mask)**2).sum(-1)+1e-8)).sqrt().mean() * self.normal_loss_weight * t_iter
                total_loss += normal_loss
        
            sdf_weight = self.sdf_regularizer_weight - (self.sdf_regularizer_weight - self.sdf_regularizer_weight/20)*min(1.0, 4.0 * t_iter)
            reg_loss = sdf_reg_loss(self.sdf, self.grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
            reg_loss += L_dev.mean() * self.remove_floaters_weight
            reg_loss += (self.weight[:,:20]).abs().mean() * self.cube_stabilizer_weight
            total_loss += reg_loss
            
            # optimize step
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            comfy_pbar.update_absolute(step + 1)
            
            #self.test_save(step, mv, mvp, grid_verts, total_loss, out_dir="C:\\Users\\reall\\Softwares\\ComfyUI_windows_portable\\ComfyUI\\output\\FlexiCubes_Output\\Test_Normals")
            
        ender.record()
        
    def get_mesh(self):
        grid_verts = self.x_nx3 + (2-1e-8) / (self.voxel_grid_res * 2) * torch.tanh(self.deform)
        vertices, faces, L_dev = self.fc(grid_verts, self.sdf, self.cube_fx8, self.voxel_grid_res, beta_fx12=self.weight[:,:12], alpha_fx8=self.weight[:,12:20],
            gamma_f=self.weight[:,20], training=False)

        v = vertices.detach().contiguous().float().to(self.device)
        f = faces.detach().contiguous().float().to(self.device)
        mesh = Mesh(v=v, f=f, device=self.device)
        mesh.auto_normal()
        mesh.auto_uv()
        
        # Trimesh seems output better mesh
        #import trimesh
        #import os
        #mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
        #mesh_np.export(os.path.join("C:\\Users\\reall\\Softwares\\ComfyUI_windows_portable\\ComfyUI\\output\\FlexiCubes_Output", 'output_trimesh.obj'))
        
        return mesh
    
    def test_save(self, step, mv, mvp, grid_verts, total_loss, out_dir, save_interval=20, display_res=[1024, 1024]):
        if (step % save_interval == 0 or step == (self.training_iterations-1)): # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = self.fc(grid_verts, self.sdf, self.cube_fx8, self.voxel_grid_res, beta_fx12=self.weight[:,:12], alpha_fx8=self.weight[:,12:20],
                gamma_f=self.weight[:,20], training=False)
                flexicubes_mesh = SimpleMesh(vertices, faces)
                flexicubes_mesh.auto_normals() # compute face normals for visualization
                
                mv, mvp = self.renderer.get_rotate_camera(step//save_interval, iter_res=display_res, device=self.device)
                mv = mv.unsqueeze(0)
                mvp = mvp.unsqueeze(0)
                val_buffers = self.renderer.render_mesh(flexicubes_mesh, mv, mvp, display_res, return_types=["normal"], white_bg=True)
                val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                #val_image = (val_buffers["depth"][0].detach().cpu().numpy()*255).astype(np.uint8)
                #val_repeat = np.repeat(val_image, 4, axis=2)
                #val_repeat[:, :, 3] = 255
                
                #gt_image = (gt_image[0].detach().cpu().numpy()*255).astype(np.uint8)
                #gt_repeat = np.repeat(gt_image, 4, axis=2)
                #gt_repeat[:, :, 3] = 255
                
                imageio.imwrite(os.path.join(out_dir, '{:04d}.png'.format(step)), val_image)
                #imageio.imwrite(os.path.join(out_dir, '{:04d}.png'.format(step)), np.concatenate([val_repeat, gt_repeat], 1))
                print(f"Optimization Step [{step}/{self.training_iterations}], Loss: {total_loss.item():.4f}")