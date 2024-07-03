import random
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.mesh_utils import laplacian_smooth_loss, normal_consistency
from pytorch_msssim import SSIM, MS_SSIM

import comfy.utils

from .diff_mesh_renderer import DiffRastRenderer

from shared_utils.camera_utils import BaseCameraController
from shared_utils.image_utils import prepare_torch_img

class DiffMeshCameraController(BaseCameraController):
    
    def get_render_result(self, render_pose, bg_color, **kwargs):
        ref_cam = (render_pose, self.cam.perspective)
        return self.renderer.render(*ref_cam, self.cam.H, self.cam.W, ssaa=1, bg_color=bg_color, **kwargs) #ssaa = min(2.0, max(0.125, 2 * np.random.random()))

class DiffMesh:
    
    def __init__(
        self, 
        mesh, 
        training_iterations, 
        batch_size, 
        texture_learning_rate, 
        train_mesh_geometry, 
        geometry_learning_rate, 
        ms_ssim_loss_weight, 
        remesh_after_n_iteration, 
        invert_bg_prob, 
        force_cuda_rasterize
    ):
        self.device = torch.device("cuda")
        
        self.train_mesh_geometry = train_mesh_geometry
        self.remesh_after_n_iteration = remesh_after_n_iteration
        
        # prepare main components for optimization
        self.renderer = DiffRastRenderer(mesh, force_cuda_rasterize).to(self.device)

        self.optimizer = torch.optim.Adam(self.renderer.get_params(texture_learning_rate, train_mesh_geometry, geometry_learning_rate))
        #self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.lambda_ssim = ms_ssim_loss_weight
        
        self.training_iterations = training_iterations
        
        self.batch_size = batch_size
        
        self.invert_bg_prob = invert_bg_prob
    
    def prepare_training(self, reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy):
        self.ref_imgs_num = len(reference_images)
    
        self.ref_size_H = reference_images[0].shape[0]
        self.ref_size_W = reference_images[0].shape[1]
        
        # default camera settings
        self.cam_controller = DiffMeshCameraController(
            self.renderer, self.ref_size_W, self.ref_size_H, reference_orbit_camera_fovy, self.invert_bg_prob, None, self.device
        )

        self.all_ref_cam_poses = reference_orbit_camera_poses
        
        # prepare reference images and masks
        ref_imgs_torch_list = []
        ref_masks_torch_list = []
        for i in range(self.ref_imgs_num):
            ref_imgs_torch_list.append(prepare_torch_img(reference_images[i].unsqueeze(0), self.ref_size_H, self.ref_size_W, self.device))
            ref_masks_torch_list.append(prepare_torch_img(reference_masks[i].unsqueeze(2).unsqueeze(0), self.ref_size_H, self.ref_size_W, self.device))
            
        self.ref_imgs_torch = torch.cat(ref_imgs_torch_list, dim=0)
        self.ref_masks_torch = torch.cat(ref_masks_torch_list, dim=0)
    
    def training(self, decimate_target=5e4):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        ref_imgs_masked = []
        for i in range(self.ref_imgs_num):
            ref_imgs_masked.append((self.ref_imgs_torch[i] * self.ref_masks_torch[i]).unsqueeze(0))
            
        ref_imgs_num_minus_1 = self.ref_imgs_num-1
        
        comfy_pbar = comfy.utils.ProgressBar(self.training_iterations)

        for step in tqdm.trange(self.training_iterations):

            ### calculate loss between reference and rendered image from known view
            loss = 0
            masked_rendered_img_batch = []
            masked_ref_img_batch = []
            for _ in range(self.batch_size):
                
                i = random.randint(0, ref_imgs_num_minus_1)

                out = self.cam_controller.render_at_pose(self.all_ref_cam_poses[i])                

                image = out["image"]    # [H, W, 3] in [0, 1]
                image = image.permute(2, 0, 1).contiguous()  # [3, H, W] in [0, 1]
                
                image_masked = (image * self.ref_masks_torch[i]).unsqueeze(0)
                
                masked_rendered_img_batch.append(image_masked)
                masked_ref_img_batch.append(ref_imgs_masked[i])
            
            masked_rendered_img_batch_torch = torch.cat(masked_rendered_img_batch, dim=0)
            masked_ref_img_batch_torch = torch.cat(masked_ref_img_batch, dim=0)
                
            # rgb loss
            loss += (1 - self.lambda_ssim) * F.mse_loss(masked_rendered_img_batch_torch, masked_ref_img_batch_torch)
            
            # D-SSIM loss
            # [1, 3, H, W] in [0, 1]
            #loss += self.lambda_ssim * (1 - self.ssim_loss(X, Y))
            loss += self.lambda_ssim * (1 - self.ms_ssim_loss(masked_ref_img_batch_torch, masked_rendered_img_batch_torch))
            
            # Regularization loss
            if self.train_mesh_geometry:
                current_v = self.renderer.mesh.v + self.renderer.v_offsets
                loss += 0.01 * laplacian_smooth_loss(current_v, self.renderer.mesh.f)
                loss += 0.001 * normal_consistency(current_v, self.renderer.mesh.f)
                loss += 0.1 * (self.renderer.v_offsets ** 2).sum(-1).mean()
                
                # remesh periodically
                if step > 0 and step % self.remesh_after_n_iteration == 0:
                    vertices = (self.renderer.mesh.v + self.renderer.v_offsets).detach().cpu().numpy()
                    triangles = self.renderer.mesh.f.detach().cpu().numpy()
                    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)
                    if triangles.shape[0] > decimate_target:
                        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target, optimalplacement=False)
                    self.renderer.mesh.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
                    self.renderer.mesh.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
                    self.renderer.v_offsets = nn.Parameter(torch.zeros_like(self.renderer.mesh.v)).to(self.device)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            comfy_pbar.update_absolute(step + 1)
            
        torch.cuda.synchronize()
            
        self.need_update = True
            
        print(f"Step: {step}")

        self.renderer.update_mesh()
        
        ender.record()
        #t = starter.elapsed_time(ender)
        
    def get_mesh_and_texture(self):
        return (self.renderer.mesh, self.renderer.mesh.albedo, )