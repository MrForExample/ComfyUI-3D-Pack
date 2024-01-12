import random
import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM

from .diff_mesh_renderer import Renderer
from ..shared_utils.camera_utils import orbit_camera, OrbitCamera

class DiffTextureBaker:
    
    def __init__(self, mesh, training_iterations, batch_size, texture_learning_rate, train_mesh_geometry, geometry_learning_rate, ms_ssim_loss_weight, force_cuda_rasterize):
        self.device = torch.device("cuda")
        
        # prepare main components for optimization
        self.renderer = Renderer(mesh, force_cuda_rasterize).to(self.device)

        self.optimizer = torch.optim.Adam(self.renderer.get_params(texture_learning_rate, train_mesh_geometry, geometry_learning_rate))
        #self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.lambda_ssim = ms_ssim_loss_weight
        
        self.training_iterations = training_iterations
        
        self.batch_size = batch_size
    
    def prepare_img(self, img):
        img_new = img.permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_new = F.interpolate(img_new, (self.ref_size_H, self.ref_size_W), mode="bilinear", align_corners=False).contiguous()
        return img_new
    
    def prepare_training(self, reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy):
        self.ref_imgs_num = len(reference_images)

        self.all_ref_cam_poses = reference_orbit_camera_poses
        self.ref_cam_fovy = reference_orbit_camera_fovy
    
        self.ref_size_H = reference_images[0].shape[0]
        self.ref_size_W = reference_images[0].shape[1]
        
        self.cam = OrbitCamera(self.ref_size_W, self.ref_size_H, fovy=reference_orbit_camera_fovy)
        
        # prepare reference images and masks
        ref_imgs_torch_list = []
        ref_masks_torch_list = []
        for i in range(self.ref_imgs_num):
            ref_imgs_torch_list.append(self.prepare_img(reference_images[i]))
            ref_masks_torch_list.append(self.prepare_img(reference_masks[i].unsqueeze(2)))
            
        self.ref_imgs_torch = torch.cat(ref_imgs_torch_list, dim=0)
        self.ref_masks_torch = torch.cat(ref_masks_torch_list, dim=0)
    
    def training(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        ref_imgs_masked = []
        for i in range(self.ref_imgs_num):
            ref_imgs_masked.append((self.ref_imgs_torch[i] * self.ref_masks_torch[i]).unsqueeze(0))
            
        ref_imgs_num_minus_1 = self.ref_imgs_num-1

        for step in tqdm.trange(self.training_iterations):

            ### calculate loss between reference and rendered image from known view
            loss = 0
            masked_rendered_img_batch = []
            masked_ref_img_batch = []
            for _ in range(self.batch_size):
                
                i = random.randint(0, ref_imgs_num_minus_1)

                radius, elevation, azimuth, center_X, center_Y, center_Z = self.all_ref_cam_poses[i]
                
                # render output
                orbit_target = np.array([center_X, center_Y, center_Z], dtype=np.float32)
                ref_pose = orbit_camera(elevation, azimuth, radius, target=orbit_target)
                ref_cam = (ref_pose, self.cam.perspective)
                out = self.renderer.render(*ref_cam, self.ref_size_H, self.ref_size_W, ssaa=1) #ssaa = min(2.0, max(0.125, 2 * np.random.random()))

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

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        torch.cuda.synchronize()
            
        self.need_update = True
            
        print(f"Step: {step}")

        self.renderer.update_mesh()
        
        ender.record()
        #t = starter.elapsed_time(ender)
        
    def get_mesh_and_texture(self):
        return (self.renderer.mesh, self.renderer.mesh.albedo, )