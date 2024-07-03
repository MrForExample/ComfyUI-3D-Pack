import random
import tqdm

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM

import comfy.utils

from .main_3DGS_renderer import GaussianSplattingRenderer

from shared_utils.camera_utils import BaseCameraController, MiniCam, calculate_fovX, get_projection_matrix
from shared_utils.image_utils import prepare_torch_img

class GSParams:
    def __init__(
        self,
        training_iterations=30_000, 
        batch_size=1,
        lambda_ssim=0.2,
        lambda_alpha=3,
        lambda_offset=0,
        lambda_offset_opacity=0,
        invert_bg_prob = 0.5,
        feature_lr=0.0025, 
        opacity_lr=0.05, 
        scaling_lr=0.005, 
        rotation_lr=0.001, 
        position_lr_init=0.00016, 
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        num_pts=10_000,
        K=3,
        percent_dense=0.01,
        density_start_iter=500,
        density_end_iter=15_000,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_grad_threshold=0.0002,
        sh_degree=3
    ):
        
        # training params
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.lambda_ssim = lambda_ssim
        self.lambda_alpha = lambda_alpha
        self.lambda_offset = lambda_offset
        self.lambda_offset_opacity = lambda_offset_opacity
        self.invert_bg_prob = invert_bg_prob
        
        # learning params
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        
        # densify and prune params
        self.num_pts = num_pts
        self.K = K
        self.percent_dense = percent_dense
        self.density_start_iter=density_start_iter
        self.density_end_iter=density_end_iter
        self.densification_interval=densification_interval
        self.opacity_reset_interval=opacity_reset_interval
        self.densify_grad_threshold=densify_grad_threshold
        
        # other gaussian params
        self.sh_degree = sh_degree
        
class GaussianSplattingCameraController(BaseCameraController):
    def get_render_result(self, render_pose, bg_color, **kwargs):
        render_cam = MiniCam(render_pose, self.cam.W, self.cam.H, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far, self.projection_matrix)
        return self.renderer.render(render_cam, bg_color=bg_color, **kwargs)
    
    def post_init(self):
        self.projection_matrix = get_projection_matrix(self.cam.near, self.cam.far, self.cam.fovx, self.cam.fovy).transpose(0, 1).cuda()

class GaussianSplatting3D:
            
    def __init__(self, gs_params=None, init_input=None, device='cuda'):
        self.device = torch.device(device)
        
        # prepare renderer for optimization
        if gs_params is None:
            gs_params = GSParams()
            
        self.renderer = GaussianSplattingRenderer(sh_degree=gs_params.sh_degree)
        self.renderer.initialize(init_input, num_pts=gs_params.num_pts)

        # setup training
        self.renderer.gaussians.training_setup(gs_params)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer
        #self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)
        
        self.gs_params = gs_params
    
    def prepare_training(self, reference_images, reference_masks, reference_orbit_camera_poses, reference_orbit_camera_fovy):
        self.ref_imgs_num = len(reference_images)
        
        self.ref_size_H = reference_images[0].shape[0]
        self.ref_size_W = reference_images[0].shape[1]
        
        # default camera settings
        self.cam_controller = GaussianSplattingCameraController(
            self.renderer, self.ref_size_W, self.ref_size_H, reference_orbit_camera_fovy, self.gs_params.invert_bg_prob, None, self.device
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
    
    def training(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        ref_imgs_masked = []
        for i in range(self.ref_imgs_num):
            ref_imgs_masked.append((self.ref_imgs_torch[i] * self.ref_masks_torch[i]).unsqueeze(0))
            
        ref_imgs_num_minus_1 = self.ref_imgs_num-1
        
        comfy_pbar = comfy.utils.ProgressBar(self.gs_params.training_iterations)

        for step in tqdm.trange(self.gs_params.training_iterations):

            ### calculate loss between reference and rendered image from known view
            loss = 0
            masked_rendered_img_batch = []
            masked_ref_img_batch = []
            masks_batch = []
            ref_masks_batch = []

            step_ratio = min(1, step / self.gs_params.training_iterations)

            # update lr
            self.renderer.gaussians.update_learning_rate(step)

            loss = 0

            for _ in range(self.gs_params.batch_size):
                ### calculate loss between reference and rendered image from known view
                    
                i = random.randint(0, ref_imgs_num_minus_1)
                
                out = self.cam_controller.render_at_pose(self.all_ref_cam_poses[i])
                
                image = out["image"] # [3, H, W] in [0, 1]
                mask = out["alpha"] # [1, H, W] in [0, 1]
                
                ref_mask = self.ref_masks_torch[i]
                image_masked = (image * ref_mask).unsqueeze(0)
                
                masked_rendered_img_batch.append(image_masked)
                masked_ref_img_batch.append(ref_imgs_masked[i])
                masks_batch.append(mask.unsqueeze(0))
                ref_masks_batch.append(ref_mask.unsqueeze(0))
            
            masked_rendered_img_batch_torch = torch.cat(masked_rendered_img_batch, dim=0)
            masked_ref_img_batch_torch = torch.cat(masked_ref_img_batch, dim=0)
            masks_batch_torch = torch.cat(masks_batch, dim=0)
            ref_masks_batch_torch = torch.cat(ref_masks_batch, dim=0)
                
            #loss_scaler = self.gs_params.loss_scale * step_ratio
                
            # rgb loss            
            loss += (1 - self.gs_params.lambda_ssim) * F.l1_loss(masked_rendered_img_batch_torch, masked_ref_img_batch_torch)

            # alpha loss
            loss += self.gs_params.lambda_alpha * F.mse_loss(masks_batch_torch, ref_masks_batch_torch)
            
            # D-SSIM loss
            # [1, 3, H, W] in [0, 1]
            #loss += self.lambda_ssim * (1 - self.ssim_loss(X, Y))
            loss += self.gs_params.lambda_ssim * (1 - self.ms_ssim_loss(masked_ref_img_batch_torch, masked_rendered_img_batch_torch))

            if self.gs_params.lambda_offset > 0:
                # Reference offset loss
                offset_norm = self.renderer.gaussians.get_xyz_offset.norm(dim=-1, keepdim=True)
                loss += self.gs_params.lambda_offset * torch.mean(offset_norm)
            
            if self.gs_params.lambda_offset_opacity > 0:
                # Alpha penalty loss
                loss += self.gs_params.lambda_offset_opacity * torch.mean(offset_norm.detach() * self.renderer.gaussians.get_opacity)


            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if step >= self.gs_params.density_start_iter and step <= self.gs_params.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if step % self.gs_params.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.gs_params.densify_grad_threshold, min_opacity=0.005, extent=4, max_screen_size=1)
                    #self.renderer.gaussians.densify_and_prune_by_compatness(self.gs_params.K, min_opacity=0.01, extent=4, max_screen_size=1)
                    
                    #self.renderer.gaussians.densify_by_clone_and_split(self.gs_params.densify_grad_threshold, extent=4)
                    #self.renderer.gaussians.densify_by_compatness(self.gs_params.K)
                    #self.renderer.gaussians.prune(min_opacity=0.01, extent=4, max_screen_size=1, max_offset=0.1)
                
                if step % self.gs_params.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()
                    
            comfy_pbar.update_absolute(step + 1)

        ender.record()
        torch.cuda.synchronize()
        #t = starter.elapsed_time(ender)

        self.need_update = True