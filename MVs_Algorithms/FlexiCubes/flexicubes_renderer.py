import os
import numpy as np
import torch

import nvdiffrast.torch as dr
from kiui.cam import orbit_camera

from FlexiCubes import util

class FlexiCubesRenderer:
    def __init__(self, force_cuda_rast):

        if force_cuda_rast or os.name != 'nt':
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()

    def get_rotate_camera(self, itr, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
        proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 10) * np.pi * 2
        mv     = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        return mv.to(device), mvp.to(device)
        
    def get_orbit_camera(self, azimuth=0, elevation=0, fovy=45, iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
        aspect = iter_res[1] / iter_res[0]
        fovy = np.deg2rad(fovy)
        proj_mtx = util.perspective(fovy, aspect, cam_near_far[0], cam_near_far[1])
        #fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect)
        #proj_mtx = get_projection_matrix(cam_near_far[0], cam_near_far[1], fovx, -fovy, z_sign=-1.0)

        # Smooth rotation for display.
        #mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-np.de2rad(elevation)) @ util.rotate_y(np.deg2rad(azimuth)))
        mv = torch.tensor(np.linalg.inv(orbit_camera(-elevation, azimuth, cam_radius)))
        mvp = proj_mtx @ mv
        return mv.to(device), mvp.to(device)

    def render_mesh(self, mesh, mv, mvp, iter_res, depth_min=-5.5, depth_max=-0.5, return_types=["mask", "depth", "normal"], white_bg=False):
        '''
        The rendering function used to produce the results in the paper.
        '''
        v_pos_clip = util.xfm_points(mesh.vertices.unsqueeze(0), mvp)  # Rotate it to camera coordinates
        rast, db = dr.rasterize(
            dr.RasterizeCudaContext(), v_pos_clip, mesh.faces.int(), iter_res)

        alpha_index = rast[..., -1:] > 0
        inv_alpha_index = rast[..., -1:] <= 0
        alpha = alpha_index.float()

        out_dict = {}
        for type in return_types:
            if type == "mask" :
                img = dr.antialias(alpha, rast, v_pos_clip, mesh.faces.int()) 
            elif type == "depth":
                v_pos_cam = util.xfm_points(mesh.vertices.unsqueeze(0), mv)
                img, _ = util.interpolate(v_pos_cam[..., [2]].contiguous(), rast, mesh.faces.int())
                img_masked = img[alpha_index]
                img_masked = torch.clamp(img_masked, min=depth_min, max=depth_max)
                img_masked = (img_masked - depth_min) / (depth_max - depth_min)
                img[alpha_index] = img_masked
                img[inv_alpha_index] = 0
            elif type == "normal" :
                normal_indices = (torch.arange(0, mesh.nrm.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
                img, _ = util.interpolate(mesh.nrm.unsqueeze(0).contiguous(), rast, normal_indices.int())
            elif type == "vertex_normal":
                img, _ = util.interpolate(mesh.v_nrm.unsqueeze(0).contiguous(), rast, mesh.faces.int())
                img = dr.antialias((img + 1) * 0.5, rast, v_pos_clip, mesh.faces.int()) 
            if white_bg:
                bg = torch.ones_like(img) 
                img = torch.lerp(bg, img, alpha)
            out_dict[type] = img
        return out_dict
