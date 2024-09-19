import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from LGM.core.options import Options

import kiui

class GaussianRenderer:
    def __init__(self, opt: Options):
        
        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size,
                    image_width=self.opt.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }


    def to_ply(self, gaussians, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        return PlyData([el])
    
    def create_from_ply(self, plydata, compatible=True):

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians