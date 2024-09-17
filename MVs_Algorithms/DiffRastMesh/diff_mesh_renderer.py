import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from kiui.op import inverse_sigmoid

from mesh_processer.mesh import safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

class DiffRastRenderer(nn.Module):
    def __init__(self, mesh, force_cuda_rast):
        
        super().__init__()

        self.mesh = mesh

        if force_cuda_rast or os.name != 'nt':
            self.glctx = dr.RasterizeCudaContext()
        else:
            self.glctx = dr.RasterizeGLContext()
        
        # extract trainable parameters
        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v), requires_grad=True)
        self.raw_albedo = nn.Parameter(inverse_sigmoid(self.mesh.albedo), requires_grad=True)

        self.train_geo = False

    def get_params(self, texture_lr, train_geo, geom_lr):

        params = [
            {'params': self.raw_albedo, 'lr': texture_lr},
        ]

        self.train_geo = train_geo
        if train_geo:
            params.append({'params': self.v_offsets, 'lr': geom_lr})

        return params

    def update_mesh(self):
        self.mesh.v = (self.mesh.v + self.v_offsets).detach()
        self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
    
    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1, texture_filter='linear', 
               optional_render_types=['depth', 'normal']):
        
        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        if self.train_geo:
            v = self.mesh.v + self.v_offsets # [N, 3]
        else:
            v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        #alpha = (rast[0, ..., 3:] > 0).float() # [H, W, 1]
        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).clamp(0, 1).squeeze(0) # [H, W, 1] important to enable gradients!
            
        # render albedo
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]
        albedo = torch.sigmoid(albedo)
        
        # render depth
        if 'depth' in optional_render_types:
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
            depth = depth.squeeze(0) # [H, W, 1]

        # get vn and render normal
        if 'normal' in optional_render_types:
            if self.train_geo:
                i0, i1, i2 = self.mesh.f[:, 0].long(), self.mesh.f[:, 1].long(), self.mesh.f[:, 2].long()
                v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

                face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
                face_normals = safe_normalize(face_normals)
                
                vn = torch.zeros_like(v)
                vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
                vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
                vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

                vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
            else:
                vn = self.mesh.vn
            
            normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal[0])

            # rotated normal (where [0, 0, 1] always faces camera)
            viewcos = normal @ pose[:3, :3]

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).contiguous() # [H, W, 3]
        albedo = alpha * albedo + (1 - alpha) * bg_color

        # ssaa
        if ssaa != 1:
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            if 'depth' in optional_render_types:
                depth = scale_img_hwc(depth, (h0, w0))
            if 'normal' in optional_render_types:
                normal = scale_img_hwc(normal, (h0, w0))
                viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = albedo.clamp(0, 1)
        results['alpha'] = alpha
        if 'depth' in optional_render_types:
            results['depth'] = depth
        if 'normal' in optional_render_types:
            results['normal'] = (normal + 1) / 2
            results['viewcos'] = (viewcos + 1) / 2

        return results