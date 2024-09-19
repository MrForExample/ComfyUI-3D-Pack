import math
import random
import numpy as np
from torchtyping import TensorType
from plyfile import PlyData, PlyElement

import torch
from torch import nn
import torch.nn.functional as F
    
from kornia.geometry.conversions import (
    quaternion_to_rotation_matrix,
)

from kiui.op import inverse_sigmoid

from shared_utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from mesh_processer.mesh import Mesh, PointCloud
from mesh_processer.mesh_utils import construct_list_of_gs_attributes, write_gs_ply, read_gs_ply, K_nearest_neighbors_func

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def random_point_in_triangle(v0, v1, v2):
    """
    Given three vertices v0, v1, v2, sample a point uniformly in the triangle
    Algorithm Reference:
    https://math.stackexchange.com/questions/538458/how-to-sample-points-on-a-triangle-surface-in-3d
    https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain
    """
    
    r1 = random.random()
    r2 = random.random()

    s1 = math.sqrt(r1)
    return v0 * (1.0 - s1) + v1 * (1.0 - r2) * s1 + v2 * r2 * s1#

def find_points_within_radius(query_points, vertex_points, d):
    """
    Finds vertex points within a given radius for each query point.

    Args:
        query_points (torch.Tensor): Tensor of shape (P1, 3) representing query points.
        vertex_points (torch.Tensor): Tensor of shape (P2, 3) representing vertex points.
        d (float): Radius within which to search for vertex points.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains indices of vertex points
                         within the radius for the corresponding query point.
    """
    num_query_points = query_points.shape[0]

    # Calculate pairwise distances between query points and vertex points
    distances = torch.norm(query_points[:, None] - vertex_points, dim=2)

    # Create a mask indicating which vertex points are within the radius
    mask = distances <= d

    # Collect indices of vertex points within the radius for each query point
    result = []
    for i in range(num_query_points):
        indices_within_radius = torch.nonzero(mask[i]).squeeze().tolist()
        result.append(indices_within_radius)

    return result

def distance_to_gaussian_surface(points, svec, rotmat, query):
    """
    Calculate the radius of the gaussian along the direction, which determined by offset between points and query
    Calculation using Mahalanobis distance

    Args:
        points (Tensor): gaussians position of shape (*, 3)
        svec (Tensor): gaussians scale vector of shape (*, 3)
        rotmat : the rotation matrix of shape (*, 3, 3)
        query (Tensor): query positions of shape (*, 3)

    Returns:
        Distance of shape (*, 1): 
    """
    offset_dir = query - points

    offset_dir = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), offset_dir)
    offset_dir = F.normalize(offset_dir, dim=-1)
    z = offset_dir[..., 2]
    y = offset_dir[..., 1]
    x = offset_dir[..., 0]
    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)
    cos_theta = z
    sin_theta = r_xy
    cos_phi = x / r_xy
    sin_phi = y / r_xy

    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    #r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2 * sin_theta**2 # same as: squared_dist = np.dot(diff.T, np.dot(np.linalg.inv(covariance), diff))

    """
        Alternatively:
        
            def gaussian_squared_distance(query_point, mean, covariance):
                # Calculate the difference vector
                diff = query_point - mean

                # Compute the squared Mahalanobis distance
                inv_covariance = np.linalg.inv(covariance)
                squared_dist = np.dot(diff.T, np.dot(inv_covariance, diff))

                return squared_dist
                
            # Example usage
            mean = np.array([x0, y0, z0])  # Gaussian mean (origin position)
            covariance = np.diag([sigma_x**2, sigma_y**2, sigma_z**2])  # Covariance matrix

            query_point = np.array([x_q, y_q, z_q])  # Query point
            squared_dist_to_gaussian = gaussian_squared_distance(query_point, mean, covariance)

            print(f"Squared distance to Gaussian surface: {squared_dist_to_gaussian:.4f}")
    """

    return torch.sqrt(r2 + 1e-10)

def qvec2rotmat_batched(qvec: TensorType["N", 4]):
    return quaternion_to_rotation_matrix(qvec)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.init_xyz = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # diffuse color
        self._features_rest = torch.empty(0) # spherical harmonic coefficients
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0) # (w, x, y, z)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_gaussians_num(self):
        return self.init_xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def rotmat(self):
        return qvec2rotmat_batched(self._rotation)
    
    @property
    def get_xyz_offset(self):
        return self.init_xyz - self._xyz
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 

        return occ
    
    def get_covariance(self, scaling_modifier = 1, gaussain_idx = None):
        if gaussain_idx is None:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        else:
            return self.covariance_activation(self.get_scaling[gaussain_idx], scaling_modifier, self._rotation[gaussain_idx])

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : PointCloud, spatial_lr_scale : float = 1):
        from simple_knn._C import distCUDA2
        
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.init_xyz = fused_point_cloud.clone().detach()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def get_points_cloud(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        shs = self.get_features.detach().cpu().numpy()
        pcd = PointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)
        return pcd

    def to_ply(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        return write_gs_ply(xyz, normals, f_dc, f_rest, opacities, scale, rotation, construct_list_of_gs_attributes(self._features_dc, self._features_rest, self._scaling, self._rotation))

    def create_from_ply(self, plydata):
        xyz, features_dc, features_extra, opacities, scales, rots = read_gs_ply(plydata)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.init_xyz = self._xyz.clone().detach()
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree
        
    def create_from_mesh(self, mesh, num_pts):
        """
            Sample gaussians uniformly for each face, this method can capture the structure of the 3D meshes well even for the fine structure e.g. fingers
        """
        # In .obj file, first index is 1 instead of zero
        vertices = np.concatenate([[[0., 0., 0.]], mesh.v.detach().cpu().numpy()]) 
        #normals = np.concatenate([[[0., 0., 0.]], mesh.n.detach().cpu().numpy()]) 
        
        all_points = []
        num_pts_per_face = math.ceil(num_pts / mesh.f.shape[0])
        for triangle_face in mesh.f.detach().cpu().numpy():
            v0 = vertices[triangle_face[0]]
            v1 = vertices[triangle_face[1]]
            v2 = vertices[triangle_face[2]]
            for i in range(num_pts_per_face):
                all_points.append(random_point_in_triangle(v0, v1, v2))
                
        num_pts = len(all_points)
                
        xyz = np.array(all_points)
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = PointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        self.create_from_pcd(pcd, 10)
        
    def create_from_uv_data(self, uv_grids_data):
        """
        Idealy this method should capture the most information for the texture, 
        but depends on the UV of 3D meshes, fine structure like fingers could have too little gaussians to form the proper structure,
        also the number of gaussians is too big even for 1K texture
        """
        if uv_grids_data is not None:
            uv_coords, uv_coords_3d, uv_normals_3d = uv_grids_data
            
        num_pts = len(uv_coords_3d)
        xyz = np.array(uv_coords_3d)
        shs = np.random.random((num_pts, 3)) / 255.0    
        pcd = PointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        self.create_from_pcd(pcd, 10)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        self.init_xyz = self.init_xyz[valid_points_mask]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        new_params = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }
        
        self.densify_with_new_params(new_params)
        
    def densify_with_new_params(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # Update reference position for gaussians
        self.init_xyz = torch.cat((self.init_xyz, self.init_xyz[selected_pts_mask].repeat(N,1)), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        # Update reference position for gaussians
        self.init_xyz = torch.cat((self.init_xyz, self.init_xyz[selected_pts_mask]), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        
    def densify_by_compatnes_with_idx(self, idx):
        nn_scaling = self._scaling[idx]
        nn_rotmat = self.rotmat[idx]
        nn_pos = self._xyz[idx]

        nn_gaussian_surface_dist = distance_to_gaussian_surface(
            nn_pos, nn_scaling, nn_rotmat, self._xyz
        )
        gaussian_surface_dist = distance_to_gaussian_surface(
            self._xyz, self._scaling, self.rotmat, nn_pos
        )

        dist_to_nn = torch.norm(nn_pos - self._xyz, dim=-1)
        selected_pts_mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn
        new_direction = (nn_pos - self._xyz) / dist_to_nn[..., None]
        new_xyz = (
            self._xyz
            + new_direction * (dist_to_nn + gaussian_surface_dist - nn_gaussian_surface_dist)[..., None]
            / 2.0
        )[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_raw_alpha = self._opacity[selected_pts_mask]
        new_raw_svec = self.scaling_inverse_activation(
            torch.ones_like(self._scaling[selected_pts_mask]) 
            * (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[selected_pts_mask][..., None]
            / 6.0
        )
        # print(torch.ones_like(self.svec.data[selected_pts_mask]).shape)
        # print(
        #     (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[selected_pts_mask].shape
        # )
        new_qvec = self._rotation[selected_pts_mask]
        
        # Update reference position for gaussians
        self.init_xyz = torch.cat((self.init_xyz, self.init_xyz[selected_pts_mask]), dim=0)

        new_params = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_raw_alpha,
            "scaling": new_raw_svec,
            "rotation": new_qvec
        }
        
        return new_params

    def densify_by_compatness(self, K=1):        
        _, idx = K_nearest_neighbors_func(self._xyz, K=K+1)
        
        new_params_list = []
        for i in range(K):
            new_params = self.densify_by_compatnes_with_idx(idx[:, i])
            new_params_list.append(new_params)
        new_params = {}
        for key in new_params_list[0].keys():
            new_params[key] = torch.cat([p[key] for p in new_params_list], dim=0)

        self.densify_with_new_params(new_params)
        
        #num_densified = new_params["xyz"].shape[0]
        
    def densify_by_clone_and_split(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        self.densify_by_clone_and_split(max_grad, extent)
        self.prune(min_opacity, extent, max_screen_size)
        
    def densify_and_prune_by_compatness(self, K, min_opacity, extent, max_screen_size):
        self.densify_by_compatness(K)
        self.prune(min_opacity, extent, max_screen_size)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def prune(self, min_opacity, extent, max_screen_size, max_offset=0):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            #big_points_os = self.get_xyz_offset.norm(dim=1) > max_offset
            #prune_mask = torch.logical_or(torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws), big_points_os)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
class GaussianSplattingRenderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if isinstance(input, Mesh):
            # load from 3D mesh
            self.gaussians.create_from_mesh(input, num_pts)
        elif isinstance(input, PlyData):
            self.gaussians.create_from_ply(input)
        elif isinstance(input, PointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        elif input is None:
            # init from random point cloud
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = PointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
        else:
            self.gaussians.create_from_uv_data(input)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        gaussain_idx=None,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        gaussians_xyz = self.gaussians.get_xyz
        gaussians_features = self.gaussians.get_features
        gaussians_opacity = self.gaussians.get_opacity
        
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        gaussians_scales = None
        gaussians_rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier, gaussain_idx)
        else:
            gaussians_scales = self.gaussians.get_scaling
            gaussians_rotations = self.gaussians.get_rotation
            
        if gaussain_idx is not None:
            gaussians_xyz = gaussians_xyz[gaussain_idx]
            gaussians_features = gaussians_features[gaussain_idx]
            gaussians_opacity = gaussians_opacity[gaussain_idx]
            if cov3D_precomp is None:
                gaussians_scales = gaussians_scales[gaussain_idx]
                gaussians_rotations = gaussians_rotations[gaussain_idx]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gaussians_xyz,
                dtype=gaussians_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.gaussians_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = gaussians_xyz - viewpoint_camera.camera_center.repeat(
                    gaussians_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = gaussians_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=gaussians_xyz,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=gaussians_opacity,
            scales=gaussians_scales,
            rotations=gaussians_rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image, # rgb
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }