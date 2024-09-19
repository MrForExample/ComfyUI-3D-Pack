from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..utils.typing import *
from ..utils.base import BaseModule
from ..utils.ops import trunc_exp
from ..utils.ops import scale_tensor
from .networks import MLP
from einops import rearrange, reduce

from mesh_processer.mesh_utils import construct_list_of_gs_attributes, write_gs_ply

inverse_sigmoid = lambda x: np.log(x / (1 - x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(w2c.device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width)

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

    def to_ply(self):
        
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy())
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        return write_gs_ply(xyz, normals, f_dc, f_rest, opacities, scale, rotation, construct_list_of_gs_attributes(features_dc, features_rest, self.scaling, self.rotation))

class GSLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 128
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        self.out_layers = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
            layer = nn.Linear(self.cfg.in_channels, out_ch)

            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))

            self.out_layers.append(layer)

    def forward(self, x, pts):
        ret = {}
        for k, layer in zip(self.cfg.feature_channels.keys(), self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.cfg.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.cfg.clip_scaling)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.cfg.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if self.cfg.restrict_offset:
                    max_step = 1.2 / 32
                    v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pts if self.cfg.xyz_offset else pts
            ret[k] = v

        return GaussianModel(**ret)

class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        mlp_network_config: Optional[dict] = None
        gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        scaling_modifier: float = 1.0
        random_background: bool = False
        radius: float = 1.0
        feature_reduction: str = "concat"
        projection_feature_dim: int = 773
        background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        if self.cfg.feature_reduction == "mean":
            mlp_in = 80
        elif self.cfg.feature_reduction == "concat":
            mlp_in = 80 * 3
        else:
            raise NotImplementedError
        mlp_in = mlp_in + self.cfg.projection_feature_dim
        if self.cfg.mlp_network_config is not None:
            self.mlp_net = MLP(mlp_in, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
        else:
            self.cfg.gs_out.in_channels = mlp_in
        self.gs_net = GSLayer(self.cfg.gs_out)

    def forward_gs(self, x, p):
        if self.cfg.mlp_network_config is not None:
            x = self.mlp_net(x)
        return self.gs_net(x, p)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        ):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.cfg.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.cfg.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
        else:
            shs = gs.shs

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        
        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": bg_color
        }
        
        if ret_mask:
            mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.height),
                image_width=int(viewpoint_camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=mask_bg_color,
                scale_modifier=self.cfg.scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform.float(),
                sh_degree=0,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                rendered_mask, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    # shs = ,
                    colors_precomp = torch.ones_like(means3D),
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret
    
    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]

        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError
        
        if not batched:
            out = out.squeeze(0)

        return out

    def forward_single_batch(
        self,
        gs_hidden_features: Float[Tensor, "Np Cp"],
        query_points: Float[Tensor, "Np 3"],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "3"]],
    ):
        gs: GaussianModel = self.forward_gs(gs_hidden_features, query_points)
        out_list = []
       
        for c2w, intrinsic in zip(c2ws, intrinsics):
            out_list.append(self.forward_single_view(
                                gs, 
                                Camera.from_c2w(c2w, intrinsic, height, width),
                                background_color
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs

        return out

    def forward(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        **kwargs):
        batch_size = gs_hidden_features.shape[0]
        out_list = []
        gs_hidden_features = self.query_triplane(query_points, gs_hidden_features)
        if additional_features is not None:
            gs_hidden_features = torch.cat([gs_hidden_features, additional_features], dim=-1)

        for b in range(batch_size):
            out_list.append(self.forward_single_batch(
                gs_hidden_features[b],
                query_points[b],
                c2w[b],
                intrinsic[b],
                height, width,
                background_color[b] if background_color is not None else None))

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v
        return out
        