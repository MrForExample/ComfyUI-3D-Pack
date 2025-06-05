import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .core import debug, find, info, warn
from .typing import *


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "negative":
        return lambda x: -x
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    if normalize:
        directions = F.normalize(directions, dim=-1)

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=False,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()


def tet_sdf_diff(
    vert_sdf: Float[Tensor, "Nv 1"], tet_edges: Integer[Tensor, "Ne 2"]
) -> Float[Tensor, ""]:
    sdf_f1x6x2 = vert_sdf[:, 0][tet_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float()
    )
    return sdf_diff


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        warn("Empty rays_indices!")
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([0]).to(ray_indices)
        t_end = torch.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end


def rays_intersect_bbox(
    rays_o: Float[Tensor, "N 3"],
    rays_d: Float[Tensor, "N 3"],
    radius: Float,
    near: Float = 0.0,
    valid_thresh: Float = 0.01,
    background: bool = False,
):
    input_shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    rays_d_valid = torch.where(
        rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d
    )
    if type(radius) in [int, float]:
        radius = torch.FloatTensor(
            [[-radius, radius], [-radius, radius], [-radius, radius]]
        ).to(rays_o.device)
    radius = (
        1.0 - 1.0e-3
    ) * radius  # tighten the radius to make sure the intersection point lies in the bounding box
    interx0 = (radius[..., 1] - rays_o) / rays_d_valid
    interx1 = (radius[..., 0] - rays_o) / rays_d_valid
    t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(interx0, interx1).amin(dim=-1)

    # check wheter a ray intersects the bbox or not
    rays_valid = t_far - t_near > valid_thresh

    t_near_valid, t_far_valid = t_near[rays_valid], t_far[rays_valid]
    global_near = t_near_valid.min().item()
    global_far = t_far_valid.max().item()

    t_near[torch.where(~rays_valid)] = 0.0
    t_far[torch.where(~rays_valid)] = 0.0

    t_near = t_near.view(*input_shape, 1)
    t_far = t_far.view(*input_shape, 1)
    rays_valid = rays_valid.view(*input_shape)

    return t_near, t_far, rays_valid


def get_plucker_rays(
    rays_o: Float[Tensor, "*N 3"], rays_d: Float[Tensor, "*N 3"]
) -> Float[Tensor, "*N 6"]:
    rays_o = F.normalize(rays_o, dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    return torch.cat([rays_o.cross(rays_d), rays_d], dim=-1)


def c2w_to_polar(c2w: Float[Tensor, "4 4"]) -> Tuple[float, float, float]:
    cam_pos = c2w[:3, 3]
    x, y, z = cam_pos.tolist()
    distance = cam_pos.norm().item()
    elevation = math.asin(z / distance)
    if abs(x) < 1.0e-5 and abs(y) < 1.0e-5:
        azimuth = 0
    else:
        azimuth = math.atan2(y, x)
        if azimuth < 0:
            azimuth += 2 * math.pi

    return elevation, azimuth, distance


def polar_to_c2w(
    elevation: float, azimuth: float, distance: float
) -> Float[Tensor, "4 4"]:
    """
    Compute L = p - C.
    Normalize L.
    Compute s = L x u. (cross product)
    Normalize s.
    Compute u' = s x L.
    rotation = [s, u, -l]
    """
    z = distance * math.sin(elevation)
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    l = -torch.as_tensor([x, y, z]).float()
    l = F.normalize(l, dim=0)
    u = torch.as_tensor([0.0, 0.0, 1.0]).float()
    s = l.cross(u)
    s = F.normalize(s, dim=0)
    u = s.cross(l)
    rot = torch.stack([s, u, -l], dim=0).T
    c2w = torch.zeros((4, 4), dtype=torch.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = torch.as_tensor([x, y, z])
    c2w[3, 3] = 1
    return c2w


def fourier_position_encoding(x, n_freq: int, dim: int):
    assert n_freq > 0
    input_shape = x.shape
    input_ndim = x.ndim
    if dim < 0:
        dim = input_ndim + dim
    bands = 2 ** torch.arange(n_freq, dtype=x.dtype, device=x.device)
    for i in range(dim + 1):
        bands = bands.unsqueeze(0)
    for i in range(input_ndim - dim - 1):
        bands = bands.unsqueeze(-1)
    x = x.view(*input_shape[: dim + 1], 1, *input_shape[dim + 1 :])
    x = torch.cat(
        [
            torch.sin(bands * x).reshape(
                *input_shape[:dim], -1, *input_shape[dim + 1 :]
            ),
            torch.cos(bands * x).reshape(
                *input_shape[:dim], -1, *input_shape[dim + 1 :]
            ),
        ],
        dim=dim,
    )
    return x
