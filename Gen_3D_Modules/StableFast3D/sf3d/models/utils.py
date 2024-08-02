import dataclasses
import importlib
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Num
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


class BaseModule(nn.Module):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError


def find_class(cls_string):
    module_string = ".".join(cls_string.split(".")[:-1])
    cls_name = cls_string.split(".")[-1]
    module = importlib.import_module(module_string, package=None)
    cls = getattr(module, cls_name)
    return cls


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    # Check if cfg.keys are in fields
    cfg_ = cfg.copy()
    keys = list(cfg_.keys())

    field_names = {f.name for f in dataclasses.fields(fields)}
    for key in keys:
        # This is helpful when swapping out modules from CLI
        if key not in field_names:
            print(f"Ignoring {key} as it's not supported by {fields}")
            cfg_.pop(key)
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg_)
    return scfg


EPS_DTYPE = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
    torch.float32: 1e-7,
    torch.float64: 1e-8,
}


def dot(x, y, dim=-1):
    return torch.sum(x * y, dim, keepdim=True)


def reflect(x, n):
    return x - 2 * dot(x, n) * n


def normalize(x, dim=-1, eps=None):
    if eps is None:
        eps = EPS_DTYPE[x.dtype]
    return F.normalize(x, dim=dim, p=2, eps=eps)


def tri_winding(tri: Float[Tensor, "*B 3 2"]) -> Float[Tensor, "*B 3 3"]:
    # One pad for determinant
    tri_sq = F.pad(tri, (0, 1), "constant", 1.0)
    det_tri = torch.det(tri_sq)
    tri_rev = torch.cat(
        (tri_sq[..., 0:1, :], tri_sq[..., 2:3, :], tri_sq[..., 1:2, :]), -2
    )
    tri_sq[det_tri < 0] = tri_rev[det_tri < 0]
    return tri_sq


def triangle_intersection_2d(
    t1: Float[Tensor, "*B 3 2"],
    t2: Float[Tensor, "*B 3 2"],
    eps=1e-12,
) -> Float[Tensor, "*B"]:  # noqa: F821
    """Returns True if triangles collide, False otherwise"""

    def chk_edge(x: Float[Tensor, "*B 3 3"]) -> Bool[Tensor, "*B"]:  # noqa: F821
        logdetx = torch.logdet(x.double())
        if eps is None:
            return ~torch.isfinite(logdetx)
        return ~(torch.isfinite(logdetx) & (logdetx > math.log(eps)))

    t1s = tri_winding(t1)
    t2s = tri_winding(t2)

    # Assume the triangles do not collide in the begging
    ret = torch.zeros(t1.shape[0], dtype=torch.bool, device=t1.device)
    for i in range(3):
        edge = torch.roll(t1s, i, dims=1)[:, :2, :]
        # Check if all points of triangle 2 lay on the external side of edge E.
        # If this is the case the triangle do not collide
        upd = (
            chk_edge(torch.cat((edge, t2s[:, 0:1]), 1))
            & chk_edge(torch.cat((edge, t2s[:, 1:2]), 1))
            & chk_edge(torch.cat((edge, t2s[:, 2:3]), 1))
        )
        # Here no collision is still True due to inversion
        ret = ret | upd

    for i in range(3):
        edge = torch.roll(t2s, i, dims=1)[:, :2, :]

        upd = (
            chk_edge(torch.cat((edge, t1s[:, 0:1]), 1))
            & chk_edge(torch.cat((edge, t1s[:, 1:2]), 1))
            & chk_edge(torch.cat((edge, t1s[:, 2:3]), 1))
        )
        # Here no collision is still True due to inversion
        ret = ret | upd

    return ~ret  # Do the inversion


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


def dilate_fill(img, mask, iterations=10):
    oldMask = mask.float()
    oldImg = img

    mask_kernel = torch.ones(
        (1, 1, 3, 3),
        dtype=oldMask.dtype,
        device=oldMask.device,
    )

    for i in range(iterations):
        newMask = torch.nn.functional.max_pool2d(oldMask, 3, 1, 1)

        # Fill the extension with mean color of old valid regions
        img_unfold = F.unfold(oldImg, (3, 3)).view(1, 3, 3 * 3, -1)
        mask_unfold = F.unfold(oldMask, (3, 3)).view(1, 1, 3 * 3, -1)
        new_mask_unfold = F.unfold(newMask, (3, 3)).view(1, 1, 3 * 3, -1)

        # Average color of the valid region
        mean_color = (img_unfold.sum(dim=2) / mask_unfold.sum(dim=2).clip(1)).unsqueeze(
            2
        )
        # Extend it to the new region
        fill_color = (mean_color * new_mask_unfold).view(1, 3 * 3 * 3, -1)

        mask_conv = F.conv2d(
            newMask, mask_kernel, padding=1
        )  # Get the sum for each kernel patch
        newImg = F.fold(
            fill_color, (img.shape[-2], img.shape[-1]), (3, 3)
        ) / mask_conv.clamp(1)

        diffMask = newMask - oldMask

        oldMask = newMask
        oldImg = torch.lerp(oldImg, newImg, diffMask)

    return oldImg


def float32_to_uint8_np(
    x: Float[np.ndarray, "*B H W C"],
    dither: bool = True,
    dither_mask: Optional[Float[np.ndarray, "*B H W C"]] = None,
    dither_strength: float = 1.0,
) -> Int[np.ndarray, "*B H W C"]:
    if dither:
        dither = (
            dither_strength * np.random.rand(*x[..., :1].shape).astype(np.float32) - 0.5
        )
        if dither_mask is not None:
            dither = dither * dither_mask
        return np.clip(np.floor((256.0 * x + dither)), 0, 255).astype(np.uint8)
    return np.clip(np.floor((256.0 * x)), 0, 255).astype(torch.uint8)


def convert_data(data):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        if data.dtype in [torch.float16, torch.bfloat16]:
            data = data.float()
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError(
            "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
            type(data),
        )


class ImageProcessor:
    def convert_and_resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        size: int,
    ):
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            pass

        batched = image.ndim == 4

        if not batched:
            image = image[None, ...]
        image = F.interpolate(
            image.permute(0, 3, 1, 2),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        if not batched:
            image = image[0]
        return image

    def __call__(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        size: int,
    ) -> Any:
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            image = self.convert_and_resize(image, size)
        else:
            if not isinstance(image, list):
                image = [image]
            image = [self.convert_and_resize(im, size) for im in image]
            image = torch.stack(image, dim=0)
        return image


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
