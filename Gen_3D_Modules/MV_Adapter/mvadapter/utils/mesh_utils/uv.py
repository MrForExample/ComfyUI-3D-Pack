from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .blend import PoissonBlendingSolver
from .camera import Camera
from .mesh import TexturedMesh
from .render import NVDiffRastContextWrapper, SimpleNormalization, render
from .utils import IMAGE_TYPE, get_clip_space_position, image_to_tensor


@dataclass
class UVPrecomputeOutput:
    height: int
    width: int
    uv_attr: torch.Tensor
    uv_mask: torch.Tensor
    uv_pos: torch.Tensor


def uv_precompute(
    ctx: NVDiffRastContextWrapper,
    mesh: TexturedMesh,
    height: int,
    width: int,
    clone_attr: bool = False,
) -> UVPrecomputeOutput:
    uv_clip = mesh.v_tex * 2.0 - 1.0
    uv_clip4 = torch.cat(
        (
            uv_clip,
            torch.zeros_like(uv_clip[..., 0:1]),
            torch.ones_like(uv_clip[..., 0:1]),
        ),
        dim=-1,
    )
    rast, _ = ctx.rasterize(uv_clip4[None], mesh.t_tex_idx, (height, width))
    uv_mask = rast[0, :, :, 3] > 0

    uv_pos, _ = ctx.interpolate(mesh.v_pos[None], rast, mesh.t_pos_idx)
    uv_pos = uv_pos[0]

    uv_precompute_output = UVPrecomputeOutput(
        height=height,
        width=width,
        uv_attr=mesh.texture.clone() if clone_attr else mesh.texture,
        uv_mask=uv_mask,
        uv_pos=uv_pos,
    )
    return uv_precompute_output


@dataclass
class UVRenderGeometryOutput:
    uv_pos_proj: torch.Tensor
    uv_pos_error: torch.Tensor
    uv_aoi_cos: torch.Tensor
    uv_pos_ndc: torch.Tensor
    view_mask: torch.Tensor
    view_normal: torch.Tensor
    view_aoi_cos: torch.Tensor
    view_position: torch.Tensor
    view_depth: torch.Tensor
    view_depth_grad: Optional[torch.Tensor] = None
    uv_depth_grad: Optional[torch.Tensor] = None
    view_attr: Optional[torch.Tensor] = None


def uv_render_geometry(
    ctx: NVDiffRastContextWrapper,
    mesh: TexturedMesh,
    cam: Camera,
    view_height: int,
    view_width: int,
    uv_precompute_output: UVPrecomputeOutput,
    grid_sample_mode="bilinear",
    compute_depth_grad: bool = False,
    depth_grad_dilation: int = 1,
    render_attr: bool = False,
) -> UVRenderGeometryOutput:
    batch_size = cam.c2w.shape[0]

    height, width, _ = uv_precompute_output.uv_pos.shape
    uv_pos_clip = get_clip_space_position(
        uv_precompute_output.uv_pos.view(-1, 3), cam.mvp_mtx
    ).view(batch_size, height, width, 4)
    uv_pos_ndc = uv_pos_clip[..., :2] / uv_pos_clip[..., 3:4]

    render_output = render(
        ctx,
        mesh,
        cam,
        view_height,
        view_width,
        render_attr=render_attr,
        render_depth=True,
        render_normal=True,
        depth_normalization_strategy=SimpleNormalization(
            scale=1.0, offset=0.0, clamp=False, bg_value=1e2
        ),
    )
    view_position = render_output.pos
    view_mask = render_output.mask
    view_normal = render_output.normal
    view_normal_cs = (
        view_normal[:, :, :, None, :] * cam.w2c[:, None, None, :3, :3]
    ).sum(-1)
    view_normal_cs = F.normalize(view_normal_cs, dim=-1, p=2)
    view_normal_cs[~render_output.mask] = render_output.normal[~render_output.mask]
    view_aoi_cos = (
        view_normal_cs
        * torch.as_tensor(
            [0.0, 0.0, 1.0], dtype=view_normal_cs.dtype, device=view_normal_cs.device
        )[None, None, None]
    ).sum(-1)
    view_aoi_cos = view_aoi_cos.clamp(0.0, 1.0)

    view_depth = render_output.depth
    if compute_depth_grad:
        grad_x_kernel = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            device=view_depth.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        grad_y_kernel = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            device=view_depth.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        view_depth_grad_x = F.conv2d(view_depth[:, None], grad_x_kernel, padding=1)
        view_depth_grad_y = F.conv2d(view_depth[:, None], grad_y_kernel, padding=1)
        view_depth_grad = (view_depth_grad_x**2 + view_depth_grad_y**2).sqrt()
        view_depth_grad = F.max_pool2d(
            view_depth_grad,
            depth_grad_dilation,
            stride=1,
            padding=depth_grad_dilation // 2,
        )

        uv_depth_grad = F.grid_sample(
            view_depth_grad, uv_pos_ndc, align_corners=False, mode=grid_sample_mode
        ).permute(0, 2, 3, 1)[..., 0]
    else:
        view_depth_grad = None
        uv_depth_grad = None

    if render_attr:
        view_attr = render_output.attr
    else:
        view_attr = None

    uv_pos_proj = F.grid_sample(
        view_position.permute(0, 3, 1, 2),
        uv_pos_ndc,
        align_corners=False,
        mode=grid_sample_mode,
    ).permute(0, 2, 3, 1)

    uv_pos_error = ((uv_pos_proj - uv_precompute_output.uv_pos) ** 2).sum(-1).sqrt()

    uv_aoi_cos = F.grid_sample(
        view_aoi_cos[..., None].permute(0, 3, 1, 2),
        uv_pos_ndc,
        align_corners=False,
        mode=grid_sample_mode,
    ).permute(0, 2, 3, 1)[..., 0]

    return UVRenderGeometryOutput(
        uv_pos_proj=uv_pos_proj,
        uv_pos_error=uv_pos_error,
        uv_aoi_cos=uv_aoi_cos,
        uv_pos_ndc=uv_pos_ndc,
        view_mask=view_mask,
        view_position=view_position,
        view_normal=view_normal,
        view_aoi_cos=view_aoi_cos,
        view_depth=view_depth,
        view_depth_grad=view_depth_grad,
        uv_depth_grad=uv_depth_grad,
        view_attr=view_attr,
    )


@dataclass
class UVRenderAttrOutput:
    uv_attr_proj: torch.Tensor
    uv_mask_proj: Optional[torch.Tensor]


def uv_render_attr(
    images: IMAGE_TYPE,
    uv_render_geometry_output: UVRenderGeometryOutput,
    masks: Optional[IMAGE_TYPE] = None,
    grid_sample_mode: str = "bilinear",
) -> UVRenderAttrOutput:
    images = image_to_tensor(images, device=uv_render_geometry_output.uv_pos_ndc.device)
    uv_attr_proj = F.grid_sample(
        images.permute(0, 3, 1, 2),
        uv_render_geometry_output.uv_pos_ndc,
        align_corners=False,
        mode=grid_sample_mode,
    ).permute(0, 2, 3, 1)

    if masks is not None:
        masks = image_to_tensor(
            masks, device=uv_render_geometry_output.uv_pos_ndc.device
        )
        if masks.ndim == 4:
            masks = masks.mean(-1)
        uv_mask_proj = F.grid_sample(
            masks[..., None].permute(0, 3, 1, 2),
            uv_render_geometry_output.uv_pos_ndc,
            align_corners=False,
            mode=grid_sample_mode,
        ).permute(0, 2, 3, 1)[..., 0]
    else:
        uv_mask_proj = None

    return UVRenderAttrOutput(uv_attr_proj=uv_attr_proj, uv_mask_proj=uv_mask_proj)


@dataclass
class UVBlendOutput:
    uv_attr_blend: Optional[torch.Tensor]
    uv_valid_mask: torch.Tensor
    uv_valid_mask_blend: torch.Tensor
    uv_blend_weight: torch.Tensor


class UVValidityStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
        self,
        uv_precompute_output: UVPrecomputeOutput,
        uv_render_geometry_output: UVRenderGeometryOutput,
        uv_render_attr_output: Optional[UVRenderAttrOutput],
    ) -> torch.BoolTensor:
        pass


class SimpleUVValidityStrategy(UVValidityStrategy):
    def __init__(
        self,
        pos_error_eps: float = 1e-3,
        aoi_cos_thresh: float = 0.1,
        mask_thresh: float = 0.9,
        depth_grad_thresh: Optional[float] = None,
        first_view_dominate: bool = False,
    ):
        self.pos_error_eps = pos_error_eps
        self.aoi_cos_thresh = aoi_cos_thresh
        self.mask_thresh = mask_thresh
        self.depth_grad_thresh = depth_grad_thresh
        self.first_view_dominate = first_view_dominate

    def __call__(
        self,
        uv_precompute_output: UVPrecomputeOutput,
        uv_render_geometry_output: UVRenderGeometryOutput,
        uv_render_attr_output: Optional[UVRenderAttrOutput],
    ) -> torch.BoolTensor:
        uv_pos_valid = uv_render_geometry_output.uv_pos_error < self.pos_error_eps
        uv_aoi_cos_valid = uv_render_geometry_output.uv_aoi_cos > self.aoi_cos_thresh
        uv_valid_mask = uv_pos_valid & uv_aoi_cos_valid

        if self.depth_grad_thresh is not None:
            if uv_render_geometry_output.uv_depth_grad is None:
                print(
                    "Warning: Depth gradient is not computed, depth gradient threshold is ignored."
                )
            else:
                uv_valid_mask &= (
                    uv_render_geometry_output.uv_depth_grad < self.depth_grad_thresh
                )

        uv_valid_mask &= uv_precompute_output.uv_mask

        if (
            uv_render_attr_output is not None
            and uv_render_attr_output.uv_mask_proj is not None
        ):
            uv_valid_mask &= uv_render_attr_output.uv_mask_proj > self.mask_thresh
        else:
            print("No view mask provided for UV blending, using all valid pixels")

        if self.first_view_dominate:
            uv_valid_mask[1:][
                uv_valid_mask[0:1].expand(uv_valid_mask.shape[0] - 1, -1, -1)
            ] = False

        return uv_valid_mask


class UVBlendWeightStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
        self,
        uv_precompute_output: UVPrecomputeOutput,
        uv_render_geometry_output: UVRenderGeometryOutput,
        uv_render_attr_output: Optional[UVRenderAttrOutput],
        uv_valid_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        pass


class ExponentialBlend(UVBlendWeightStrategy):
    def __init__(
        self,
        alpha: float = 1.0,
        normalization: str = "linear",
        view_weight: Optional[torch.FloatTensor] = None,
    ):
        self.alpha = alpha
        self.normalization = normalization
        self.view_weight = view_weight

    def __call__(
        self,
        uv_precompute_output: UVPrecomputeOutput,
        uv_render_geometry_output: UVRenderGeometryOutput,
        uv_render_attr_output: Optional[UVRenderAttrOutput],
        uv_valid_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        weight = uv_render_geometry_output.uv_aoi_cos
        weight = weight * uv_valid_mask.float()
        if self.view_weight is not None:
            weight = weight ** (self.alpha / self.view_weight[:, None, None].to(weight))
        else:
            weight = weight**self.alpha
        if self.normalization == "linear":
            weight_norm = (weight / weight.sum(axis=0, keepdim=True).clamp(1e-5)).clamp(
                0.0, 1.0
            )
        elif self.normalization == "softmax":
            weight[~uv_valid_mask] = -1e5
            weight_norm = F.softmax(weight, dim=0)
        return weight_norm


class RandomChoiceBlend(UVBlendWeightStrategy):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(
        self,
        uv_precompute_output: UVPrecomputeOutput,
        uv_render_geometry_output: UVRenderGeometryOutput,
        uv_render_attr_output: Optional[UVRenderAttrOutput],
        uv_valid_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        weight = uv_render_geometry_output.uv_aoi_cos
        weight = weight * uv_valid_mask.float()
        weight[weight > 0] = torch.rand_like(weight[weight > 0])
        weight_norm = (
            F.one_hot(weight.max(dim=0).indices, num_classes=weight.shape[0])
            .float()
            .permute(2, 0, 1)
        )
        return weight_norm


def uv_padding(attr: torch.FloatTensor, inside_mask: torch.BoolTensor, radius: int):
    if attr.min() < -1e-5 or attr.max() > 1 + 1e-5:
        print(
            "Warning: UV blending result is out of range [0, 1], force clamped before UV padding."
        )

    from .cv_ops import inpaint_torch

    attr_padded = inpaint_torch(attr.clamp(0.0, 1.0), ~inside_mask, radius)
    return attr_padded


def uv_blend(
    uv_precompute_output: UVPrecomputeOutput,
    uv_render_geometry_output: UVRenderGeometryOutput,
    uv_render_attr_output: Optional[UVRenderAttrOutput],
    uv_validity_strategy: UVValidityStrategy = SimpleUVValidityStrategy(),
    uv_blend_weight_strategy: UVBlendWeightStrategy = ExponentialBlend(),
    empty_value: float = 0.0,
    do_uv_padding: bool = True,
    uv_padding_radius: int = 3,
    pad_unseen_area: bool = False,
    poisson_blending: bool = False,
    pb_solver: Optional[PoissonBlendingSolver] = None,
    pb_num_iters: int = 1000,
    pb_keep_original_border: bool = True,
    pb_inplace: bool = False,
    pb_grad_mode: str = "src",
) -> UVBlendOutput:
    uv_valid_mask = uv_validity_strategy(
        uv_precompute_output, uv_render_geometry_output, uv_render_attr_output
    )
    uv_blend_weight = uv_blend_weight_strategy(
        uv_precompute_output,
        uv_render_geometry_output,
        uv_render_attr_output,
        uv_valid_mask,
    )
    uv_valid_mask_blend = uv_valid_mask.any(dim=0)

    if uv_render_attr_output is None:
        return UVBlendOutput(
            uv_attr_blend=None,
            uv_valid_mask=uv_valid_mask,
            uv_valid_mask_blend=uv_valid_mask_blend,
            uv_blend_weight=uv_blend_weight,
        )

    uv_attr_blend = (
        uv_render_attr_output.uv_attr_proj * uv_blend_weight[..., None]
    ).sum(axis=0)
    if poisson_blending:
        assert do_uv_padding
        assert pb_solver is not None
        uv_attr_blend_padded = uv_padding(
            uv_attr_blend, uv_valid_mask_blend, uv_padding_radius
        )
        if pb_keep_original_border:
            pb_tgt = uv_precompute_output.uv_attr
        else:
            uv_attr_hard_stitch = (
                uv_attr_blend * uv_valid_mask_blend[..., None].float()
                + uv_precompute_output.uv_attr
                * (~uv_valid_mask_blend)[..., None].float()
            )
            uv_attr_hard_stitch_padded = uv_padding(
                uv_attr_hard_stitch, uv_precompute_output.uv_mask, uv_padding_radius
            )
            pb_tgt = uv_attr_hard_stitch_padded

        uv_attr_blend = pb_solver(
            uv_attr_blend_padded,
            uv_valid_mask_blend,
            pb_tgt,
            pb_num_iters,
            inplace=pb_inplace,
            grad_mode=pb_grad_mode,
        )
    else:
        uv_attr_blend = (
            uv_attr_blend * uv_valid_mask_blend[..., None].float()
            + uv_precompute_output.uv_attr * (~uv_valid_mask_blend)[..., None].float()
        )

    if do_uv_padding:
        content_mask = (
            uv_valid_mask_blend if pad_unseen_area else uv_precompute_output.uv_mask
        )
        uv_attr_blend = uv_padding(uv_attr_blend, content_mask, uv_padding_radius)

    return UVBlendOutput(
        uv_attr_blend=uv_attr_blend,
        uv_valid_mask=uv_valid_mask,
        uv_valid_mask_blend=uv_valid_mask_blend,
        uv_blend_weight=uv_blend_weight,
    )
