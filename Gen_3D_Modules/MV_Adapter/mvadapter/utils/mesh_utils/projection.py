import glob
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .blend import PoissonBlendingSolver
from .camera import Camera, get_camera, get_orthogonal_camera
from .mesh import TexturedMesh, load_mesh, replace_mesh_texture_and_save
from .render import NVDiffRastContextWrapper, SimpleNormalization, render
from .seg import SegmentationModel
from .utils import (
    IMAGE_TYPE,
    LIST_TYPE,
    image_to_tensor,
    make_image_grid,
    tensor_to_image,
)
from .uv import (
    ExponentialBlend,
    SimpleUVValidityStrategy,
    uv_blend,
    uv_precompute,
    uv_render_attr,
    uv_render_geometry,
)
from .warp import compute_warp_field


@dataclass
class CameraProjectionOutput:
    uv_proj: torch.FloatTensor
    uv_proj_mask: torch.BoolTensor
    uv_depth_grad: Optional[torch.FloatTensor]
    uv_aoi_cos: Optional[torch.FloatTensor]


class CameraProjection:
    def __init__(
        self,
        pb_backend: str,
        bg_remover: Optional[SegmentationModel],
        device: str,
        context_type: str = "gl",
    ) -> None:
        self.pb_solver = PoissonBlendingSolver(pb_backend, device)
        self.ctx = NVDiffRastContextWrapper(device, context_type)
        self.bg_remover = bg_remover
        self.device = device

    def __call__(
        self,
        images: IMAGE_TYPE,
        mesh: TexturedMesh,
        cam: Optional[Camera] = None,
        fovy_deg: Optional[LIST_TYPE] = None,
        masks: Optional[IMAGE_TYPE] = None,
        remove_bg: bool = False,
        c2w: Optional[torch.FloatTensor] = None,
        elevation_deg: Optional[LIST_TYPE] = None,
        distance: Optional[LIST_TYPE] = None,
        azimuth_deg: Optional[LIST_TYPE] = None,
        num_views: Optional[int] = None,
        uv_size: int = 2048,
        warp_images: bool = False,
        images_background: Optional[float] = None,
        iou_rejection_threshold: Optional[float] = 0.8,
        aoi_cos_valid_threshold: float = 0.3,
        depth_grad_dilation: int = 5,
        depth_grad_threshold: float = 0.1,
        uv_exp_blend_alpha: float = 6,
        uv_exp_blend_view_weight: Optional[torch.FloatTensor] = None,
        poisson_blending: bool = True,
        pb_num_iters: int = 1000,
        pb_keep_original_border: bool = True,
        from_scratch: bool = False,
        uv_padding: bool = True,
        return_uv_projection_mask: bool = False,
        return_dict: bool = False,
    ) -> Optional[torch.FloatTensor]:
        images_pt = image_to_tensor(images, device=self.device)
        assert images_pt.ndim == 4
        Nv, H, W, _ = images_pt.shape

        if masks is not None:
            masks_pt = image_to_tensor(masks, device=self.device)
        else:
            if remove_bg:
                assert self.bg_remover is not None
                masks_pt = self.bg_remover(images_pt)
            else:
                masks_pt = None

        if masks_pt is not None and masks_pt.ndim == 4:
            masks_pt = masks_pt.mean(-1)

        if cam is None:
            cam = get_camera(
                elevation_deg,
                distance,
                fovy_deg,
                azimuth_deg,
                num_views,
                c2w,
                aspect_wh=W / H,
                device=self.device,
            )
        uv_precompute_output = uv_precompute(
            self.ctx, mesh, height=uv_size, width=uv_size
        )
        uv_render_geometry_output = uv_render_geometry(
            self.ctx,
            mesh,
            cam,
            view_height=H,
            view_width=W,
            uv_precompute_output=uv_precompute_output,
            compute_depth_grad=True,
            depth_grad_dilation=depth_grad_dilation,
        )

        # IoU rejection
        if masks_pt is not None and iou_rejection_threshold is not None:
            given_masks = (masks_pt > 0.5).float()
            render_masks = uv_render_geometry_output.view_mask.float()
            intersection = given_masks * render_masks
            union = given_masks + render_masks - intersection
            iou = intersection.sum((1, 2)) / union.sum((1, 2))
            iou_min = iou.min()
            print(f"Debug: Per view IoU: {iou.tolist()}")
            if iou_min < iou_rejection_threshold:
                print(
                    f"Warning: Minimum view IoU {iou_min} below threshold {iou_rejection_threshold}, skipping camera projection!"
                )
                return None

        if warp_images:
            # TODO: clean code
            assert images_background is not None
            render_attr = render(
                self.ctx,
                mesh,
                cam,
                height=H,
                width=W,
                render_attr=True,
                attr_background=images_background,
            ).attr
            images_pt = compute_warp_field(
                self.ctx.ctx,
                images_pt,
                render_attr,
                n_grid=10,
                optim_res=[64, 128],
                optim_step_per_res=20,
                lambda_reg=2.0,
                temp_dir="debug_warp",
                verbose=False,
                device=self.device,
            )

        uv_render_attr_output = uv_render_attr(
            images=images_pt,
            masks=masks_pt,
            uv_render_geometry_output=uv_render_geometry_output,
        )
        uv_blend_output = uv_blend(
            uv_precompute_output,
            uv_render_geometry_output,
            uv_render_attr_output,
            uv_validity_strategy=SimpleUVValidityStrategy(
                aoi_cos_thresh=aoi_cos_valid_threshold,
                depth_grad_thresh=depth_grad_threshold,
            ),
            uv_blend_weight_strategy=ExponentialBlend(
                alpha=uv_exp_blend_alpha, view_weight=uv_exp_blend_view_weight
            ),
            empty_value=1.0,
            do_uv_padding=uv_padding,
            pad_unseen_area=from_scratch,
            poisson_blending=poisson_blending,
            pb_solver=self.pb_solver,
            pb_num_iters=pb_num_iters,
            pb_keep_original_border=pb_keep_original_border,
        )

        if return_dict:
            # recommonded new way to get return value
            return CameraProjectionOutput(
                uv_proj=uv_blend_output.uv_attr_blend,
                uv_proj_mask=uv_blend_output.uv_valid_mask_blend,
                uv_depth_grad=uv_render_geometry_output.uv_depth_grad,
                uv_aoi_cos=uv_render_geometry_output.uv_aoi_cos,
            )
        else:
            if return_uv_projection_mask:
                return (
                    uv_blend_output.uv_attr_blend,
                    uv_blend_output.uv_valid_mask_blend,
                )
            return uv_blend_output.uv_attr_blend
