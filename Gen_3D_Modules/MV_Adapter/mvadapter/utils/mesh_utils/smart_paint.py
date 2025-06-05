import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import Camera, get_camera
from .mesh import (
    TexturedMesh,
    load_mesh,
    mesh_use_texture,
    replace_mesh_texture_and_save,
)
from .projection import CameraProjection
from .render import NVDiffRastContextWrapper, render
from .utils import image_to_tensor, make_image_grid, tensor_to_image
from .uv import (
    ExponentialBlend,
    NVDiffRastContextWrapper,
    PoissonBlendingSolver,
    SimpleUVValidityStrategy,
    uv_blend,
    uv_padding,
    uv_precompute,
    uv_render_attr,
    uv_render_geometry,
)


class SmartPainter:
    def __init__(self, device: str):
        self.device = device
        self.cam_proj = CameraProjection(
            pb_backend="torch-cuda", bg_remover=None, device=device
        )
        self.ctx = NVDiffRastContextWrapper(device=self.device)

    def __call__(
        self,
        mod_name: str,
        mesh: TexturedMesh,
        inpaint_func: callable,
        uv_texture: torch.FloatTensor,
        uv_inpaint_mask: torch.BoolTensor,
        max_view_score_thresh: float = 0.02,
        min_rounds: int = 3,
        max_rounds: int = 8,
        uv_padding_end: bool = True,
        debug_dir: Optional[str] = None,
        debug_visualize_details: bool = False,
    ):
        def make_view_selection_cam_params():
            elevation_anchors = range(-60, 61, 15)
            azimuth_anchors = range(0, 360, 30)
            distance_anchors = [1.2]
            fovy_deg_anchors = [40]
            params = list(
                product(
                    elevation_anchors,
                    azimuth_anchors,
                    distance_anchors,
                    fovy_deg_anchors,
                )
            )
            params = list(zip(*params))
            return params

        elevation_deg, azimuth_deg, distance, fovy_deg = (
            make_view_selection_cam_params()
        )
        view_selection_cams = get_camera(
            elevation_deg=list(elevation_deg),
            azimuth_deg=list(azimuth_deg),
            distance=list(distance),
            fovy_deg=list(fovy_deg),
            perturb_camera_position=0.1,
            device=self.device,
        )

        texture_update = uv_texture.clone()
        uv_valid_mask_update = ~uv_inpaint_mask.clone()
        score_map_update = torch.zeros_like(uv_valid_mask_update, dtype=torch.float32)
        score_map_update[uv_valid_mask_update] = 1.0

        max_view_score = 1
        i = 0
        while i < min_rounds or (
            max_view_score > max_view_score_thresh and i < max_rounds
        ):
            score_map_image = score_map_update.float()[:, :, None].repeat_interleave(
                3, dim=-1
            )
            render_size = 256
            with mesh_use_texture(mesh, score_map_image):
                render_out = render(
                    self.ctx,
                    mesh,
                    view_selection_cams,
                    height=render_size,
                    width=render_size,
                    attr_background=1.0,
                    texture_filter_mode="nearest",
                )

            def get_view_aoi_cos(render_out, cameras):
                view_normal = render_out.normal
                view_normal_cs = (
                    view_normal[:, :, :, None, :] * cameras.w2c[:, None, None, :3, :3]
                ).sum(-1)
                view_normal_cs = F.normalize(view_normal_cs, dim=-1, p=2)
                view_normal_cs[~render_out.mask] = 0
                view_aoi_cos = (
                    (
                        view_normal_cs
                        * torch.as_tensor(
                            [0.0, 0.0, 1.0],
                            dtype=view_normal_cs.dtype,
                            device=view_normal_cs.device,
                        )[None, None, None]
                    )
                    .sum(-1)
                    .clamp(0, 1)
                )
                return view_aoi_cos

            view_aoi_cos = get_view_aoi_cos(render_out, view_selection_cams)
            if debug_dir is not None and debug_visualize_details:
                make_image_grid(tensor_to_image(render_out.attr, batched=True)).save(
                    os.path.join(debug_dir, f"{mod_name}_render_score_{i:02d}.jpg")
                )
                make_image_grid(tensor_to_image(view_aoi_cos, batched=True)).save(
                    os.path.join(
                        debug_dir, f"{mod_name}_render_view_aoi_cos_{i:02d}.jpg"
                    )
                )
            view_score = [
                (
                    ((attr < 1e-3) & (aoi_cos > 0.1)).sum().item()
                    + (
                        ((attr > 1e-3) & (aoi_cos > 0.1)).float()
                        * (aoi_cos - attr - 0.3).clamp_min(0)
                    )
                    .sum()
                    .item()
                )
                / render_size**2
                for attr, aoi_cos in zip(render_out.attr[..., 0], view_aoi_cos)
            ]

            max_view_score = np.max(view_score)
            best_view = np.argmax(view_score)

            inpaint_render_size = 1024
            best_cam = view_selection_cams[best_view : best_view + 1]

            def shrink_mask(mask: torch.BoolTensor, radius: int) -> torch.BoolTensor:
                return (
                    (
                        -torch.nn.functional.max_pool2d(
                            -(mask[None, None].float()),
                            kernel_size=radius * 2 + 1,
                            stride=1,
                            padding=radius,
                        )
                    )
                    .squeeze()
                    .bool()
                )

            def enlarge_mask(mask: torch.BoolTensor, radius: int) -> torch.BoolTensor:
                return (
                    torch.nn.functional.max_pool2d(
                        mask[None, None].float(),
                        kernel_size=radius * 2 + 1,
                        stride=1,
                        padding=radius,
                    )
                    .squeeze()
                    .bool()
                )

            def blur_mask(mask: torch.BoolTensor, radius: int) -> torch.FloatTensor:
                # Create a Gaussian kernel
                x = torch.arange(
                    -radius, radius + 1, dtype=torch.float32, device=mask.device
                )
                kernel = torch.exp(-(x**2) / (2 * (radius / 2) ** 2))
                kernel = kernel[None, None, None]
                kernel = kernel * kernel.transpose(2, 3)
                kernel = kernel / kernel.sum()
                # Apply convolution to blur the mask
                blurred_mask = F.conv2d(
                    mask[None, None].float(), kernel, padding=radius
                )
                return blurred_mask.squeeze()

            def get_occulusion_boundary(
                view_depth: torch.FloatTensor, dilation: int, thresh: float
            ) -> torch.BoolTensor:
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
                view_depth_grad_x = F.conv2d(
                    view_depth[None, None], grad_x_kernel, padding=1
                )
                view_depth_grad_y = F.conv2d(
                    view_depth[None, None], grad_y_kernel, padding=1
                )
                view_depth_grad = (view_depth_grad_x**2 + view_depth_grad_y**2).sqrt()
                occ_boundary = (view_depth_grad > thresh)[0][0]
                if dilation > 0:
                    occ_boundary = enlarge_mask(occ_boundary, dilation)
                return occ_boundary

            with mesh_use_texture(mesh, score_map_image):
                render_out = render(
                    self.ctx,
                    mesh,
                    best_cam,
                    height=inpaint_render_size,
                    width=inpaint_render_size,
                    attr_background=1.0,
                    texture_filter_mode="nearest",
                )

            view_aoi_cos = get_view_aoi_cos(render_out, best_cam)
            inpaint_mask = (render_out.attr[0, :, :, 0] < 1e-3) | (
                view_aoi_cos[0] - render_out.attr[0, :, :, 0] > 0.3
            )
            occ_boundary = get_occulusion_boundary(
                render_out.depth[0], dilation=0, thresh=0.1
            )
            # shrink to remove uv seam
            # enlarge the mask gives better results
            # remove the occlusion boundary to avoid bleeding
            inpaint_mask = enlarge_mask(shrink_mask(inpaint_mask, 3), 5) & ~occ_boundary
            if debug_dir is not None and debug_visualize_details:
                tensor_to_image(occ_boundary).save(
                    os.path.join(debug_dir, f"{mod_name}_occ_boundary_{i:02d}.jpg")
                )

            with mesh_use_texture(mesh, texture_update):
                inpaint_image = render(
                    self.ctx,
                    mesh,
                    best_cam,
                    height=inpaint_render_size,
                    width=inpaint_render_size,
                    texture_filter_mode="linear",
                ).attr[0]

            # according to https://github.com/advimman/lama/issues/262
            # blur the sharp edge mask gives better results
            # but it gives worse results in our case
            # inpaint_mask = blur_mask(inpaint_mask, 1)

            inpaint_mask_input = inpaint_mask.float()[None, None]
            inpaint_image_input = inpaint_image.permute(2, 0, 1)[None]
            inpaint_result = inpaint_func(inpaint_image_input, inpaint_mask_input)[
                0
            ].permute(1, 2, 0)

            if debug_dir is not None:
                make_image_grid(
                    [
                        tensor_to_image(inpaint_image),
                        tensor_to_image(inpaint_mask),
                        tensor_to_image(inpaint_result),
                    ],
                    rows=1,
                ).save(
                    os.path.join(debug_dir, f"{mod_name}_inpaint_result_{i:02d}.jpg")
                )
            with mesh_use_texture(mesh, texture_update):
                proj_out = self.cam_proj(
                    inpaint_result[None],
                    mesh,
                    best_cam,
                    masks=inpaint_mask[None].float(),
                    from_scratch=False,
                    poisson_blending=False,
                    depth_grad_dilation=3,
                    uv_exp_blend_alpha=3,
                    aoi_cos_valid_threshold=0.1,
                    uv_size=mesh.uv_size,
                    uv_padding=True,
                    iou_rejection_threshold=None,
                    return_dict=True,
                )
            uv_proj_inpaint, uv_valid_mask_inpaint = (
                proj_out.uv_proj,
                proj_out.uv_proj_mask,
            )
            if debug_dir is not None and debug_visualize_details:
                tensor_to_image(uv_valid_mask_inpaint).save(
                    os.path.join(debug_dir, f"{mod_name}_uv_inpaint_mask_{i:02d}.jpg")
                )
            texture_update = uv_proj_inpaint
            uv_valid_mask_update = uv_valid_mask_inpaint | uv_valid_mask_update
            score_map_inpaint = torch.where(
                uv_valid_mask_inpaint,
                proj_out.uv_aoi_cos[0],
                torch.zeros_like(proj_out.uv_aoi_cos[0]),
            )
            score_map = torch.max(score_map_update, score_map_inpaint)
            if debug_dir is not None and debug_visualize_details:
                make_image_grid(
                    [
                        tensor_to_image(score_map_update),
                        tensor_to_image(score_map_inpaint),
                        tensor_to_image(score_map),
                    ]
                ).save(os.path.join(debug_dir, f"{mod_name}_score_map_{i:02d}.jpg"))
            score_map_update = score_map
            i += 1

        if uv_padding_end:
            texture_update = uv_padding(texture_update, uv_valid_mask_update, 3)

        return texture_update, uv_valid_mask_update
