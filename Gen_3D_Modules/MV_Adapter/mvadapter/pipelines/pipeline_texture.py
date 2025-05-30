import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spandrel import ModelLoader

from mvadapter.utils import image_to_tensor, make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import (
    Camera,
    CameraProjection,
    NVDiffRastContextWrapper,
    SmartPainter,
    TexturedMesh,
    get_camera,
    get_orthogonal_camera,
    load_mesh,
    render,
    replace_mesh_texture_and_save,
)
from mvadapter.utils.mesh_utils.mesh_process import process_raw


def clear():
    torch.cuda.empty_cache()


@contextmanager
def mesh_use_texture(mesh: TexturedMesh, texture: torch.FloatTensor):
    texture_ = mesh.texture
    mesh.texture = texture
    try:
        yield
    finally:
        mesh.texture = texture_


@dataclass
class ModProcessConfig:
    view_upscale: bool = False
    view_upscale_factor: int = 2
    inpaint_mode: str = "uv"  # in ["none", "uv", "view"]
    view_inpaint_max_view_score_thresh: float = 0.02
    view_inpaint_min_rounds: int = 4
    view_inpaint_max_rounds: int = 8
    view_inpaint_uv_padding_end: bool = True


@dataclass
class TexturePipelineOutput:
    shaded_model_save_path: Optional[str] = None
    pbr_model_save_path: Optional[str] = None


class TexturePipeline:
    def __init__(self, upscaler_ckpt_path: str, inpaint_ckpt_path: str, device: str):
        self.device = device
        self.ctx = NVDiffRastContextWrapper(device=self.device)
        self.cam_proj = CameraProjection(
            pb_backend="torch-cuda", bg_remover=None, device=self.device
        )
        if upscaler_ckpt_path is not None:
            self.upscaler = ModelLoader().load_from_file(upscaler_ckpt_path)
            self.upscaler.to(self.device).eval().half()
        if inpaint_ckpt_path is not None:
            self.inpainter = ModelLoader().load_from_file(inpaint_ckpt_path)
            self.inpainter.to(self.device).eval()

        self.smart_painter = SmartPainter(self.device)

    def load_packed_images(self, packed_image_path: Optional[str]) -> List[Image.Image]:
        if packed_image_path is None:
            return None
        packed_image = Image.open(packed_image_path)
        images = np.array_split(np.array(packed_image), 6, axis=1)
        images = [Image.fromarray(im) for im in images]
        return images

    def maybe_upscale_image(
        self,
        tensor: Optional[torch.FloatTensor],
        upscale: bool,
        upscale_factor: int,
        batched: bool = False,
    ) -> Optional[torch.FloatTensor]:
        if upscale:
            with torch.no_grad():
                tensor = tensor.permute(0, 3, 1, 2)
                if batched:
                    tensor = self.upscaler(tensor.half()).float()
                else:
                    tensor = torch.concat(
                        [
                            self.upscaler(im.unsqueeze(0).half()).float()
                            for im in tensor
                        ],
                        dim=0,
                    )
                tensor = tensor.clamp(0, 1).permute(0, 2, 3, 1)
            clear()
        return tensor

    def view_inpaint(
        self,
        mod_name: str,
        mesh: TexturedMesh,
        uv_proj: torch.FloatTensor,
        uv_valid_mask: torch.BoolTensor,
        config: ModProcessConfig,
        debug_dir: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        def inpaint_func(
            image: torch.FloatTensor, mask: torch.FloatTensor
        ) -> torch.FloatTensor:
            with torch.no_grad():
                return self.inpainter(image.to(torch.float32), mask.to(torch.float32))

        return self.smart_painter(
            mod_name,
            mesh,
            inpaint_func,
            uv_proj,
            ~uv_valid_mask,
            max_view_score_thresh=config.view_inpaint_max_view_score_thresh,
            min_rounds=config.view_inpaint_min_rounds,
            max_rounds=config.view_inpaint_max_rounds,
            uv_padding_end=config.view_inpaint_uv_padding_end,
            debug_dir=debug_dir,
            debug_visualize_details=False,
        )

    def __call__(
        self,
        mesh_path: str,
        save_dir: str,
        save_name: str = "default",
        # mesh load settings
        move_to_center: bool = False,
        front_x: bool = True,
        # uv unwarp
        uv_unwarp: bool = False,
        preprocess_mesh: bool = False,
        # projection
        uv_size: int = 4096,
        # modes
        rgb_path: Optional[str] = None,
        rgb_process_config: ModProcessConfig = ModProcessConfig(),
        base_color_path: Optional[str] = None,
        base_color_process_config: ModProcessConfig = ModProcessConfig(),
        orm_path: Optional[str] = None,
        orm_process_config: ModProcessConfig = ModProcessConfig(),
        normal_path: Optional[str] = None,
        normal_strength: float = 1.0,
        normal_process_config: ModProcessConfig = ModProcessConfig(),
        # inpaint
        uv_inpaint_use_network: bool = False,
        view_inpaint_include_occlusion_boundary: bool = False,
        poisson_reprojection: bool = False,
        # camera
        camera_projection_type: str = "ORTHO",
        camera_elevation_deg: List[float] = [0, 0, 0, 0, 89.99, -89.99],
        camera_azimuth_deg: List[float] = [0, 90, 180, 270, 180, 180],
        camera_distance: float = 1.0,
        camera_ortho_scale: float = 1.1,
        camera_fov_deg: float = 40,
        # debug
        debug_mode: bool = False,
    ):
        clear()

        if debug_mode:
            debug_dir = os.path.join(save_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

        if uv_unwarp:
            file_suffix = os.path.splitext(mesh_path)[-1]
            mesh_path_new = mesh_path.replace(file_suffix, f"_unwarp{file_suffix}")
            process_raw(mesh_path, mesh_path_new, preprocess=preprocess_mesh)
            mesh_path = mesh_path_new

        mesh: TexturedMesh = load_mesh(
            mesh_path,
            rescale=True,
            move_to_center=move_to_center,
            front_x_to_y=front_x,
            default_uv_size=uv_size,
            device=self.device,
        )

        # projection
        if camera_projection_type == "PERSP":
            raise NotImplementedError
        elif camera_projection_type == "ORTHO":
            cameras = get_orthogonal_camera(
                elevation_deg=camera_elevation_deg,
                distance=[camera_distance] * 6,
                left=-camera_ortho_scale / 2,
                right=camera_ortho_scale / 2,
                bottom=-camera_ortho_scale / 2,
                top=camera_ortho_scale / 2,
                azimuth_deg=[x - 90 for x in camera_azimuth_deg],  # -y as front
                device=self.device,
            )

        mod_kwargs = {
            "rgb": (rgb_path, rgb_process_config),
            "base_color": (base_color_path, base_color_process_config),
            "orm": (orm_path, orm_process_config),
            "normal": (normal_path, normal_process_config),
        }
        mod_uv_image, mod_uv_tensor = {}, {}
        for mod_name, (mod_path, mod_process_config) in mod_kwargs.items():
            if mod_path is None:
                mod_uv_image[mod_name] = None
                mod_uv_tensor[mod_name] = None
                continue
            mod_images = self.load_packed_images(mod_path)
            mod_tensor = image_to_tensor(mod_images, device=self.device)
            mod_tensor = self.maybe_upscale_image(
                mod_tensor,
                mod_process_config.view_upscale,
                mod_process_config.view_upscale_factor,
            )
            if mod_process_config.view_upscale and debug_mode:
                make_image_grid(tensor_to_image(mod_tensor, batched=True), rows=1).save(
                    os.path.join(debug_dir, f"{mod_name}_upscaled.jpg")
                )

            if mod_name == "normal":
                _, height, width, _ = mod_tensor.shape
                render_out = render(
                    self.ctx,
                    mesh,
                    cameras,
                    height=height,
                    width=width,
                    render_attr=False,
                    render_depth=False,
                    render_normal=True,
                    render_tangent=True,
                )

                # compute UV tangent space
                vN = render_out.normal
                vT = render_out.tangent
                vB = torch.cross(vN, vT, dim=-1)
                tang_space = F.normalize(torch.stack([vT, vB, vN], dim=-2), dim=-1)

                # compute geometry tangent space
                vGN = vN
                vGT = torch.as_tensor(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [-1, 0, 0],
                        [-1, 0, 0],
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )[:, None, None, :]
                vGB = torch.cross(vGN, vGT, dim=-1)
                vGT = torch.cross(vGB, vGN, dim=-1)
                geo_tang_space = F.normalize(
                    torch.stack([vGT, vGB, vGN], dim=-2), dim=-1
                )

                # restore world-space normal from geometry tangent space
                mod_tensor = mod_tensor * 2 - 1
                mod_tensor = F.normalize(
                    (
                        mod_tensor[:, :, :, None, :]
                        * geo_tang_space.permute(0, 1, 2, 4, 3)
                    ).sum(-1),
                    dim=-1,
                )

                # bake world-space normal to UV tangent space
                mod_tensor = F.normalize(
                    (mod_tensor[:, :, :, None, :] * tang_space).sum(-1), dim=-1
                )
                mod_tensor = (mod_tensor * 0.5 + 0.5).clamp(0, 1)

                uv_proj, uv_valid_mask = self.cam_proj(
                    mod_tensor,
                    mesh,
                    cameras,
                    from_scratch=mod_process_config.inpaint_mode != "none",
                    poisson_blending=False,
                    depth_grad_dilation=5,
                    uv_exp_blend_alpha=3,
                    uv_exp_blend_view_weight=torch.as_tensor([1, 1, 1, 1, 1, 1]),
                    aoi_cos_valid_threshold=0.2,
                    uv_size=uv_size,
                    return_uv_projection_mask=True,
                )
                uv_proj[~uv_valid_mask] = torch.as_tensor([0.5, 0.5, 1]).to(uv_proj)
            else:
                # TODO: tweak depth_grad_dilation
                cam_proj_out = self.cam_proj(
                    mod_tensor,
                    mesh,
                    cameras,
                    from_scratch=mod_process_config.inpaint_mode != "none",
                    poisson_blending=False,
                    depth_grad_dilation=5,
                    depth_grad_threshold=0.1,
                    uv_exp_blend_alpha=3,
                    uv_exp_blend_view_weight=torch.as_tensor([1, 1, 1, 1, 1, 1]),
                    aoi_cos_valid_threshold=0.2,
                    uv_size=uv_size,
                    uv_padding=not uv_inpaint_use_network,
                    return_dict=True,
                )
                uv_proj, uv_valid_mask, uv_depth_grad = (
                    cam_proj_out.uv_proj,
                    cam_proj_out.uv_proj_mask,
                    cam_proj_out.uv_depth_grad,
                )
                if uv_inpaint_use_network:
                    uv_inpaint_mask_input = 1 - uv_valid_mask[None, None].float()
                    uv_inpaint_image_input = uv_proj[None].permute(0, 3, 1, 2)
                    with torch.no_grad():
                        uv_inpaint_result = self.inpainter(
                            uv_inpaint_image_input, uv_inpaint_mask_input
                        )[0].permute(1, 2, 0)
                    clear()
                    if debug_mode:
                        make_image_grid(
                            [
                                tensor_to_image(uv_proj),
                                tensor_to_image(uv_valid_mask),
                                tensor_to_image(uv_inpaint_result),
                            ]
                        ).save(os.path.join(debug_dir, f"{mod_name}_uv_inpaint.jpg"))
                    uv_proj = uv_inpaint_result.contiguous()

                if mod_process_config.inpaint_mode == "view":
                    if view_inpaint_include_occlusion_boundary:
                        uv_max_depth_grad = uv_depth_grad.max(dim=0)[0]
                        uv_valid_mask = uv_valid_mask & (uv_max_depth_grad < 0.1)
                    uv_proj, uv_valid_mask = self.view_inpaint(
                        mod_name,
                        mesh,
                        uv_proj,
                        uv_valid_mask,
                        mod_process_config,
                        debug_dir=debug_dir if debug_mode else None,
                    )

                if poisson_reprojection:
                    # up and down
                    mesh.texture = uv_proj
                    uv_proj = self.cam_proj(
                        mod_tensor[4:5],
                        mesh,
                        cameras[4:5],
                        from_scratch=False,
                        poisson_blending=True,
                        pb_keep_original_border=True,
                        depth_grad_dilation=5,
                        uv_exp_blend_alpha=3,
                        uv_exp_blend_view_weight=torch.as_tensor([1, 1]),
                        aoi_cos_valid_threshold=0.2,
                        uv_size=uv_size,
                        uv_padding=True,
                        return_dict=False,
                    )
                    # front, sides and back
                    mesh.texture = uv_proj
                    uv_proj = self.cam_proj(
                        mod_tensor[0:4],
                        mesh,
                        cameras[0:4],
                        from_scratch=False,
                        poisson_blending=True,
                        pb_keep_original_border=True,
                        depth_grad_dilation=5,
                        uv_exp_blend_alpha=3,
                        uv_exp_blend_view_weight=torch.as_tensor([1, 1, 1, 1]),
                        aoi_cos_valid_threshold=0.2,
                        uv_size=uv_size,
                        uv_padding=True,
                        return_dict=False,
                    )

                if mod_name == "orm":
                    uv_proj[:, :, 0] = 1.0

            mod_uv_image[mod_name] = tensor_to_image(uv_proj)
            mod_uv_tensor[mod_name] = uv_proj
            clear()

        shaded_model_save_path = None
        if mod_uv_image["rgb"] is not None:
            shaded_model_save_path = os.path.join(save_dir, f"{save_name}_shaded.glb")
            replace_mesh_texture_and_save(
                mesh_path,
                shaded_model_save_path,
                texture=mod_uv_image["rgb"],
                backend="gltflib",
                task_id=save_name,
            )
        pbr_model_save_path = None
        if mod_uv_image["base_color"] is not None:
            pbr_model_save_path = os.path.join(save_dir, f"{save_name}_pbr.glb")
            replace_mesh_texture_and_save(
                mesh_path,
                pbr_model_save_path,
                texture=mod_uv_image["base_color"],
                metallic_roughness_texture=mod_uv_image["orm"],
                normal_texture=mod_uv_image["normal"],
                normal_strength=normal_strength,
                backend="gltflib",
                task_id=save_name,
            )

        clear()

        return TexturePipelineOutput(
            shaded_model_save_path=shaded_model_save_path,
            pbr_model_save_path=pbr_model_save_path,
        )
