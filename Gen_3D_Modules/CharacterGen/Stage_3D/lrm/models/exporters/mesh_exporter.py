from dataclasses import dataclass, field
import os

import cv2
import numpy as np
import torch

from CharacterGen.Stage_3D import lrm
from ..renderers.base import BaseRenderer
from .base import Exporter, ExporterOutput
from ..mesh import Mesh
from ...utils.rasterize import NVDiffRasterizerContext
from ...utils.typing import *
from ...utils.misc import time_recorder as tr, time_recorder_enabled


def uv_padding_cpu(image, hole_mask, padding):
    uv_padding_size = padding
    inpaint_image = (
        cv2.inpaint(
            (image.detach().cpu().numpy() * 255).astype(np.uint8),
            (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
            uv_padding_size,
            cv2.INPAINT_TELEA,
        )
        / 255.0
    )
    return torch.from_numpy(inpaint_image).to(image)


def uv_padding_cvc(image, hole_mask, padding):
    import cvcuda

    torch_to_cvc = lambda x, layout: cvcuda.as_tensor(x, layout)
    cvc_to_torch = lambda x: torch.as_tensor(x.cuda())

    uv_padding_size = padding
    image_cvc = torch_to_cvc((image.detach() * 255).to(torch.uint8), "HWC")
    hole_mask_cvc = torch_to_cvc((hole_mask.detach() * 255).to(torch.uint8), "HW")
    inpaint_image = cvcuda.inpaint(image_cvc, hole_mask_cvc, uv_padding_size)
    inpaint_image = cvc_to_torch(inpaint_image) / 255.0
    return inpaint_image.to(image)


def uv_padding(image, hole_mask, padding):
    try:
        inpaint_image = uv_padding_cvc(image, hole_mask, padding)
    except:
        lrm.info(f"CVCUDA not available, fallback to CPU UV padding.")
        inpaint_image = uv_padding_cpu(image, hole_mask, padding)
    return inpaint_image


class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj"  # in ['obj', 'glb']
        visual: str = "uv"  # in ['uv', 'vertex']
        save_name: str = "model"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 1024
        texture_format: str = "jpg"
        uv_unwrap_method: str = "xatlas"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        smartuv_options: dict = field(default_factory=dict)
        uv_padding_size: int = 2
        subdivide: bool = False
        post_process: bool = False
        post_process_options: dict = field(default_factory=dict)
        context_type: str = "gl"
        output_path: str = "outputs"

    cfg: Config

    def configure(self, renderer: BaseRenderer) -> None:
        super().configure(renderer)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)
        if self.cfg.fmt == "obj-mtl":
            lrm.warn(
                f"fmt=obj-mtl is deprecated, please us fmt=obj and visual=uv instead."
            )
            self.cfg.fmt = "obj"
            self.cfg.visual = "uv"

        if self.cfg.fmt == "glb":
            assert self.cfg.visual in [
                "vertex",
                "uv-blender",
            ], "GLB format only supports visual=vertex and visual=uv-blender!"

    def get_geometry(self, scene_code: torch.Tensor) -> Mesh:
        tr.start("Surface extraction")
        mesh: Mesh = self.renderer.isosurface(scene_code)
        tr.end("Surface extraction")
        return mesh

    def get_texture_maps(
        self, scene_code: torch.Tensor, mesh: Mesh
    ) -> Dict[str, torch.Tensor]:
        assert mesh.has_uv
        # clip space transform
        uv_clip = mesh.v_tex * 2.0 - 1.0
        # pad to four component coordinate
        uv_clip4 = torch.cat(
            (
                uv_clip,
                torch.zeros_like(uv_clip[..., 0:1]),
                torch.ones_like(uv_clip[..., 0:1]),
            ),
            dim=-1,
        )
        # rasterize
        rast, _ = self.ctx.rasterize_one(
            uv_clip4,
            mesh.t_tex_idx,
            (self.cfg.texture_size, self.cfg.texture_size),
        )

        hole_mask = ~(rast[:, :, 3] > 0)

        # Interpolate world space position
        gb_pos, _ = self.ctx.interpolate_one(
            mesh.v_pos, rast[None, ...], mesh.t_pos_idx
        )
        gb_pos = gb_pos[0]

        # Sample out textures from MLP
        tr.start("Query color")
        geo_out = self.renderer.query(scene_code, points=gb_pos)
        tr.end("Query color")
        mat_out = self.renderer.material.export(points=gb_pos, **geo_out)

        textures = {}
        tr.start("UV padding")
        if "albedo" in mat_out:
            textures["map_Kd"] = uv_padding(
                mat_out["albedo"], hole_mask, self.cfg.uv_padding_size
            )
        else:
            lrm.warn(
                "save_texture is True but no albedo texture found, using default white texture"
            )
        if "metallic" in mat_out:
            textures["map_Pm"] = uv_padding(
                mat_out["metallic"], hole_mask, self.cfg.uv_padding_size
            )
        if "roughness" in mat_out:
            textures["map_Pr"] = uv_padding(
                mat_out["roughness"], hole_mask, self.cfg.uv_padding_size
            )
        if "bump" in mat_out:
            textures["map_Bump"] = uv_padding(
                mat_out["bump"], hole_mask, self.cfg.uv_padding_size
            )
        tr.end("UV padding")
        return textures

    def __call__(self, names, scene_codes) -> List[ExporterOutput]:
        outputs = []
        for name, scene_code in zip(names, scene_codes):
            mesh = self.get_geometry(scene_code)
            if self.cfg.post_process:
                tr.start("Mesh post-processing")
                mesh = mesh.post_process(self.cfg.post_process_options)
                tr.end("Mesh post-processing")
            if self.cfg.visual == "uv":
                output = self.export_model_with_mtl(
                    name, self.cfg.fmt, scene_code, mesh
                )
            elif self.cfg.visual == "vertex":
                output = self.export_model(name, self.cfg.fmt, scene_code, mesh)
            elif self.cfg.visual == "uv-blender":
                output = self.export_model_blender(name, self.cfg.fmt, scene_code, mesh)
            else:
                raise ValueError(f"Unsupported visual format: {self.cfg.visual}")
            outputs.append(output)
        return outputs

    def export_model_with_mtl(
        self, name: str, fmt: str, scene_code: torch.Tensor, mesh: Mesh
    ) -> ExporterOutput:
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(
                self.cfg.uv_unwrap_method,
                self.cfg.xatlas_chart_options,
                self.cfg.xatlas_pack_options,
                self.cfg.smartuv_options,
            )

        if self.cfg.save_texture:
            lrm.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"

            with time_recorder_enabled():
                textures = self.get_texture_maps(scene_code, mesh)
                params.update(textures)
        os.makedirs(self.cfg.output_path, exist_ok=True)
        np.savez(f"{self.cfg.output_path}/tex_info.npz", v_tex=mesh.v_tex.cpu().numpy(), t_tex_idx=mesh.t_tex_idx.cpu().numpy())
        return ExporterOutput(
            save_name=f"{self.cfg.save_name}-{name}.{fmt}", save_type=fmt, params=params
        )

    def export_model(
        self, name: str, fmt: str, scene_code, mesh: Mesh
    ) -> ExporterOutput:
        params = {
            "mesh": mesh,
            "save_mat": False,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(
                self.cfg.uv_unwrap_method,
                self.cfg.xatlas_chart_options,
                self.cfg.xatlas_pack_options,
                self.cfg.smartuv_options,
            )

        if self.cfg.save_texture:
            lrm.info("Exporting textures ...")
            geo_out = self.renderer.query(scene_code, points=mesh.v_pos)
            mat_out = self.renderer.material.export(points=mesh.v_pos, **geo_out)

            if "albedo" in mat_out:
                mesh.set_vertex_color(mat_out["albedo"])
                params["save_vertex_color"] = True
            else:
                lrm.warn(
                    "save_texture is True but no albedo texture found, not saving vertex color"
                )

        return ExporterOutput(
            save_name=f"{self.cfg.save_name}-{name}.{fmt}", save_type=fmt, params=params
        )
