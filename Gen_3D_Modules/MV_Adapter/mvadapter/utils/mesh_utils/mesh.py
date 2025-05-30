import io
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch import BoolTensor, FloatTensor

from .utils import tensor_to_image


def dot(x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
    return torch.sum(x * y, -1, keepdim=True)


@dataclass
class TexturedMesh:
    v_pos: torch.FloatTensor
    t_pos_idx: torch.LongTensor

    # texture coordinates
    v_tex: Optional[torch.FloatTensor] = None
    t_tex_idx: Optional[torch.LongTensor] = None

    # texture map
    texture: Optional[torch.FloatTensor] = None

    # vertices, faces after vertex merging
    _stitched_v_pos: Optional[torch.FloatTensor] = None
    _stitched_t_pos_idx: Optional[torch.LongTensor] = None

    _v_nrm: Optional[torch.FloatTensor] = None
    _v_tang: Optional[torch.FloatTensor] = None

    @property
    def v_nrm(self) -> torch.FloatTensor:
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tang(self) -> torch.FloatTensor:
        if self._v_tang is None:
            self._v_tang = self._compute_tangent()
        return self._v_tang

    def set_vertex_normal(self, v_nrm: torch.FloatTensor) -> None:
        assert v_nrm.shape == self.v_pos.shape
        self._v_nrm = v_nrm.to(self.v_pos)

    def set_stitched_mesh(
        self, v_pos: torch.FloatTensor, t_pos_idx: torch.LongTensor
    ) -> None:
        self._stitched_v_pos = v_pos
        self._stitched_t_pos_idx = t_pos_idx

    @property
    def stitched_v_pos(self) -> torch.FloatTensor:
        if self._stitched_v_pos is None:
            print("Warning: Stitched vertices not available, using original vertices!")
            return self.v_pos
        return self._stitched_v_pos

    @property
    def stitched_t_pos_idx(self) -> torch.LongTensor:
        if self._stitched_t_pos_idx is None:
            print("Warning: Stitched faces not available, using original faces!")
            return self.t_pos_idx
        return self._stitched_t_pos_idx

    @property
    def uv_size(self) -> Optional[int]:
        if self.texture is None:
            return None
        return self.texture.shape[0]

    def _compute_vertex_normal(self) -> torch.FloatTensor:
        if self._stitched_v_pos is None or self._stitched_t_pos_idx is None:
            print(
                "Warning: Stitched vertices and faces not available, computing vertex normals on original mesh, which can be erroneous!"
            )
            v_pos, t_pos_idx = self.v_pos, self.t_pos_idx
        else:
            v_pos, t_pos_idx = self._stitched_v_pos, self._stitched_t_pos_idx

        i0 = t_pos_idx[:, 0]
        i1 = t_pos_idx[:, 1]
        i2 = t_pos_idx[:, 2]

        v0 = v_pos[i0, :]
        v1 = v_pos[i1, :]
        v2 = v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_tangent(self) -> torch.FloatTensor:
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self._v_nrm)
        tansum = torch.zeros_like(self._v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(
            tangents - dot(tangents, self._v_nrm) * self._v_nrm, dim=1
        )

        v_tng = tangents

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_tng))

        return v_tng

    def to(self, device: Optional[str] = None):
        self.v_pos = self.v_pos.to(device)
        self.t_pos_idx = self.t_pos_idx.to(device)
        if self.v_tex is not None:
            self.v_tex = self.v_tex.to(device)
        if self.t_tex_idx is not None:
            self.t_tex_idx = self.t_tex_idx.to(device)
        if self.texture is not None:
            self.texture = self.texture.to(device)
        if self._stitched_v_pos is not None:
            self._stitched_v_pos = self._stitched_v_pos.to(device)
        if self._stitched_t_pos_idx is not None:
            self._stitched_t_pos_idx = self._stitched_t_pos_idx.to(device)
        if self._v_nrm is not None:
            self._v_nrm = self._v_nrm.to(device)
        if self._v_tang is not None:
            self._v_tang = self._v_tang.to(device)


@contextmanager
def mesh_use_texture(mesh: TexturedMesh, texture: torch.FloatTensor):
    texture_ = mesh.texture
    mesh.texture = texture
    try:
        yield
    finally:
        mesh.texture = texture_


def load_mesh(
    mesh_path: str,
    rescale: bool = False,
    move_to_center: bool = False,
    scale: float = 0.5,
    flip_uv: bool = True,
    merge_vertices: bool = True,
    default_uv_size: Optional[int] = None,
    shape_init_mesh_up: str = "+y",
    shape_init_mesh_front: str = "+x",
    front_x_to_y: bool = False,
    device: Optional[str] = None,
    return_transform: bool = False,
) -> TexturedMesh:
    if mesh_path.endswith(".npz"):

        class Mesh:
            vertices = None
            faces = None

        data = np.load(mesh_path)
        mesh = Mesh()
        mesh.vertices = data["vertices"]
        mesh.faces = data["faces"]
        merge_vertices = False
    else:
        scene = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
        elif isinstance(scene, trimesh.scene.Scene):
            mesh = trimesh.Trimesh()
            for obj in scene.geometry.values():
                mesh = trimesh.util.concatenate([mesh, obj])
        else:
            raise ValueError(f"Unknown mesh type at {mesh_path}.")

    vertex_normals = getattr(mesh, "vertex_normals", None)

    # move to center
    transform_offset = None
    if move_to_center:
        centroid = mesh.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid
        transform_offset = centroid  # record the offset

    # rescale
    transform_scale = None
    if rescale:
        max_scale = np.abs(mesh.vertices).max()
        mesh.vertices = mesh.vertices / max_scale * scale
        transform_scale = max_scale / scale

    dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    if shape_init_mesh_up not in dirs or shape_init_mesh_front not in dirs:
        raise ValueError(
            f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
        )
    if shape_init_mesh_up[1] == shape_init_mesh_front[1]:
        raise ValueError(
            "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
        )
    z_, x_ = (
        dir2vec[shape_init_mesh_up],
        dir2vec[shape_init_mesh_front],
    )
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)
    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T
    if vertex_normals is not None:
        vertex_normals = np.dot(mesh2std, vertex_normals.T).T
    if front_x_to_y:
        x = mesh.vertices[:, 1].copy()
        y = -mesh.vertices[:, 0].copy()
        mesh.vertices[:, 0] = x
        mesh.vertices[:, 1] = y
        if vertex_normals is not None:
            vx = vertex_normals[:, 1].copy()
            vy = -vertex_normals[:, 0].copy()
            vertex_normals[:, 0] = vx
            vertex_normals[:, 1] = vy

    v_pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64)

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        if flip_uv:
            v_tex[:, 1] = 1.0 - v_tex[:, 1]
        t_tex_idx = t_pos_idx.clone()
        if (
            default_uv_size is not None
            or getattr(mesh.visual.material, "baseColorTexture", None) is None
        ):
            assert default_uv_size is not None
            texture = torch.zeros(
                (default_uv_size, default_uv_size, 3), dtype=torch.float32
            )
        else:
            texture = torch.tensor(
                np.array(mesh.visual.material.baseColorTexture) / 255.0,
                dtype=torch.float32,
            )[..., :3]
    else:
        v_tex = None
        t_tex_idx = None
        texture = None

    textured_mesh = TexturedMesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        texture=texture,
    )

    if vertex_normals is not None:
        v_nrm = F.normalize(torch.tensor(vertex_normals, dtype=torch.float32), dim=-1)
        textured_mesh.set_vertex_normal(v_nrm)

    if merge_vertices and vertex_normals is None:
        # only merge vertices when vertex normals are not available
        mesh.merge_vertices(merge_tex=True)
        textured_mesh.set_stitched_mesh(
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64),
        )
    else:
        textured_mesh.set_stitched_mesh(textured_mesh.v_pos, textured_mesh.t_pos_idx)

    textured_mesh.to(device)

    if return_transform:
        return textured_mesh, transform_offset, transform_scale
    else:
        return textured_mesh


def replace_mesh_texture_and_save_trimesh(
    input_path: str,
    output_path: str,
    texture: Union[Image.Image, np.ndarray, torch.Tensor],
    metallic_roughness_texture: Optional[
        Union[Image.Image, np.ndarray, torch.Tensor]
    ] = None,
    normal_texture: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    texture_format: str = "JPEG",
    task_id: str = "",
    **kwargs,
) -> None:
    tmesh = trimesh.load(input_path, force="mesh", process=False)

    def convert_image(tensor):
        texture = tensor_to_image(tensor)
        texture.format = texture_format
        return texture

    texture = convert_image(texture)
    if metallic_roughness_texture is not None:
        metallic_roughness_texture = convert_image(metallic_roughness_texture)

    if normal_texture is not None:
        normal_texture = convert_image(normal_texture)

    if not isinstance(tmesh.visual.material, trimesh.visual.material.PBRMaterial):
        material = trimesh.visual.texture.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            roughnessFactor=0.9 if metallic_roughness_texture is None else 1.0,
            metallicFactor=0.0 if metallic_roughness_texture is None else 1.0,
            baseColorTexture=texture,
            metallicRoughnessTexture=metallic_roughness_texture,
            normalTexture=normal_texture,
        )
        tmesh.visual = trimesh.visual.TextureVisuals(
            uv=tmesh.visual.uv, image=texture, material=material
        )
    else:
        tmesh.visual.material.baseColorTexture = texture
        tmesh.visual.material.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
        if metallic_roughness_texture is None:
            tmesh.visual.material.roughnessFactor = 0.9
            tmesh.visual.material.metallicFactor = 0.0
        else:
            tmesh.visual.material.roughnessFactor = 1.0
            tmesh.visual.material.metallicFactor = 1.0
            tmesh.visual.material.metallicRoughnessTexture = metallic_roughness_texture
            tmesh.visual.material.normalTexture = normal_texture

    def p_func(tree, task_id_in=task_id):
        if "nodes" in tree and tree["nodes"][0]["name"].find("world") >= 0:
            tree["nodes"].pop(0)
        tree["asset"]["generator"] = "https://github.com/huanngzh/MV-Adapter"
        change_dict = {
            "nodes": "node",
            "meshes": "mesh",
            "materials": "mat",
            "images": "image",
        }
        for k, v in change_dict.items():
            if k in tree:
                for index in range(len(tree[k])):
                    tree[k][index]["name"] = f"mvadapter_{v}_{task_id_in}"

    tmesh.export(output_path, tree_postprocessor=p_func)


def replace_mesh_texture_and_save_gltflib(
    input_path: str,
    output_path: str,
    texture: Union[Image.Image, np.ndarray, torch.Tensor],
    metallic_roughness_texture: Optional[
        Union[Image.Image, np.ndarray, torch.Tensor]
    ] = None,
    normal_texture: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    texture_format: str = "JPEG",
    normal_strength: float = 1.0,
    task_id: str = "",
    **kwargs,
) -> None:
    from gltflib import GLTF
    from gltflib import Image as gltfImage
    from gltflib import NormalTextureInfo, Texture, TextureInfo

    def convert_image(tensor):
        texture = tensor_to_image(tensor)
        texture.format = texture_format
        return texture

    def add_texture(gltf: GLTF, image: Image.Image) -> int:
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer, format=texture_format)
        new_texture_data = bytearray(jpeg_buffer.getvalue())
        _, offset, bytelen = gltf._create_or_extend_glb_resource(new_texture_data)
        bufferView = gltf._create_embedded_image_buffer_view(offset, bytelen)
        image = gltfImage(mimeType="image/jpeg", bufferView=bufferView)
        if gltf.model.images:
            gltf.model.images.append(image)
        else:
            gltf.model.images = [image]
        texture = Texture(source=len(gltf.model.images) - 1)
        if gltf.model.textures:
            gltf.model.textures.append(texture)
        else:
            gltf.model.textures = [texture]
        return len(gltf.model.textures) - 1

    gltf = GLTF.load(input_path)

    texture = convert_image(texture)
    if metallic_roughness_texture is not None:
        metallic_roughness_texture = convert_image(metallic_roughness_texture)

    if normal_texture is not None:
        normal_texture = convert_image(normal_texture)

    pbr_material = gltf.model.materials[0].pbrMetallicRoughness

    pbr_material.baseColorTexture = TextureInfo(index=add_texture(gltf, texture))
    pbr_material.baseColorFactor = [1, 1, 1, 1]

    if metallic_roughness_texture is not None:
        pbr_material.metallicRoughnessTexture = TextureInfo(
            index=add_texture(gltf, metallic_roughness_texture)
        )
        pbr_material.metallicFactor = 1.0
        pbr_material.roughnessFactor = 1.0
    else:
        pbr_material.metallicFactor = 0.0
        pbr_material.roughnessFactor = 0.9

    if normal_texture is not None:
        gltf.model.materials[0].normalTexture = NormalTextureInfo(
            index=add_texture(gltf, normal_texture), scale=normal_strength
        )

    gltf.model.nodes[0].name = f"mvadapter_node_{task_id}"
    gltf.model.meshes[0].name = f"mvadapter_mesh_{task_id}"
    gltf.model.materials[0].name = f"mvadapter_material_{task_id}"
    gltf.model.asset.generator = "https://github.com/huanngzh/MV-Adapter"
    gltf.export(output_path)


def replace_mesh_texture_and_save(
    input_path: str,
    output_path: str,
    texture: Union[Image.Image, np.ndarray, torch.Tensor],
    metallic_roughness_texture: Optional[
        Union[Image.Image, np.ndarray, torch.Tensor]
    ] = None,
    normal_texture: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    normal_strength: float = 1.0,
    texture_format: str = "JPEG",
    task_id: str = "",
    backend: str = "trimesh",
) -> None:
    if backend == "trimesh":
        replace_mesh_texture_and_save_trimesh(
            input_path,
            output_path,
            texture,
            metallic_roughness_texture=metallic_roughness_texture,
            normal_texture=normal_texture,
            texture_format=texture_format,
            task_id=task_id,
        )
    elif backend == "gltflib":
        replace_mesh_texture_and_save_gltflib(
            input_path,
            output_path,
            texture,
            metallic_roughness_texture=metallic_roughness_texture,
            normal_texture=normal_texture,
            normal_strength=normal_strength,
            task_id=task_id,
        )
    else:
        raise NotImplementedError
