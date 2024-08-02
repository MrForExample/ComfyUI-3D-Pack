from __future__ import annotations

from typing import Any, Dict, Optional

import gpytoolbox
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from jaxtyping import Float, Integer
from torch import Tensor

from StableFast3D.sf3d.box_uv_unwrap import box_projection_uv_unwrap
from StableFast3D.sf3d.models.utils import dot


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self):
        if self._v_tex is None:
            self.unwrap_uv()
        return self._v_tex

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
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

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_pos_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        duv1 = tex[1] - tex[0]
        duv2 = tex[2] - tex[0]
        dpos1 = pos[1] - pos[0]
        dpos2 = pos[2] - pos[0]

        tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]

        denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        denom_safe = denom.clip(1e-6)
        tang = tng_nom / denom_safe

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        # Also normalize it. Here we do not normalize the individual triangles first so larger area
        # triangles influence the tangent space more
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def quad_remesh(
        self,
        quad_vertex_count: int = 12_000,
        quad_rosy: int = 4,
        quad_crease_angle: float = 0.0,
        quad_smooth_iter: int = 2,
        quad_align_to_boundaries: bool = False,
    ) -> Mesh:
        import pynim
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float32)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        new_vert, new_faces = pynim.remesh(
            v_pos,
            t_pos_idx,
            quad_vertex_count // 4,
            rosy=quad_rosy,
            posy=4,
            creaseAngle=quad_crease_angle,
            align_to_boundaries=quad_align_to_boundaries,
            smooth_iter=quad_smooth_iter,
            deterministic=True,
        )

        # Briefly load in trimesh
        mesh = trimesh.Trimesh(vertices=new_vert, faces=new_faces)

        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx).contiguous()

        # Create new mesh
        return Mesh(v_pos, t_pos_idx)

    def triangle_remesh(
        self,
        triangle_average_edge_length_multiplier: float = 1.0,
        triangle_remesh_steps: int = 10,
    ):
        edges = self.edges
        average_edge_length = (
            torch.linalg.norm(self.v_pos[edges[:, 0]] - self.v_pos[edges[:, 1]], dim=1)
            .mean()
            .item()
        )

        # Convert to numpy
        v_pos = self.v_pos.detach().cpu().numpy().astype(np.float64)
        t_pos_idx = self.t_pos_idx.detach().cpu().numpy().astype(np.int32)

        # Remesh
        v_remesh, f_remesh = gpytoolbox.remesh_botsch(
            v_pos,
            t_pos_idx,
            triangle_remesh_steps,
            float(average_edge_length * triangle_average_edge_length_multiplier),
        )

        # Convert back to torch
        v_pos = torch.from_numpy(v_remesh).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(f_remesh).to(self.t_pos_idx).contiguous()

        # Create new mesh
        return Mesh(v_pos, t_pos_idx)

    @torch.no_grad()
    def unwrap_uv(
        self,
        island_padding: float = 0.02,
    ) -> Mesh:
        uv, indices = box_projection_uv_unwrap(
            self.v_pos, self.v_nrm, self.t_pos_idx, island_padding
        )

        # Do store per vertex UVs.
        # This means we need to duplicate some vertices at the seams
        individual_vertices = self.v_pos[self.t_pos_idx].reshape(-1, 3)
        individual_faces = torch.arange(
            individual_vertices.shape[0],
            device=individual_vertices.device,
            dtype=self.t_pos_idx.dtype,
        ).reshape(-1, 3)
        uv_flat = uv[indices].reshape((-1, 2))
        # uv_flat[:, 1] = 1 - uv_flat[:, 1]

        self.v_pos = individual_vertices
        self.t_pos_idx = individual_faces
        self._v_tex = uv_flat
        self._v_nrm = self._compute_vertex_normal()
        self._v_tng = self._compute_vertex_tangent()

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges
