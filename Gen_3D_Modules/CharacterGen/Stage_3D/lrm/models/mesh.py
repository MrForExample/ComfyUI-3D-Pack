from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from CharacterGen.Stage_3D import lrm
from ..utils.ops import dot
from ..utils.typing import *
from ..utils.misc import time_recorder as tr, time_recorder_enabled


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = None
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v) -> None:
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]) -> Mesh:
        if self.requires_grad:
            lrm.debug("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        lrm.debug(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        lrm.debug(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        lrm.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            lrm.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    def subdivide(self):
        if self.requires_grad:
            lrm.debug("Mesh is differentiable, not performing subdivision")
            return self

        import trimesh

        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        mesh.subdivide_loop()

        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        subdivided_mesh = Mesh(v_pos, t_pos_idx)

        if len(self.extras) > 0:
            subdivided_mesh.extras = self.extras
            lrm.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )

        return subdivided_mesh

    def post_process(self, options):
        if self.requires_grad:
            lrm.debug("Mesh is differentiable, not performing post processing")
            return self

        from extern.mesh_process.MeshProcess import process_mesh

        v_pos, t_pos_idx = process_mesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
            **options,
        )

        v_pos = torch.from_numpy(v_pos).to(self.v_pos).contiguous()
        t_pos_idx = torch.from_numpy(t_pos_idx).to(self.t_pos_idx).contiguous()

        processed_mesh = Mesh(v_pos, t_pos_idx)

        if len(self.extras) > 0:
            processed_mesh.extras = self.extras
            lrm.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )

        return processed_mesh

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
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self):
        if self._t_tex_idx is None:
            self._v_tex, self._t_tex_idx = self._unwrap_uv()
        return self._t_tex_idx

    @property
    def v_rgb(self):
        return self._v_rgb

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
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

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
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv_open3d(
        self
    ):
        import open3d as o3d
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self.v_pos.detach().cpu().numpy())
        mesh.triangle.indices = o3d.core.Tensor(self.t_pos_idx.cpu().numpy())
        mesh.compute_uvatlas(size=1024)
        texture_uvs = torch.from_numpy(mesh.triangle.texture_uvs.numpy()).reshape(-1, 2).cuda()
        indices = torch.arange(self.t_pos_idx.shape[0] * 3).reshape(-1, 3).to(torch.int64).cuda()
        # Add a wood texture and visualize
        return texture_uvs, indices
    
    def _unwrap_uv_xatlas(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        lrm.info("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = (
            torch.from_numpy(
                vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = (
            torch.from_numpy(
                indices.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        return uvs, indices

    def _unwrap_uv_smartuv(self, options: dict = {}):
        from extern.mesh_process.MeshProcess import (
            mesh_to_bpy,
            get_uv_from_bpy,
            bpy_context,
            bpy_export,
        )
        from CharacterGen.Stage_3D.lrm.utils.misc import time_recorder as tr

        v_pos, t_pos_idx = self.v_pos.cpu().numpy(), self.t_pos_idx.cpu().numpy()
        with bpy_context():
            mesh_bpy = mesh_to_bpy("_", v_pos, t_pos_idx)
            v_tex = get_uv_from_bpy(mesh_bpy, **options).astype(np.float32)

        assert v_tex.shape[0] == self.t_pos_idx.shape[0] * 3

        t_tex_idx = torch.arange(
            self.t_pos_idx.shape[0] * 3, device=self.t_pos_idx.device, dtype=torch.long
        ).reshape(-1, 3)

        """
        # super efficient de-duplication
        v_tex_u_uint32 = v_tex[..., 0].view(np.uint32)
        v_tex_v_uint32 = v_tex[..., 1].view(np.uint32)
        v_hashed = (v_tex_u_uint32.astype(np.uint64) << 32) | v_tex_v_uint32
        v_hashed = torch.from_numpy(v_hashed.view(np.int64)).to(self.v_pos.device)

        v_tex = torch.from_numpy(v_tex).to(
            device=self.v_pos.device, dtype=torch.float32
        )
        t_pos_idx_f3 = torch.arange(
            self.t_pos_idx.shape[0] * 3, device=self.t_pos_idx.device, dtype=torch.long
        ).reshape(-1, 3)
        v_pos_f3 = self.v_pos[self.t_pos_idx].reshape(-1, 3)

        # super efficient de-duplication
        v_hashed_dedup, inverse_indices = torch.unique(v_hashed, return_inverse=True)
        dedup_size, full_size = v_hashed_dedup.shape[0], inverse_indices.shape[0]
        indices = torch.scatter_reduce(
            torch.full(
                [dedup_size],
                fill_value=full_size,
                device=inverse_indices.device,
                dtype=torch.long,
            ),
            index=inverse_indices,
            src=torch.arange(
                full_size, device=inverse_indices.device, dtype=torch.int64
            ),
            dim=0,
            reduce="amin",
        )
        v_tex = v_tex[indices]
        t_tex_idx = inverse_indices.reshape(-1, 3)

        v_pos = v_pos_f3[indices]
        t_pos_idx = inverse_indices[t_pos_idx_f3]
        """

        return self.v_pos, self.t_pos_idx, v_tex, t_tex_idx

    def unwrap_uv(
        self,
        method: str,
        xatlas_chart_options: dict = {},
        xatlas_pack_options: dict = {},
        smartuv_options: dict = {},
    ):
        if method == "xatlas":
            with time_recorder_enabled():
                tr.start("UV unwrapping xatlas")
                self._v_tex, self._t_tex_idx = self._unwrap_uv_xatlas(
                    xatlas_chart_options, xatlas_pack_options
                )
                tr.end("UV unwrapping xatlas")
        elif method == "open3d":
            with time_recorder_enabled():
                tr.start("UV unwrapping o3d")
                self._v_tex, self._t_tex_idx = self._unwrap_uv_open3d()
                tr.end("UV unwrapping o3d")
        elif method == "smartuv":
            with time_recorder_enabled():
                tr.start("UV unwrapping smartuv")
                (
                    self.v_pos,
                    self.t_pos_idx,
                    self._v_tex,
                    self._t_tex_idx,
                ) = self._unwrap_uv_smartuv(smartuv_options)
                tr.end("UV unwrapping smartuv")
        else:
            raise NotImplementedError

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb
    
    def set_uv(self, v_tex, t_tex_idx):
        self._v_tex = v_tex
        self._t_tex_idx = t_tex_idx
    
    @property
    def has_uv(self):
        return self._v_tex is not None and self._t_tex_idx is not None

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

    def normal_consistency(self) -> Float[Tensor, ""]:
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(
            dim=1
        )
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self) -> Float[Tensor, ""]:
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss
