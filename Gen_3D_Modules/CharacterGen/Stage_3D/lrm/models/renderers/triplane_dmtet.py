import os
from dataclasses import dataclass
from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from CharacterGen.Stage_3D import lrm
from ..renderers.base import BaseRenderer
from ..isosurface import MarchingTetrahedraHelper
from ...utils.ops import (
    get_activation,
    scale_tensor,
    dot,
    chunk_batch
)
from ...utils.rasterize import NVDiffRasterizerContext
from ..mesh import Mesh
from ...utils.typing import *


class TriplaneDMTetRenderer(BaseRenderer):
    @dataclass
    class Config(BaseRenderer.Config):
        feature_reduction: str = "concat"
        sdf_activation: Optional[str] = None
        sdf_bias: Union[str, float] = 0.0
        sdf_bias_params: Any = None
        inside_out: bool = False

        isosurface_resolution: int = 256
        tet_dir: str = "tets/"
        enable_isosurface_grid_deformation: bool = False
        eval_chunk_size: int = 262144
        context_type: str = "gl"

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        super().configure(*args, **kwargs)

        assert self.cfg.feature_reduction in ["concat", "mean"]

        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            os.path.join(self.cfg.tet_dir, f"{self.cfg.isosurface_resolution}_tets.npz"),
        ).to(self.device)

    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        # import ipdb
        # ipdb.set_trace()
        assert triplanes.ndim == 5 and positions.ndim == 3

        # assume positions in [-1, 1]
        # normalized to (-1, 1) for grid sample
        positions = scale_tensor(
            positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
        )

        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError

        net_out: Dict[str, Float[Tensor, "B N ..."]] = self.decoder(out)
        assert "sdf" in net_out
        net_out["sdf"] = get_activation(self.cfg.sdf_activation)(
            self.get_shifted_sdf(positions, net_out["sdf"])
        )

        if not batched:
            net_out = {k: v.squeeze(0) for k, v in net_out.items()}

        return net_out

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def forward_single(
        self,
        triplane: Float[Tensor, "3 Cp Hp Wp"],
        mvp_mtx: Float[Tensor, "Nv 4 4"],
        camera_positions: Float[Tensor, "Nv 3"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "3"]],
        extra_sdf_query: Any = None,
    ) -> Dict[str, Tensor]:
        Nv = mvp_mtx.shape[0]

        out = {}

        query_vertices = []
        query_sizes = []

        grid_vertices = scale_tensor(
            self.isosurface_helper.grid_vertices,
            self.isosurface_helper.points_range,
            self.bbox,
        )

        query_vertices.append(grid_vertices)
        query_sizes.append(len(grid_vertices))

        if extra_sdf_query is not None:
            query_vertices.append(extra_sdf_query)
            query_sizes.append(len(extra_sdf_query))

        query_vertices = torch.cat(query_vertices, dim=0)
        triplane_out = self.query_triplane(query_vertices, triplane)

        all_sdf = triplane_out["sdf"]
        if extra_sdf_query is not None:
            sdf, sdf_ex_query = torch.split(all_sdf, query_sizes)
        else:
            sdf, sdf_ex_query = all_sdf, None

        out.update({"sdf_ex_query": sdf_ex_query})

        if self.cfg.enable_isosurface_grid_deformation:
            all_deformation = triplane_out["deformation"]
            if extra_sdf_query is not None:
                deformation, _ = torch.split(all_deformation, query_sizes)
            else:
                deformation, _ = all_deformation, None
        else:
            deformation = None

        # Fix some sdf if we observe empty shape (full positive or full negative)
        pos_shape = torch.sum((sdf.squeeze(dim=-1) > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf.squeeze(dim=-1) < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            lrm.warn("Empty mesh! Fixing by adding fake faces.")
            sdf = torch.nan_to_num(sdf, nan=0.0, posinf=1.0, neginf=-1.0)
            update_sdf = torch.zeros_like(sdf)
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[self.isosurface_helper.center_indices] += (
                -1.0 - max_sdf
            )  # greater than zero
            update_sdf[self.isosurface_helper.boundary_indices] += (
                1.0 - min_sdf
            )  # smaller than zero
            new_sdf = sdf.clone().detach()
            if zero_surface:
                new_sdf += update_sdf
            update_mask = (update_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = sdf.abs().mean()
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)
            lrm.debug(
                "max sdf: {}, min sdf: {}".format(sdf.max().item(), sdf.min().item())
            )
            out.update({"sdf_reg": sdf_reg_loss})

        # Here we remove the gradient for the bad sdf (full positive or full negative)
        if zero_surface:
            sdf = sdf.detach()

        mesh: Mesh = self.isosurface_helper(sdf, deformation=deformation)

        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        # import ipdb
        # ipdb.set_trace()
        v_pos_clip: Float[Tensor, "Nv V 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out.update({"opacity": mask_aa, "mesh": mesh})

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)

        # FIXME: this depth corresponds to the one provided in the dataset, but assumes camera looking at scene center
        gb_depth = dot(
            gb_pos - camera_positions[:, None, None, :],
            F.normalize(-camera_positions[:, None, None, :], dim=-1),
        )

        gb_depth = torch.lerp(torch.zeros_like(gb_depth), gb_depth, mask.float())
        out.update({"depth": gb_depth})

        gb_viewdirs = F.normalize(gb_pos - camera_positions[:, None, None, :], dim=-1)
        gb_rgb_fg = torch.zeros(
            (Nv, height, width, 3), device=self.device, dtype=torch.float32
        )
        gb_rgb_bg = self.background(dirs=gb_viewdirs, color_spec=background_color)

        selector = mask[..., 0]
        if selector.sum() > 0:
            positions = gb_pos[selector]
            geo_out = self.query_triplane(positions, triplane)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                **extra_geo_info,
                **geo_out,
            )

            gb_rgb_fg[selector] = rgb_fg.to(
                gb_rgb_fg.dtype
            )  # TODO: don't know if this is correct

        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

        out.update(
            {"comp_rgb": gb_rgb_aa, "comp_rgb_fg": gb_rgb_fg, "comp_rgb_bg": gb_rgb_bg}
        )

        return out

    def forward(
        self,
        triplanes: Float[Tensor, "B 3 Cp Hp Wp"],
        mvp_mtx: Float[Tensor, "B Nv 4 4"],
        camera_positions: Float[Tensor, "B Nv 3"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        extra_sdf_query: Optional[List[Float[Tensor, "N 3"]]] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        batch_size = triplanes.shape[0]
        out_list = []
        for b in range(batch_size):
            out_list.append(
                self.forward_single(
                    triplanes[b],
                    mvp_mtx[b],
                    camera_positions[b],
                    height,
                    width,
                    background_color=background_color[b]
                    if background_color is not None
                    else None,
                    extra_sdf_query=extra_sdf_query[b]
                    if extra_sdf_query is not None
                    else None,
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)

        for k, v in out.items():
            # some properties cannot be batched
            if isinstance(v[0], torch.Tensor) and (
                all([vv.ndim == 0 for vv in v])
                or all([vv.shape[0] == v[0].shape[0] for vv in v])
            ):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v

        return out

    def isosurface(self, triplane: Float[Tensor, "3 Cp Hp Wp"]):
        grid_vertices = scale_tensor(
            self.isosurface_helper.grid_vertices,
            self.isosurface_helper.points_range,
            self.bbox,
        )
        triplane_out = chunk_batch(
            partial(self.query_triplane, triplanes=triplane), self.cfg.eval_chunk_size, grid_vertices,
        )

        sdf = triplane_out["sdf"]

        if self.cfg.inside_out:
            sdf = -sdf

        if self.cfg.enable_isosurface_grid_deformation:
            deformation = triplane_out["deformation"]
        else:
            deformation = None

        mesh: Mesh = self.isosurface_helper(sdf, deformation=deformation)

        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.bbox
        )

        return mesh

    def query(
        self, triplane: Float[Tensor, "3 Cp Hp Wp"], points: Float[Tensor, "*N 3"]
    ):
        input_shape = points.shape[:-1]
        triplane_out = chunk_batch(
            partial(self.query_triplane, triplanes=triplane), self.cfg.eval_chunk_size, points.view(-1, 3)
        )
        triplane_out = {
            k: v.view(*input_shape, v.shape[-1]) for k, v in triplane_out.items()
        }
        return triplane_out
