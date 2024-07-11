
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
from .flexicubes_geometry import FlexiCubesGeometry

class Renderer(nn.Module):
    def __init__(self, tet_grid_size, camera_angle_num, scale, geo_type):
        super().__init__()

        self.tet_grid_size = tet_grid_size
        self.camera_angle_num = camera_angle_num
        self.scale = scale
        self.geo_type = geo_type
        self.glctx = dr.RasterizeCudaContext()

        if self.geo_type == "flex":
            self.flexicubes = FlexiCubesGeometry(grid_res = self.tet_grid_size)   

    def forward(self, data, sdf, deform, verts, tets, training=False, weight = None):

        results = {}

        deform = torch.tanh(deform) / self.tet_grid_size * self.scale / 0.95
        if self.geo_type == "flex":
            deform = deform *0.5

            v_deformed = verts + deform

            verts_list = []
            faces_list = []
            reg_list = []
            n_shape = verts.shape[0]
            for i in range(n_shape): 
                verts_i, faces_i, reg_i = self.flexicubes.get_mesh(v_deformed[i], sdf[i].squeeze(dim=-1),
                with_uv=False, indices=tets, weight_n=weight[i], is_training=training)

                verts_list.append(verts_i)
                faces_list.append(faces_i)
                reg_list.append(reg_i)       
            verts = verts_list
            faces = faces_list

            flexicubes_surface_reg = torch.cat(reg_list).mean()
            flexicubes_weight_reg = (weight ** 2).mean()
            results["flex_surf_loss"] = flexicubes_surface_reg
            results["flex_weight_loss"] = flexicubes_weight_reg

        return results, verts, faces