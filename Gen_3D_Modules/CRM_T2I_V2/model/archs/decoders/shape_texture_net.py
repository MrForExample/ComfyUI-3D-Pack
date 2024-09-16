import torch
import torch.nn as nn
import torch.nn.functional as F


class TetTexNet(nn.Module):
    def __init__(self, plane_reso=64, padding=0.1, fea_concat=True):
        super().__init__()
        # self.c_dim = c_dim
        self.plane_reso = plane_reso
        self.padding = padding
        self.fea_concat = fea_concat

    def forward(self, rolled_out_feature, query):
        # rolled_out_feature: rolled-out triplane feature
        # query: queried xyz coordinates (should be scaled consistently to ptr cloud)

        plane_reso = self.plane_reso

        triplane_feature = dict()
        triplane_feature['xy'] = rolled_out_feature[:, :, :, 0: plane_reso]
        triplane_feature['yz'] = rolled_out_feature[:, :, :, plane_reso: 2 * plane_reso]
        triplane_feature['zx'] = rolled_out_feature[:, :, :, 2 * plane_reso:]

        query_feature_xy = self.sample_plane_feature(query, triplane_feature['xy'], 'xy')
        query_feature_yz = self.sample_plane_feature(query, triplane_feature['yz'], 'yz')
        query_feature_zx = self.sample_plane_feature(query, triplane_feature['zx'], 'zx')

        if self.fea_concat:
            query_feature = torch.cat((query_feature_xy, query_feature_yz, query_feature_zx), dim=1)
        else:
            query_feature = query_feature_xy + query_feature_yz + query_feature_zx

        output = query_feature.permute(0, 2, 1)

        return output

    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        # CYF note:
        # for pretraining, query are uniformly sampled positions w.i. [-scale, scale]
        # for training, query are essentially tetrahedra grid vertices, which are
        # also within [-scale, scale] in the current version!
        # xy range [-scale, scale]
        if plane == 'xy':
            xy = query[:, :, [0, 1]]
        elif plane == 'yz':
            xy = query[:, :, [1, 2]]
        elif plane == 'zx':
            xy = query[:, :, [2, 0]]
        else:
            raise ValueError("Error! Invalid plane type!")

        xy = xy[:, :, None].float()
        # not seem necessary to rescale the grid, because from
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html,
        # it specifies sampling locations normalized by plane_feature's spatial dimension,
        # which is within [-scale, scale] as specified by encoder's calling of coordinate2index()
        vgrid = 1.0 * xy
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)

        return sampled_feat
