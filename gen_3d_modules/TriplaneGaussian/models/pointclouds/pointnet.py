# modified from https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/pointnet.py
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

from TriplaneGaussian.utils.base import BaseModule
from TriplaneGaussian.models.networks import ResnetBlockFC
from TriplaneGaussian.utils.ops import scale_tensor

class LocalPoolPointnet(BaseModule):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        plane_resolution (int): defined resolution for plane feature
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    @dataclass
    class Config(BaseModule.Config):
        input_channels: int = 3
        c_dim: int = 128
        hidden_dim: int = 128
        scatter_type: str = "max"
        plane_size: int = 32
        n_blocks: int = 5
        radius: float = 1.

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.fc_pos = nn.Linear(self.cfg.input_channels, 2 * self.cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * self.cfg.hidden_dim, self.cfg.hidden_dim) for i in range(self.cfg.n_blocks)
        ])
        self.fc_c = nn.Linear(self.cfg.hidden_dim, self.cfg.c_dim)

        self.actvn = nn.ReLU()

        if self.cfg.scatter_type == 'max':
            self.scatter = scatter_max
        elif self.cfg.scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')


    def generate_plane_features(self, index, c):
        # acquire indices of features in plane
        # xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        # index = self.coordinate2index(x, self.cfg.plane_size)

        # scatter plane features from points
        fea_plane = c.new_zeros(index.shape[0], self.cfg.c_dim, self.cfg.plane_size ** 2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(index.shape[0], self.cfg.c_dim, self.cfg.plane_size, self.cfg.plane_size) # sparce matrix (B x 512 x reso x reso)

        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.shape[0], c.shape[2]
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.cfg.plane_size ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def coordinate2index(self, x):
        x = (x * self.cfg.plane_size).long()
        index = x[..., 0] + self.cfg.plane_size * x[..., 1]
        assert index.max() < self.cfg.plane_size ** 2
        return index[:, None, :]

    def forward(self, p):
        batch_size, T, D = p.shape

        # acquire the index for each point
        coord = {}
        index = {}

        position = torch.clamp(p[..., :3], -self.cfg.radius + 1e-6, self.cfg.radius - 1e-6)
        position_norm = scale_tensor(position, (-self.cfg.radius, self.cfg.radius), (0, 1))
        coord["xy"] = position_norm[..., [0, 1]]
        coord["xz"] = position_norm[..., [0, 2]]
        coord["yz"] = position_norm[..., [1, 2]]
        index["xy"] = self.coordinate2index(coord["xy"])
        index["xz"] = self.coordinate2index(coord["xz"])
        index["yz"] = self.coordinate2index(coord["yz"])
        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        features = torch.stack([
            self.generate_plane_features(index["xy"], c),
            self.generate_plane_features(index["xz"], c),
            self.generate_plane_features(index["yz"], c)
        ], dim=1)
        
        return features
