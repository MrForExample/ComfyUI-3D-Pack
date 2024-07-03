from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from ..utils.base import BaseModule
from ..utils.ops import get_activation
from ..utils.typing import *

class PointOutLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 3
    cfg: Config
    def configure(self) -> None:
        super().configure()
        self.point_layer = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.point_layer.weight, 0)
        nn.init.constant_(self.point_layer.bias, 0)

    def forward(self, x):
        return self.point_layer(x)

class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 80

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
        )

    def forward(
        self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]
    ) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
        triplanes_up = rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )
        return triplanes_up


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

class GSProjection(nn.Module):
    def __init__(self, 
                in_channels: int = 80,
                sh_degree: int = 3,
                init_scaling: float = -5.0,
                init_density: float = 0.1) -> None:
        super().__init__()

        self.out_keys = GS_KEYS + ["shs"]
        self.out_channels = GS_CHANNELS + [(sh_degree + 1) ** 2 * 3]

        self.out_layers = nn.ModuleList()
        for key, ch in zip(self.out_keys, self.out_channels):
            layer = nn.Linear(in_channels, ch)
            # initialize
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                inverse_sigmoid = lambda x: np.log(x / (1 - x))
                nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            
            self.out_layers.append(layer)
    
    def forward(self, x):
        ret = []
        for k, layer in zip(self.out_keys, self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = torch.exp(v)
                # v = v.detach() # FIXME: for DEBUG
            elif k == "opacity":
                v = torch.sigmoid(v)
            # elif k == "shs":
            #     v = torch.reshape(v, (v.shape[0], -1, 3))
            ret.append(v)
        ret = torch.cat(ret, dim=-1)
        return ret

def get_encoding(n_input_dims: int, config) -> nn.Module:
    raise NotImplementedError


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    raise NotImplementedError


# Resnet Blocks for pointnet
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx