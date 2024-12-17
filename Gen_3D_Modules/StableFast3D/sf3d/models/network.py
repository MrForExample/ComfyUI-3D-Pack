from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.amp import custom_bwd, custom_fwd

from StableFast3D.sf3d.models.utils import BaseModule, normalize


class PixelShuffleUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 40
        scale_factor: int = 4

        conv_layers: int = 4
        conv_kernel_size: int = 3

    cfg: Config

    def configure(self) -> None:
        layers = []
        output_channels = self.cfg.out_channels * self.cfg.scale_factor**2

        in_channels = self.cfg.in_channels
        for i in range(self.cfg.conv_layers):
            cur_out_channels = (
                in_channels if i != self.cfg.conv_layers - 1 else output_channels
            )
            layers.append(
                nn.Conv2d(
                    in_channels,
                    cur_out_channels,
                    self.cfg.conv_kernel_size,
                    padding=(self.cfg.conv_kernel_size - 1) // 2,
                )
            )
            if i != self.cfg.conv_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.PixelShuffle(self.cfg.scale_factor))

        self.upsample = nn.Sequential(*layers)

    def forward(
        self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]
    ) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
        return rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none" or name == "linear" or name == "identity":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "negative":
        return lambda x: -x
    elif name == "normalize_channel_last":
        return lambda x: normalize(x)
    elif name == "normalize_channel_first":
        return lambda x: normalize(x, dim=1)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None
    out_bias: float = 0.0


class MaterialMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 120
        n_neurons: int = 64
        activation: str = "silu"
        heads: List[HeadSpec] = field(default_factory=lambda: [])

    cfg: Config

    def configure(self) -> None:
        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [
                    nn.Linear(
                        self.cfg.in_channels if i == 0 else self.cfg.n_neurons,
                        self.cfg.n_neurons,
                    ),
                    self.make_activation(self.cfg.activation),
                ]
            head_layers += [
                nn.Linear(
                    self.cfg.n_neurons,
                    head.out_channels,
                ),
            ]
            heads[head.name] = nn.Sequential(*head_layers)
        self.heads = nn.ModuleDict(heads)

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def keys(self):
        return self.heads.keys()

    def forward(
        self, x, include: Optional[List] = None, exclude: Optional[List] = None
    ):
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude.")
        if include is not None:
            heads = [h for h in self.cfg.heads if h.name in include]
        elif exclude is not None:
            heads = [h for h in self.cfg.heads if h.name not in exclude]
        else:
            heads = self.cfg.heads

        out = {
            head.name: get_activation(head.output_activation)(
                self.heads[head.name](x) + head.out_bias
            )
            for head in heads
        }

        return out
