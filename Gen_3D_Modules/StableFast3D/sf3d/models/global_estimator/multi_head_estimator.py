from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from StableFast3D.sf3d.models.network import get_activation
from StableFast3D.sf3d.models.utils import BaseModule


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None
    output_bias: float = 0.0
    add_to_decoder_features: bool = False
    shape: Optional[list[int]] = None


class MultiHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        triplane_features: int = 1024

        n_layers: int = 2
        hidden_features: int = 512
        activation: str = "relu"

        pool: str = "max"
        # Literal["mean", "max"] = "mean"  # noqa: F821

        heads: List[HeadSpec] = field(default_factory=lambda: [])

    cfg: Config

    def configure(self):
        layers = []
        cur_features = self.cfg.triplane_features * 3
        for _ in range(self.cfg.n_layers):
            layers.append(
                nn.Conv2d(
                    cur_features,
                    self.cfg.hidden_features,
                    kernel_size=3,
                    padding=0,
                    stride=2,
                )
            )
            layers.append(self.make_activation(self.cfg.activation))

            cur_features = self.cfg.hidden_features

        self.layers = nn.Sequential(*layers)

        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [
                    nn.Linear(
                        self.cfg.hidden_features,
                        self.cfg.hidden_features,
                    ),
                    self.make_activation(self.cfg.activation),
                ]
            head_layers += [
                nn.Linear(
                    self.cfg.hidden_features,
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

    def forward(
        self,
        triplane: Float[Tensor, "B 3 F Ht Wt"],
    ) -> dict[str, Any]:
        x = self.layers(
            triplane.reshape(
                triplane.shape[0], -1, triplane.shape[-2], triplane.shape[-1]
            )
        )

        if self.cfg.pool == "max":
            x = x.amax(dim=[-2, -1])
        elif self.cfg.pool == "mean":
            x = x.mean(dim=[-2, -1])
        else:
            raise NotImplementedError

        out = {
            ("decoder_" if head.add_to_decoder_features else "")
            + head.name: get_activation(head.output_activation)(
                self.heads[head.name](x) + head.output_bias
            )
            for head in self.cfg.heads
        }
        for head in self.cfg.heads:
            if head.shape:
                head_name = (
                    "decoder_" if head.add_to_decoder_features else ""
                ) + head.name
                out[head_name] = out[head_name].reshape(*head.shape)

        return out
