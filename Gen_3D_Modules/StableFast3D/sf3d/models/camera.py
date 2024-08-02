from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from StableFast3D.sf3d.models.utils import BaseModule


class LinearCameraEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 25
        out_channels: int = 768
        conditions: List[str] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def forward(self, **kwargs):
        cond_tensors = []
        for cond_name in self.cfg.conditions:
            assert cond_name in kwargs
            cond = kwargs[cond_name]
            # cond in shape (B, Nv, ...)
            cond_tensors.append(cond.view(*cond.shape[:2], -1))
        cond_tensor = torch.cat(cond_tensors, dim=-1)
        assert cond_tensor.shape[-1] == self.cfg.in_channels
        embedding = self.linear(cond_tensor)
        return embedding
