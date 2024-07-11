from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...utils.base import BaseModule
from ...utils.typing import *


class TriplaneLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int = 32
        num_channels: int = 1024

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, batch_size: int) -> Float[Tensor, "B Ct Nt"]:
        return rearrange(
            repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(
        self, tokens: Float[Tensor, "B Ct Nt"]
    ) -> Float[Tensor, "B 3 Ct Hp Wp"]:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.cfg.plane_size**2 * 3
        assert Ct == self.cfg.num_channels
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.cfg.plane_size,
            Wp=self.cfg.plane_size,
        )
