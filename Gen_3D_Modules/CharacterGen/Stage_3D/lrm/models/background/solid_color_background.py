import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseBackground
from ...utils.typing import *


class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False
        random_aug: bool = False
        random_aug_prob: float = 0.5

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )

    def forward(
        self,
        dirs: Float[Tensor, "B H W Nc"],
        color_spec: Optional[Float[Tensor, "Nc"]] = None,
    ) -> Float[Tensor, "B H W Nc"]:
        color = torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs) * (
            color_spec if color_spec is not None else self.env_color
        )
        if (
            self.training
            and self.cfg.random_aug
            and random.random() < self.cfg.random_aug_prob
        ):
            # use random background color with probability random_aug_prob
            # color = color * 0 + (
            #     torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs) *
            #     torch.rand(self.cfg.n_output_dims).to(dirs)
            # )
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                torch.rand(dirs.shape[0], 1, 1, self.cfg.n_output_dims)
                .to(dirs)
                .expand(*dirs.shape[:-1], -1)
            )
        return color
