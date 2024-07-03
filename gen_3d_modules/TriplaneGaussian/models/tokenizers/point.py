from dataclasses import dataclass
import torch.nn as nn
from TriplaneGaussian.utils.base import BaseModule
from TriplaneGaussian.utils.typing import *
import torch

class PointLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_pcl: int = 2048
        num_channels: int = 512

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pcl_embeddings = nn.Embedding(
               self.cfg.num_pcl , self.cfg.num_channels
            )

    def forward(self, batch_size: int) -> Float[Tensor, "B Ct Nt"]:
        range_ = torch.arange(self.cfg.num_pcl, device=self.device)
        embeddings =  self.pcl_embeddings(range_).unsqueeze(0).repeat((batch_size,1,1))
        return torch.permute(embeddings, (0,2,1))

    def detokenize(
        self, tokens: Float[Tensor, "B Ct Nt"]
    ) -> Float[Tensor, "B 3 Ct Hp Wp"]:
        return torch.permute(tokens, (0,2,1))