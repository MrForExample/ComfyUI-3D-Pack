from dataclasses import dataclass

import torch


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor
