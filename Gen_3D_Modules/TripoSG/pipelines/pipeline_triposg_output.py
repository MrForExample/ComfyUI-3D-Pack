from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput


@dataclass
class TripoSGPipelineOutput(BaseOutput):
    r"""
    Output class for ShapeDiff pipelines.
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]
