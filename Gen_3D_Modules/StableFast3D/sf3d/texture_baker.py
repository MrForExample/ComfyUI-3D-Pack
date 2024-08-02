import os

import slangtorch
import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor


class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.baker = slangtorch.loadModule(
            os.path.join(os.path.dirname(__file__), "texture_baker.slang")
        )

    def rasterize(
        self,
        uv: Float[Tensor, "Nv 2"],
        face_indices: Float[Tensor, "Nf 3"],
        bake_resolution: int,
    ) -> Float[Tensor, "bake_resolution bake_resolution 4"]:
        if not face_indices.is_cuda or not uv.is_cuda:
            raise ValueError("All input tensors must be on cuda")

        face_indices = face_indices.to(torch.int32)
        uv = uv.to(torch.float32)

        rast_result = torch.empty(
            bake_resolution, bake_resolution, 4, device=uv.device, dtype=torch.float32
        )

        block_size = 16
        grid_size = bake_resolution // block_size
        self.baker.bake_uv(uv=uv, indices=face_indices, output=rast_result).launchRaw(
            blockSize=(block_size, block_size, 1), gridSize=(grid_size, grid_size, 1)
        )

        return rast_result

    def get_mask(
        self, rast: Float[Tensor, "bake_resolution bake_resolution 4"]
    ) -> Bool[Tensor, "bake_resolution bake_resolution"]:
        return rast[..., -1] >= 0

    def interpolate(
        self,
        attr: Float[Tensor, "Nv 3"],
        rast: Float[Tensor, "bake_resolution bake_resolution 4"],
        face_indices: Float[Tensor, "Nf 3"],
        uv: Float[Tensor, "Nv 2"],
    ) -> Float[Tensor, "bake_resolution bake_resolution 3"]:
        # Make sure all input tensors are on torch
        if not attr.is_cuda or not face_indices.is_cuda or not rast.is_cuda:
            raise ValueError("All input tensors must be on cuda")

        attr = attr.to(torch.float32)
        face_indices = face_indices.to(torch.int32)
        uv = uv.to(torch.float32)

        pos_bake = torch.zeros(
            rast.shape[0],
            rast.shape[1],
            3,
            device=attr.device,
            dtype=attr.dtype,
        )

        block_size = 16
        grid_size = rast.shape[0] // block_size
        self.baker.interpolate(
            attr=attr, indices=face_indices, rast=rast, output=pos_bake
        ).launchRaw(
            blockSize=(block_size, block_size, 1), gridSize=(grid_size, grid_size, 1)
        )

        return pos_bake

    def forward(
        self,
        attr: Float[Tensor, "Nv 3"],
        uv: Float[Tensor, "Nv 2"],
        face_indices: Float[Tensor, "Nf 3"],
        bake_resolution: int,
    ) -> Float[Tensor, "bake_resolution bake_resolution 3"]:
        rast = self.rasterize(uv, face_indices, bake_resolution)
        return self.interpolate(attr, rast, face_indices, uv)
