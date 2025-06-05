import io
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch import BoolTensor, FloatTensor

LIST_TYPE = Union[list, np.ndarray, torch.Tensor]
IMAGE_TYPE = Union[Image.Image, List[Image.Image], np.ndarray, torch.Tensor]
SINGLE_IMAGE_TYPE = Union[Image.Image, np.ndarray, torch.Tensor]


def tensor_to_image(
    data: Union[Image.Image, torch.Tensor, np.ndarray],
    batched: bool = False,
    format: str = "HWC",
) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype == np.float32 or data.dtype == np.float16:
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    assert data.dtype == np.uint8
    if format == "CHW":
        if batched and data.ndim == 4:
            data = data.transpose((0, 2, 3, 1))
        elif not batched and data.ndim == 3:
            data = data.transpose((1, 2, 0))

    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)


def image_to_tensor(image: IMAGE_TYPE, return_type="pt", device: Optional[str] = None):
    assert return_type in ["np", "pt"]
    batched = True
    if isinstance(image, Image.Image):
        batched = False
        image = [image]
    if isinstance(image, list):
        image = np.stack([np.array(img) for img in image], axis=0)
        image = image.astype(np.float32) / 255.0
    if isinstance(image, np.ndarray) and return_type == "pt":
        image = torch.tensor(image, device=device)
    if isinstance(image, torch.Tensor):
        image = image.to(dtype=torch.float32, device=device)

    if not batched:
        image = image[0]
    return image


def largest_factor_near_sqrt(n: int) -> int:
    """
    Finds the largest factor of n that is closest to the square root of n.

    Args:
        n (int): The integer for which to find the largest factor near its square root.

    Returns:
        int: The largest factor of n that is closest to the square root of n.
    """
    sqrt_n = int(math.sqrt(n))  # Get the integer part of the square root

    # First, check if the square root itself is a factor
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Otherwise, find the largest factor by iterating from sqrt_n downwards
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

    # If n is 1, return 1
    return 1


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_current_timestamp(fmt: str = "%Y%m%d%H%M%S") -> str:
    return datetime.now().strftime(fmt)


def get_clip_space_position(pos: torch.FloatTensor, mvp_mtx: torch.FloatTensor):
    pos_homo = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos)], dim=-1)
    return torch.matmul(pos_homo, mvp_mtx.permute(0, 2, 1))


def transform_points_homo(pos: torch.FloatTensor, mtx: torch.FloatTensor):
    batch_size = pos.shape[0]
    pos_shape = pos.shape[1:-1]
    pos = pos.reshape(batch_size, -1, 3)
    pos_homo = torch.cat([pos, torch.ones_like(pos[..., 0:1])], dim=-1)
    pos = (pos_homo.unsqueeze(2) * mtx.unsqueeze(1)).sum(-1)[..., :3]
    pos = pos.reshape(batch_size, *pos_shape, 3)
    return pos
