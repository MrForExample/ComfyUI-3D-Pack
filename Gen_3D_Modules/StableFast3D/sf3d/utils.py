from typing import Any

import numpy as np
import torch
from PIL import Image

import StableFast3D.sf3d.models.utils as sf3d_utils


def create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    intrinsic = sf3d_utils.get_intrinsic_from_fov(
        np.deg2rad(fov_deg),
        H=cond_height,
        W=cond_width,
    )
    intrinsic_normed_cond = intrinsic.clone()
    intrinsic_normed_cond[..., 0, 2] /= cond_width
    intrinsic_normed_cond[..., 1, 2] /= cond_height
    intrinsic_normed_cond[..., 0, 0] /= cond_width
    intrinsic_normed_cond[..., 1, 1] /= cond_height

    return intrinsic, intrinsic_normed_cond


def default_cond_c2w(distance: float):
    c2w_cond = torch.as_tensor(
        [
            [0, 0, 1, distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).float()
    return c2w_cond


def remove_background(
    image: Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image:
    import rembg
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: Image,
    ratio: float,
) -> Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image, mode="RGBA")
    return new_image
