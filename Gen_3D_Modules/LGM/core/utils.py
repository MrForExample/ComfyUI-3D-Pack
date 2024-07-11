import numpy as np
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiui.op import safe_normalize

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate
    import roma
    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

class WarningFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            record.msg = f"Warn!: {record.msg}"
        return True

def create_handler(stream, levels, formatter):
    handler = logging.StreamHandler(stream)
    handler.setLevel(min(levels))
    handler.addFilter(lambda record: record.levelno in levels)
    handler.addFilter(WarningFilter())  # Apply the custom filter
    handler.setFormatter(formatter)
    
    return handler
def setup_logger(logger_name, level, stdout_levels, stderr_levels, formatter):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(level)
    stdout_handler = create_handler(sys.stdout, stdout_levels, formatter)
    stderr_handler = create_handler(sys.stderr, stderr_levels, formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)