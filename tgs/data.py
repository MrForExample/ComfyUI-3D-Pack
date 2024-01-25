import json
import math
from dataclasses import dataclass, field

import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .utils.config import parse_structured
from .utils.ops import get_intrinsic_from_fov, get_ray_directions, get_rays
from .utils.typing import *

@dataclass
class CustomImageDataModuleConfig:
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )

    relative_pose: bool = False
    cond_height: int = 512
    cond_width: int = 512
    cond_camera_distance: float = 1.6
    cond_fovy_deg: float = 40.0
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 0.0
    eval_camera_distance: float = 1.6
    eval_fovy_deg: float = 40.0
    n_test_views: int = 120
    num_views_output: int = 120
    only_3dgs: bool = False

class CustomImageOrbitDataset(Dataset):
    def __init__(self, reference_image, reference_mask, cfg: Any) -> None:
        super().__init__()
        self.cfg: CustomImageDataModuleConfig = parse_structured(CustomImageDataModuleConfig, cfg)

        self.n_views = self.cfg.n_test_views
        assert self.n_views % self.cfg.num_views_output == 0

        self.all_images = reference_image
        self.all_masks = reference_mask

        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, self.n_views + 1)[
            : self.n_views
        ]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            focal=1.0,
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        # must use normalize=True to normalize directions here
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        intrinsic: Float[Tensor, "B 3 3"] = get_intrinsic_from_fov(
            self.cfg.eval_fovy_deg * math.pi / 180,
            H=self.cfg.eval_height,
            W=self.cfg.eval_width,
            bs=self.n_views,
        )
        intrinsic_normed: Float[Tensor, "B 3 3"] = intrinsic.clone()
        intrinsic_normed[..., 0, 2] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 2] /= self.cfg.eval_height
        intrinsic_normed[..., 0, 0] /= self.cfg.eval_width
        intrinsic_normed[..., 1, 1] /= self.cfg.eval_height

        self.rays_o, self.rays_d = rays_o, rays_d
        self.intrinsic = intrinsic
        self.intrinsic_normed = intrinsic_normed
        self.c2w = c2w
        self.camera_positions = camera_positions

        self.background_color = torch.as_tensor(self.cfg.background_color)

        # condition
        self.intrinsic_cond = get_intrinsic_from_fov(
            np.deg2rad(self.cfg.cond_fovy_deg),
            H=self.cfg.cond_height,
            W=self.cfg.cond_width,
        )
        self.intrinsic_normed_cond = self.intrinsic_cond.clone()
        self.intrinsic_normed_cond[..., 0, 2] /= self.cfg.cond_width
        self.intrinsic_normed_cond[..., 1, 2] /= self.cfg.cond_height
        self.intrinsic_normed_cond[..., 0, 0] /= self.cfg.cond_width
        self.intrinsic_normed_cond[..., 1, 1] /= self.cfg.cond_height


        if self.cfg.relative_pose:
            self.c2w_cond = torch.as_tensor(
                [
                    [0, 0, 1, self.cfg.cond_camera_distance],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ).float()
        else:
            cond_elevation = self.cfg.cond_elevation_deg * math.pi / 180
            cond_azimuth = self.cfg.cond_azimuth_deg * math.pi / 180
            cond_camera_position: Float[Tensor, "3"] = torch.as_tensor(
                    [
                        self.cfg.cond_camera_distance * np.cos(cond_elevation) * np.cos(cond_azimuth),
                        self.cfg.cond_camera_distance * np.cos(cond_elevation) * np.sin(cond_azimuth),
                        self.cfg.cond_camera_distance * np.sin(cond_elevation),
                    ], dtype=torch.float32
            )

            cond_center: Float[Tensor, "3"] = torch.zeros_like(cond_camera_position)
            cond_up: Float[Tensor, "3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)
            cond_lookat: Float[Tensor, "3"] = F.normalize(cond_center - cond_camera_position, dim=-1)
            cond_right: Float[Tensor, "3"] = F.normalize(torch.cross(cond_lookat, cond_up), dim=-1)
            cond_up = F.normalize(torch.cross(cond_right, cond_lookat), dim=-1)
            cond_c2w3x4: Float[Tensor, "3 4"] = torch.cat(
                [torch.stack([cond_right, cond_up, -cond_lookat], dim=-1), cond_camera_position[:, None]],
                dim=-1,
            )
            cond_c2w: Float[Tensor, "4 4"] = torch.cat(
                [cond_c2w3x4, torch.zeros_like(cond_c2w3x4[:1])], dim=0
            )
            cond_c2w[3, 3] = 1.0
            self.c2w_cond = cond_c2w

    def __len__(self):
        if self.cfg.only_3dgs:
            return len(self.all_images)
        else:
            return len(self.all_images) * self.n_views // self.cfg.num_views_output

    def __getitem__(self, index):
        if self.cfg.only_3dgs:
            scene_index = index
            view_index = [0]
        else:
            scene_index = index * self.cfg.num_views_output // self.n_views
            view_start = index % (self.n_views // self.cfg.num_views_output)
            view_index = list(range(self.n_views))[view_start * self.cfg.num_views_output : 
                                                (view_start + 1) * self.cfg.num_views_output]

        mask_cond: Float[Tensor, "Hc Wc 1"] = self.resize_image(self.all_masks[scene_index].unsqueeze(2))
        rgb_cond: Float[Tensor, "Hc Wc 3"] = self.resize_image(self.all_images[scene_index]) * mask_cond + self.background_color[None, None, :] * (1 - mask_cond)        

        out = {
            "rgb_cond": rgb_cond,
            "c2w_cond": self.c2w_cond.unsqueeze(0),
            "mask_cond": mask_cond,
            "intrinsic_cond": self.intrinsic_cond.unsqueeze(0),
            "intrinsic_normed_cond": self.intrinsic_normed_cond.unsqueeze(0),
            "view_index": torch.as_tensor(view_index),
            "rays_o": self.rays_o[view_index],
            "rays_d": self.rays_d[view_index],
            "intrinsic": self.intrinsic[view_index],
            "intrinsic_normed": self.intrinsic_normed[view_index],
            "c2w": self.c2w[view_index],
            "camera_positions": self.camera_positions[view_index],
        }
        out["c2w"][..., :3, 1:3] *= -1
        out["c2w_cond"][..., :3, 1:3] *= -1
        out["index"] = torch.as_tensor(scene_index)
        out["background_color"] = self.background_color
        instance_id = str(scene_index)
        out["instance_id"] = instance_id
        return out
    
    def resize_image(self, img):
        img_new = img.permute(2, 0, 1).unsqueeze(0)
        img_new = F.interpolate(img_new, (self.cfg.cond_width, self.cfg.cond_height), mode="bilinear", align_corners=False).contiguous()
        return img_new.permute(0, 2, 3, 1)

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch