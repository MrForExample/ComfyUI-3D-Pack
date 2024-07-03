# import decord
# decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import glob
import os
import json
import random
import cv2
import math
import numpy as np
import torch
from PIL import Image
from .normal_utils import trans_normal, img2normal, normal2img

"""
load normal and color images together
"""
class MVDiffusionDatasetV2(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_views: int,
        bg_color: Any,
        img_wh: Tuple[int, int],
        validation: bool = False,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        caption_path: Optional[str] = None,
        elevation_range_deg: Tuple[float,float] = (-90, 90),
        azimuth_range_deg: Tuple[float, float] = (0, 360),
    ):
        self.all_obj_paths = sorted(glob.glob(os.path.join(root_dir, "*/*")))
        if not validation:
            self.all_obj_paths = self.all_obj_paths[:-num_validation_samples]
        else:
            self.all_obj_paths = self.all_obj_paths[-num_validation_samples:]
        if num_samples is not None:
            self.all_obj_paths = self.all_obj_paths[:num_samples]
        self.all_obj_ids = [os.path.basename(path) for path in self.all_obj_paths]
        self.num_views = num_views
        self.bg_color = bg_color
        self.img_wh = img_wh
    
    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    def load_image(self, img_path, bg_color, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255. # [0, 1]
        assert img.shape[-1] == 4 # RGBA

        alpha = img[...,3:4]
        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img, alpha
    
    def load_normal(self, img_path, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type='np'):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).resize(self.img_wh))

        assert normal.shape[-1] == 3 # RGB

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)
        img = normal2img(normal)

        img = img.astype(np.float32) / 255. # [0, 1]

        img = img[...,:3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError
        
        return img

    def __len__(self):
        return len(self.all_obj_ids)

    def __getitem__(self, index):
        obj_path = self.all_obj_paths[index]
        obj_id = self.all_obj_ids[index]
        with open(os.path.join(obj_path, 'meta.json')) as f:
            meta = json.loads(f.read())

        num_views_all = len(meta['locations'])
        num_groups = num_views_all // self.num_views

        # random a set of 4 views
        # the data is arranged in ascending order of the azimuth angle
        group_ids = random.sample(range(num_groups), k=2)
        cond_group_id, tgt_group_id = group_ids
        cond_location = meta['locations'][cond_group_id * self.num_views + random.randint(0, self.num_views - 1)]
        tgt_locations = meta['locations'][tgt_group_id * self.num_views : tgt_group_id * self.num_views + self.num_views]
        # random an order
        start_id = random.randint(0, self.num_views - 1)
        tgt_locations = tgt_locations[start_id:] + tgt_locations[:start_id]
        
        cond_elevation = cond_location['elevation']
        cond_azimuth = cond_location['azimuth']
        cond_c2w = cond_location['transform_matrix']
        cond_w2c = np.linalg.inv(cond_c2w)
        tgt_elevations = [loc['elevation'] for loc in tgt_locations]
        tgt_azimuths = [loc['azimuth'] for loc in tgt_locations]
        tgt_c2ws = [loc['transform_matrix'] for loc in tgt_locations]
        tgt_w2cs = [np.linalg.inv(loc['transform_matrix']) for loc in tgt_locations]

        elevations = [ele - cond_elevation for ele in tgt_elevations]
        azimuths = [(azi - cond_azimuth) % (math.pi * 2) for azi in tgt_azimuths]
        elevations = torch.as_tensor(elevations).float()
        azimuths = torch.as_tensor(azimuths).float()
        elevations_cond = torch.as_tensor([cond_elevation] * self.num_views).float()

        bg_color = self.get_bg_color()
        img_tensors_in = [
            self.load_image(os.path.join(obj_path, cond_location['frames'][0]['name']), bg_color, return_type='pt')[0].permute(2, 0, 1)
        ] * self.num_views
        img_tensors_out = []
        normal_tensors_out = []
        for loc, tgt_w2c in zip(tgt_locations, tgt_w2cs):
            img_path = os.path.join(obj_path, loc['frames'][0]['name'])
            img_tensor, alpha = self.load_image(img_path, bg_color, return_type="pt")
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensors_out.append(img_tensor)

            normal_path = os.path.join(obj_path, loc['frames'][1]['name'])
            normal_tensor = self.load_normal(normal_path, bg_color, alpha, RT_w2c=tgt_w2c, RT_w2c_cond=cond_w2c, return_type="pt").permute(2, 0, 1)
            normal_tensors_out.append(normal_tensor)
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float() # (Nv, 3, H, W)
        normal_tensors_out = torch.stack(normal_tensors_out, dim=0).float() # (Nv, 3, H, W)

        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1) # (Nv, 3)

        return {
            'elevations_cond': elevations_cond,
            'elevations_cond_deg': torch.rad2deg(elevations_cond),
            'elevations': elevations,
            'azimuths': azimuths,
            'elevations_deg': torch.rad2deg(elevations),
            'azimuths_deg': torch.rad2deg(azimuths),
            'imgs_in': img_tensors_in,
            'imgs_out': img_tensors_out,
            'normals_out': normal_tensors_out,
            'camera_embeddings': camera_embeddings
        }

