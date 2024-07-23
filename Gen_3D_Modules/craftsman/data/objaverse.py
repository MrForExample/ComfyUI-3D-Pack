import math
import os
import json
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

from craftsman import register
from craftsman.utils.base import Updateable
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

@dataclass
class ObjaverseDataModuleConfig:
    root_dir: str = None
    data_type: str = "occupancy"         # occupancy or sdf
    n_samples: int = 4096                # number of points in input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 10000           # number of points in supervision
    
    load_image: bool = False             # whether to load images 
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision

    load_caption: bool = False           # whether to load captions
    caption_type: str = "text"           # text, clip_embeds
    tokenizer_pretrained_model_name_or_path: str = ""

    batch_size: int = 32
    num_workers: int = 0

class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split

        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))
        print(f"Loaded {len(self.uids)} {split} uids")

        if self.cfg.load_caption:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.tokenizer_pretrained_model_name_or_path)

        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.distance = 1.0
        self.camera_embedding = torch.as_tensor([
            [[1, 0, 0, 0],
            [0, 0, -1, -self.distance],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], # front to back

            [[0, 0, 1, self.distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], # right to left

            [[-1, 0, 0, 0],
            [0, 0, 1, self.distance],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], # back to front

            [[0, 0, -1, -self.distance],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]], # left to right
        ], dtype=torch.float32)
        if self.cfg.n_views != 1:
            assert self.cfg.n_views == self.camera_embedding.shape[0]

    def __len__(self):
        return len(self.uids)

    def _load_shape(self, index: int) -> Dict[str, Any]:
        if self.cfg.data_type == "occupancy":
            # for input point cloud
            pointcloud = np.load(f'{self.cfg.root_dir}/{self.uids[index]}/pointcloud.npz')
            surface = np.asarray(pointcloud['points']) * 2 # range from -1 to 1
            normal = np.asarray(pointcloud['normals'])
            surface = np.concatenate([surface, normal], axis=1)
        elif self.cfg.data_type == "sdf":
            data = np.load(f'{self.cfg.root_dir}/{self.uids[index]}.npz')
            # for input point cloud
            surface = data["surface"]
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")
        
        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(surface.shape[0], self.cfg.n_samples, replace=False)
        surface = surface[ind]
        # rescale data
        surface[:, :3] = surface[:, :3] * self.cfg.scale # target scale
        # add noise to input point cloud
        surface[:, :3] += (np.random.rand(surface.shape[0], 3) * 2 - 1) * self.cfg.noise_sigma
        ret = {
            "uid": self.uids[index].split('/')[-1],
            "surface": surface.astype(np.float32),
        }

        return ret

    def _load_shape_supervision(self, index: int) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.data_type == "occupancy":
            points = np.load(f'{self.cfg.root_dir}/{self.uids[index]}/points.npz')
            rand_points = np.asarray(points['points']) * 2 # range from -1.1 to 1.1
            occupancies = np.asarray(points['occupancies'])
            occupancies = np.unpackbits(occupancies)
        elif self.cfg.data_type == "sdf":
            data = np.load(f'{self.cfg.root_dir}/{self.uids[index]}.npz')
            rand_points = data['rand_points']
            sdfs = data['sdfs']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        ret["rand_points"] = rand_points.astype(np.float32)

        if self.cfg.data_type == "occupancy":
            assert self.cfg.supervision_type == "occupancy", "Only occupancy supervision is supported for occupancy data"
            occupancies = occupancies[ind]
            ret["occupancies"] = occupancies.astype(np.float32)
        elif self.cfg.data_type == "sdf":
            if self.cfg.supervision_type == "sdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32)
            elif self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(np.float32)
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret

    def _load_image(self, index: int) -> Dict[str, Any]:
        def _load_single_image(img_path):
            img = torch.from_numpy(
                np.asarray(
                    Image.fromarray(imageio.v2.imread(img_path))
                    .convert("RGBA")
                )
                / 255.0
            ).float()
            mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
            image: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + self.background_color[
                None, None, :
            ] * (1 - mask)
            return image
        
        ret = {}
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert self.cfg.n_views == 1, "Only single view is supported for single image"
            sel_idx = random.choice(self.cfg.idx)
            ret["sel_image_idx"] = sel_idx
            if self.cfg.image_type == "rgb":
                img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{sel_idx}.png"
            elif self.cfg.image_type == "normal":
                img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{sel_idx}_normal.png"
            ret["image"] = _load_single_image(img_path)
            ret["c2w"] = self.camera_embedding[sel_idx % 4]
        elif self.cfg.image_type == "mvrgb" or self.cfg.image_type == "mvnormal":
            sel_idx = random.choice(self.cfg.idx)
            ret["sel_image_idx"] = sel_idx
            mvimages = []
            for i in range(self.cfg.n_views):
                if self.cfg.image_type == "mvrgb":
                    img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{sel_idx+i}.png"
                elif self.cfg.image_type == "mvnormal":
                    img_path = f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f"/{sel_idx+i}_normal.png"
                mvimages.append(_load_single_image(img_path))
            ret["mvimages"] = torch.stack(mvimages)
            ret["c2ws"] = self.camera_embedding
        else:
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        
        return ret

    def _load_caption(self, index: int, drop_text_embed: bool = False) -> Dict[str, Any]:
        ret = {}
        if self.cfg.caption_type == "text":
            caption = eval(json.load(open(f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f'/annotation.json')))
            texts = [v for k, v in caption.items()]
            sel_idx = random.randint(0, len(texts) - 1)
            ret["sel_caption_idx"] = sel_idx
            ret['text_input_ids'] = self.tokenizer(
                texts[sel_idx] if not drop_text_embed else "",
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.detach()
        else:
            raise NotImplementedError(f"Caption type {self.cfg.caption_type} not implemented")
        
        return ret

    def get_data(self, index):
        # load shape
        ret = self._load_shape(index)

        # load supervision for shape
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index))

        # load image
        if self.cfg.load_image:
            ret.update(self._load_image(index))

            # load the rotation of the object and rotate the camera
            rots = np.load(f'{self.cfg.image_data_path}/' + "/".join(self.uids[index].split('/')[-2:]) + f'/rots.npy')[ret['sel_image_idx']].astype(np.float32)
            rots = torch.tensor(rots[:3, :3], dtype=torch.float32)
            if "c2ws" in ret.keys():
                ret["c2ws"][:, :3, :3] = torch.matmul(rots, ret["c2ws"][:, :3, :3])
                ret["c2ws"][:, :3, 3] = torch.matmul(rots, ret["c2ws"][:, :3, 3].unsqueeze(-1)).squeeze(-1)
            elif "c2w" in ret.keys():
                ret["c2w"][:3, :3] = torch.matmul(rots, ret["c2w"][:3, :3])
                ret["c2w"][:3, 3] = torch.matmul(rots, ret["c2w"][:3, 3].unsqueeze(-1)).squeeze(-1)

        # load caption
        if self.cfg.load_caption:
            ret.update(self._load_caption(index))

        return ret
        
    def __getitem__(self, index):
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))


    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch


@register("objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)