import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.config import parse_structured
from ..utils.geometry import (
    get_plucker_embeds_from_cameras,
    get_plucker_embeds_from_cameras_ortho,
    get_position_map_from_depth,
    get_position_map_from_depth_ortho,
)
from ..utils.typing import *


def _parse_scene_list_single(scene_list_path: str, root_data_dir: str):
    all_scenes = []
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            for p in json.loads(f.read()):
                if "/" in p:
                    all_scenes.append(os.path.join(root_data_dir, p))
                else:
                    all_scenes.append(os.path.join(root_data_dir, p[:2], p))
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            for p in f.readlines():
                p = p.strip()
                if "/" in p:
                    all_scenes.append(os.path.join(root_data_dir, p))
                else:
                    all_scenes.append(os.path.join(root_data_dir, p[:2], p))
    else:
        raise NotImplementedError

    return all_scenes


def _parse_scene_list(
    scene_list_path: Union[str, List[str]], root_data_dir: Union[str, List[str]]
):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    if isinstance(root_data_dir, str):
        root_data_dir = [root_data_dir]
    for scene_list_path_, root_data_dir_ in zip(scene_list_path, root_data_dir):
        all_scenes += _parse_scene_list_single(scene_list_path_, root_data_dir_)
    return all_scenes


def _parse_reference_scene_list(reference_scenes: List[str], all_scenes: List[str]):
    all_ids = set(scene.split("/")[-1] for scene in all_scenes)
    ref_ids = set(scene.split("/")[-1] for scene in reference_scenes)
    common_ids = ref_ids.intersection(all_ids)
    all_scenes = [scene for scene in all_scenes if scene.split("/")[-1] in common_ids]
    all_ids = {scene.split("/")[-1]: idx for idx, scene in enumerate(all_scenes)}

    ref_scenes = [
        scene for scene in reference_scenes if scene.split("/")[-1] in all_ids
    ]
    sorted_ref_scenes = sorted(ref_scenes, key=lambda x: all_ids[x.split("/")[-1]])
    scene2ref = {
        scene: ref_scene for scene, ref_scene in zip(all_scenes, sorted_ref_scenes)
    }

    return all_scenes, scene2ref


@dataclass
class MultiviewDataModuleConfig:
    root_dir: Any = ""
    scene_list: Any = ""
    image_suffix: str = "webp"
    background_color: Union[str, float] = "gray"
    image_names: List[str] = field(default_factory=lambda: [])
    image_modality: str = "render"
    num_views: int = 1
    random_view_list: Optional[List[List[int]]] = None

    prompt_db_path: Optional[str] = None
    return_prompt: bool = False
    use_empty_prompt: bool = False
    prompt_prefix: Optional[Any] = None
    return_one_prompt: bool = True

    projection_type: str = "ORTHO"

    # source conditions
    source_image_modality: Any = "position"
    use_camera_space_normal: bool = False
    position_offset: float = 0.5
    position_scale: float = 1.0
    plucker_offset: float = 1.0
    plucker_scale: float = 2.0

    # reference image
    reference_root_dir: Optional[Any] = None
    reference_scene_list: Optional[Any] = None
    reference_image_modality: str = "render"
    reference_image_names: List[str] = field(default_factory=lambda: [])
    reference_augment_resolutions: Optional[List[int]] = None
    reference_mask_aug: bool = False

    repeat: int = 1  # for debugging purpose

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    height: int = 768
    width: int = 768

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16


class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiviewDataModuleConfig = cfg
        self.all_scenes = _parse_scene_list(self.cfg.scene_list, self.cfg.root_dir)

        if (
            self.cfg.reference_root_dir is not None
            and self.cfg.reference_scene_list is not None
        ):
            reference_scenes = _parse_scene_list(
                self.cfg.reference_scene_list, self.cfg.reference_root_dir
            )
            self.all_scenes, self.reference_scenes = _parse_reference_scene_list(
                reference_scenes, self.all_scenes
            )
        else:
            self.reference_scenes = None

        self.split = split
        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.train_indices[0] : self.cfg.train_indices[1]
            ]
            self.all_scenes = self.all_scenes * self.cfg.repeat
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.val_indices[0] : self.cfg.val_indices[1]
            ]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[
                self.cfg.test_indices[0] : self.cfg.test_indices[1]
            ]

        if self.cfg.prompt_db_path is not None:
            self.prompt_db = json.load(open(self.cfg.prompt_db_path))
        else:
            self.prompt_db = None

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(
        self,
        image: Union[str, Image.Image],
        height: int,
        width: int,
        background_color: torch.Tensor,
        rescale: bool = False,
        mask_aug: bool = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.resize((width, height))
        image = torch.from_numpy(np.array(image)).float() / 255.0

        if mask_aug:
            alpha = image[:, :, 3]  # Extract alpha channel
            h, w = alpha.shape
            y_indices, x_indices = torch.where(alpha > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                idx = torch.randint(len(y_indices), (1,)).item()
                y_center = y_indices[idx].item()
                x_center = x_indices[idx].item()
                mask_h = random.randint(h // 8, h // 4)
                mask_w = random.randint(w // 8, w // 4)

                y1 = max(0, y_center - mask_h // 2)
                y2 = min(h, y_center + mask_h // 2)
                x1 = max(0, x_center - mask_w // 2)
                x2 = min(w, x_center + mask_w // 2)

                alpha[y1:y2, x1:x2] = 0.0
                image[:, :, 3] = alpha

        image = image[:, :, :3] * image[:, :, 3:4] + background_color * (
            1 - image[:, :, 3:4]
        )
        if rescale:
            image = image * 2.0 - 1.0
        return image

    def load_normal_image(
        self,
        path,
        height,
        width,
        background_color,
        camera_space: bool = False,
        c2w: Optional[torch.FloatTensor] = None,
    ):
        image = Image.open(path).resize((width, height), resample=Image.NEAREST)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        alpha = image[:, :, 3:4]
        image = image[:, :, :3]
        if camera_space:
            w2c = torch.linalg.inv(c2w)[:3, :3]
            image = (
                F.normalize(((image * 2 - 1)[:, :, None, :] * w2c).sum(-1), dim=-1)
                * 0.5
                + 0.5
            )
        image = image * alpha + background_color * (1 - alpha)
        return image

    def load_depth(self, path, height, width):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth[..., 0:1]).float()
        mask = torch.ones_like(depth)
        mask[depth > 1000.0] = 0.0  # depth = 65535 is the invalid value
        depth[~(mask > 0.5)] = 0.0
        return depth, mask

    def retrieve_prompt(self, scene_dir):
        assert self.prompt_db is not None
        source_id = os.path.basename(scene_dir)
        return self.prompt_db.get(source_id, "")

    def __getitem__(self, index):
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))
        scene_dir = self.all_scenes[index]
        with open(os.path.join(scene_dir, "meta.json")) as f:
            meta = json.load(f)
        name2loc = {loc["index"]: loc for loc in meta["locations"]}

        # target multi-view images
        image_paths = [
            os.path.join(
                scene_dir, f"{self.cfg.image_modality}_{f}.{self.cfg.image_suffix}"
            )
            for f in self.cfg.image_names
        ]
        images = [
            self.load_image(
                p,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
            )
            for p in image_paths
        ]
        images = torch.stack(images, dim=0).permute(0, 3, 1, 2)

        # camera
        c2w = [
            torch.as_tensor(name2loc[name]["transform_matrix"])
            for name in self.cfg.image_names
        ]
        c2w = torch.stack(c2w, dim=0)

        if self.cfg.projection_type == "PERSP":
            camera_angle_x = (
                meta.get("camera_angle_x", None)
                or meta["locations"][0]["camera_angle_x"]
            )
            focal_length = 0.5 * self.cfg.width / np.tan(0.5 * camera_angle_x)
            intrinsics = (
                torch.as_tensor(
                    [
                        [focal_length, 0.0, 0.5 * self.cfg.width],
                        [0.0, focal_length, 0.5 * self.cfg.height],
                        [0.0, 0.0, 1.0],
                    ]
                )
                .unsqueeze(0)
                .float()
                .repeat(len(self.cfg.image_names), 1, 1)
            )
        elif self.cfg.projection_type == "ORTHO":
            ortho_scale = (
                meta.get("ortho_scale", None) or meta["locations"][0]["ortho_scale"]
            )

        # source conditions
        source_image_modality = self.cfg.source_image_modality
        if isinstance(source_image_modality, str):
            source_image_modality = [source_image_modality]
        source_images = []
        for modality in source_image_modality:
            if modality == "position":
                depth_masks = [
                    self.load_depth(
                        os.path.join(scene_dir, f"depth_{f}.exr"),
                        self.cfg.height,
                        self.cfg.width,
                    )
                    for f in self.cfg.image_names
                ]
                depths = torch.stack([d for d, _ in depth_masks])
                masks = torch.stack([m for _, m in depth_masks])
                c2w_ = c2w.clone()
                c2w_[:, :, 1:3] *= -1

                if self.cfg.projection_type == "PERSP":
                    position_maps = get_position_map_from_depth(
                        depths,
                        masks,
                        intrinsics,
                        c2w_,
                        image_wh=(self.cfg.width, self.cfg.height),
                    )
                elif self.cfg.projection_type == "ORTHO":
                    position_maps = get_position_map_from_depth_ortho(
                        depths,
                        masks,
                        c2w_,
                        ortho_scale,
                        image_wh=(self.cfg.width, self.cfg.height),
                    )
                position_maps = (
                    (position_maps + self.cfg.position_offset) / self.cfg.position_scale
                ).clamp(0.0, 1.0)
                source_images.append(position_maps)
            elif modality == "normal":
                normal_maps = [
                    self.load_normal_image(
                        os.path.join(
                            scene_dir, f"{modality}_{f}.{self.cfg.image_suffix}"
                        ),
                        height=self.cfg.height,
                        width=self.cfg.width,
                        background_color=background_color,
                        camera_space=self.cfg.use_camera_space_normal,
                        c2w=c,
                    )
                    for c, f in zip(c2w, self.cfg.image_names)
                ]
                source_images.append(torch.stack(normal_maps, dim=0))
            elif modality == "plucker":
                if self.cfg.projection_type == "ORTHO":
                    plucker_embed = get_plucker_embeds_from_cameras_ortho(
                        c2w, [ortho_scale] * len(c2w), self.cfg.width
                    )
                elif self.cfg.projection_type == "PERSP":
                    plucker_embed = get_plucker_embeds_from_cameras(
                        c2w, [camera_angle_x] * len(c2w), self.cfg.width
                    )
                else:
                    raise NotImplementedError
                plucker_embed = plucker_embed.permute(0, 2, 3, 1)
                plucker_embed = (
                    (plucker_embed + self.cfg.plucker_offset) / self.cfg.plucker_scale
                ).clamp(0.0, 1.0)
                source_images.append(plucker_embed)
            else:
                raise NotImplementedError
        source_images = torch.cat(source_images, dim=-1).permute(0, 3, 1, 2)
        rv = {"rgb": images, "c2w": c2w, "source_rgb": source_images}

        num_images = len(self.cfg.image_names)
        # prompt
        if self.cfg.return_prompt:
            if self.cfg.use_empty_prompt:
                prompt = ""
            else:
                prompt = self.retrieve_prompt(scene_dir)
            prompts = [prompt] * num_images

            if self.cfg.prompt_prefix is not None:
                prompt_prefix = self.cfg.prompt_prefix
                if isinstance(prompt_prefix, str):
                    prompt_prefix = [prompt_prefix] * num_images

                for i, prompt in enumerate(prompts):
                    prompts[i] = f"{prompt_prefix[i]} {prompt}"

            if self.cfg.return_one_prompt:
                rv.update({"prompts": prompts[0]})
            else:
                rv.update({"prompts": prompts})

        # reference image
        if self.reference_scenes is not None:
            reference_scene_dir = self.reference_scenes[scene_dir]
            reference_image_paths = [
                os.path.join(
                    reference_scene_dir,
                    f"{self.cfg.reference_image_modality}_{f}.{self.cfg.image_suffix}",
                )
                for f in self.cfg.reference_image_names
            ]
            reference_image_path = random.choice(reference_image_paths)

            if self.cfg.reference_augment_resolutions is None:
                reference_image = self.load_image(
                    reference_image_path,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    background_color=background_color,
                    mask_aug=self.cfg.reference_mask_aug,
                ).permute(2, 0, 1)
                rv.update({"reference_rgb": reference_image})
            else:
                random_resolution = random.choice(
                    self.cfg.reference_augment_resolutions
                )
                reference_image_ = Image.open(reference_image_path).resize(
                    (random_resolution, random_resolution)
                )
                reference_image = self.load_image(
                    reference_image_,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    background_color=background_color,
                    mask_aug=self.cfg.reference_mask_aug,
                ).permute(2, 0, 1)
                rv.update({"reference_rgb": reference_image})

        return rv

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])

        if self.cfg.random_view_list is not None:
            indices = random.choice(self.cfg.random_view_list)
        else:
            indices = list(range(self.cfg.num_views))
        num_views = len(indices)

        for k in batch.keys():
            if k in ["rgb", "source_rgb", "c2w"]:
                batch[k] = batch[k][:, indices]
                batch[k] = pack(batch[k])
        for k in ["prompts"]:
            if not self.cfg.return_one_prompt:
                batch[k] = [item for pair in zip(*batch[k]) for item in pair]

        batch.update(
            {
                "num_views": num_views,
                # For SDXL
                "original_size": (self.cfg.height, self.cfg.width),
                "target_size": (self.cfg.height, self.cfg.width),
                "crops_coords_top_left": (0, 0),
            }
        )
        return batch


class MultiviewDataModule(pl.LightningDataModule):
    cfg: MultiviewDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    import torchvision
    from omegaconf import OmegaConf

    config_file = "configs/view-guidance/mvadapter_t2mv_flux.yaml"
    data_cfg = OmegaConf.load(config_file)["data"]
    cfg: MultiviewDataModuleConfig = MultiviewDataModuleConfig(**data_cfg)
    data_module = MultiviewDataModule(cfg)
    data_module.setup()

    for batch in data_module.test_dataloader():
        # ref_rgb = batch["reference_rgb"]  # bchw
        rgb = batch["rgb"]
        source_rgb = batch["source_rgb"]

        print(batch["prompts"])
        print(rgb.shape, source_rgb.shape)

        torchvision.utils.save_image(rgb[:, :3], "debug_rgb.png")
        torchvision.utils.save_image(source_rgb[:, :3], "debug_source_rgb.png")

        exit(0)
