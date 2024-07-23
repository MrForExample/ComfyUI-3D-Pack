import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from dataclasses import dataclass
from torchvision.transforms import Normalize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.transforms import _interpolation_modes_from_int

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *

ImageType = Union[np.ndarray, torch.Tensor, Image.Image]


class BaseEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None # the pretrained model name or path
        
        encode_camera: bool = False # whether to encode camera
        camera_embeds_type: str = "sincos" # the type of camera embeds
        camera_embeds_dim: Optional[int] = None # the dimension of camera embeds
        n_views: int = 1 # the number of views

        empty_embeds_ratio: float = 0.1 # the ratio of empty embeds
        zero_uncond_embeds: bool = True

        normalize_embeds: bool = False # whether to normalize the embeds

    cfg: Config

    def configure(self) -> None:
        super().configure()

        if self.cfg.encode_camera:
            self.distance = 1.0
            self.register_buffer(
                "cameras",
                    torch.as_tensor([
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
                ], dtype=torch.float32),
            )

    def encode_image(self, images: Iterable[Optional[ImageType]], camera_embeds: Optional[torch.Tensor] = None, **kwargs) -> torch.FloatTensor:
        pass

    def encode_text(self, texts: List[str], **kwargs) -> torch.FloatTensor:
        pass

    def encode_camera(self, c2ws: torch.Tensor):
        if self.cfg.camera_embeds_type == "sincos":
            assert c2ws.shape[-1] == 4 and c2ws.shape[-2] == 4, f"Invalid c2ws shape: {c2ws.shape}"
            c2ws = c2ws.view(-1, 16)
            return torch.cat([torch.sin(c2ws), torch.cos(c2ws)], dim=-1)
        else:
            raise NotImplementedError(f"Unknown camera_embeds_type: {self.cfg.camera_embeds_type}")

    def post_process_embeds(self, text_embeds, visual_embeds):
        bs = text_embeds.shape[0] if text_embeds is not None else visual_embeds.shape[0]

        if self.cfg.normalize_embeds:
            # post-process the text/visual embeds
            if text_embeds is not None:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            if visual_embeds is not None:
                visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)

        assert text_embeds is not None or visual_embeds is not None

        # return text_embeds, visual_embeds
        if text_embeds is not None and visual_embeds is not None:
            return torch.cat([text_embeds, visual_embeds], dim=1)
        elif text_embeds is not None:
            return text_embeds
        else:
            return visual_embeds

    def forward(self, batch):
        bs = batch["surface"].shape[0]

        text_embeds, visual_embeds = None, None
        
        if random.random() < self.cfg.empty_embeds_ratio:
            if "text_input_ids" in batch or "text_embeds" in batch:
                if self.empty_text_embeds is None:
                    if not self.cfg.zero_uncond_embeds:
                        self.empty_text_embeds = self.encode_text([""]).detach() # [1, 77, 768]
                text_embeds = self.empty_text_embeds.repeat(bs, 1, 1)
            if "image" in batch or "image_embeds" in batch:
                visual_embeds = self.empty_image_embeds.repeat(bs, 1, 1)
            elif "mvimages" in batch or "mvimage_embeds" in batch:
                visual_embeds = self.empty_image_embeds.unsqueeze(1).repeat(bs, 1, 1, 1)
        else:
            # for text inputs
            if "text_input_ids" in batch:
                text_embeds = self.encode_text(batch["text_input_ids"])

            # for visual inputs
            if "image" in batch:
                if self.cfg.encode_camera:
                    visual_embeds = self.encode_image(batch["image"], cameras=batch["c2w"])
                else:
                    visual_embeds = self.encode_image(batch["image"])
            elif "mvimages" in batch:
                n_views = batch["mvimages"].shape[1]
                if self.cfg.encode_camera:
                    visual_embeds = self.encode_image(
                        batch["mvimages"].view(-1, *batch["mvimages"].shape[-3:]), \
                        cameras=batch["c2ws"]).view(bs, n_views, *self.empty_image_embeds.shape[-2:])
                else:
                    visual_embeds =  self.encode_image(
                        batch["mvimages"].view(-1, *batch["mvimages"].shape[-3:])).view(bs, n_views, *self.empty_image_embeds.shape[-2:])

        return self.post_process_embeds(text_embeds, visual_embeds)
