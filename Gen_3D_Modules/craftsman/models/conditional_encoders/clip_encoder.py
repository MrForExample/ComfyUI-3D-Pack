import random
import torch
from torch import nn
import numpy as np
from PIL import Image
from einops import rearrange
from dataclasses import dataclass
from torchvision.transforms import Normalize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.transforms import _interpolation_modes_from_int
from torchvision import transforms

from transformers import CLIPTokenizer, CLIPImageProcessor
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import craftsman
from craftsman.utils.typing import *
from .clip.modeling_clip import CLIPModel
from .clip.modeling_conditional_clip import ConditionalCLIPModel
from .base import BaseEmbedder, ImageType

@dataclass
class CLIPEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    embeds: torch.FloatTensor = None

@craftsman.register("clip-embedder")
class CLIPEmbedder(BaseEmbedder):

    @dataclass
    class Config(BaseEmbedder.Config):
        freeze_modulation: bool = False
        config_path: str = ''

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the CLIP model and processor
        if not self.cfg.encode_camera:
            self.model: CLIPModel = CLIPModel.from_pretrained(self.cfg.pretrained_model_name_or_path)
        else:
            if self.cfg.pretrained_model_name_or_path == '':
                assert self.cfg.config_path is not None, "The config path should be provided"
                conditional_clip_config = ConditionalCLIPModel.config_class.from_json_file(self.cfg.config_path)
                conditional_clip_config.vision_config.modulation_dim = self.cfg.camera_embeds_dim
                self.model: CLIPModel = ConditionalCLIPModel(conditional_clip_config)
            else:
                conditional_clip_config = ConditionalCLIPModel.config_class.from_pretrained(
                    self.cfg.pretrained_model_name_or_path,
                )
                conditional_clip_config.vision_config.modulation_dim = self.cfg.camera_embeds_dim
                self.model: CLIPModel = ConditionalCLIPModel.from_pretrained(
                    self.cfg.pretrained_model_name_or_path, 
                    vision_config=conditional_clip_config.vision_config
                )
                
        self.tokenizer = None
        self.image_preprocess = CLIPImageProcessor()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.logit_scale = self.model.logit_scale.exp()

        if self.cfg.zero_uncond_embeds:
            self.empty_text_embeds = torch.zeros((1, 77, 768)).detach()
            self.empty_image_embeds = torch.zeros((self.cfg.n_views, 257, 1024)).detach()
        else:
            try:
                self.empty_text_embeds = self.encode_text([""]).detach() # [1, 77, 768]
            except:
                self.empty_text_embeds = None
            if self.cfg.encode_camera:
                self.empty_image_embeds = self.encode_image(torch.zeros(self.cfg.n_views, 224, 224, 3), self.cameras[:self.cfg.n_views]).detach()
            else:
                self.empty_image_embeds = self.encode_image(torch.zeros(self.cfg.n_views, 224, 224, 3)).detach()

        # Freeze the model parameters
        self.model.eval()
        for k, p in self.model.named_parameters():
            ks = k.split('.')
            if 'mod_norm1' in ks or 'mod_norm2' in ks and not self.cfg.freeze_modulation:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    def encode_image(self, images: Iterable[Optional[ImageType]], cameras: Optional[torch.Tensor] = None, force_none_camera_embeds: bool = False, return_dict: bool = False, **kwargs) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)): # for training process
            assert images.min() >= 0.0 and images.max() <= 1.0, "The pixel values should be in the range of [0, 1]"
            do_rescale = False
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform(images.permute(0, 3, 1, 2))
        else: # for inference process
            do_rescale = True
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = self.cameras[:self.cfg.n_views].repeat(bs, 1, 1).to(self.model.device)
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess.preprocess(images, return_tensors='pt', do_rescale=do_rescale).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        packed = False
        if pixel_values.ndim == 4:
            packed = True
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)

        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.model.vision_model(
                pixel_values=rearrange(pixel_values.to(self.model.device), "B N C H W -> (B N) C H W"),
                condition=rearrange(camera_embeds, "B N C -> (B N) C")
            )
        else:
            vision_outputs = self.model.vision_model(
                pixel_values=rearrange(pixel_values.to(self.model.device), "B N C H W -> (B N) C H W"), 
            )

        if return_dict:
            pooler_output = vision_outputs[1]  # pooled_output
            image_features = self.model.visual_projection(pooler_output)

            return CLIPEmbedOutput(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=pooler_output,
                embeds=image_features
            )
        else:
            return vision_outputs.last_hidden_state

    @torch.no_grad()
    def encode_text(self, text_inputs: torch.Tensor, return_dict: bool = False) -> torch.FloatTensor:
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path)
            
        if isinstance(text_inputs, list):
            text_inputs = self.tokenizer(
                text_inputs, 
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        text_outputs = self.model.text_model(input_ids=text_inputs.to(self.model.device))

        pooler_output = text_outputs[1]  # pooled_output
        text_features = self.model.text_projection(pooler_output)

        if return_dict:
            return CLIPEmbedOutput(
                last_hidden_state=text_outputs.last_hidden_state,
                pooler_output=pooler_output,
                embeds=text_features
            )
        else:
            return text_outputs.last_hidden_state