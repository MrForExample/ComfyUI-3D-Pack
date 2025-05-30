from abc import ABC, abstractmethod

import torch
import transformers

from .utils import IMAGE_TYPE, image_to_tensor


class SegmentationModel(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, images: IMAGE_TYPE) -> torch.FloatTensor:
        pass


class RMBGModel(SegmentationModel):
    def __init__(self, pretrained_model_name_or_path: str, device: str):
        self.model = transformers.AutoModelForImageSegmentation.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        ).to(device)
        self.device = device

    def __call__(self, images: IMAGE_TYPE) -> torch.FloatTensor:
        images = image_to_tensor(images, device=self.device)
        batched = True
        if images.ndim == 3:
            images = images.unsqueeze(0)
            batched = False

        out = (
            self.model(images.permute(0, 3, 1, 2) - 0.5)[0][0]
            .clamp(0.0, 1.0)
            .permute(0, 2, 3, 1)
        )
        if not batched:
            out = out.squeeze(0)
        return out
