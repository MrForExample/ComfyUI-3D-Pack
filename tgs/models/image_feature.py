from dataclasses import dataclass
import torch
import torch.nn.functional as F
from einops import rearrange

from ..utils.base import BaseModule
from ..utils.ops import compute_distance_transform
from ..utils.typing import *

class ImageFeature(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        use_rgb: bool = True
        use_feature: bool = True
        use_mask: bool = True
        feature_dim: int = 128
        out_dim: int = 133
        backbone: str = "default"
        freeze_backbone_params: bool = True

    cfg: Config

    def forward(self, rgb, mask=None, feature=None):
        B, Nv, H, W = rgb.shape[:4]
        rgb = rearrange(rgb, "B Nv H W C -> (B Nv) C H W")
        if mask is not None:
            mask = rearrange(mask, "B Nv H W C -> (B Nv) C H W")

        assert feature is not None
        # reshape dino tokens to image-like size
        feature = rearrange(feature, "B (Nv Nt) C -> (B Nv) Nt C", Nv=Nv)
        feature = feature[:, 1:].reshape(B * Nv, H // 14, W // 14, -1).permute(0, 3, 1, 2).contiguous()
        feature = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)

        if mask is not None and mask.is_floating_point():
            mask = mask > 0.5
        
        image_features = []
        if self.cfg.use_rgb:
            image_features.append(rgb)
        if self.cfg.use_feature:
            image_features.append(feature)
        if self.cfg.use_mask:
            image_features += [mask, compute_distance_transform(mask)]

        # detach features, occur error when with grad
        image_features = torch.cat(image_features, dim=1)#.detach()
        return rearrange(image_features, "(B Nv) C H W -> B Nv C H W", B=B, Nv=Nv).squeeze(1)