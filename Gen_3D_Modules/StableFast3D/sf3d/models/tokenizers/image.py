from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from StableFast3D.sf3d.models.tokenizers.dinov2 import Dinov2Model
from StableFast3D.sf3d.models.transformers.attention import Modulation
from StableFast3D.sf3d.models.utils import BaseModule


class DINOV2SingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dinov2-large"
        width: int = 512
        height: int = 512
        modulation_cond_dim: int = 768

    cfg: Config

    def configure(self) -> None:
        self.model = Dinov2Model.from_pretrained(self.cfg.pretrained_model_name_or_path)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.model.set_gradient_checkpointing(False)

        # add modulation
        modulations = []
        for layer in self.model.encoder.layer:
            norm1_modulation = Modulation(
                self.model.config.hidden_size,
                self.cfg.modulation_cond_dim,
                zero_init=True,
                single_layer=True,
            )
            norm2_modulation = Modulation(
                self.model.config.hidden_size,
                self.cfg.modulation_cond_dim,
                zero_init=True,
                single_layer=True,
            )
            layer.register_ada_norm_modulation(norm1_modulation, norm2_modulation)
            modulations += [norm1_modulation, norm2_modulation]
        self.modulations = nn.ModuleList(modulations)

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        images: Float[Tensor, "B *N C H W"],
        modulation_cond: Optional[Float[Tensor, "B *N Cc"]],
        **kwargs,
    ) -> Float[Tensor, "B *N Ct Nt"]:
        model = self.model

        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                assert modulation_cond.ndim == 2
                modulation_cond = modulation_cond.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=rearrange(modulation_cond, "B N Cc -> (B N) Cc")
            if modulation_cond is not None
            else None,
        )
        local_features = out.last_hidden_state
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
