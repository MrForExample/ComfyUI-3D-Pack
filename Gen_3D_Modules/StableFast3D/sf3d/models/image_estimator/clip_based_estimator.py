from dataclasses import dataclass, field
from typing import Any, List, Optional

import open_clip
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torchvision.transforms import Normalize

from StableFast3D.sf3d.models.network import get_activation
from StableFast3D.sf3d.models.utils import BaseModule


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None
    output_bias: float = 0.0
    add_to_decoder_features: bool = False
    shape: Optional[list[int]] = None


class ClipBasedHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model: str = "ViT-B-32"
        pretrain: str = "laion2b_s34b_b79k"

        distribution: str = "beta"

        # ["mean", "mode", "sample", "sample_mean"]
        distribution_eval: str = "mode"

        activation: str = "relu"
        hidden_features: int = 512
        heads: List[HeadSpec] = field(default_factory=lambda: [])

    cfg: Config

    def configure(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.cfg.model, pretrained=self.cfg.pretrain
        )
        self.model.eval()

        # Do not add the weights in self.model to the optimizer
        for param in self.model.parameters():
            param.requires_grad = False

        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []

            for i in range(head.n_hidden_layers):
                head_layers += [
                    nn.Linear(
                        self.cfg.hidden_features,
                        self.cfg.hidden_features,
                    ),
                    self.make_activation(self.cfg.activation),
                ]

            head_layers = [nn.Sequential(*head_layers)]
            head_layers += [
                nn.Sequential(
                    nn.Linear(
                        self.cfg.hidden_features,
                        self.cfg.hidden_features,
                    ),
                    self.make_activation(self.cfg.activation),
                    nn.Linear(self.cfg.hidden_features, 1),
                )
                for _ in range(2)
            ]
            heads[head.name] = nn.ModuleList(head_layers)
        self.heads = nn.ModuleDict(heads)

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(
        self,
        cond_image: Float[Tensor, "B 1 H W 3"],
        sample: bool = True,
    ) -> dict[str, Any]:
        # Run the model
        # Resize cond_image to 224
        cond_image = nn.functional.interpolate(
            cond_image.flatten(0, 1).permute(0, 3, 1, 2),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        cond_image = Normalize(
            mean=open_clip.constants.OPENAI_DATASET_MEAN,
            std=open_clip.constants.OPENAI_DATASET_STD,
        )(cond_image)
        image_features = self.model.encode_image(cond_image)

        # Run the heads
        outputs = {}

        for head_dict in self.cfg.heads:
            head_name = head_dict.name
            shared_head, d1_h, d2_h = self.heads[head_name]
            shared_features = shared_head(image_features)
            d1, d2 = [head(shared_features).squeeze(-1) for head in [d1_h, d2_h]]
            if self.cfg.distribution == "normal":
                mean = d1
                var = d2
                if mean.shape[-1] == 1:
                    outputs[head_name] = torch.distributions.Normal(
                        mean + head_dict.output_bias,
                        torch.nn.functional.softplus(var),
                    )
                else:
                    outputs[head_name] = torch.distributions.MultivariateNormal(
                        mean + head_dict.output_bias,
                        torch.nn.functional.softplus(var).diag_embed(),
                    )
            elif self.cfg.distribution == "beta":
                outputs[head_name] = torch.distributions.Beta(
                    torch.nn.functional.softplus(d1 + head_dict.output_bias),
                    torch.nn.functional.softplus(d2 + head_dict.output_bias),
                )
            else:
                raise NotImplementedError

        if sample:
            for head_dict in self.cfg.heads:
                head_name = head_dict.name
                dist = outputs[head_name]

                if self.cfg.distribution_eval == "mean":
                    out = dist.mean
                elif self.cfg.distribution_eval == "mode":
                    out = dist.mode
                elif self.cfg.distribution_eval == "sample_mean":
                    out = dist.sample([10]).mean(-1)
                else:
                    # use rsample if gradient is needed
                    out = dist.rsample() if self.training else dist.sample()

                outputs[head_name] = get_activation(head_dict.output_activation)(out)
                outputs[f"{head_name}_dist"] = dist

        for head in self.cfg.heads:
            if head.shape:
                if not sample:
                    raise ValueError(
                        "Cannot reshape non-sampled probabilisitic outputs"
                    )
                outputs[head.name] = outputs[head.name].reshape(*head.shape)

            if head.add_to_decoder_features:
                outputs[f"decoder_{head.name}"] = outputs[head.name]
                del outputs[head.name]

        return outputs
