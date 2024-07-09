from dataclasses import dataclass

import torch

from ..networks import MultiHeadMLP
from ..background.base import BaseBackground
from ..materials.base import BaseMaterial
from ...utils.base import BaseModule
from ...utils.typing import *


class BaseRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0

    cfg: Config

    def configure(
        self,
        decoder: MultiHeadMLP,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure()

        self.set_decoder(decoder)
        self.set_material(material)
        self.set_background(background)

        # set up bounding box
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def decoder(self) -> MultiHeadMLP:
        return self.non_module("decoder")

    @property
    def material(self) -> BaseMaterial:
        return self.non_module("material")

    @property
    def background(self) -> BaseBackground:
        return self.non_module("background")

    def set_decoder(self, decoder: MultiHeadMLP) -> None:
        self.register_non_module("decoder", decoder)

    def set_material(self, material: BaseMaterial) -> None:
        self.register_non_module("material", material)

    def set_background(self, background: BaseBackground) -> None:
        self.register_non_module("background", background)
