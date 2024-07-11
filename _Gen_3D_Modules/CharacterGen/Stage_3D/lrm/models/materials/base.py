from dataclasses import dataclass

from ...utils.base import BaseModule
from ...utils.typing import *


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config
    requires_normal: bool = False
    requires_tangent: bool = False

    def configure(self):
        pass

    def forward(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        raise NotImplementedError

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}
