from dataclasses import dataclass

from ..renderers.base import BaseRenderer
from ...utils.base import BaseObject
from ...utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(self, renderer: BaseRenderer) -> None:
        self.renderer = renderer

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError


class DummyExporter(Exporter):
    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        # DummyExporter does not export anything
        return []
