import importlib
from .utils.typing import *

def find(cls_string) -> Type:
    cls_full_name = cls_string.split(".")
    module_string = ".".join(cls_full_name[:-1])
    cls_name = cls_full_name[-1]
    print(module_string)
    module = importlib.import_module(module_string, package=__name__)
    cls = getattr(module, cls_name)
    return cls