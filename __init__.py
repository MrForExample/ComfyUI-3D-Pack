import os
import sys
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

sys.path.append('.')   # for portable version

all_module_names = (
    "diff_rast",
    "gaussian_splatting",
    "shared_utils",
    "mesh_processer",
)
for module_name in all_module_names:
    module_path = os.path.join(os.path.dirname(__file__), module_name)
    sys.path.append(module_path)

    
import diff_rast.diff_texturing

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
