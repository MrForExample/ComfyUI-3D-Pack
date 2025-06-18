# __init__.py for Hunyuan3D-2.1
import os
import sys
import torch
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-Hunyuan3D-2.1')

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add paths for Hunyuan3D-2.1 modules
hy3dshape_path = os.path.join(current_dir, "hy3dshape")
hy3dpaint_path = os.path.join(current_dir, "hy3dpaint")

# Add all necessary paths
paths_to_add = [
    current_dir,
    hy3dshape_path,
    hy3dpaint_path,
]

for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        logger.info(f"Added Hunyuan3D-2.1 path to sys.path: {path}")

# Apply torchvision fix before other imports
try:
    from hy3dpaint.utils.torchvision_fix import apply_fix
    apply_fix()
    logger.info("Torchvision fix applied for Hunyuan3D-2.1")
except Exception as e:
    logger.error(f"Warning: Failed to apply torchvision fix: {e}")

# Import mmgp for memory management
try:
    from mmgp import offload, profile_type
    logger.info("mmgp available for memory management")
except ImportError:
    logger.warning("Warning: mmgp module not found")
except Exception as e:
    logger.error(f"Warning: Failed to import mmgp: {e}")

# Import key modules with aliases to avoid conflicts with old Hunyuan3D
from hy3dshape import FaceReducer as FaceReducer_2_1
from hy3dshape import FloaterRemover as FloaterRemover_2_1  
from hy3dshape import DegenerateFaceRemover as DegenerateFaceRemover_2_1
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as Hunyuan3DDiTFlowMatchingPipeline_2_1
from hy3dshape.pipelines import export_to_trimesh as export_to_trimesh_2_1
from hy3dshape.rembg import BackgroundRemover as BackgroundRemover_2_1
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline as Hunyuan3DPaintPipeline_2_1
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig as Hunyuan3DPaintConfig_2_1
from hy3dpaint.convert_utils import create_glb_with_pbr_materials as create_glb_with_pbr_materials_2_1

logger.info("Hunyuan3D-2.1 modules loaded successfully")

# Export modules with safe aliases
__all__ = [
    'FaceReducer_2_1', 
    'FloaterRemover_2_1', 
    'DegenerateFaceRemover_2_1',
    'Hunyuan3DDiTFlowMatchingPipeline_2_1', 
    'export_to_trimesh_2_1',
    'BackgroundRemover_2_1',
    'Hunyuan3DPaintPipeline_2_1', 
    'Hunyuan3DPaintConfig_2_1',
    'create_glb_with_pbr_materials_2_1',
] 