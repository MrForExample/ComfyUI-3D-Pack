#__init__.py
import os
import sys
import torch
import logging
import platform
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-Hi3DGen')

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both current and parent dir to handle different installation scenarios
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add trellis package path
trellis_path = os.path.join(current_dir, "trellis")
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.insert(0, trellis_path)
    # logger.info(f"Added trellis path to sys.path: {trellis_path}")

# Add stablx package path
stablex_path = os.path.join(current_dir, "stablex")
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.insert(0, trellis_path)
    # logger.info(f"Added stablex path to sys.path: {trellis_path}")

# Verify trellis package is importable
try:
    import trellis
    # logger.info("Trellis package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import trellis package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Verify stablex package is importable
try:
    import stablex
    # logger.info("stablex package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import stablex package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("trellis", os.path.join(folder_paths.models_dir, "trellis"))
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.models_dir, "checkpoints"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("stablex", os.path.join(folder_paths.models_dir, "stablex"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

