# trellis_model_manager.py
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Dict, Union
import json
import importlib  # Import the importlib module
from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32

logger = logging.getLogger('model_manager')

__attributes = {
    'SparseStructureDecoder': 'trellis.models.sparse_structure_vae',
    'SparseStructureFlowModel': 'trellis.models.sparse_structure_flow',
    'SLatFlowModel': 'trellis.models.structured_latent_flow',
}

__all__ = list(__attributes.keys())

def __getattr__(name):
    if name in __attributes:
        module_name = __attributes[name]
        module = importlib.import_module(module_name, package=None) # Import the module
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

class TrellisModelManager:
    """
    Basic manager for Trellis models, using ComfyUI's new model path.
    """
    def __init__(self, model_dir: str, config=None, device: str = "cuda"):
        """
        Initialize the model manager with a specific model directory.
        
        Args:
            model_dir (str): Path to model directory (e.g. "models/checkpoints/TRELLIS-image-large")
            config (dict or object): Global configuration for Trellis
            device (str): Device to load models on (e.g. "cuda")
        """
        self.model_dir = model_dir
        # Handle config being either a dict or an object
        if config is None:
            self.device = device
        elif isinstance(config, dict):
            self.device = config.get('device', device)
            self.config = config
        else:
            self.device = getattr(config, 'device', device)
            self.config = config
        self.model = None
        self.dinov2_model = None
        
    def load(self) -> None:
        """Load model configuration and checkpoints"""
        try:
            # Ensure directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            ckpts_folder = os.path.join(self.model_dir, "ckpts")
            os.makedirs(ckpts_folder, exist_ok=True)
            
            # Download model files if needed
            if not os.path.exists(os.path.join(self.model_dir, "pipeline.json")):
                logger.info("Downloading TRELLIS models...")
                try:
                    # Download main pipeline files
                    snapshot_download(
                        repo_id="Stable-X/trellis-normal-v0-1",
                        local_dir=self.model_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=["pipeline.json", "README.md"]
                    )
                    # Download checkpoint files
                    snapshot_download(
                        repo_id="Stable-X/trellis-normal-v0-1",
                        local_dir=ckpts_folder,
                        local_dir_use_symlinks=False,
                        allow_patterns=["*.safetensors", "*.json"],
                        cache_dir=os.path.join(self.model_dir, ".cache")
                    )
                    logger.info("Model files downloaded successfully")
                except Exception as e:
                    logger.error(f"Error downloading model files: {str(e)}")
                    raise
            
            # Load configuration
            self.config = self._load_config()
            
        except Exception as e:
            logger.error(f"Error in load(): {str(e)}")
            raise

    def get_checkpoint_path(self, filename: str) -> str:
        """
        Returns the full path to a checkpoint file.
        """
        ckpts_folder = os.path.join(self.model_dir, "ckpts")
        # Add .safetensors extension if not present
        if not filename.endswith('.safetensors'):
            filename = f"{filename}.safetensors"
        full_path = os.path.join(ckpts_folder, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
        return full_path

    def _load_config(self) -> Dict:
        """Load model configuration from pipeline.json"""
        try:
            config_path = os.path.join(self.model_dir, "pipeline.json")
            
            if os.path.exists(config_path):
                logger.info(f"Loading config from {config_path}")
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                logger.info(f"Config not found locally, downloading from HuggingFace")
                config_path = hf_hub_download(
                    repo_id=f"JeffreyXiang/{os.path.basename(self.model_dir)}", 
                    filename="pipeline.json",
                    cache_dir=os.path.join(self.model_dir, ".cache")
                )
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Debug: Print raw config
            logger.info("Raw config contents:")
            logger.info(json.dumps(config, indent=2))
            
            if not config:
                raise ValueError(f"Could not load valid configuration from {self.model_dir}")
                
            if 'name' not in config:
                config['name'] = 'TrellisImageTo3DPipeline'
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.model_dir}: {e}")
            return {
                'name': 'TrellisImageTo3DPipeline',
                'version': '1.0'
            }

    def load_models(self) -> Dict[str, nn.Module]:
        """Load all required models with current configuration"""
        return {
            'sparse_structure_flow_model': self.get_checkpoint_path("ss_flow_img_dit_L_16l8_fp16"),
            'slat_flow_model': self.get_checkpoint_path("slat_flow_img_dit_L_64l8p2_fp16")
        }

    def load_model_components(self) -> Dict[str, nn.Module]:
        """Loads individual model components."""
        models = {}
        model_paths = self.load_models()
        for name, path in model_paths.items():
            models[name] = models.from_pretrained(path, config=self.config)
            
            # Ensure each model is converted to the desired precision
            if self.config.get('use_fp16', True):
                convert_module_to_f16(models[name])
            else:
                convert_module_to_f32(models[name])
    
        # DINOv2 is handled separately
        # models['image_cond_model'] = self.load_dinov2(self.config.get("dinov2_model", "dinov2_vitl14"))
        
        return models

    def load_dinov2(self, model_name: str):
        """Load DINOv2 model with device, precision, and attention backend management"""
        try:
            # Get configuration values
            use_fp16 = (self.config.get('use_fp16', True) 
                    if isinstance(self.config, dict) 
                    else getattr(self.config, 'use_fp16', True))
            
            # Get attention backend from config
            attention_backend = (self.config.get('attention_backend', 'default')
                    if isinstance(self.config, dict)
                    else getattr(self.config, 'attention_backend', 'default'))

            # Try to load from local path first
            model_path = folder_paths.get_full_path("classifiers", f"{model_name}.pth")
            
            if model_path is None:
                print(f"Downloading {model_name} from torch hub...")
                try:
                    # Load model architecture with specified attention backend
                    model = torch.hub.load('facebookresearch/dinov2', model_name, 
                                         pretrained=True, 
                                         force_reload=False,
                                         trust_repo=True)
                    
                    # Save model for future use
                    save_dir = os.path.join(folder_paths.models_dir, "classifiers")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{model_name}.pth")
                    
                    # Save on CPU to avoid memory issues
                    model = model.cpu()
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved DINOv2 model to {save_path}")
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to download DINOv2 model: {str(e)}")
            else:
                # Load from local path
                print(f"Loading DINOv2 model from {model_path}")
                model = torch.hub.load('facebookresearch/dinov2', model_name, 
                                     pretrained=False,
                                     force_reload=False,
                                     trust_repo=True)
                model.load_state_dict(torch.load(model_path))

            # Move model to specified device and apply precision settings
            model = model.to(self.device)
            if use_fp16:
                model = model.half()
            
            # Set attention backend if specified in config
            if hasattr(model, 'set_attention_backend') and attention_backend != 'default':
                model.set_attention_backend(attention_backend)
            
            model.eval()
            return model
                
        except Exception as e:
            raise RuntimeError(f"Error loading DINOv2 model: {str(e)}")