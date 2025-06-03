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
            os.makedirs(self.model_dir, exist_ok=True)
            ckpts_folder = os.path.join(self.model_dir, "ckpts")
            os.makedirs(ckpts_folder, exist_ok=True)
            
            if not os.path.exists(os.path.join(self.model_dir, "pipeline.json")):
                logger.info("Downloading TRELLIS models...")
                try:
                    snapshot_download(
                        repo_id="Stable-X/trellis-normal-v0-1",
                        local_dir=self.model_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=["pipeline.json", "README.md"]
                    )
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
            
            self.config = self._load_config()
            
        except Exception as e:
            logger.error(f"Error in load(): {str(e)}")
            raise

    def get_checkpoint_path(self, filename: str) -> str:
        """
        Returns the full path to a checkpoint file.
        """
        ckpts_folder = os.path.join(self.model_dir, "ckpts")
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
        """Load DINOv2 model from local repo and weights"""
        try:
            local_repo_path = self._ensure_dinov2_repo()
            weights_path = self._ensure_dinov2_weights(model_name)
            
            use_fp16 = (self.config.get('use_fp16', True) 
                    if isinstance(self.config, dict) 
                    else getattr(self.config, 'use_fp16', True))
            
            model = torch.hub.load(
                local_repo_path,
                model_name,
                source='local',
                pretrained=False,  
                force_reload=False,
                trust_repo=True
            )
            
            state_dict = torch.load(weights_path, map_location='cpu')
            
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            model.load_state_dict(state_dict, strict=False)
            
            model = model.to(self.device)
            if use_fp16:
                model = model.half()
            
            model.eval()
            logger.info(f"DINOv2 model {model_name} loaded successfully")
            return model
                
        except Exception as e:
            raise RuntimeError(f"Error loading DINOv2 model: {str(e)}")

    def _ensure_dinov2_repo(self):
        repo_path = os.path.join(os.path.dirname(__file__), 'dinov2')
        
        if not os.path.exists(repo_path):
            logger.error("DINOv2 repository not found. Please ensure dinov2/ folder exists with hubconf.py")
            raise FileNotFoundError(f"DINOv2 repo not found at {repo_path}")
        
        return repo_path

    def _ensure_dinov2_weights(self, model_name: str):
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'Checkpoints', 'facebookresearch', 'dinov2')
        os.makedirs(weights_dir, exist_ok=True)
        
        weights_file = f"{model_name}.pth"
        weights_path = os.path.join(weights_dir, weights_file)
        
        if not os.path.exists(weights_path):
            logger.info(f"Downloading {model_name} weights...")
            
            success = self._download_from_facebook(model_name, weights_path)
            
            if not success:
                logger.info("Trying HuggingFace...")
                self._download_from_huggingface(model_name, weights_path)
                logger.info("Weights downloaded via HuggingFace")
        
        return weights_path
    
    def _download_from_facebook(self, model_name: str, weights_path: str) -> bool:
        try:
            import urllib.request
            import shutil
            
            facebook_urls = {
                'dinov2_vitl14_reg': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'
            }
            
            if model_name not in facebook_urls:
                return False
            
            url = facebook_urls[model_name]
            logger.info(f"Downloading from Facebook...")
            
            temp_path = weights_path + '.tmp'
            urllib.request.urlretrieve(url, temp_path)
            
            shutil.move(temp_path, weights_path)
            
            return True
            
        except Exception as e:
            logger.warning(f"Facebook download failed: {str(e)}")
            temp_path = weights_path + '.tmp'
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def _download_from_huggingface(self, model_name: str, weights_path: str):
        """Download DINOv2 weights from HuggingFace as fallback"""
        try:
            weights_dir = os.path.dirname(weights_path)
            
            model_mapping = {
                'dinov2_vitl14_reg': ('DenisKochetov/dinov2_vitl14_reg', 'dinov2_vitl14_reg.pth')
            }
            
            if model_name not in model_mapping:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(model_mapping.keys())}")
            
            repo_id, filename = model_mapping[model_name]
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=weights_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            weights_file = os.path.basename(weights_path)
            if filename != weights_file:
                final_path = weights_path
                if os.path.exists(downloaded_path) and downloaded_path != final_path:
                    import shutil
                    shutil.move(downloaded_path, final_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download weights for {model_name}: {str(e)}")