import importlib
import torch
import torch.nn as nn

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'SLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str) -> nn.Module:
    """
    Load a pretrained model.
    
    Args:
        path (str): Full path to model file or HuggingFace repo ID with model name
    """
    import os
    import json
    from safetensors.torch import load_file

    # Split path into directory and model name
    path = os.path.normpath(path)
    model_dir = os.path.dirname(os.path.dirname(path))  # Go up two levels (past ckpts/)
    model_name = os.path.basename(path)

    is_local = os.path.exists(model_dir)

    if is_local:
        # For local paths
        print(f"Loading local model: {model_name}")
        model_name = model_name.replace('ckpts/', '').replace('ckpts\\', '')
        config_path = os.path.normpath(os.path.join(model_dir, "ckpts", f"{model_name}.json"))
        weights_path = os.path.normpath(os.path.join(model_dir, "ckpts", f"{model_name}.safetensors"))
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create model
        model = create_model_from_config(config)
        
        # Load weights
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        
    else:
        # For HuggingFace paths
        from huggingface_hub import hf_hub_download
        
        config_file = hf_hub_download(path, f"{model_name}.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        model = create_model_from_config(config)
        
        weights_file = hf_hub_download(path, f"{model_name}.safetensors")
        state_dict = load_file(weights_file)
        model.load_state_dict(state_dict)
    
    return model

def create_model_from_config(config):
    """Helper function to create model from config"""
    #print(f"Creating model from config: {config}")
    model_type = config.get('type') or config.get('name')
    #print(f"Model type: {model_type}")
    #print(f"Available model types: {list(__attributes.keys())}")
    if not model_type in __attributes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = __getattr__(model_type)
    #print(f"Model class: {model_class}")
    args = config.get('args', {})
    #print(f"Model args: {args}")
    return model_class(**args)


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_vae import SLatEncoder, SLatGaussianDecoder, SLatRadianceFieldDecoder, SLatMeshDecoder
    from .structured_latent_flow import SLatFlowModel
