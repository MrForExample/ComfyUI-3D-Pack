import os
import json
import torch
import shutil
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from huggingface_hub import snapshot_download

from .stablex.pipeline_yoso import YosoPipeline
from .stablex.controlnetvae import ControlNetVAEModel
from .trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline as Stable3DGenTrellisImageTo3DPipeline
from .trellis.backend_config import (
    set_attention_backend,
    set_sparse_backend,
    get_available_backends,
    get_available_sparse_backends
)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
HF_DOWNLOAD_IGNORE = ["*.yaml", "*.json", "*.py", ".png", ".jpg", ".gif"]


class StableGenPipelineBuilder:
    
    @staticmethod
    def get_available_backends():
        try:
            attn_backends = get_available_backends()
            sparse_backends = get_available_sparse_backends()
        except:
            attn_backends = {'flash_attn': True, 'sdpa': True}
            sparse_backends = {'spconv': True}

        available_attn = [k for k, v in attn_backends.items() if v]
        if not available_attn:
            available_attn = ['flash_attn']

        available_sparse = [k for k, v in sparse_backends.items() if v]
        if not available_sparse:
            available_sparse = ['spconv']
            
        return available_attn, available_sparse
    
    @staticmethod
    def create_trellis_pipeline_params(
        seed: int = 1234,
        ss_sampling_steps: int = 12,
        ss_guidance_strength: float = 7.5,
        slat_sampling_steps: int = 12,
        slat_guidance_strength: float = 3.0,
        formats: list = None
    ):
        if formats is None:
            formats = ["mesh"]
            
        return {
            "num_samples": 1,
            "seed": seed,
            "formats": formats,
            "preprocess_image": True,
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            "slat_sampler_params": {
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            }
        }
    
    @staticmethod
    def build_trellis_pipeline(
        repo: str, 
        dinov2_model: str, 
        use_fp16: bool, 
        attn_backend: str,
        sparse_backend: str, 
        spconv_algo: str, 
        smooth_k: bool,
        ckpt_path: str
    ):
        
        StableGenPipelineBuilder._ensure_trellis_weights(repo, ckpt_path)
        
        StableGenPipelineBuilder._setup_environment(attn_backend, sparse_backend, spconv_algo, smooth_k)

        model_dir = os.path.join(ckpt_path, "trellis", repo)

        try:
            pipe = Stable3DGenTrellisImageTo3DPipeline.from_pretrained(
                model_dir,
                dinov2_model=dinov2_model
            )

            if pipe is None:
                raise Exception("Pipeline returned None from from_pretrained")

            pipe = StableGenPipelineBuilder._optimize_trellis_pipeline(pipe, use_fp16)
            return pipe
            
        except FileNotFoundError as e:
            if "Weights file not found" in str(e):
                print(f"Trellis weights missing, re-downloading {repo}...")
                StableGenPipelineBuilder._ensure_trellis_weights(repo, ckpt_path)
                
                try:
                    pipe = Stable3DGenTrellisImageTo3DPipeline.from_pretrained(
                        model_dir,
                        dinov2_model=dinov2_model
                    )
                    
                    if pipe is None:
                        raise Exception("Pipeline returned None from from_pretrained")

                    pipe = StableGenPipelineBuilder._optimize_trellis_pipeline(pipe, use_fp16)
                    return pipe
                    
                except Exception as retry_e:
                    raise Exception(f"Failed to build Trellis pipeline after re-download: {retry_e}")
            else:
                raise Exception(f"Failed to build Trellis pipeline: {e}")
        except Exception as e:
            raise Exception(f"Failed to build Trellis pipeline: {e}")
    
    @staticmethod
    def build_stablex_pipeline(repo: str, use_fp16: bool, ckpt_path: str):
        
        StableGenPipelineBuilder._ensure_stablex_weights(repo, ckpt_path)

        model_dir = os.path.join(ckpt_path, "stablex", repo)
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        try:
            config_path = os.path.join(model_dir, 'unet', 'config.json')
            
            if not os.path.exists(config_path):
                print(f"UNet config not found, re-downloading {repo}...")
                StableGenPipelineBuilder._ensure_stablex_weights(repo, ckpt_path)
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config not found at {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)

            with init_empty_weights():
                unet = UNet2DConditionModel(**config)

            unet_file_candidates = [
                'diffusion_pytorch_model.fp16.safetensors',
                'diffusion_pytorch_model.safetensors', 
                'pytorch_model.fp16.safetensors',
                'pytorch_model.safetensors'
            ]
            
            unet_sd = None
            unet_ckpt_path = None
            
            for filename in unet_file_candidates:
                candidate_path = os.path.join(model_dir, 'unet', filename)
                if os.path.exists(candidate_path):
                    unet_ckpt_path = candidate_path
                    print(f"Found UNet checkpoint: {unet_ckpt_path}")
                    unet_sd = load_file(unet_ckpt_path)
                    break
            
            if unet_sd is None:
                print(f"UNet checkpoint not found, re-downloading {repo}...")
                StableGenPipelineBuilder._ensure_stablex_weights(repo, ckpt_path)
                
                for filename in unet_file_candidates:
                    candidate_path = os.path.join(model_dir, 'unet', filename)
                    if os.path.exists(candidate_path):
                        unet_ckpt_path = candidate_path
                        print(f"Found UNet checkpoint after re-download: {unet_ckpt_path}")
                        unet_sd = load_file(unet_ckpt_path)
                        break
                
                if unet_sd is None:
                    unet_dir = os.path.join(model_dir, 'unet')
                    if os.path.exists(unet_dir):
                        available_files = [f for f in os.listdir(unet_dir) if os.path.isfile(os.path.join(unet_dir, f))]
                        print(f"Available files in unet directory: {available_files}")
                    
                    raise FileNotFoundError(f"No UNet checkpoint found in {unet_dir}. Searched for: {unet_file_candidates}")

            for name, param in unet.named_parameters():
                set_module_tensor_to_device(unet, name, device=DEVICE, dtype=torch_dtype, value=unet_sd[name])

            try:
                vae = AutoencoderKL.from_pretrained(
                    model_dir, 
                    subfolder="vae", 
                    variant="fp16" if use_fp16 else None,
                    torch_dtype=torch_dtype
                )
            except Exception as e:
                print(f"Warning: Failed to load VAE with variant, trying without: {e}")
                vae = AutoencoderKL.from_pretrained(
                    model_dir, 
                    subfolder="vae", 
                    torch_dtype=torch_dtype
                )
            
            try:
                controlnet = ControlNetVAEModel.from_pretrained(
                    model_dir, 
                    subfolder="controlnet", 
                    variant="fp16" if use_fp16 else None,
                    torch_dtype=torch_dtype
                )
            except Exception as e:
                print(f"Warning: Failed to load ControlNet with variant, trying without: {e}")
                controlnet = ControlNetVAEModel.from_pretrained(
                    model_dir, 
                    subfolder="controlnet", 
                    torch_dtype=torch_dtype
                )

            pipeline = YosoPipeline(
                unet=unet,
                vae=vae,
                controlnet=controlnet,
            )

            pipeline.to(DEVICE)
            
            if torch_dtype == torch.float16 and DEVICE.type == "cuda":
                pipeline.unet = pipeline.unet.half()
                pipeline.vae = pipeline.vae.half()
                pipeline.controlnet = pipeline.controlnet.half()
            
            return pipeline
            
        except Exception as e:
            raise Exception(f"Failed to build StableX pipeline: {e}")
    
    @staticmethod
    def _ensure_trellis_weights(repo: str, ckpt_path: str):
        base_dir = os.path.join(ckpt_path, "trellis", repo)
        
        def check_files_exist():
            required_files = [
                "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                "ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors", 
                "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors",
                "ckpts/slat_flow_normal_dit_L_64l8p2_fp16.safetensors",
                "ckpts/ss_dec_conv3d_16l8_fp16.json",
                "ckpts/ss_flow_normal_dit_L_16l8_fp16.json",
                "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json", 
                "ckpts/slat_flow_normal_dit_L_64l8p2_fp16.json"
            ]
            
            for req_file in required_files:
                if not os.path.exists(os.path.join(base_dir, req_file)):
                    print(f"Missing required file: {req_file}")
                    return False
                    
            return True
        
        if not os.path.exists(base_dir) or not os.listdir(base_dir) or not check_files_exist():
            print(f"Downloading {repo} to {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Download attempt {attempt + 1}/{max_retries}")
                    snapshot_download(
                        repo_id=f"Stable-X/{repo}",
                        repo_type="model",
                        local_dir=base_dir,
                        resume_download=True,
                        ignore_patterns=HF_DOWNLOAD_IGNORE
                    )
                    
                    if check_files_exist():
                        print(f"Successfully downloaded {repo}")
                        break
                    else:
                        print(f"Download incomplete, retrying... ({attempt + 1}/{max_retries})")
                        
                except Exception as e:
                    print(f"Download failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to download {repo} after {max_retries} attempts")
                    
            if not check_files_exist():
                raise Exception(f"Failed to download required files for {repo}")
    
    @staticmethod
    def _ensure_stablex_weights(repo: str, ckpt_path: str):
        base_dir = os.path.join(ckpt_path, "stablex", repo)
        
        def check_files_exist():
            required_files = [
                "unet/config.json",
                "vae/config.json", 
                "controlnet/config.json",
                "model_index.json"
            ]
            
            unet_files = [
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "unet/diffusion_pytorch_model.safetensors"
            ]
            
            for req_file in required_files:
                if not os.path.exists(os.path.join(base_dir, req_file)):
                    print(f"Missing required file: {req_file}")
                    return False
            
            unet_found = False
            for unet_file in unet_files:
                if os.path.exists(os.path.join(base_dir, unet_file)):
                    unet_found = True
                    break
            
            if not unet_found:
                print(f"No UNet model file found. Checked: {unet_files}")
                return False
                
            return True
        
        if not os.path.exists(base_dir) or not os.listdir(base_dir) or not check_files_exist():
            print(f"Downloading {repo} to {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Download attempt {attempt + 1}/{max_retries}")
                    snapshot_download(
                        repo_id=f"Stable-X/{repo}",
                        repo_type="model",
                        local_dir=base_dir,
                        resume_download=True,
                        ignore_patterns=["*text_encoder*", "tokenizer*", "*scheduler*"]
                    )
                    
                    if check_files_exist():
                        print(f"Successfully downloaded {repo}")
                        break
                    else:
                        print(f"Download incomplete, retrying... ({attempt + 1}/{max_retries})")
                        
                except Exception as e:
                    print(f"Download failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to download {repo} after {max_retries} attempts")
                    
            if not check_files_exist():
                raise Exception(f"Failed to download required files for {repo}")
    
    @staticmethod
    def _setup_environment(attn_backend: str, sparse_backend: str, spconv_algo: str, smooth_k: bool):
        try:
            success = set_attention_backend(attn_backend)
            if not success:
                print(f"Warning: Failed to set {attn_backend}, fallback to default")

            success2 = set_sparse_backend(sparse_backend, spconv_algo)
            if not success2:
                print(f"Warning: Failed to set {sparse_backend}, fallback to default")

            os.environ['SAGEATTN_SMOOTH_K'] = '1' if smooth_k else '0'
        except Exception as e:
            print(f"Warning: Could not setup environment: {e}")
    
    @staticmethod
    def _optimize_trellis_pipeline(pipeline, use_fp16: bool = True):
        if DEVICE.type == "cuda":
            try:
                if hasattr(pipeline, 'cuda'):
                    pipeline.cuda()

                if use_fp16:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing(slice_size="auto")
                    if hasattr(pipeline, 'half'):
                        pipeline.half()
            except Exception as e:
                print(f"Warning: Some pipeline optimizations failed: {str(e)}")

        return pipeline 