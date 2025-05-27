import os
import json
import torch
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
            
        except Exception as e:
            raise Exception(f"Failed to build Trellis pipeline: {e}")
    
    @staticmethod
    def build_stablex_pipeline(repo: str, use_fp16: bool, ckpt_path: str):
        
        StableGenPipelineBuilder._ensure_stablex_weights(repo, ckpt_path)

        model_dir = os.path.join(ckpt_path, "stablex", repo)
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        try:
            config_path = os.path.join(model_dir, 'unet', 'config.json')
            unet_ckpt_path = os.path.join(model_dir, 'unet', 'diffusion_pytorch_model.fp16.safetensors')
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found at {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)

            with init_empty_weights():
                unet = UNet2DConditionModel(**config)

            if os.path.exists(unet_ckpt_path):
                unet_sd = load_file(unet_ckpt_path)
            else:
                raise FileNotFoundError(f"No checkpoint found at {unet_ckpt_path}")

            for name, param in unet.named_parameters():
                set_module_tensor_to_device(unet, name, device=DEVICE, dtype=torch_dtype, value=unet_sd[name])

            vae = AutoencoderKL.from_pretrained(
                model_dir, 
                subfolder="vae", 
                variant="fp16" if use_fp16 else None,
                torch_dtype=torch_dtype
            )
            
            controlnet = ControlNetVAEModel.from_pretrained(
                model_dir, 
                subfolder="controlnet", 
                variant="fp16" if use_fp16 else None,
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
        
        if not os.path.exists(base_dir) or not os.listdir(base_dir):
            print(f"Downloading {repo} to {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            snapshot_download(
                repo_id=f"Stable-X/{repo}",
                repo_type="model",
                local_dir=base_dir,
                resume_download=True,
                ignore_patterns=HF_DOWNLOAD_IGNORE
            )
    
    @staticmethod
    def _ensure_stablex_weights(repo: str, ckpt_path: str):
        base_dir = os.path.join(ckpt_path, "stablex", repo)
        
        if not os.path.exists(base_dir) or not os.listdir(base_dir):
            print(f"Downloading {repo} to {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            snapshot_download(
                repo_id=f"Stable-X/{repo}",
                repo_type="model",
                local_dir=base_dir,
                resume_download=True,
                ignore_patterns=["*text_encoder*", "tokenizer*", "*scheduler*"]
            )
    
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