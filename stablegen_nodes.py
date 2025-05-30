import torch
import os
import trimesh
import numpy as np
from collections import OrderedDict
from huggingface_hub import snapshot_download

# Import StableGen modules
try:
    from .Gen_3D_Modules.Stable3DGen.trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline as Stable3DGenTrellisImageTo3DPipeline
    from .Gen_3D_Modules.Stable3DGen.stablex.pipeline_yoso import YosoPipeline
    from .Gen_3D_Modules.Stable3DGen.stablex.controlnetvae import ControlNetVAEModel

    from .Gen_3D_Modules.Stable3DGen.pipeline_builders import StableGenPipelineBuilder
except ImportError as e:
    print(f"Warning: Could not import StableGen modules: {e}")
    # Create dummy functions for fallback
    def get_available_backends():
        return {'flash_attn': True, 'sdpa': True}
    def get_available_sparse_backends():
        return {'spconv': True}
    def set_attention_backend(backend):
        return True
    def set_sparse_backend(backend, algo):
        return True
    
    class StableGenPipelineBuilder:
        @staticmethod
        def build_trellis_pipeline(*args, **kwargs):
            raise ImportError("StableGen modules not available")
        @staticmethod
        def build_stablex_pipeline(*args, **kwargs):
            raise ImportError("StableGen modules not available")

# Import diffusers components for StableX
try:
    from diffusers.models import AutoencoderKL, UNet2DConditionModel
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import load_file
except ImportError as e:
    print(f"Warning: Could not import diffusers/accelerate: {e}")

import json

from .shared_utils.mesh_utils import Mesh
from .shared_utils.image_utils import torch_imgs_to_pils, pils_to_torch_imgs, pil_make_image_grid

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CKPT_ROOT_PATH = os.path.join(ROOT_PATH, "Checkpoints")
CKPT_DIFFUSERS_PATH = os.path.join(CKPT_ROOT_PATH, "Diffusers")

WEIGHT_DTYPE = torch.float16
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

HF_DOWNLOAD_IGNORE = ["*.yaml", "*.json", "*.py", ".png", ".jpg", ".gif"]


class Load_StableGen_Trellis_Pipeline:
    CATEGORY      = "Comfy3D/Algorithm"
    RETURN_TYPES  = ("DIFFUSERS_PIPE",)
    RETURN_NAMES  = ("trellis_pipe",)
    FUNCTION      = "load"

    _REPO_ID_BASE = "Stable-X"
    CKPT_STABLEGEN_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "Stable3DGen")

    _MODES = {
        "trellis-normal-v0-1": ("trellis-normal-v0-1", 12, 12),  # (repo, ss_steps, slat_steps)
    }

    @classmethod
    def INPUT_TYPES(cls):
        available_attn, available_sparse = StableGenPipelineBuilder.get_available_backends()

        return {
            "required": {
                "model_name": (list(cls._MODES.keys()),),
                "dinov2_model": (["dinov2_vitl14_reg"],),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "attn_backend": (available_attn,),
                "sparse_backend": (available_sparse,),
                "spconv_algo": (["implicit_gemm", "native", "auto"],),
                "smooth_k": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def _build_pipe(cls, repo: str, dinov2_model: str, use_fp16: bool, attn_backend: str, 
                   sparse_backend: str, spconv_algo: str, smooth_k: bool):
        
        return StableGenPipelineBuilder.build_trellis_pipeline(
            repo=repo,
            dinov2_model=dinov2_model,
            use_fp16=use_fp16,
            attn_backend=attn_backend,
            sparse_backend=sparse_backend,
            spconv_algo=spconv_algo,
            smooth_k=smooth_k,
            ckpt_path=cls.CKPT_STABLEGEN_PATH
        )

    def load(self, model_name, dinov2_model, use_fp16, attn_backend, sparse_backend, spconv_algo, smooth_k):
        repo, ss_steps, slat_steps = self._MODES[model_name]
        
        pipe = self.__class__._build_pipe(repo, dinov2_model, use_fp16, attn_backend, sparse_backend, spconv_algo, smooth_k)
        # Store default steps
        pipe.default_ss_steps = ss_steps
        pipe.default_slat_steps = slat_steps
        
        return (pipe,)


class Load_StableGen_StableX_Pipeline:
    CATEGORY     = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("stablex_pipe",)
    FUNCTION     = "load"

    _REPO_ID_BASE = "Stable-X"
    CKPT_STABLEGEN_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "Stable3DGen")

    _MODES = {
        "yoso-normal-v1-8-1": "yoso-normal-v1-8-1",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls._MODES.keys()),),
                "use_fp16": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def _build_pipe(cls, repo: str, use_fp16: bool):
        return StableGenPipelineBuilder.build_stablex_pipeline(
            repo=repo,
            use_fp16=use_fp16,
            ckpt_path=cls.CKPT_STABLEGEN_PATH
        )

    def load(self, model_name, use_fp16):
        repo = self._MODES[model_name]
        pipe = self.__class__._build_pipe(repo, use_fp16)
        return (pipe,)


class StableGen_Trellis_Image_To_3D:
    """
    3D generation pipeline using Trellis model.
    """

    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_pipe": ("DIFFUSERS_PIPE",),
                "images": ("IMAGE", {"list": True}),
                "mode": (["single", "multi"], {"default": "single"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01}),
            }
        }

    @torch.no_grad()
    def run(
        self,
        trellis_pipe,
        images,
        mode="single",
        seed=1234,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.0,
        slat_sampling_steps=12,
        mesh_simplify=0.95
    ):
        if isinstance(images, torch.Tensor):
            images = torch_imgs_to_pils(images)
        
        if not isinstance(images, list) or len(images) == 0:
            raise Exception(f"[StableGen_Trellis_Image_To_3D] 'images' must be a non-empty list of PIL images. Got type: {type(images)}, len: {len(images) if hasattr(images, '__len__') else 'no len'}")

        # Use inference context like in working version
        with trellis_pipe.inference_context():
            # Prepare input based on mode
            if mode == "single":
                if len(images) > 1:
                    print(f"Warning: Single mode selected but {len(images)} images provided. Using first image.")
                image_input = images[0]
                
                # Set up generation parameters for single image
                pipeline_params = StableGenPipelineBuilder.create_trellis_pipeline_params(
                    seed=seed,
                    ss_sampling_steps=ss_sampling_steps,
                    ss_guidance_strength=ss_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    formats=["mesh"]
                )
                
                # Generate 3D mesh using single image method
                outputs = trellis_pipe.run(
                    image_input,
                    **pipeline_params
                )
            else:  # multi mode
                # Set up generation parameters for multi image
                pipeline_params = StableGenPipelineBuilder.create_trellis_pipeline_params(
                    seed=seed,
                    ss_sampling_steps=ss_sampling_steps,
                    ss_guidance_strength=ss_guidance_strength,
                    slat_sampling_steps=slat_sampling_steps,
                    slat_guidance_strength=slat_guidance_strength,
                    formats=["mesh"]
                )
                
                outputs = trellis_pipe.run_multi_image(
                    images,
                    **pipeline_params
                )
            
        try:
            mesh_output = outputs['mesh'][0]
            
            vertices = mesh_output.vertices.cpu().numpy()
            faces = mesh_output.faces.cpu().numpy()
            
            transformation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            vertices = vertices @ transformation_matrix
            
            tri_mesh = trimesh.Trimesh(vertices, faces)
            
            if mesh_simplify < 1.0:
                try:
                    target_faces = int(len(faces) * mesh_simplify)
                    tri_mesh = tri_mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    print(f"Warning: Mesh simplification failed: {e}. Using original mesh.")
            
            # Convert to Mesh object
            mesh = Mesh.load_trimesh(given_mesh=tri_mesh)
            mesh.auto_normal()
            
            return (mesh,)
            
        except Exception as e:
            raise Exception(f"[StableGen_Trellis_Image_To_3D] 3D generation failed: {str(e)}")


# Import MV-Adapter modules
try:
    from .Gen_3D_Modules.MV_Adapter.mvadapter_node_utils import (
        prepare_pipeline as mvadapter_prepare_pipeline,
        run_pipeline as mvadapter_run_pipeline, 
        create_bg_remover,
    )
    from .Gen_3D_Modules.MV_Adapter.mvadapter.utils import make_image_grid
    MVADAPTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MV-Adapter modules: {e}")
    MVADAPTER_AVAILABLE = False


class Load_MVAdapter_Pipeline:
    """Loader pipeline for MV-Adapter (Image to Multi-View)"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("DIFFUSERS_PIPE",)
    RETURN_NAMES = ("mvadapter_pipe",)
    FUNCTION = "load"

    CKPT_MVADAPTER_PATH = os.path.join(CKPT_DIFFUSERS_PATH, "huanngzh", "MV-Adapter")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (["stabilityai/stable-diffusion-xl-base-1.0"], 
                             {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "vae_model": (["madebyollin/sdxl-vae-fp16-fix", "None"], 
                             {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "adapter_path": (["huanngzh/mv-adapter"], {"default": "huanngzh/mv-adapter"}),
                "scheduler": (["default", "ddpm", "lcm"], {"default": "default"}),
                "num_views": ("INT", {"default": 6, "min": 1, "max": 16}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "use_mmgp": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "unet_model": ("STRING", {"default": ""}),
                "lora_model": ("STRING", {"default": ""}),
            }
        }
    @classmethod
    def load(self, base_model, vae_model, adapter_path, scheduler, num_views, 
             use_fp16, use_mmgp, unet_model="", lora_model=""):
        
        
        dtype = torch.float16 if use_fp16 else torch.float32
        vae_model = None if vae_model == "None" else vae_model
        unet_model = None if not unet_model else unet_model
        lora_model = None if not lora_model else lora_model
        
        pipe = mvadapter_prepare_pipeline(
            base_model=base_model,
            vae_model=vae_model,
            unet_model=unet_model,
            lora_model=lora_model,
            adapter_path=adapter_path,
            scheduler=scheduler,
            num_views=num_views,
            device=DEVICE_STR,
            dtype=dtype,
            use_mmgp=use_mmgp,
            adapter_local_path=cls.CKPT_MVADAPTER_PATH
        )
        
        print("MV-Adapter pipeline loaded successfully")
        return (pipe,)
            

class MVAdapter_Image_To_MultiView:
    """Generate multi-view images from single image and 3D mesh"""
    CATEGORY = "Comfy3D/Algorithm"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("multiview_images",)
    FUNCTION = "run"

    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mvadapter_pipe": ("DIFFUSERS_PIPE",),
                "mesh_path": ("STRING", {"default": ""}),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "high quality", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "watermark, ugly, deformed, noisy, blurry, low contrast", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "reference_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "remove_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    def run(self, mvadapter_pipe, mesh_path, reference_image, prompt, negative_prompt, 
            num_inference_steps, guidance_scale, reference_conditioning_scale,
            height, width, seed, remove_background, lora_scale=1.0):
        
        # Convert input image
        if isinstance(reference_image, torch.Tensor):
            reference_images = torch_imgs_to_pils(reference_image)
            reference_image = reference_images[0]
        
        # Check if mesh_path exists
        if not mesh_path or not os.path.exists(mesh_path):
            raise ValueError(f"Mesh path does not exist: {mesh_path}")
        
        # Create background removal function if needed
        remove_bg_fn = None
        if remove_background:
            remove_bg_fn = create_bg_remover(DEVICE_STR)
        
        # Execute generation
        num_views = 6  # Standard number of views
        images, pos_images, normal_images, processed_ref_image = mvadapter_run_pipeline(
            pipe=mvadapter_pipe,
            mesh_path=mesh_path,
            num_views=num_views,
            text=prompt,
            image=reference_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            remove_bg_fn=remove_bg_fn,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            lora_scale=lora_scale,
            device=DEVICE_STR,
        )
        
        # Create image grid  
        grid_image = make_image_grid(images, rows=1)
        
        # Convert PIL image to torch tensor using shared utils
        grid_tensor = pils_to_torch_imgs([grid_image], device=DEVICE_STR)
        
        print(f"Generated multiview images: {grid_tensor.shape}")
        return (grid_tensor,)
            
