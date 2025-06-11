"""
Utilities for ComfyUI MV-Adapter nodes
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Union, List
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download

from .mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from .mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from .mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from .mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)
from .mvadapter.utils import make_image_grid, tensor_to_image
from .mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho

try:
    from mmgp import offload, profile_type
    MMGP_AVAILABLE = True
except ImportError:
    MMGP_AVAILABLE = False
    print("Warning: mmgp not available, memory optimization disabled")


def prepare_pipeline(
    base_model: str,
    vae_model: Optional[str] = None,
    lora_model: Optional[str] = None,
    adapter_path: str = "huanngzh/mv-adapter",
    scheduler: str = "ddpm",
    num_views: int = 6,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_mmgp: bool = False,
    adapter_local_path: Optional[str] = None,
):
    """
    Prepare MV-Adapter pipeline
    
    Args:
        base_model: Base Stable Diffusion XL model
        vae_model: Custom VAE model (optional)
        lora_model: LoRA model (optional)
        adapter_path: Path to adapter weights
        scheduler: Scheduler type ("ddpm")
        num_views: Number of views to generate
        device: Device for computation
        dtype: Data type
        use_mmgp: Use mmgp for memory optimization
        adapter_local_path: Local path for adapter download
    """
    # Load vae if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    
    # Prepare pipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load DDPM scheduler
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    
    # Load adapter weights
    weight_name = "mvadapter_ig2mv_sdxl.safetensors"
    pipe.load_custom_adapter(adapter_path, weight_name=weight_name, local_cache_dir=adapter_local_path)

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    # Apply mmgp optimization if available and requested
    if use_mmgp and MMGP_AVAILABLE:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        all_models = {
            "vae": pipe.vae,
            "text_encoder": pipe.text_encoder,
            "text_encoder_2": pipe.text_encoder_2,
            "unet": pipe.unet,
            "cond_encoder": pipe.cond_encoder,  
        }
        
        if hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
            all_models["image_encoder"] = pipe.image_encoder
        
        offload.profile(all_models, profile_type.HighRAM_LowVRAM)
        print("mmgp profiling applied successfully.")
    
    # Enable VAE slicing for memory efficiency
    pipe.enable_vae_slicing()

    return pipe


def preprocess_image(image: Image.Image, height: int, width: int) -> Image.Image:
    """Preprocess image for MV-Adapter"""
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def run_pipeline(
    pipe,
    mesh_path: str,
    num_views: int,
    text: str,
    image: Union[str, Image.Image],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    remove_bg_fn=None,
    reference_conditioning_scale: float = 1.0,
    negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale: float = 1.0,
    device: str = "cuda",
):
    """
    Execute multi-view image generation
    """
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device)

    mesh = load_mesh(mesh_path, rescale=True, device=device)
    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    
    pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
    normal_images = tensor_to_image(
        (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(device)
    )

    # Prepare image
    reference_image = Image.open(image) if isinstance(image, str) else image
    if reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference_image,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images

    return images, pos_images, normal_images, reference_image


def create_image_grid(images: List[Image.Image], rows: int = 1) -> Image.Image:
    return make_image_grid(images, rows=rows)


def pil_to_torch_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np
    array = np.array(image)
    tensor = torch.from_numpy(array).float() / 255.0
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def pils_to_torch_tensor(images: List[Image.Image]) -> torch.Tensor:
    tensors = [pil_to_torch_tensor(img).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0)


def prepare_texture_pipeline(
    upscaler_ckpt_path: Optional[str] = None,
    inpaint_ckpt_path: Optional[str] = None,
    device: str = "cuda",
    use_mmgp: bool = False,
):
    """
    Prepare texture projection pipeline
    
    Args:
        upscaler_ckpt_path: Path to upscaler checkpoint (optional)
        inpaint_ckpt_path: Path to inpaint checkpoint (optional)
        device: Device for computation
        use_mmgp: Use mmgp for memory optimization
    """
    from .mvadapter.pipelines.pipeline_texture import TexturePipeline
    
    if upscaler_ckpt_path and not os.path.exists(upscaler_ckpt_path):
        print(f"Warning: Upscaler checkpoint not found: {upscaler_ckpt_path}, using None")
        upscaler_ckpt_path = None
        
    if inpaint_ckpt_path and not os.path.exists(inpaint_ckpt_path):
        print(f"Warning: Inpaint checkpoint not found: {inpaint_ckpt_path}, using None")
        inpaint_ckpt_path = None
    
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path=upscaler_ckpt_path,
        inpaint_ckpt_path=inpaint_ckpt_path,
        device=device
    )
    
    # Apply mmgp optimization if available and requested
    if use_mmgp and MMGP_AVAILABLE:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        all_models = {}
        
        if hasattr(texture_pipe, 'upscaler') and texture_pipe.upscaler is not None:
            all_models["upscaler"] = texture_pipe.upscaler
            
        if hasattr(texture_pipe, 'inpainter') and texture_pipe.inpainter is not None:
            all_models["inpainter"] = texture_pipe.inpainter
        
        if all_models:
            offload.profile(all_models, profile_type.HighRAM_LowVRAM)
            print("mmgp profiling applied to texture pipeline successfully.")
        else:
            print("No models loaded for texture pipeline, skipping mmgp profiling.")
    
    return texture_pipe


def run_texture_pipeline(
    texture_pipe,
    multiview_images: List[Image.Image],
    mesh_path: str,
    save_dir: str,
    save_name: str = "textured_model",
    uv_size: int = 4096,
    view_upscale: bool = True,
    inpaint_mode: str = "view",
    uv_unwarp: bool = True,
    preprocess_mesh: bool = False,
    move_to_center: bool = False,
    front_x: bool = True,
    camera_azimuth_deg: List[float] = None,
    camera_elevation_deg: List[float] = None,
    camera_distance: float = 1.0,
    camera_ortho_scale: float = 1.1,
    debug_mode: bool = False,
    apply_dilate: bool = True,
):
    """
    Execute texture projection pipeline
    
    Args:
        texture_pipe: Prepared texture pipeline
        multiview_images: List of PIL images (6 views)
        mesh_path: Path to 3D mesh file
        save_dir: Directory to save results
        save_name: Name for saved files
        uv_size: UV texture resolution
        view_upscale: Enable view upscaling
        inpaint_mode: Inpainting mode ("none", "uv", "view")
        uv_unwarp: Enable UV unwrapping
        preprocess_mesh: Enable mesh preprocessing
        move_to_center: Move mesh to center
        front_x: Front face along X axis
        camera_azimuth_deg: Camera azimuth angles
        camera_elevation_deg: Camera elevation angles
        camera_distance: Camera distance
        camera_ortho_scale: Orthographic scale
        debug_mode: Enable debug mode
        apply_dilate: Apply dilate to remove texture seams
    """
    from .mvadapter.pipelines.pipeline_texture import ModProcessConfig
    from .mvadapter.utils import make_image_grid
    
    if not mesh_path or not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file does not exist: {mesh_path}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if camera_azimuth_deg is None:
        camera_azimuth_deg = [0, 90, 180, 270, 180, 180]
    if camera_elevation_deg is None:
        camera_elevation_deg = [0, 0, 0, 0, 89.99, -89.99]
    
    azimuth_deg_corrected = [x - 90 for x in camera_azimuth_deg]
    
    grid_image = make_image_grid(multiview_images, rows=1)
    temp_grid_path = os.path.join(save_dir, f"{save_name}_temp_grid.png")
    grid_image.save(temp_grid_path)
    
    try:
        rgb_process_config = ModProcessConfig(
            view_upscale=view_upscale,
            inpaint_mode=inpaint_mode
        )
        
        output = texture_pipe(
            mesh_path=mesh_path,
            save_dir=save_dir,
            save_name=save_name,
            move_to_center=move_to_center,
            front_x=front_x,
            uv_unwarp=uv_unwarp,
            preprocess_mesh=preprocess_mesh,
            uv_size=uv_size,
            rgb_path=temp_grid_path,
            rgb_process_config=rgb_process_config,
            camera_elevation_deg=camera_elevation_deg,
            camera_azimuth_deg=azimuth_deg_corrected,
            camera_distance=camera_distance,
            camera_ortho_scale=camera_ortho_scale,
            debug_mode=debug_mode,
            apply_dilate=apply_dilate,
        )
        
        if os.path.exists(temp_grid_path):
            os.remove(temp_grid_path)
        
        return output
        
    except Exception as e:
        if os.path.exists(temp_grid_path):
            os.remove(temp_grid_path)
        raise e


def download_texture_checkpoints(texture_ckpt_dir, upscaler_ckpt_path, inpaint_ckpt_path):
    """Download texture checkpoint files if they don't exist"""
    import subprocess
    
    os.makedirs(texture_ckpt_dir, exist_ok=True)
    
    checkpoints = [
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "path": upscaler_ckpt_path,
            "name": "RealESRGAN upscaler"
        },
        {
            "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt", 
            "path": inpaint_ckpt_path,
            "name": "Big-LaMa inpainter"
        }
    ]
    
    for ckpt in checkpoints:
        if not os.path.exists(ckpt["path"]):
            print(f"Downloading {ckpt['name']} to {ckpt['path']}...")
            
            success = False
            
            # Try wget first
            try:
                subprocess.run([
                    "wget", ckpt["url"], "-O", ckpt["path"]
                ], check=True, capture_output=True, text=True)
                print(f"Successfully downloaded {ckpt['name']} using wget")
                success = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            if not success:
                try:
                    subprocess.run([
                        "curl", "-L", ckpt["url"], "-o", ckpt["path"]
                    ], check=True, capture_output=True, text=True)
                    print(f"Successfully downloaded {ckpt['name']} using curl")
                    success = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            if not success:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(ckpt["url"], ckpt["path"])
                    print(f"Successfully downloaded {ckpt['name']} using urllib")
                    success = True
                except Exception as e:
                    print(f"Failed to download {ckpt['name']} using urllib: {e}")
            
            if not success:
                print(f"Failed to download {ckpt['name']}. Please download manually:")
                print(f"wget {ckpt['url']} -O {ckpt['path']}")
                print(f"or")
                print(f"curl -L {ckpt['url']} -o {ckpt['path']}")
        else:
            print(f"{ckpt['name']} already exists at {ckpt['path']}") 


def prepare_tg2mv_pipeline(
    base_model: str,
    vae_model: Optional[str] = None,
    lora_model: Optional[str] = None,
    adapter_path: str = "huanngzh/mv-adapter",
    scheduler: str = "ddpm",
    num_views: int = 6,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_mmgp: bool = False,
    adapter_local_path: Optional[str] = None,
):
    """
    Prepare MV-Adapter TG2MV (Text-Guided to Multi-View) pipeline
    
    Args:
        base_model: Base Stable Diffusion XL model
        vae_model: Custom VAE model (optional)
        lora_model: LoRA model (optional)
        adapter_path: Path to adapter weights
        scheduler: Scheduler type ("ddpm")
        num_views: Number of views to generate
        device: Device for computation
        dtype: Data type
        use_mmgp: Use mmgp for memory optimization
        adapter_local_path: Local path for adapter download
    """
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)

    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, torch_dtype=dtype, **pipe_kwargs)

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    
    weight_name = "mvadapter_tg2mv_sdxl.safetensors"
    pipe.load_custom_adapter(adapter_path, weight_name=weight_name, local_cache_dir=adapter_local_path)

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()  
    
    if hasattr(pipe, 'enable_model_cpu_offload'):
        if not use_mmgp:  
            pipe.enable_model_cpu_offload()
            print("[INFO] Включен CPU offload для экономии VRAM")

    if use_mmgp and MMGP_AVAILABLE:
        torch.cuda.empty_cache()
        gc.collect()
        
        all_models = {
            "vae": pipe.vae,
            "text_encoder": pipe.text_encoder,
            "text_encoder_2": pipe.text_encoder_2,
            "unet": pipe.unet,
            "cond_encoder": pipe.cond_encoder,  
        }
        
        if hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
            all_models["image_encoder"] = pipe.image_encoder
        
        try:
            offload.profile(all_models, profile_type.HighRAM_LowVRAM)
            print("mmgp profiling applied successfully.")
        except Exception as e:
            print(f"[WARNING] mmgp failed, continuing without it: {e}")
    
    torch.cuda.empty_cache()
    gc.collect()

    return pipe


def run_tg2mv_pipeline(
    pipe,
    mesh_path: str,
    num_views: int,
    text: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale: float = 1.0,
    device: str = "cuda",
):
    """
    Execute text-guided multiview generation (TG2MV)
    
    Args:
        pipe: TG2MV pipeline
        mesh_path: Path to 3D mesh file for guidance
        num_views: Number of views to generate
        text: Text prompt
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed
        negative_prompt: Negative prompt
        lora_scale: LoRA scale
        device: Device for computation
    """
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device)

    mesh = load_mesh(mesh_path, rescale=True, device=device)
    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    
    pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
    normal_images = tensor_to_image(
        (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(device)
    )

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images

    return images, pos_images, normal_images 