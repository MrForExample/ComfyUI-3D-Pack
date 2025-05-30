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

from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download

from .mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from .mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from .mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from .mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)
from .mvadapter.utils import make_image_grid, tensor_to_image

try:
    from mmgp import offload, profile_type
    MMGP_AVAILABLE = True
except ImportError:
    MMGP_AVAILABLE = False
    print("Warning: mmgp not available, memory optimization disabled")


def prepare_pipeline(
    base_model: str,
    vae_model: Optional[str] = None,
    unet_model: Optional[str] = None,
    lora_model: Optional[str] = None,
    adapter_path: str = "huanngzh/mv-adapter",
    scheduler: str = "default",
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
        unet_model: Custom UNet model (optional)
        lora_model: LoRA model (optional)
        adapter_path: Path to adapter weights
        scheduler: Scheduler type ("default", "ddpm", "lcm")
        num_views: Number of views to generate
        device: Device for computation
        dtype: Data type
        use_mmgp: Use mmgp for memory optimization
        adapter_local_path: Local path for adapter download
    """
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    if scheduler != "default":
        scheduler_class = None
        if scheduler == "ddpm":
            scheduler_class = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class = LCMScheduler

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
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


def create_bg_remover(device: str = "cuda"):
    """Create background removal function"""
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    def remove_bg(image):
        image_size = image.size
        input_images = transform_image(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
    return remove_bg


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
    if remove_bg_fn is not None:
        reference_image = remove_bg_fn(reference_image)
        reference_image = preprocess_image(reference_image, height, width)
    elif reference_image.mode == "RGBA":
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
    """Создание сетки изображений"""
    return make_image_grid(images, rows=rows)


def pil_to_torch_tensor(image: Image.Image) -> torch.Tensor:
    """Конвертация PIL изображения в torch tensor"""
    import numpy as np
    array = np.array(image)
    tensor = torch.from_numpy(array).float() / 255.0
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def pils_to_torch_tensor(images: List[Image.Image]) -> torch.Tensor:
    """Конвертация списка PIL изображений в torch tensor"""
    tensors = [pil_to_torch_tensor(img).squeeze(0) for img in images]
    return torch.stack(tensors, dim=0) 