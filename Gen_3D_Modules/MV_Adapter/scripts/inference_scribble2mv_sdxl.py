import argparse
import random

import cv2
import numpy as np
import torch
from controlnet_aux import HEDdetector, PidiNetDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from PIL import Image

from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import get_orthogonal_camera
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils import make_image_grid


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterT2MVSDXLPipeline
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
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
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
    )

    # ControlNet
    pipe.controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)
    pipe.controlnet.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def preprocess_controlnet_image(image_path, height, width):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (width, height))

    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3] / 255.0
        rgb_channels = image[:, :, :3] / 255.0

        gray_background = np.ones_like(rgb_channels) * 0.5

        image = (
            alpha_channel[..., None] * rgb_channels
            + (1 - alpha_channel[..., None]) * gray_background
        )
        image = (image * 255).astype(np.uint8)

    processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
    image = processor(image, scribble=False)

    # following is some processing to simulate human sketch draw, different threshold can generate different width of lines
    image = np.array(image)
    image = nms(image, 127, 3)
    image = cv2.GaussianBlur(image, (0, 0), 3)

    # higher threshold, thiner line
    random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
    image[image > random_val] = 255
    image[image < 255] = 0

    return Image.fromarray(image)


def run_pipeline(
    pipe,
    num_views,
    text,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    controlnet_images,
    controlnet_conditioning_scale,
    lora_scale=1.0,
    device="cuda",
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 0, 0],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    pipe_kwargs = {}
    if seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    # Prepare controlnet images
    controlnet_image = [
        preprocess_controlnet_image(path, height, width) for path in controlnet_images
    ]
    pipe_kwargs.update(
        {
            "controlnet_image": controlnet_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }
    )

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images

    return images, controlnet_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--controlnet_images", type=str, nargs="+", required=True)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    images, controlnet_images = run_pipeline(
        pipe,
        num_views=args.num_views,
        text=args.text,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        controlnet_images=args.controlnet_images,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        lora_scale=args.lora_scale,
        device=args.device,
    )
    make_image_grid(images, rows=1).save(args.output)
    make_image_grid(controlnet_images, rows=1).save(
        args.output.rsplit(".", 1)[0] + "_controlnet.png"
    )
