import argparse
import os
import sys

import torch

from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from mvadapter.utils import make_image_grid

from .inference_tg2mv_sdxl import prepare_pipeline, run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    # I/O
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--save_name", type=str, default="t2tex_sample")
    # Extra
    parser.add_argument("--preprocess_mesh", action="store_true")
    args = parser.parse_args()

    device = args.device
    num_views = 6

    # Prepare pipelines
    pipe = prepare_pipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=num_views,
        device=device,
        dtype=torch.float16,
    )
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
    )
    print("Pipeline ready.")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. run MV-Adapter to generate multi-view images
    images, pos_images, normal_images = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=num_views,
        text=args.text,
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=7.0,
        seed=args.seed,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        device=device,
    )
    mv_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    make_image_grid(images, rows=1).save(mv_path)

    torch.cuda.empty_cache()

    # 2. un-project and complete texture
    out = texture_pipe(
        mesh_path=args.mesh,
        save_dir=args.save_dir,
        save_name=args.save_name,
        uv_unwarp=True,
        preprocess_mesh=args.preprocess_mesh,
        uv_size=4096,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )
    print(f"Output saved to {out.shaded_model_save_path}")
