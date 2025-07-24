import argparse
import os
import sys
from glob import glob
import time
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "partcrafter_src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from partcrafter_src.utils.data_utils import get_colored_mesh_composition, scene_to_parts, load_surfaces
from partcrafter_src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from partcrafter_src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from partcrafter_src.utils.image_utils import prepare_image
from partcrafter_src.models.briarmbg import BriaRMBG

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    num_parts: int,
    rmbg_net: Any,
    seed: int,
    num_tokens: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    rmbg: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> trimesh.Scene:

    if rmbg:
        img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    else:
        img_pil = Image.open(image_input)
    start_time = time.time()
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    for i in range(len(outputs)):
        if outputs[i] is None:
            # If the generated mesh is None (decoding error), use a dummy mesh
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
    return outputs, img_pil

MAX_NUM_PARTS = 16

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16


    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--num_parts", type=int, required=True, help="number of parts to generate")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9)
    parser.add_argument("--use_flash_decoder", action="store_true")
    parser.add_argument("--rmbg", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sampling_version", type=int, default=1, choices=[1, 2], 
                        help="Sampling version: 1=original, 2=fixed (per-batch with normalization)")
    args = parser.parse_args()

    assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"

    # download pretrained weights
    partcrafter_weights_dir = "pretrained_weights/PartCrafter"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    # ignore json file
    snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=partcrafter_weights_dir, ignore_patterns=["*.json"])
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
    
    # DEBUG: Set sampling version
    print(f"🎛️  Setting sampling version to {args.sampling_version}")
    if hasattr(pipe.vae, 'set_sampling_version'):
        pipe.vae.set_sampling_version(args.sampling_version)
    else:
        print("⚠️  Warning: VAE doesn't have set_sampling_version method")

    set_seed(args.seed)

    # run inference
    outputs, processed_image = run_triposg(
        pipe,
        image_input=args.image_path,
        num_parts=args.num_parts,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        rmbg=args.rmbg,
        dtype=dtype,
        device=device,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    
    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)

    for i, mesh in enumerate(outputs):
        mesh.export(os.path.join(export_dir, f"part_{i:02}.glb"))
    
    merged_mesh = get_colored_mesh_composition(outputs)
    merged_mesh.export(os.path.join(export_dir, "object.glb"))
    print(f"Generated {len(outputs)} parts and saved to {export_dir}")

    if args.render:
        print("Start rendering...")
        num_views = 36
        radius = 4
        fps = 18
        rendered_images = render_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_normals = render_normal_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_grids = make_grid_for_images_or_videos(
            [
                [processed_image] * num_views,
                rendered_images,
                rendered_normals,
            ], 
            nrow=3
        )
        export_renderings(
            rendered_images,
            os.path.join(export_dir, "rendering.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_normals,
            os.path.join(export_dir, "rendering_normal.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_grids,
            os.path.join(export_dir, "rendering_grid.gif"),
            fps=fps,
        )

        rendered_image, rendered_normal, rendered_grid = rendered_images[0], rendered_normals[0], rendered_grids[0]
        rendered_image.save(os.path.join(export_dir, "rendering.png"))
        rendered_normal.save(os.path.join(export_dir, "rendering_normal.png"))
        rendered_grid.save(os.path.join(export_dir, "rendering_grid.png"))
        print("Rendering done.")

