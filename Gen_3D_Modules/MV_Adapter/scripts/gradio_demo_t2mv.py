import argparse
import random

import gradio as gr
import numpy as np

# import spaces
import torch

from .inference_t2mv_sdxl import prepare_pipeline, run_pipeline

# Base model
parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--scheduler", type=str, default=None)
args = parser.parse_args()
base_model = args.base_model
scheduler = args.scheduler

# Device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_VIEWS = 6
HEIGHT = 768
WIDTH = 768
MAX_SEED = np.iinfo(np.int32).max

pipe = prepare_pipeline(
    base_model=base_model,
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_model=None,
    lora_model=None,
    adapter_path="huanngzh/mv-adapter",
    scheduler=scheduler,
    num_views=NUM_VIEWS,
    device=device,
    dtype=dtype,
)


# @spaces.GPU()
def infer(
    prompt,
    seed=42,
    randomize_seed=False,
    guidance_scale=7.0,
    num_inference_steps=50,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    images = run_pipeline(
        pipe,
        num_views=NUM_VIEWS,
        text=prompt,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        negative_prompt=negative_prompt,
        device=device,
    )
    return images, seed


examples = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        ["An astronaut riding a horse", 42],
        ["A DSLR photo of a frog wearing a sweater", 42],
    ],
    "cagliostrolab/animagine-xl-3.1": [
        [
            "1girl, izayoi sakuya, touhou, solo, maid headdress, maid, apron, short sleeves, dress, closed mouth, white apron, serious face, upper body, masterpiece, best quality, very aesthetic, absurdres",
            0,
        ],
        [
            "1boy, male focus, ikari shinji, neon genesis evangelion, solo, serious face,(masterpiece), (best quality), (ultra-detailed), very aesthetic, illustration, disheveled hair, moist skin, intricate details",
            0,
        ],
        [
            "1girl, pink hair, pink shirts, smile, shy, masterpiece, anime",
            0,
        ],
    ],
    "Lykon/dreamshaper-xl-1-0": [
        ["the warrior Aragorn from Lord of the Rings, film grain, 8k hd", 0],
        [
            "Oil painting, masterpiece, regal, fancy.  A well-dressed dog named Puproy Doggerson III wearing reading glasses types an important letter on a typewriter and enjoys a cup of coffee with the newspaper.",
            42,
        ],
    ],
}

css = """
#col-container {
    margin: 0 auto;
    max-width: 600px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            f"""# MV-Adapter [Text-to-Multi-View]
Generate 768x768 multi-view images using {base_model} <br>
[[page](https://huanngzh.github.io/MV-Adapter-Page/)] [[repo](https://github.com/huanngzh/MV-Adapter)]
        """
        )

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Gallery(
            label="Result",
            show_label=False,
            columns=[3],
            rows=[2],
            object_fit="contain",
            height="auto",
        )

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.0,
                )

            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    placeholder="Enter your negative prompt",
                    value="watermark, ugly, deformed, noisy, blurry, low contrast",
                )

        if base_model in examples:
            gr.Examples(
                examples=examples[base_model],
                fn=infer,
                inputs=[prompt, seed],
                outputs=[result, seed],
                cache_examples=True,
            )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
            negative_prompt,
        ],
        outputs=[result, seed],
    )

demo.launch()
