import argparse
import random

import gradio as gr
import numpy as np

# import spaces
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .inference_i2mv_sdxl import prepare_pipeline, remove_bg, run_pipeline

# Device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_VIEWS = 6
HEIGHT = 768
WIDTH = 768
MAX_SEED = np.iinfo(np.int32).max

pipe = prepare_pipeline(
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_model=None,
    lora_model=None,
    adapter_path="huanngzh/mv-adapter",
    scheduler=None,
    num_views=NUM_VIEWS,
    device=device,
    dtype=dtype,
)

# remove bg
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(device)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# @spaces.GPU()
def infer(
    prompt,
    image,
    do_rembg=True,
    seed=42,
    randomize_seed=False,
    guidance_scale=3.0,
    num_inference_steps=50,
    reference_conditioning_scale=1.0,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    progress=gr.Progress(track_tqdm=True),
):
    if do_rembg:
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, device)
    else:
        remove_bg_fn = None
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    images, preprocessed_image = run_pipeline(
        pipe,
        num_views=NUM_VIEWS,
        text=prompt,
        image=image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_bg_fn=remove_bg_fn,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        device=device,
    )
    return images, preprocessed_image, seed


examples = [
    [
        "A decorative figurine of a young anime-style girl",
        "assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png",
        True,
        21,
    ],
    [
        "A juvenile emperor penguin chick",
        "assets/demo/i2mv/A_juvenile_emperor_penguin_chick.png",
        True,
        0,
    ],
    [
        "A striped tabby cat with white fur sitting upright",
        "assets/demo/i2mv/A_striped_tabby_cat_with_white_fur_sitting_upright.png",
        True,
        0,
    ],
]


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            f"""# MV-Adapter [Image-to-Multi-View]
Generate 768x768 multi-view images from a single image using SDXL <br>
[[page](https://huanngzh.github.io/MV-Adapter-Page/)] [[repo](https://github.com/huanngzh/MV-Adapter)]
        """
        )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    sources=["upload", "webcam", "clipboard"],
                    type="pil",
                )
                preprocessed_image = gr.Image(label="Preprocessed Image", type="pil")

            prompt = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt", value="high quality"
            )
            do_rembg = gr.Checkbox(label="Remove background", value=True)
            run_button = gr.Button("Run")

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
                        value=3.0,
                    )

                with gr.Row():
                    reference_conditioning_scale = gr.Slider(
                        label="Image conditioning scale",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                    )

                with gr.Row():
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="Enter your negative prompt",
                        value="watermark, ugly, deformed, noisy, blurry, low contrast",
                    )

        with gr.Column():
            result = gr.Gallery(
                label="Result",
                show_label=False,
                columns=[3],
                rows=[2],
                object_fit="contain",
                height="auto",
            )

    with gr.Row():
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt, input_image, do_rembg, seed],
            outputs=[result, preprocessed_image, seed],
            cache_examples=True,
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            input_image,
            do_rembg,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
            reference_conditioning_scale,
            negative_prompt,
        ],
        outputs=[result, preprocessed_image, seed],
    )

demo.launch()
