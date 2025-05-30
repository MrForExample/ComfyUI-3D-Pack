# MV-Adapter: Multi-view Consistent Image Generation Made Easy🚀

## 🏠 <a href="https://huanngzh.github.io/MV-Adapter-Page/" target="_blank">Project Page</a> | <a href="https://arxiv.org/abs/2412.03632" target="_blank">Paper</a> | <a href="https://huggingface.co/huanngzh/mv-adapter" target="_blank">Model Weights</a> | [Demo](https://huggingface.co/collections/huanngzh/mv-adapter-spaces-677e497578747fd734a1b999) | <a href="https://github.com/huanngzh/ComfyUI-MVAdapter" target="_blank">ComfyUI</a>

![teaser](assets/doc/teaser.jpg)

MV-Adapter is a **versatile plug-and-play adapter** that adapt T2I models and their derivatives to multi-view generators.

Highlight Features: Generate multi-view images

- with 768 Resolution using SDXL
- using personalized models (e.g. <a href="https://civitai.com/models/112902/dreamshaper-xl" target="_blank">DreamShaper</a>), distilled models (e.g. <a href="https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models" target="_blank">LCM</a>), or extensions (e.g. <a href="https://github.com/lllyasviel/ControlNet" target="_blank">ControlNet</a>)
- from text or image condition
- can be guided by geometry for texture generation

## 🔥 Updates

* [2025-05-15] Release full pipeline for text-to-texture and image-to-texture generation. [See [guidelines](#usage-texture-generation)]
* [2025-04-23] Release dataset ([Objaverse-Ortho10View](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View) and [Objaverse-Rand6View](https://huggingface.co/datasets/huanngzh/Objaverse-Rand6View)) and training code. [See [guidelines](#️-training)]
* [2025-03-31] Release text/image-conditioned 3D texture generation demos on [Text2Texture](https://huggingface.co/spaces/VAST-AI/MV-Adapter-Text2Texture) and [Image2Texture](https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture). Feel free to try them!
* [2025-03-17] Release model weights for partial-image conditioned geometry-to-multiview generation, which can be used to generate textured 3D scenes combined with [MIDI](https://github.com/VAST-AI-Research/MIDI-3D). [See [guidelines](#partial-image--geometry-to-multiview)]
* [2025-03-07] Release model weights for geometry-guided multi-view generation. [See [guidelines](#text-geometry-to-multiview-generation)]
* [2024-12-27] Release model weights, gradio demo, inference scripts and comfyui of text-/image- to multi-view generation models.

## TOC

- [MV-Adapter: Multi-view Consistent Image Generation Made Easy🚀](#mv-adapter-multi-view-consistent-image-generation-made-easy)
  - [🏠 Project Page | Paper | Model Weights | Demo | ComfyUI](#-project-page--paper--model-weights--demo--comfyui)
  - [🔥 Updates](#-updates)
  - [TOC](#toc)
  - [Model Zoo \& Demos](#model-zoo--demos)
  - [Installation](#installation)
  - [Notes](#notes)
    - [System Requirements](#system-requirements)
  - [Usage: Multiview Generation](#usage-multiview-generation)
    - [Launch Demo](#launch-demo)
      - [Text to Multiview Generation](#text-to-multiview-generation)
      - [Image to Multiview Generation](#image-to-multiview-generation)
    - [Inference Scripts](#inference-scripts)
      - [Text to Multiview Generation](#text-to-multiview-generation-1)
      - [Image to Multiview Generation](#image-to-multiview-generation-1)
      - [Text-Geometry to Multiview Generation](#text-geometry-to-multiview-generation)
      - [Image-Geometry to Multiview Generation](#image-geometry-to-multiview-generation)
      - [Partial Image + Geometry to Multiview](#partial-image--geometry-to-multiview)
    - [ComfyUI](#comfyui)
  - [Usage: Texture Generation](#usage-texture-generation)
  - [Citation](#citation)

## Model Zoo & Demos

No need to download manually. Running the scripts will download model weights automatically.

Notes: Running MV-Adapter for SDXL may need higher GPU memory and more time, but produce higher-quality and higher-resolution results. On the other hand, running its SD2.1 variant needs lower computing cost, but shows a bit lower performance.

|            Model            | Base Model |                                                                                                                                 HF Weights                                                                                                                                  |                                                                   Demo Link                                                                   |
| :-------------------------: | :--------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------: |
|      Text-to-Multiview      |   SD2.1    |                                                                         [mvadapter_t2mv_sd21.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_t2mv_sd21.safetensors)                                                                          |                                                                                                                                               |
|      Text-to-Multiview      |    SDXL    |                                                                         [mvadapter_t2mv_sdxl.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_t2mv_sdxl.safetensors)                                                                          | [General](https://huggingface.co/spaces/VAST-AI/MV-Adapter-T2MV-SDXL) / [Anime](https://huggingface.co/spaces/huanngzh/MV-Adapter-T2MV-Anime) |
|     Image-to-Multiview      |   SD2.1    |                                                                         [mvadapter_i2mv_sd21.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_i2mv_sd21.safetensors)                                                                          |                                                                                                                                               |
|     Image-to-Multiview      |    SDXL    |                                                                         [mvadapter_i2mv_sdxl.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_i2mv_sdxl.safetensors)                                                                          |                                      [Demo](https://huggingface.co/spaces/VAST-AI/MV-Adapter-I2MV-SDXL)                                       |
| Text-Geometry-to-Multiview  |    SDXL    |                                                                        [mvadapter_tg2mv_sdxl.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_tg2mv_sdxl.safetensors)                                                                         |                                     [Demo](https://huggingface.co/spaces/VAST-AI/MV-Adapter-Text2Texture)                                     |
| Image-Geometry-to-Multiview |    SDXL    | [mvadapter_ig2mv_sdxl.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_ig2mv_sdxl.safetensors) / [mvadapter_ig2mv_partial_sdxl.safetensors](https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_ig2mv_partial_sdxl.safetensors) |                                     [Demo](https://huggingface.co/spaces/VAST-AI/MV-Adapter-Img2Texture)                                      |
|  Image-to-Arbitrary-Views   |    SDXL    |                                                                                                                                                                                                                                                                             |                                                                                                                                               |

## Installation

Clone the repo first:

```Bash
git clone https://github.com/huanngzh/MV-Adapter.git
cd MV-Adapter
```

(Optional) Create a fresh conda env:

```Bash
conda create -n mvadapter python=3.10
conda activate mvadapter
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt
```

For texture generation, you need to install `CV-CUDA` according to [CVCUDA/CV-CUDA](https://github.com/CVCUDA/CV-CUDA?tab=readme-ov-file#installation).

## Notes

### System Requirements

In the model zoo of MV-Adapter, running image-to-multiview generation has the highest system requirements, which requires about 14G GPU memory.

## Usage: Multiview Generation

### Launch Demo

#### Text to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "stabilityai/stable-diffusion-xl-base-1.0"
```

![demo_t2mv_1](assets/doc/demo_t2mv_1.png)

![demo_t2mv_2](assets/doc/demo_t2mv_2.png)

> Reminder: When switching the demo to another base model, delete the `gradio_cached_examples` directory, otherwise it will affect the examples results of the next demo.

**With anime-themed <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1" target="_blank">Animagine XL 3.1</a>:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "cagliostrolab/animagine-xl-3.1"
```

![demo_t2mv_anime_1](assets/doc/demo_t2mv_anime_1.png)

![demo_t2mv_anime_2](assets/doc/demo_t2mv_anime_2.png)

**With general <a href="https://huggingface.co/Lykon/dreamshaper-xl-1-0" target="_blank">Dreamshaper</a>:**

```Bash
python -m scripts.gradio_demo_t2mv --base_model "Lykon/dreamshaper-xl-1-0" --scheduler ddpm
```

![demo_t2mv_dreamshaper_1](assets/doc/demo_t2mv_dreamshaper_1.png)

You can also specify a new diffusers-format text-to-image diffusion model using `--base_model`. Note that it should be the model name in huggingface, such as `stabilityai/stable-diffusion-xl-base-1.0`, or a local path refer to a text-to-image pipeline directory. Note that if you specify `latent-consistency/lcm-sdxl` to use latent consistency models, please add `--scheduler lcm` to the command.

#### Image to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.gradio_demo_i2mv
```

![demo_i2mv_1](assets/doc/demo_i2mv_1.png)

![demo_i2mv_2](assets/doc/demo_i2mv_2.png)

### Inference Scripts

We recommend that experienced users check the files in the scripts directory to adjust the parameters appropriately to try the best "card drawing" results.

#### Text to Multiview Generation

Note that you can specify a diffusers-format text-to-image diffusion model as the base model using `--base_model xxx`. It should be the model name in huggingface, such as `stabilityai/stable-diffusion-xl-base-1.0`, or a local path refer to a text-to-image pipeline directory.

**With SDXL:**

```Bash
python -m scripts.inference_t2mv_sdxl --text "an astronaut riding a horse" \
--seed 42 \
--output output.png
```

**With personalized models:**

anime-themed <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1" target="_blank">Animagine XL 3.1</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "cagliostrolab/animagine-xl-3.1" \
--text "1girl, izayoi sakuya, touhou, solo, maid headdress, maid, apron, short sleeves, dress, closed mouth, white apron, serious face, upper body, masterpiece, best quality, very aesthetic, absurdres" \
--seed 0 \
--output output.png
```

general <a href="https://huggingface.co/Lykon/dreamshaper-xl-1-0" target="_blank">Dreamshaper</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "Lykon/dreamshaper-xl-1-0" \
--scheduler ddpm \
--text "the warrior Aragorn from Lord of the Rings, film grain, 8k hd" \
--seed 0 \
--output output.png
```

realistic <a href="https://huggingface.co/stablediffusionapi/real-dream-sdxl" target="_blank">real-dream-sdxl</a>

```Bash
python -m scripts.inference_t2mv_sdxl --base_model "stablediffusionapi/real-dream-sdxl" \
--scheduler ddpm \
--text "macro shot, parrot, colorful, dark shot, film grain, extremely detailed" \
--seed 42 \
--output output.png
```

**With <a href="https://huggingface.co/latent-consistency/lcm-sdxl" target="_blank">LCM</a>:**

```Bash
python -m scripts.inference_t2mv_sdxl --unet_model "latent-consistency/lcm-sdxl" \
--scheduler lcm \
--text "Samurai koala bear" \
--num_inference_steps 8 \
--seed 42 \
--output output.png
```

**With LoRA:**

stylized lora <a href="https://huggingface.co/goofyai/3d_render_style_xl" target="_blank">3d_render_style_xl</a>

```Bash
python -m scripts.inference_t2mv_sdxl --lora_model "goofyai/3d_render_style_xl/3d_render_style_xl.safetensors" \
--text "3d style, a fox with flowers around it" \
--seed 20 \
--lora_scale 1.0 \
--output output.png
```

**With ControlNet:**

Scribble to Multiview with <a href="https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0" target="_blank">controlnet-scribble-sdxl-1.0</a>

```Bash
python -m scripts.inference_scribble2mv_sdxl --text "A 3D model of Finn the Human from the animated television series Adventure Time. He is wearing his iconic blue shirt and green backpack and has a neutral expression on his face. He is standing in a relaxed pose with his left foot slightly forward and his right foot back. His arms are at his sides and his head is turned slightly to the right. The model is made up of simple shapes and has a stylized, cartoon-like appearance. It is textured to resemble the character's appearance in the show." \
--seed 0 \
--output output.png \
--guidance_scale 5.0 \
--controlnet_images "assets/demo/scribble2mv/color_0000.webp" "assets/demo/scribble2mv/color_0001.webp" "assets/demo/scribble2mv/color_0002.webp" "assets/demo/scribble2mv/color_0003.webp" "assets/demo/scribble2mv/color_0004.webp" "assets/demo/scribble2mv/color_0005.webp" \
--controlnet_conditioning_scale 0.7
```

**With SD2.1:**

> SD2.1 has lower demand for computing resources and higher inference speed, but a bit lower performance than SDXL.
> In our tests, ddpm scheduler works better than other schedulers here.

```Bash
python -m scripts.inference_t2mv_sd --text "a corgi puppy" \
--seed 42 --scheduler ddpm \
--output output.png
```

#### Image to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.inference_i2mv_sdxl \
--image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
--text "A decorative figurine of a young anime-style girl" \
--seed 21 --output output.png --remove_bg
```

**With LCM:**

```Bash
python -m scripts.inference_i2mv_sdxl \
--unet_model "latent-consistency/lcm-sdxl" \
--scheduler lcm \
--image assets/demo/i2mv/A_juvenile_emperor_penguin_chick.png \
--text "A juvenile emperor penguin chick" \
--num_inference_steps 8 \
--seed 0 --output output.png --remove_bg
```

**With SD2.1:** (lower demand for computing resources and higher inference speed)

> In our tests, ddpm scheduler works better than other schedulers here.

```Bash
python -m scripts.inference_i2mv_sd \
--image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
--text "A decorative figurine of a young anime-style girl" \
--output output.png --remove_bg --scheduler ddpm
```

#### Text-Geometry to Multiview Generation

**Importantly**, when using geometry-condition generation, please make sure that the orientation of the mesh you provide is consistent with the following example. Otherwise, you need to adjust the angles in the scripts when rendering the view.

**With SDXL:**

```Bash
python -m scripts.inference_tg2mv_sdxl \
--mesh assets/demo/tg2mv/ac9d4e4f44f34775ad46878ba8fbfd86.glb \
--text "Mater, a rusty and beat-up tow truck from the 2006 Disney/Pixar animated film 'Cars', with a rusty brown exterior, big blue eyes."
```

![tg2mv_example_out](assets/demo/tg2mv/ac9d4e4f44f34775ad46878ba8fbfd86_mv.png)

```Bash
python -m scripts.inference_tg2mv_sdxl \
--mesh assets/demo/tg2mv/b5f0f0f33e3644d1ba73576ceb486d42.glb \
--text "Optimus Prime, a character from Transformers, with blue, red and gray colors, and has a flame-like pattern on the body"
```

#### Image-Geometry to Multiview Generation

**With SDXL:**

```Bash
python -m scripts.inference_ig2mv_sdxl \
--image assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.jpeg \
--mesh assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb \
--output output.png --remove_bg
```

![example_ig2mv_out](assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c_mv.png)

#### Partial Image + Geometry to Multiview

**With SDXL:**

```Bash
python -m scripts.inference_ig2mv_partial_sdxl \
--image assets/demo/ig2mv/cartoon_style_table.png \
--mesh assets/demo/ig2mv/cartoon_style_table.glb \
--output output.png
```

Example input:
<img src="assets/demo/ig2mv/cartoon_style_table.png" alt="partial input" style="width: 20%">

Example output:
![example_partial_ig2mv](assets/demo/ig2mv/cartoon_style_table_mv.png)

The above command will save a `*_transform.json` file in the output dir, which contains transformation information like this:
```json
{
    "offset": [
        0.7446051140100826,
        -0.3421213991056582,
        0.1360104325533671
    ],
    "scale": 1.0086087120792668
}
```

You can use it to transform your mesh into the canonical space, map the generated multi-view images onto the mesh, and then re-transform the mesh back to the original spatial position.

### ComfyUI

Please check <a href="https://github.com/huanngzh/ComfyUI-MVAdapter" target="_blank">ComfyUI-MVAdapter Repo</a> for details.

**Text to Multiview Generation**

![comfyui_t2mv](assets/doc/comfyui_t2mv.png)

**Text to Multiview Generation with LoRA**

![comfyui_t2mv_lora](assets/doc/comfyui_t2mv_lora.png)

**Image to Multiview Generation**

![comfyui_i2mv](assets/doc/comfyui_i2mv.png)


## Usage: Texture Generation

**Prepare Models**

Download pre-trained [RealESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) for upscaling images and [LaMa](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) for view in-painting.

```Bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O ./checkpoints/RealESRGAN_x2plus.pth
wget https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O ./checkpoints/big-lama.pt
```

**Usage**

> All in one script

Text-conditioned texture generation:

```Bash
python -m scripts.texture_t2tex \
--mesh assets/demo/tg2mv/ac9d4e4f44f34775ad46878ba8fbfd86.glb \
--text "Mater, a rusty and beat-up tow truck from the 2006 Disney/Pixar animated film 'Cars', with a rusty brown exterior, big blue eyes." \
--save_dir outputs --save_name t2tex_sample
```

Image-conditioned texture generation:

```Bash
python -m scripts.texture_i2tex \
--image assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.jpeg \
--mesh assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb \
--save_dir outputs --save_name i2tex_sample \
--remove_bg
```

It will save the textured model into `<save_dir>/<save_name>_shaded.glb`.

## 📊 Dataset

Our training dataset, rendered from [Objaverse](https://huggingface.co/datasets/allenai/objaverse), can be downloaded from [Objaverse-Ortho10View](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View) and [Objaverse-Rand6View](https://huggingface.co/datasets/huanngzh/Objaverse-Rand6View).

* [Objaverse-Ortho10View](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View) contains 10 orthographic views of 1024x1024 resolution, and is used as ground truth.
* [Objaverse-Rand6View](https://huggingface.co/datasets/huanngzh/Objaverse-Rand6View) contains 6 randomly distributed views, and is used as reference image conditions.

Please refer to their dataset cards to extract the data files, and organize them into the following structures:

```Bash
data
├── texture_ortho10view_easylight_objaverse # Objaverse-Ortho10View
│   ├── 00
│   │   ├── 00a4d2b0c4c240289ed456e87d8b9e02
│   ...
├── texture_rand_easylight_objaverse # Objaverse-Rand6View
│   ├── 00
│   │   ├── 00a4d2b0c4c240289ed456e87d8b9e02
│   ...
├── objaverse_list_6w.json # objaverse ids
└── objaverse_short_captions.json # id to captions
```

## 🏋️ Training

The key training code can be found in `mvadapter/systems`:

* `MVAdapterTextSDXLSystem` in `mvadapter_text_sdxl.py` is used for text or text+geometry conditioned multi-view generation.
* `MVAdapterImageSDXLSystem` in `mvadapter_image_sdxl.py` is used for image or image+geometry conditioned multi-view generation.

The specific training commands are as follows:

For text to 6 view generation:

```Bash
python launch.py --config configs/view-guidance/mvadapter_t2mv_sdxl.yaml --train --gpu 0,1,2,3,4,5,6,7
```

For single image to 6 view generation:

```Bash
python launch.py --config configs/view-guidance/mvadapter_i2mv_sdxl.yaml --train --gpu 0,1,2,3,4,5,6,7
```

For single image to 2/3/4/6 view generation:

```Bash
python launch.py --config configs/view-guidance/mvadapter_i2mv_sdxl_aug_quantity.yaml --train --gpu 0,1,2,3,4,5,6,7
```

For text + geometry to 6 view generation:

```Bash
python launch.py --config configs/geometry-guidance/mvadapter_tg2mv_sdxl.yaml --train --gpu 0,1,2,3,4,5,6,7
```

For single image + geometry to 6 view generation:

```Bash
python launch.py --config configs/geometry-guidance/mvadapter_ig2mv_sdxl.yaml --train --gpu 0,1,2,3,4,5,6,7
```

For single partial image + geometry to 6 view generation (used for texture generation conditioned on occluded image, for example, used in [MIDI-3D](https://github.com/VAST-AI-Research/MIDI-3D)):

```Bash
python launch.py --config configs/geometry-guidance/mvadapter_ig2mv_partialimg_sdxl.yaml --train --gpu 0,1,2,3,4,5,6,7
```

## Citation

```
@article{huang2024mvadapter,
  title={MV-Adapter: Multi-view Consistent Image Generation Made Easy},
  author={Huang, Zehuan and Guo, Yuanchen and Wang, Haoran and Yi, Ran and Ma, Lizhuang and Cao, Yan-Pei and Sheng, Lu},
  journal={arXiv preprint arXiv:2412.03632},
  year={2024}
}
```
