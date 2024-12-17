---
library_name: hunyuan3d-1.0
license: other
license_name: tencent-hunyuan-community
license_link: https://huggingface.co/tencent/Hunyuan3D-1/blob/main/LICENSE.txt
language:
- en
- zh
tags:
- image-to-3d
- text-to-3d
pipeline_tag: text-to-3d
---
<!-- ## **Hunyuan3D-1.0** -->

<p align="center">
  <img src="./assets/logo.png"  height=200>
</p>

# Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation

<div align="center">
  <a href="https://github.com/tencent/Hunyuan3D-1"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://3d.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Homepage&message=Tencent Hunyuan3D&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/pdf/2411.02293"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/Tencent/Hunyuan3D-1"><img src="https://img.shields.io/static/v1?label=Checkpoints&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/Tencent/Hunyuan3D-1"><img src="https://img.shields.io/static/v1?label=Demo&message=HuggingFace&color=yellow"></a> &ensp;
</div>


## 🔥🔥🔥 News!!

* Nov 5, 2024: 💬 We support demo running  image_to_3d generation now. Please check the [script](#using-gradio) below.
* Nov 5, 2024: 💬 We support demo running  text_to_3d generation now. Please check the [script](#using-gradio) below.


## 📑 Open-source Plan

- [x] Inference 
- [x] Checkpoints
- [ ] Baking related
- [ ] Training
- [ ] ComfyUI
- [ ] Distillation Version
- [ ] TensorRT Version



## **Abstract**
<p align="center">
  <img src="./assets/teaser.png"  height=450>
</p>

While 3D generative models have greatly improved artists' workflows, the existing diffusion models for 3D generation suffer from slow generation and poor generalization. To address this issue, we propose a two-stage approach named Hunyuan3D-1.0 including a lite version and a standard version, that both support text- and image-conditioned generation.

In the first stage, we employ a multi-view diffusion model that efficiently generates multi-view RGB in approximately 4 seconds. These multi-view images capture rich details of the 3D asset from different viewpoints, relaxing the tasks from single-view to multi-view reconstruction. In the second stage, we introduce a feed-forward reconstruction model that rapidly and faithfully reconstructs the 3D asset given the generated multi-view images in approximately 7 seconds. The reconstruction network learns to handle noises and in-consistency introduced by the multi-view diffusion and leverages the available information from the condition image to efficiently recover the 3D structure.

Our framework involves the text-to-image model, i.e., Hunyuan-DiT, making it a unified framework to support both text- and image-conditioned 3D generation. Our standard version has 3x more parameters than our lite and other existing model. Our Hunyuan3D-1.0 achieves an impressive balance between speed and quality, significantly reducing generation time while maintaining the quality and diversity of the produced assets.


## 🎉 **Hunyuan3D-1 Architecture**

<p align="center">
  <img src="./assets/overview_3.png"  height=400>
</p>


## 📈 Comparisons

We have evaluated Hunyuan3D-1.0 with other open-source 3d-generation methods, our Hunyuan3D-1.0 received the highest user preference across 5 metrics. Details in the picture on the lower left.

The lite model takes around 10 seconds to produce a 3D mesh from a single image on an NVIDIA A100 GPU, while the standard model takes roughly 25 seconds. The plot laid out in the lower right demonstrates that Hunyuan3D-1.0 achieves an optimal balance between quality and efficiency.

<p align="center">
  <img src="./assets/radar.png"  height=300>
  <img src="./assets/runtime.png"  height=300>
</p>

## Get Started

#### Begin by cloning the repository:

```shell
git clone https://github.com/tencent/Hunyuan3D-1
cd Hunyuan3D-1
```

#### Installation Guide for Linux

We provide an env_install.sh script file for setting up environment. 

```
# step 1, create conda env
conda create -n hunyuan3d-1 python=3.9 or 3.10 or 3.11 or 3.12
conda activate hunyuan3d-1

# step 2. install torch realated package
which pip # check pip corresponds to python

# modify the cuda version according to your machine (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# step 3. install other packages
bash env_install.sh
```
<details>
<summary>💡Other tips for envrionment installation</summary>
    
Optionally, you can install xformers or flash_attn to acclerate computation:

```
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```
```
pip install flash_attn
```

Most environment errors are caused by a mismatch between machine and packages. You can try manually specifying the version, as shown in the following successful cases:
```
# python3.9
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 
```

when install pytorch3d, the gcc version is preferably greater than 9, and the gpu driver should not be too old.

</details>

#### Download Pretrained Models

The models are available at [https://huggingface.co/tencent/Hunyuan3D-1](https://huggingface.co/tencent/Hunyuan3D-1):

+ `Hunyuan3D-1/lite`, lite model for multi-view generation.
+ `Hunyuan3D-1/std`, standard model for multi-view generation.
+ `Hunyuan3D-1/svrm`, sparse-view reconstruction model.


To download the model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python3 -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
mkdir weights
huggingface-cli download tencent/Hunyuan3D-1 --local-dir ./weights

mkdir weights/hunyuanDiT
huggingface-cli download Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled --local-dir ./weights/hunyuanDiT
```

#### Inference 
For text to 3d generation, we supports bilingual Chinese and English, you can use the following command to inference.
```python
python3 main.py \
    --text_prompt "a lovely rabbit" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000 \
    --do_texture_mapping \
    --do_render
```

For image to 3d generation, you can use the following command to inference.
```python
python3 main.py \
    --image_prompt "/path/to/your/image" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000 \
    --do_texture_mapping \
    --do_render
```
We list some more useful configurations for easy usage:

|    Argument        |  Default  |                     Description                     |
|:------------------:|:---------:|:---------------------------------------------------:|
|`--text_prompt`  |   None    |The text prompt for 3D generation         |
|`--image_prompt` |   None    |The image prompt for 3D generation         |
|`--t2i_seed`     |    0      |The random seed for generating images        |
|`--t2i_steps`    |    25     |The number of steps for sampling of text to image  |
|`--gen_seed`     |    0      |The random seed for generating 3d generation        |
|`--gen_steps`    |    50     |The number of steps for sampling of 3d generation  |
|`--max_faces_numm` | 90000  |The limit number of faces of 3d mesh |
|`--save_memory`   | False   |module will move to cpu automatically|
|`--do_texture_mapping` |   False    |Change vertex shadding to texture shading  |
|`--do_render`  |   False   |render gif   |


We have also prepared scripts with different configurations for reference
- Inference Std-pipeline requires 30GB VRAM (24G VRAM with --save_memory).
- Inference Lite-pipeline requires 22GB VRAM (18G VRAM with --save_memory).
- Note: --save_memory will increase inference time

```bash
bash scripts/text_to_3d_std.sh 
bash scripts/text_to_3d_lite.sh 
bash scripts/image_to_3d_std.sh 
bash scripts/image_to_3d_lite.sh 
```

If your gpu memory is 16G, you can try to run modules in pipeline seperately:
```bash
bash scripts/text_to_3d_std_separately.sh 'a lovely rabbit' ./outputs/test # >= 16G
bash scripts/text_to_3d_lite_separately.sh 'a lovely rabbit' ./outputs/test # >= 14G
bash scripts/image_to_3d_std_separately.sh ./demos/example_000.png ./outputs/test  # >= 16G
bash scripts/image_to_3d_lite_separately.sh ./demos/example_000.png ./outputs/test # >= 10G
```

#### Using Gradio

We have prepared two versions of multi-view generation, std and lite.

```shell
# std 
python3 app.py
python3 app.py --save_memory

# lite
python3 app.py --use_lite
python3 app.py --use_lite --save_memory
```

Then the demo can be accessed through http://0.0.0.0:8080. It should be noted that the 0.0.0.0 here needs to be X.X.X.X with your server IP.

## Camera Parameters

Output views are a fixed set of camera poses:

+ Azimuth (relative to input view): `+0, +60, +120, +180, +240, +300`.


## Citation

If you found this repository helpful, please cite our report:
```bibtex
@misc{yang2024tencent,
    title={Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Xianghui Yang and Huiwen Shi and Bowen Zhang and Fan Yang and Jiacheng Wang and Hongxu Zhao and Xinhai Liu and Xinzhou Wang and Qingxiang Lin and Jiaao Yu and Lifu Wang and Zhuo Chen and Sicong Liu and Yuhong Liu and Yong Yang and Di Wang and Jie Jiang and Chunchao Guo},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```