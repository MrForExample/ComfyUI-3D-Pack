---
license: mit
pipeline_tag: image-to-3d
tags:
- triposg
- 3d-generation
- rectified-flow
---
# TripoSG-scribble - Fast 3D Shape Prototyping with Scribble and Prompt

TripoSG-scribble converts a scribble image and a text prompt to a 3D shape. TripoSG-scribble is a variant of TripoSG. TripoSG is a state-of-the-art image-to-3D generation foundation model that leverages large-scale rectified flow transformers to produce high-fidelity 3D shapes from single images.

## Model Description

### Model Architecture

TripoSG utilizes a novel architecture combining:
- Rectified Flow (RF) based Transformer for stable, linear trajectory modeling
- Advanced VAE with SDF-based representation and hybrid geometric supervision
- Cross-attention mechanism for image feature condition
- 1.5B parameters operating on 2048 latent tokens

For inference efficiency, TripoSG-scribble is different from TripoSG in:
- TripoSG-scribble is a CFG-distilled model and should be used with CFG=0
- TripoSG-scribble is trained with 512 latent tokens

## Intended Uses

This model is designed for:
- Converting scribble image and text prompt to high-quality 3D meshes
- Creative and design applications
- Gaming and VFX asset creation
- Prototyping and visualization

## Requirements

- CUDA-capable GPU (>8GB VRAM)

## Usage

For detailed usage instructions, please visit our [GitHub repository](https://github.com/VAST-AI-Research/TripoSG).

## About

TripoSG-scribble is developed by [Tripo](https://www.tripo3d.ai), [VAST AI Research](https://github.com/orgs/VAST-AI-Research), pushing the boundaries of 3D Generative AI.
For more information:
- [GitHub Repository](https://github.com/VAST-AI-Research/TripoSG)
- [Paper](https://arxiv.org/abs/2502.06608)
- [Gradio Demo](https://huggingface.co/spaces/VAST-AI/TripoSG-scribble)