---
license: mit
pipeline_tag: image-to-3d
tags:
- triposg
- 3d-generation
- rectified-flow
---
# TripoSG - High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models

TripoSG is a state-of-the-art image-to-3D generation foundation model that leverages large-scale rectified flow transformers to produce high-fidelity 3D shapes from single images.

## Model Description

### Model Architecture

TripoSG utilizes a novel architecture combining:
- Rectified Flow (RF) based Transformer for stable, linear trajectory modeling
- Advanced VAE with SDF-based representation and hybrid geometric supervision
- Cross-attention mechanism for image feature condition
- 1.5B parameters operating on 2048 latent tokens

## Intended Uses

This model is designed for:
- Converting single images to high-quality 3D meshes
- Creative and design applications
- Gaming and VFX asset creation
- Prototyping and visualization

## Requirements

- CUDA-capable GPU (>8GB VRAM)

## Usage

For detailed usage instructions, please visit our [GitHub repository](https://github.com/VAST-AI-Research/TripoSG).

## About

TripoSG is developed by [Tripo](https://www.tripo3d.ai), [VAST AI Research](https://github.com/orgs/VAST-AI-Research), pushing the boundaries of 3D Generative AI.
For more information:
- [GitHub Repository](https://github.com/VAST-AI-Research/TripoSG)
- [Paper](https://arxiv.org/abs/2502.06608)
- [Gradio Demo](https://huggingface.co/spaces/VAST-AI/TripoSG)