---
license: openrail++
tags:
- stable-diffusion
- text-to-image
---

# SD v2.1-base with Zero Terminal SNR (LAION Aesthetic 6+)

This model is used in [Diffusion Model with Perceptual Loss](https://arxiv.org/abs/2401.00110) paper as the MSE baseline.

This model is trained using zero terminal SNR schedule following [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891) paper on LAION aesthetic 6+ data.

This model is finetuned from [stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

This model is meant for research demonstration, not for production use.

## Usage

```python
from diffusers import StableDiffusionPipeline
prompt = "A young girl smiling"
pipe = StableDiffusionPipeline.from_pretrained("ByteDance/sd2.1-base-zsnr-laionaes6").to("cuda")
pipe(prompt, guidance_scale=7.5, guidance_rescale=0.7).images[0].save("out.jpg")
```

## Related Models

* [bytedance/sd2.1-base-zsnr-laionaes5](https://huggingface.co/ByteDance/sd2.1-base-zsnr-laionaes5)
* [bytedance/sd2.1-base-zsnr-laionaes6](https://huggingface.co/ByteDance/sd2.1-base-zsnr-laionaes6)
* [bytedance/sd2.1-base-zsnr-laionaes6-perceptual](https://huggingface.co/ByteDance/sd2.1-base-zsnr-laionaes6-perceptual)


## Cite as
```
@misc{lin2024diffusion,
      title={Diffusion Model with Perceptual Loss}, 
      author={Shanchuan Lin and Xiao Yang},
      year={2024},
      eprint={2401.00110},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lin2023common,
      title={Common Diffusion Noise Schedules and Sample Steps are Flawed}, 
      author={Shanchuan Lin and Bingchen Liu and Jiashi Li and Xiao Yang},
      year={2023},
      eprint={2305.08891},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```