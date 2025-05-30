import random

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch.optim import lr_scheduler

from ..utils.core import debug, find, info, warn
from ..utils.typing import *

"""Diffusers Model Utils"""


def vae_encode(
    vae: AutoencoderKL,
    pixel_values: Float[Tensor, "B 3 H W"],
    sample: bool = True,
    apply_scale: bool = True,
):
    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.sample() if sample else latent_dist.mode()
    if apply_scale:
        latents = latents * vae.config.scaling_factor
    return latents


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True
):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


CLIP_INPUT_MEAN = torch.as_tensor(
    [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
)[None, :, None, None]
CLIP_INPUT_STD = torch.as_tensor(
    [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
)[None, :, None, None]


def normalize_image_for_clip(image: Float[Tensor, "B C H W"]):
    return (image - CLIP_INPUT_MEAN.to(image)) / CLIP_INPUT_STD.to(image)


"""Training"""


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adam8bit", "AdamW8bit"]:
        import bitsandbytes as bnb

        optim = bnb.optim.Adam8bit(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
