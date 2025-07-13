"""
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

import importlib
<<<<<<< HEAD

=======
>>>>>>> 20c359f (WIP: temporary updates before switching branch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms
from transformers import Dinov2Model

from .configs.schema import ModelConfig
from .flow_matching import FlowMatchingScheduler
from .modules.dit import DiT
from ..vae.model import Model as VAE
from ..vae.utils import sync_timer
<<<<<<< HEAD

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
=======
from comfy.utils import ProgressBar


>>>>>>> 20c359f (WIP: temporary updates before switching branch)
class Model(nn.Module):
    def __init__(self, config: ModelConfig,device,dino_path,cpu_offload=False) -> None:
        super().__init__()
        self.config = config
        self.precision = torch.bfloat16
        self.cpu_offload = cpu_offload

        # image condition model (dinov2)
        # if self.config.dino_model == "dinov2_vitg14":
        #     #self.dino = Dinov2Model.from_pretrained("facebook/dinov2-giant")
        # elif self.config.dino_model == "dinov2_vitl14_reg":
        #     #self.dino = Dinov2Model.from_pretrained("facebook/dinov2-with-registers-large")
        self.device = device
        if dino_path:
            self.dino = Dinov2Model.from_pretrained(dino_path)
        else:
            raise ValueError(f"DINOv2 model {self.config.dino_model} not supported")

        # hack to match our implementation
        self.dino.layernorm = torch.nn.Identity()

<<<<<<< HEAD
        self.dino.eval().to(dtype=self.precision)
=======
        # self.dino.eval().to(dtype=self.precision)
        self.dino.eval().to(dtype=self.precision)
        if self.cpu_offload:
            self.dino.to("cpu")  # Force to CPU
        else:
            self.dino.to(device)  # Only use GPU if enough VRAM

>>>>>>> 20c359f (WIP: temporary updates before switching branch)
        self.dino.requires_grad_(False)

        cond_dim = 1024 if self.config.dino_model == "dinov2_vitl14_reg" else 1536
        assert cond_dim == config.hidden_dim, "DINOv2 dim must match backbone dim"

        self.preprocess_cond_image = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # vae encoder
        vae_config = importlib.import_module(config.vae_conf).make_config()
<<<<<<< HEAD


=======
>>>>>>> 20c359f (WIP: temporary updates before switching branch)
        self.vae = VAE(vae_config).eval().to(dtype=self.precision)
        self.vae.requires_grad_(False)

        # load vae
        if self.config.preload_vae:
            try:
                vae_ckpt = torch.load(self.config.vae_ckpt_path, weights_only=True)  # local path
                if "model" in vae_ckpt:
                    vae_ckpt = vae_ckpt["model"]
                self.vae.load_state_dict(vae_ckpt, strict=True)
                del vae_ckpt
                print(f"Loaded VAE from {self.config.vae_ckpt_path}")
            except Exception as e:
                print(
                    f"Failed to load VAE from {self.config.vae_ckpt_path}: {e}, make sure you resumed from a valid checkpoint!"
                )

        # load info from vae config
        if config.latent_size is None:
            config.latent_size = self.vae.config.latent_size
        if config.latent_dim is None:
            config.latent_dim = self.vae.config.latent_dim

        # dit
        self.dit = DiT(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            latent_size=config.latent_size,
            latent_dim=config.latent_dim,
            qknorm=config.qknorm,
            qknorm_type=config.qknorm_type,
            use_pos_embed=config.use_pos_embed,
            use_parts=config.use_parts,
            part_embed_mode=config.part_embed_mode,
        )
        if cpu_offload:
<<<<<<< HEAD
            print("🔁 Using CPU offload mode to save VRAM.")
=======
            print("Using CPU offload mode to save VRAM.")
>>>>>>> 20c359f (WIP: temporary updates before switching branch)
            self.vae.to("cpu")
            self.dit.to("cpu")
        else:
            self.vae.to(device)
            self.dit.to(device)

        # num_part condition
        if self.config.use_num_parts_cond:
            assert self.config.use_parts, "use_num_parts_cond requires use_parts"
            self.num_part_embed = nn.Embedding(5, config.hidden_dim)

        # preload from a checkpoint (NOTE: this happens BEFORE checkpointer loading latest checkpoint!)
        if self.config.pretrain_path is not None:
            try:
                ckpt = torch.load(self.config.pretrain_path)  # local path
                self.load_state_dict(ckpt["model"], strict=True)
                del ckpt
                print(f"Loaded DiT from {self.config.pretrain_path}")
            except Exception as e:
                print(
                    f"Failed to load DiT from {self.config.pretrain_path}: {e}, make sure you resumed from a valid checkpoint!"
                )

        # sampler
        self.scheduler = FlowMatchingScheduler(shift=config.flow_shift)

        n_params = 0
        for p in self.dit.parameters():
            n_params += p.numel()
        print(f"Number of parameters in DiT: {n_params/1e6:.2f}M")

    # override state_dict to exclude vae and dino, so we only save the trainable params.
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        keys_to_del = []
        for k in state_dict.keys():
            if "vae" in k or "dino" in k:
                keys_to_del.append(k)

        for k in keys_to_del:
            del state_dict[k]

        return state_dict

    # override to support tolerant loading (only load matched shape)
    def load_state_dict(self, state_dict, strict=True, assign=False):
        local_state_dict = self.state_dict()
        seen_keys = {k: False for k in local_state_dict.keys()}
        for k, v in state_dict.items():
            if k in local_state_dict:
                seen_keys[k] = True
                if local_state_dict[k].shape == v.shape:
                    local_state_dict[k].copy_(v)
                else:
                    print(f"mismatching shape for key {k}: loaded {local_state_dict[k].shape} but model has {v.shape}")
            else:
                print(f"unexpected key {k} in loaded state dict")
        for k in seen_keys:
            if not seen_keys[k]:
                print(f"missing key {k} in loaded state dict")

    # this happens before checkpointer loading old models !!!
    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format=memory_format)
        device = next(self.dit.parameters()).device

        self.dit.to(dtype=self.precision)

        if self.config.use_num_parts_cond:
            self.num_part_embed.to(dtype=self.precision)

        # cast scheduler to device
        self.scheduler.to(device)

    def get_cond(self, cond_image, num_part=None):
        # image condition
<<<<<<< HEAD
        cond_image = cond_image.to(dtype=self.precision)
=======
        #device = next(self.dino.parameters()).device
        #cond_image = cond_image.to(device=device, dtype=self.precision)
        cond_image = cond_image.to(device=self.dino.device, dtype=self.precision)
>>>>>>> 20c359f (WIP: temporary updates before switching branch)
        with torch.no_grad():
            cond = self.dino(cond_image).last_hidden_state
        cond = F.layer_norm(cond.float(), cond.shape[-1:]).to(dtype=self.precision)  # [B, L, C]
        
        # num_part condition
        if self.config.use_num_parts_cond:
            if num_part is None:
                # use a default value (2-10 parts)
                num_part_coarse = torch.ones(cond.shape[0], dtype=torch.int64, device=cond.device) * 2
            else:
                # coarse range
                num_part_coarse = torch.ones(cond.shape[0], dtype=torch.int64, device=cond.device)
                num_part_coarse[num_part == 2] = 1
                num_part_coarse[(num_part > 2) & (num_part <= 10)] = 2
                num_part_coarse[(num_part > 10) & (num_part <= 100)] = 3
                num_part_coarse[num_part > 100] = 4
            num_part_embed = self.num_part_embed(num_part_coarse).unsqueeze(1)  # [B, 1, C]
            cond = torch.cat([cond, num_part_embed], dim=1)  # [B, L+1, C]

        return cond

    def training_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output = {}
        loss = 0

        cond_images = self.preprocess_cond_image(
            data["cond_images"]
        )  # [B, N, 3, 518, 518], we may load multiple (N) cond images for the same shape
        B, N, C, H, W = cond_images.shape

        if self.config.use_num_parts_cond:
            cond_num_part = data["num_part"].repeat_interleave(N, dim=0)
        else:
            cond_num_part = None

        cond = self.get_cond(cond_images.view(-1, C, H, W), cond_num_part)  # [B*N, L, C]

        # random CFG dropout
        if self.training:
            mask = torch.rand((B * N, 1, 1), device=cond.device, dtype=cond.dtype) >= 0.1
            cond = cond * mask

        with torch.no_grad():
            # encode latent
            if self.config.use_parts:
                # encode two parts and concat latent
                part0_data = {k.replace("_part0", ""): v for k, v in data.items() if "_part0" in k}
                part1_data = {k.replace("_part1", ""): v for k, v in data.items() if "_part1" in k}
                posterior0 = self.vae.encode(part0_data)
                posterior1 = self.vae.encode(part1_data)
                if self.training and self.config.shuffle_parts:
                    if np.random.rand() < 0.5:
                        posterior0, posterior1 = posterior1, posterior0
                latent = torch.cat(
                    [
                        posterior0.mode().float().nan_to_num_(0),
                        posterior1.mode().float().nan_to_num_(0),
                    ],
                    dim=1,
                )  # [B, 2L, C]
            else:
                posterior = self.vae.encode(data)
                latent = posterior.mode().float().nan_to_num_(0)  # use mean as the latent, [B, L, C]

            # repeat latent for each cond image
            if N != 1:
                latent = latent.repeat_interleave(N, dim=0)

            # random sample timesteps and add noise
            noisy_latent, noise, timesteps = self.scheduler.add_noise(
                latent, self.config.logitnorm_mean, self.config.logitnorm_std
            )

        noisy_latent = noisy_latent.to(dtype=self.precision)
        model_pred = self.dit(noisy_latent, cond, timesteps)

        # flow-matching loss
        target = noise - latent
        loss = F.mse_loss(model_pred.float(), target.float())

        # metrics
        with torch.no_grad():
            output["scalar"] = {}  # for wandb logging
            output["scalar"]["loss_mse"] = loss.detach()

        return output, loss

    @torch.no_grad()
    def validation_step(
        self,
        data: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return self.training_step(data, iteration)

    @torch.inference_mode()
    @sync_timer("flow forward")
<<<<<<< HEAD
    def forward(
=======
    def forward_old(
>>>>>>> 20c359f (WIP: temporary updates before switching branch)
        self,
        data: dict[str, torch.Tensor],
        num_steps: int = 30,
        cfg_scale: float = 7.0,
        verbose: bool = True,
        generator: torch.Generator | None = None,
<<<<<<< HEAD
    ) -> dict[str, torch.Tensor]:
        # the inference sampling
        cond_images = self.preprocess_cond_image(data["cond_images"])  # [B, 3, 518, 518]
        B = cond_images.shape[0]
        assert B == 1, "Only support batch size 1 for now."
=======
        node_id :str | None = None) -> dict[str, torch.Tensor]:
        # the inference sampling
        cond_images = self.preprocess_cond_image(data["cond_images"])  # [B, 3, 518, 518]
        if self.cpu_offload:
            try:
                self.dino.to(device=self.device, dtype=self.precision, non_blocking=True)
            except RuntimeError as e:
                print(f"Failed to move DINO to GPU: {e}")
                self.device = "cpu"
                self.dino.to("cpu")
                torch.cuda.empty_cache()
        #cond_images = self.preprocess_cond_image(data["cond_images"]).to(device=self.dino.device, dtype=self.precision)
        cond_images = cond_images.to(device=self.device, dtype=self.precision)

        B = cond_images.shape[0]
        assert B == 1, "Only support batch size 1 for now."
        # num_part condition
        cond_num_part = data.get("num_part") if self.config.use_num_parts_cond else None
        cond = self.get_cond(cond_images, cond_num_part)
>>>>>>> 20c359f (WIP: temporary updates before switching branch)

        # num_part condition
        if self.config.use_num_parts_cond and "num_part" in data:
            cond_num_part = data["num_part"]  # [B,], int
        else:
            cond_num_part = None

        if self.cpu_offload:
            self.dino.to(device=self.device)


        # if self.cpu_offload:
        #     self.dino.to(device=self.device)

        if self.cpu_offload:
            try:
                self.dino.to(device=self.device, dtype=self.precision, non_blocking=True)
            except RuntimeError as e:
                print(f"Failed to move DINO to GPU: {e}")
                self.dino.to("cpu")
                torch.cuda.empty_cache()
                self.device = "cpu"
        # Force image tensor to follow dino's device
        cond_images = cond_images.to(self.device, dtype=self.precision)
        cond = self.get_cond(cond_images, cond_num_part)
        if self.cpu_offload:
            self.dino.to(device="cpu")
            torch.cuda.empty_cache()

        if self.config.use_parts:
            x = torch.randn(
                B,
                self.config.latent_size * 2,
                self.config.latent_dim,
                device=cond.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            x = torch.randn(
                B,
                self.config.latent_size,
                self.config.latent_dim,
                device=cond.device,
                dtype=torch.float32,
                generator=generator,
            )

        cond_input = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        # flow-matching
        sigmas = np.linspace(1, 0, num_steps + 1)
        sigmas = self.scheduler.shift * sigmas / (1 + (self.scheduler.shift - 1) * sigmas)
        sigmas_pair = list((sigmas[i], sigmas[i + 1]) for i in range(num_steps))


        for sigma, sigma_prev in tqdm.tqdm(sigmas_pair, desc="Flow Sampling", disable=not verbose):

        # Add ProgressBar that connects to ComfyUI UI
        pbar = ProgressBar(total=len(sigmas_pair), node_id=node_id)
        #for sigma, sigma_prev in tqdm.tqdm(sigmas_pair, desc="Flow Sampling", disable=not verbose):
        for i, (sigma, sigma_prev) in enumerate(sigmas_pair):
            pbar.update_absolute(i + 1)
            # classifier-free guidance
            timesteps = torch.tensor([1000 * sigma] * B * 2, device=x.device, dtype=x.dtype)
            x_input = torch.cat([x, x], dim=0)

            # predict v
            x_input = x_input.to(dtype=self.precision)
            pred = self.dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale

            # scheduler step
            x = x - (sigma - sigma_prev) * pred_v

        output = {}
        output["latent"] = x

        # leave mesh extraction to vae
        return output

    @torch.inference_mode()
    @sync_timer("flow forward")
    def forward(
            self,
            data: dict[str, torch.Tensor],
            num_steps: int = 30,
            cfg_scale: float = 7.0,
            verbose: bool = True,
            generator: torch.Generator | None = None,
            node_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        # Preprocess conditioning image and move to device
        cond_images = self.preprocess_cond_image(data["cond_images"])

        if self.cpu_offload:
            try:
                self.dino.to(device=self.device, dtype=self.precision, non_blocking=True)
            except RuntimeError as e:
                print(f"Failed to move DINO to GPU: {e}")
                self.device = "cpu"
                self.dino.to("cpu")
                torch.cuda.empty_cache()

        cond_images = cond_images.to(device=self.device, dtype=self.precision)
        B = cond_images.shape[0]
        assert B == 1, "Only batch size 1 is supported."

        cond_num_part = data.get("num_part") if self.config.use_num_parts_cond else None
        cond = self.get_cond(cond_images, cond_num_part)

        if self.cpu_offload:
            self.dino.to("cpu")
            torch.cuda.empty_cache()

        # Prepare latent
        latent_size = self.config.latent_size * 2 if self.config.use_parts else self.config.latent_size
        x = torch.randn(
            B, latent_size, self.config.latent_dim,
            device=cond.device, dtype=torch.float32, generator=generator
        )

        cond_input = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        # Flow sampling
        sigmas = np.linspace(1, 0, num_steps + 1)
        sigmas = self.scheduler.shift * sigmas / (1 + (self.scheduler.shift - 1) * sigmas)
        sigmas_pair = [(sigmas[i], sigmas[i + 1]) for i in range(num_steps)]

        #  Progress bar
        pbar = ProgressBar(total=len(sigmas_pair), node_id=node_id)

        for i, (sigma, sigma_prev) in enumerate(sigmas_pair):
            pbar.update_absolute(i + 1)
            timesteps = torch.tensor([1000 * sigma] * B * 2, device=x.device, dtype=x.dtype)
            x_input = torch.cat([x, x], dim=0).to(dtype=self.precision)
            pred = self.dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * pred_v

        return {"latent": x}

