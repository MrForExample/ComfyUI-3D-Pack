import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from diffusers.training_utils import compute_snr
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ..pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from ..schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils.core import find
from ..utils.typing import *
from .base import BaseSystem
from .utils import encode_prompt, vae_encode


def compute_embeddings(
    prompt_batch,
    empty_prompt_indices,
    text_encoders,
    tokenizers,
    is_train=True,
    **kwargs,
):
    original_size = kwargs["original_size"]
    target_size = kwargs["target_size"]
    crops_coords_top_left = kwargs["crops_coords_top_left"]

    for i in range(empty_prompt_indices.shape[0]):
        if empty_prompt_indices[i]:
            prompt_batch[i] = ""

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, 0, is_train
    )
    add_text_embeds = pooled_prompt_embeds.to(
        device=prompt_embeds.device, dtype=prompt_embeds.dtype
    )

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(
        device=prompt_embeds.device, dtype=prompt_embeds.dtype
    )

    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}


class MVAdapterTextSDXLSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        # Model / Adapter
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        pretrained_vae_name_or_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
        pretrained_adapter_name_or_path: Optional[str] = None
        pretrained_unet_name_or_path: Optional[str] = None
        init_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

        use_fp16_vae: bool = True
        use_fp16_clip: bool = True

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        train_cond_encoder: bool = True
        prompt_drop_prob: float = 0.0
        image_drop_prob: float = 0.0
        cond_drop_prob: float = 0.0

        gradient_checkpointing: bool = False

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        noise_offset: float = 0.0
        input_perturbation: float = 0.0
        snr_gamma: Optional[float] = 5.0
        prediction_type: Optional[str] = None
        shift_noise: bool = False
        shift_noise_mode: str = "interpolated"
        shift_noise_scale: float = 1.0

        # Evaluation
        eval_seed: int = 0
        eval_num_inference_steps: int = 30
        eval_guidance_scale: float = 1.0
        eval_height: int = 512
        eval_width: int = 512

    cfg: Config

    def configure(self):
        super().configure()

        # Prepare pipeline
        pipeline_kwargs = {}
        if self.cfg.pretrained_vae_name_or_path is not None:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_vae_name_or_path
            )
        if self.cfg.pretrained_unet_name_or_path is not None:
            pipeline_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_unet_name_or_path
            )

        pipeline: MVAdapterT2MVSDXLPipeline
        pipeline = MVAdapterT2MVSDXLPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipeline_kwargs
        )

        init_adapter_kwargs = OmegaConf.to_container(self.cfg.init_adapter_kwargs)
        if "self_attn_processor" in init_adapter_kwargs:
            self_attn_processor = init_adapter_kwargs["self_attn_processor"]
            if self_attn_processor is not None and isinstance(self_attn_processor, str):
                self_attn_processor = find(self_attn_processor)
                init_adapter_kwargs["self_attn_processor"] = self_attn_processor
        pipeline.init_custom_adapter(**init_adapter_kwargs)

        if self.cfg.pretrained_adapter_name_or_path:
            pretrained_path = os.path.dirname(self.cfg.pretrained_adapter_name_or_path)
            adapter_name = os.path.basename(self.cfg.pretrained_adapter_name_or_path)
            pipeline.load_custom_adapter(pretrained_path, weight_name=adapter_name)

        noise_scheduler = DDPMScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        if self.cfg.shift_noise:
            noise_scheduler = ShiftSNRScheduler.from_scheduler(
                noise_scheduler,
                shift_mode=self.cfg.shift_noise_mode,
                shift_scale=self.cfg.shift_noise_scale,
                scheduler_class=DDPMScheduler,
            )
        pipeline.scheduler = noise_scheduler

        # Prepare models
        self.pipeline: MVAdapterT2MVSDXLPipeline = pipeline
        self.vae = self.pipeline.vae.to(
            dtype=torch.float16 if self.cfg.use_fp16_vae else torch.float32
        )
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.text_encoder = self.pipeline.text_encoder.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.text_encoder_2 = self.pipeline.text_encoder_2.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )

        self.cond_encoder = self.pipeline.cond_encoder
        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler
        self.inference_scheduler = DDPMScheduler.from_config(
            self.noise_scheduler.config
        )
        self.pipeline.scheduler = self.inference_scheduler
        if self.cfg.prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=self.cfg.prediction_type
            )

        # Prepare trainable / non-trainable modules
        trainable_modules = self.cfg.trainable_modules
        if trainable_modules and len(trainable_modules) > 0:
            self.unet.requires_grad_(False)
            for name, module in self.unet.named_modules():
                for trainable_module in trainable_modules:
                    if trainable_module in name:
                        module.requires_grad_(True)
        else:
            self.unet.requires_grad_(True)
        self.cond_encoder.requires_grad_(self.cfg.train_cond_encoder)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Others
        # Prepare gradient checkpointing
        if self.cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(
        self,
        noisy_latents: Tensor,
        conditioning_pixel_values: Tensor,
        timesteps: Tensor,
        prompts: List[str],
        num_views: int,
        **kwargs,
    ) -> Dict[str, Any]:
        bsz = noisy_latents.shape[0]
        b_samples = bsz // num_views
        num_batch_images = num_views

        prompt_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.prompt_drop_prob
        )
        image_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.image_drop_prob
        )
        cond_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device) < self.cfg.cond_drop_prob
        )
        prompt_drop_mask = prompt_drop_mask | cond_drop_mask
        image_drop_mask = image_drop_mask | cond_drop_mask

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Here, we compute not just the text embeddings but also the additional embeddings
            # needed for the SD XL UNet to operate.
            additional_embeds = compute_embeddings(
                prompts,
                prompt_drop_mask,
                [self.text_encoder, self.text_encoder_2],
                [self.tokenizer, self.tokenizer_2],
                **kwargs,
            )

        for key, value in additional_embeds.items():
            kwargs[key] = value.repeat_interleave(num_batch_images, dim=0)

        conditioning_features = self.cond_encoder(conditioning_pixel_values)

        added_cond_kwargs = {
            "text_embeds": kwargs["text_embeds"],
            "time_ids": kwargs["time_ids"],
        }

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=kwargs["prompt_embeds"],
            added_cond_kwargs=added_cond_kwargs,
            down_intrablock_additional_residuals=conditioning_features,
            cross_attention_kwargs={"num_views": num_views},
        ).sample

        return {"noise_pred": noise_pred}

    def training_step(self, batch, batch_idx):
        num_views = batch["num_views"]

        vae_max_slice = 8
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = []
            for i in range(0, batch["rgb"].shape[0], vae_max_slice):
                latents.append(
                    vae_encode(
                        self.vae,
                        batch["rgb"][i : i + vae_max_slice].to(self.vae.dtype) * 2 - 1,
                        sample=True,
                        apply_scale=True,
                    ).float()
                )
            latents = torch.cat(latents, dim=0)

        bsz = latents.shape[0]
        b_samples = bsz // num_views

        noise = torch.randn_like(latents)
        if self.cfg.noise_offset is not None:
            # # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        noise_mask = (
            batch["noise_mask"]
            if "noise_mask" in batch
            else torch.ones((bsz,), dtype=torch.bool, device=latents.device)
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b_samples,),
            device=latents.device,
            dtype=torch.long,
        )
        timesteps = timesteps.repeat_interleave(num_views)
        timesteps[~noise_mask] = 0

        if self.cfg.input_perturbation is not None:
            new_noise = noise + self.cfg.input_perturbation * torch.randn_like(noise)
            noisy_latents = self.noise_scheduler.add_noise(
                latents, new_noise, timesteps
            )
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noisy_latents[~noise_mask] = latents[~noise_mask]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}"
            )

        model_pred = self(noisy_latents, batch["source_rgb"], timesteps, **batch)[
            "noise_pred"
        ]

        model_pred = model_pred[noise_mask]
        target = target[noise_mask]

        if self.cfg.snr_gamma is None:
            loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            loss = F.mse_loss(model_pred, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        self.log("train/loss", loss, prog_bar=True)

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch)

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def get_input_visualizations(self, batch):
        return [
            {
                "type": "rgb",
                "img": rearrange(
                    batch["source_rgb"],
                    "(B N) C H W -> (B H) (N W) C",
                    N=batch["num_views"],
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]

    def get_output_visualizations(self, batch, outputs):
        images = [
            {
                "type": "rgb",
                "img": rearrange(
                    batch["source_rgb"],
                    "(B N) C H W -> (B H) (N W) C",
                    N=batch["num_views"],
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    outputs, "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]
        return images

    def generate_images(self, batch, **kwargs):
        return self.pipeline(
            prompt=batch["prompts"],
            control_image=batch["source_rgb"],
            num_images_per_prompt=batch["num_views"],
            generator=torch.Generator(device=self.device).manual_seed(
                self.cfg.eval_seed
            ),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
            output_type="pt",
        ).images

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            self.pipeline.save_custom_adapter(
                os.path.dirname(self.get_save_dir()),
                "custom_adapter.safetensors",
                safe_serialization=True,
                include_keys=self.cfg.trainable_modules,
            )

    def on_check_train(self, batch):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_input_visualizations(batch),
            name="train_step_input",
            step=self.true_global_step,
        )

    def validation_step(self, batch, batch_idx):
        out = self.generate_images(batch)

        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                self.get_output_visualizations(batch, out),
                name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self.generate_images(batch)

        self.save_image_grid(
            f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}.jpg",
            self.get_output_visualizations(batch, out),
            name=f"test_step_output_{self.global_rank}_{batch_idx}",
            step=self.true_global_step,
        )

    def on_test_end(self):
        pass
