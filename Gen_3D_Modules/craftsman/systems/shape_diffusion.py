from dataclasses import dataclass, field

import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.utils.misc import get_rank
from craftsman.utils.typing import *

import comfy.utils

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def ddim_sample(ddim_scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2

    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma
    # set timesteps
    ddim_scheduler.set_timesteps(steps)
    timesteps = ddim_scheduler.timesteps.to(device)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, and between [0, 1]
    extra_step_kwargs = {
        # "eta": eta,
        "generator": generator
    }

    # reverse
    comfy_pbar = comfy.utils.ProgressBar(steps)
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample
        
        comfy_pbar.update_absolute(i + 1)

        yield latents, t


@craftsman.register("shape-diffusion-system")
class ShapeDiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        val_samples_json: str = None
        z_scale_factor: float = 1.0
        guidance_scale: float = 7.5
        num_inference_steps: int = 50
        eta: float = 0.0
        snr_gamma: float = 5.0

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        condition_model_type: str = None
        condition_model: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        self.condition = craftsman.find(self.cfg.condition_model_type)(self.cfg.condition_model)

        self.denoiser_model = craftsman.find(self.cfg.denoiser_model_type)(self.cfg.denoiser_model)

        self.noise_scheduler = craftsman.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)
        self.denoise_scheduler = craftsman.find(self.cfg.denoise_scheduler_type)(**self.cfg.denoise_scheduler)

        self.z_scale_factor = self.cfg.z_scale_factor

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # encode shape latents
        shape_embeds, kl_embed, posterior = self.shape_model.encode(
            batch["surface"][..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True
        )
        latents = kl_embed * self.z_scale_factor

        cond_latents = self.condition(batch)
        cond_latents = cond_latents.to(latents).view(latents.shape[0], -1, cond_latents.shape[-1])

        # Sample noise that we"ll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents).to(latents)
        bs = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        # import pdb; pdb.set_trace()

        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # x_t
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # diffusion model forward
        noise_pred = self.denoiser_model(noisy_z, timesteps, cond_latents)

        # compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise 
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Prediction Type: {self.noise_scheduler.prediction_type} not supported.")
        if self.cfg.snr_gamma == 0:
            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="mean")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="mean")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            
            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="none")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="none")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return {
            "loss_diffusion": loss,
            "latents": latents,
            "x_0": noisy_z,
            "noise": noise,
            "noise_pred": noise_pred,
            "timesteps": timesteps,
            }

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        if get_rank() == 0:
            sample_inputs = json.loads(open(self.cfg.val_samples_json).read()) # condition
            sample_inputs_ = copy.deepcopy(sample_inputs)
            sample_outputs = self.sample(sample_inputs) # list
            for i, sample_output in enumerate(sample_outputs):
                mesh_v_f, has_surface = self.shape_model.extract_geometry(sample_output, octree_depth=7)
                for j in range(len(mesh_v_f)):
                    if "text" in sample_inputs_ and "image" in sample_inputs_:
                        name = sample_inputs_["image"][j].split("/")[-1].replace(".png", "")
                    elif "text" in sample_inputs_ and "mvimage" in sample_inputs_:
                        name = sample_inputs_["mvimages"][j][0].split("/")[-2].replace(".png", "")
                    elif "text" in sample_inputs_:
                        name = sample_inputs_["text"][j].replace(" ", "_")
                    elif "image" in sample_inputs_:
                        name = sample_inputs_["image"][j].split("/")[-1].replace(".png", "")
                    elif "mvimages" in sample_inputs_:
                        name = sample_inputs_["mvimages"][j][0].split("/")[-2].replace(".png", "")
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}_{i}.obj",
                        mesh_v_f[j][0], mesh_v_f[j][1]
                    )

        out = self(batch)
        if self.global_step == 0:
            latents = self.shape_model.decode(out["latents"])
            mesh_v_f, has_surface = self.shape_model.extract_geometry(latents)
            self.save_mesh(
                f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}.obj",
                mesh_v_f[0][0], mesh_v_f[0][1]
            )
            # exit()
        torch.cuda.empty_cache()

        return {"val/loss": out["loss_diffusion"]}

    @torch.no_grad()
    def sample(self,
               sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
               sample_times: int = 1,
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               eta: float = 0.0,
               return_intermediates: bool = False, 
               camera_embeds: Optional[torch.Tensor] = None,
               seed: Optional[int] = None,
               **kwargs):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale > 0

        # conditional encode
        if "image" in sample_inputs:
            sample_inputs["image"] = [Image.open(img) for img in sample_inputs["image"]]
            cond = self.condition.encode_image(sample_inputs["image"])
            if do_classifier_free_guidance:
                un_cond = self.condition.empty_image_embeds.repeat(len(sample_inputs["image"]), 1, 1).to(cond)
                cond = torch.cat([un_cond, cond], dim=0)
        elif "mvimages" in sample_inputs: # by default 4 views
            bs = len(sample_inputs["mvimages"])
            cond = []
            for image in sample_inputs["mvimages"]:
                if isinstance(image, list) and isinstance(image[0], str):
                    sample_inputs["image"] = [Image.open(img) for img in image] # List[PIL]
                else:
                    sample_inputs["image"] = image
                cond += [self.condition.encode_image(sample_inputs["image"])]
            cond = torch.stack(cond, dim=0)# tensor  shape 为[len(sample_inputs["mvimages"], 4*(num_latents+1), context_dim]
            if do_classifier_free_guidance:
                un_cond = self.condition.empty_image_embeds.unsqueeze(0).repeat(len(sample_inputs["mvimages"]), cond.shape[1] // self.condition.cfg.n_views, 1, 1).to(cond) # shape 为[len(sample_inputs["mvimages"], 4*(num_latents+1), context_dim]
                cond = torch.cat([un_cond, cond], dim=0).view(bs * 2, -1, cond[0].shape[-1]) 
        else:
            raise NotImplementedError("Only text, image or mvimages condition is supported.")

        outputs = []
        latents = None
        
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None
            
        if not return_intermediates:
            for _ in range(sample_times):
                sample_loop = ddim_sample(
                    self.denoise_scheduler,
                    self.denoiser_model.eval(),
                    shape=self.shape_model.latent_shape,
                    cond=cond,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=self.device,
                    eta=eta,
                    disable_prog=False,
                    generator= generator
                )
                for sample, t in sample_loop:
                    latents = sample
                outputs.append(self.shape_model.decode(latents / self.z_scale_factor, **kwargs))
        else:
            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator= generator
            )

            iter_size = steps // sample_times
            i = 0
            for sample, t in sample_loop:
                latents = sample
                if i % iter_size == 0 or i == steps - 1:
                    outputs.append(self.shape_model.decode(latents / self.z_scale_factor, **kwargs))
                i += 1

        return outputs
    

    def on_validation_epoch_end(self):
        pass
