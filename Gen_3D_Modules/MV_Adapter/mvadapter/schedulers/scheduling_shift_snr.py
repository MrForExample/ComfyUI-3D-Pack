from typing import Any

import torch

from .scheduler_utils import SNR_to_betas, compute_snr


class ShiftSNRScheduler:
    def __init__(
        self,
        noise_scheduler: Any,
        timesteps: Any,
        shift_scale: float,
        scheduler_class: Any,
    ):
        self.noise_scheduler = noise_scheduler
        self.timesteps = timesteps
        self.shift_scale = shift_scale
        self.scheduler_class = scheduler_class

    def _get_shift_scheduler(self):
        """
        Prepare scheduler for shifted betas.

        :return: A scheduler object configured with shifted betas
        """
        snr = compute_snr(self.timesteps, self.noise_scheduler)
        shifted_betas = SNR_to_betas(snr / self.shift_scale)

        return self.scheduler_class.from_config(
            self.noise_scheduler.config, trained_betas=shifted_betas.numpy()
        )

    def _get_interpolated_shift_scheduler(self):
        """
        Prepare scheduler for shifted betas and interpolate with the original betas in log space.

        :return: A scheduler object configured with interpolated shifted betas
        """
        snr = compute_snr(self.timesteps, self.noise_scheduler)
        shifted_snr = snr / self.shift_scale

        weighting = self.timesteps.float() / (
            self.noise_scheduler.config.num_train_timesteps - 1
        )
        interpolated_snr = torch.exp(
            torch.log(snr) * (1 - weighting) + torch.log(shifted_snr) * weighting
        )

        shifted_betas = SNR_to_betas(interpolated_snr)

        return self.scheduler_class.from_config(
            self.noise_scheduler.config, trained_betas=shifted_betas.numpy()
        )

    @classmethod
    def from_scheduler(
        cls,
        noise_scheduler: Any,
        shift_mode: str = "default",
        timesteps: Any = None,
        shift_scale: float = 1.0,
        scheduler_class: Any = None,
    ):
        # Check input
        if timesteps is None:
            timesteps = torch.arange(0, noise_scheduler.config.num_train_timesteps)
        if scheduler_class is None:
            scheduler_class = noise_scheduler.__class__

        # Create scheduler
        shift_scheduler = cls(
            noise_scheduler=noise_scheduler,
            timesteps=timesteps,
            shift_scale=shift_scale,
            scheduler_class=scheduler_class,
        )

        if shift_mode == "default":
            return shift_scheduler._get_shift_scheduler()
        elif shift_mode == "interpolated":
            return shift_scheduler._get_interpolated_shift_scheduler()
        else:
            raise ValueError(f"Unknown shift_mode: {shift_mode}")


if __name__ == "__main__":
    """
    Compare the alpha values for different noise schedulers.
    """
    import matplotlib.pyplot as plt
    from diffusers import DDPMScheduler

    from .scheduler_utils import compute_alpha

    # Base
    timesteps = torch.arange(0, 1000)
    noise_scheduler_base = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
    )
    alpha = compute_alpha(timesteps, noise_scheduler_base)
    plt.plot(timesteps.numpy(), alpha.numpy(), label="Base")

    # Kolors
    num_train_timesteps_ = 1100
    timesteps_ = torch.arange(0, num_train_timesteps_)
    noise_kwargs = {"beta_end": 0.014, "num_train_timesteps": num_train_timesteps_}
    noise_scheduler_kolors = DDPMScheduler.from_config(
        noise_scheduler_base.config, **noise_kwargs
    )
    alpha = compute_alpha(timesteps_, noise_scheduler_kolors)
    plt.plot(timesteps_.numpy(), alpha.numpy(), label="Kolors")

    # Shift betas
    shift_scale = 8.0
    noise_scheduler_shift = ShiftSNRScheduler.from_scheduler(
        noise_scheduler_base, shift_mode="default", shift_scale=shift_scale
    )
    alpha = compute_alpha(timesteps, noise_scheduler_shift)
    plt.plot(timesteps.numpy(), alpha.numpy(), label="Shift Noise (scale 8.0)")

    # Shift betas (interpolated)
    noise_scheduler_inter = ShiftSNRScheduler.from_scheduler(
        noise_scheduler_base, shift_mode="interpolated", shift_scale=shift_scale
    )
    alpha = compute_alpha(timesteps, noise_scheduler_inter)
    plt.plot(timesteps.numpy(), alpha.numpy(), label="Interpolated (scale 8.0)")

    # ZeroSNR
    noise_scheduler = DDPMScheduler.from_config(
        noise_scheduler_base.config, rescale_betas_zero_snr=True
    )
    alpha = compute_alpha(timesteps, noise_scheduler)
    plt.plot(timesteps.numpy(), alpha.numpy(), label="ZeroSNR")

    plt.legend()
    plt.grid()
    plt.savefig("check_alpha.png")
