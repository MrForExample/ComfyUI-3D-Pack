import torch


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device=None):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def SNR_to_betas(snr):
    """
    Converts SNR to betas
    """
    # alphas_cumprod = pass
    # snr = (alpha / ) ** 2
    # alpha_t^2 / (1 - alpha_t^2) = snr
    alpha_t = (snr / (1 + snr)) ** 0.5
    alphas_cumprod = alpha_t**2
    alphas = alphas_cumprod / torch.cat(
        [torch.ones(1, device=snr.device), alphas_cumprod[:-1]]
    )
    betas = 1 - alphas
    return betas


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per Min-SNR-Diffusion-Training/guided_diffusion/gaussian_diffusion.py at 521b624bd70c67cee4bdf49225915f5
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from Min-SNR-Diffusion-Training/guided_diffusion/gaussian_diffusion.py at 521b624bd70c67cee4bdf49225915f5
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def compute_alpha(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    return alpha
