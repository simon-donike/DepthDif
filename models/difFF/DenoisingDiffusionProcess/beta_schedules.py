# taken from https://huggingface.co/blog/annotated-diffusion
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def get_beta_schedule(variant, timesteps, beta_start=0.0001, beta_end=0.02):
    if not (0.0 < float(beta_start) < float(beta_end) < 1.0):
        raise ValueError(
            f"Invalid beta range: beta_start={beta_start}, beta_end={beta_end}. Expected 0 < start < end < 1."
        )
    
    if variant=='cosine':
        return cosine_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif variant=='linear':
        return linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif variant=='quadratic':
        return quadratic_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif variant=='sigmoid':
        return sigmoid_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    else:
        raise ValueError(
            f"Unknown beta schedule '{variant}'. Supported: cosine, linear, quadratic, sigmoid."
        )

def cosine_beta_schedule(timesteps, s=0.008, beta_start=0.0001, beta_end=0.02):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
