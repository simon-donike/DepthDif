# taken from https://huggingface.co/blog/annotated-diffusion
from __future__ import annotations

import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def get_beta_schedule(
    variant: str,
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Compute get beta schedule and return the result.

    Args:
        variant (str): Input value.
        timesteps (int): Step or timestep value.
        beta_start (float): Input value.
        beta_end (float): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    if not (0.0 < float(beta_start) < float(beta_end) < 1.0):
        raise ValueError(
            f"Invalid beta range: beta_start={beta_start}, beta_end={beta_end}. Expected 0 < start < end < 1."
        )

    if variant == "cosine":
        return cosine_beta_schedule(timesteps, beta_start=None, beta_end=None) # pass None to use default values for cosine schedule
    elif variant == "linear":
        return linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif variant == "quadratic":
        return quadratic_beta_schedule(
            timesteps, beta_start=beta_start, beta_end=beta_end
        )
    elif variant == "sigmoid":
        return sigmoid_beta_schedule(
            timesteps, beta_start=beta_start, beta_end=beta_end
        )
    else:
        raise ValueError(
            f"Unknown beta schedule '{variant}'. Supported: cosine, linear, quadratic, sigmoid."
        )


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    beta_start: float | None = 0.0001,
    beta_end: float | None = None,
) -> torch.Tensor:
    """Compute cosine beta schedule and return the result.

    Args:
        timesteps (int): Step or timestep value.
        s (float): Input value.
        beta_start (float | None): Input value.
        beta_end (float | None): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    # Set beta_end to a default value if not provided
    if beta_end is None:
        beta_end = 0.999
    if beta_start is None:
        beta_start = 1e-8
        
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end) #ToDo: investigate clipping effects, might not be ideal here because it prevents very strong noise at the end


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    """Compute linear beta schedule and return the result.

    Args:
        timesteps (int): Step or timestep value.
        beta_start (float): Input value.
        beta_end (float): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    """Compute quadratic beta schedule and return the result.

    Args:
        timesteps (int): Step or timestep value.
        beta_start (float): Input value.
        beta_end (float): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    """Compute sigmoid beta schedule and return the result.

    Args:
        timesteps (int): Step or timestep value.
        beta_start (float): Input value.
        beta_end (float): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
