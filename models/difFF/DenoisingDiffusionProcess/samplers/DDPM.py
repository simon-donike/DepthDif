"""

This file contains the DDPM sampler class for a diffusion process

"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..beta_schedules import *


class DDPM_Sampler(nn.Module):

    """DDPM sampler that performs one reverse-diffusion step at a time."""
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        parameterization: str = "epsilon",
    ) -> None:

        """Initialize DDPM_Sampler with configured parameters.

        Args:
            num_timesteps (int): Step or timestep value.
            schedule (str): Input value.
            beta_start (float): Input value.
            beta_end (float): Input value.
            parameterization (str): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()

        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.parameterization = self._normalize_parameterization(parameterization)

        self.register_buffer(
            "betas",
            get_beta_schedule(
                self.schedule,
                self.num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
            ),
        )
        self.register_buffer("betas_sqrt", self.betas.sqrt())
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, 0))
        self.register_buffer("alphas_cumprod_sqrt", self.alphas_cumprod.sqrt())
        self.register_buffer(
            "alphas_one_minus_cumprod_sqrt", (1 - self.alphas_cumprod).sqrt()
        )
        self.register_buffer("alphas_sqrt", self.alphas.sqrt())
        self.register_buffer("alphas_sqrt_recip", 1 / (self.alphas_sqrt))

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run the sampler call and return the next sample.

        Args:
            *args (Any): Additional positional arguments forwarded to the underlying call.
            **kwargs (Any): Additional keyword arguments forwarded to the underlying call.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        return self.step(*args, **kwargs)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Predict the previous diffusion sample for one timestep.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            z_t (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        assert (t < self.num_timesteps).all()

        noise_pred = self._prediction_to_noise(x_t, t, z_t)

        # 2. Approximate Distribution of Previous Sample in the chain
        mean_pred, std_pred = self.posterior_params(x_t, t, noise_pred)

        # 3. Sample from the distribution
        z = torch.randn_like(x_t) if any(t > 0) else torch.zeros_like(x_t)
        return mean_pred + std_pred * z

    def posterior_params(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """Compute posterior params and return the result.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            noise_pred (torch.Tensor): Tensor input for the computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing computed outputs.
        """
        assert (t < self.num_timesteps).all()

        beta_t = self.betas[t].view(x_t.shape[0], 1, 1, 1)
        alpha_one_minus_cumprod_sqrt_t = self.alphas_one_minus_cumprod_sqrt[t].view(
            x_t.shape[0], 1, 1, 1
        )
        alpha_sqrt_recip_t = self.alphas_sqrt_recip[t].view(x_t.shape[0], 1, 1, 1)

        mean = alpha_sqrt_recip_t * (
            x_t - beta_t * noise_pred / alpha_one_minus_cumprod_sqrt_t
        )
        std = self.betas_sqrt[t].view(x_t.shape[0], 1, 1, 1)

        return mean, std

    @staticmethod
    def _normalize_parameterization(parameterization: str) -> str:
        """Helper that computes normalize parameterization.

        Args:
            parameterization (str): Input value.

        Returns:
            str: Computed scalar output.
        """
        value = str(parameterization).strip().lower().replace("-", "").replace("_", "")
        if value in {"epsilon", "eps", "noise"}:
            return "epsilon"
        if value in {"x0", "xstart"}:
            return "x0"
        raise ValueError(
            "parameterization must be one of {'epsilon', 'x0'} "
            f"(got '{parameterization}')."
        )

    def set_parameterization(self, parameterization: str) -> None:
        """Compute set parameterization and return the result.

        Args:
            parameterization (str): Input value.

        Returns:
            None: No value is returned.
        """
        self.parameterization = self._normalize_parameterization(parameterization)

    def _prediction_to_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Helper that computes prediction to noise.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            prediction (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if self.parameterization == "epsilon":
            return prediction

        b = x_t.shape[0]
        alpha_cumprod_sqrt_t = self.alphas_cumprod_sqrt[t].view(b, 1, 1, 1)
        alpha_one_minus_cumprod_sqrt_t = self.alphas_one_minus_cumprod_sqrt[t].view(
            b, 1, 1, 1
        )
        return (
            x_t - alpha_cumprod_sqrt_t * prediction
        ) / alpha_one_minus_cumprod_sqrt_t
