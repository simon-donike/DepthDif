"""

This file contains implementations of the forward diffusion process

Current Models:

1) Gaussian Diffusion

"""

from __future__ import annotations

import torch
from torch import nn

from .beta_schedules import *


class ForwardModel(nn.Module):
    """Base interface for forward diffusion process implementations."""

    def __init__(self, num_timesteps: int = 1000, schedule: str = "linear") -> None:

        """Initialize ForwardModel with configured parameters.

        Args:
            num_timesteps (int): Step or timestep value.
            schedule (str): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.schedule = schedule
        self.num_timesteps = num_timesteps

    @torch.no_grad()
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the module forward computation.

        Args:
            x_0 (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        raise NotImplemented

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run one update step.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        raise NotImplemented


class GaussianForwardProcess(ForwardModel):
    """Forward diffusion process based on Gaussian noise transitions."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:

        """Initialize GaussianForwardProcess with configured parameters.

        Args:
            num_timesteps (int): Step or timestep value.
            schedule (str): Input value.
            beta_start (float): Input value.
            beta_end (float): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__(num_timesteps=num_timesteps, schedule=schedule)

        # get process parameters
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

    @torch.no_grad()
    def forward(
        self, x_0: torch.Tensor, t: torch.Tensor, return_noise: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run reverse diffusion and return generated outputs.

        Args:
            x_0 (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            return_noise (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Tensor output produced by this call.
        """
        assert (t < self.num_timesteps).all()

        b = x_0.shape[0]
        mean = x_0 * self.alphas_cumprod_sqrt[t].view(b, 1, 1, 1)
        std = self.alphas_one_minus_cumprod_sqrt[t].view(b, 1, 1, 1)

        noise = torch.randn_like(x_0)
        output = mean + std * noise

        if not return_noise:
            return output
        else:
            return output, noise

    @torch.no_grad()
    def step(
        self, x_t: torch.Tensor, t: torch.Tensor, return_noise: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply one forward-diffusion transition step.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            return_noise (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Tensor output produced by this call.
        """
        assert (t < self.num_timesteps).all()

        mean = self.alphas_sqrt[t] * x_t
        std = self.betas_sqrt[t]

        noise = torch.randn_like(x_t)
        output = mean + std * noise

        if not return_noise:
            return output
        else:
            return output, noise
