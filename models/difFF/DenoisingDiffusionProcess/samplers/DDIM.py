"""

This file contains the DDIM sampler class for a diffusion process

"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..beta_schedules import *


class DDIM_Sampler(nn.Module):

    """DDIM sampler that performs accelerated reverse-diffusion updates."""
    def __init__(
        self,
        num_timesteps: int = 100,
        train_timesteps: int = 1000,
        clip_sample: bool = True,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        eta: float = 0.0,
        betas: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        parameterization: str = "epsilon",
    ) -> None:

        """Initialize DDIM_Sampler with configured parameters.

        Args:
            num_timesteps (int): Step or timestep value.
            train_timesteps (int): Step or timestep value.
            clip_sample (bool): Boolean flag controlling behavior.
            schedule (str): Input value.
            beta_start (float): Input value.
            beta_end (float): Input value.
            eta (float): Input value.
            betas (torch.Tensor | list[float] | tuple[float, ...] | None): Tensor input for the computation.
            parameterization (str): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()

        self.num_timesteps = int(num_timesteps)
        self.train_timesteps = int(train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.schedule = schedule
        self.eta = float(eta)
        self.parameterization = self._normalize_parameterization(parameterization)
        self.final_alpha_cumprod = torch.tensor(1.0)

        if betas is not None:
            betas_tensor = torch.as_tensor(betas, dtype=torch.float32)
            if betas_tensor.ndim != 1:
                raise ValueError("DDIM_Sampler betas must be a 1D tensor.")
            self.train_timesteps = int(betas_tensor.numel())
            self.register_buffer("betas", betas_tensor)
        else:
            self.register_buffer(
                "betas",
                get_beta_schedule(
                    self.schedule,
                    self.train_timesteps,
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
        # DDIM inference indices mapped into the training diffusion timeline.
        ddim_steps = (
            torch.linspace(0, self.train_timesteps - 1, self.num_timesteps)
            .round()
            .long()
        )
        self.register_buffer("ddim_train_steps", ddim_steps)
        ddim_prev = torch.cat(
            [torch.tensor([-1], dtype=torch.long), ddim_steps[:-1]], dim=0
        )
        self.register_buffer("ddim_prev_steps", ddim_prev)

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

        b = z_t.shape[0]
        device = z_t.device

        ddim_t = t.long()
        train_t = self.ddim_train_steps[ddim_t]
        train_t_prev = self.ddim_prev_steps[ddim_t]

        alpha_cumprod_t = self.alphas_cumprod[train_t].view(b, 1, 1, 1)
        alpha_cumprod_prev = torch.where(
            train_t_prev >= 0,
            self.alphas_cumprod[train_t_prev],
            self.final_alpha_cumprod.to(device),
        ).view(b, 1, 1, 1)

        x_0_pred, noise_pred = self._prediction_to_x0_and_noise(
            x_t=x_t,
            train_t=train_t,
            prediction=z_t,
            alpha_cumprod_t=alpha_cumprod_t,
        )
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
            noise_pred = self._x0_to_noise(
                x_t=x_t,
                x0_pred=x_0_pred,
                alpha_cumprod_t=alpha_cumprod_t,
                train_t=train_t,
            )

        sigma_t = self.eta * self.estimate_std(alpha_cumprod_t, alpha_cumprod_prev)
        dir_xt = (
            torch.clamp(1 - alpha_cumprod_prev - sigma_t**2, min=0.0).sqrt()
            * noise_pred
        )
        prev_sample = alpha_cumprod_prev.sqrt() * x_0_pred + dir_xt

        if self.eta > 0:
            prev_sample = prev_sample + sigma_t * torch.randn_like(x_t)

        return prev_sample

    def estimate_std(
        self, alpha_cumprod: torch.Tensor, alpha_cumprod_prev: torch.Tensor
    ) -> torch.Tensor:
        """Compute estimate std and return the result.

        Args:
            alpha_cumprod (torch.Tensor): Tensor input for the computation.
            alpha_cumprod_prev (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod) * (
            1 - alpha_cumprod / alpha_cumprod_prev
        )

        return var.sqrt()

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

    def _x0_to_noise(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        alpha_cumprod_t: torch.Tensor,
        train_t: torch.Tensor,
    ) -> torch.Tensor:
        """Helper that computes x0 to noise.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            x0_pred (torch.Tensor): Tensor input for the computation.
            alpha_cumprod_t (torch.Tensor): Tensor input for the computation.
            train_t (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        b = x_t.shape[0]
        one_minus_alpha_cumprod_sqrt_t = self.alphas_one_minus_cumprod_sqrt[
            train_t
        ].view(b, 1, 1, 1)
        return (
            x_t - alpha_cumprod_t.sqrt() * x0_pred
        ) / one_minus_alpha_cumprod_sqrt_t

    def _prediction_to_x0_and_noise(
        self,
        x_t: torch.Tensor,
        train_t: torch.Tensor,
        prediction: torch.Tensor,
        alpha_cumprod_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper that computes prediction to x0 and noise.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            train_t (torch.Tensor): Tensor input for the computation.
            prediction (torch.Tensor): Tensor input for the computation.
            alpha_cumprod_t (torch.Tensor): Tensor input for the computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing computed outputs.
        """
        b = x_t.shape[0]
        if self.parameterization == "epsilon":
            noise_pred = prediction
            x0_pred = (
                x_t
                - self.alphas_one_minus_cumprod_sqrt[train_t].view(b, 1, 1, 1)
                * noise_pred
            ) / alpha_cumprod_t.sqrt()
            return x0_pred, noise_pred

        x0_pred = prediction
        noise_pred = self._x0_to_noise(
            x_t=x_t,
            x0_pred=x0_pred,
            alpha_cumprod_t=alpha_cumprod_t,
            train_t=train_t,
        )
        return x0_pred, noise_pred
