from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *
from utils.validation_denoise import build_capture_indices


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


def _set_sampler_parameterization(sampler: nn.Module, parameterization: str) -> None:
    """Helper that computes set sampler parameterization.

    Args:
        sampler (nn.Module): Sampler instance used for reverse diffusion.
        parameterization (str): Input value.

    Returns:
        None: No value is returned.
    """
    if hasattr(sampler, "set_parameterization"):
        sampler.set_parameterization(parameterization)
        return
    if hasattr(sampler, "parameterization"):
        sampler.parameterization = parameterization


class DenoisingDiffusionProcess(nn.Module):

    """Unconditional diffusion process module for training and sampling."""
    def __init__(
        self,
        generated_channels: int = 3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_timesteps: int = 1000,
        unet_dim: int = 64,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        unet_with_time_emb: bool = True,
        unet_output_mean_scale: bool = False,
        unet_residual: bool = False,
        parameterization: str = "epsilon",
        sampler: nn.Module | None = None,
    ) -> None:
        """Initialize DenoisingDiffusionProcess with configured parameters.

        Args:
            generated_channels (int): Input value.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Tensor input for the computation.
            schedule (str): Input value.
            beta_start (float): Input value.
            beta_end (float): Input value.
            num_timesteps (int): Step or timestep value.
            unet_dim (int): Input value.
            unet_dim_mults (tuple[int, ...]): Input value.
            unet_with_time_emb (bool): Boolean flag controlling behavior.
            unet_output_mean_scale (bool): Boolean flag controlling behavior.
            unet_residual (bool): Boolean flag controlling behavior.
            parameterization (str): Input value.
            sampler (nn.Module | None): Sampler instance used for reverse diffusion.

        Returns:
            None: No value is returned.
        """
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn
        self.parameterization = _normalize_parameterization(parameterization)

        # Forward Process Used for Training
        self.forward_process = GaussianForwardProcess(
            num_timesteps=self.num_timesteps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.model = UnetConvNextBlock(
            dim=unet_dim,
            dim_mults=unet_dim_mults,
            channels=self.generated_channels,
            out_dim=self.generated_channels,
            with_time_emb=unet_with_time_emb,
            output_mean_scale=unet_output_mean_scale,
            residual=unet_residual,
        )

        # defaults to a DDPM sampler if None is provided
        self.sampler = (
            DDPM_Sampler(
                num_timesteps=self.num_timesteps,
                schedule=schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                parameterization=self.parameterization,
            )
            if sampler is None
            else sampler
        )

    @torch.no_grad()
    def forward(
        self,
        shape: tuple[int, int] = (256, 256),
        batch_size: int = 1,
        sampler: nn.Module | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Run reverse diffusion and return generated outputs.

        Args:
            shape (tuple[int, int]): Input value.
            batch_size (int): Size/count parameter.
            sampler (nn.Module | None): Sampler instance used for reverse diffusion.
            verbose (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """

        # read dimensions
        b, c, h, w = batch_size, self.generated_channels, *shape
        device = next(self.model.parameters()).device

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)
        _set_sampler_parameterization(sampler, self.parameterization)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        x_t = torch.randn([b, self.generated_channels, h, w], device=device)

        for i in (
            tqdm(it, desc="diffusion sampling", total=num_timesteps) if verbose else it
        ):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            prediction = self.model(x_t, t)
            x_t = sampler(x_t, t, prediction)

        return x_t

    def p_loss(self, output: torch.Tensor) -> torch.Tensor:
        """Compute the diffusion training loss for the current batch.

        Args:
            output (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """

        b, c, h, w = output.shape
        device = output.device

        # loss for training

        # input is the optional condition
        t = torch.randint(
            0, self.forward_process.num_timesteps, (b,), device=device
        ).long()
        output_noisy, noise = self.forward_process(output, t, return_noise=True)

        # reverse pass
        prediction = self.model(output_noisy, t)

        # apply loss
        target = noise if self.parameterization == "epsilon" else output
        return self.loss_fn(target, prediction)


class DenoisingDiffusionConditionalProcess(nn.Module):

    """Conditional diffusion process module for guided reconstruction."""
    def __init__(
        self,
        generated_channels: int = 3,
        condition_channels: int = 3,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_timesteps: int = 1000,
        unet_dim: int = 64,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        unet_with_time_emb: bool = True,
        unet_output_mean_scale: bool = False,
        unet_residual: bool = False,
        coord_conditioning_enabled: bool = False,
        coord_encoding: str = "unit_sphere",
        date_conditioning_enabled: bool = False,
        date_encoding: str = "day_of_year_sincos",
        coord_embed_dim: int | None = None,
        parameterization: str = "epsilon",
        sampler: nn.Module | None = None,
    ) -> None:
        """Initialize DenoisingDiffusionConditionalProcess with configured parameters.

        Args:
            generated_channels (int): Input value.
            condition_channels (int): Input value.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Tensor input for the computation.
            schedule (str): Input value.
            beta_start (float): Input value.
            beta_end (float): Input value.
            num_timesteps (int): Step or timestep value.
            unet_dim (int): Input value.
            unet_dim_mults (tuple[int, ...]): Input value.
            unet_with_time_emb (bool): Boolean flag controlling behavior.
            unet_output_mean_scale (bool): Boolean flag controlling behavior.
            unet_residual (bool): Boolean flag controlling behavior.
            coord_conditioning_enabled (bool): Boolean flag controlling behavior.
            coord_encoding (str): Input value.
            date_conditioning_enabled (bool): Boolean flag controlling behavior.
            date_encoding (str): Input value.
            coord_embed_dim (int | None): Input value.
            parameterization (str): Input value.
            sampler (nn.Module | None): Sampler instance used for reverse diffusion.

        Returns:
            None: No value is returned.
        """
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn
        self.parameterization = _normalize_parameterization(parameterization)
        self.coord_conditioning_enabled = bool(coord_conditioning_enabled)
        self.coord_encoding = str(coord_encoding).strip().lower()
        self.date_conditioning_enabled = bool(date_conditioning_enabled)
        self.date_encoding = str(date_encoding).strip().lower()
        self.coord_embed_dim = None
        self.coord_mlp = None
        if self.date_conditioning_enabled and not self.coord_conditioning_enabled:
            raise ValueError(
                "date conditioning requires coord_conditioning.enabled=true."
            )
        if self.coord_conditioning_enabled:
            valid_encodings = {"unit_sphere", "sincos", "raw"}
            if self.coord_encoding not in valid_encodings:
                raise ValueError(
                    "coord_encoding must be one of "
                    f"{sorted(valid_encodings)} (got '{self.coord_encoding}')."
                )
            if self.date_conditioning_enabled:
                valid_date_encodings = {"day_of_year_sincos"}
                if self.date_encoding not in valid_date_encodings:
                    raise ValueError(
                        "date_encoding must be one of "
                        f"{sorted(valid_date_encodings)} (got '{self.date_encoding}')."
                    )
            if coord_embed_dim is None:
                coord_embed_dim = unet_dim
            self.coord_embed_dim = int(coord_embed_dim)
            enc_dim = self._coord_encoding_dim(self.coord_encoding)
            if self.date_conditioning_enabled:
                enc_dim += self._date_encoding_dim(self.date_encoding)
            self.coord_mlp = nn.Sequential(
                nn.Linear(enc_dim, self.coord_embed_dim * 4),
                nn.GELU(),
                nn.Linear(self.coord_embed_dim * 4, self.coord_embed_dim),
            )

        # Forward Process
        self.forward_process = GaussianForwardProcess(
            num_timesteps=self.num_timesteps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Neural Network Backbone
        self.model = UnetConvNextBlock(
            dim=unet_dim,
            dim_mults=unet_dim_mults,
            channels=self.generated_channels + condition_channels,
            out_dim=self.generated_channels,
            with_time_emb=unet_with_time_emb,
            coord_emb_dim=self.coord_embed_dim if self.coord_mlp is not None else None,
            output_mean_scale=unet_output_mean_scale,
            residual=unet_residual,
        )

        # defaults to a DDPM sampler if None is provided
        self.sampler = (
            DDPM_Sampler(
                num_timesteps=self.num_timesteps,
                schedule=schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                parameterization=self.parameterization,
            )
            if sampler is None
            else sampler
        )

    @torch.no_grad()
    def forward(
        self,
        condition: torch.Tensor,
        sampler: nn.Module | None = None,
        verbose: bool = False,
        known_mask: torch.Tensor | None = None,
        known_values: torch.Tensor | None = None,
        coord: torch.Tensor | None = None,
        date: torch.Tensor | None = None,
        return_intermediates: bool = False,
        intermediate_step_indices: list[int] | None = None,
        return_x0_intermediates: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]
        | tuple[
            torch.Tensor,
            list[tuple[int, torch.Tensor]],
            list[tuple[int, torch.Tensor]],
        ]
    ):
        """Run reverse diffusion and return generated outputs.

        Args:
            condition (torch.Tensor): Tensor input for the computation.
            sampler (nn.Module | None): Sampler instance used for reverse diffusion.
            verbose (bool): Boolean flag controlling behavior.
            known_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            known_values (torch.Tensor | None): Tensor input for the computation.
            coord (torch.Tensor | None): Coordinate conditioning values.
            date (torch.Tensor | None): Date conditioning values.
            return_intermediates (bool): Boolean flag controlling behavior.
            intermediate_step_indices (list[int] | None): Input value.
            return_x0_intermediates (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]] | tuple[torch.Tensor, list[tuple[int, torch.Tensor]], list[tuple[int, torch.Tensor]]]: Tensor output produced by this call.
        """

        # read dimensions
        b, c, h, w = condition.shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        coord_emb = self._maybe_embed_coords(coord, condition, date=date)

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)
        _set_sampler_parameterization(sampler, self.parameterization)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        x_t = torch.randn([b, self.generated_channels, h, w], device=device)
        intermediates: list[tuple[int, torch.Tensor]] = []
        x0_intermediates: list[tuple[int, torch.Tensor]] = []
        capture_indices: set[int] = set()
        if return_intermediates:
            capture_indices = build_capture_indices(
                total_steps=int(num_timesteps),
                intermediate_step_indices=intermediate_step_indices,
            )

        apply_known = False
        if known_mask is not None and known_values is not None:
            known_mask = known_mask.to(device=device, dtype=x_t.dtype)
            known_values = known_values.to(device=device, dtype=x_t.dtype)
            if known_mask.ndim == 3:
                known_mask = known_mask.unsqueeze(1)
            if known_mask.ndim == 4:
                # Preserve per-channel masks when they already match generated channels.
                if known_mask.size(1) == x_t.size(1):
                    pass
                elif known_mask.size(1) == 1 and x_t.size(1) > 1:
                    known_mask = known_mask.expand(-1, x_t.size(1), -1, -1)
                else:
                    known_mask = known_mask.amax(dim=1, keepdim=True)
                    if x_t.size(1) > 1:
                        known_mask = known_mask.expand(-1, x_t.size(1), -1, -1)
            if known_values.ndim == 3:
                known_values = known_values.unsqueeze(1)
            if (
                known_values.ndim == 4
                and known_values.size(1) != x_t.size(1)
                and known_values.size(1) == 1
            ):
                known_values = known_values.repeat(1, x_t.size(1), 1, 1)
            if (
                known_mask.shape[:1] == x_t.shape[:1]
                and known_mask.shape[2:] == x_t.shape[2:]
                and known_values.shape == x_t.shape
            ):
                apply_known = True

        if apply_known:
            x_t = x_t * (1.0 - known_mask) + known_values * known_mask
        if return_intermediates and 0 in capture_indices:
            intermediates.append((0, x_t.detach().clone()))

        for step_index, i in enumerate(
            tqdm(it, desc="diffusion sampling", total=num_timesteps) if verbose else it
        ):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_input = torch.cat([x_t, condition], 1).to(device)
            prediction = self.model(model_input, t, coord_emb=coord_emb)
            x0_pred = self._prediction_to_x0(
                x_t=x_t,
                t=t,
                prediction=prediction,
                sampler=sampler,
            )
            x_t = sampler(x_t, t, prediction)
            if apply_known:
                x_t = x_t * (1.0 - known_mask) + known_values * known_mask
            if return_intermediates:
                capture_step = step_index + 1
                if capture_step in capture_indices:
                    intermediates.append((capture_step, x_t.detach().clone()))
                    if return_x0_intermediates:
                        x0_intermediates.append((capture_step, x0_pred.detach().clone()))

        if return_intermediates:
            if return_x0_intermediates:
                return x_t, intermediates, x0_intermediates
            return x_t, intermediates
        return x_t

    @staticmethod
    def _sampler_train_timestep(sampler: nn.Module, t: torch.Tensor) -> torch.Tensor:
        """Helper that computes sampler train timestep.

        Args:
            sampler (nn.Module): Sampler instance used for reverse diffusion.
            t (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        t_long = t.long()
        if hasattr(sampler, "ddim_train_steps"):
            ddim_train_steps = getattr(sampler, "ddim_train_steps")
            if torch.is_tensor(ddim_train_steps):
                return ddim_train_steps[t_long]
        return t_long

    def _prediction_to_x0(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor,
        sampler: nn.Module,
    ) -> torch.Tensor:
        """Helper that computes prediction to x0.

        Args:
            x_t (torch.Tensor): Tensor input for the computation.
            t (torch.Tensor): Tensor input for the computation.
            prediction (torch.Tensor): Tensor input for the computation.
            sampler (nn.Module): Sampler instance used for reverse diffusion.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if self.parameterization == "x0":
            return prediction

        if not hasattr(sampler, "alphas_cumprod"):
            return prediction

        b = x_t.shape[0]
        train_t = self._sampler_train_timestep(sampler, t)
        alpha_cumprod = sampler.alphas_cumprod[train_t].view(b, 1, 1, 1)
        alpha_cumprod_sqrt = torch.clamp(alpha_cumprod, min=1e-20).sqrt()
        one_minus_alpha_sqrt = torch.clamp(1.0 - alpha_cumprod, min=1e-20).sqrt()
        return (x_t - one_minus_alpha_sqrt * prediction) / alpha_cumprod_sqrt

    @staticmethod
    def _build_valid_mask(
        valid_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Helper that computes build valid mask.

        Args:
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            reference (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor | None: Tensor output produced by this call.
        """
        if valid_mask is None:
            return None
        mask = (valid_mask > 0.5).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim == 4 and reference.ndim == 4:
            if mask.size(1) == reference.size(1):
                pass
            elif mask.size(1) == 1 and reference.size(1) > 1:
                mask = mask.expand(-1, reference.size(1), -1, -1)
            elif reference.size(1) == 1 and mask.size(1) > 1:
                mask = mask.amax(dim=1, keepdim=True)
            else:
                mask = mask.amax(dim=1, keepdim=True)
                if reference.size(1) > 1:
                    mask = mask.expand(-1, reference.size(1), -1, -1)
        mask = 1.0 - mask  # now 1 = missing
        mask = mask.clamp(0.0, 1.0)
        return mask.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _build_land_mask(
        land_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Helper that computes build land mask.

        Args:
            land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            reference (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor | None: Tensor output produced by this call.
        """
        if land_mask is None:
            return None
        mask = (land_mask > 0.5).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim == 4 and reference.ndim == 4:
            if mask.size(1) == reference.size(1):
                pass
            elif mask.size(1) == 1 and reference.size(1) > 1:
                mask = mask.expand(-1, reference.size(1), -1, -1)
            elif reference.size(1) == 1 and mask.size(1) > 1:
                mask = mask.amax(dim=1, keepdim=True)
            else:
                mask = mask.amax(dim=1, keepdim=True)
                if reference.size(1) > 1:
                    mask = mask.expand(-1, reference.size(1), -1, -1)
        mask = mask.clamp(0.0, 1.0)
        return mask.to(device=reference.device, dtype=reference.dtype)

    def p_loss(
        self,
        output: torch.Tensor,
        condition: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        land_mask: torch.Tensor | None = None,
        mask_loss: bool = False,
        coord: torch.Tensor | None = None,
        date: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the diffusion training loss for the current batch.

        Args:
            output (torch.Tensor): Tensor input for the computation.
            condition (torch.Tensor): Tensor input for the computation.
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            mask_loss (bool): Mask tensor controlling valid or known pixels.
            coord (torch.Tensor | None): Coordinate conditioning values.
            date (torch.Tensor | None): Date conditioning values.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """

        b, c, h, w = output.shape
        device = output.device

        # loss for training

        # input is the optional condition
        t = torch.randint(
            0, self.forward_process.num_timesteps, (b,), device=device
        ).long()
        output_noisy, noise = self.forward_process(output, t, return_noise=True)

        # reverse pass
        model_input = torch.cat([output_noisy, condition], 1).to(device)
        coord_emb = self._maybe_embed_coords(coord, model_input, date=date)
        prediction = self.model(model_input, t, coord_emb=coord_emb)
        target = noise if self.parameterization == "epsilon" else output

        # apply loss
        if not mask_loss:
            return self.loss_fn(target, prediction)

        mask = self._build_valid_mask(valid_mask, target)
        if mask is None:
            return self.loss_fn(target, prediction)
        ocean_mask = self._build_land_mask(land_mask, target)
        if ocean_mask is not None:
            mask = mask * ocean_mask

        # manually computes MSE loss over masked pixels
        diff = (target - prediction) ** 2
        masked_diff = diff * mask
        denom = mask.sum()
        if denom.item() <= 0:
            return torch.zeros((), device=diff.device, dtype=diff.dtype)
        return masked_diff.sum() / denom

    @staticmethod
    def _coord_encoding_dim(encoding: str) -> int:
        """Helper that computes coord encoding dim.

        Args:
            encoding (str): Input value.

        Returns:
            int: Computed scalar output.
        """
        if encoding == "unit_sphere":
            return 3
        if encoding == "sincos":
            return 4
        if encoding == "raw":
            return 2
        raise ValueError(f"Unsupported coord encoding '{encoding}'.")

    @staticmethod
    def _date_encoding_dim(encoding: str) -> int:
        """Helper that computes date encoding dim.

        Args:
            encoding (str): Input value.

        Returns:
            int: Computed scalar output.
        """
        if encoding == "day_of_year_sincos":
            return 2
        raise ValueError(f"Unsupported date encoding '{encoding}'.")

    def _encode_coords(self, coord: torch.Tensor) -> torch.Tensor:
        """Helper that computes encode coords.

        Args:
            coord (torch.Tensor): Coordinate conditioning values.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if coord.ndim != 2 or coord.size(1) != 2:
            raise ValueError(
                "coords must have shape (B, 2) with [lat, lon] in degrees."
            )
        lat = coord[:, 0]
        lon = coord[:, 1]
        deg2rad = math.pi / 180.0
        lat_rad = lat * deg2rad
        lon_rad = lon * deg2rad
        if self.coord_encoding == "unit_sphere":
            cos_lat = torch.cos(lat_rad)
            x = cos_lat * torch.cos(lon_rad)
            y = cos_lat * torch.sin(lon_rad)
            z = torch.sin(lat_rad)
            return torch.stack([x, y, z], dim=1)
        if self.coord_encoding == "sincos":
            return torch.stack(
                [
                    torch.sin(lat_rad),
                    torch.cos(lat_rad),
                    torch.sin(lon_rad),
                    torch.cos(lon_rad),
                ],
                dim=1,
            )
        if self.coord_encoding == "raw":
            lat_norm = lat / 90.0
            lon_norm = lon / 180.0
            return torch.stack([lat_norm, lon_norm], dim=1)
        raise ValueError(f"Unsupported coord encoding '{self.coord_encoding}'.")

    def _encode_date(self, date: torch.Tensor) -> torch.Tensor:
        """Helper that computes encode date.

        Args:
            date (torch.Tensor): Date conditioning values.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if date.ndim == 2 and date.size(1) == 1:
            date = date[:, 0]
        if date.ndim != 1:
            raise ValueError("date must have shape (B,) or (B, 1) as YYYYMMDD integers.")

        date_int = torch.round(date).to(dtype=torch.long)
        month = (date_int // 100) % 100
        day = date_int % 100

        month_lengths = torch.tensor(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
            device=date_int.device,
            dtype=torch.long,
        )
        valid_month = (month >= 1) & (month <= 12)
        clamped_month = torch.clamp(month, min=1, max=12)
        max_day = month_lengths[clamped_month - 1]
        valid_day = (day >= 1) & (day <= max_day)
        if not bool(torch.all(valid_month & valid_day)):
            raise ValueError("date values must be valid YYYYMMDD integers.")

        month_offsets = torch.tensor(
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
            device=date_int.device,
            dtype=torch.long,
        )
        # Use a fixed non-leap calendar to keep day-of-year encoding stable year-to-year.
        day_of_year = month_offsets[month - 1] + day
        phase = (2.0 * math.pi * day_of_year.to(dtype=torch.float32)) / 365.0
        if self.date_encoding == "day_of_year_sincos":
            return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
        raise ValueError(f"Unsupported date encoding '{self.date_encoding}'.")

    def _maybe_embed_coords(
        self,
        coord: torch.Tensor | None,
        reference: torch.Tensor,
        *,
        date: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Helper that computes maybe embed coords.

        Args:
            coord (torch.Tensor | None): Coordinate conditioning values.
            reference (torch.Tensor): Tensor input for the computation.
            date (torch.Tensor | None): Date conditioning values.

        Returns:
            torch.Tensor | None: Tensor output produced by this call.
        """
        if self.coord_mlp is None:
            return None
        if coord is None:
            raise ValueError(
                "coord_conditioning is enabled but no coords were provided."
            )
        coord = coord.to(device=reference.device, dtype=reference.dtype)
        enc = self._encode_coords(coord)
        if self.date_conditioning_enabled:
            if date is None:
                raise ValueError(
                    "date conditioning is enabled but no date values were provided."
                )
            if not torch.is_tensor(date):
                date = torch.as_tensor(date, device=reference.device)
            date = date.to(device=reference.device)
            date_enc = self._encode_date(date).to(dtype=reference.dtype)
            enc = torch.cat([enc, date_enc], dim=1)
        return self.coord_mlp(enc)
