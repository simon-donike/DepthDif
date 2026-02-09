import math
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


class DenoisingDiffusionProcess(nn.Module):

    def __init__(
        self,
        generated_channels=3,
        loss_fn=F.mse_loss,
        schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_timesteps=1000,
        unet_dim=64,
        unet_dim_mults=(1, 2, 4, 8),
        unet_with_time_emb=True,
        unet_output_mean_scale=False,
        unet_residual=False,
        sampler=None,
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn

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
            )
            if sampler is None
            else sampler
        )

    @torch.no_grad()
    def forward(self, shape=(256, 256), batch_size=1, sampler=None, verbose=False):
        """
        forward() function triggers a complete inference cycle

        A custom sampler can be provided as an argument!
        """

        # read dimensions
        b, c, h, w = batch_size, self.generated_channels, *shape
        device = next(self.model.parameters()).device

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        x_t = torch.randn([b, self.generated_channels, h, w], device=device)

        for i in (
            tqdm(it, desc="diffusion sampling", total=num_timesteps) if verbose else it
        ):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t = self.model(x_t, t)  # prediction of noise
            x_t = sampler(x_t, t, z_t)  # prediction of next state

        return x_t

    def p_loss(self, output):
        """
        Computes denoising objective in whatever normalized space the caller uses.
        (In this project, caller feeds standardized temperature targets.)
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
        noise_hat = self.model(output_noisy, t)

        # apply loss
        return self.loss_fn(noise, noise_hat)


class DenoisingDiffusionConditionalProcess(nn.Module):

    def __init__(
        self,
        generated_channels=3,
        condition_channels=3,
        loss_fn=F.mse_loss,
        schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_timesteps=1000,
        unet_dim=64,
        unet_dim_mults=(1, 2, 4, 8),
        unet_with_time_emb=True,
        unet_output_mean_scale=False,
        unet_residual=False,
        coord_conditioning_enabled=False,
        coord_encoding="unit_sphere",
        coord_embed_dim=None,
        sampler=None,
    ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn
        self.coord_conditioning_enabled = bool(coord_conditioning_enabled)
        self.coord_encoding = str(coord_encoding).strip().lower()
        self.coord_embed_dim = None
        self.coord_mlp = None
        if self.coord_conditioning_enabled:
            valid_encodings = {"unit_sphere", "sincos", "raw"}
            if self.coord_encoding not in valid_encodings:
                raise ValueError(
                    "coord_encoding must be one of "
                    f"{sorted(valid_encodings)} (got '{self.coord_encoding}')."
                )
            if coord_embed_dim is None:
                coord_embed_dim = unet_dim
            self.coord_embed_dim = int(coord_embed_dim)
            enc_dim = self._coord_encoding_dim(self.coord_encoding)
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
            )
            if sampler is None
            else sampler
        )

    @torch.no_grad()
    def forward(
        self,
        condition,
        sampler=None,
        verbose=False,
        known_mask: torch.Tensor | None = None,
        known_values: torch.Tensor | None = None,
        coord: torch.Tensor | None = None,
        return_intermediates: bool = False,
        intermediate_step_indices: list[int] | None = None,
    ):
        """
        forward() function triggers a complete inference cycle

        A custom sampler can be provided as an argument!
        """

        # read dimensions
        b, c, h, w = condition.shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        coord_emb = self._maybe_embed_coords(coord, condition)

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        x_t = torch.randn([b, self.generated_channels, h, w], device=device)
        intermediates: list[tuple[int, torch.Tensor]] = []
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
            if known_mask.ndim == 4 and known_mask.size(1) != 1:
                known_mask = known_mask.amax(dim=1, keepdim=True)
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
            z_t = self.model(model_input, t, coord_emb=coord_emb)  # prediction of noise
            x_t = sampler(x_t, t, z_t)  # prediction of next state
            if apply_known:
                x_t = x_t * (1.0 - known_mask) + known_values * known_mask
            if return_intermediates:
                capture_step = step_index + 1
                if capture_step in capture_indices:
                    intermediates.append((capture_step, x_t.detach().clone()))

        if return_intermediates:
            return x_t, intermediates
        return x_t

    @staticmethod
    def _build_valid_mask(
        valid_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        if valid_mask is None:
            return None
        mask = (valid_mask > 0.5).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim == 4 and mask.size(1) != 1:
            mask = mask.amax(dim=1, keepdim=True)
        if mask.ndim == 4 and mask.size(1) == 1 and reference.ndim == 4:
            if reference.size(1) > 1:
                mask = mask.expand(-1, reference.size(1), -1, -1)
        mask = 1.0 - mask  # now 1 = missing
        mask = mask.clamp_(0.0, 1.0)
        return mask.to(device=reference.device, dtype=reference.dtype)

    def p_loss(
        self,
        output,
        condition,
        *,
        valid_mask: torch.Tensor | None = None,
        mask_loss: bool = False,
        coord: torch.Tensor | None = None,
    ):
        """
        Computes conditional denoising objective in caller-provided normalized space.
        (In this project, output/condition data channels are standardized temperatures.)
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
        coord_emb = self._maybe_embed_coords(coord, model_input)
        noise_hat = self.model(model_input, t, coord_emb=coord_emb)

        # apply loss
        if not mask_loss:
            return self.loss_fn(noise, noise_hat)

        mask = self._build_valid_mask(valid_mask, noise)
        if mask is None:
            return self.loss_fn(noise, noise_hat)

        diff = (noise - noise_hat) ** 2
        masked_diff = diff * mask
        denom = mask.sum()
        if denom.item() <= 0:
            return torch.zeros((), device=diff.device, dtype=diff.dtype)
        return masked_diff.sum() / denom

    @staticmethod
    def _coord_encoding_dim(encoding: str) -> int:
        if encoding == "unit_sphere":
            return 3
        if encoding == "sincos":
            return 4
        if encoding == "raw":
            return 2
        raise ValueError(f"Unsupported coord encoding '{encoding}'.")

    def _encode_coords(self, coord: torch.Tensor) -> torch.Tensor:
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

    def _maybe_embed_coords(
        self, coord: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        if self.coord_mlp is None:
            return None
        if coord is None:
            raise ValueError(
                "coord_conditioning is enabled but no coords were provided."
            )
        coord = coord.to(device=reference.device, dtype=reference.dtype)
        enc = self._encode_coords(coord)
        return self.coord_mlp(enc)
