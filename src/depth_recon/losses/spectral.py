"""GLORYS spectral energy floor prior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from depth_recon.paths import resolve_config_path


def _load_spectral_reference(path: str | Path) -> torch.Tensor:
    """Load radial-band reference energy from a torch stats file."""
    payload: Any = torch.load(resolve_config_path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Spectral reference file {path} must contain a mapping.")
    for key in ("power_ref", "energy_ref", "band_energy_ref", "spectral_energy_ref"):
        if key in payload:
            tensor = torch.as_tensor(payload[key], dtype=torch.float32).detach()
            if tensor.ndim not in {1, 2}:
                raise RuntimeError(f"{key} must be shaped [bands] or [C,bands].")
            return tensor
    raise RuntimeError(
        f"Spectral reference file {path} must contain one of: "
        "power_ref, energy_ref, band_energy_ref, spectral_energy_ref."
    )


class SpectralEnergyFloorLoss(nn.Module):
    """Penalize generated spectra only when below GLORYS reference energy."""

    def __init__(
        self,
        *,
        reference_path: str | Path,
        eps: float = 1e-8,
        margin: float = 0.0,
        min_band: int = 1,
        max_band: int | None = None,
        per_depth: bool = True,
    ) -> None:
        """Initialize the spectral floor prior from precomputed radial bands."""
        super().__init__()
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "spectral_energy_floor.reference_path is required when enabled."
            )
        energy_ref = _load_spectral_reference(reference_path)
        self.register_buffer("energy_ref", energy_ref, persistent=False)
        self.eps = float(eps)
        self.margin = float(margin)
        self.min_band = max(0, int(min_band))
        self.max_band = None if max_band is None else max(0, int(max_band))
        self.per_depth = bool(per_depth)

    @staticmethod
    def _radial_band_ids(height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return integer radial FFT band ids for an ``H x W`` grid."""
        fy = torch.fft.fftfreq(height, device=device) * height
        fx = torch.fft.fftfreq(width, device=device) * width
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        return torch.sqrt(yy.square() + xx.square()).floor().long()

    def _radial_power(self, x0_pred: torch.Tensor, num_bands: int) -> torch.Tensor:
        """Compute radial-band FFT power as ``[C,bands]`` or ``[bands]``."""
        bsz, channels, height, width = x0_pred.shape
        spectrum = torch.fft.fft2(x0_pred, dim=(-2, -1))
        power = spectrum.abs().square().mean(dim=0)
        band_ids = self._radial_band_ids(height, width, x0_pred.device)
        valid = band_ids < num_bands
        flat_ids = band_ids[valid].reshape(-1)
        flat_power = power[:, valid].reshape(channels, -1)
        sums = torch.zeros(
            channels,
            num_bands,
            device=x0_pred.device,
            dtype=flat_power.dtype,
        )
        counts = torch.zeros_like(sums)
        scatter_idx = flat_ids.unsqueeze(0).expand(channels, -1)
        sums.scatter_add_(1, scatter_idx, flat_power)
        counts.scatter_add_(1, scatter_idx, torch.ones_like(flat_power))
        radial = sums / counts.clamp_min(1.0)
        if self.per_depth:
            return radial
        return radial.mean(dim=0)

    def forward(self, x0_pred: torch.Tensor) -> torch.Tensor:
        """Compute the lower-bound spectral hinge loss."""
        if x0_pred.ndim != 4:
            raise RuntimeError(
                f"x0_pred must be shaped [B,C,H,W], got {tuple(x0_pred.shape)}."
            )
        ref = self.energy_ref.to(device=x0_pred.device, dtype=x0_pred.dtype)
        num_bands = int(ref.shape[-1])
        radial = self._radial_power(x0_pred, num_bands)
        if radial.ndim == 2 and ref.ndim == 1:
            ref = ref.unsqueeze(0).expand_as(radial)
        if radial.ndim == 1 and ref.ndim == 2:
            ref = ref.mean(dim=0)
        if tuple(ref.shape) != tuple(radial.shape):
            raise RuntimeError(
                f"Spectral reference shape {tuple(ref.shape)} is incompatible with "
                f"generated radial power {tuple(radial.shape)}."
            )
        max_band = (
            num_bands if self.max_band is None else min(num_bands, self.max_band + 1)
        )
        band_mask = torch.zeros(num_bands, device=x0_pred.device, dtype=torch.bool)
        band_mask[self.min_band : max_band] = True
        while band_mask.ndim < radial.ndim:
            band_mask = band_mask.unsqueeze(0)
        valid = band_mask.expand_as(radial) & torch.isfinite(ref) & (ref > 0)
        if not bool(valid.any()):
            return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
        hinge = F.relu(
            torch.log(ref + self.eps) - torch.log(radial + self.eps) - self.margin
        )
        return hinge[valid].mean()


class PairedSpectralEnergyFloorLoss(nn.Module):
    """Penalize prediction spectra when below paired GLORYS target spectra."""

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        margin: float = 0.0,
        min_band: int = 1,
        max_band: int | None = None,
        per_depth: bool = True,
    ) -> None:
        """Initialize same-sample paired spectral floor comparison."""
        super().__init__()
        self.eps = float(eps)
        self.margin = float(margin)
        self.min_band = max(0, int(min_band))
        self.max_band = None if max_band is None else max(0, int(max_band))
        self.per_depth = bool(per_depth)

    @staticmethod
    def _radial_band_ids(height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return integer radial FFT band ids for an ``H x W`` grid."""
        fy = torch.fft.fftfreq(height, device=device) * height
        fx = torch.fft.fftfreq(width, device=device) * width
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        return torch.sqrt(yy.square() + xx.square()).floor().long()

    @staticmethod
    def _expand_valid_mask(
        valid_mask: torch.Tensor | None,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return a support mask broadcastable to the field tensor."""
        if valid_mask is None:
            return None
        mask = valid_mask.to(device=reference.device, dtype=reference.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1:
            mask = mask.expand(-1, reference.size(1), -1, -1)
        return mask

    def _radial_power(self, field: torch.Tensor) -> torch.Tensor:
        """Compute radial-band FFT power as ``[C,bands]`` or ``[bands]``."""
        _, channels, height, width = field.shape
        band_ids = self._radial_band_ids(height, width, field.device)
        num_bands = int(band_ids.max().item()) + 1
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        power = spectrum.abs().square().mean(dim=0)
        flat_ids = band_ids.reshape(-1)
        flat_power = power.reshape(channels, -1)
        sums = torch.zeros(
            channels,
            num_bands,
            device=field.device,
            dtype=flat_power.dtype,
        )
        counts = torch.zeros_like(sums)
        scatter_idx = flat_ids.unsqueeze(0).expand(channels, -1)
        sums.scatter_add_(1, scatter_idx, flat_power)
        counts.scatter_add_(1, scatter_idx, torch.ones_like(flat_power))
        radial = sums / counts.clamp_min(1.0)
        if self.per_depth:
            return radial
        return radial.mean(dim=0)

    def forward(
        self,
        x0_pred: torch.Tensor,
        target_grid: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute paired lower-bound spectral hinge loss."""
        if x0_pred.ndim != 4:
            raise RuntimeError(
                f"x0_pred must be shaped [B,C,H,W], got {tuple(x0_pred.shape)}."
            )
        if target_grid.shape != x0_pred.shape:
            raise RuntimeError(
                f"target_grid shape {tuple(target_grid.shape)} must match x0_pred "
                f"{tuple(x0_pred.shape)}."
            )
        target_grid = target_grid.to(device=x0_pred.device, dtype=x0_pred.dtype)
        support = self._expand_valid_mask(valid_mask, reference=x0_pred)
        if support is not None:
            # Apply identical support before FFT so prediction and GLORYS see the same domain.
            x0_pred = x0_pred * support
            target_grid = target_grid * support
        radial = self._radial_power(x0_pred)
        ref = self._radial_power(target_grid)
        if tuple(ref.shape) != tuple(radial.shape):
            raise RuntimeError(
                f"Paired spectral reference shape {tuple(ref.shape)} is incompatible "
                f"with prediction radial power {tuple(radial.shape)}."
            )
        num_bands = int(radial.shape[-1])
        max_band = (
            num_bands if self.max_band is None else min(num_bands, self.max_band + 1)
        )
        band_mask = torch.zeros(num_bands, device=x0_pred.device, dtype=torch.bool)
        band_mask[self.min_band : max_band] = True
        while band_mask.ndim < radial.ndim:
            band_mask = band_mask.unsqueeze(0)
        valid = band_mask.expand_as(radial) & torch.isfinite(ref) & (ref > 0)
        if not bool(valid.any()):
            return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
        hinge = F.relu(
            torch.log(ref + self.eps) - torch.log(radial + self.eps) - self.margin
        )
        return hinge[valid].mean()
