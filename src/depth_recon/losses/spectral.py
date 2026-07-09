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
            dtype=x0_pred.dtype,
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
