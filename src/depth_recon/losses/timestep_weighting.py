"""Timestep-dependent auxiliary loss weighting."""

from __future__ import annotations

from typing import Any

import torch


def _clamp_weight(
    weight: torch.Tensor,
    *,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Clamp a timestep weight tensor to configured scalar bounds."""
    if max_weight < min_weight:
        raise ValueError("aux_timestep_weighting.max_weight must be >= min_weight.")
    return weight.clamp(min=float(min_weight), max=float(max_weight))


def aux_timestep_weight(
    config: dict[str, Any] | None,
    *,
    t: torch.Tensor | None,
    alphas_cumprod: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a scalar batch-mean auxiliary loss weight."""
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return torch.ones((), device=reference.device, dtype=reference.dtype)
    if t is None or alphas_cumprod is None:
        raise RuntimeError(
            "aux_timestep_weighting.enabled=true requires sampled timesteps and "
            "alphas_cumprod."
        )

    mode = str(cfg.get("mode", "snr")).strip().lower()
    t = t.to(device=reference.device).long()
    alpha_bar = alphas_cumprod.to(device=reference.device, dtype=reference.dtype)
    if alpha_bar.ndim != 1:
        raise RuntimeError("alphas_cumprod must be a 1D tensor.")
    if bool((t < 0).any()) or bool((t >= int(alpha_bar.numel())).any()):
        raise RuntimeError("sampled timesteps are outside alphas_cumprod.")

    min_weight = float(cfg.get("min_weight", 0.0))
    max_weight = float(cfg.get("max_weight", 1.0))
    if mode == "snr":
        gamma = float(cfg.get("snr_gamma", 5.0))
        if gamma <= 0.0:
            raise ValueError("aux_timestep_weighting.snr_gamma must be > 0.")
        selected = alpha_bar[t]
        snr = selected / torch.clamp(1.0 - selected, min=1.0e-12)
        weight = torch.minimum(snr, torch.full_like(snr, gamma)) / gamma
    elif mode == "linear":
        start = float(cfg.get("linear_start_weight", 0.0))
        end = float(cfg.get("linear_end_weight", 1.0))
        denom = max(1, int(alpha_bar.numel()) - 1)
        clean_fraction = 1.0 - t.to(dtype=reference.dtype) / float(denom)
        weight = start + clean_fraction * (end - start)
    else:
        raise ValueError(
            "aux_timestep_weighting.mode must be one of {'snr', 'linear'} "
            f"(got {mode!r})."
        )

    return _clamp_weight(
        weight.mean(),
        min_weight=min_weight,
        max_weight=max_weight,
    )
