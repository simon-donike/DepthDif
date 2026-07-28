"""Sparse observation consistency loss."""

from __future__ import annotations

import torch

from .robust import charbonnier_loss


def _flatten_spatial_field(x: torch.Tensor) -> torch.Tensor:
    """Return field values as ``[B, S]`` for sparse indexing."""
    if x.ndim < 3:
        raise RuntimeError(f"x0_pred must have at least 3 dims, got {tuple(x.shape)}.")
    return x.reshape(x.shape[0], -1)


def _gather_observations(
    x0_pred: torch.Tensor, obs_indices: torch.Tensor
) -> torch.Tensor:
    """Gather predicted values at integer observation indices."""
    if obs_indices.ndim != 3:
        raise RuntimeError(
            f"obs_indices must be shaped [B,N,D], got {tuple(obs_indices.shape)}."
        )
    if obs_indices.shape[0] != x0_pred.shape[0]:
        raise RuntimeError("obs_indices batch dimension must match x0_pred.")
    spatial_shape = tuple(int(v) for v in x0_pred.shape[1:])
    if obs_indices.shape[-1] != len(spatial_shape):
        raise RuntimeError(
            "obs_indices last dimension must index all non-batch dimensions: "
            f"got {int(obs_indices.shape[-1])}, expected {len(spatial_shape)}."
        )

    multipliers: list[int] = []
    stride = 1
    for size in reversed(spatial_shape):
        multipliers.insert(0, stride)
        stride *= size
    flat_idx = torch.zeros(
        obs_indices.shape[:2], device=obs_indices.device, dtype=torch.long
    )
    for dim, multiplier in enumerate(multipliers):
        idx = obs_indices[..., dim].long()
        if bool((idx < 0).any()) or bool((idx >= spatial_shape[dim]).any()):
            raise RuntimeError("obs_indices contains entries outside x0_pred shape.")
        flat_idx = flat_idx + idx * int(multiplier)
    return torch.gather(_flatten_spatial_field(x0_pred), 1, flat_idx)


def sparse_observation_loss(
    x0_pred: torch.Tensor,
    obs_values: torch.Tensor | None = None,
    obs_indices: torch.Tensor | None = None,
    obs_mask: torch.Tensor | None = None,
    obs_grid: torch.Tensor | None = None,
    obs_mask_grid: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute Charbonnier consistency at sparse observed locations."""
    if obs_grid is not None:
        if obs_grid.shape != x0_pred.shape:
            raise RuntimeError(
                f"obs_grid shape {tuple(obs_grid.shape)} must match x0_pred "
                f"{tuple(x0_pred.shape)}."
            )
        if obs_mask_grid is None:
            obs_mask_grid = torch.isfinite(obs_grid)
        residual = x0_pred - obs_grid.to(device=x0_pred.device, dtype=x0_pred.dtype)
        return charbonnier_loss(
            residual,
            mask=obs_mask_grid,
            weights=weights,
            eps=eps,
        )

    if obs_values is None or obs_indices is None:
        return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
    pred_values = _gather_observations(x0_pred, obs_indices.to(device=x0_pred.device))
    residual = pred_values - obs_values.to(device=x0_pred.device, dtype=x0_pred.dtype)
    return charbonnier_loss(residual, mask=obs_mask, weights=weights, eps=eps)
