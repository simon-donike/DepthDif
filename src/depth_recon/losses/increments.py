"""Sparse pairwise increment loss."""

from __future__ import annotations

import torch

from .robust import charbonnier_loss
from .sparse_observation import _gather_observations


def _subsample_pairs(
    residual: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample valid pair residuals without introducing CPU-side pixel loops."""
    flat_residual = residual.reshape(-1)
    flat_mask = mask.reshape(-1)
    valid_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    if int(valid_idx.numel()) == 0:
        return flat_residual[:0], flat_mask[:0]
    if max_pairs > 0 and int(valid_idx.numel()) > max_pairs:
        choice = torch.randperm(int(valid_idx.numel()), device=valid_idx.device)[
            :max_pairs
        ]
        valid_idx = valid_idx[choice]
    return flat_residual[valid_idx], torch.ones_like(
        flat_residual[valid_idx], dtype=torch.bool
    )


def _gridded_vertical_increment_loss(
    x0_pred: torch.Tensor,
    obs_grid: torch.Tensor,
    obs_mask_grid: torch.Tensor,
    *,
    eps: float,
    max_pairs_per_sample: int,
) -> torch.Tensor:
    """Compute adjacent-depth sparse increment loss for gridded observations."""
    if x0_pred.size(1) < 2:
        return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
    pred_inc = x0_pred[:, 1:] - x0_pred[:, :-1]
    obs_inc = obs_grid[:, 1:] - obs_grid[:, :-1]
    pair_mask = (obs_mask_grid[:, 1:] > 0) & (obs_mask_grid[:, :-1] > 0)
    residual, mask = _subsample_pairs(
        pred_inc - obs_inc,
        pair_mask,
        max_pairs=max_pairs_per_sample * int(x0_pred.shape[0]),
    )
    return charbonnier_loss(residual, mask=mask, eps=eps)


def _gridded_horizontal_increment_loss(
    x0_pred: torch.Tensor,
    obs_grid: torch.Tensor,
    obs_mask_grid: torch.Tensor,
    *,
    eps: float,
    max_pairs_per_sample: int,
) -> torch.Tensor:
    """Compute same-depth right/down sparse increment loss for gridded observations."""
    residuals: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for dy, dx in ((0, 1), (1, 0)):
        pred_inc = (
            x0_pred[:, :, dy:, dx:]
            - x0_pred[:, :, : x0_pred.size(2) - dy, : x0_pred.size(3) - dx]
        )
        obs_inc = (
            obs_grid[:, :, dy:, dx:]
            - obs_grid[:, :, : obs_grid.size(2) - dy, : obs_grid.size(3) - dx]
        )
        pair_mask = (obs_mask_grid[:, :, dy:, dx:] > 0) & (
            obs_mask_grid[
                :, :, : obs_mask_grid.size(2) - dy, : obs_mask_grid.size(3) - dx
            ]
            > 0
        )
        residuals.append(pred_inc - obs_inc)
        masks.append(pair_mask)
    residual = torch.cat([item.reshape(-1) for item in residuals], dim=0)
    mask = torch.cat([item.reshape(-1) for item in masks], dim=0)
    residual, mask = _subsample_pairs(
        residual,
        mask,
        max_pairs=max_pairs_per_sample * int(x0_pred.shape[0]),
    )
    return charbonnier_loss(residual, mask=mask, eps=eps)


def _packed_increment_loss(
    x0_pred: torch.Tensor,
    obs_values: torch.Tensor,
    obs_indices: torch.Tensor,
    pair_indices: torch.Tensor,
    obs_mask: torch.Tensor | None,
    *,
    eps: float,
) -> torch.Tensor:
    """Compute pair-index increment loss for packed sparse observations."""
    pred_values = _gather_observations(x0_pred, obs_indices.to(device=x0_pred.device))
    pair_indices = pair_indices.to(device=x0_pred.device).long()
    if pair_indices.ndim != 3 or pair_indices.shape[-1] != 2:
        raise RuntimeError(
            f"pair_indices must be shaped [B,P,2], got {tuple(pair_indices.shape)}."
        )
    gather_i = pair_indices[..., 0]
    gather_j = pair_indices[..., 1]
    pred_inc = torch.gather(pred_values, 1, gather_i) - torch.gather(
        pred_values, 1, gather_j
    )
    obs_values = obs_values.to(device=x0_pred.device, dtype=x0_pred.dtype)
    obs_inc = torch.gather(obs_values, 1, gather_i) - torch.gather(
        obs_values, 1, gather_j
    )
    pair_mask = torch.ones_like(pred_inc, dtype=torch.bool)
    if obs_mask is not None:
        obs_mask = obs_mask.to(device=x0_pred.device) > 0
        pair_mask = torch.gather(obs_mask, 1, gather_i) & torch.gather(
            obs_mask, 1, gather_j
        )
    return charbonnier_loss(pred_inc - obs_inc, mask=pair_mask, eps=eps)


def sparse_increment_loss(
    x0_pred: torch.Tensor,
    obs_values: torch.Tensor | None = None,
    obs_indices: torch.Tensor | None = None,
    obs_mask: torch.Tensor | None = None,
    obs_grid: torch.Tensor | None = None,
    obs_mask_grid: torch.Tensor | None = None,
    pair_indices: torch.Tensor | None = None,
    eps: float = 1e-3,
    max_pairs_per_sample: int = 4096,
    vertical_pairs: bool = True,
    horizontal_pairs: bool = False,
    horizontal_max_distance: float | None = None,
) -> torch.Tensor:
    """Compute sparse pairwise increment consistency loss."""
    _ = horizontal_max_distance
    if pair_indices is not None:
        if obs_values is None or obs_indices is None:
            raise RuntimeError("pair_indices require obs_values and obs_indices.")
        return _packed_increment_loss(
            x0_pred,
            obs_values,
            obs_indices,
            pair_indices,
            obs_mask,
            eps=eps,
        )

    if obs_grid is None:
        return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
    if obs_grid.shape != x0_pred.shape:
        raise RuntimeError(
            f"obs_grid shape {tuple(obs_grid.shape)} must match x0_pred "
            f"{tuple(x0_pred.shape)}."
        )
    if obs_mask_grid is None:
        obs_mask_grid = torch.isfinite(obs_grid)
    obs_grid = obs_grid.to(device=x0_pred.device, dtype=x0_pred.dtype)
    obs_mask_grid = obs_mask_grid.to(device=x0_pred.device)

    losses: list[torch.Tensor] = []
    if vertical_pairs:
        losses.append(
            _gridded_vertical_increment_loss(
                x0_pred,
                obs_grid,
                obs_mask_grid,
                eps=eps,
                max_pairs_per_sample=max_pairs_per_sample,
            )
        )
    if horizontal_pairs:
        losses.append(
            _gridded_horizontal_increment_loss(
                x0_pred,
                obs_grid,
                obs_mask_grid,
                eps=eps,
                max_pairs_per_sample=max_pairs_per_sample,
            )
        )
    if not losses:
        return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
    return torch.stack(losses).mean()
