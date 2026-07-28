"""Robust scalar losses and masked reductions."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return the mean over finite values selected by an optional mask."""
    finite = torch.isfinite(values)
    if mask is not None:
        finite = finite & (mask.to(device=values.device) > 0)
    if not bool(finite.any()):
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return values[finite].mean()


def charbonnier_loss(
    residual: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute masked mean Charbonnier loss for ``residual``."""
    loss = torch.sqrt(residual * residual + float(eps) ** 2)
    reduction_mask = torch.isfinite(residual)
    if mask is not None:
        reduction_mask = reduction_mask & (mask.to(device=residual.device) > 0)
    if weights is not None:
        weight_tensor = weights.to(device=residual.device, dtype=residual.dtype)
        reduction_mask = reduction_mask & torch.isfinite(weight_tensor)
        weighted = loss * torch.where(reduction_mask, weight_tensor, 0.0)
        denom = torch.where(reduction_mask, weight_tensor, 0.0).sum()
        if denom.item() <= 0:
            return torch.zeros((), device=residual.device, dtype=residual.dtype)
        return weighted.sum() / denom
    return masked_mean(loss, reduction_mask)
