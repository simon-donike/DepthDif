"""GLORYS structure-function prior loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from depth_recon.paths import resolve_config_path


def _load_reference_tensor(path: str | Path, key: str) -> torch.Tensor:
    """Load a tensor-like value from a torch reference-stat file."""
    payload: Any = torch.load(resolve_config_path(path), map_location="cpu")
    if not isinstance(payload, dict) or key not in payload:
        raise RuntimeError(f"Reference file {path} must contain key {key!r}.")
    tensor = torch.as_tensor(payload[key], dtype=torch.float32).detach()
    if tensor.numel() == 0:
        raise RuntimeError(f"Reference tensor {key!r} in {path} must be non-empty.")
    return tensor


class StructureFunctionPriorLoss(nn.Module):
    """Match generated log structure functions to GLORYS reference statistics."""

    def __init__(
        self,
        *,
        reference_path: str | Path,
        num_pairs: int = 8192,
        eps: float = 1e-6,
        per_depth: bool = True,
    ) -> None:
        """Initialize the structure-function prior from precomputed stats."""
        super().__init__()
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "structure_function_prior.reference_path is required when enabled."
            )
        distance_bins = _load_reference_tensor(reference_path, "distance_bins")
        s2_ref = _load_reference_tensor(reference_path, "s2_ref")
        if distance_bins.ndim != 1 or int(distance_bins.numel()) < 2:
            raise RuntimeError(
                "distance_bins must be a 1D tensor with at least 2 edges."
            )
        if s2_ref.ndim not in {1, 2}:
            raise RuntimeError("s2_ref must be shaped [num_bins] or [C,num_bins].")
        num_bins = int(distance_bins.numel()) - 1
        if int(s2_ref.shape[-1]) != num_bins:
            raise RuntimeError(
                "s2_ref last dimension must equal len(distance_bins) - 1."
            )
        self.register_buffer("distance_bins", distance_bins, persistent=False)
        self.register_buffer("s2_ref", s2_ref, persistent=False)
        self.num_pairs = max(1, int(num_pairs))
        self.eps = float(eps)
        self.per_depth = bool(per_depth)

    def _sample_pair_stats(
        self,
        x0_pred: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random same-depth pairs and return binned sums/counts."""
        bsz, channels, height, width = x0_pred.shape
        device = x0_pred.device
        pairs = self.num_pairs
        b_idx = torch.randint(bsz, (pairs,), device=device)
        c_idx = torch.randint(channels, (pairs,), device=device)
        y1 = torch.randint(height, (pairs,), device=device)
        x1 = torch.randint(width, (pairs,), device=device)
        y2 = torch.randint(height, (pairs,), device=device)
        x2 = torch.randint(width, (pairs,), device=device)
        distance = torch.sqrt((y2 - y1).float().square() + (x2 - x1).float().square())
        bin_idx = torch.bucketize(distance, self.distance_bins.to(device=device)) - 1
        valid = (bin_idx >= 0) & (bin_idx < int(self.distance_bins.numel()) - 1)
        valid = valid & (distance > 0)
        if valid_mask is not None:
            mask = valid_mask.to(device=device) > 0
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1:
                mask = mask.expand(-1, channels, -1, -1)
            valid = valid & mask[b_idx, c_idx, y1, x1] & mask[b_idx, c_idx, y2, x2]

        increments = x0_pred[b_idx, c_idx, y2, x2] - x0_pred[b_idx, c_idx, y1, x1]
        # Autocast can promote the square to float32 while x0_pred is fp16.
        squared = increments.square().to(dtype=torch.float32)
        num_bins = int(self.distance_bins.numel()) - 1
        if self.per_depth:
            flat_idx = c_idx * num_bins + bin_idx
            sums = torch.zeros(channels * num_bins, device=device, dtype=squared.dtype)
            counts = torch.zeros_like(sums)
            flat_valid = valid & (flat_idx >= 0) & (flat_idx < channels * num_bins)
            sums.scatter_add_(0, flat_idx[flat_valid], squared[flat_valid])
            counts.scatter_add_(
                0,
                flat_idx[flat_valid],
                torch.ones_like(squared[flat_valid]),
            )
            return sums.view(channels, num_bins), counts.view(channels, num_bins)

        sums = torch.zeros(num_bins, device=device, dtype=squared.dtype)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, bin_idx[valid], squared[valid])
        counts.scatter_add_(0, bin_idx[valid], torch.ones_like(squared[valid]))
        return sums, counts

    def forward(
        self,
        x0_pred: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked log structure-function prior loss."""
        if x0_pred.ndim != 4:
            raise RuntimeError(
                f"x0_pred must be shaped [B,C,H,W], got {tuple(x0_pred.shape)}."
            )
        sums, counts = self._sample_pair_stats(x0_pred, valid_mask)
        s2_hat = sums / counts.clamp_min(1.0)
        ref = self.s2_ref.to(device=x0_pred.device, dtype=x0_pred.dtype)
        if s2_hat.ndim == 2 and ref.ndim == 1:
            ref = ref.unsqueeze(0).expand_as(s2_hat)
        if s2_hat.ndim == 1 and ref.ndim == 2:
            ref = ref.mean(dim=0)
        if tuple(ref.shape) != tuple(s2_hat.shape):
            raise RuntimeError(
                f"s2_ref shape {tuple(ref.shape)} is incompatible with sampled "
                f"structure function {tuple(s2_hat.shape)}."
            )
        valid_bins = (counts > 0) & torch.isfinite(ref) & (ref > 0)
        if not bool(valid_bins.any()):
            return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
        diff = torch.abs(torch.log(s2_hat + self.eps) - torch.log(ref + self.eps))
        return diff[valid_bins].mean()


class PairedStructureFunctionLoss(nn.Module):
    """Match prediction structure functions to the paired GLORYS target."""

    def __init__(
        self,
        *,
        num_pairs: int = 8192,
        num_bins: int = 16,
        eps: float = 1e-6,
        per_depth: bool = True,
    ) -> None:
        """Initialize same-sample paired structure-function comparison."""
        super().__init__()
        self.num_pairs = max(1, int(num_pairs))
        self.num_bins = max(1, int(num_bins))
        self.eps = float(eps)
        self.per_depth = bool(per_depth)

    @staticmethod
    def _expand_valid_mask(
        valid_mask: torch.Tensor | None,
        *,
        channels: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a channel-expanded boolean support mask."""
        if valid_mask is None:
            return None
        mask = valid_mask.to(device=device) > 0
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1:
            mask = mask.expand(-1, channels, -1, -1)
        return mask

    def _sample_indices(self, field: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample same-depth point pairs and dynamic distance bins."""
        bsz, channels, height, width = field.shape
        device = field.device
        pairs = self.num_pairs
        b_idx = torch.randint(bsz, (pairs,), device=device)
        c_idx = torch.randint(channels, (pairs,), device=device)
        y1 = torch.randint(height, (pairs,), device=device)
        x1 = torch.randint(width, (pairs,), device=device)
        y2 = torch.randint(height, (pairs,), device=device)
        x2 = torch.randint(width, (pairs,), device=device)
        distance = torch.sqrt((y2 - y1).float().square() + (x2 - x1).float().square())
        max_distance = torch.sqrt(
            torch.tensor(float((height - 1) ** 2 + (width - 1) ** 2), device=device)
        )
        distance_bins = torch.linspace(0.0, max_distance, self.num_bins + 1).to(
            device=device
        )
        bin_idx = torch.bucketize(distance, distance_bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < self.num_bins) & (distance > 0)
        return {
            "b_idx": b_idx,
            "c_idx": c_idx,
            "y1": y1,
            "x1": x1,
            "y2": y2,
            "x2": x2,
            "bin_idx": bin_idx,
            "valid": valid,
        }

    def _sample_pair_stats(
        self,
        field: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
        indices: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binned squared-increment sums and counts for sampled pairs."""
        _, channels, _, _ = field.shape
        device = field.device
        b_idx = indices["b_idx"]
        c_idx = indices["c_idx"]
        y1 = indices["y1"]
        x1 = indices["x1"]
        y2 = indices["y2"]
        x2 = indices["x2"]
        bin_idx = indices["bin_idx"]
        valid = indices["valid"].clone()
        mask = self._expand_valid_mask(valid_mask, channels=channels, device=device)
        if mask is not None:
            valid = valid & mask[b_idx, c_idx, y1, x1] & mask[b_idx, c_idx, y2, x2]

        increments = field[b_idx, c_idx, y2, x2] - field[b_idx, c_idx, y1, x1]
        # Keep scatter_add_ source and destination dtypes identical under autocast.
        squared = increments.square().to(dtype=torch.float32)
        if self.per_depth:
            flat_idx = c_idx * self.num_bins + bin_idx
            sums = torch.zeros(
                channels * self.num_bins, device=device, dtype=squared.dtype
            )
            counts = torch.zeros_like(sums)
            flat_valid = valid & (flat_idx >= 0) & (flat_idx < channels * self.num_bins)
            sums.scatter_add_(0, flat_idx[flat_valid], squared[flat_valid])
            counts.scatter_add_(
                0,
                flat_idx[flat_valid],
                torch.ones_like(squared[flat_valid]),
            )
            return sums.view(channels, self.num_bins), counts.view(
                channels, self.num_bins
            )

        sums = torch.zeros(self.num_bins, device=device, dtype=squared.dtype)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, bin_idx[valid], squared[valid])
        counts.scatter_add_(0, bin_idx[valid], torch.ones_like(squared[valid]))
        return sums, counts

    def forward(
        self,
        x0_pred: torch.Tensor,
        target_grid: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute paired log structure-function difference."""
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
        indices = self._sample_indices(x0_pred)
        pred_sums, pred_counts = self._sample_pair_stats(
            x0_pred,
            valid_mask=valid_mask,
            indices=indices,
        )
        target_sums, target_counts = self._sample_pair_stats(
            target_grid,
            valid_mask=valid_mask,
            indices=indices,
        )
        pred_s2 = pred_sums / pred_counts.clamp_min(1.0)
        target_s2 = target_sums / target_counts.clamp_min(1.0)
        valid_bins = (pred_counts > 0) & (target_counts > 0)
        if not bool(valid_bins.any()):
            return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)
        diff = torch.abs(
            torch.log(pred_s2 + self.eps) - torch.log(target_s2 + self.eps)
        )
        return diff[valid_bins].mean()
