from __future__ import annotations

import torch

# Dataset-level statistics provided by user.
Y_MEAN = 16.592671779467846
Y_STD = 10.933397487585731


def temperature_standardize(mode: str, tensor: torch.Tensor) -> torch.Tensor:
    """
    mode="norm"   -> (x - mean) / std
    mode="denorm" -> x * std + mean
    """
    if mode not in {"norm", "denorm"}:
        raise ValueError("mode must be 'norm' or 'denorm'")

    mean = torch.as_tensor(Y_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(Y_STD, dtype=tensor.dtype, device=tensor.device)

    if mode == "norm":
        return (tensor - mean) / std
    return tensor * std + mean
