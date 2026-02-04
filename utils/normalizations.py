from __future__ import annotations

import torch

# Dataset-level statistics provided by user.
Y_MEAN = 16.592671779467846
Y_STD = 10.933397487585731
PLOT_STD_MULTIPLIER = 2.5
PLOT_TEMP_MIN = Y_MEAN - (PLOT_STD_MULTIPLIER * Y_STD)
PLOT_TEMP_MAX = Y_MEAN + (PLOT_STD_MULTIPLIER * Y_STD)
PLOT_CMAP = "turbo"


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


def temperature_to_plot_unit(
    tensor: torch.Tensor,
    *,
    tensor_is_standardized: bool = True,
) -> torch.Tensor:
    """
    Convert temperature data to [0, 1] for plotting using fixed dataset-level
    stretch limits from this module.
    """
    temp = temperature_standardize(mode="denorm", tensor=tensor) if tensor_is_standardized else tensor
    t_min = torch.as_tensor(PLOT_TEMP_MIN, dtype=temp.dtype, device=temp.device)
    t_max = torch.as_tensor(PLOT_TEMP_MAX, dtype=temp.dtype, device=temp.device)
    denom = torch.clamp(t_max - t_min, min=torch.finfo(temp.dtype).eps)
    return ((temp - t_min) / denom).clamp(0.0, 1.0)
