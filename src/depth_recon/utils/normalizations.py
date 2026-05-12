from __future__ import annotations

import torch

# Dataset-level statistics provided by user.
CELSIUS_TO_KELVIN_OFFSET = 273.15
Y_MEAN = 289.74267177946783
Y_STD = 10.933397487585731
PLOT_STD_MULTIPLIER = 2.5
PLOT_TEMP_MIN = -10.740821939496481
PLOT_TEMP_MAX = 43.92616549843217
PLOT_CMAP = "turbo"


def temperature_normalize(mode: str, tensor: torch.Tensor) -> torch.Tensor:
    """Compute temperature normalize and return the result.

    Args:
        mode (str): Input value.
        tensor (torch.Tensor): Tensor input for the computation.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    if mode not in {"norm", "denorm"}:
        raise ValueError("mode must be 'norm' or 'denorm'")

    mean = torch.as_tensor(Y_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(Y_STD, dtype=tensor.dtype, device=tensor.device)
    kelvin_offset = torch.as_tensor(
        CELSIUS_TO_KELVIN_OFFSET, dtype=tensor.dtype, device=tensor.device
    )

    if mode == "norm":
        tensor_kelvin = tensor + kelvin_offset
        return (tensor_kelvin - mean) / std
    denorm_kelvin = tensor * std + mean
    # Convert back to Celsius so callers keep receiving physical temperatures in C.
    return denorm_kelvin - kelvin_offset


def temperature_to_plot_unit(
    tensor: torch.Tensor,
    *,
    tensor_is_normalized: bool = True,
) -> torch.Tensor:
    """Compute temperature to plot unit and return the result.

    Args:
        tensor (torch.Tensor): Tensor input for the computation.
        tensor_is_normalized (bool): Boolean flag controlling behavior.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """
    temp = (
        temperature_normalize(mode="denorm", tensor=tensor)
        if tensor_is_normalized
        else tensor
    )
    t_min = torch.as_tensor(PLOT_TEMP_MIN, dtype=temp.dtype, device=temp.device)
    t_max = torch.as_tensor(PLOT_TEMP_MAX, dtype=temp.dtype, device=temp.device)
    denom = torch.clamp(t_max - t_min, min=torch.finfo(temp.dtype).eps)
    return ((temp - t_min) / denom).clamp(0.0, 1.0)
