from __future__ import annotations

import torch


def minmax_stretch(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    nodata_value: float | None = None,
) -> torch.Tensor:
    """
    Linearly stretch tensor values to [0, 1] based on min and max values in the tensor.
    Optionally exclude a nodata value (default: 0.0) from the min/max computation.
    """
    t = tensor.detach()
    valid = torch.isfinite(t)
    if mask is not None:
        m = mask.detach()
        if m.shape != t.shape:
            m = m.expand_as(t)
        valid = valid & (m > 0.0)
    if nodata_value is not None:
        nodata = torch.as_tensor(nodata_value, dtype=t.dtype, device=t.device)
        valid = valid & ~torch.isclose(t, nodata, atol=1e-6, rtol=0.0)

    if torch.any(valid):
        t_min = t[valid].min()
        t_max = t[valid].max()
    else:
        t_min = torch.as_tensor(0.0, dtype=t.dtype, device=t.device)
        t_max = t_min
    denom = torch.clamp(t_max - t_min, min=torch.finfo(t.dtype).eps)
    stretched = ((t - t_min) / denom).clamp(0.0, 1.0)
    if nodata_value is not None:
        stretched = stretched.masked_fill(~valid, 0.0)
    return stretched
