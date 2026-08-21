"""Deterministic surface-duplication pretraining target helpers."""

from typing import Any

from depth_recon.data.synthetic_dataset_creation.vertical_offset_prior import (
    PriorSample,
    VerticalOffsetAccumulator,
    VerticalOffsetPrior,
)


def fit_vertical_offset_prior(*args: Any, **kwargs: Any) -> VerticalOffsetPrior:
    """Fit offsets lazily so module execution remains warning-free."""
    from depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior import (
        fit_vertical_offset_prior as _fit,
    )

    return _fit(*args, **kwargs)


__all__ = [
    "PriorSample",
    "VerticalOffsetAccumulator",
    "VerticalOffsetPrior",
    "fit_vertical_offset_prior",
]
