"""Baseline models for Ocean Depth Reconstruction experiments."""

from .IDW import IDWInterpolationBaseline
from .LSTM import PointwiseLSTMBaseline
from .UNet import UNetInfillingBaseline

__all__ = [
    "IDWInterpolationBaseline",
    "PointwiseLSTMBaseline",
    "UNetInfillingBaseline",
]
