"""Baseline models for DepthDif experiments."""

from .IDW import IDWInterpolationBaseline
from .LSTM import PointwiseLSTMBaseline

__all__ = ["IDWInterpolationBaseline", "PointwiseLSTMBaseline"]
