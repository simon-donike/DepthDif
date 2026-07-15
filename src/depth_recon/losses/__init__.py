"""Loss modules for ambient ocean-field diffusion training."""

from .ambient_ocean_loss import AmbientOceanLoss
from .increments import sparse_increment_loss
from .robust import charbonnier_loss, masked_mean
from .sparse_observation import sparse_observation_loss
from .spectral import PairedSpectralEnergyFloorLoss, SpectralEnergyFloorLoss
from .structure_function import PairedStructureFunctionLoss, StructureFunctionPriorLoss
from .timestep_weighting import aux_timestep_weight

__all__ = [
    "AmbientOceanLoss",
    "StructureFunctionPriorLoss",
    "SpectralEnergyFloorLoss",
    "PairedStructureFunctionLoss",
    "PairedSpectralEnergyFloorLoss",
    "charbonnier_loss",
    "masked_mean",
    "sparse_observation_loss",
    "sparse_increment_loss",
    "aux_timestep_weight",
]
