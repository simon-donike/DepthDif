"""Backbone modules used by the denoising diffusion process."""

# Make `backbones` an explicit package so mkdocstrings/griffe can resolve
# `models.diffusion.DenoisingDiffusionProcess.backbones.unet_convnext` reliably.
from .unet_convnext import *
