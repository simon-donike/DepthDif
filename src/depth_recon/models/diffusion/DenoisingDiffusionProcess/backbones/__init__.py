"""Backbone modules used by the denoising diffusion process."""

# Make `backbones` an explicit package so mkdocstrings/griffe can resolve
# `depth_recon.models.diffusion.DenoisingDiffusionProcess.backbones.unet_convnext`.
from .unet_convnext import *
