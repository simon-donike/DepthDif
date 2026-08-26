# Python API Reference

## Supported public API

The package root intentionally exposes a small, lazy-loaded inference surface.

::: depth_recon

::: depth_recon.inference.api
    options:
      members:
        - InferenceAssets
        - PublicInferenceAssets
        - resolve_hf_assets
        - resolve_public_inference_assets
        - download_argo_for_week
        - run_week_inference

Use [Public inference](public-inference-package.md) for the checkpoint interface,
source downloads, outputs, and CLI equivalents.

## Repository building blocks

The following interfaces support repository training and exporters. They are
documented for contributors but are not the stable PyPI surface.

### Data

::: depth_recon.data.datamodule

::: depth_recon.data.dataset_argo_geotiff_gridded

### Diffusion model and EMA

::: depth_recon.models.diffusion.PixelDiffusion

::: depth_recon.models.diffusion.EMA

### Diffusion process and samplers

::: depth_recon.models.diffusion.DenoisingDiffusionProcess.DenoisingDiffusionProcess

::: depth_recon.models.diffusion.DenoisingDiffusionProcess.forward

::: depth_recon.models.diffusion.DenoisingDiffusionProcess.beta_schedules

::: depth_recon.models.diffusion.DenoisingDiffusionProcess.samplers.DDPM

::: depth_recon.models.diffusion.DenoisingDiffusionProcess.samplers.DDIM

### Utilities

::: depth_recon.utils.normalizations

::: depth_recon.utils.stretching

::: depth_recon.utils.validation_denoise
