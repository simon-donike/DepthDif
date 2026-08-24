<p align="center">
  <img src="assets/branding/banner_depthdif.webp" width="65%" alt="DepthDif" />
</p>

# DepthDif

DepthDif is a research codebase for reconstructing dense subsurface temperature
or salinity fields from sparse EN4/ARGO observations. The maintained pixel-space
model uses three ordered surface conditions—SST, SSS, and ADT—plus sparse profile
support, ocean masks, coordinates, and date context.

!!! important "Two checkpoint interfaces"
    Current repository configs use `[sst, sss, adt]`. The downloadable public
    `depthdif_v1.ckpt` uses the earlier single-OSTIA interface. Configs and
    checkpoints from those interfaces are not interchangeable.

## Supported workflows

- Export and download the GeoTIFF/Zarr training dataset.
- Train temperature, salinity, or joint pixel-space diffusion models.
- Pretrain with deterministic surface-plus-depth-offset synthetic targets.
- Run public regional inference for one ISO week.
- Export global, paired-variable, temporal, paper-metric, and spectral products.
- View generated products in the spatial, temporal, comparison, and spectral
  analysis pages.

<div class="globe-cta">
  <div class="globe-cta__body">
    <p class="globe-cta__eyebrow">Analysis</p>
    <h2 class="globe-cta__title">Explore generated DepthDif products</h2>
    <p class="globe-cta__text">
      Open the analysis landing page for spatial and temporal globes,
      error dashboards, model comparisons, and wavenumber spectra.
    </p>
  </div>
  <a class="globe-cta__button" href="analysis/">Open Analysis</a>
</div>

## Start here

- [Quick start](quickstart.md): install, train, and run inference.
- [Model](model.md): tensor contract, conditioning, diffusion, and outputs.
- [Training](training.md): current presets, scenarios, resume behavior, and monitors.
- [Data overview](data.md): active products and model-facing dataset.
- [Inference](inference.md): local, global, temporal, and paper workflows.
- [Public package](public-inference-package.md): stable one-week API and legacy checkpoint.
- [CLI reference](cli.md): maintained commands and module entry points.
- [Development](development.md): contributor workflow and repository structure.
- [Tests](tests.md): validation coverage and commands.

The documentation describes current code and committed configuration. Historical
experiment notes and speculative roadmaps are intentionally not part of the site.
