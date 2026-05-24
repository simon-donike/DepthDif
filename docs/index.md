<p align="center">
  <img src="assets/branding/banner_depthdif.png" width="65%" style="border-radius: 12px;" />
</p>

# Densifying Sparse Ocean Depth Observations
DepthDif explores conditional diffusion for reconstructing dense subsurface ocean temperature fields from sparse, masked observations.

The repository currently supports:
- EO-conditioned multi-band reconstruction (surface condition + deeper target bands)
- cross-source conditioning where EO surface SST can come from OSTIA while deeper targets remain Copernicus reanalysis
- public PyPI inference through the `depth-recon` package, including no-GLORYS ARGO/OSTIA week exports
- latent diffusion workflow with autoencoder-based depth compression (see [Autoencoder](autoencoder.md))

## Project Links
- [GitHub Repository](https://github.com/simon-donike/DepthDif)
- [Open Issues](https://github.com/simon-donike/DepthDif/issues)
- [Releases](https://github.com/simon-donike/DepthDif/releases)

<div class="globe-cta">
  <div class="globe-cta__body">
    <p class="globe-cta__eyebrow">New viewer</p>
    <h2 class="globe-cta__title">Inspect DepthDif outputs on a 3D globe</h2>
    <p class="globe-cta__text">
      Open the Cesium web viewer to compare stitched prediction and GLORYS
      depth levels with observed Argo points on one globe.
    </p>
  </div>
  <a class="globe-cta__button" href="globe/">Open 3D Globe</a>
</div>

## Model Description

![depthdif_schema](assets/figures/depthdif_schema.png)

DepthDif is a conditional diffusion model: it reconstructs dense GLORYS depth fields from sparse ARGO profile observations, conditioned on scenario-selected surface EO context (OSTIA SST for temperature/joint, SSS `sos` for salinity), ARGO observation support, GLORYS spatial support, plus coordinate/date context. See the full model details in [Model](model.md).

In the GeoTIFF training workflow, EO surface conditioning comes from the scenario-selected surface raster, subsurface targets come from GLORYS, and sparse inputs come from ARGO/EN4 profiles after depth alignment. Salinity is a scenario-selected field: `--scenario salinity` trains salinity only, and `--scenario joint` trains temperature and salinity together.

Ambient diffusion (short): at step `t`, `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`, `epsilon ~ N(0, I)`.
For ambient-occlusion training with observed mask `m` and further-corrupted mask `m' <= m`, optimize
`L` on the original `x` support intersected with valid target support and GLORYS spatial support (`x_valid_mask ∩ y_valid_mask ∩ land_mask`) while conditioning on the stronger corruption `m'`.

## Documentation Map
- [Quick Start](quickstart.md): environment setup + fastest train/infer path
- [Production Datasets](production-datasets.md): OSTIA L4 + EN4 profile dataset specs and download workflows
- [Data Overview](data.md): high-level modalities, variables, shared axes, and cadence
- [Data Source](data-source.md): source product and raw variable tables
- [Data Export](data-export.md): GeoTIFF dataset layout, quantization, and export command
- [Dataset Statistics](dataset-statistics.md): measured ARGO, raster, patch, and overlap counts
- [Data Contract](data-contract.md): model-facing tensor shapes, masks, and normalization
- [Model](model.md): architecture and diffusion conditioning flow
- [Temporal Dimension Ideas](temporal_dimension.md): options and tradeoffs for extending from `B,C,H,W` to `B,T,C,H,W` on real dataset windows
- [Autoencoder + Latent Diffusion](autoencoder.md): AE architecture, latent task setup, launch commands, and constraints
- [Data + Coordinate Injection](data-coordinate-injection.md): coordinate/date FiLM conditioning details
- [Training](training.md): CLI usage, run outputs, logging, checkpoints
- [Inference](inference.md): public API, global export, script, and direct `predict_step` workflows
- [Public Inference Package](public-inference-package.md): `depth-recon` install, API, CLI, asset resolution, and outputs
- [FUll settings documentation](settings.md#full-settings-documentation): per-file config keys, defaults, and explanations
- [Sampling Diagnostics](sampling-diagnostics.md): denoising intermediates, MAE-vs-step, and schedule profiling
- [Experiments](experiments.md): qualitative test results
- [Model Settings](settings.md): key config knobs, runtime mapping, and full settings reference
- [Development](development.md): known issues, TODOs, and roadmap
- [API Reference](api.md): auto-generated module reference via `mkdocstrings`
