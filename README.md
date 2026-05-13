<p align="center">
  <a href="https://pypi.org/project/depth-recon/">
    <img src="https://img.shields.io/pypi/v/depth-recon?style=for-the-badge&label=PyPI" alt="PyPI version" />
  </a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python >= 3.12" />
  <img src="https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.10.0" />
  <img src="https://img.shields.io/badge/Lightning-2.6.1-792EE5?style=for-the-badge&logo=lightning&logoColor=white" alt="PyTorch Lightning 2.6.1" />
  <a href="https://github.com/simon-donike/DepthDif/actions/workflows/publish-pypi.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/simon-donike/DepthDif/publish-pypi.yml?branch=main&label=tests&style=for-the-badge" alt="Test workflow status" />
  </a>
  <img src="https://img.shields.io/badge/license-not%20declared-lightgrey?style=for-the-badge" alt="License not declared" />
  <a href="https://depthdif.donike.net/">
    <img src="https://img.shields.io/badge/docs-online-0b2e4f?style=for-the-badge" alt="Open Documentation" />
  </a>
  <a href="https://depthdif.donike.net/experiments/">
    <img src="https://img.shields.io/badge/experiments-online-0f3f68?style=for-the-badge" alt="Check Experiments" />
  </a>
  <a href="https://colab.research.google.com/github/simon-donike/DepthDif/blob/main/Colab_Demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>
</p>


<p align="center">
  <img src="docs/assets/branding/banner_depthdif.png" width="65%" style="border-radius: 12px;" />
</p>

# DepthDif

DepthDif is a conditional diffusion project for densifying sparse ocean temperature observations. Visit the [Documentation](https://depthdif.donike.net/) for more info on the models, datasets, and auxiliary data - or follow along with the [Experiments](https://depthdif.donike.net/experiments/).


## Demo

Run the public inference notebook directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simon-donike/DepthDif/blob/main/Colab_Demo.ipynb)

## Installation

This project uses Python 3.12.3.

```bash
python -m pip install -r requirements.txt
```

The root `requirements.txt` delegates to `pyproject.toml` so local installs and
package metadata use the same curated dependency list.

For public inference only, install the published PyPI package:

```bash
python -m pip install depth-recon
```

To install a branch or tag directly from GitHub, use the same package metadata:

```bash
python -m pip install "depth-recon @ git+https://github.com/simon-donike/DepthDif.git@main"
```

The equivalent explicit editable install is:

```bash
python -m pip install -e .
```

PyPI releases are published by GitHub Actions when a version tag such as
`v0.1.0` is pushed on `main`. The tag must match `project.version` in
`pyproject.toml`, and the repository's `pypi` environment must provide PyPI
publishing credentials.

## Model Overview

- Model: `PixelDiffusionConditional` (conditional pixel-space diffusion with ConvNeXt U-Net denoiser).
- Active dataset: `src/depth_recon/data/dataset_argo_netcdf_gridded.py` (`ArgoNetCDFGriddedPatchDataset`) lazily builds model-ready patches from ARGO/EN4, GLORYS, OSTIA, and sea-level NetCDF files without writing patch exports.
- Optional dataset ablation: `dataset.synthetic.enabled=true` builds sparse `x` from random GLORYS `y` pixels, controlled by `dataset.synthetic.pixel_count`.
- Config layout:
  - `src/depth_recon/configs/px_space/`: active pixel-space diffusion configs
  - `src/depth_recon/configs/lat_space/`: latent-space model/training/autoencoder configs

DepthDif is a conditional diffusion model: it reconstructs dense GLORYS depth fields from sparse ARGO profile observations, conditioned on OSTIA surface SST plus coordinate/date context.

Ambient-occlusion training is available via `model.ambient_occlusion.*`: the model receives a further-corrupted sparse Argo input during training while loss is evaluated on the original `x` support intersected with valid `y` support (`x_valid_mask ∩ y_valid_mask`). With the current `x0` training preset, the model predicts the clean target on that masked support rather than the old missing-pixel region. At inference time, both standard and ambient outputs are masked back to `NaN` wherever `y_valid_mask==0`; ambient mode does not do a post-hoc overwrite with observed `x` values when `clamp_known_pixels=false`.
See `docs/ambient-occlusion-objective.md` for the full mathematical objective, figure walkthrough, and citation.
![depthdif_schema](docs/assets/figures/depthdif_schema.png)

## Data Example

Representative surface-level training patches:

<p align="center">
  <img src="docs/assets/data/geotiff_dataset_random100_surface.png" width="85%" alt="Random surface-level training dataset patches" />
</p>

## Training

OSTIA + Argo NetCDF training:

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config.yaml
```

Ambient-occlusion objective example:

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config_ambient.yaml \
  --set training.wandb.run_name=ambient_ostia_argo_netcdf_v1
```

Notes:
- `--train-config` and `--training-config` are equivalent.
- Training outputs are written under `logs/<timestamp>/` with `best.ckpt` and `last.ckpt`.
- `model.resume_checkpoint` resumes full Lightning state; `model.load_checkpoint` warm-starts by loading only model weights.
- Latent diffusion workflow configs live in `src/depth_recon/configs/lat_space/`; see `docs/autoencoder.md` for AE + latent setup and launch commands.
- Latent launcher scripts: `src/depth_recon/scripts/train_autoencoder.sh`, `src/depth_recon/scripts/train_latent_diffusion.sh`.

## Inference

Public ISO-week inference is available from the `depth-recon` PyPI package. It
downloads the public model/config artifacts from Hugging Face, downloads EN4/ARGO
and optionally OSTIA source files, and writes stitched prediction GeoTIFFs plus
metadata for one ISO-week Wednesday.

```python
from depth_recon import run_week_inference

run_dir = run_week_inference(
    year=2015,
    iso_week=25,
    rectangle=(-20.0, 30.0, 10.0, 50.0),
    device="cuda",
    config_repo="simon-donike/DepthDif",
)
```

The public API downloads configs/checkpoints and the land mask from Hugging Face,
downloads EN4/ARGO and, by default, OSTIA for the selected ISO week, and returns
the GeoTIFF run directory. Existing cached files are reused automatically. Pass
`auto_download_ostia=False` without `ostia_dir` to run ARGO-only inference.
The package API uses non-overlapping public inference patches by default
(`patch_stride=tile_size`, normally 128), so small rectangles select compact
patch sets.
GLORYS is not required for the standard public inference path; it is only needed
for training or optional ground-truth comparison exports.
EN4/ARGO downloads use the Met Office annual EN.4.2.2 profile archives for each
calendar month touched by the selected ISO week.
OSTIA downloads use the Copernicus Marine CLI credentials configured in the
environment, or credentials passed to `run_week_inference` via
`copernicus_username` plus `copernicus_token`. The Copernicus Marine toolbox
accepts that token through its password field, so `copernicus_password` remains
supported as a backwards-compatible alias.

By default, the package uses `simon-donike/DepthDif` at revision `main`,
`model_config.yaml`, `data_config.yaml`, `training_config.yaml`,
`depthdif_v1.ckpt`, and `world_land_mask_glorys_0p1.tif`.

To prepare the public model files and land mask before a run:

```python
from depth_recon import resolve_public_inference_assets

bundle = resolve_public_inference_assets()
print(bundle.assets.checkpoint)
print(bundle.land_mask_path)
```

To fetch source files separately:

```bash
depth-recon-download-argo --year 2015 --iso-week 25 --output-dir ./en4_profiles
depth-recon-download-ostia --year 2015 --iso-week 25 --output-dir ./ostia
```

The same inference call is also exposed as a console script:

```bash
depth-recon-infer-week \
  --year 2015 \
  --iso-week 25 \
  --rectangle -20 30 10 50 \
  --device cuda
```

Use `src/depth_recon/inference/run_single.py`:

1. Set config/checkpoint constants at the top of `src/depth_recon/inference/run_single.py` (`MODEL_CONFIG_PATH`, `DATA_CONFIG_PATH`, `TRAIN_CONFIG_PATH`, `CHECKPOINT_PATH`).
   For the active EO setup in this repository, use:
   `src/depth_recon/configs/px_space/model_config.yaml`, `src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml`, `src/depth_recon/configs/px_space/training_config.yaml`
2. Choose `MODE` (`"dataloader"` or `"random"`).
3. Run:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.run_single
```

For a full spatial export, use `src/depth_recon/inference/export_global.py`. It selects one exact daily snapshot from the configured patch dataset (directly or via ISO week/year), runs inference on every patch for that day, streams the accumulation to disk, and writes stitched prediction and GLORYS GeoTIFFs for Surface, 10m, 50m, 100m, 250m, 500m, 1000m, 2000m, 2500m, and 5000m under `inference/outputs/global_top_band_<YYYYMMDD>/`. Requested depths are mapped to the nearest GLORYS channel and each TIFF records both the requested and actual source depth in metadata. By default it also writes GeoJSON exports for observed Argo point locations, sampled full-profile locations with per-point graphs, and train/val patch squares. The exporter defaults to the GeoTIFF-backed dataset, forces a world land-mask grid with 75% overlap, stitches overlaps with deterministic spatial weights, and zeroes final land pixels. Extra export-time Gaussian blur is disabled by default; pass a positive `--sigma` only when explicitly needed.

For a pooled validation-set depth summary, use `src/depth_recon/inference/export_validation_error_summary.py`. It loads the configured dataset `val` split, runs inference across the whole split, computes per-depth median absolute error against both GLORYS and the observed ARGO values, writes `validation_error_by_depth.csv`, and saves both a single-panel error graph and a two-panel median-profile/error figure under `inference/outputs/validation_error_summary/` by default.

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_validation_error_summary \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --checkpoint logs/<run>/best.ckpt \
  --split val \
  --year 2015 \
  --iso-week 25 \
  --device cuda
```

To package one exported run for the Cesium globe viewer in the docs, use:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_cesium_globe_assets \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```

The globe packager tiles every exported depth level into Cesium-ready folders and uploads those tiled assets, GeoJSON, graph PNGs, and `globe-config.json` when `--rclone-sync-scope globe` is used. Raw GeoTIFFs remain local in the run directory. The standalone viewer page lives at `docs/globe/index.html` and can load a hosted `globe-config.json`.

## Experiment Script

Use `experiments.py` for quick qualitative ablations on a single dataloader sample. It loads the configured model and checkpoint, runs a few fixed conditioning cases (`eo_plus_x`, `x_only_no_eo`, `coords_date_only_no_eo_no_x`), saves comparison plots under `temp/images/`, and prints compact tensor statistics for each case.

Typical run:

```bash
/work/envs/depth/bin/python experiments.py
```

Before running, check the config and checkpoint constants at the top of `experiments.py` if you want a different model, dataset split, or checkpoint.

## Documentation

- Full documentation: `docs/` (or build/serve with MkDocs).
- Autoencoder + latent workflow guide: `docs/autoencoder.md`.
- Experiments page: `docs/experiments.md`.
