# Inference
There are two practical inference workflows in this repository:
- run the standalone script `inference/run_single.py`
- call `PixelDiffusionConditional.predict_step(...)` directly

DepthDif supports pixel-space configs (`configs/px_space/*`) and latent-workflow configs (`configs/lat_space/*`).
For latent workflow setup and command flow, see [Autoencoder + Latent Diffusion](autoencoder.md).

## Workflow 1: Use `inference/run_single.py`
`inference/run_single.py` is a configurable script for quick prediction sanity checks.

### What it supports
- load config files and instantiate model/datamodule
- load checkpoint (explicit override or `model.load_checkpoint` / `model.resume_checkpoint`)
- run from:
  - dataloader sample (`MODE="dataloader"`)
  - random tensor batch (`MODE="random"`)
- optional intermediate sample capture

### Important script settings
At the top of `inference/run_single.py`, set:
- `MODEL_CONFIG_PATH`
- `DATA_CONFIG_PATH`
- `TRAIN_CONFIG_PATH`
- `CHECKPOINT_PATH` (or keep `None` to use `model.load_checkpoint` then `model.resume_checkpoint`)
- `MODE`, `LOADER_SPLIT`, `DEVICE`, `INCLUDE_INTERMEDIATES`

### Note on default paths
The script constants should be set explicitly. In this repository, the actively used configs are:
- OSTIA + Argo NetCDF setup: `configs/px_space/model_config.yaml`, `configs/px_space/data_ostia_argo_netcdf.yaml`, `configs/px_space/training_config.yaml`

## Workflow 1b: Export Global Depth Rasters
Use `inference/export_global.py` when you want the standard production inference path: one spatially complete ISO-week globe from raw ARGO/GLORYS/OSTIA/sea-surface products or an equivalent patch dataset. The script:
- requires `--year ... --iso-week ...` and uses the ISO-week Wednesday as the single target date
- forces the inference grid to `patch_grid_source=land_mask`, `require_argo_for_all=false`, and `patch_stride=tile_size/4` for 75% overlapping patches
- keeps every tile with at least `--min-ocean-fraction` ocean cover; the default `0.05` includes all patches with 5% or more ocean
- runs batched `predict_step(...)` over all global patches for that week-centered date
- runs one stochastic prediction per patch; the global smoothing/variance reduction comes from 75% spatial overlap and overlap-weighted stitching
- can fan out inference over all visible CUDA devices via `--multi-gpu` / `--no-multi-gpu`
- streams patch outputs into on-disk accumulation buffers instead of holding the full world tensor in RAM
- stitches prediction GeoTIFFs for Surface, 100m, 250m, 500m, 1000m, 2500m, and 5000m by averaging overlap counts, then conservatively fills tiny nodata seams
- applies the configured land-mask GeoTIFF at the final write step so land pixels are `0.0` and uncovered water remains nodata
- maps requested depths to the nearest GLORYS/model channel and records requested depth, actual source depth, and channel index in TIFF metadata and `run_summary.yaml`
- exports matching GLORYS rasters for the same seven depth levels by default via `--export-ground-truth` / `--no-export-ground-truth`
- writes all observed Argo point locations for that timestep as a GeoJSON alongside the rasters
- exports full-profile metadata for all observed Argo locations by default, saves their full `(Argo, prediction, GLORYS)` depth stacks plus graph references into a second GeoJSON, and renders one two-panel PNG per location under `graphs/` with an OSTIA SST marker at depth 0 plus a side-by-side absolute-error panel; pass `--full-sample-count 0` to disable or a positive count to keep a capped subset
- writes a second GeoJSON of patch-square polygons carrying only the `train`/`val` split labels for that timestep
- optionally packages Cesium globe assets and uploads them with `rclone` in the same command

Typical run:
```bash
/work/envs/depth/bin/python inference/export_global.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --checkpoint logs/<run>/best.ckpt \
  --year 2010 \
  --iso-week 1 \
  --device cuda \
  --min-ocean-fraction 0.05 \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```
Increase overlap or lower `--min-ocean-fraction` for coverage changes; per-tile multi-generation is intentionally disabled.

Outputs land under `inference/outputs/<run_name>/` and include:
- `<run_name>_prediction_surface.tif`, `<run_name>_prediction_100m.tif`, `<run_name>_prediction_250m.tif`, `<run_name>_prediction_500m.tif`, `<run_name>_prediction_1000m.tif`, `<run_name>_prediction_2500m.tif`, `<run_name>_prediction_5000m.tif`: stitched prediction rasters
- `<run_name>_glorys_surface.tif`, `<run_name>_glorys_100m.tif`, `<run_name>_glorys_250m.tif`, `<run_name>_glorys_500m.tif`, `<run_name>_glorys_1000m.tif`, `<run_name>_glorys_2500m.tif`, `<run_name>_glorys_5000m.tif`: stitched GLORYS truth rasters by default
- `<run_name>_argo_points.geojson`: all observed Argo point locations for the selected timestep
- `<run_name>_full_sample_locations.geojson`: sampled full-profile Argo locations with full depth-stack properties and `graph_png_path` pointers
- `<run_name>_patch_splits.geojson`: patch polygons for the selected timestep with `split=train|val` properties only
- `graphs/`: one large PNG per sampled full-profile location with side-by-side temperature-vs-depth and absolute-error-vs-depth panels
- `globe/`: Cesium tiles, hosted GeoJSON, copied graphs, and `globe-config.json` when `--public-base-url` or `--rclone-remote` is supplied
- `selected_patches.csv`: the dataset rows used for the run
- `run_summary.yaml`: checkpoint/config/date, forced inference-grid, land-mask, packaging, and upload metadata for traceability
When `--output-name` is omitted, `<run_name>` defaults to `global_top_band_<YYYYMMDD>` and the run directory matches that name under `inference/outputs/`.

## Workflow 1c: Export One Pooled Validation Error Summary
Use `inference/export_validation_error_summary.py` when you want one depth-vs-error summary across the whole dataset split instead of one map export or one sampled batch. The script:
- loads the configured checkpoint and the explicit dataset split selected by `--split` (`val` by default)
- optionally narrows that split to one ISO week via `--year ... --iso-week ...`, matching the same week-style selection used by the global export workflow
- forces real-observation semantics by evaluating `|Prediction - GLORYS|` only on valid `y` support and `|Prediction - ARGO|` only on observed `x` support
- pools all eligible validation pixels by depth level across the entire split and reports pooled medians rather than per-patch averages
- writes `validation_error_by_depth.csv` with per-depth medians, counts, and the support-aware median profiles
- saves `validation_median_absolute_error_by_depth.png` with the two pooled error traces
- saves `validation_median_profile_and_error_by_depth.png` with the median `(ARGO, prediction, GLORYS)` profiles plus the pooled error panel
- writes `run_summary.yaml` with checkpoint/config/device/split metadata and artifact names

Typical run:
```bash
/work/envs/depth/bin/python inference/export_validation_error_summary.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --checkpoint logs/<run>/best.ckpt \
  --split val \
  --year 2015 \
  --iso-week 25 \
  --device cuda
```

Default outputs land under `inference/outputs/validation_error_summary/`:
- `validation_error_by_depth.csv`: per-depth pooled error/profile summary table
- `validation_median_absolute_error_by_depth.png`: single-panel pooled median absolute-error graph
- `validation_median_profile_and_error_by_depth.png`: two-panel pooled median-profile/error figure
- `run_summary.yaml`: checkpoint/config/split metadata plus artifact filenames

## Workflow 1d: Package One Run for the Cesium Globe
The standard path is to let `inference/export_global.py` package and upload the globe assets by passing `--public-base-url` and `--rclone-remote`. `inference/export_cesium_globe_assets.py` remains available when you need to re-package an existing run directory without re-running model inference. The packaging step:
- reads one completed `inference/outputs/<run_name>/` directory
- tiles every stitched prediction and ground-truth depth GeoTIFF with `gdal2tiles.py`
- rewrites the hosted Argo points GeoJSON with rounded coordinates and no extra properties
- rewrites the sampled full-profile GeoJSON with rounded coordinates and only the popup properties, then copies its `graphs/` folder
- merges both point exports into one hosted `argo_sample_locations.geojson` so the globe uses one toggleable ARGO layer with distinct markers for ordinary points and full-depth-profile points
- rewrites the train/val patch-split GeoJSON with rounded coordinates and only the `split` property
- writes `globe/globe-config.json` with a `depth_levels` list used by the static Cesium page depth slider

Typical run:
```bash
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/
```

The hosted output lands under `inference/outputs/global_top_band_<YYYYMMDD>/globe/` locally and under `inference_production/globe/` in the bucket when synced. It includes:
- `prediction_tiles_surface/`, `prediction_tiles_100m/`, etc.: TMS imagery tiles for each prediction depth raster
- `ground_truth_tiles_surface/`, `ground_truth_tiles_100m/`, etc.: TMS imagery tiles for each GLORYS depth raster when present
- `argo_sample_locations.geojson`: hosted combined ARGO point overlay used by the single ARGO globe layer, with per-feature marker metadata for ordinary points versus full-depth profiles
- `argo_points.geojson`: hosted raw observed-point overlay source retained alongside the combined file
- `full_sample_locations.geojson`: hosted sampled-profile point overlay source retained alongside the combined file
- `graphs/`: hosted PNGs opened by the sampled-profile popup
- `patch_splits.geojson`: hosted train/val patch grid overlay rendered as solid red/green fills at fixed 50% opacity in the globe viewer
- `globe-config.json`: the viewer manifest consumed by the standalone `globe/` viewer route

Raw GeoTIFFs stay in the run directory and are not copied into `globe/` for bucket upload.

When serving from a bucket, enable CORS for the docs origin so the standalone static globe page can fetch the tiled layers and GeoJSON.

## Workflow 2: Direct `predict_step`
The model inference entry point is:
- `PixelDiffusionConditional.predict_step(batch, batch_idx=0)`

Minimum required batch keys:
- `x`
- `x_valid_mask`
- `y_valid_mask`

Common optional keys:
- `eo`
- `x_valid_mask_1d`
- `land_mask`
- `coords`
- `date`
- `sampler`
- `clamp_known_pixels`
- `return_intermediates`
- `intermediate_step_indices`

### Returned outputs
`predict_step` returns a dictionary containing:
- `y_hat`: standardized prediction
- `y_hat_denorm`: temperature-denormalized prediction, masked to `NaN` where `y_valid_mask==0`
- `denoise_samples`: reverse samples (if requested)
- `x0_denoise_samples`: per-step x0 predictions (if requested)
- `sampler`: sampler used for prediction

## Example (`argo_netcdf_gridded` config)
```python
import torch

from data.datamodule import DepthTileDataModule
from data.dataset_argo_netcdf_gridded import ArgoNetCDFGriddedPatchDataset
from models.difFF import PixelDiffusionConditional

model_config = "configs/px_space/model_config.yaml"
data_config = "configs/px_space/data_ostia_argo_netcdf.yaml"
train_config = "configs/px_space/training_config.yaml"
ckpt_path = "logs/<run>/best-epochXXX.ckpt"

train_dataset = ArgoNetCDFGriddedPatchDataset.from_config(data_config, split="train")
val_dataset = ArgoNetCDFGriddedPatchDataset.from_config(data_config, split="val")
datamodule = DepthTileDataModule(dataset=train_dataset, val_dataset=val_dataset)

model = PixelDiffusionConditional.from_config(
    model_config_path=model_config,
    data_config_path=data_config,
    training_config_path=train_config,
    datamodule=datamodule,
)

state = torch.load(ckpt_path, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state
model.load_state_dict(state_dict, strict=False)
model.eval()

batch = next(iter(datamodule.val_dataloader()))
with torch.no_grad():
    pred = model.predict_step(batch, batch_idx=0)

y_hat = pred["y_hat"]
y_hat_denorm = pred["y_hat_denorm"]
```

## Sampler Choice
Validation/inference sampler can be switched via training config:
- `training.validation_sampling.sampler: "ddpm"` or `"ddim"`
- DDIM controls:
  - `ddim_num_timesteps`
  - `ddim_eta`
  - `ddim_temperature`

The same sampler can also be injected per batch through `batch["sampler"]` in direct prediction calls.
