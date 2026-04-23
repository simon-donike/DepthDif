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
  - synthetic random batch (`MODE="random"`)  
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
- OSTIA + Argo disk setup: `configs/px_space/model_config.yaml`, `configs/px_space/data_ostia_argo_disk.yaml`, `configs/px_space/training_config.yaml`  

## Workflow 1b: Export Global Depth Rasters  
Use `inference/export_global.py` when you want one spatially complete raster from the `ostia_argo_disk` manifest rather than a single sampled batch. The script:
- loads the configured checkpoint and disk-manifest dataset  
- selects one exact daily snapshot either from `--date YYYYMMDD` or from the earliest available day inside `--year ... --iso-week ...`  
- runs batched `predict_step(...)` over all spatial patches for that day  
- can fan out inference over all visible CUDA devices via `--multi-gpu` / `--no-multi-gpu`  
- streams patch outputs into on-disk accumulation buffers instead of holding the full world tensor in RAM  
- stitches prediction GeoTIFFs for Surface, 100m, 250m, 500m, 1000m, 2500m, and 5000m, then conservatively fills 1-2 pixel nodata seams in each written TIFF before finalizing it  
- maps requested depths to the nearest GLORYS/model channel and records requested depth, actual source depth, and channel index in TIFF metadata and `run_summary.yaml`  
- exports matching GLORYS rasters for the same seven depth levels by default via `--export-ground-truth` / `--no-export-ground-truth`  
- writes all observed Argo point locations for that timestep as a GeoJSON alongside the rasters  
- samples `200` observed Argo locations by default, saves their full `(Argo, prediction, GLORYS)` depth stacks plus graph references into a second GeoJSON, and renders one compact two-panel PNG per sampled location under `graphs/` with an OSTIA SST marker at depth 0 plus a side-by-side absolute-error panel  
- writes a second GeoJSON of patch-square polygons carrying only the `train`/`val` split labels for that timestep

Typical run:  
```bash
/work/envs/depth/bin/python inference/export_global.py \
  --data-config configs/px_space/data_ostia_argo_disk_actual.yaml \
  --checkpoint logs/<run>/best.ckpt \
  --year 2010 \
  --iso-week 1 \
  --device cuda
```  

Outputs land under `inference/outputs/<run_name>/` and include:
- `<run_name>_prediction_surface.tif`, `<run_name>_prediction_100m.tif`, `<run_name>_prediction_250m.tif`, `<run_name>_prediction_500m.tif`, `<run_name>_prediction_1000m.tif`, `<run_name>_prediction_2500m.tif`, `<run_name>_prediction_5000m.tif`: stitched prediction rasters  
- `<run_name>_glorys_surface.tif`, `<run_name>_glorys_100m.tif`, `<run_name>_glorys_250m.tif`, `<run_name>_glorys_500m.tif`, `<run_name>_glorys_1000m.tif`, `<run_name>_glorys_2500m.tif`, `<run_name>_glorys_5000m.tif`: stitched GLORYS truth rasters by default  
- `<run_name>_argo_points.geojson`: all observed Argo point locations for the selected timestep  
- `<run_name>_full_sample_locations.geojson`: sampled full-profile Argo locations with full depth-stack properties and `graph_png_path` pointers  
- `<run_name>_patch_splits.geojson`: patch polygons for the selected timestep with `split=train|val` properties only  
- `graphs/`: one compact PNG per sampled full-profile location with side-by-side temperature-vs-depth and absolute-error-vs-depth panels  
- `selected_patches.csv`: the manifest rows used for the run  
- `run_summary.yaml`: checkpoint/config/date metadata for traceability  
When `--output-name` is omitted, `<run_name>` defaults to `global_top_band_<YYYYMMDD>` and the run directory matches that name under `inference/outputs/`.

## Workflow 1c: Package One Run for the Cesium Globe
Use `inference/export_cesium_globe_assets.py` after the global export when you want one hosted asset bundle for the docs globe viewer. The script:
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

To upload one selected run into the hosted production area in the same step, provide an `rclone` destination:
```bash
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```

The hosted output lands under `inference/outputs/global_top_band_<YYYYMMDD>/globe/` locally and under `inference_production/globe/` in the bucket when synced with the example above. It includes:
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
  
## Example (`ostia_argo_disk` config)  
```python  
import torch  
  
from data.datamodule import DepthTileDataModule  
from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset  
from models.difFF import PixelDiffusionConditional  
  
model_config = "configs/px_space/model_config.yaml"  
data_config = "configs/px_space/data_ostia_argo_disk.yaml"  
train_config = "configs/px_space/training_config.yaml"  
ckpt_path = "logs/<run>/best-epochXXX.ckpt"  
  
train_dataset = OstiaArgoTiffDataset.from_config(data_config, split="train")  
val_dataset = OstiaArgoTiffDataset.from_config(data_config, split="val")  
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
  
