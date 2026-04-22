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

## Workflow 1b: Export One Global Top-Band Raster  
Use `inference/export_global.py` when you want one spatially complete raster from the `ostia_argo_disk` manifest rather than a single sampled batch. The script:
- loads the configured checkpoint and disk-manifest dataset  
- selects one exact daily snapshot either from `--date YYYYMMDD` or from the earliest available day inside `--year ... --iso-week ...`  
- runs batched `predict_step(...)` over all spatial patches for that day  
- can fan out inference over all visible CUDA devices via `--multi-gpu` / `--no-multi-gpu`  
- streams patch outputs into on-disk accumulation buffers instead of holding the full world tensor in RAM  
- stitches `y_hat_denorm[:, 0, :, :]` into one large tiled GeoTIFF with internal overviews  
- exports the matching GLORYS top-band raster by default via `--export-ground-truth` / `--no-export-ground-truth`  
- writes all observed Argo point locations for that timestep as a GeoJSON alongside the rasters  
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
- `<run_name>_prediction.tif`: stitched global top-band prediction  
- `<run_name>_glorys_top_band.tif`: stitched GLORYS top-band truth export by default  
- `<run_name>_argo_points.geojson`: all observed Argo point locations for the selected timestep  
- `<run_name>_patch_splits.geojson`: patch polygons for the selected timestep with `split=train|val` properties only  
- `selected_patches.csv`: the manifest rows used for the run  
- `run_summary.yaml`: checkpoint/config/date metadata for traceability  
When `--output-name` is omitted, `<run_name>` defaults to `global_top_band_<YYYYMMDD>` and the run directory matches that name under `inference/outputs/`.

## Workflow 1c: Package One Run for the Cesium Globe
Use `inference/export_cesium_globe_assets.py` after the global export when you want one hosted asset bundle for the docs globe viewer. The script:
- reads one completed `inference/outputs/<run_name>/` directory
- tiles the stitched prediction and ground-truth GeoTIFFs with `gdal2tiles.py`
- copies the Argo points GeoJSON into the hosted globe bundle
- copies the train/val patch-split GeoJSON into the hosted globe bundle
- writes `globe/globe-config.json` for the static Cesium page

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
- `prediction_tiles/`: TMS imagery tiles for the prediction raster
- `ground_truth_tiles/`: TMS imagery tiles for the GLORYS raster when present
- `argo_points.geojson`: hosted point overlay
- `patch_splits.geojson`: hosted train/val patch grid overlay rendered as dashed outlines in the globe viewer
- `globe-config.json`: the viewer manifest consumed by [Globe Viewer](globe.md)

When serving from a bucket, enable CORS for the docs origin so the static MkDocs page can fetch the tiled layers and GeoJSON.
  
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
  
The same sampler can also be injected per batch through `batch["sampler"]` in direct prediction calls.  
  
