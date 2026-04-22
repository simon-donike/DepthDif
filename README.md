<p align="center">  
  <img src="docs/assets/branding/banner_depthdif.png" width="65%" style="border-radius: 12px;" />  
</p>  
  
<p align="center">  
  <a href="https://depthdif.donike.net/">  
    <img src="https://img.shields.io/badge/Visit-Documentation-0b2e4f?style=for-the-badge" alt="Open Documentation" />  
  </a>  
  <a href="https://depthdif.donike.net/experiments/">  
    <img src="https://img.shields.io/badge/Open-Experiments-0f3f68?style=for-the-badge" alt="Check Experiments" />  
  </a>  
</p>  
  
# DepthDif  
  
DepthDif is a conditional diffusion project for densifying sparse ocean temperature observations. Visit the [Documentation](https://depthdif.donike.net/) for more info on the models, datasets, and auxiliary data - or follow along with the [Experiments](https://depthdif.donike.net/experiments/).  
  
  
  
## Installation  
  
This project uses Python 3.12.3.  
  
```bash  
python -m pip install -r requirements.txt  
```  
  
## Model Overview  
  
- Model: `PixelDiffusionConditional` (conditional pixel-space diffusion with ConvNeXt U-Net denoiser).  
- Main task modes:  
  - `eo_4band`: EO-conditioned multiband reconstruction (`[eo, x, x_valid_mask] -> y`).  
- Additional standalone raw-source datasets: `data/dataset_ostia_argo.py` (`OstiaArgoTileDataset`) for CSV-driven OSTIA condition tile retrieval plus date-matched EN4 profile extraction from `argo_file_path`, with each Argo profile resampled onto the fixed 50-level GLORYS depth grid, optional temporal-window averaging via `days`, georeferenced GeoTIFF export via `save_to_disk(...)`, explicit `x_valid_mask` / `y_valid_mask` depth masks, a horizontal `land_mask`, and `x_valid_mask_1d` for profile-column support; and `data/dataset_ostia_argo_disk.py` (`OstiaArgoTiffDataset`) for loading the exported OSTIA/ARGO/GLORYS GeoTIFF triplets back from the manifest CSV, with GLORYS stored on disk as packed `int16` (`0.01°C` precision) and decoded back to Celsius on read. The disk loader also supports a synthetic mode that replaces exported Argo `x` with sparse GLORYS profile columns sampled at a Gaussian-distributed pixel count around `synthetic.pixel_count` and clamped to `+-10%`.
- Config layout:  
  - `configs/px_space/`: active pixel-space diffusion configs  
  - `configs/lat_space/`: latent-space config set (`model_config.yaml`, `training_config.yaml`, `data_config.yaml`, `ae_config.yaml`)  
  
DepthDif is a conditional diffusion model: it reconstructs dense depth fields from corrupted submarine observations, conditioned on EO (surface) data plus sparse corrupted subsurface input. Synthetic sparse inputs are generated with continuous curved trajectory masks to mimic submarine movement; in the current dataset version, each track keeps one measurement every few pixels (random 2-8 pixel stride) until the configured corruption percentage is reached. It can inject coordinate/date context via FiLM conditioning and reconstruct the full target image.  
  
Ambient-occlusion training is available via `model.ambient_occlusion.*`: the model receives a further-corrupted sparse Argo input during training while loss is evaluated on the original `x` support intersected with valid `y` support (`x_valid_mask ∩ y_valid_mask`). With the current `x0` training preset, the model predicts the clean target on that masked support rather than the old missing-pixel region. At inference time, both standard and ambient outputs are masked back to `NaN` wherever `y_valid_mask==0`; ambient mode does not do a post-hoc overwrite with observed `x` values when `clamp_known_pixels=false`.  
See `docs/ambient-occlusion-objective.md` for the full mathematical objective, figure walkthrough, and citation.  
![depthdif_schema](docs/assets/figures/depthdif_schema.png)  
  
## Training  
  
OSTIA + Argo disk training:  

```bash  
python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config.yaml  
```  

Ambient-occlusion objective example:  

```bash  
python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config_ambient.yaml \  
  --set training.wandb.run_name=ambient_ostia_argo_disk_v1  
```  
  
Notes:  
- `--train-config` and `--training-config` are equivalent.  
- Training outputs are written under `logs/<timestamp>/` with `best.ckpt` and `last.ckpt`.  
- `model.resume_checkpoint` resumes full Lightning state; `model.load_checkpoint` warm-starts by loading only model weights.  
- Latent diffusion workflow configs live in `configs/lat_space/`; see `docs/autoencoder.md` for AE + latent setup and launch commands.  
- Latent launcher scripts: `scripts/train_autoencoder.sh`, `scripts/train_latent_diffusion.sh`.  
  
## Inference  
  
Use `inference/run_single.py`:  

1. Set config/checkpoint constants at the top of `inference/run_single.py` (`MODEL_CONFIG_PATH`, `DATA_CONFIG_PATH`, `TRAIN_CONFIG_PATH`, `CHECKPOINT_PATH`).  
   For the active EO setup in this repository, use:  
   `configs/px_space/model_config.yaml`, `configs/px_space/data_ostia_argo_disk.yaml`, `configs/px_space/training_config.yaml`  
2. Choose `MODE` (`"dataloader"` or `"random"`).  
3. Run:  
  
```bash  
/work/envs/depth/bin/python inference/run_single.py  
```  

For a full spatial export, use `inference/export_global.py`. It selects one exact daily snapshot from the `ostia_argo_disk` manifest (directly or via ISO week/year), runs inference on every patch for that day, streams the accumulation to disk, writes the stitched top-band prediction under `inference/outputs/global_top_band_<YYYYMMDD>/`, and by default also writes the matching GLORYS top-band raster plus GeoJSON exports for observed Argo point locations and train/val patch squares.

To package one exported run for the Cesium globe viewer in the docs, use:

```bash
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```

The docs viewer page lives at `docs/globe.md` and can load a hosted `globe-config.json`.

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
  
