# Training  
Training is launched via `train.py` and is fully config-driven.  

## Recommended CLI Usage  
Use explicit config paths to avoid ambiguity:  

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config.yaml
```

CLI aliases:  
- `--train-config` and `--training-config` are equivalent  
- `--model-config` also accepts the typo alias `--mdoel-config`  
- `--set <root.path=value>` is repeatable for strict nested overrides (`root` in `data`, `training`, `model`)  
- because `src/depth_recon/configs/px_space/model_config.yaml` itself is nested under top-level `model:`, model overrides must use `model.model.*` (example below)  

Override example:  

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config.yaml \
  --set data.dataset.output.return_info=true \
  --set training.trainer.max_epochs=100 \
  --set training.wandb.run_name=null
```

Ambient-occlusion objective example (self-supervised on `x`):  

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config_ambient.yaml \
  --set training.wandb.run_name=ambient_ostia_argo_netcdf_v1
```

When enabled, training logs:  
- `train/ambient_further_drop_fraction`  
- `train/ambient_observed_fraction_original`  
- `train/ambient_observed_fraction_further`  
- same metrics under `val/*` on validation epochs  

See [Ambient Occlusion Objective](ambient-occlusion-objective.md) for the full derivation, figure walkthrough, and paper citation.  

Note: turning `model.ambient_occlusion.enabled` back to `false` switches training back to direct `y` reconstruction over `y_valid_mask`. With `model.mask_loss_with_valid_pixels=true`, the standard task uses `y_valid_mask ∩ land_mask`, while ambient uses `x_valid_mask ∩ y_valid_mask ∩ land_mask`. `x_valid_mask` is ARGO observation support; `land_mask` is GLORYS spatial support.  
For CLI overrides, the corresponding path is `model.model.ambient_occlusion.enabled=false`.  

## Joint Temperature + Salinity Training

Temperature-only training is the default. For joint pixel training, enable both
sides of the contract:

- `data.dataset.output.include_salinity=true` makes the GeoTIFF loader return the normalized salinity side-channel tensors and masks.
- `model.output_fields=["temperature", "salinity"]` makes `PixelDiffusionConditional` stack temperature and salinity inside the model path.

Example:

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_geotiff.yaml \
  --train-config src/depth_recon/configs/px_space/training_config.yaml \
  --model-config src/depth_recon/configs/px_space/model_config_joint_temp_salinity.yaml \
  --set data.dataset.output.include_salinity=true
```

The joint preset predicts 100 output channels: 50 normalized temperature channels followed by 50 normalized salinity channels. Conditioning is 103 channels with EO enabled: one OSTIA channel, 100 stacked sparse ARGO channels, one collapsed `x_valid_mask` ARGO-observation support channel, and one GLORYS `land_mask` support channel. `train.py` fails early if the model asks for salinity but the data config does not enable `include_salinity`, because the model requires `x_salinity`/`y_salinity` batch keys.

Start from scratch or from a checkpoint trained with the same 100-channel architecture; existing 50-channel temperature checkpoints are not compatible.

## Important Config Notes  
- `dataset.core.dataloader_type` is expected to be `"light"` in the training runner.  
- `model.model_type="cond_px_dif"` runs pixel-space diffusion.  
- `model.model_type="latent_cond_dif"` runs latent diffusion with the autoencoder bridge.  
- dataset variant is selected by `dataset.core.dataset_variant`; use `"argo_geotiff_gridded"` for the active GeoTIFF workflow. `"argo_netcdf_gridded"` is legacy.  
- `dataset.output.include_salinity` is `false` by default; set it to `true` only when the model config consumes salinity.  
- parser defaults in `train.py` now point to `src/depth_recon/configs/px_space/*.yaml`; explicit CLI paths are still recommended for reproducibility  

## Autoencoder and Latent Workflow  
For latent diffusion training, use the latent config domain under `src/depth_recon/configs/lat_space/`.  

Autoencoder pretraining command:  

```bash
/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml
```

Latent diffusion command:  

```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --model-config src/depth_recon/configs/lat_space/model_config.yaml
```

Repo launcher scripts:  
- `./src/depth_recon/scripts/train_autoencoder.sh`  
- `./src/depth_recon/scripts/train_latent_diffusion.sh`  

Design details, model goals, and limitations are documented in [Autoencoder + Latent Diffusion](autoencoder.md).  

## What `train.py` Does During Startup  
1. Resolves distributed rank and creates a run directory under `logs/<timestamp>` on global rank 0.  
2. Copies exact config files into the run directory for reproducibility.  
3. Loads configs and validates `model.resume_checkpoint` / `model.load_checkpoint_only` early.  
4. Builds dataset and datamodule.  
5. Instantiates `PixelDiffusionConditional.from_config(...)`.  
6. Sets up W&B logger and callbacks.  

## Checkpointing and Resume  
ModelCheckpoint behavior:  
- best checkpoint: `best-epoch{epoch:03d}.ckpt` (monitor from `trainer.ckpt_monitor`)  
- always saved: `last.ckpt`  
- location: current run folder under `logs/`  

Resume and warm-start behavior:  
- set `model.resume_checkpoint` to `false/null` to train from scratch, or to a valid `.ckpt` path to load a checkpoint  
- set `model.load_checkpoint_only: true` to load only model `state_dict` before training starts (optimizer/scheduler state is re-initialized)  
- set `model.load_checkpoint_only: false` to resume full Lightning state from `model.resume_checkpoint` (model + optimizer/scheduler/trainer state)  
- invalid path fails early before trainer start  

## Device, Precision, and Validation Controls  
From `training_config` trainer section:  
- accelerator/devices strategy (`accelerator`, `devices`, optional legacy `num_gpus`)  
- mixed precision (`precision`)  
- optional validation cap via `val_batches_per_epoch` or `limit_val_batches`  
- gradient clipping (`gradient_clip_val`)  
- epoch-end full-reconstruction validation diagnostics run on global rank 0 only; regular `validation_step` loss metrics still use distributed reduction  

## Learning Rate Behavior  
`PixelDiffusionConditional` supports:  
- step-based linear warmup in `optimizer_step`  
- `ReduceLROnPlateau` scheduler when enabled  

Warmup and scheduler are configured via:  
- `scheduler.warmup.*`  
- `scheduler.reduce_on_plateau.*`  

`scheduler.reduce_on_plateau.interval` selects whether `patience` counts optimizer  
steps or epochs. The default monitor is `val/loss_ckpt`; with step-based patience,  
Lightning skips scheduler updates until validation has logged that metric.  
Increasing cheap validation-loss batches does not increase the full reverse-chain  
reconstruction count, which remains one cached first-batch pass per validation run.  

## Logging  
W&B logging is configured in `training_config.wandb`.  

Notable behavior:  
- gradients/parameters watching is opt-in via `watch_gradients` / `watch_parameters`  
- periodic scalar/image logging intervals are configurable  
- config files are uploaded to W&B run files (when experiment handle is available)  

- fixed overrides:  
  - `data.dataset.conditioning.eo_dropout_prob=0.0`  
  - `training.trainer.max_epochs=100`  
  - `training.wandb.run_name=null` (auto-generated run names)  

Launch:  

```bash
./src/depth_recon/scripts/start_occlusion_sweep.sh
```

Equivalent manual steps:  

```bash
/work/envs/depth/bin/wandb sweep src/depth_recon/configs/px_space/sweeps/eo_occlusion_grid_no_eodrop.yaml
/work/envs/depth/bin/wandb agent <entity/project/sweep_id>
```
