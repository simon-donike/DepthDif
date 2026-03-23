# Training  
Training is launched via `train.py` and is fully config-driven.  
  
## Recommended CLI Usage  
Use explicit config paths to avoid ambiguity:  
  
```bash  
/work/envs/depth/bin/python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config.yaml  
```  
  
CLI aliases:  
- `--train-config` and `--training-config` are equivalent  
- `--model-config` also accepts the typo alias `--mdoel-config`  
- `--set <root.path=value>` is repeatable for strict nested overrides (`root` in `data`, `training`, `model`)  
- because `configs/px_space/model_config.yaml` itself is nested under top-level `model:`, model overrides must use `model.model.*` (example below)  
  
Override example:  
  
```bash  
/work/envs/depth/bin/python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config.yaml \  
  --set data.dataset.output.return_info=true \  
  --set training.trainer.max_epochs=100 \  
  --set training.wandb.run_name=null  
```  
  
Ambient-occlusion objective example (self-supervised on `x`):  
  
```bash  
/work/envs/depth/bin/python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config_ambient.yaml \  
  --set training.wandb.run_name=ambient_ostia_argo_disk_v1  
```  
  
When enabled, training logs:  
- `train/ambient_further_drop_fraction`  
- `train/ambient_observed_fraction_original`  
- `train/ambient_observed_fraction_further`  
- same metrics under `val/*` on validation epochs  
  
See [Ambient Occlusion Objective](ambient-occlusion-objective.md) for the full derivation, figure walkthrough, and paper citation.  
  
Note: turning `model.ambient_occlusion.enabled` back to `false` does not switch to full-image loss. With `model.mask_loss_with_valid_pixels=true`, loss falls back to missing pixels (`1 - valid_mask`).  
For CLI overrides, the corresponding path is `model.model.ambient_occlusion.enabled=false`.  
  
## Important Config Notes  
- `dataset.core.dataloader_type` is expected to be `"light"` in the training runner.  
- `model.model_type="cond_px_dif"` runs pixel-space diffusion.  
- `model.model_type="latent_cond_dif"` runs latent diffusion with the autoencoder bridge.  
- dataset variant is selected by `dataset.core.dataset_variant` (or inferred from data config filename)  
- `dataset_variant` selects the dataset implementation in code (`"eo_4band"`, `"ostia"`, or `"ostia_argo_disk"`)  
- `SurfaceTempPatchOstiaLightDataset` still does not apply EO degradation (no EO dropout/random-scale/speckle) when you train the legacy OSTIA overlap dataset through an older config.  
- EO dropout from data config is injected into dataset object for both train and val  
- parser defaults in `train.py` now point to `configs/px_space/*.yaml`; explicit CLI paths are still recommended for reproducibility  
  
## Autoencoder and Latent Workflow  
For latent diffusion training, use the latent config domain under `configs/lat_space/`.  
  
Autoencoder pretraining command:  
  
```bash  
/work/envs/depth/bin/python train_autoencoder.py \  
  --ae-config configs/lat_space/ae_config.yaml \  
  --data-config configs/lat_space/data_config.yaml \  
  --train-config configs/lat_space/training_config.yaml  
```  
  
Latent diffusion command:  
  
```bash  
/work/envs/depth/bin/python train.py \  
  --data-config configs/lat_space/data_config.yaml \  
  --train-config configs/lat_space/training_config.yaml \  
  --model-config configs/lat_space/model_config.yaml  
```  
  
Repo launcher scripts:  
- `./scripts/train_autoencoder.sh`  
- `./scripts/train_latent_diffusion.sh`  
  
Design details, model goals, and limitations are documented in [Autoencoder + Latent Diffusion](autoencoder.md).  
  
## What `train.py` Does During Startup  
1. Resolves distributed rank and creates a run directory under `logs/<timestamp>` on global rank 0.  
2. Copies exact config files into the run directory for reproducibility.  
3. Loads configs and validates `model.resume_checkpoint` / `model.load_checkpoint` early.  
4. Builds dataset and datamodule.  
5. Instantiates `PixelDiffusionConditional.from_config(...)`.  
6. Sets up W&B logger and callbacks.  
  
## Checkpointing and Resume  
ModelCheckpoint behavior:  
- best checkpoint: `best-epoch{epoch:03d}.ckpt` (monitor from `trainer.ckpt_monitor`)  
- always saved: `last.ckpt`  
- location: current run folder under `logs/`  
  
Resume and warm-start behavior:  
- set `model.resume_checkpoint` to a valid `.ckpt` path to resume full Lightning state (model + optimizer/scheduler/trainer state)  
- set `model.load_checkpoint` to a valid `.ckpt` path to load only model `state_dict` before training starts (optimizer/scheduler state is re-initialized)  
- `model.resume_checkpoint` and `model.load_checkpoint` are mutually exclusive  
- invalid path fails early before trainer start  
  
## Device, Precision, and Validation Controls  
From `training_config` trainer section:  
- accelerator/devices strategy (`accelerator`, `devices`, optional legacy `num_gpus`)  
- mixed precision (`precision`)  
- optional validation cap via `val_batches_per_epoch` or `limit_val_batches`  
- gradient clipping (`gradient_clip_val`)  
  
## Learning Rate Behavior  
`PixelDiffusionConditional` supports:  
- step-based linear warmup in `optimizer_step`  
- `ReduceLROnPlateau` scheduler when enabled  
  
Warmup and scheduler are configured via:  
- `scheduler.warmup.*`  
- `scheduler.reduce_on_plateau.*`  
  
## Logging  
W&B logging is configured in `training_config.wandb`.  
  
Notable behavior:  
- gradients/parameters watching is opt-in via `watch_gradients` / `watch_parameters`  
- periodic scalar/image logging intervals are configurable  
- config files are uploaded to W&B run files (when experiment handle is available)  
  
## W&B Occlusion Sweep (EO Always Available)  
Sweep config:  
- `configs/px_space/sweeps/eo_occlusion_grid_no_eodrop.yaml`  
  
This sweep runs grid values:  
- `mask_fraction`: `0.95, 0.96, 0.97, 0.98, 0.99, 0.995`  
- fixed overrides:  
  - `data.dataset.conditioning.eo_dropout_prob=0.0`  
  - `training.trainer.max_epochs=100`  
  - `training.wandb.run_name=null` (auto-generated run names)  
  
Launch:  
  
```bash  
./scripts/start_occlusion_sweep.sh  
```  
  
Equivalent manual steps:  
  
```bash  
/work/envs/depth/bin/wandb sweep configs/px_space/sweeps/eo_occlusion_grid_no_eodrop.yaml  
/work/envs/depth/bin/wandb agent <entity/project/sweep_id>  
```  
