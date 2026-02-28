# Training
Training is launched via `train.py` and is fully config-driven.

## Recommended CLI Usage
Use explicit config paths to avoid ambiguity:  

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_ostia.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

For legacy same-source EO (`eo_4band`), use `--data-config configs/data.yaml`.

CLI aliases:  
- `--train-config` and `--training-config` are equivalent  
- `--model-config` also accepts the typo alias `--mdoel-config`  
- `--set <root.path=value>` is repeatable for strict nested overrides (`root` in `data`, `training`, `model`)  

Override example:  

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_ostia.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --set data.dataset.degradation.mask_fraction=0.99 \
  --set data.dataset.conditioning.eo_dropout_prob=0.0 \
  --set training.trainer.max_epochs=100 \
  --set training.wandb.run_name=null
```

Sparse X-only objective override example (`eo_4band` and `ostia`):

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_ostia.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --set model.training_objective.mode=x_holdout_sparse \
  --set model.training_objective.holdout_fraction=0.15
```

## Important Config Notes
- `train.py` currently supports only:  
  - `dataset.core.dataloader_type: "light"`  
  - `model.model_type: "cond_px_dif"`  
- dataset variant is selected by `dataset.core.dataset_variant` (or inferred from data config filename)  
- for `dataset_variant="ostia"`, set `dataset.source.light_index_csv` to the overlap index (with `ostia_npy_path`) and ensure sibling `ostia_npy/` tiles exist  
- `SurfaceTempPatchOstiaLightDataset` does not apply EO degradation (no EO dropout/random-scale/speckle)  
- EO dropout from data config is injected into dataset object for both train and val  
- `model.training_objective.mode="x_holdout_sparse"` keeps EO conditioning and trains against held-out observed `x` pixels only (no `y` required for the loss path); dataset returns `batch["loss_mask"]` + `batch["x_supervision_target"]`
- `x_holdout_sparse` supports `eo_4band` and `ostia`; unsupported variants still fall back to `standard` objective with a warning
- with current shared dataset object, `training_objective.deterministic_val_mask=true` makes dataset-side holdout masks deterministic for both train and val
- Epoch-end validation reconstruction logging to W&B includes `x`, `eo`, reconstruction, `y`, context valid mask, and the sparse loss mask panel
- Optional local PNG dumps remain available via `model.training_objective.dump_val_reconstruction.*` (disabled by default)
- parser defaults in `train.py` still point to legacy `configs/*_config.yaml` names, so explicit CLI paths are recommended  

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
- `configs/sweeps/eo_occlusion_grid_no_eodrop.yaml`  

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
/work/envs/depth/bin/wandb sweep configs/sweeps/eo_occlusion_grid_no_eodrop.yaml
/work/envs/depth/bin/wandb agent <entity/project/sweep_id>
```
