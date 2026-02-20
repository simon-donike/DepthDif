# Training
Training is launched via `train.py` and is fully config-driven.

## Recommended CLI Usage
Use explicit config paths to avoid ambiguity:

```bash
python3 train.py \
  --data-config configs/data_config_eo_4band.yaml \
  --train-config configs/training_config_eo_4band.yaml \
  --model-config configs/model_config_eo_4band.yaml
```

CLI aliases:
- `--train-config` and `--training-config` are equivalent
- `--model-config` also accepts the typo alias `--mdoel-config`

## Single-Band (Legacy Config Set)
For the single-band setup in this repo, use `configs/older_configs/*`:

```bash
python3 train.py \
  --data-config configs/older_configs/data_config.yaml \
  --train-config configs/older_configs/training_config.yaml \
  --model-config configs/older_configs/model_config.yaml
```

Before launching a fresh run with this legacy set, set
`model.resume_checkpoint: false` (or `null`) in `configs/older_configs/model_config.yaml`.

## Important Config Notes
- `train.py` currently supports only:
  - `dataset.dataloader_type: "light"`
  - `model.model_type: "cond_px_dif"`
- dataset variant is selected by `dataset.dataset_variant` (or inferred from data config filename)
- EO dropout from data config is injected into dataset object for both train and val
- parser defaults in `train.py` still point to legacy `configs/*_config.yaml` names, so explicit CLI paths are recommended

## What `train.py` Does During Startup
1. Resolves distributed rank and creates a run directory under `logs/<timestamp>` on global rank 0.
2. Copies exact config files into the run directory for reproducibility.
3. Loads configs and validates `model.resume_checkpoint` early.
4. Builds dataset and datamodule.
5. Instantiates `PixelDiffusionConditional.from_config(...)`.
6. Sets up W&B logger and callbacks.

## Checkpointing and Resume
ModelCheckpoint behavior:
- best checkpoint: `best-epoch{epoch:03d}.ckpt` (monitor from `trainer.ckpt_monitor`)
- always saved: `last.ckpt`
- location: current run folder under `logs/`

Resume behavior:
- set `model.resume_checkpoint` to a valid `.ckpt` path
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
