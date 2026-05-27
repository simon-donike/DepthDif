# Training
Training is launched via `train.py` and is fully config-driven.

## Recommended CLI Usage
Use `--scenario` for pixel-space GeoTIFF training so the data/model channel contract is derived automatically:

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
```

CLI controls:
- `--config` defaults to `src/depth_recon/configs/px_space/training_super_config.yaml`
- `--scenario temperature|salinity|joint` derives `model.output_fields`, `model.generated_channels`, `model.condition_channels`, `data.dataset.output.fields`, `data.dataset.output.include_salinity`, and the EO raster source
- `--set <root.path=value>` is repeatable for strict nested overrides (`root` in `data`, `training`, `model`) after scenario resolution; inference helpers also accept `inference.*` overrides
- because the super-config has top-level `data`, `model`, and `training` sections, model overrides use `model.*` paths

Override example:

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set data.dataset.output.return_info=true \
  --set training.trainer.max_epochs=100 \
  --set training.wandb.run_name=null
```

Hard-area finetuning example:

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set data.dataset.finetune_sampling.enabled=true
```

This keeps validation on the normal validation split, while the train dataset is filtered to the configured hard-region/easy-row mix. When `data.dataset.finetune_sampling.relax_land_filter=true`, hard-region boxes also relax patch-grid land filtering for the finetune run only. The model can also emphasize coastal supervised pixels with `model.coastal_loss.*`; see [Coastal Loss Weighting For Finetuning](model.md#coastal-loss-weighting-for-finetuning).

Ambient-occlusion objective example (self-supervised on `x`):

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set model.ambient_occlusion.enabled=true \
  --set model.ambient_occlusion.further_drop_prob=0.25 \
  --set training.wandb.run_name=ambient_ostia_argo_geotiff_v1
```

When enabled, training logs:
- `train/ambient_further_drop_fraction`
- `train/ambient_observed_fraction_original`
- `train/ambient_observed_fraction_further`
- same metrics under `val/*` on validation epochs

See [Ambient Occlusion Objective](ambient-occlusion-objective.md) for the full derivation, figure walkthrough, and paper citation.

Note: turning `model.ambient_occlusion.enabled` back to `false` switches training back to direct `y` reconstruction over `y_valid_mask`. With `model.mask_loss_with_valid_pixels=true`, the standard task uses `y_valid_mask ∩ land_mask`, while ambient uses `x_valid_mask ∩ y_valid_mask ∩ land_mask`. `x_valid_mask` is ARGO observation support; `land_mask` is GLORYS spatial support.
For CLI overrides, the corresponding path is `model.ambient_occlusion.enabled=false`.

## Temperature, Salinity, And Joint Training

The scenario selector supports three pixel-space contracts and applies the coupled data/model settings together:

| Scenario | Output fields | Salinity data | Generated channels | Condition channels |
| --- | --- | --- | ---: | ---: |
| `temperature` | `['temperature']` | disabled | `50` | `53` |
| `salinity` | `['salinity']` | enabled | `50` | `53` |
| `joint` | `['temperature', 'salinity']` | enabled | `100` | `103` |

`condition_channels` is derived from selected output channels plus the enabled conditioning inputs: scenario-selected EO, collapsed valid mask, and GLORYS land mask. The salinity scenario uses SSS `sos` as the EO channel; temperature and joint use OSTIA `analysed_sst`. Do not maintain `model.output_fields`, `model.generated_channels`, `model.condition_channels`, `data.dataset.output.fields`, or `data.dataset.output.include_salinity` manually in normal super-configs; use `--scenario` and let the resolver write effective configs. `--set` still runs after scenario resolution for intentional experiments.

Every run snapshots the original super-config plus resolved effective `data_config_effective.yaml`, `model_config_effective.yaml`, and `training_config_effective.yaml` under `logs/<timestamp>/`, and uploads those files to W&B. Validation shuffling stays enabled by default in the super-config for the current experimentation workflow.

Start from scratch or from a checkpoint trained with the same architecture; temperature-only, salinity-only, and joint checkpoints are not channel-compatible with each other.

## Important Config Notes
- `dataset.core.dataloader_type` is expected to be `"light"` in the training runner.
- `model.model_type="cond_px_dif"` runs pixel-space diffusion.
- `train.py` super-config workflow is pixel-space only; latent diffusion still uses the latent config files documented below.
- dataset variant is selected by `dataset.core.dataset_variant`; use `"argo_geotiff_gridded"` for the active GeoTIFF workflow. `"argo_netcdf_gridded"` is legacy.
- `dataset.output.fields` and `dataset.output.include_salinity` are derived by `--scenario`; do not maintain them by hand in the super-config.
- Pixel split data/model/training YAML files were removed; use the super-configs for pixel training and inference.

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
From the `training.trainer` section:
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
W&B logging is configured in `training.wandb`.

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
