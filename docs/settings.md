# Config Settings
This page maps the current config files to runtime behavior. Pixel-space training and inference now use super-configs plus a scenario selector; the old pixel split data/model/training YAMLs are gone.

## Active Pixel Configs

| File | Used by | Purpose |
| --- | --- | --- |
| `src/depth_recon/configs/px_space/training_super_config.yaml` | `train.py` | Pixel GeoTIFF training defaults. Contains top-level `scenario`, `data`, `model`, and `training` sections. |
| `src/depth_recon/configs/px_space/inference_super_config.yaml` | inference exporters and smoke scripts | Pixel GeoTIFF inference defaults. Contains top-level `scenario`, `data`, `model`, `training`, and `inference` sections. |

Latent-space workflows still use `src/depth_recon/configs/lat_space/model_config.yaml`, `training_config.yaml`, and `ae_config.yaml`; see [Autoencoder + Latent Diffusion](autoencoder.md).

## Scenario Resolution

Select the pixel task with `--scenario temperature|salinity|joint` or with the top-level `scenario` key in the super-config. CLI `--scenario` wins over the file. The resolver lives in `depth_recon.configs.config_resolver_pixel` and materializes effective split configs for existing dataset/model constructors.

| Scenario | Derived `model.output_fields` | Derived `data.dataset.output.fields` | Derived `data.dataset.output.include_salinity` | Derived EO source | Derived `model.generated_channels` | Derived `model.condition_channels` |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `temperature` | `['temperature']` | `['temperature']` | `false` | `ostia/analysed_sst` | `50` | `53` |
| `salinity` | `['salinity']` | `['salinity']` | `true` | `sss/sos` | `50` | `53` |
| `joint` | `['temperature', 'salinity']` | `['temperature', 'salinity']` | `true` | `ostia/analysed_sst` | `100` | `103` |

`model.condition_channels` is computed from the selected generated channels plus enabled conditioning inputs: EO channel, `condition_mask_channels`, and land-mask channel. Do not maintain `output_fields`, `fields`, `include_salinity`, `eo_source`, `eo_var_name`, `generated_channels`, or `condition_channels` by hand in the super-config for normal runs. Use repeatable `--set` overrides only for intentional experiments; overrides are applied after scenario derivation.

Examples:

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
/work/envs/depth/bin/python train.py --scenario temperature --set training.trainer.max_epochs=100
```

Override paths are rooted at the super-config sections: `data.*`, `model.*`, `training.*`, and for inference helpers `inference.*`.

## Effective Config Snapshots

The resolver writes effective split YAMLs whenever a run directory is available:

- `config_original.yaml`: the original super-config snapshot
- `data_config_effective.yaml`: resolved data section wrapped for dataset constructors
- `model_config_effective.yaml`: resolved model section wrapped for model constructors
- `training_config_effective.yaml`: resolved training section
- `inference_config_effective.yaml`: inference-only snapshot for inference super-config loads

Training stores these under `logs/<timestamp>/`. Inference exporters store them in the output run directory or a temporary runtime directory, depending on the caller.

## Data Keys

These keys live under top-level `data` in both pixel super-configs.

| Key | Default | Meaning |
| --- | --- | --- |
| `data.dataset.core.dataset_variant` | `argo_geotiff_gridded` | Dataset implementation. The GeoTIFF workflow is the only supported dataset variant. |
| `data.dataset.core.dataloader_type` | `light` | Training runner expects the lightweight dataloader path. |
| `data.dataset.core.geotiff_root_dir` | `/work/data/OceanVariableReconstruction` | Packaged dataset root containing `manifest.yaml`, root-level `rasters/`, `argo/argo_profiles_on_grid.zarr`, and `masks/`. |
| `data.dataset.core.metadata_cache_dir` | `/work/data/OceanVariableReconstruction/depthdif_cache` | Patch/date metadata cache directory inside the packaged dataset root. |
| `data.dataset.grid.tile_size` | `128` | Patch height and width in pixels. |
| `data.dataset.grid.resolution_deg` | `0.1` | Horizontal grid resolution. |
| `data.dataset.grid.patch_grid_source` | `land_mask` | Builds patch origins from the configured land-mask GeoTIFF. |
| `data.dataset.grid.land_mask_path` | `masks/world_land_mask_glorys_0p1.tif` | Dataset-root-relative land/ocean mask used for patch selection and fallback support. |
| `data.dataset.grid.patch_stride` | `32` | Pixel stride between patch origins. `32` gives 75% overlap for 128-pixel tiles. |
| `data.dataset.grid.max_land_fraction` | `0.3` | Maximum land fraction allowed for default patch candidates. |
| `data.dataset.grid.force_include_regions` | named regional boxes | Relaxed patch-inclusion rules for specific ocean regions. |
| `data.dataset.sampling.temporal_window_days` | `7` | Centered ARGO/OSTIA/auxiliary window around each GLORYS date. |
| `data.dataset.sampling.glorys_var_name` | `thetao` | Dense GLORYS temperature target variable. |
| `data.dataset.sampling.ostia_var_name` | `analysed_sst` | Legacy OSTIA variable key used when `eo_source=ostia`. |
| `data.dataset.sampling.eo_source` | scenario-derived | Dense surface EO raster group: `ostia` for temperature/joint, `sss` for salinity. |
| `data.dataset.sampling.eo_var_name` | scenario-derived | Dense surface EO raster variable: `analysed_sst` for OSTIA, `sos` for SSS. |
| `data.dataset.selection.require_argo_for_train` | `false` | Drops train rows without ARGO support when enabled. |
| `data.dataset.selection.require_argo_for_val` | `true` | Drops validation rows without ARGO support. |
| `data.dataset.selection.require_argo_for_all` | `false` | Keeps no-ARGO rows for full-grid inference. |
| `data.dataset.synthetic.enabled` | `false` | If true, builds sparse `x` by sampling dense `y` instead of ARGO. |
| `data.dataset.synthetic.pixel_count` | `250` | Number of horizontal pixels sampled when synthetic mode is enabled. |
| `data.dataset.finetune_sampling.enabled` | `false` | Enables train-split hard-area row filtering for coastal finetuning. |
| `data.dataset.finetune_sampling.hard_fraction` | `0.75` | Target retained-row fraction from configured hard regions. |
| `data.dataset.finetune_sampling.apply_to_splits` | `[train]` | Splits affected by hard/easy row filtering; validation is unchanged by default. |
| `data.dataset.finetune_sampling.relax_land_filter` | `true` | Adds hard-region boxes as relaxed land-fraction regions before row filtering. |
| `data.dataset.finetune_sampling.default_max_land_fraction` | `0.85` | Land-fraction cap used for hard-region grid inclusion when a box has no override. |
| `data.dataset.finetune_sampling.hard_regions` | named regional boxes | Patch-center boxes used to classify hard finetuning rows. |
| `data.dataset.output.return_info` | `false` | Adds metadata under `batch['info']`. |
| `data.dataset.output.return_coords` | `true` | Adds patch-center coordinates under `batch['coords']`. Required for coordinate conditioning. |
| `data.dataset.output.fields` | scenario-derived | Physical fields loaded by the GeoTIFF dataset: `temperature`, `salinity`, or both. |
| `data.dataset.output.include_salinity` | scenario-derived | Enables salinity raster/profile support. Derived from scenario. |
| `data.dataset.runtime.random_seed` | `7` | Deterministic split/sampling seed. |
| `data.dataset.runtime.cache_size` | `8` | Maximum open raster/source cache size. |
| `data.split.val_year` | `2018` | Calendar year assigned to validation. Prevents spatial leakage when overlapping tiles are used. |
| `data.split.val_fraction` | `0.2` | Fallback validation fraction when no validation year is set. |
| `data.dataloader.num_workers` | `6` | Dataset-side dataloader worker default used by helper paths. |
| `data.dataloader.prefetch_factor` | `2` | Prefetched batches per worker when workers are enabled. |
| `data.dataloader.val_shuffle` | `true` | Validation loader shuffle remains enabled intentionally. |

## Model Keys

These keys live under top-level `model` in both pixel super-configs.

| Key | Default | Meaning |
| --- | --- | --- |
| `model.model_type` | `cond_px_dif` | Model selector: `cond_px_dif`, `latent_cond_dif`, or checkpoint-free `idw_baseline`. |
| `model.depth_channels` | `50` | Depth channels per active output field. Used by scenario derivation. |
| `model.resume_checkpoint` | `false` | `false`/`null` starts from scratch; a path resumes or warm-starts from that checkpoint. |
| `model.load_checkpoint_only` | `false` | When true, loads model weights only and reinitializes optimizer/trainer state. |
| `model.output_fields` | scenario-derived | Active predicted fields. Derived from scenario. |
| `model.generated_channels` | scenario-derived | Number of generated output channels. Derived from scenario. |
| `model.condition_channels` | scenario-derived | Total input channels to the denoiser. Derived from scenario and conditioning toggles. |
| `model.condition_mask_channels` | `1` | Number of valid-mask channels appended to conditioning when enabled. |
| `model.condition_include_eo` | `true` | Prepends the scenario-selected EO channel to model conditioning. |
| `model.condition_use_valid_mask` | `true` | Appends ARGO observation support to conditioning. |
| `model.condition_use_land_mask` | `true` | Appends GLORYS spatial support to conditioning. |
| `model.clamp_known_pixels` | `false` | Re-injects known values during reverse sampling when enabled. |
| `model.mask_loss_with_valid_pixels` | `true` | Restricts loss to task-valid support intersected with `land_mask`. |
| `model.coastal_loss.*` | `enabled=true`, `radius_px=5`, `weight=3.0`, `ramp=linear` | Upweights supervised ocean pixels within a configurable pixel radius of land. |
| `model.parameterization` | `x0` | Diffusion target, either `x0` or `epsilon`. |
| `model.log_intermediates` | `false` | Captures reverse-process intermediates when enabled by the caller. |
| `model.idw.*` | `power=2.0`, `eps=1e-6`, `chunk_size=4096` | IDW baseline controls used when `model.model_type=idw_baseline`; bands with no ARGO observations are emitted as nodata. |
| `model.ema.*` | enabled by default | Exponential moving average callback and validation-swap settings. |
| `model.ambient_occlusion.*` | disabled by default | Self-supervised occlusion objective controls. |
| `model.post_process.gaussian_blur.*` | disabled by default | Optional denormalized prediction blur. |
| `model.coord_conditioning.*` | enabled, date included | Coordinate/date FiLM conditioning controls. |
| `model.unet.*` | `dim=64`, `dim_mults=[1,2,4,8]` | ConvNeXt U-Net width/depth and output behavior. |

## Training Keys

These keys live under top-level `training` in `training_super_config.yaml` and are also present in `inference_super_config.yaml` so checkpoints can rebuild the model consistently.

| Key | Default | Meaning |
| --- | --- | --- |
| `training.training.lr` | `1.0e-4` | Optimizer learning rate. |
| `training.training.batch_size` | `4` | Informational batch size; `training.dataloader.batch_size` is the dataloader source of truth. |
| `training.training.noise.num_timesteps` | `1000` | Diffusion training timesteps. |
| `training.training.noise.schedule` | `cosine` | Noise schedule: `linear`, `cosine`, `quadratic`, or `sigmoid`. |
| `training.training.noise.beta_start` | `1.0e-4` | First-step beta for schedules that use explicit endpoints. |
| `training.training.noise.beta_end` | `2.0e-2` | Final beta for schedules that use explicit endpoints. |
| `training.training.validation_sampling.sampler` | `ddpm` | Validation reconstruction sampler. |
| `training.training.validation_sampling.ddim_num_timesteps` | `100` | DDIM step count when the sampler is `ddim`. |
| `training.training.validation_sampling.ddim_eta` | `0.0` | DDIM stochasticity. |
| `training.training.validation_sampling.ddim_temperature` | `1.0` | Reverse-process noise scale. |
| `training.training.validation_sampling.max_full_reconstruction_samples` | `5` | Cap for expensive full-reconstruction validation examples. |
| `training.trainer.max_epochs` | `1500` | Lightning epoch cap. |
| `training.trainer.accelerator` / `devices` | `auto` / `auto` | Lightning device selection. |
| `training.trainer.precision` | `16-mixed` | Mixed-precision mode. |
| `training.trainer.ckpt_monitor` | `val/loss_ckpt` | Best-checkpoint metric. |
| `training.trainer.val_check_interval` | `0.1` | Validation cadence within each epoch. |
| `training.trainer.limit_val_batches` | `64` | Validation batches per validation run. |
| `training.trainer.gradient_clip_val` | `1.0` | Gradient clipping threshold. |
| `training.wandb.*` | project/run/logging defaults | W&B project, run naming, watch, scalar, and image logging controls. |
| `training.dataloader.batch_size` | `4` | Training dataloader batch size. |
| `training.dataloader.val_batch_size` | `2` | Validation dataloader batch size. |
| `training.dataloader.num_workers` | `6` | Training dataloader workers. |
| `training.dataloader.val_num_workers` | `0` | Validation dataloader workers. |
| `training.dataloader.shuffle` | `true` | Training shuffle. |
| `training.dataloader.val_shuffle` | `true` | Validation shuffle. This is intended behavior. |
| `training.dataloader.pin_memory` | `true` | Enables pinned host memory. |
| `training.scheduler.warmup.*` | disabled by default | Optional step-based linear warmup. |
| `training.scheduler.reduce_on_plateau.*` | enabled by default | Plateau LR scheduler settings. |

## Inference Keys

These keys live under top-level `inference` in `inference_super_config.yaml`.

| Key | Default | Meaning |
| --- | --- | --- |
| `inference.grid.patch_stride` | `96` | Inference-time patch stride override. Smaller values increase overlap and runtime. |
| `inference.grid.min_ocean_fraction` | `0.05` | Minimum ocean fraction for inference patch selection. |
| `inference.grid.land_mask_path` | `masks/world_land_mask_glorys_0p1.tif` | Dataset-root-relative land-mask grid used by inference patch selection and final cleanup. |
| `inference.dataloader.batch_size` | `64` | Prediction batch size. |
| `inference.dataloader.num_workers` | `6` | Prediction dataloader workers. |
| `inference.dataloader.prefetch_factor` | `2` | Prefetched batches per prediction worker. |

Export scripts may expose CLI flags such as `--patch-stride` or `--min-ocean-fraction`; those one-off flags override the inference super-config for that run.

## Runtime Mapping Notes

- `x_valid_mask` is ARGO observation support; it is collapsed to one conditioning channel when `condition_mask_channels=1`.
- `land_mask` is GLORYS spatial/domain support and gates loss when mask-based loss is enabled.
- `output_land_mask` is an optional predict-time cleanup overlay, not a training dataloader key.
- For salinity-only runs, the dataloader skips temperature tensors and returns only `x_salinity`, `y_salinity`, and their salinity masks.
- For joint runs, temperature channels come first, followed by salinity channels.
- Existing checkpoints are only shape-compatible with runs that use the same scenario-derived channel counts.
