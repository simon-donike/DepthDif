# Configs
This page documents the default configuration files in `configs/` using key paths, default values, and short explanations aligned with comments in each file.

## `configs/data_config.yaml`
Dataset settings are grouped by intent (`core`, `source`, `validity`, `degradation`, `conditioning`, `augmentation`, `output`, `runtime`).  

| Config key | Default value | Explanation |
|---|---|---|
| `dataset.core.dataset_variant` | `"eo_4band"` | Selects `SurfaceTempPatch4BandsLightDataset` in `train.py`. |
| `dataset.core.dataloader_type` | `"light"` | `"raw"` loads NetCDF on-the-fly, `"light"` uses CSV + saved patch paths. |
| `dataset.source.light_index_csv` | `"/work/data/depth/4_bands_v2/patch_index_with_paths_split.csv"` | Index CSV used when `dataloader_type="light"`. |
| `dataset.source.bands` | `["thetao"]` | Variables/bands to extract and write (order preserved). |
| `dataset.source.edge_size` | `128` | Patch size in pixels; also used as stride for non-overlapping tiles. |
| `dataset.validity.max_nodata_fraction` | `0.15` | Maximum invalid/nodata ratio per tile when validity filtering is enabled. |
| `dataset.validity.nan_fill_value` | `0.0` | Fill value used for invalid/land pixels before tensor conversion. |
| `dataset.validity.valid_from_fill_value` | `true` | In light mode, infer valid mask from standardized fill value. |
| `dataset.validity.enforce_validity` | `true` | Drops indexed tiles with too much nodata using `max_nodata_fraction`. |
| `dataset.degradation.mask_fraction` | `0.975` | Fraction of pixels hidden in `x`; streak generation continues until this corruption target is reached. |
| `dataset.degradation.mask_strategy` | `"tracks"` | Corruption strategy (`"tracks"` = continuous curved submarine-like streaks, `"rectangles"` = legacy fallback). |
| `dataset.degradation.mask_patch_min` | `2` | Minimum rectangle patch side length (pixels) for legacy `mask_strategy="rectangles"`. |
| `dataset.degradation.mask_patch_max` | `5` | Maximum rectangle patch side length (pixels) for legacy `mask_strategy="rectangles"`. |
| `dataset.conditioning.eo_dropout_prob` | `0.50` | Probability of zeroing EO conditioning per sample (train and val). |
| `dataset.conditioning.eo_random_scale_enabled` | `false` | If enabled, applies additive EO offset in `[-2.0, 2.0]` (temperature units). |
| `dataset.conditioning.eo_speckle_noise_enabled` | `true` | If enabled, applies multiplicative EO speckle noise clamped to `[0.9, 1.1]`. |
| `dataset.augmentation.enable_transform` | `false` | Enables random geometric augmentation. |
| `dataset.output.x_return_mode` | `"currupted_plus_mask"` | Return mode for `x` (`"corrputed"` or `"currupted_plus_mask"` in file comments). |
| `dataset.output.return_info` | `false` | Returns per-sample metadata under `batch["info"]`. |
| `dataset.output.return_coords` | `true` | Returns patch-center coordinates under `batch["coords"]`. |
| `dataset.runtime.rebuild_index` | `false` | Rebuilds tile index from raw files on startup. |
| `dataset.runtime.random_seed` | `7` | Seed used for deterministic split and random dataset sampling behavior. |
| `split.val_fraction` | `0.2` | Fraction of dataset reserved for validation. |

## `configs/model_config.yaml`
| Config key | Default value | Explanation |
|---|---|---|
| `model.model_type` | `"cond_px_dif"` | Model type (conditional diffusion). |
| `model.resume_checkpoint` | `false` | `false/null` starts from scratch; checkpoint path resumes training. |
| `model.generated_channels` | `3` | Number of predicted target channels. |
| `model.condition_channels` | `5` | Condition channel count: EO + corrupted target + valid mask. |
| `model.condition_mask_channels` | `1` | Number of valid-mask condition channels. |
| `model.condition_include_eo` | `true` | Includes `batch["eo"]` as condition input. |
| `model.condition_use_valid_mask` | `true` | Includes valid mask in condition input. |
| `model.clamp_known_pixels` | `false` | Clamps known pixels each reverse step for inpainting-style stability. |
| `model.mask_loss_with_valid_pixels` | `true` | Computes loss on missing pixels (`1-valid_mask`) with optional gating. |
| `model.parameterization` | `"x0"` | Diffusion training target (`"epsilon"` or `"x0"`). |
| `model.log_intermediates` | `true` | Default validation intermediate logging behavior. |
| `model.post_process.gaussian_blur.enabled` | `false` | Enables final denormalized Gaussian blur post-process. |
| `model.post_process.gaussian_blur.sigma` | `0.5` | Gaussian blur sigma in pixels. |
| `model.post_process.gaussian_blur.kernel_size` | `3` | Blur kernel size; even values are adjusted to odd. |
| `model.coord_conditioning.enabled` | `true` | Enables coordinate conditioning with FiLM. |
| `model.coord_conditioning.encoding` | `"unit_sphere"` | Coordinate encoding type (`"unit_sphere"`, `"sincos"`, `"raw"`). |
| `model.coord_conditioning.include_date` | `true` | Includes date encoding with coordinates. |
| `model.coord_conditioning.date_encoding` | `"day_of_year_sincos"` | Date encoding mode (day-of-year sin/cos, denominator 365). |
| `model.coord_conditioning.embed_dim` | `null` | FiLM embedding dimension; defaults to `unet.dim` when null. |
| `model.unet.dim` | `64` | Base channel width of U-Net denoiser. |
| `model.unet.dim_mults` | `[1, 2, 4, 8]` | Per-stage width multipliers; controls depth/width scaling. |
| `model.unet.with_time_emb` | `true` | Enables timestep embeddings in denoiser. |
| `model.unet.output_mean_scale` | `false` | Optional output mean correction for diffusion variants. |
| `model.unet.residual` | `false` | If enabled, predicts residual added to input. |

## `configs/training_config.yaml`
| Config key | Default value | Explanation |
|---|---|---|
| `training.lr` | `1.0e-4` | Optimizer learning rate. |
| `training.batch_size` | `24` | Informational training batch size (dataloader section is source of truth). |
| `training.noise.num_timesteps` | `1000` | Number of diffusion timesteps. |
| `training.noise.schedule` | `"cosine"` | Noise schedule: `linear`, `cosine`, `quadratic`, `sigmoid`. |
| `training.noise.beta_start` | `1.0e-4` | First-step noise level (must be positive and below `beta_end`). |
| `training.noise.beta_end` | `2.0e-2` | Final-step noise level (must be below `1` and above `beta_start`). |
| `training.validation_sampling.sampler` | `"ddpm"` | Validation sampler (`ddpm` full chain, `ddim` faster). |
| `training.validation_sampling.ddim_num_timesteps` | `200` | DDIM steps when `sampler="ddim"`. |
| `training.validation_sampling.ddim_eta` | `0.0` | DDIM eta; `0.0` is deterministic DDIM. |
| `training.validation_sampling.log_intermediates` | `true` | Captures/logs denoising intermediate images in validation. |
| `training.validation_sampling.skip_full_reconstruction_in_sanity_check` | `false` | Skips expensive full reconstruction during Lightning sanity checks when true. |
| `training.validation_sampling.max_full_reconstruction_samples` | `4` | Max first-batch val samples used for full reconstruction pass. |
| `trainer.max_epochs` | `5000` | Maximum training epochs. |
| `trainer.accelerator` | `"auto"` | Lightning accelerator backend selection. |
| `trainer.devices` | `"auto"` | Device selection (`auto`, int, list). |
| `trainer.num_gpus` | `2` | Legacy explicit GPU count override. |
| `trainer.strategy` | `"auto"` | Distributed strategy selection. |
| `trainer.precision` | `"16-mixed"` | Mixed precision mode. |
| `trainer.matmul_precision` | `"high"` | `torch.set_float32_matmul_precision` mode. |
| `trainer.suppress_accumulate_grad_stream_mismatch_warning` | `true` | Suppresses PyTorch stream mismatch warning noise. |
| `trainer.suppress_lightning_pytree_warning` | `true` | Suppresses Lightning LeafSpec deprecation warning noise. |
| `trainer.ckpt_monitor` | `"val/loss_ckpt"` | Metric monitored for best-checkpoint saving. |
| `trainer.lr_logging_interval` | `"step"` | Learning-rate logging cadence (`step` or `epoch`). |
| `trainer.log_every_n_steps` | `1` | Trainer logging interval in steps. |
| `trainer.num_sanity_val_steps` | `2` | Number of startup sanity-validation steps. |
| `trainer.val_batches_per_epoch` | `200` | Absolute cap on validation batches per epoch. |
| `trainer.limit_val_batches` | `4` | Number/fraction of validation batches per epoch. |
| `trainer.enable_model_summary` | `true` | Enables Lightning model summary printout. |
| `trainer.gradient_clip_val` | `1.0` | Gradient clipping threshold (`0.0` disables). |
| `trainer.rebuild_index` | `false` | Compatibility field; dataset config controls index rebuilding. |
| `wandb.project` | `"DepthDif_Simon"` | W&B project name. |
| `wandb.entity` | `"esa-phi-lab"` | W&B entity/team (null uses default account). |
| `wandb.run_name` | `"eo_4band_cond"` | Explicit run name. |
| `wandb.log_model` | `"false"` | W&B model artifact logging policy. |
| `wandb.verbose` | `true` | Enables extra metric/image logging. |
| `wandb.watch_gradients` | `false` | Enables gradient history logging via `wandb.watch`. |
| `wandb.watch_parameters` | `false` | Enables parameter history logging via `wandb.watch`. |
| `wandb.watch_log_freq` | `100` | `wandb.watch` logging frequency in steps. |
| `wandb.watch_log_graph` | `false` | Logs computation graph when watch is enabled. |
| `wandb.log_stats_every_n_steps` | `100` | Step interval for scalar debug stats. |
| `wandb.log_images_every_n_steps` | `10` | Step interval for validation preview images. |
| `dataloader.batch_size` | `24` | Training dataloader batch size. |
| `dataloader.val_batch_size` | `4` | Validation batch size (falls back to `batch_size` if omitted). |
| `dataloader.num_workers` | `6` | Number of training dataloader workers. |
| `dataloader.val_num_workers` | `0` | Validation workers (`0` avoids h5netcdf sanity-check instability). |
| `dataloader.persistent_workers` | `false` | Keeps train workers alive across epochs when true. |
| `dataloader.val_persistent_workers` | `false` | Validation worker persistence (when `val_num_workers > 0`). |
| `dataloader.prefetch_factor` | `4` | Prefetched batches per worker (only used when workers > 0). |
| `dataloader.shuffle` | `true` | Shuffles training dataset each epoch. |
| `dataloader.val_shuffle` | `true` | Shuffles validation set (often used with limited val batches). |
| `dataloader.pin_memory` | `false` | Enables pinned host memory for faster H2D transfer. |
| `scheduler.warmup.enabled` | `true` | Enables linear warmup before plateau scheduling. |
| `scheduler.warmup.steps` | `15000` | Warmup step count to ramp LR from `start_ratio` to base LR. |
| `scheduler.warmup.start_ratio` | `0.2` | Initial warmup LR as ratio of `training.lr`. |
| `scheduler.reduce_on_plateau.enabled` | `true` | Enables `ReduceLROnPlateau`. |
| `scheduler.reduce_on_plateau.monitor` | `"val/loss_ckpt"` | Metric monitored for LR reduction. |
| `scheduler.reduce_on_plateau.mode` | `"min"` | Plateau mode (`min` or `max`). |
| `scheduler.reduce_on_plateau.factor` | `0.5` | Multiplicative LR decay factor on plateau. |
| `scheduler.reduce_on_plateau.patience` | `15` | Validation epochs with no improvement before reducing LR. |
| `scheduler.reduce_on_plateau.threshold` | `1.0e-4` | Minimum significant metric change. |
| `scheduler.reduce_on_plateau.threshold_mode` | `"rel"` | Threshold mode (`rel` or `abs`). |
| `scheduler.reduce_on_plateau.cooldown` | `0` | Epoch cooldown after LR reduction. |
| `scheduler.reduce_on_plateau.min_lr` | `0.0` | Lower bound for LR. |
| `scheduler.reduce_on_plateau.eps` | `1.0e-8` | Minimum effective LR change. |
