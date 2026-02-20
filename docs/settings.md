# A1: Major and Minor Model Settings

These are the model/training behaviors in this repo and where they are wired in config.

## Major model settings

- **Input channels with mask (conditioning channels)**  
  The conditional diffusion model receives corrupted data plus mask channels.  
  Configure in `configs/model_config.yaml`:
  - `model.condition_channels`: total conditioning channels passed to the denoiser (data + mask).
  - `model.condition_mask_channels`: number of condition channels that are mask semantics.
  - `model.condition_include_eo`: if true, prepend `eo` as an additional condition channel when present.
  - `model.condition_use_valid_mask`: if false, keep `valid_mask` out of condition input (still available for masked loss/logging).  
  In EO 3-band mode this is set to 5 condition channels total: `eo (1) + x (3) + mask (1)`.

- **EO dropout (conditioning regularization)**  
  Configure in `configs/data_config.yaml` or `configs/data_config_eo_4band.yaml`:
  - `dataset.eo_dropout_prob` (for example `0.25`)  
  EO is randomly zeroed for the configured fraction of samples in both train and val.  
  This is used to prevent EO over-reliance and encourage reconstruction from the sparse/deeper `x` signal itself.

- **Masked pixel loss computation**  
  Loss can be restricted to valid (ocean) pixels to avoid land/no-data bias.  
  Configure in `configs/model_config.yaml`:
  - `model.mask_loss_with_valid_pixels: true`  
  When enabled, the conditional loss is computed over missing pixels (`1 - valid_mask`),
  optionally restricted to ocean pixels via `land_mask`, then normalized by mask sum.

- **Anchoring known pixels at inference (inpainting clamp)**  
  During sampling, known pixels are overwritten at every diffusion step for stability.  
  Configure in `configs/model_config.yaml`:
  - `model.clamp_known_pixels: true`  
  This uses the conditioning input + mask to keep known values fixed while denoising.

  ![img](assets/clamped_pixels.png)  

- **Coordinate encoding + FiLM injection**  
  Patch-center coordinates are encoded and injected via FiLM scale/shift in ConvNeXt blocks.  
  Configure in:
  - `configs/data_config.yaml`: `dataset.return_coords: true` (adds coords to each batch)
  - `configs/model_config.yaml`: `model.coord_conditioning.enabled: true`
  - `model.coord_conditioning.encoding`: `unit_sphere`, `sincos`, or `raw`
  - `model.coord_conditioning.include_date`: include `batch["date"]` (`YYYYMMDD`) in FiLM conditioning
  - `model.coord_conditioning.date_encoding`: currently `day_of_year_sincos` (fixed denominator `365`)
  - `model.coord_conditioning.embed_dim`: embedding width (defaults to `unet.dim` if null)
  Exact mechanism is described in [Date + Coordination Injection](date-coordination-injection.md).

## Minor model/training settings

- **Learning-rate scheduler (Warmup + ReduceLROnPlateau)**  
  Configure in `configs/training_config.yaml`:
  - `scheduler.warmup.enabled`, `steps`, `start_ratio`
  - `scheduler.reduce_on_plateau.enabled`
  - `scheduler.reduce_on_plateau.monitor`, `mode`, `factor`, `patience`, `threshold`, `cooldown`
  - `trainer.lr_logging_interval` (learning-rate monitor cadence)
  Notes:
  - Warmup is step-based (optimizer updates), not epoch-based.
  - Default warmup ramps LR linearly from `0.1 * training.lr` to `training.lr` over the first `1000` optimizer steps.
  - Warmup is currently wired in the conditional model path; after warmup, `ReduceLROnPlateau` controls LR as usual.

- **Checkpointing + resume**  
  Configure in:
  - `configs/model_config.yaml`: `model.resume_checkpoint` (false/null or a `.ckpt` path)
  - `configs/training_config.yaml`: `trainer.ckpt_monitor` (metric used for `best.ckpt`; `last.ckpt` is always saved)

- **W&B logging (metrics/images/watch)**  
  Configure in `configs/training_config.yaml`:
  - `wandb.project`, `wandb.entity`, `wandb.run_name`, `wandb.log_model`
  - `wandb.verbose`, `wandb.log_images_every_n_steps`, `wandb.log_stats_every_n_steps`
  - `wandb.watch_gradients`, `wandb.watch_parameters`, `wandb.watch_log_freq`, `wandb.watch_log_graph`

- **Validation-time sampling + diagnostics**  
  Configure in `configs/training_config.yaml`:
  - `training.validation_sampling.sampler` (`ddpm` or `ddim`)
  - `training.validation_sampling.ddim_num_timesteps`, `ddim_eta`
  Notes: PSNR/SSIM are computed when `skimage` is available; one cached val example per epoch is used for full reconstruction logging.
