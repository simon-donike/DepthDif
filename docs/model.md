# Model  
DepthDif uses a conditional pixel-space diffusion model implemented in `src/depth_recon/models/diffusion/PixelDiffusion.py`.  
  
Model schema:  
![depthdif_schema](assets/figures/depthdif_schema.png)  
  
Core stack:  
- Lightning wrapper: `PixelDiffusionConditional`  
- diffusion core: `DenoisingDiffusionConditionalProcess`  
- denoiser backbone: `UnetConvNextBlock` (ConvNeXt-style U-Net)  
  
The model learns to generate `y` while conditioning on observed channels (`x`), optional `eo`, ARGO observation support (`x_valid_mask`), and GLORYS spatial support (`land_mask`).  
  
## Conditioning Setup  
Three conditioning layouts are supported by code/config:  
  
- Single-band task: `x -> y`  
- EO multiband task: `[eo, x, x_valid_mask, land_mask] -> y`  
- Joint temperature/salinity task: `[eo, cat(x, x_salinity), collapsed x_valid_mask, land_mask] -> cat(y, y_salinity)` when `model.output_fields=["temperature", "salinity"]` and the dataloader includes salinity keys.  
  
Condition assembly happens in `_prepare_condition_for_model`:  
- optionally prepend `eo` (`condition_include_eo=true`)  
- append data channels from `x`  
- optionally append ARGO observation-support `x_valid_mask` channels (`condition_use_valid_mask=true`)  
- optionally append GLORYS spatial-support `land_mask` (`condition_use_land_mask=true`)  
- enforce channel count equals `model.condition_channels`  
  
## Architecture Summary  
`UnetConvNextBlock` follows a U-Net encoder/decoder with ConvNeXt blocks and linear attention.  
  
With default `dim_mults=[1,2,4,8]`:  
- 4 downsampling stages  
- bottleneck block with attention  
- 3 upsampling stages with skip connections  
- final ConvNeXt block + `1x1` output conv to `generated_channels`  

For the ambient EO preset in `src/depth_recon/configs/px_space/model_config_ambient.yaml`, the U-Net base width is `dim: 64`. This keeps the same depth (`dim_mults=[1,2,4,8]`) while matching the current 50 generated channels + 53 condition channels.  
  
Time conditioning:  
- sinusoidal timestep embedding -> MLP -> additive bias in ConvNeXt blocks  
  
Coordinate/date conditioning (when enabled):  
- per-channel FiLM scale/shift in ConvNeXt blocks  
- details in [Data + Coordinate Injection](data-coordinate-injection.md)  
  
## Training Objective  
Training step (`training_step`) calls conditional diffusion `p_loss` on normalized model-output tensors. By default this is temperature only; joint mode stacks normalized temperature and salinity channels before loss computation. The dataset still returns temperature and salinity as separate keys; stacking is owned by `PixelDiffusionConditional`.  
  
Behavior:  
- sample random timestep `t`  
- forward diffuse the selected training target to the noisy target branch  
- predict either:  
  - noise (`epsilon` parameterization), or  
  - clean sample (`x0` parameterization)  
  
Loss options:  
- unmasked MSE (default behavior when masking disabled)  
- masked MSE with mode-specific supervision support:  
  - standard mode: over `y_valid_mask` intersected with GLORYS `land_mask` on the full `y` target  
  - ambient mode: over `x_valid_mask` intersected with `y_valid_mask` and GLORYS `land_mask` on the degraded `x` target  
  - the common on-disk mask is not loaded by train/validation dataloaders; optional `output_land_mask` is only final prediction cleanup support  
  
Ambient occlusion objective (`model.ambient_occlusion.enabled: true`):  
- sample an additional Bernoulli keep-mask over already observed pixels (`~A = B * A`)  
- feed the model a further-corrupted condition (`x_tilde = x * ~A`) and `~A` as condition mask  
- switch the diffusion target from `y` to the original sparse-observation tensor `x`  
- optionally apply `~A` to noisy target branch during `p_loss` (`~A * x_t`)  
- compute masked MSE on the originally valid `x` support intersected with valid `y` support and GLORYS `land_mask` (`A ∩ Y ∩ land_mask`, not `~A`)  
- detailed walkthrough and citation: [Ambient Occlusion Objective](ambient-occlusion-objective.md)  
  
Current EO config (`src/depth_recon/configs/px_space/model_config.yaml`) uses:  
- `parameterization: "x0"`  
- `mask_loss_with_valid_pixels: true`  

## Joint Temperature + Salinity Mode  

Joint pixel training requires two independent opt-ins:  

- data side: `dataset.output.include_salinity=true`, so batches contain `x_salinity`, `y_salinity`, `x_salinity_valid_mask`, `y_salinity_valid_mask`, and `x_salinity_valid_mask_1d`  
- model side: `model.output_fields=["temperature", "salinity"]`, so `PixelDiffusionConditional` requires those keys and stacks them internally  

`train.py` validates this pairing before dataloaders are built. If the model asks  
for salinity while the data flag is still false, startup fails with a clear  
configuration error instead of failing later inside a training step.  

The preset is `src/depth_recon/configs/px_space/model_config_joint_temp_salinity.yaml`:  

- `generated_channels: 100` for 50 temperature + 50 salinity target channels  
- `condition_channels: 103` for OSTIA + 100 stacked sparse ARGO channels + one collapsed `x_valid_mask` support channel + one GLORYS `land_mask` channel  
- `condition_mask_channels: 1`, so the stacked 100-channel support mask is collapsed with `amax` into one spatial support channel  
- existing 50-channel temperature checkpoints are not shape-compatible with this preset  

Inside the model path, `_prepare_model_batch_tensors` builds:  

```text
x_model            = cat([x, x_salinity], dim=1)
y_model            = cat([y, y_salinity], dim=1)
x_valid_mask_model = cat([x_valid_mask, x_salinity_valid_mask], dim=1)
y_valid_mask_model = cat([y_valid_mask, y_salinity_valid_mask], dim=1)
```

The diffusion core remains channel-agnostic and sees one 100-channel normalized  
target. The loss is the existing unweighted diffusion MSE over the stacked  
normalized channels, masked by the stacked task-valid support intersected with GLORYS `land_mask` when mask-based  
loss is enabled. `predict_step` splits sampled outputs back into temperature and  
salinity fields and denormalizes them with their own normalization helpers.  

For non-ambient training, the loss is pulled over all valid target pixels via  
`y_valid_mask ∩ land_mask`, or the stacked target-valid mask intersected with `land_mask` in joint temperature/salinity mode.  

Latent model workflow is configured via `src/depth_recon/configs/lat_space/model_config.yaml` with AE controls in `src/depth_recon/configs/lat_space/ae_config.yaml`; see [Autoencoder + Latent Diffusion](autoencoder.md) for the full setup.  
  
EMA weight averaging can be enabled through `model.ema`; see [Exponential Moving Average Weights](ema.md) for the implementation details, validation logging behavior, and metric definitions.  
  
## Inference Flow  
Prediction entry point is `predict_step`.  
  
At inference:  
- build condition tensor from batch inputs  
- start reverse process from Gaussian latent  
- keep condition fixed during reverse sampling  
- use configured sampler (`ddpm` by default, `ddim` optional)  
- optional known-pixel clamping can overwrite known pixels each step  
  
Output dictionary from `predict_step`:  
- `y_hat`: normalized model output; stacked in joint mode  
- `y_hat_denorm`: denormalized temperature output, kept as a legacy alias  
- `y_hat_temperature` / `y_hat_salinity`: normalized split outputs in joint mode  
- `y_hat_temperature_denorm` / `y_hat_salinity_denorm`: Celsius and PSU split outputs in joint mode  
- `denoise_samples`: optional intermediate reverse samples  
- `x0_denoise_samples`: optional per-step `x0` predictions  
- `sampler`: sampler object used  
  
## Post-Processing in Lightning Inference  
After denormalization, inference can apply:  
- optional Gaussian blur (`model.post_process.gaussian_blur.*`)  
- direct `y` prediction: keep the generated field and set `y_valid_mask==0` pixels to `NaN`  
- ambient `x` completion: return the model prediction as-is after optional sampler-time `clamp_known_pixels`, then set `y_valid_mask==0` pixels to `NaN`  

This post-processing is centralized in `predict_step`.  
  
## Validation Diagnostics  
Validation computes two paths:  
- per-batch validation loss (`validation_step`) using the same objective as training  
- one full reverse-diffusion reconstruction per validation run from the global-rank-0 cached first validation batch (`on_validation_epoch_end`)  

When available, full reconstruction logging includes:  
- MSE  
- PSNR/SSIM (if `skimage` is installed)  
- qualitative reconstruction grid  
- denoising-intermediate grid and MAE-vs-step curve (when intermediates enabled)  
- reconstruction plotting keeps the unmerged model prediction panel and masks invalid output support through `y_valid_mask`  
- joint mode logs a separate salinity reconstruction grid under `val_salinity_imgs` and separate per-band PSU L1 charts under `val_salinity_absolute_band_error` using a blue-to-green color scale  
- these epoch-end diagnostics stay rank-local on global rank 0 to avoid DDP logging mismatches for optional metrics like PSNR/SSIM  
  
