# Model
DepthDif uses a conditional pixel-space diffusion model implemented in `models/difFF/PixelDiffusion.py`.

Model schema:  
![depthdif_schema](assets/depthdif_schema.png)

Core stack:
- Lightning wrapper: `PixelDiffusionConditional`
- diffusion core: `DenoisingDiffusionConditionalProcess`
- denoiser backbone: `UnetConvNextBlock` (ConvNeXt-style U-Net)

The model learns to generate `y` while conditioning on observed channels (`x`, optional `eo`, optional mask channels).

## Conditioning Setup
Two conditioning layouts are supported by code/config:

- Single-band task: `x -> y`
- EO multiband task: `[eo, x, valid_mask] -> y`

Condition assembly happens in `_prepare_condition_for_model`:
- optionally prepend `eo` (`condition_include_eo=true`)
- append data channels from `x`
- optionally append valid-mask channels (`condition_use_valid_mask=true`)
- enforce channel count equals `model.condition_channels`

## Architecture Summary
`UnetConvNextBlock` follows a U-Net encoder/decoder with ConvNeXt blocks and linear attention.

With default `dim_mults=[1,2,4,8]`:
- 4 downsampling stages
- bottleneck block with attention
- 3 upsampling stages with skip connections
- final ConvNeXt block + `1x1` output conv to `generated_channels`

Time conditioning:
- sinusoidal timestep embedding -> MLP -> additive bias in ConvNeXt blocks

Coordinate/date conditioning (when enabled):
- per-channel FiLM scale/shift in ConvNeXt blocks
- details in [Date + Coordination Injection](date-coordination-injection.md)

## Training Objective
Training step (`training_step`) calls conditional diffusion `p_loss` on standardized temperature tensors.

Behavior:
- sample random timestep `t`
- forward diffuse `y` to noisy target branch
- predict either:
  - noise (`epsilon` parameterization), or
  - clean sample (`x0` parameterization)

Loss options:
- unmasked MSE (default behavior when masking disabled)
- masked MSE over missing pixels (`1 - valid_mask`) with optional ocean gating via `land_mask`

Current EO config (`configs/model_config.yaml`) uses:
- `parameterization: "x0"`
- `mask_loss_with_valid_pixels: true`

## Inference Flow
Prediction entry point is `predict_step`.

At inference:
- build condition tensor from batch inputs
- start reverse process from Gaussian latent
- keep condition fixed during reverse sampling
- use configured sampler (`ddpm` by default, `ddim` optional)
- optional known-pixel clamping can overwrite known pixels each step

Output dictionary from `predict_step`:
- `y_hat`: standardized model output
- `y_hat_denorm`: denormalized output
- `denoise_samples`: optional intermediate reverse samples
- `x0_denoise_samples`: optional per-step `x0` predictions
- `sampler`: sampler object used

## Post-Processing in Lightning Inference
After denormalization, inference can apply:
- optional Gaussian blur (`model.post_process.gaussian_blur.*`)
- merge observed pixels from `x` (where `valid_mask=1`) with generated pixels (where `valid_mask=0`)
- zero land pixels using `land_mask`

This post-processing is centralized in `predict_step`.

## Validation Diagnostics
Validation computes two paths:
- per-batch validation loss (`validation_step`) using the same objective as training
- one full reverse-diffusion reconstruction per epoch from cached first validation batch (`on_validation_epoch_end`)

When available, full reconstruction logging includes:
- MSE
- PSNR/SSIM (if `skimage` is installed)
- qualitative reconstruction grid
- denoising-intermediate grid and MAE-vs-step curve (when intermediates enabled)
- reconstruction plotting applies `land_mask` and does not copy observed `valid_mask` pixels into the displayed prediction panel
