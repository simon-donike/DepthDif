# Baselines

DepthDif baseline models live in `src/depth_recon/models/baselines/`. They are Lightning modules and share the same dataloader-facing batch contract and `predict_step` output contract as the diffusion model. This keeps training, inference, global export, uncertainty export, and comparison-globe tooling on the same path where possible.

The currently supported baseline selectors are:

| `model.model_type` | Model | Trainable | Checkpoint required |
| --- | --- | ---: | ---: |
| `idw_baseline` | `IDWInterpolationBaseline` | No | No |
| `lstm_baseline` | `PointwiseLSTMBaseline` | Yes | Yes |
| `cnn_baseline` | `ProfileCNNInfillingBaseline` | Yes | Yes |
| `unet_baseline` | `UNetInfillingBaseline` | Yes | Yes |
| `unet2d_baseline` | `UNet2DInfillingBaseline` | Yes | Yes |

## Shared Batch Contract

All baselines use the existing GeoTIFF dataloaders unchanged. The dataset still returns sparse ARGO/profile observations, dense GLORYS targets, masks, and optional EO surface context with field-specific keys.

Temperature mode uses:

```text
x
y
x_valid_mask
y_valid_mask
eo
land_mask
```

Salinity mode uses:

```text
x_salinity
y_salinity
x_salinity_valid_mask
y_salinity_valid_mask
eo
land_mask
```

Joint mode keeps temperature first and salinity second. The baseline helper path stacks model-facing tensors as:

```text
x_model            = cat([x, x_salinity], dim=1)
y_model            = cat([y, y_salinity], dim=1)
x_valid_mask_model = cat([x_valid_mask, x_salinity_valid_mask], dim=1)
y_valid_mask_model = cat([y_valid_mask, y_salinity_valid_mask], dim=1)
```

`land_mask` is always used for loss support and output cleanup. The point-wise LSTM does not use it as an input feature; the profile CNN and 3D U-Net can use it as configured conditioning. `output_land_mask`, when present during export, is only used for final prediction masking.

## IDW Baseline

`IDWInterpolationBaseline` is a non-learned inverse-distance weighting baseline selected with:

```yaml
model:
  model_type: idw_baseline
  idw:
    power: 2.0
    eps: 1.0e-6
    chunk_size: 4096
```

For each active field and depth band, IDW reconstructs a dense 2D patch from sparse observed pixels in that same band. It does not use EO, neighboring depth channels, temporal context, or learned parameters.

For one band:

```text
observed sparse pixels in x[:, band]
observed support in x_valid_mask[:, band]
-> inverse-distance weighted 2D interpolation
-> y_hat[:, band]
```

Observed pixels are preserved exactly. If a band has no ARGO/profile observations in a patch, the output for that band is `NaN`, so global raster export writes nodata.

Because IDW has no trainable weights:

- `configure_optimizers()` returns no optimizers.
- `model_requires_checkpoint(...)` is false.
- Inference can run without loading a checkpoint.
- `uncertainty_step` returns deterministic zero uncertainty.

## Point-Wise LSTM Baseline

`PointwiseLSTMBaseline` is selected with:

```yaml
model:
  model_type: lstm_baseline
  lstm:
    hidden_size: 64
    num_layers: 2
    dropout: 0.0
    bidirectional: true
    weight_decay: 0.0
```

The learning problem is intentionally point-wise. Each spatial pixel is treated as an independent vertical profile. The model does not see neighboring pixels, convolutions, coordinate features, or patch-level context.

For one field with shape `(B, C, H, W)`, the model reshapes internally:

```text
(B, C, H, W)
-> (B * H * W, C, features)
-> field-specific LSTM over depth
-> linear head at each depth step
-> (B, C, H, W)
```

At each pixel and depth step, the LSTM sees:

```text
[
  sparse normalized profile value,
  observed-depth mask,
  normalized depth coordinate,
  per-pixel EO surface value repeated over depth,
]
```

The depth coordinate comes from the dataset/datamodule `depth_axis_m` when available. If that metadata is not available, the model falls back to evenly spaced normalized channel positions.

## Profile CNN Baseline

`ProfileCNNInfillingBaseline` is selected with:

```yaml
model:
  model_type: cnn_baseline
  cnn_baseline:
    hidden_channels: 64
    seed_length: 8
    transpose_layers: null
    conv_layers: 3
    kernel_size: 3
    batch_norm_momentum: 0.3
    dropout: 0.0
    activation: selu
    weight_decay: 0.0001
```

This baseline is inspired by profile-reconstruction CNNs that decode a fixed input vector into a vertical profile with `ConvTranspose1d`, then refine it with `Conv1d` layers. In DepthDif, each spatial pixel is still handled independently. The per-pixel vector contains the sparse normalized depth profile, an optional depth-wise observation mask, optional EO surface value, and optional land/ocean support scalar. It does not use neighboring pixels.

For one field with shape `(B, C, H, W)`, the model reshapes internally:

```text
(B, C, H, W)
-> (B * H * W, vector_features)
-> linear seed projection
-> ConvTranspose1d depth upsampling
-> Conv1d profile refinement
-> (B, C, H, W)
```

The default initializer is LeCun normal and the default activation is SELU. Training and validation loss are computed only at spatial columns with ARGO observations, comparing the decoded profile to the matching GLORYS profile at that location. Dense `predict_step` still runs over the full patch, and sample/field outputs with no ARGO support are emitted as nodata to match the other baselines.

## 3D U-Net Baseline

`UNetInfillingBaseline` is selected with `model.model_type: unet_baseline`. It keeps depth as a 3D convolution axis, uses sparse fields plus optional EO, valid-mask, and land-mask condition volumes, and predicts dense normalized patches directly. Unlike the LSTM and profile CNN, it can use local spatial context inside the patch.

## 2D U-Net Baseline

`UNet2DInfillingBaseline` is selected with `model.model_type: unet2d_baseline`. It uses the same `model.unet_baseline` settings, loss, validation logging, prediction contract, and no-ARGO behavior as the 3D U-Net. The difference is the tensor layout: depth bands stay flattened in `(B, C, H, W)` and are treated as 2D channels instead of a 3D convolution axis.

## LSTM Training Step

One training step uses the existing dataloader batch directly:

1. Stack active fields into model-facing tensors.
2. Split the stacked input back by physical field inside the baseline.
3. For each field, reshape every pixel into an independent depth sequence.
4. Run the field-specific LSTM and linear head.
5. Concatenate field predictions back in the configured output order.
6. Compute normalized masked MSE against the dense target.
7. Log `train/loss`.

The supervised loss is over finite predictions and targets, intersected with `y_valid_mask` and `land_mask`. This means the LSTM is trained against dense GLORYS target pixels on valid ocean support, not only at ARGO-observed pixels.

For validation, the same masked normalized MSE is logged as:

```text
val/loss
val/loss_ckpt
```

The first validation batch is cached, reconstructed at validation epoch end, denormalized, and logged with the same full-reconstruction metric keys used by the diffusion model:

```text
val/recon_mse_full_recon
val/recon_l1_full_recon
val/recon_psnr_full_recon
val/recon_ssim_full_recon
```

In joint mode, salinity metrics are also logged under:

```text
val_salinity/recon_mse_full_recon
val_salinity/recon_l1_full_recon
val_salinity/recon_psnr_full_recon
val_salinity/recon_ssim_full_recon
```

## No-ARGO Handling

The baselines handle missing ARGO support slightly differently because IDW operates per 2D band while trainable neural baselines operate per field/sample.

For IDW:

- A depth band with observations is interpolated from those observations.
- A depth band with no observations emits `NaN`.

For the U-Net baselines:

- A pixel with no observed profile can still predict if the patch has ARGO support elsewhere. Its sparse value feature is zero, its observed-mask feature is all zeros, and enabled auxiliary features remain available.
- A whole patch/sample with no ARGO support for a field emits `NaN` for that field.
- Those `NaN` values are preserved in normalized and denormalized `predict_step` outputs so export writes nodata.

For the point-wise LSTM and profile CNN, dense inference does not require ARGO support and can predict from EO/surface features with zero ARGO inputs and masks. The profile CNN still uses ARGO support to choose supervised training profile locations.

## Predict Step Outputs

All baselines return diffusion-compatible prediction keys:

```text
y_hat
y_hat_denorm
y_hat_denorm_for_plot
y_hat_temperature
y_hat_temperature_denorm
y_hat_temperature_denorm_for_plot
y_hat_salinity
y_hat_salinity_denorm
y_hat_salinity_denorm_for_plot
denoise_samples: []
x0_denoise_samples: []
sampler: None
further_valid_mask: None
```

Field-specific salinity keys are present only when salinity is an active output field. `y_hat_denorm` aliases the primary field, using temperature when temperature is present.

All baselines also implement deterministic uncertainty:

```text
uncertainty_stat: deterministic_zero
```

## Inference

Inference uses the same model factory and export path as the diffusion model:

1. Set `model.model_type` to `idw_baseline`, `lstm_baseline`, `cnn_baseline`, `unet_baseline`, or `unet2d_baseline`.
2. Keep the same data config and datamodule setup.
3. For `idw_baseline`, no checkpoint is loaded.
4. For `lstm_baseline`, `cnn_baseline`, `unet_baseline`, or `unet2d_baseline`, provide a trained Lightning checkpoint.
5. The exporter calls `predict_step` and consumes the same normalized, denormalized, and field-specific keys.

For whole-world export, the existing patch inference flow can therefore run any baseline. IDW is checkpoint-free and deterministic. The trainable neural baselines require trained weights but otherwise follow the same output path. For all baselines, sample/field predictions with no ARGO support become nodata.
