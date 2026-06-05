# Baselines

DepthDif baseline models live in `src/depth_recon/models/baselines/`. They are Lightning modules and share the same dataloader-facing batch contract and `predict_step` output contract as the diffusion model. This keeps training, inference, global export, uncertainty export, and comparison-globe tooling on the same path where possible.

The currently supported baseline selectors are:

| `model.model_type` | Model | Trainable | Checkpoint required |
| --- | --- | ---: | ---: |
| `idw_baseline` | `IDWInterpolationBaseline` | No | No |
| `lstm_baseline` | `PointwiseLSTMBaseline` | Yes | Yes |

## Shared Batch Contract

Both baselines use the existing GeoTIFF dataloaders unchanged. The dataset still returns sparse ARGO/profile observations, dense GLORYS targets, masks, and optional EO surface context with field-specific keys.

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

`land_mask` is not an input feature for the point-wise LSTM. It is used for loss support and output cleanup. `output_land_mask`, when present during export, is only used for final prediction masking.

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

The two baselines handle missing ARGO support slightly differently because IDW operates per 2D band and the LSTM operates per field profile.

For IDW:

- A depth band with observations is interpolated from those observations.
- A depth band with no observations emits `NaN`.

For the point-wise LSTM:

- A pixel with no observed profile can still predict if the patch has ARGO support elsewhere. Its sparse value feature is zero, its observed-mask feature is all zeros, and it still receives EO and depth features.
- A whole patch/sample with no ARGO support for a field emits `NaN` for that field.
- Those `NaN` values are preserved in normalized and denormalized `predict_step` outputs so export writes nodata.

This behavior keeps no-support global inference regions from turning into arbitrary learned prior predictions while still allowing learned predictions for unobserved pixels inside supported patches.

## Predict Step Outputs

Both baselines return diffusion-compatible prediction keys:

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

Both baselines also implement deterministic uncertainty:

```text
uncertainty_stat: deterministic_zero
```

## Inference

Inference uses the same model factory and export path as the diffusion model:

1. Set `model.model_type` to `idw_baseline` or `lstm_baseline`.
2. Keep the same data config and datamodule setup.
3. For `idw_baseline`, no checkpoint is loaded.
4. For `lstm_baseline`, provide a trained Lightning checkpoint.
5. The exporter calls `predict_step` and consumes the same normalized, denormalized, and field-specific keys.

For whole-world export, the existing patch inference flow can therefore run either baseline. IDW is checkpoint-free and deterministic. The LSTM requires trained weights but otherwise follows the same output path; no-ARGO patches become nodata during raster stitching/export.
