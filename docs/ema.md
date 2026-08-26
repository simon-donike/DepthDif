# Exponential Moving Average Weights

Current pixel training and inference presets enable the EMA callback. It tracks a
smoothed copy of floating-point model state while copying non-floating state
directly.

For parameter value \(\theta_t\) and EMA value \(\theta_t^{EMA}\):

\[
\theta_t^{EMA} = d\theta_{t-1}^{EMA} + (1-d)\theta_t.
\]

## Current configuration

```yaml
model:
  ema:
    enabled: true
    decay: 0.9999
    apply_every_n_steps: 1
    start_step: 0
    save_ema_weights_in_callback_state: true
    evaluate_ema_weights_instead: true
```

EMA starts at step zero and updates every optimizer step. The callback validates
its decay, cadence, and start step, keeps state keys aligned with the model, and
moves tracked tensors to the active device.

## Validation behavior

When EMA evaluation is enabled, the callback temporarily swaps EMA values into
the model during validation and restores raw training weights afterward. Full
reconstruction logging computes explicit `val_standard/*` and `val_ema/*`
variants; the configured EMA branch also owns the ordinary validation metrics.

The callback logs weight-distance diagnostics once per validation epoch:

- `ema/decay`
- `ema/weight_mean_abs_delta`
- `ema/weight_rms_delta`
- `ema/weight_relative_rms_delta`
- `ema/weight_max_abs_delta`
- `ema/tracked_floating_tensors`

These describe distance between raw and smoothed weights. They do not by
themselves establish that EMA improves reconstruction quality; compare the raw
and EMA validation metrics for the same checkpoint and sampler.

## Checkpoints

With `save_ema_weights_in_callback_state=true`, Lightning checkpoints include the
EMA tensors and current callback step. Loading validates the state against the
model keys. The callback also understands associated `-EMA.ckpt` files when that
checkpoint style is used.

Repository inference must load a checkpoint and config that agree about the
model architecture. EMA cannot make a legacy one-surface checkpoint compatible
with the current three-surface model.
