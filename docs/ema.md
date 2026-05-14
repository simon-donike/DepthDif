# Exponential Moving Average Weights  

DepthDif can maintain exponential moving average (EMA) weights during diffusion  
training. EMA is optional and disabled by default in the model configs.  

## What EMA Tracks  

EMA keeps a second copy of the model `state_dict` alongside the trainable model  
weights. The trainable model is still optimized normally by backpropagation. EMA  
does not change gradients, optimizer state, or the training update itself.  

After an optimizer step, the EMA callback updates each floating-point tensor with:  

```text
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

With `decay: 0.9999`, each EMA tensor changes slowly and represents a smoothed  
history of recent model weights. Non-floating buffers are copied from the current  
model state instead of being averaged, because integer counters and similar state  
cannot be meaningfully interpolated.  

The callback tracks weights by `state_dict` key. This makes resume behavior safer  
than relying on positional order alone.  

## Why It Can Improve Generations  

Diffusion training can move through noisy local weight states, especially with  
mixed precision, large U-Nets, sparse supervision masks, and high-variance batches.  
A checkpoint from a single optimizer step may contain weights that are valid for  
training but slightly less stable for reverse diffusion sampling.  

EMA reduces this step-to-step weight noise. At validation or inference time, the  
EMA copy can behave like an ensemble of nearby recent checkpoints without running  
multiple models. In practice, this often improves generated samples by:  

- reducing speckle and high-frequency instability  
- making reverse-chain outputs less sensitive to the exact checkpoint step  
- improving qualitative consistency between neighboring pixels and depth bands  
- producing smoother validation curves when the raw model weights are still noisy  

EMA is not guaranteed to improve every run. If the decay is too high early in  
training, EMA can lag behind real progress. If the decay is too low, it behaves  
nearly like the raw weights. That is why the configs keep EMA disabled by default  
until it is validated on full training runs.  

## Configuration  

EMA is configured in the model config under `model.ema`:  

```yaml
model:
  ema:
    enabled: false
    decay: 0.9999
    apply_every_n_steps: 1
    start_step: 0
    save_ema_weights_in_callback_state: true
    evaluate_ema_weights_instead: true
```

Fields:  

- `enabled`: turns EMA on when set to `true`.  
- `decay`: smoothing factor for EMA weights. Higher values change more slowly.  
- `apply_every_n_steps`: update cadence in optimizer steps.  
- `start_step`: first global step where EMA updates may run.  
- `save_ema_weights_in_callback_state`: stores EMA weights in Lightning checkpoint  
  callback state, so resume can continue the same EMA history.  
- `evaluate_ema_weights_instead`: swaps EMA weights into the model for  
  validation/test, then restores the raw training weights afterward.  

The training script builds the EMA callback from `model.ema`. The model class does  
not own EMA state; EMA is a trainer-level concern because it affects checkpointing,  
validation, and test-time weight swapping.  

## Checkpoints And Inference  

When `save_ema_weights_in_callback_state: true`, EMA weights are stored in the  
Lightning checkpoint callback state under `ema_weights`. This lets resumed  
training continue the same smoothed history instead of restarting EMA from the  
raw checkpoint weights.  

Inference helpers load EMA weights by default when they are present in a  
checkpoint. If a checkpoint does not contain EMA state, they fall back to the  
standard model `state_dict`.  

## Validation Logging  

When EMA is disabled, validation logging behaves as before.  

When EMA is enabled, the validation full-reconstruction diagnostic logs both  
weight states from the same cached validation mini-batch:  

- `val_imgs/x_y_full_reconstruction_standard`: reconstruction with raw training  
  weights  
- `val_imgs/x_y_full_reconstruction_ema`: reconstruction with EMA weights  

The standard and EMA grids are intentionally separate images so they can be  
compared directly in W&B.  

Scalar reconstruction metrics are also split:  

- `val_standard/recon_mse_full_recon`  
- `val_standard/recon_psnr_full_recon`  
- `val_standard/recon_ssim_full_recon`  
- `val_ema/recon_mse_full_recon`  
- `val_ema/recon_psnr_full_recon`  
- `val_ema/recon_ssim_full_recon`  

The existing `val/recon_*` keys follow the configured validation behavior:  

- if `evaluate_ema_weights_instead: true`, `val/recon_*` reflects EMA weights  
- if `evaluate_ema_weights_instead: false`, `val/recon_*` reflects standard weights  

This keeps checkpoint monitoring compatible with the selected validation mode  
while still logging both qualitative variants for comparison.  

## EMA Weight Metrics  

The callback logs raw-vs-EMA distance metrics under `ema/*` once per validation  
epoch. These metrics describe how different the EMA shadow weights are from the  
current raw training weights.  

### `ema/decay`  

The configured EMA decay. This is logged as a scalar so W&B charts and run  
comparisons show which smoothing factor was used.  

### `ema/weight_mean_abs_delta`  

Mean absolute difference between raw and EMA floating-point weights:  

```text
mean(abs(ema_weight - raw_weight))
```

Interpretation:  

- near zero: EMA is very close to the raw model  
- increasing value: EMA is lagging farther behind current weights or smoothing a  
  volatile training phase  
- sudden jump: training weights changed sharply, or a checkpoint/resume boundary  
  should be inspected  

### `ema/weight_rms_delta`  

Root-mean-square difference between raw and EMA floating-point weights:  

```text
sqrt(mean((ema_weight - raw_weight)^2))
```

This is more sensitive to large outlier differences than mean absolute delta.  
Use it to detect whether a small subset of tensors is drifting strongly.  

### `ema/weight_relative_rms_delta`  

RMS delta normalized by the RMS magnitude of the raw weights:  

```text
weight_rms_delta / rms(raw_weight)
```

This is useful because absolute weight scales differ across layers. A relative  
value gives a dimensionless sense of how far EMA is from the trainable model.  

Interpretation depends on the run, but broad guidance is:  

- very small: EMA and raw weights are effectively the same  
- moderate and stable: EMA is smoothing recent training movement  
- large or growing: EMA may be lagging, training may be unstable, or decay may be  
  too high for the current phase  

### `ema/weight_max_abs_delta`  

Maximum absolute raw-vs-EMA weight difference over all tracked floating tensors.  
This is an outlier metric. It can reveal a single tensor changing sharply even  
when average metrics look mild.  

### `ema/tracked_floating_tensors`  

Number of floating-point `state_dict` tensors included in the EMA distance  
metrics. Non-floating buffers are excluded from distance calculations because  
they are copied, not averaged.  

This value should remain stable for a fixed model architecture. A change usually  
means the model state structure changed.  

## How To Read EMA Runs  

A useful EMA run should show both qualitative and scalar evidence:  

- the EMA reconstruction image should be at least as coherent as the standard  
  reconstruction image  
- `val_ema/recon_mse_full_recon` should be competitive with  
  `val_standard/recon_mse_full_recon`  
- `ema/weight_relative_rms_delta` should not grow without bound  
- `ema/weight_max_abs_delta` should not show unexplained spikes  

If EMA images look smoother but metrics get worse, the model may be over-smoothed  
for the supervised validation support. If EMA metrics are nearly identical to  
standard metrics, the decay may be too low, the run may be too short, or the model  
may already be stable enough that EMA has little visible effect.  

## Practical Defaults  

The current config defaults are conservative:  

- EMA disabled until full training runs validate it  
- `decay: 0.9999`  
- update every optimizer step  
- save EMA state in checkpoints  
- use EMA for validation/test when enabled  

For quick experiments, enable EMA without changing the other fields first. Tune  
`decay` only after comparing standard and EMA validation images and metrics over  
several validation intervals.  
