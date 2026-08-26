# Model

DepthDif's maintained model is `PixelDiffusionConditional`, a conditional
pixel-space diffusion model with a ConvNeXt-style U-Net denoiser.

## Model-facing fields

The active dataset returns 128×128 patches. A scalar scenario has 50 generated
depth channels and 53 condition channels; the joint scenario concatenates
temperature and salinity into 100 generated and 103 condition channels.

Dense surface conditioning uses one scenario-derived channel: OSTIA sea-surface
temperature (`sst`) for temperature and joint scenarios, or sea-surface salinity
(`sss`) for the salinity scenario.

Sparse depth-aligned EN4/ARGO values, observation masks, ocean/bathymetry support,
coordinates, and date context complete the condition. The scenario resolver owns
the final channel contract.

## Conditioning and denoising

Coordinates are encoded with the configured scheme and the date is represented
by periodic day-of-year features. The resulting context is injected through FiLM
in ConvNeXt blocks throughout the denoiser. See [Coordinate and date
conditioning](data-coordinate-injection.md).

For diffusion timestep `t`, the forward process is

\[
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
\]

The configured objective determines whether the model predicts noise or the
clean target. Current pixel presets use the values recorded in their committed
YAML snapshots; checkpoints must be loaded with a compatible objective and
channel layout.

## Training objectives

- Direct paired supervision uses the dense GLORYS target.
- Synthetic pretraining uses observed surfaces plus a fitted monthly/spatial
  GLORYS depth-delta prior, with exact ARGO anchors.
- Ambient training further corrupts sparse observations for the condition while
  computing loss on the original observed support intersected with valid target
  and ocean support.

Coastal distance weighting exists but is disabled in all current pixel presets.
Optional observation, increment, structure, and spectral losses are also
disabled by default. Feature Gram configuration is reserved but not implemented.

## Sampling and output

DDPM and DDIM samplers are implemented. Training validation uses 100-step DDIM;
the inference super-config defaults to DDPM unless explicitly overridden.

Prediction applies the target field's inverse normalization, preserves invalid
support as `NaN` internally, and lets exporters apply final land/nodata masks.
Repository exporters can retain all native depth channels or select requested
physical depths by nearest GLORYS channel.

EMA weights are maintained by current pixel presets and can be compared with
standard weights during validation. See [EMA](ema.md), [Losses](losses.md), and
[Uncertainty](uncertainty.md).
