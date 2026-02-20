# Model
As a first prototype, a conditional pixel-space Diffuser is modeled after [DiffusionFF](https://github.com/mikonvergence/DiffusionFastForward).  

Current implemented conditioning/target setups:
- Single-band: `x -> y` (corrupted temp to clean temp)
- EO multiband: `[eo, x, valid_mask] -> y` (EO + corrupted deeper bands + mask to clean deeper bands)

Loss can be computed with or without masking by valid pixels (`mask_loss_with_valid_pixels`).

## Model Description
- **Model type**: Conditional diffusion model in pixel space (`PixelDiffusionConditional`) with a **ConvNeXt U-Net denoiser** (`UnetConvNextBlock`). CNN/U-Net-style model.
- **Architecture (default config in this repo)**:
  - `dim_mults=[1,2,4,8]` gives 4 encoder stages and 3 decoder stages.
  - Encoder stage block pattern: `ConvNextBlock -> ConvNextBlock -> LinearAttention -> Downsample` (last encoder stage uses identity downsample).
  - Bottleneck block pattern: `ConvNextBlock -> LinearAttention -> ConvNextBlock`.
  - Decoder stage block pattern: `concat skip -> ConvNextBlock -> ConvNextBlock -> LinearAttention -> Upsample` (last decoder stage uses identity upsample).
  - Output head: final `ConvNextBlock` followed by `1x1 Conv2d` to `generated_channels`.
- **Parameter count**: current EO config is approximately **57M parameters**.
- **Backbone I/O**:
  - Input to denoiser at each reverse/training step: `cat([x_t, condition], dim=1)` with shape `(B, generated_channels + condition_channels, H, W)`.
  - Output from denoiser: `(B, generated_channels, H, W)` (predicts `epsilon` noise or `x0`, based on `model.parameterization`).
- **Conditioning construction (EO task)**:
  - Dataset returns `eo: (B,1,H,W)`, `x: (B,3,H,W)`, `valid_mask: (B,3,H,W)`, `y: (B,3,H,W)`.
  - With EO config (`condition_include_eo=true`, `condition_mask_channels=1`), condition is:
    - `condition = cat([eo, x, valid_mask_reduced], dim=1) = (B,5,H,W)`
    - where `valid_mask_reduced` is `(B,1,H,W)`.
- **Training flow (EO task)**:
  - Target `y` is the diffusion sample branch: `(B,3,H,W)`.
  - Random timestep `t` is sampled per item: `(B,)`.
  - Forward process adds Gaussian noise to `y` only: `y_t = q(y_t | y_0)` and returns sampled `noise` `(B,3,H,W)`.
  - Denoiser input becomes `cat([y_t, condition], dim=1) = (B,8,H,W)`.
  - With `parameterization: "epsilon"` (default), loss compares predicted noise vs sampled noise.
- **Inference flow**:
  - Start from random latent `x_T ~ N(0,I)` with shape `(B,3,H,W)`.
  - Keep `condition` fixed (`(B,5,H,W)` in EO mode) through all reverse steps.
  - At each step, denoiser input is `cat([x_t, condition], dim=1) = (B,8,H,W)`, output `(B,3,H,W)`.
  - Sampler update:
    - DDPM: injects stochastic noise each step for `t>0`.
    - DDIM: deterministic when `eta=0`, stochastic when `eta>0`.
- **Where noise is injected**:
  - Noise is **added to the generated/target branch** (`y` in training, latent `x_t` in inference).
  - Noise is **not added to conditioning channels** (`eo`, `x`, `valid_mask`).
  - Conditioning is provided by **channel concatenation**, not by adding noise to condition tensors.

For model/training knobs, see [Model Settings](settings.md) (major + minor settings) and [Date + Coordination Injection](date-coordination-injection.md) (FiLM coordinate injection details).
