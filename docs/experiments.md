# Experiments
This page summarizes qualitative experiments from the README, with the same scenarios preserved and clarified.

## Experiment 1 (1-band corruption, 50%)
Preliminary sub-surface reconstruction with the single-band setup and 50% clustered occlusion after ~24h training.

Notes from the original run:
- valid masks were used in training
- land mask was used for visualization
- loss was computed over the whole image
- no inpainting-style known-pixel clamping in DDPM sampling
- reported metrics: PSNR ~40 dB, SSIM ~0.90

![img](assets/prelim_results2.png)

Applying the same checkpoint to 75% occlusion still produced plausible reconstructions:
![img](assets/prelim_results_75perc.png)

## Experiment 2 (3-band x and y, EO condition, 95% corruption)
EO-conditioned multiband setup:
- `x`: corrupted 3-band deeper input (depth levels 3,4,5)
- `y`: clean 3-band deeper target
- `eo`: additional surface condition

At 95% corruption, reconstructions remain strong, with noted risk of EO over-reliance.

![img](assets/95per_obstr_with_condition.png)

## Experiment 3 (High corruption and EO dropout)
Tested 75% corruption with 50% EO dropout while keeping the same checkpoint for comparison.

Observation:
- removing EO condition reduces fidelity
- outputs still match plausible distributional structure

![img](assets/75perc_50drop.png)

After continued training (100 epochs in that setting), corruption increased to 95% while keeping 50% dropout.

Observation:
- EO-present samples reconstruct faithfully
- EO-absent samples remain realistic but can deviate from exact ground-truth patterns

![img](assets/95perc_50drop.png)

## Experiment 4 (High corruption, EO dropout, EO degradation)
To reduce direct EO value shortcutting, EO condition was degraded using random perturbations and speckle-like noise.

Observation:
- reconstruction remains workable with degraded EO
- model still performs plausible `x -> y` reconstruction when EO is absent or weak

![img](assets/eo_x0_maskedLoss_95perc_50percDO_EOdegradation.png)

## Experiment 5 (Lower temperature levels)
Harder target depths were tested with levels:
- 0 (EO)
- 4 (5m)
- 10 (16m)
- 17 (50m)

Training setup: 95% corruption, 50% EO dropout, EO degradation.

Reported outcome: quality remained comparable on these depth levels.

![img](assets/deeper.png)

## Sampling Process Diagnostics
Current validation sampling in these experiments uses a cosine-guided noise schedule.

Logged diagnostics include:
- intermediate denoising frames
- MAE vs reverse denoising step (using per-step `x0` prediction)
- diffusion schedule profiles (`sqrt(alpha_bar_t)`, `sqrt(1-alpha_bar_t)`, `beta_tilde_t`, `log10(SNR+eps)`)

Observed tradeoff in DDPM mode:
- many early steps remain highly noisy
- compute is spent on low-visual-information stages

Potential improvement directions already noted in project docs:
- DDIM sampling for faster useful denoising trajectory
- alternate schedules
- parameterization choices (`x0` vs `epsilon`)

Intermediate reconstructions over the denoising path:
<p align="center">
  <img src="assets/intermediate_steps.png" alt="intermediate_steps" width="40%">
</p>

MAE trend across intermediate denoising steps:
<p align="center">
  <img src="assets/mae_vs_intermediate.png" alt="mae_vs_intermediate" width="50%">
</p>

Implemented schedule options in code:
- `linear`
- `cosine`
- `quadratic`
- `sigmoid`

Example schedule profile image:
<p align="center">
  <img src="assets/noise_schedules.png" alt="noise_schedules" width="85%">
</p>
