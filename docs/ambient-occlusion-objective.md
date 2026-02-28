# Ambient Occlusion Objective in DepthDif

This document describes changes implemented to go from the standard DepthDif and the occlusion branch to a re-implementation of the Ambient Diffusion training objective from:

- Daras et al., *Ambient Diffusion: Learning Clean Distributions from Corrupted Data* (arXiv:2305.19256).  
  Paper: <https://arxiv.org/abs/2305.19256>


## TL;DR

Before this change, the model saw one masked input and trained mostly on the pixels that were already missing.

Now, during training only, we do one extra step: we hide some of the still-visible pixels again at random. So the model gets a harder, more incomplete input.

The important part is the loss target: we still score the model on the original known pixels (from the first mask), not only on the pixels that stayed visible after the second random drop.

Why this matters: it avoids a weak objective where the model can learn shortcuts from the currently visible subset, and instead teaches it to recover stable structure under extra random occlusion.

Inference/sampling did not change. This is a training-objective change only.

## Visual Walkthrough (One Real Sample)

![Ambient objective step-by-step example](assets/sample_12370_t653.png)

How to read this image from left to right (each row is one depth band):

1. `x0 (clean y)`: the clean target field used by diffusion training.
2. `A (orig mask)`: original observed/valid mask from the dataset (`valid_mask`).
3. `x = A * x0`: sparse input after first occlusion (this is the normal conditioning input before ambient mode).
4. `B (random keep)`: random keep/drop mask sampled during training only.
5. `A_tilde = B * A`: further-corrupted mask; it can only remove points from `A`, never add new ones.
6. `x_tilde = A_tilde * x`: the actual harder condition given to the model in ambient mode.
7. `x_t (noisy x0)`: diffusion forward-noised target branch at timestep `t`.
8. `A_tilde * x_t`: optional masking of the noisy branch (enabled by default in this implementation).

Key point in context of this figure:
- Input to the model is based on `A_tilde` (harder than original `A`),
- but the loss is still weighted on the original mask `A` (not on `A_tilde`).

That is the core difference from the sparse-holdout style objective: sparse holdout trains on a held-out subset itself, while ambient uses extra corruption for conditioning difficulty and keeps supervision anchored to the original observation support.

## 1. Top-Level Perspective

DepthDif previously trained a conditional diffusion model with a **single corruption stage** (dataset occlusion mask) and (typically) a loss focused on **missing pixels**.

The new procedure adds a second stochastic corruption stage during training:

1. Start from the original observation mask \(A\) (from `valid_mask`).
2. Sample an additional random keep/drop operator \(B\).
3. Form a further-corrupted mask \(\tilde{A} = B \odot A\).
4. Feed the model condition built from \(\tilde{A}\)-corrupted input.
5. Supervise the prediction on the **original** observed subset \(A\), not on \(\tilde{A}\).

Intuition: the model is forced to infer values in pixels that are sometimes hidden by the additional corruption, while still being evaluated where ground truth is known from the original observation set.

## 2. Notation

For one sample:

- \(x_0 \in \mathbb{R}^{C \times H \times W}\): clean diffusion target (in this repo: normalized `y`).
- \(A \in \{0,1\}^{C \times H \times W}\): original validity/observation mask (`valid_mask`).
- \(x = A \odot x_0\): original sparse observed input (in this repo, `x` already carries this structure).
- \(t \sim \mathrm{Unif}\{0,\dots,T-1\}\): diffusion timestep.
- \(x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ \epsilon\sim\mathcal{N}(0,I)\): noisy target branch sample.
- \(B \in \{0,1\}^{C \times H \times W}\): further keep mask sampled with keep-probability \(1-\delta\) on observed entries.
- \(\tilde{A} = B \odot A\): further-corrupted observation mask.

In implementation, \(\delta =\) `model.ambient_occlusion.further_drop_prob`.

## 3. Previous Objective (Repository Before This Change)

With `mask_loss_with_valid_pixels=true`, the loss was computed on missing pixels:

\[
\mathcal{L}_{\text{prev}}(\theta)
=
\frac{
\left\|(1-A)\odot\left(\text{target}_t-\hat{x}_{\theta}\right)\right\|_2^2
}{
\|(1-A)\|_1
},
\]

where:

- \(\hat{x}_{\theta}\) is the denoiser output.
- \(\text{target}_t = x_0\) for `parameterization="x0"` or \(\epsilon\) for `parameterization="epsilon"`.

Conditioning used the original sparse input/mask pair \((x, A)\) (plus EO, if enabled), without extra stochastic masking during training.

## 4. New Ambient Objective (Implemented)

### 4.1 Training Inputs

Define:

\[
\tilde{x} = \tilde{A}\odot x.
\]

The model condition is built from \((\tilde{x}, \tilde{A}, \text{EO})\) instead of \((x, A, \text{EO})\).

Optionally (enabled by default), the noisy branch is also masked:

\[
\tilde{x}_t = \tilde{A}\odot x_t.
\]

### 4.2 Loss Region

The implemented ambient mode uses `loss_mask_mode="observed"`, so supervision mask is \(A\) (not \(1-A\), and not \(\tilde{A}\)):

\[
\mathcal{L}_{\text{ambient}}(\theta)
=
\frac{
\left\|A\odot\left(\text{target}_t-\hat{x}_{\theta}\right)\right\|_2^2
}{
\|A\|_1
}.
\]

If `land_mask` is provided, the effective mask is \(A\odot L\), exactly matching existing ocean-gating behavior.

### 4.3 Relation to Paper Objective

The procedure matches the paper’s core structure:

\[
J_{\mathrm{corr}}(\theta)=\frac12\,\mathbb{E}\left[\left\|A\left(h_{\theta}(\tilde{A}\,x_t,\tilde{A},t)-x_0\right)\right\|_2^2\right],
\]

up to the repository’s existing normalization/parameterization conventions and per-mask normalization by mask cardinality.

## 5. What Changed vs What Stayed the Same

### Changed

1. **Two-stage masking during training** (\(A \rightarrow \tilde{A}\)).
2. **Condition path uses \(\tilde{A}\)** and \(\tilde{x}\).
3. **Loss region switches to observed mask \(A\)** in ambient mode.
4. New ambient metrics are logged:
   - `train/ambient_further_drop_fraction`
   - `train/ambient_observed_fraction_original`
   - `train/ambient_observed_fraction_further`
   - same keys under `val/*`.

### Unchanged

1. Inference/sampler algorithms (DDPM/DDIM) are unchanged.
2. Dataset generation of original corruption mask \(A\) is unchanged.
3. Optional land-mask gating remains identical.
4. If ambient mode is disabled, behavior remains backward-compatible with previous training flow.
5. Important: disabling ambient does **not** imply full-image loss. With `model.mask_loss_with_valid_pixels=true` (current default), loss reverts to missing-pixel masking (`1 - valid_mask`), not all of \(Y\).

## 6. Implemented Safety and Constraints

1. **Parameterization guard**: if ambient mode is enabled and `require_x0_parameterization=true`, then `parameterization` must be `"x0"`; otherwise construction raises `ValueError`.
2. **Mask monotonicity**: \(\tilde{A} \le A\) elementwise by construction.
3. **Degeneracy guard**: at least `min_kept_observed_pixels` are kept per sample when possible, preventing empty effective supervision from the further corruption stage.
4. `shared_spatial_mask=true` enforces one spatial \(B\) per sample, broadcast across channels.

## 7. Code Mapping (Equation to Implementation)

- Ambient config surface:
  - `configs/model_config.yaml` (`model.ambient_occlusion.*`)
- Runtime config wiring and safety:
  - `models/difFF/PixelDiffusion.py`
    - `PixelDiffusionConditional.from_config(...)`
    - `PixelDiffusionConditional.__init__(...)`
- \(\tilde{A}\) construction:
  - `PixelDiffusionConditional._build_ambient_further_valid_mask(...)`
- Condition path replacement \((x,A)\to(\tilde{x},\tilde{A})`:
  - `training_step(...)`, `validation_step(...)`
- Ambient loss execution:
  - `models/difFF/DenoisingDiffusionProcess/DenoisingDiffusionProcess.py`
    - `DenoisingDiffusionConditionalProcess.p_loss(...)`
    - `loss_mask_mode="observed"`
    - optional `apply_further_corruption_to_noisy_branch`

## 8. Practical Interpretation

The old setup primarily asked the model to reconstruct hidden regions given fixed observed context.

The new setup introduces random context removal during training while preserving supervision on the original observed support. This makes the learning problem closer to the ambient objective: robustly estimate clean content under stochastic measurement degradation, not only under one fixed missingness pattern per sample.

## Citation

If you use this objective adaptation, please cite:

```bibtex
@article{daras2023ambient,
  title={Ambient Diffusion: Learning Clean Distributions from Corrupted Data},
  author={Daras, Giannis and Shah, Kulin and Dagan, Yuval and Gollakota, Aravind and Dimakis, Alexandros G. and Klivans, Adam},
  journal={arXiv preprint arXiv:2305.19256},
  year={2023},
  url={https://arxiv.org/abs/2305.19256}
}
```
