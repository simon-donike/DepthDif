# Ambient Ocean Losses

This page documents the optional loss stack under `model.losses.*` for pixel-space ambient ocean reconstruction. These terms are designed for sparse ARGO-conditioned diffusion training where temperature and salinity are trained as separate scalar fields.

The central constraint is that GLORYS is not used as paired dense supervision for these auxiliary losses. GLORYS may provide distributional, spectral, or statistical reference values, but the model must not minimize a direct same-sample error such as `mse(x0_pred, glorys_same_time_location)` or `mse(phi(x0_pred), phi(glorys_same_sample))`.

## Notation

For one normalized model sample:

- \(\hat{x}_0 \in \mathbb{R}^{C \times H \times W}\): predicted clean field recovered from the denoiser output. In the current scalar-field setup, \(C\) is the number of depth channels for one variable.
- \(x \in \mathbb{R}^{C \times H \times W}\): sparse observed ARGO field in model-normalized units.
- \(A \in \{0,1\}^{C \times H \times W}\): sparse ARGO observation mask, `x_valid_mask`.
- \(Y \in \{0,1\}^{C \times H \times W}\): valid target/support mask, `y_valid_mask`.
- \(G \in \{0,1\}^{1 \times H \times W}\): GLORYS-derived ocean/domain support, `land_mask` in the dataloader contract.
- \(t\): diffusion timestep.
- \(\epsilon\): Gaussian noise used by the forward diffusion process.
- \(\lambda_k\): scalar config weight for loss term \(k\).

The implementation operates in normalized model units. This is intentional: ARGO sparse observations, predictions, and diffusion targets are already standardized by the dataset and model path. Physical-unit interpretation belongs to denormalized diagnostics, not the training objective.

## Combined Objective

The total optimized loss is:

\[
\mathcal{L}_{\text{total}}
=
\lambda_{\text{amb}}\mathcal{L}_{\text{amb}}
+
\lambda_{\text{obs}}\mathcal{L}_{\text{obs}}
+
\lambda_{\text{inc}}\mathcal{L}_{\text{inc}}
+
\lambda_{\text{S2}}\mathcal{L}_{\text{S2}}
+
\lambda_{\text{spec}}\mathcal{L}_{\text{spec}}.
\]

The shipped defaults keep all auxiliary terms disabled while retaining their intended weights:

| Term | Config key | Default enabled | Default weight |
| --- | --- | --- | ---: |
| Base diffusion / ambient loss | `model.losses.ambient` | always present | `1.0` |
| Sparse observation consistency | `model.losses.sparse_observation` | `false` | `1.0` |
| Sparse increment consistency | `model.losses.increment` | `false` | `0.5` |
| GLORYS structure-function prior | `model.losses.structure_function_prior` | `false` | `0.1` |
| GLORYS spectral energy floor | `model.losses.spectral_energy_floor` | `false` | `0.05` |
| Feature Gram prior | `model.losses.feature_gram_prior` | `false` | `0.01` reserved, not implemented |

`train/loss` and `val/loss` are the total loss. Component logs are emitted as `*/loss_ambient`, `*/loss_obs`, `*/loss_increment`, `*/loss_s2_glorys`, `*/loss_spectral_glorys`, and `*/loss_total`.

## Base Diffusion And Ambient Loss

The base term is the existing diffusion training loss. It is unchanged by the auxiliary loss stack.

For `parameterization: x0`, the denoiser predicts the clean target directly:

\[
\mathcal{L}_{\text{amb}}
=
\frac{\sum_p M(p)\left(\hat{x}_0(p)-x_0(p)\right)^2}{\sum_p M(p)},
\]

where \(M\) is the active supervised support. In standard mode, \(M\) is based on `y_valid_mask` and `land_mask`. In ambient occlusion mode, \(M = A \odot Y \odot G\), and the clean target is the original sparse observation tensor `x` rather than dense GLORYS `y`.

For `parameterization: epsilon`, the base loss remains the existing noise-prediction MSE. The auxiliary losses still need a clean estimate, so the diffusion process converts the predicted noise to \(\hat{x}_0\):

\[
\hat{x}_0
=
\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}.
\]

Why this term is kept: it is the diffusion objective that makes the reverse process learn denoising dynamics. The auxiliary terms shape the clean estimate, but they do not replace the diffusion objective.

## Sparse Observation Consistency

The sparse observation term compares the clean prediction to real sparse ARGO observations at observed locations:

\[
\mathcal{L}_{\text{obs}}
=
\frac{\sum_p A(p)\,\rho\left(\hat{x}_0(p)-x(p)\right)}{\sum_p A(p)}.
\]

The robust penalty is Charbonnier loss:

\[
\rho(r)=\sqrt{r^2 + \varepsilon^2}.
\]

Default \(\varepsilon = 10^{-3}\).

Implementation details:

- Primary path uses gridded sparse tensors: `x` and `x_valid_mask`.
- Packed observation values and indices are supported for future batch formats and tests.
- Missing or invalid observations are masked out.
- Optional weights can be applied when supplied by a caller.
- Reduction is a masked mean.

Why we assume it helps: the diffusion model is free to generate plausible fields, but reconstruction still has to honor the measurements that are actually available. Charbonnier is less brittle than MSE when individual ARGO observations have representativeness error, interpolation noise, or local mismatch with gridded support.

What it does not do: it does not compare against dense GLORYS values at unobserved pixels.

## Sparse Increment Consistency

Pointwise observation consistency can still allow over-smoothed profiles. The increment loss compares local differences between observed pairs:

\[
\Delta_{ij}^{\text{pred}} = \hat{x}_0(i)-\hat{x}_0(j),
\]

\[
\Delta_{ij}^{\text{obs}} = x(i)-x(j),
\]

\[
\mathcal{L}_{\text{inc}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j)\in\mathcal{P}}
\rho\left(\Delta_{ij}^{\text{pred}}-\Delta_{ij}^{\text{obs}}\right).
\]

Here \(\mathcal{P}\) is the sampled set of valid observed pairs.

The default implementation builds vertical adjacent-depth pairs from the gridded ARGO tensor:

\[
\mathcal{P}_{\text{vertical}}
=
\{((z,h,w),(z+1,h,w)) : A(z,h,w)=1, A(z+1,h,w)=1\}.
\]

Optional horizontal pairs use same-depth right/down neighbor pairs when enabled:

\[
\mathcal{P}_{\text{horizontal}}
=
\{((z,h,w),(z,h+\delta_h,w+\delta_w)) : A\text{ is valid at both points}\}.
\]

Pairs are randomly capped by `max_pairs_per_sample` for efficiency.

Why we assume it helps: smoothing often preserves coarse values while shrinking gradients and profile contrasts. Increment matching directly penalizes that gradient shrinkage at observed locations. It is especially useful for vertical ocean structure because thermocline and halocline sharpness is carried more by differences between nearby depths than by isolated absolute values.

What it does not do: it does not invent a dense target gradient from GLORYS. It only uses observed sparse ARGO increments unless future explicit pair indices are provided by the batch.

## GLORYS Structure-Function Prior

The structure-function prior is distributional. It compares generated field increment statistics to precomputed GLORYS reference statistics for a class such as variable, region, month, or depth. It does not compare a generated field to the same sample's GLORYS field.

For a spatial displacement vector \(r\), the second-order structure function is:

\[
S_2(r)=\mathbb{E}_p\left[\left(x(p+r)-x(p)\right)^2\right].
\]

The model estimate \(\hat{S}_2\) is computed by randomly sampling same-depth point pairs from \(\hat{x}_0\), binning them by pair distance, and averaging squared increments inside each distance bin.

For bin \(b\):

\[
\hat{S}_2(b)=
\frac{1}{|\mathcal{P}_b|}
\sum_{(i,j)\in\mathcal{P}_b}
\left(\hat{x}_0(i)-\hat{x}_0(j)\right)^2.
\]

The loss compares log structure functions:

\[
\mathcal{L}_{\text{S2}}
=
\frac{1}{|\mathcal{B}_{\text{valid}}|}
\sum_{b\in\mathcal{B}_{\text{valid}}}
\left|
\log\left(\hat{S}_2(b)+\varepsilon\right)
-
\log\left(S_2^{\text{ref}}(b)+\varepsilon\right)
\right|.
\]

Default \(\varepsilon = 10^{-6}\).

Reference file format:

```python
{
    "distance_bins": tensor([...]),      # bin edges, shape [num_bins + 1]
    "s2_ref": tensor([...]),             # shape [num_bins] or [C, num_bins]
}
```

If `per_depth: true`, a `[C, num_bins]` reference compares each depth channel to its corresponding reference row. Empty sampled bins are skipped.

Why we assume it helps: ocean textures are scale-dependent. Over-smoothed predictions usually have too little variance at short and intermediate separations, even if their large-scale mean looks plausible. Structure functions summarize how roughness grows with spatial scale, so matching them nudges the generated distribution toward realistic multiscale variability without requiring paired GLORYS supervision.

Why log space: ocean variance can differ by orders of magnitude across distances and depths. Log comparison makes the loss care about relative scale errors instead of letting high-energy bins dominate everything.

## GLORYS Spectral Energy Floor

The spectral prior is also distributional. It prevents generated fields from becoming too smooth by enforcing a lower bound on radial Fourier-band energy.

For each depth slice, the model computes a 2D Fourier power spectrum:

\[
P(k_x,k_y)=\left|\mathcal{F}_{2D}\{\hat{x}_0\}(k_x,k_y)\right|^2.
\]

Power is radially binned into bands \(b\):

\[
\hat{P}(b)=\mathbb{E}_{(k_x,k_y)\in b}\left[P(k_x,k_y)\right].
\]

The loss is a lower-bound hinge in log-energy space:

\[
\mathcal{L}_{\text{spec}}
=
\frac{1}{|\mathcal{B}_{\text{valid}}|}
\sum_{b\in\mathcal{B}_{\text{valid}}}
\max\left(
0,
\log(P^{\text{ref}}(b)+\varepsilon)
-
\log(\hat{P}(b)+\varepsilon)
-
m
\right).
\]

Here \(m\) is `margin`, and default \(\varepsilon = 10^{-8}\).

Reference file format accepts one of these keys:

```python
{
    "energy_ref": tensor([...]),          # or power_ref / band_energy_ref / spectral_energy_ref
}
```

The tensor may be shaped `[bands]` or `[C, bands]`. `min_band: 1` skips the DC band by default. `max_band: null` uses all available bands.

Why we assume it helps: averaged reconstructions lose high-frequency energy. An exact spectral match would be too restrictive because each generated sample need not reproduce a particular GLORYS spectrum. The hinge only penalizes missing energy below the GLORYS reference floor, so it pushes against excessive smoothness while still allowing the model to produce more energetic fields when observations and conditioning support it.

Current limitation: the FFT path does not apply an irregular land/ocean mask before transforming. The loss relies on the generated patch tensor and radial bins. If masked FFTs become necessary, they should be added deliberately with windowing or inpainting choices documented, because naive masks can introduce spectral artifacts.

## Feature Gram Prior

`model.losses.feature_gram_prior` is present in the config as a reserved disabled section. It is not implemented. If enabled, construction raises `NotImplementedError`.

The intended future form is a frozen encoder and class-level GLORYS feature statistics:

\[
\mathcal{L}_{\text{gram}}
=
\left\|\operatorname{Gram}(\phi(\hat{x}_0))-G^{\text{ref}}\right\|_1.
\]

This would still be distributional: \(G^{\text{ref}}\) must be precomputed for a class or bin, not taken from the same sample's paired GLORYS field.

## Reference Statistics And Class Conditioning

The implementation loads one reference file per enabled prior. The meaning of that reference file is controlled outside the loss by how the file was built. For example, a valid reference could represent:

- temperature, North Atlantic, winter, all depths;
- salinity, Mediterranean, monthly bin, per depth;
- global temperature, per-depth climatological statistics.

The loss does not currently select different reference rows by date/region metadata at runtime. If multiple classes are needed, the recommended extension is to precompute separate files and select the desired file in the training config for that run.

## Why These Losses Are Compatible With Ambient Diffusion

Ambient diffusion learns from corrupted observations by supervising only what is legitimately observed under the measurement process. In this repository, sparse ARGO terms use actual observed values, and GLORYS terms use aggregate distributional statistics. This preserves the distinction between:

- measurement consistency: match sparse ARGO where measurements exist;
- distributional realism: match GLORYS-derived statistics across classes;
- forbidden dense supervision: match the paired GLORYS field at the same time and location.

That distinction matters because direct dense GLORYS losses would turn training into paired reconstruction against reanalysis, while the goal here is sparse observation reconstruction regularized by realistic ocean-field statistics.

## Practical Tuning

Start with the base ambient objective and enable sparse terms first:

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set model.ambient_occlusion.enabled=true \
  --set model.losses.sparse_observation.enabled=true \
  --set model.losses.increment.enabled=true
```

Then raise anti-smoothing terms gradually:

- Increase `model.losses.increment.weight` when observed vertical profiles look too flat.
- Increase `model.losses.structure_function_prior.weight` when spatial roughness is too weak across distance scales.
- Increase `model.losses.spectral_energy_floor.weight` when mid/high-frequency texture is underpowered.
- Keep GLORYS-prior weights smaller than sparse-observation weights unless the reference statistics are well matched to the training subset.

For salinity, use `--scenario salinity` and salinity-specific reference statistics. Do not enable auxiliary losses for `--scenario joint`; temperature and salinity are intentionally trained as separate scalar-field models for this loss stack.
