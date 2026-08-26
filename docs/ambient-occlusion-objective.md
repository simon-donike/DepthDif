# Ambient Occlusion Objective

Ambient training learns from sparse EN4/ARGO observations without converting
unobserved ocean cells into supervised targets. The model is conditioned on a
further-corrupted version of the observed field, while loss is evaluated only
where the original sparse field and the paired target are valid over ocean.

## Notation

- `x`: sparse depth-aligned EN4/ARGO values.
- `m`: original sparse observation mask (`x_valid_mask`).
- `m'`: further-corrupted mask, with `m' ≤ m`.
- `y`: paired dense target used to establish valid depth/ocean support.
- `v`: target-valid mask (`y_valid_mask`).
- `l`: ocean support (`land_mask` in the batch contract, where ocean is valid).

The model condition contains `x ⊙ m'`, `m'`, the configured dense surface
channels, and coordinate/date context. In the current `x0` objective, the
supervision mask is

\[
M_{loss} = m \cap v \cap l,
\]

and the supervised target on that support is the original sparse field `x`.
Unobserved pixels are not assigned pseudo-labels by this objective.

## Corruption policy

The local pixel preset uses:

```yaml
model:
  parameterization: x0
  clamp_known_pixels: false
  ambient_occlusion:
    enabled: true
    further_drop_prob: 0.25
    apply_to_noisy_branch: true
    shared_spatial_mask: true
    min_kept_observed_pixels: 50
    require_x0_parameterization: true
```

The drop mask is shared spatially across field channels when configured. A
minimum-support guard restores observations when random dropping would leave too
few supervised pixels. With `apply_to_noisy_branch=true`, the further-corrupted
support also masks the noisy target branch seen by the denoiser.

## Constraints

- Ambient mode requires `x_valid_mask` in each batch.
- With `require_x0_parameterization=true`, any other parameterization is rejected.
- Loss is normalized over valid weighted support, not the full patch area.
- `clamp_known_pixels=false` means sampling does not overwrite predictions with
  observed `x` after each reverse step.
- Exporters restore geospatial invalid support and final land nodata after model
  prediction; that output masking is separate from the training objective.

## Relationship to other losses

The ambient diffusion term has weight 1.0 in the local preset. Optional sparse
observation, profile-increment, GLORYS structure, and spectral terms are disabled.
Auxiliary timestep weighting is therefore dormant until an auxiliary term is
enabled. See [Auxiliary losses](losses.md).

The HPC synthetic-target and direct-GLORYS presets disable ambient mode and use
ordinary dense supervision instead.
