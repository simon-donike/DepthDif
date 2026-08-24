# GLORYS-comparative evaluation

This page separates the validation signals used while fitting a model from the
post-training evaluation needed to compare DepthDif with GLORYS12. They answer
different questions and should not be reported interchangeably.

## Evaluation overview

| Stage | Reference | Samples | Purpose | Suitable for a “better than GLORYS” claim? |
| --- | --- | --- | --- | --- |
| Lightning `validation_step` | Active training target | Up to 64 shuffled 2016 validation batches | Select checkpoints and monitor the training objective | No |
| Lightning full reconstruction | Active target plus paired GLORYS where available | A small cached validation subset | Qualitative reconstruction diagnostics | No |
| Lightning EN4 candidate callback | Exact held-out EN4 profiles; GLORYS is evaluated against the same profiles | Fixed 2016 ISO-week candidate subset, capped to one reconstructed patch by default | In-training comparison monitor | Indicative, but not the final paper result |
| Lightning hard-region callback | Dense GLORYS | One fixed 2016 patch per provisional region by default | Detect regional disagreement with GLORYS | No; GLORYS is the reference here |
| Post-training paper workflow | Held-out EN4 profiles and persisted dense GLORYS | The exported paper-week population | Final method and baseline comparison | Yes for the EN4 target, with the candidate-evidence caveat below |

There is currently no separate Lightning `test` split. The year 2016 is the
configured validation year and is excluded from training by the temporal split.
A final reported evaluation should therefore come from the frozen-checkpoint
paper workflow, not from choosing the best-looking epoch-time callback result.

## Data split and ordinary Lightning validation

The active GeoTIFF dataset creates separate `train` and `val` dataset objects.
With `data.split.val_year: 2016`, every dated row from 2016 is assigned to
validation and dates from other years are assigned to training. The fallback
`val_fraction` is used only when a validation year is not configured. Validation
rows must contain at least one QC-valid EN4/ARGO profile in the current presets.
The accepted profile and level QC flags are 1 and 2.

One dataset row is a 128 by 128 grid patch at 0.1-degree resolution for one
target date. The patch contains:

- sparse EN4/ARGO temperature and/or salinity profiles rasterized at their grid
  locations over the native depth channels;
- dense surface SST, SSS, and ADT conditioning fields;
- a land/ocean mask, patch coordinates, and date conditioning; and
- either a dense synthetic prior target or a dense GLORYS target, depending on
  the training preset.

The ordinary validation loader intentionally remains shuffled. It evaluates at
most 64 batches per validation run in the current presets. `val/loss` and
`val/loss_ckpt` are diffusion-objective losses, not physical-unit RMSE against
independent observations. `val/loss_ckpt` selects the best checkpoint.

The meaning of that loss depends on the active stage:

- `training_super_config_hpc.yaml` is the dense synthetic-prior pretraining
  preset. Its ordinary validation target is the deterministic vertical-offset
  prior. Paired 2016 GLORYS fields are returned separately for diagnostics.
- `training_super_config.yaml` and `training_super_config_standard.yaml` are the
  sparse ambient fine-tuning presets. Their target is the original sparse ARGO
  field and their condition is a further-corrupted version of that field. The
  current hard/easy sampling configuration applies to both train and validation
  in these two presets.
- `training_super_config_spacehpc_glorys.yaml` directly supervises against dense
  GLORYS. Here the checkpoint loss measures the GLORYS-trained objective, so it
  still cannot demonstrate improvement over GLORYS.

EMA weights are used for validation when `evaluate_ema_weights_instead: true`.
The normal validation loss is distributed across workers; expensive full
reconstructions and comparison callbacks run separately at validation end.

## EN4 candidate-profile monitor

This is the epoch-time monitor that directly compares DepthDif and GLORYS with
the same observed profiles.

### Candidate and holdout selection

For the current configuration, the callback:

1. Finds the single dataset target date in ISO week 25 of 2016.
2. Reads the external candidate parquet and exact-matches records to the compact
   EN4 store using `profile_source_file` and `source_profile_idx`.
3. Applies the same profile and level QC filtering as the dataloader.
4. Keeps candidates with at least one valid temperature or salinity depth.
5. Groups profiles by `(date, grid_row, grid_col)` and selects 20% of unique
   locations without replacement using seed 7.
6. Removes every profile at each selected location from the sparse temperature
   and salinity model inputs before rasterization. The removal applies to every
   overlapping validation patch, preventing the held-out profile from entering
   the model through another patch.
7. Greedily chooses patches that cover the most selected locations. The
   epoch-time default `max_patches: 1` means only held-out profiles covered by
   that one patch contribute to the monitor. The logged eligible, selected, and
   monitored counts make this reduction visible.

This location holdout is deterministic and is not the ambient random dropout.
The model still receives all other QC-valid ARGO profiles in the selected patch,
along with SST, SSS, ADT, the land mask, coordinates, and date. The reconstruction
seed is fixed so changes between epochs primarily reflect model changes.

### Metrics

At every monitored held-out location and native depth, the callback samples:

- the DepthDif prediction;
- GLORYS12 at the same grid cell; and
- the exact held-out EN4 profile value.

A value contributes only where EN4, DepthDif, and GLORYS are all finite. Errors
are pooled over those common values. For temperature and salinity separately,
W&B logs prediction-versus-EN4 RMSE and MAE, GLORYS-versus-EN4 RMSE and MAE,
support counts, and

\[
\mathrm{skill}_{\mathrm{GLORYS}} = 1 -
\frac{\mathrm{RMSE}(\mathrm{DepthDif},\,\mathrm{EN4})}
     {\mathrm{RMSE}(\mathrm{GLORYS},\,\mathrm{EN4})}.
\]

Positive skill means DepthDif has lower pooled RMSE than GLORYS on exactly the
same held-out values; zero means equal RMSE; negative skill means worse RMSE.
Profile images show both reconstructed curves and both absolute-error curves.

## What the candidate parquet establishes

The parquet is an audit-derived candidate list, not a definitive GLORYS
assimilation blacklist. It contains 3,632,248 records marked
`no_spatiotemporal_candidate` or `proxy_no_spatiotemporal_candidate`. Its own
metadata labels 2,881,955 records as historical-release candidates and 750,293
as current `013_030` proxy candidates. The expected archives include CORA 4.1,
CORA 5.0, and CORA 5.1; some source coverage is explicitly marked unresolved or
unavailable. This repository consumes the supplied provenance IDs but does not
independently reproduce the upstream audit.

The defensible interpretation is:

> These are EN4 profiles for which the audit found no spatiotemporally nearby
> profile in the inspected GLORYS source release or proxy.

This reduces the risk that the comparison merely evaluates GLORYS at an
observation implicitly used in its construction, but it does not prove that an
exact observation was never assimilated by GLORYS. A stronger claim would need
official assimilation feedback, rejection/usage flags, or an authoritative
observation-ID inventory from the GLORYS producer.

Accordingly, paper text should call these “no-nearby-source candidates” or
“reduced-assimilation-leakage candidates,” not “guaranteed unassimilated
profiles.” The important experimental comparison remains fair: DepthDif and
GLORYS are scored against the same profile values, while those values are hidden
from DepthDif's sparse input.

## Ambient dropout and other masking

Several different operations can look like dropout:

### Fixed EN4 location holdout

The 20% candidate holdout described above is fixed by seed and location. It is
used specifically for independent-profile comparison and removes the selected
points from every overlapping validation input.

### Ambient observation dropout

In the current fine-tuning presets, `ambient_occlusion.enabled: true` and
`further_drop_prob: 0.25`. For every train and ordinary validation batch, the
model starts from the actual sparse ARGO validity mask and independently hides
approximately 25% of its observed spatial positions. With
`shared_spatial_mask: true`, the same spatial draw is expanded across depth
channels and active variables. Dropout can only remove observations; it never
creates new ones. A minimum-support guard restores observations if too few are
left.

The corrupted sparse field is passed as the condition. The original sparse
field is the target, and supervision is gated by the intersection of original
ARGO support and valid target support. Thus this is a stochastic self-supervised
reconstruction objective; it is not the same as the fixed candidate holdout.
Ambient dropout is disabled in the synthetic-prior and direct-GLORYS HPC
presets, even though their inactive dropout parameters remain in the YAML.

### EO dropout and diffusion noise

No SST/SSS/ADT dropout is active in the current dataset path. The model therefore
receives the configured surface fields during these evaluations. Diffusion
timestep/noise sampling is part of the diffusion loss and is separate from both
ARGO holdout mechanisms.

## Hard-region GLORYS monitor

The hard-region callback runs during every non-sanity Lightning validation. It
uses the 2016 validation dataset and the provisional hand-drawn Greenland,
California/Baja, and Beaufort/Western-Arctic polygons. For each region it:

1. finds validation patches whose footprints intersect the polygon;
2. sorts them by date and grid position and selects the middle candidate when
   `max_patches_per_region: 1`;
3. runs a seeded full reconstruction through the normal validation dataset;
4. intersects the result with the polygon, valid GLORYS support, and ocean mask;
5. logs physical-unit RMSE and MAE between the prediction and GLORYS; and
6. logs GLORYS, prediction, and absolute-error images at the nearest native
   channels to 0 m, 100 m, and 500 m.

This monitor asks, “Where and by how much does DepthDif disagree with GLORYS in
known difficult regions?” It does not ask which one is closer to reality. Lower
error here means greater GLORYS agreement, not evidence that DepthDif outperforms
GLORYS. The polygons can be replaced in
`src/depth_recon/configs/evaluation/hard_regions_2016.yaml` without changing
their stable IDs.

## Recommended final comparison

Use the Lightning EN4 callback for regression detection during training. For
reported results, freeze the chosen checkpoint and run the paper-week workflow
described in [Inference](inference.md#paper-week-inference-bundle-and-metrics-export).
It exports the selected holdout once, applies the same hidden locations to every
learned method, samples GLORYS at those exact locations, and computes comparable
per-depth and pooled metrics from persisted artifacts.

The central paper result should therefore compare DepthDif and GLORYS against
held-out EN4 on identical support, report the candidate-audit limitation, and
keep dense hard-region prediction-versus-GLORYS maps as a separate diagnostic.
