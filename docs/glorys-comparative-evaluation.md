# GLORYS-Comparative Evaluation

DepthDif has three complementary validation views. None is independent ground
truth for the complete ocean: they answer different repository-level questions
and must retain their source and split provenance.

## Lightning validation

The ordinary validation dataloader uses the configured 2016 split and remains
shuffled intentionally. It measures the same objective and optional metrics used
during training. Resolved run configs record the target mode, sampler, scenario,
and callback settings required to interpret the values.

When EMA evaluation is enabled, standard and EMA reconstruction branches are
logged separately. Do not compare metrics across checkpoints without also
matching sampler and target configuration.

## EN4 candidate-profile monitor

The EN4 callback selects candidate profile locations for a fixed ISO week in the
validation year. A seeded location-level holdout is removed from the shared
sparse validation input, including overlapping patches and both sparse variables.
The callback then compares reconstructed values at the held-out profile cells and
depths.

This tests recovery at real withheld observations. Candidate Parquet files and
run metadata provide the exact locations, fraction, seed, and source provenance;
the documentation does not embed machine-local candidate counts.

## Hard-region GLORYS monitor

The hard-region callback selects deterministic validation patches from named
polygons in `hard_regions_2016.yaml` and compares reconstruction with GLORYS over
valid ocean support. The polygons are provisional hand-authored diagnostic
regions. They are not literature-backed basin definitions and are unsuitable as
standalone evidence for scientific claims.

Current pixel presets enable both candidate-profile and hard-region callbacks.
The local preset also applies a 50% hard-region row mix to both train and
validation datasets; the HPC presets disable that row filtering while retaining
the evaluation callbacks.

## Masking distinction

- Location holdout removes selected EN4 cells before model conditioning.
- Ambient occlusion randomly drops a subset of the remaining observations during
  a training or validation step.
- Diffusion noise perturbs the target/noisy branch according to timestep.
- Dense surface dropout, where configured, affects conditioning modalities.

These mechanisms are not interchangeable. Reports should state which were
active and should use the resolved config stored beside the checkpoint.

## Recommended reporting

For one checkpoint and scenario, report:

1. ordinary 2016 validation metrics with sampler and EMA/raw branch;
2. withheld EN4 candidate metrics with candidate artifact, week, fraction, and seed;
3. hard-region GLORYS metrics with region-config revision;
4. optional global or paper-bundle metrics with exact exported artifact provenance.

This combination exposes objective-level behavior, observation recovery, and
dense regional comparison without presenting any one source as universal truth.
