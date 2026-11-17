# Corrected residual HPC handoff

This dedicated preset is based on `dev-ruben` (`6deeeeaa`) and leaves Ruben's
existing presets unchanged. It preserves his Lustre paths, automatic GPU
allocation, workers, mixed precision, and offline W&B setup. The temperature/SST
residual architecture uses seed 7, learning rate `1e-4`, EMA `0.999`, QC flags
1/2 with the existing missing-QC policy, and dense corrected GLORYS supervision.
Ambient training, synthetic targets, and hard-region sampling are disabled.

The effective training batch is exactly 48, with accumulation 1. The loader
uses the actual distributed world size, rejects non-divisors of 48, and drops
both sampler padding and incomplete training batches. Ordinary validation
remains shuffled.

| GPUs | Samples per GPU | Global batch |
| ---: | ---: | ---: |
| 1 | 48 | 48 |
| 2 | 24 | 48 |
| 4 | 12 | 48 |
| 8 | 6 | 48 |

The corrected residual run requires a complete copy of the exported dataset at
`/lustre/projects/1000/rubenc/ocean_data/ocean_data` and its small immutable
bundle at `hpc_residual/` below that root. The bundle preserves the Italian
reference manifest exactly as produced; its historical local paths are checked
against their relocated files without rewriting provenance.

## Obtain the code

The local handoff branch is `hpc-residual-corrected`, based on `dev-ruben`.
The handoff directory also contains `hpc-residual-corrected.patch` and an
incremental `hpc-residual-corrected.bundle`. The bundle requires the recorded
`dev-ruben` base commit already present in the receiving clone:

```bash
git fetch /path/to/hpc-residual-corrected.bundle hpc-residual-corrected:hpc-residual-corrected
git switch hpc-residual-corrected
```

The large data bundle is transferred separately using the mapping below.

## Prepare the local transfer bundle

The handoff already includes the frozen bias and staged assets. The following
commands reproduce their preparation; use the supplied frozen artifact for the
HPC run.

Fit the signed EN4-minus-GLORYS bias before staging. It excludes both profile
and assigned GLORYS-target dates from 2016. Source file/profile identity is
counted once at each depth, independently of overlapping patches. The frozen
artifact records counts, sample variance, support, fitting settings, depth/grid
metadata, and hashes of its source data. Fewer than 30 profiles gives a zero
offset and an unsupported flag at that depth.

The signed correction is `b[z] = mean(EN4[z] - GLORYS[z])` in potential
temperature degrees Celsius, and the target is `G' = G + b`. Correction occurs
before normalization in both training and validation. The exact existing
climatology `C` stays unchanged: training predicts `G + b - C`, and inference
adds `C` once. No further bias is added at inference. Actual 2016 observation
dates are excluded from training inputs even when assigned across a year boundary.

```bash
PYTHONPATH=src /work/envs/depth/bin/python -m depth_recon.data.fit_target_bias \
  --root /work/data/OceanVariableReconstruction \
  --output /work/code/DepthDif/outputs/hpc_residual_handoff_20261117/temperature_bias_excluding_2016.npz \
  --val-year 2016 --min-profiles 30 --chunk-size 50000
```

Create the full, strict inventory and small bundle. This streams SHA-256 over
every relevant dataset file and reports progress every 1,000 files. It excludes
only `.cache`, `.git`, `depthdif_cache`, Python bytecode caches, and a nested
`hpc_residual` bundle. It does not copy the 118 GB dataset tree itself.

```bash
PYTHONPATH=src /work/envs/depth/bin/python -m depth_recon.scripts.prepare_hpc_handoff stage \
  --dataset-root /work/data/OceanVariableReconstruction \
  --climatology /work/code/DepthDif/outputs/checkpoint_selection_20261005/baselines/temperature_monthly_stride4.npz \
  --target-bias /work/code/DepthDif/outputs/hpc_residual_handoff_20261117/temperature_bias_excluding_2016.npz \
  --italian-profiles /work/code/DepthDif/val/results/xbt_reference/profiles.csv \
  --italian-manifest /work/code/DepthDif/val/results/xbt_reference/manifest.json \
  --italian-screening /work/code/DepthDif/evaluations/outputs/italian_en4_no_match_profiles.parquet \
  --source-archive /work/code/DepthDif/evaluations/focus_evaluation.tar.gz \
  --output-dir /work/code/DepthDif/outputs/hpc_residual_handoff_20261117/hpc_residual \
  --hpc-destination /lustre/projects/1000/rubenc/ocean_data/ocean_data
```

For a non-authoritative readiness listing, use `--inventory-only --no-full-hash`.
It cannot pass preflight. `--reuse-prior-hashes` is an explicit staging shortcut;
preflight still streams and hashes every transferred file.

## Transfer and strict preflight

Use `path_map.json` as the source-to-HPC mapping: the complete dataset source
maps to the HPC dataset root, while every copied asset maps below
`hpc_residual/`. Transfer the dataset and bundle separately, then run preflight
on the cluster filesystem before requesting a long GPU allocation.

```bash
rsync -aH --info=progress2 --exclude=.cache --exclude=.git --exclude=depthdif_cache \
  --exclude=__pycache__ --exclude=hpc_residual /work/data/OceanVariableReconstruction/ \
  hpc-login:/lustre/projects/1000/rubenc/ocean_data/ocean_data/
rsync -aH --info=progress2 \
  /work/code/DepthDif/outputs/hpc_residual_handoff_20261117/hpc_residual/ \
  hpc-login:/lustre/projects/1000/rubenc/ocean_data/ocean_data/hpc_residual/

/work/envs/depth/bin/python -m depth_recon.scripts.prepare_hpc_handoff preflight \
  --bundle-dir /lustre/projects/1000/rubenc/ocean_data/ocean_data/hpc_residual \
  --dataset-root /lustre/projects/1000/rubenc/ocean_data/ocean_data
```

Preflight requires the full asset roster, hashes every inventory entry and
checks size and digest, expands manifest-referenced directory trees including
all Zarr chunks, validates the climatology’s shape/grid/depth/year, loads and
validates the target-bias artifact, and checks the original Italian manifest’s
prepared-profile, dataset-manifest, and raw-salinity hashes after relocation.

## Local verification status

As of this handoff, the complete local suite has passed 441 tests. Strict
preflight verified 72,620 files in the transfer inventory, covering a 110.2
GiB source dataset. The frozen bias fit used 6,111,078 profiles and supports
49 of 50 depth bands; the unsupported 5727.9169921875 m band has a zero
correction. The final normal two-GPU smoke run passed on two local RTX 3090s
in 244.14 seconds, with peak GPU memory of 9,890 and 9,886 MiB. It completed
two optimizer updates at global batch 48, saved fresh optimizer/scheduler/EMA
state, and scored all 102 Italian profiles (79,367 points, 14 patches) with
DDPM-1000 and active EMA. The completed Italian epoch is in the checkpoint.

The local smoke used zero loader workers to fit host RAM; the HPC preset
retains Ruben's four workers and its other operational settings. Ordinary
validation previews were enabled during the final smoke. See `verification.json`,
`checkpoint_audit_006.json`, and `logs/local_smoke_command.json` in the handoff
directory for exact evidence and local path overrides. No HPC job was submitted.

## Cluster environment, smoke run, and training

Install the package and the TEOS-10 dependency on a login node before moving to
offline compute nodes:

```bash
/work/envs/depth/bin/python -m pip install -e '.[hpc]'
```

After a successful preflight, use the branch’s normal scheduler wrapper for a
bounded smoke allocation. Limit the run to two optimizer updates and one full
validation epoch so it includes one DDPM-1000 Italian evaluation; record GPU
memory and wall-clock time from the scheduler output. Do not resume a previous
checkpoint.

```bash
/usr/bin/time -v /work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config_spacehpc_residual_corrected.yaml \
  --run-dir logs/hpc_corrected_smoke_001 \
  --set training.trainer.max_epochs=1 \
  --set training.trainer.limit_train_batches=2 \
  --set training.trainer.limit_val_batches=1 \
  --set training.training.reconstruction_eval.sample_count=2 \
  --set training.training.reconstruction_eval.batch_size=1 \
  --set training.training.validation_sampling.full_reconstruction_logging_enabled=false \
  --set training.training.hard_region_eval.enabled=false
```

Italian validation still evaluates the full cohort with DDPM-1000 in this smoke
command. The two checkpoint samples use separate inference batches, and redundant
ordinary-validation preview plots are disabled for this bounded check. Check
`resolved_training_batch.json`, `final.ckpt` (global step 2), and
the Italian epoch JSON for the prepared/eligible/scored counts. Use a new run
directory for every attempt. Record peak GPU memory with the cluster's GPU
monitor alongside the wall time. This implementation is prepared locally;
the cluster smoke run is the colleague's pre-production check.

For production, use a new directory and omit all smoke overrides:

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config_spacehpc_residual_corrected.yaml \
  --run-dir logs/hpc_corrected_residual_001
```

The preset rejects checkpoint initialization/resume and W&B identity reuse.
Reconstruction score `val/full_reconstruction_score` selects checkpoints and
drives the epoch-based plateau scheduler. The Italian monitor is diagnostic:
an independent dataset holds out Italian conditioning locations, runs one
DDPM-1000 reconstruction per patch with EMA weights and fixed batches/seeds,
and logs under `italian_validation/`. Its last completed epoch is checkpointed.
Python, NumPy, and torch RNG state and sampling settings are restored afterward.

Model, original GLORYS, corrected GLORYS, and climatology use identical
observational support after TEOS-10 in-situ-to-potential conversion. Metrics
include MAE, RMSE, signed bias, counts, and skill against original GLORYS overall
and by depth band, both pooled and equally weighted by profile. Profile-balanced
RMSE is `sqrt(mean(profile MSE))`. Preserve the salinity rasters referenced by
the Italian manifest. This is development validation; its labels never fit the
correction or automatically select checkpoints. Only the corrected run is
planned, so comparison with the existing checkpoint cannot isolate the
correction's contribution.

The prepared Italian cohort is development validation. Its compact profile
store lacks platform and cruise metadata, so a whole-platform or
whole-cruise conditioning exclusion cannot be verified. The implemented
holdout excludes the weekly coordinates and their one-cell buffer from
conditioning; interpret its metrics with that limitation.

The climatology runtime cache extracts only needed arrays to shared read-only
memory maps. `SLURM_TMPDIR` is used when available; set
`DEPTH_RECON_CLIMATOLOGY_CACHE_DIR` to another writable node-local directory if
needed. This derived cache does not replace the original checksummed NPZ.
