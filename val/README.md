# Frozen residual validation

Everything added for this validation lives here. The training application,
existing tests, checkpoints, and deliberately shuffled validation loader are
unchanged. Commands run from the repository root with
`/work/envs/depth/bin/python`; add `PYTHONPATH=src` if the package is not installed.

The scientific protocol and primary references are in
[VALIDATION_PLAN.md](VALIDATION_PLAN.md). Start with the regional scan, then freeze
the regions and endpoint before opening an independent confirmation cohort.
The existing 2016 split was used for checkpoint selection and is **discovery
data**, even when an input profile is withheld at inference.

[VERIFICATION.md](VERIFICATION.md) records the completed tests, real checkpoint
smoke, recovered XBT data, and remaining scientific evidence gates.

## Quick start

Inspect data readiness without loading the model:

```bash
/work/envs/depth/bin/python -m val.run preflight \
  --config val/config.yaml --output val/results/preflight
```

An executable demonstration exercises temperature and salinity, matched baselines,
uncertainty scores, regional comparisons, and report generation without ocean
data or a GPU:

```bash
/work/envs/depth/bin/python -m val.run demo --output val/results/demo
```

Open `val/results/demo/report/report.html`. Its results are synthetic and clearly
labelled. Output directories must be new and underneath `val/`; failed runs retain
their status and error instead of being overwritten.

After restoring the data, run the frozen EN4 benchmark:

```bash
/work/envs/depth/bin/python -m val.run evaluate \
  --config val/config.yaml --output val/results/en4
```

The default pins **QC residual run `1419568d`, best step 2,000, EMA weights**,
using the recorded SHA-256. This is the best retained checkpoint within the chosen
QC fine-tuning run. It is not selected by new regional results. Its resolved
architecture is temperature-only, with SST surface conditioning.
`val/config_salinity.yaml` is prepared for a salinity checkpoint. A joint checkpoint
can use `sampling.variables: [temperature, salinity]`. Switching the scenario of
temperature weights does not create a salinity model; capability and strict weight
checks reject that mistake.

The default uses 20 independent DDPM members, matching the original sampler. This
can take substantial GPU time. For a faster development scan, copy the config
under `val/` and explicitly test DDIM with fewer steps on discovery data first.
Use the same frozen sampler for the comparisons you report. For final probability
claims, increase ensemble size and check convergence on a fixed subset.

## What to open

Each successful evaluation writes:

| File | Purpose |
| --- | --- |
| `report/report.html` | Readable regional scorecard, confidence intervals, coverage, and evidence gates |
| `report/report.md` | Text version of the conclusions |
| `report/scores.csv` | Point and probabilistic scores, separate physical variables and support counts |
| `report/comparisons.csv` | Paired cluster-bootstrap differences versus each baseline; negative favours DepthDif |
| `report/reliability.csv` | Coverage, interval width, and rank histogram |
| `report/by_depth.csv` | Depth-resolved diagnostics |
| `report/regional_decisions.csv`, `report/counts.csv` | Regional verdicts and actual support |
| `report/coverage.csv` (EN4) | Eligible versus selected versus scored profiles, including untested regions |
| `report/summary.json` | Machine-readable conclusions and confirmation eligibility |
| `bundle/points.csv`, `bundle/members.npz` | Matched observations/references and the complete ensemble at those points |
| `bundle/metadata.json` | Exact checkpoint/config/input hashes, seeds, timings, and provenance |
| `bundle/exclusions.csv` | Cases excluded using QC, reference support, or conditioning criteria |
| `cohort.csv`, `holdouts.json` | Selected points and global conditioning exclusions |
| `preflight.json`, `status.json` | Readiness and completion/failure record |

Errors are reported in °C or practical salinity units separately. Ensemble-mean
RMSE and ensemble-median MAE are point scores; CRPS scores all members. Deterministic
GLORYS and climatology are point-mass distributions, whose CRPS equals absolute
error. Beating those point masses alone does not establish superiority over a
well-calibrated probabilistic climatology. Calibration width matters alongside
coverage; a very wide interval can cover everything without being useful.

Regional intervals resample clusters, after averaging losses within each profile.
EN4 defaults to 5° tile × month clusters when platform identities
are unavailable. These are spatial/time blocks, not proof of independent floats.
XBT clusters use entire cruises. The default support requirement is 30 profiles
and 20 clusters; change it only in a versioned protocol. Sparse cruise cohorts
may be descriptive rather than statistically decisive. All searched regional
comparisons retain multiplicity correction and exploratory labels.

The EN4 adapter reuses the existing compact profile store. Its observations have
already been interpolated to model levels and quantized by the dataset exporter;
they are not native raw NetCDF profiles. Preserve that representation and its
quantization uncertainty in any publication. Small apparent gains below the
observation/export precision need a native-data sensitivity check.

## Italian XBT data found locally

The extracted `evaluations/inputs/italian_xbt/` directory is empty, but the raw
`xbt_PNRA_*.txt` files are inside **`evaluations/focus_evaluation.tar.gz`**.
`evaluations/outputs/italian_en4_no_match_profiles.parquet` holds 2,137 screened
profile identities, including cruise, station, timestamp, and source filename.
The importer reads the archive without changing or extracting into `evaluations/`:

```bash
/work/envs/depth/bin/python -m val.import_xbt \
  evaluations/focus_evaluation.tar.gz val/results/xbt_import/profiles.csv \
  --metadata evaluations/outputs/italian_en4_no_match_profiles.parquet \
  --manifest val/results/xbt_import/manifest.json --year 2016 --accepted-qc-flag 1
```

Use corrected `Depth 3` and `Temperature 2`, retaining `QF=1`. The importer matches
the screened identity table; the absence of an archive match remains screening
evidence, not verified non-assimilation. XBT measures temperature only. The source
dataset and corrections are described by
[Aulicino et al. (2025)](https://essd.copernicus.org/articles/17/2625/2025/).

Prepare the salinity reference used by the common temperature conversion:

```bash
/work/envs/depth/bin/python -m val.prepare_xbt val/results/xbt_import/profiles.csv \
  --data-root /work/data/OceanVariableReconstruction \
  --output val/results/xbt_reference/profiles.csv \
  --manifest val/results/xbt_reference/manifest.json --year 2016 \
  --max-time-delta-days 3.5 --max-vertical-gap-m 100
```

`val/config_xbt.yaml` selects the prepared cohort and reads its SHA-256 from the
preparation manifest. It intentionally records
`temperature_kind: in_situ`: GLORYS `thetao` and the trained temperature model use
potential temperature. Follow the explicit conversion option in the observation
adapter before running the primary comparison; never simply relabel the XBT
temperature. The optional TEOS-10 dependency can be installed locally:

```bash
/work/envs/depth/bin/python -m pip install --no-deps --target val/vendor -r val/requirements-optional.txt
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run evaluate \
  --config val/config_xbt.yaml --output val/results/xbt
```

This leaves the shared environment unchanged. The prepared salinity comes from
GLORYS, so its use in the observation conversion is a common assumption, not
independently observed salinity. Report a salinity sensitivity check for small
temperature gains. The native-level gap cap also limits deep support; its exclusions
are listed in the preparation manifest and should be kept in the report.
The data include training-era cruises: keep the default discovery
label unless the exact checkpoint's training windows were independently excluded.

## Other observation cohorts and baselines

External CSVs use one row per observed depth, with:

```text
profile_id,cluster_id,source,observation_time,longitude,latitude,depth_m,
variable,observation,independence
```

Use globally stable IDs and whole float/deployment/cruise IDs for clusters.
`variable` is `temperature` or `salinity`; observations are °C or practical salinity.
Optional fields include `qc_flag`, `region`, and `temperature_kind`. The adapter
preserves raw observation depths, uses the nearest product date within 3.5 days,
and interpolates every method vertically within valid model brackets without
extrapolation. Horizontal matching uses the containing raster cell, consistently
for all methods. Explicit regions are preserved; otherwise 20° tiles give a
predeclared scan grid. These tiles are analysis masks, not physical ocean basins.

Both variables and all depths of evaluation locations are removed from every
overlapping input patch. Platform exclusion is expanded when matching metadata
exist; otherwise the report records the coordinate-cell limitation. Observations
and references establish the common scoring support before predictions are
generated. Invalid predictions fail the run rather than improving scores by
silently dropping difficult points.

`baselines` accepts existing baseline model configs and checkpoints, for example:

```yaml
baselines:
  - name: lstm
    config: val/inputs/lstm_config.yaml
    checkpoint: val/inputs/lstm.ckpt
    checkpoint_sha256: <sha256>
    weights: standard
```

The same masked batches and point assignments feed every baseline. A missing
learned checkpoint is an error. A checkpoint-free IDW configuration is supported;
depths with no interpolation support fail unless a fallback is explicitly
implemented and named in that baseline. No AI comparison is claimed when the
list is empty. Published reconstruction approaches and information-matching
requirements are listed in the scientific plan.

You can score a saved bundle again without sampling:

```bash
/work/envs/depth/bin/python -m val.run score \
  --bundle val/results/en4/bundle --output val/results/rescore
```

For extremes, supply training-only `threshold_low`/`threshold_high` columns and
the required external threshold provenance metadata. Do not estimate thresholds
from the verification observations. Joint energy scores, physically integrated
diagnostics, probabilistic climatology fitting, and publication section figures
are follow-up experiments in the plan, not results supplied by the current runner.

## Verification

```bash
BLACK_NUM_WORKERS=1 /work/envs/depth/bin/python -m black .
bash val/run_tests.sh
```

The test wrapper runs the complete existing `tests/run_tests.sh` and then all
additional `val/test_*.py` tests, with logs under `val/results/verification/`.
Missing ocean data blocks real evaluation; it does not block the synthetic demo,
observation-operator fixtures, or scoring tests.
