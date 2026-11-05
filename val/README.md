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

[SCIENTIFIC_OPPORTUNITIES.md](SCIENTIFIC_OPPORTUNITIES.md) examines additional
publication-oriented tests, calibrated GLORYS baselines, joint dependence,
conditioning value, native-data matching, and the current literature.

## Quick start

Run the complete campaign with one command:

```bash
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run validate \
  --config val/validation.yaml --output val/results/paper_all
```

Open `val/results/paper_all/index.html`. This runs EN4 temperature, Italian XBT
and the salinity configuration in sequence. Missing data and compatible
salinity weights appear as blocked cohorts. Within each cohort, ten independent
modules run in sequence. Each stage exports CSV/LaTeX tables and, where applicable,
vector PDF/SVG figures plus 300 dpi PNG previews. Unexpected errors preserve a
traceback, allow subsequent stages to finish, and make the command fail.

For the frozen temperature checkpoint and both local temperature cohorts:

```bash
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run validate \
  --config val/validation_temperature.yaml --output val/results/temperature_full_20261105
```

This runs all ten stages with **one DepthDif generation** per batch and
conditioning budget, using 1,000-step DDPM and the five conditioning budgets.
It preserves the frozen EN4 cap of 256 selected patches;
it does not evaluate every validation patch. XBT uses its configured
2016 cohort without a patch cap. The cohort configs select `cuda:0`; check GPU
availability before launch. The full campaign has not been launched. Readiness
checks from 5 November 2026 are under `results/preflight_temperature_20261105/`.

| Stage | Question | Module |
| --- | --- | --- |
| Reference | Do gains survive time matching and native precision? | `techniques/reference.py` |
| AI comparators | Are trained/published comparator inputs and predictions matched? | `techniques/comparators.py` |
| Baselines | Does the ensemble beat calibrated GLORYS and climatology? | `techniques/baselines.py` |
| Regional | Where do paired, profile-balanced gains hold up? | `techniques/regional.py` |
| Conditioning | What do 0/1/2/4/8 retained context profiles contribute? | `conditioning.py`, `techniques/conditioning.py` |
| Physical | Are thermal layers and joint T/S quantities better? | `techniques/physical.py` |
| Dependence | Is vertical, T/S and spatial member dependence useful? | `techniques/dependence.py` |
| Extremes | Are frozen tail-event probabilities and tail scores better? | `techniques/extremes.py` |
| Selective | Can forecast uncertainty select reliable cases? | `techniques/selective.py` |
| Efficiency | How do quality and measured cost change with member count? | `techniques/efficiency.py` |

`paper_defaults.yaml` freezes common scientific options; override individual
stages under `paper:` in a cohort config. For discovery, 25% of complete bootstrap
clusters are reserved for calibration by a frozen hash. The paper report scores
the remaining cases. `paper/partition.csv` records membership. The older `report/`
scorecard uses all selected profiles as a separate diagnostic. Confirmation data
never supply automatic calibration labels: configure a disjoint saved bundle as
`paper.calibration_bundle`. Point, profile, cluster and physical-observation
overlap checks protect this boundary.

Set `paper.reference.native_paths` to floating-point native GLORYS NetCDF files.
Until those are supplied, the reference is explicitly **packaged GLORYS**.
Stage completion does not establish superiority. Independent confirmation,
adequate cluster support and predeclared endpoints remain necessary evidence.

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

Open `val/results/demo/paper/index.html` for all stages, or the original
`val/results/demo/report/report.html`. Its results are synthetic and clearly
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

The current validation default is `sampling.members: 1` in both the standalone
cohort configs and campaign configs. Every baseline remains included. MAE,
RMSE, signed bias and point-forecast CRPS (equal to absolute error) are available.
Ensemble fair CRPS, interval coverage, rank histograms and uncertainty-based
selection for DepthDif are marked unavailable with one generation. The sampler
still uses 1,000 DDPM steps; a generation means one complete reconstruction,
not a single denoising step. Calibration-reference sampling and the published
model's fixed output heads do not add DepthDif model passes.

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

[Cavaiola 2026](CAVAIOLA_2026.md) documents the supplied LSTM paper and an
optional adapter for its released ERA5-driven model. It produces a separate
5–200 m temperature diagnostic: the weights include 2016 in their training
period, and the fixed CRPS output heads are scored without an iid correction.

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

EN4 temperature, Italian XBT and salinity configurations all load the shared
[`baselines.yaml`](baselines.yaml) registry: **IDW, LSTM, CNN, U-Net 3D and U-Net
2D**. Neural checkpoints and their effective architecture/data/training snapshots
are pinned by SHA256 from `logs/selection/baselines/`; inference uses their standard
saved weights. Preflight verifies these inputs before evaluation. No training
configurations are edited.

Each baseline receives the identical masked batches and observation operator.
Its full-conditioning predictions are saved in `points.csv` and `members.npz`
and included in the scientific score tables, paired regional comparisons,
physical/dependence diagnostics, extreme-event scores, selective-risk curves,
and measured efficiency tables. The Cavaiola diagnostic includes all available
methods on its restricted shared support, including calibration baselines once
they have been fitted. Deterministic methods remain single point estimates.

Conditioning experiments rerun all five baselines for each retained-profile
budget. `bundle/baseline_availability.csv` and the conditioning-stage availability
table record every attempted method/budget. A method with nonfinite output on
the frozen cohort is labelled blocked for that budget; its missing values are
never filled or used to reduce another method's scoring population. In
particular, zero-input IDW and U-Net cases can be unavailable by model design.
Older saved bundles without captured baseline predictions report the missing
methods explicitly and need a new `evaluate` run to produce them.

`baselines` accepts existing baseline model configs and checkpoints, for example:

```yaml
baselines:
  - name: additional_lstm
    config: val/inputs/lstm_config.yaml
    checkpoint: val/inputs/lstm.ckpt
    checkpoint_sha256: <sha256>
    weights: standard
```

Additional method names must differ from the registry names. The same masked
batches and point assignments feed every baseline. A missing
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

Rebuild every paper stage from saved inference without loading the model:

```bash
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run suite \
  --bundle val/results/en4/bundle --config val/config.yaml --output val/results/paper_rescore
```

Conditioning experiments require arrays captured with
`paper.conditioning.enabled: true` (the default). Old bundles cannot reconstruct
missing counterfactual predictions. Both campaigns currently request one
generation per conditioning budget, and the efficiency stage scores the
single-generation forecast. Ensemble convergence curves require a separately
configured run with more members. Timing tables use measured local inference
and do not estimate GLORYS production cost.

Extremes thresholds are fitted only on calibration profiles. Joint density
diagnostics require paired T/S truth, compatible joint weights and optional
`gsw`; the frozen checkpoint supplies temperature only. Archive membership alone
does not prove non-assimilation.

## Verification

```bash
BLACK_NUM_WORKERS=1 /work/envs/depth/bin/python -m black .
bash val/run_tests.sh
```

The test wrapper runs the complete existing `tests/run_tests.sh` and then all
additional `val/test_*.py` tests, with logs under `val/results/verification/`.
Missing ocean data blocks real evaluation; it does not block the synthetic demo,
observation-operator fixtures, or scoring tests.
