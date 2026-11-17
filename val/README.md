# Frozen residual validation

The validation workflow lives here; its published dashboard and snapshots live
under `docs/`. The training application, checkpoints, and deliberately shuffled
validation loader are unchanged. Commands run from the repository root with
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
availability before launch. On 17 November 2026 the initial sequential campaign
was stopped at the user's request and restarted under
`results/temperature_parallel_20261117/`: EN4 on physical GPU 0 and XBT on physical
GPU 1. That directory's `index.html`, `status.json` and separate cohort logs track
the run; final results are pending. The frozen config snapshots and exact runner
are retained there. Registry checks are under `results/screening_restart_20261117/`.

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
| `profile_membership.csv`, `profile_membership.json` | Every selected profile checked against the frozen screened registry before inference |

EN4's registry is `instructions/en4_no_spatiotemporal_candidate_profiles.parquet`;
only the configured year and `no_spatiotemporal_candidate` status qualify. XBT's
registry is `evaluations/outputs/italian_en4_no_match_profiles.parquet`, as named
in `evaluations/plans/glorys12_weaknesses_plan_short.md`. The temperature configs
pin both registry hashes. Missing IDs, conflicting timestamps, altered registry
contents, or mismatched XBT cruise/station/file/coordinates/QC abort the run.
XBT's entire prepared input is also checked under `input_screening/` before
matching. Registry membership verifies the documented screened selection; it
does not establish that GLORYS never assimilated a profile.

The paper regional stage additionally writes `paper/04_regional/depth_scores.csv`
and `.tex` for **every available method**, using only the evaluation partition.
EN4 retains exact model depths; raw XBT depths use fixed 25 m bins (left-closed,
right-open), configurable with `paper.regional.depth_bin_m`. Losses are averaged
within each profile before averaging across profiles; RMSE is the square root of
that profile-balanced mean squared error. Tables include MAE, RMSE, signed bias,
CRPS, and point/profile/cluster counts. `depth_accuracy_temperature.pdf`, `.svg`
and `.png` plot MAE, RMSE and bias against depth. These curves are descriptive;
the paired depth-band comparisons provide cluster-bootstrap intervals where
support permits. No calibration profiles enter these depth tables or figures.

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
# Website validation snapshot

### Italian ensemble and spatial metrics

`val.italian_ensemble` generates 20 members for every frozen Italian XBT batch,
split across physical GPUs 0 and 1. It preserves the 1,000-step DDPM sampler,
EMA checkpoint, original batch pairing, full masked conditioning, and the
single-generation seed as member zero. Conditioning hashes and member-zero
point predictions must match the original campaign. Samples are saved atomically
with checksums and can be resumed using identical arguments.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/italian_ensemble_20261117 --shard 0 --members 20
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/italian_ensemble_20261117 --shard 1 --members 20
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.score_italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --ensemble val/results/italian_ensemble_20261117 --spatial val/results/xbt_spatial_all_20261117 --output docs/assets/validation/italian_xbt_temperature/ensemble --device cuda:0
```

Launch the workers concurrently; score only after both finish. CRPS and fair
CRPS use all 102 casts / 79,367 measurements, with equal cast weights and native
depth interpolation. Coverage and width use sample 5th–95th percentiles.
Original single-generation tables remain separate. No EN4 inference is repeated.

Before scoring, replay `val.export_spatial` with `--all-profiles` and work directory
`val/results/xbt_spatial_all_20261117` to include both missions. Spatial comparisons
use all 14 patches with verified saved neural baseline fields for every method,
checking wet support across all 50 model depths. LPIPS 0.1 uses
pretrained VGG16, repeats each physical temperature layer across three channels,
and maps a fixed clipped range of −2 to 35 °C to [−1, 1]. Common invalid pixels
have identical zero fills; spatial distances are averaged over wet masks eroded
16 cells. Receptive fields may still cross coasts: this is an exploratory
natural-image metric, not an ocean-trained perceptual score.

Spectra reuse the repository's physical-spacing, plane-detrended, Hann-windowed
2D FFT implementation, with common masks and 32 wavelength bins over 30–1,000 km.
The scalar spectral error is RMSE of log10 power against packaged GLORYS,
with a common 1e-12 °C² power floor so zero-power predictions are not discarded.
Individual-member power is averaged before comparison; the ensemble mean is
reported separately. Spatial scores average depths and then patches equally.
GLORYS self-scores are zero by construction; these metrics measure similarity to
the reference rather than independent observational accuracy. CSV exports retain
patch/depth scores and spectra, and the website offers depth-selectable curves.

The `/validation/` dashboard exports saved evaluation results only. Refresh it after
additional campaign stages or EN4 inference finish:

```bash
/work/envs/depth/bin/python -m val.export_dashboard --campaign val/results/temperature_parallel_20261117 --output docs/assets/validation
```

The exporter checks bundle hashes, rejects overlapping calibration/evaluation
profiles or clusters, preserves the paper's aggregate scores, and publishes every
evaluation profile at its saved observation depths. Unfinished cohorts have no
placeholder scores. Only completed-stage artifacts are included in the download
pack; pointwise diagnostic files above 2 MB remain in the original campaign.
The static site must be rebuilt and published after refreshing the snapshot.

Italian fixed-model comparisons now use all 102 saved casts (79,367 points)
from PNRAXXXI and PNRAXXXII, with no calibration. The earlier report reserved
14 casts from PNRAXXXII for fitted correction/uncertainty comparators; DepthDif
itself was never fitted to them. Preserve that original report and generate the
new analysis from the existing prediction bundle:

```bash
/work/envs/depth/bin/python -m val.score_all_profiles --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/temperature_parallel_20261117/italian_xbt_temperature/paper_all_profiles
```

The exporter prefers this report only when its completion marker, bundle hash
and complete observation membership pass validation. Combined and per-mission
scores use fixed predictions without fitting corrections. Two missions remain
below the independent-cluster threshold for confidence/significance claims.
`config_xbt.yaml` also disables calibration and fitted baseline stages for future
Italian runs. EN4 retains its own separate analysis configuration.

Full-context XBT inference uses all available QC-valid EN4 context in the matched
patches/times, irrespective of GLORYS archive membership, after target-cell and
one-cell-buffer masking. The screened registries constrain evaluation targets,
not all conditioning profiles. Italian and EN4 scores remain separate; reduced
context budgets are separate ablations. Italian archive screening does not prove
non-assimilation from GLORYS assimilation logs.

### Spatial prediction images

The dashboard comparisons retain original packaged GLORYS as the GLORYS
reference. Bias-calibrated GLORYS and both GLORYS- and climatology-background optimal interpolation
(including all context budgets) are excluded from the published dashboard
tables, curves, profile series, downloadable reports and CSV/LaTeX exports. The export filter
preserves retained scores and does not modify the frozen campaign evidence.

The validation page uses a Leaflet map to select evaluation profiles and compare
DepthDif, packaged GLORYS and XBT temperature curves. The spatial viewer displays
all model depths with a shared colour range across methods and depths per patch.
The map includes all 102 casts in separate mission layers and verified EN4
temperature support cells from all 14 saved input patches. The default animation
patch is selected by coverage, not error: more than five XBT casts and more than
five EN4 cells, then largest cast count (patch 12896: 13 casts / 15 EN4 cells).
The all-profile replay extends the original 12-patch image archive to all 14
Italian patches, including both missions.
Playback advances every 500 ms and loops; manual depth changes, patch/cohort
changes and hiding the browser tab pause it. Cyan circles show every retained
EN4 temperature input cell, with filled centres at supported depths. XBT casts
use crosses (red for the selected cast). These are masked model-grid support
locations, not unique source-profile identities. Method and metric information
buttons explain definitions, weighting, conditioning and interpretation limits
on hover, keyboard focus or tap.

The point bundles do not retain complete prediction grids. Generate verified
spatial replays separately, without restarting the campaign:

```bash
CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -m val.export_spatial --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --work val/results/xbt_spatial_20261117 --output docs/assets/validation/italian_xbt_temperature/spatial --device cuda:0
```

Use a new `--work` directory for each replay. The exporter preserves the original
batch sizes and seeds, verifies conditioning tensor hashes, checks native-depth
predictions against saved values (absolute tolerance 1e-5 °C, relative tolerance
1e-6), and exports only patches containing evaluation profiles. GLORYS validity
and land masks apply to every field; missing model values are transparent. Each
PNG contains the original grid at all model depths, stacked vertically. Physical
arrays remain in the local work directory. Re-run `val.export_dashboard` after
the spatial export to add the manifest to the website snapshot.

To add verified support markers to existing images without rerunning inference:

```bash
/work/envs/depth/bin/python -m val.export_spatial --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --work val/results/xbt_supports --output docs/assets/validation/italian_xbt_temperature/spatial --device cpu --supports-only
```

This reads the saved holdouts, rebuilds the original batches and checks every
input tensor hash before exporting `x_valid_mask` cell locations and depth
availability. It checks the existing manifest's checkpoint, depth axis and
evaluation patch set before atomically updating the manifest.

For LSTM spatial replay, independent pixel sequences are processed in groups of
1,024 on the GPU to bound recurrent workspace memory. Each full depth sequence,
conditioning feature, and outer patch batch is preserved. Frozen point predictions
must still pass the same numerical agreement check before publication.

### Preliminary EN4 subset on two GPUs

`val.quick_en4` selects 64 patches from the frozen 256-patch EN4 campaign using
month and 20-degree region coverage, with seeded tie ordering and no prediction
values. It retains original two-patch batches, residual member seeds, checkpoint,
DDPM steps, and **all** original masked target cells, including cells outside the
subset. Both workers use all remaining inputs. The additional conditioning-budget
experiments and fitted calibration methods are disabled; every supported selected
profile is evaluated. This is a preliminary coverage sample, not an area-weighted
global estimate or independent confirmation.

```bash
/work/envs/depth/bin/python -m val.quick_en4 prepare --source val/results/temperature_parallel_20261117/en4_temperature --output val/results/en4_quick_20261117/en4_temperature --patches 64 --seed 20261117
CUDA_VISIBLE_DEVICES=0 /work/envs/depth/bin/python -m val.quick_en4 worker --output val/results/en4_quick_20261117/en4_temperature --shard 0
CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -m val.quick_en4 worker --output val/results/en4_quick_20261117/en4_temperature --shard 1
/work/envs/depth/bin/python -m val.quick_en4 merge --output val/results/en4_quick_20261117/en4_temperature
```

Run the two worker commands in separate processes. Each two-patch bundle is saved
with checksums before the next batch. Completed bundles can be reused; incomplete
batch directories are deliberately not overwritten. Merge requires every frozen
batch and excludes an incomplete baseline from the entire shared comparison,
without removing observations to improve its coverage. Tables and figures use
exact EN4 model depths; the paper report has no calibration partition.
