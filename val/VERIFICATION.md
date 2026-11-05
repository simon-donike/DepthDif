# Verification on 5 November 2026

## TL;DR: comparison datasets and implemented checks

**We have two temperature comparison datasets, but neither is yet proven absent
from GLORYS assimilation.** EN4 and XBT are archive-screened candidates.
All counts below concern **2016**.

| Dataset | Available data | Evidence of absence from GLORYS |
| --- | --- | --- |
| **EN4** | **297,561 archive-screened profiles**, before compact-store matching, QC and selection. The current run selects up to **256 patches**; the final scored profile count remains unknown. | No nearby match in the supplied historical archive within 24 hours / 25 km. **Not verified non-assimilation.** |
| **Italian PNRA XBT** | **102 profiles, two cruises; 79,367 prepared temperature measurements**, approximately 0.84–541 m. | Screened against the historical archive. **Not verified non-assimilation.** |

These are available/prepared counts. Matching, support checks and calibration
splitting reduce the final evaluation population. The EN4 count is the number
of unique source-file/profile identities in the audit with status
`no_spatiotemporal_candidate` and observation year 2016; it is not the number
of QC-supported profiles selected for scoring.

The implemented checks and their data dependencies are:

- **Input integrity:** checks checkpoint hash, strict EMA loading, configurations
  and required files. Uses the checkpoint, dataset manifest, rasters, profile
  store, mask and climatology.
- **Matching and withholding:** matches observation time/location/depth, removes
  evaluation cells from sparse conditioning, and requires common valid support.
  Uses EN4 or XBT plus packaged GLORYS, climatology and remaining input
  profiles.
- **Calibration separation:** reserves approximately 25% of whole clusters for
  baseline fitting and thresholds. Uses each cohort separately; evaluation
  clusters remain separate.
- **Reference precision and timing:** tests sensitivity to quantization and
  12-hour/24-hour matching windows. Uses observations, packaged GLORYS and
  predictions. **Native floating-point GLORYS comparison is unavailable.**
- **AI comparator compatibility:** checks identities and provenance of supplied
  learned-baseline predictions. The defaults now configure IDW and pinned LSTM,
  CNN, U-Net 3D and U-Net 2D checkpoints. Their full real-data predictions have
  not yet been captured; earlier saved results do not gain them automatically.
- **Calibrated baselines:** compares raw GLORYS, bias-corrected GLORYS,
  probabilistic GLORYS and probabilistic climatology. Uses calibration
  observations to fit adjustments and evaluation observations to score them.
- **Regional skill:** compares MAE, RMSE and CRPS by geography, depth and season,
  with cluster-bootstrap intervals and multiple-comparison correction. Uses
  matched observations and model/reference predictions.
- **Conditioning value:** compares full context with **0/1/2/4/8 retained profile
  locations**, including local optimal-interpolation baselines. Uses the same
  withheld targets and remaining sparse observations.
- **Physical structure:** checks thermocline, isotherm and thermal-layer depths
  and layer-mean temperatures. Uses sufficiently complete observed and predicted
  profiles. **Joint temperature/salinity diagnostics are unavailable.**
- **Ensemble dependence:** compares coherent members with shuffled depth/spatial
  alignment. Uses complete matched profiles and ensemble members.
- **Extremes:** checks warm/cold anomaly probabilities, Brier scores, reliability
  and tail-focused CRPS. Uses calibration-frozen thresholds, climatology and
  evaluation observations.
- **Selective reliability:** tests whether spread, context distance or density
  identify more reliable predictions. Uses ensemble uncertainty, conditioning
  metadata and withheld observations.
- **Generation count and cost:** the current run uses **one generation** and
  reports point-forecast scores and measured runtime. Ensemble convergence,
  interval coverage/width and DepthDif calibration require a future ensemble run.

**Current interpretation:** exploratory validation, not independent confirmation.
XBT's two cruises are below the configured 20-cluster requirement for regional
statistical conclusions.

## Temperature campaign preparation

`validation_temperature.yaml` runs EN4 and Italian XBT with the pinned
run `1419568d` step-2,000 EMA checkpoint, one generation, 1,000-step DDPM and all
five conditioning budgets. It retains the frozen 256-patch EN4 cap and discovery
labels. The full inference campaign has not been launched; both local RTX 3090
GPUs were occupied by `sr-dit` jobs during preparation.

Fresh evidence is under `results/preflight_temperature_20261105/`:

- Both EN4 and XBT cohort preflights passed, including the pinned checkpoint hash.
- Strict CPU EMA loading passed; both CUDA devices are visible. Validation
  shuffle remains enabled.
- Small windows decoded from all 53 temperature and 53 SST rasters for 2016;
  the land mask and nine known EN4 profile rows read successfully. These are
  sampled reads, not a complete chunk-integrity scan.
- The climatology archive passed its CRC check, matches the depth axis, and
  excludes 2016 from all 1,230 training dates.
- All 79,367 prepared XBT temperature values converted successfully.
- The current EN4/XBT campaign passed **409 repository tests** and **64 validation
  tests**, plus fresh preflights for both cohorts, Black and `git diff --check`.
  Latest logs and preflights are under `results/verification_en4_xbt_20261105/`.
  Repository tests and Black required host execution because sandbox
  multiprocessing stalled.

The reference remains packaged GLORYS; native floating-point GLORYS, trained AI
comparators and independent confirmation evidence are still absent.
These limitations affect scientific conclusions, not execution readiness.

## What this validation measures

The temperature campaign asks whether the **frozen DepthDif temperature
ensemble predicts withheld profile measurements better than the matched GLORYS
reference**, and whether its uncertainty and vertical structure are useful.
Withheld measurements are the scoring observations. GLORYS is a comparator;
agreement with its dense raster is not the primary accuracy target here.

The active command is:

```bash
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run validate \
  --config val/validation_temperature.yaml --output val/results/temperature_full_20261105
```

Use a new output directory for each execution. The campaign runs EN4 and Italian
XBT sequentially, using [config.yaml](config.yaml) and
[config_xbt.yaml](config_xbt.yaml).
Each cohort gets its own predictions, calibration partition and analysis report;
the two observation sources are not pooled into one calibration population.
The model is temperature-only, so this command does not validate salinity.

## Exactly what data enter the run

### Shared model inputs and references

The packaged dataset is rooted at `/work/data/OceanVariableReconstruction`.
Its `manifest.yaml` describes exports over January 2000–July 2024; **this campaign
selects only 2016**. The shared grid is EPSG:4326, 3,600 longitude cells by 1,800
latitude cells at 0.1°. The model processes 128 × 128-cell patches on 50
nonuniform model depths, approximately 0.494–5,727.917 m. This full model depth
range does not imply that an observation cohort supports every depth.

| Data | Local source | Role and representation |
| --- | --- | --- |
| Sparse profile temperature | `argo/argo_profiles_on_grid.zarr` below the dataset root | Remaining, non-held-out profiles condition the reconstruction. The store contains 9,451,644 profiles across its complete date range, not 9.45 million validation profiles. EN4 scoring observations also come from this store. |
| Surface temperature | `rasters/ostia/analysed_sst/` | OSTIA SST conditions the model. There are 53 target-date rasters in 2016. The manifest records seven-day surface aggregation onto the target dates. SST is not a withheld subsurface observation. |
| GLORYS potential temperature | `rasters/glorys/thetao/` | The deterministic reference, with 53 dated 50-band rasters in 2016. These are daily-mean fields sampled at seven-day intervals, rather than seven-day means. They have been exported onto the 0.1° grid. |
| Land and wet support | `masks/world_land_mask_glorys_0p1.tif` and valid raster cells | Define patch coverage and valid scoring support. The model also receives the configured land, wet-domain and observation masks. |
| Frozen monthly climatology | `outputs/checkpoint_selection_20261005/baselines/temperature_monthly_stride4.npz` | A monthly, spatial temperature background used by the residual model and as a reference. Its 1,230 contributing dates exclude 2016; it is not fitted to the validation observations. |

The model additionally uses location and date conditioning. Its surface source
is SST only: the presence of sea-surface salinity or sea-level rasters on disk
does not make them inputs to this checkpoint.

Packaged GLORYS temperature and compact EN4 temperature use an 8-bit stretch:
valid codes 0–254 cover 270.15–308.15 K; code 255 is missing. The temperature
increment is approximately **0.149606°C**, with a rounding bound of about
**±0.074803°C** before considering other errors. EN4 values were already
interpolated to the model levels during export. They are not native, unquantized
EN4 casts. The store declares potential temperature from `POTM_CORRECTED`,
referenced to 0 dbar. Scores are computed in °C after decoding.

### EN4 archive-screened observations

[config.yaml](config.yaml) combines the compact profile store with
`instructions/en4_no_spatiotemporal_candidate_profiles.parquet`. It selects 2016
profiles whose audit status is `no_spatiotemporal_candidate`, with supported
temperature depths and the configured profile/level QC checks accepting flags
1 and 2. The audit means no candidate was found in the supplied historical
archive within **24 hours / 25 km**; it does not verify that GLORYS never
assimilated the observation.

The selector shuffles dates and candidate patches with the fixed seed, visiting
dates in rounds. It selects **up to 256 patches**. Within a selected patch it
aims to withhold 20% of eligible candidate grid locations, while retaining at
least eight input profile locations in every selected patch after all global
holdouts. Several source profiles can occupy one held-out grid location.
The observation is its stored potential-temperature value at each valid model
depth. Final selected/scored counts are produced by the actual run; 256 is a
patch cap, not a profile or depth-measurement count.

EN4 regions use 20° geographic tiles. Its bootstrap clusters are **5° tile ×
target month**, because full float/platform identities are unavailable. Those
blocks are a dependence approximation, not verified independent floats.

### Italian PNRA XBT observations

The original text files are inside `evaluations/focus_evaluation.tar.gz`, under
`inputs/italian_xbt/`. Identities were matched to the screened table
`evaluations/outputs/italian_en4_no_match_profiles.parquet`. Import uses corrected
`Depth 3`, corrected `Temperature 2`, and source quality flag `QF=1`.

The 2016 import contains **108,428 temperature measurements from 102 profiles**
on two cruises. Reference-salinity preparation retained **79,367 measurements**
in `results/xbt_reference/profiles.csv`: 67,803 from 88 `PNRAXXXI` profiles and
11,564 from 14 `PNRAXXXII` profiles. The retained depths are approximately
0.839–541.088 m, spanning 47.49–72.41°S and 171.57–175.00°E. Another 29,061
measurements lacked acceptable wet salinity support or a permitted depth bracket.
These are prepared-input counts; final matched scoring counts can be smaller.

XBT measures **in-situ temperature, without observed salinity**. The adapter
converts it to potential temperature using TEOS-10/GSW, observation position and
depth, and GLORYS reference salinity documented in
`results/xbt_reference/manifest.json`. That salinity is a conversion assumption
shared by all temperature comparisons, not an independent salinity observation.
Bootstrap clusters are entire cruises: there are only **two** before matching
and calibration partitioning. Archive screening does not establish independence
from training or GLORYS assimilation.

## Steps before the ten analysis stages

1. **Check and freeze inputs.** Verify the checkpoint SHA-256, temperature output
   capability, required rasters, compact store, mask, climatology and cohort
   files. Resolve evaluation configs without changing training configs. Load
   step-2,000 run `1419568d` EMA weights strictly; do not retrain or select a new
   checkpoint using these results.
2. **Build the matched observation cohort.** EN4 uses its stored model-level
   observations and selected target dates. External XBT profiles match the
   nearest available 2016 target date within ±3.5 days and the containing raster
   cell. Each external profile is assigned to the first covering eligible
   validation patch. Preserve the observation depth; interpolate model, GLORYS
   and climatology between model levels with a maximum 100 m vertical bracket,
   without extrapolation. Profiles without a supported match are recorded as
   exclusions; individual unsupported depths within a matched profile are omitted
   during assignment and are not all listed separately.
3. **Remove evaluation observations from conditioning.** Mask every depth of
   both sparse fields, where present, at held-out cells in every overlapping
   patch. External cohorts also mask a one-cell buffer. The current compact store cannot verify
   complete platform/cruise exclusion across other positions or times, so this
   remains coordinate-cell exclusion rather than proven whole-platform holdout.
4. **Establish common scoring support.** Require finite observations, wet GLORYS
   support, finite climatology and enough remaining context: eight locations for
   EN4, one for XBT. Keep signed observation-to-target time differences.
   These exclusions are determined before predictions. A nonfinite prediction
   on an accepted point fails the run instead of dropping that difficult point.
5. **Generate and save ensembles.** Use batches of two patches, 100 independent
   DDPM members with 1,000 denoising steps, and seed `20261103`. Repeat inference
   with nested retained-context budgets 0, 1, 2, 4 and 8 per patch, using the same
   member seeds to isolate the effect of conditioning. A budget counts grid
   locations with profile support, not necessarily distinct instruments; fewer
   locations can be retained when fewer are available. Compute local optimal
   interpolation (OI) references using the same available locations. Save
   point-aligned physical predictions, exclusions, timings and hashes.
6. **Separate calibration from evaluation.** Within each discovery cohort,
   deterministically reserve approximately 25% of whole clusters for calibration
   using a hash and the seed; rounding and small cluster counts affect the exact
   fraction. Fit baseline adjustments and event thresholds on that partition
   only. Score the paper stages on the remaining clusters. The model weights
   remain frozen. `paper/partition.csv` records the split. The older `report/`
   scorecard separately describes all saved scoring points, including calibration
   points; it must not be mistaken for the held-out paper evaluation.

## The ten analysis stages, in execution order

The stage order comes from [paper.py](paper.py); frozen options are in
[paper_defaults.yaml](paper_defaults.yaml). After inference these stages use the
saved bundle, so they can be repeated without generating the ensemble again.

1. **Reference audit — `techniques/reference.py`.** Record packaged reference
   precision and observation-to-product time offsets. Recompute matched errors
   for observations within 12 hours, within 24 hours and the full matching
   window. Native floating-point GLORYS sampling would compare native and
   packaged references on identical rows, but no native paths are configured:
   this portion is unavailable and the stage is partial. The filename dates do
   not verify native source time bounds.
2. **AI comparators — `techniques/comparators.py`.** Check whether learned
   baseline outputs or supplied external predictions have compatible point
   identities, variable definitions, ensemble representation and provenance.
   The shared baseline registry now supplies all five built-in methods. Earlier
   bundles without captured predictions report those methods as blocked and
   require a new evaluation. The optional Cavaiola diagnostic runs after
   calibrated references are fitted. No stage trains a baseline or treats missing
   comparisons as wins.
3. **Calibrated references — `techniques/baselines.py`.** Compare DepthDif with
   raw GLORYS, GLORYS with calibration-fitted bias correction, probabilistic
   GLORYS and probabilistic climatology. Bias groups use variable, rounded 50 m
   depth bins and target month, with shrinkage count 5. Probabilistic references
   draw complete residual profiles from calibration data, with 50 members by
   default. Report profile-balanced errors, ordinary/fair CRPS and reliability
   diagnostics. Deterministic references alone cannot establish an uncertainty
   advantage over a calibrated probabilistic forecast.
4. **Regional comparisons — `techniques/regional.py`.** Compare paired errors
   overall and by region, depth band, season, source and other available
   provenance strata. Fixed geographic boxes cover the Southern Ocean,
   59–60°N North Atlantic section and Weddell Sea; these are analysis masks,
   not newly collected section data. Depth bands are 0–50,
   50–200, 200–700, 700–2,000 and >2,000 m. Average depth losses within profiles
   before comparing methods; use 2,000 whole-cluster bootstrap replicates for
   95% intervals and Holm adjustment across searched comparisons. A negative
   DepthDif-minus-reference difference favours DepthDif. Inference requires at
   least 30 profiles and 20 clusters per comparison. Fewer units give descriptive
   results with insufficient support, regardless of the number of depth rows.
5. **Conditioning — `techniques/conditioning.py`.** Compare full context and
   the 0/1/2/4/8 retained-location ensembles on the same targets, reporting
   profile-balanced errors and CRPS. Compare with GLORYS-plus-OI and
   climatology-plus-OI at the same budgets. The frozen OI settings use 150 km
   horizontal and 50 m vertical scales, temperature noise 0.1°C and at most
   32 profiles. Zero sparse context still retains SST, climatology, masks and
   location/date conditioning; it is not an unconditional model.
6. **Physical structure — `techniques/physical.py`.** Derive diagnostics from
   each ensemble member before scoring: thermocline depth/maximum gradient in
   10–300 m, a thermal layer defined by a 0.2°C departure from 10 m, 5/10/15/20°C
   isotherm depths and mean temperature over 10–200 and 10–700 m. Compare those
   derived quantities with observation and reference profiles using diagnostic
   errors and CRPS. Missing crossings or unsupported layers are marked
   incomplete, not extrapolated. In particular, the prepared XBT data
   cannot support the full 10–700 m layer. Density, barrier-layer and other joint
   T/S diagnostics are unavailable for this temperature-only campaign.
7. **Ensemble dependence — `techniques/dependence.py`.** Test whether coherent
   ensemble members across depths/profiles add value beyond correct scalar
   distributions. Compare intact members with five permutations that break
   vertical or spatial member alignment while preserving scalar marginals.
   Report multivariate energy/variogram scores and depth-weighted profile-mean
   diagnostics on supported cases. Joint T/S dependence requires a joint model
   and paired observations and is unavailable here.
8. **Extremes — `techniques/extremes.py`.** Define warm/cold anomalies relative
   to frozen climatology. Fit profile-balanced upper/lower thresholds at the
   configured 90th and 95th quantiles using calibration observations, separately
   by variable and depth band. On evaluation data, compare member-derived event
   probabilities with observed events using Brier scores, reliability and
   tail-focused CRPS, including available calibrated reference distributions.
   These are temperature-anomaly events at sampled profiles, not a duration-based
   marine-heatwave detection product. Missing calibration support blocks the
   affected analysis rather than fitting thresholds on evaluation labels.
9. **Selective reliability — `techniques/selective.py`.** Rank profiles by
   available forecast covariates: ensemble spread, calibrated GLORYS spread,
   nearest-context distance and context density. For retained fractions from
   10% to 100%, compare profile MAE for DepthDif and GLORYS on the same selected
   profiles. Observed errors evaluate the selection; they never determine its
   ordering. Missing covariates are reported as unavailable. This tests whether
   the model can identify cases on which it should be trusted more.
10. **Ensemble size and cost — `techniques/efficiency.py`.** Reuse prefixes of
    2, 5, 10, 20, 50 and 100 members to measure fair CRPS, 90% interval coverage
    and width, and member diversity. Relate these to measured member timings,
    profile throughput and available resource metadata. Prefixes reuse one
    ensemble rather than independently rerunning every size. GLORYS production
    cost and unmeasured energy savings are not inferred from local lookup cost.

CRPS evaluates the full predictive distribution against an observation; lower
is better. Fair CRPS corrects the finite sampled-ensemble contribution. Coverage
must be interpreted together with interval width: broad intervals can cover
observations without providing useful precision. Individual stages document
their aggregation; not every descriptive plot or legacy score is profile-balanced.

## Outputs and interpretation

The campaign's `index.html` links to each cohort. Each cohort saves:

| Output | What it establishes |
| --- | --- |
| `preflight.json`, `config.yaml`, `effective/` | Required-input availability and resolved run configuration. |
| `cohort.csv`, `holdouts.json`, EN4 `en4_selection.json` | Selected observation identities, patch assignments and conditioning exclusions. |
| `bundle/points.csv`, `bundle/members.npz`, `bundle/metadata.json`, `bundle/exclusions.csv` | Matched observations/references, physical ensemble arrays, input/config hashes, timings and rejected cases. The array named `residual` stores physical temperature predictions, not just normalized residual increments. |
| `paper/partition.csv`, `paper/split.json` | Exact calibration/evaluation membership and split provenance. |
| `paper/stages.csv`, `paper/index.html`, stage subdirectories | Stage completion/partial/blocked status, tables and figures. Completing a stage is not evidence that DepthDif won it. |
| `report/` and EN4 coverage tables | Legacy all-point diagnostics and eligible-versus-selected-versus-scored coverage. |

The counts documented above describe available/prepared data, not final evaluated
sample sizes. Actual counts must be read after time/depth/wet-support matching,
conditioning checks and calibration partitioning. XBT starts with two cruise
clusters, below the regional 20-cluster requirement even before the split.
EN4 blocks do not establish independent float sampling.

Both cohorts remain **discovery**: 2016 contributed to checkpoint selection,
training exclusion has not been independently verified for every observation,
and non-assimilation/whole-platform conditioning exclusion are unverified.
Therefore the current run can report exploratory skill against packaged GLORYS
and the calibrated references. It cannot establish untouched confirmation,
native-product superiority, global coverage or salinity skill. The readiness
checks at the top of this file establish that the software and local inputs are
available; the historical short-chain smoke runs establish execution only.

# Verification on 3 November 2026

## Sequential paper validation implementation

The complete campaign is now `val/validation.yaml`, executed with:

```bash
PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run validate --config val/validation.yaml --output val/results/paper_all
```

Every available cohort runs ten separate stages in sequence. Campaign and
standalone cohort defaults now use one generation per batch and conditioning
budget; ensemble uncertainty diagnostics are marked unavailable. The initial
implementation checks passed **409 repository tests**
(21.391 s) and **65 validation tests** (34.628 s). Black formatted only `val/`
files; its final check left all 166 files unchanged. `git diff --check` passed.
No production source or original test changed.

A real QC-checkpoint smoke including every retained-profile budget and local OI
passed with 237 values, seven profiles and three bootstrap blocks. The persisted
bundle contains the full ensemble, five ablation ensembles and twelve OI
prediction arrays. All 237 points have finite nearest-context distances; measured
member timings are available. It uses two members and two DDIM steps and is a
functional test, never an accuracy or publication result. Read the
[final saved-bundle report](results/paper_reviewed_final/paper/index.html), or inspect
`results/paper_real_smoke_03/` for original inference provenance. The first
attempt stalled on restricted-sandbox multiprocessing; the second found a sparse
DataFrame-index bug in OI, which is fixed and regression-tested.

The native-reference tests include successful CF time bounds, a singleton depth,
dry bilinear corners, model ensemble alignment and explicit unavailable data.

Native floating-point GLORYS and trained external AI comparators remain explicit
prerequisites. Salinity/joint weights and independence evidence are still missing.
Full campaign accuracy runs have not been launched. The earlier checks below
describe the original validation harness and recovered XBT observations.

## Earlier harness checks

The existing application source, tests, configs, and validation-loader shuffle
setting are unchanged. `git status --short` lists only the new `val/` directory;
`git diff --check` passes.

- Required formatter: `/work/envs/depth/bin/python -m black .`, exit 0,
  145 files unchanged on the final pass. Black prints an existing Python
  target-version warning. The sandbox formatter stalled; host execution completed.
- Full repository suite: **409 tests passed**, final run 34.909 seconds.
- Additional validation suite: **28 tests passed**, final run 5.780 seconds.
- Logs: `results/verification/{repository_tests,validation_tests,black_host}.log`.
  The sandboxed repository suite stalled during dataset setup; the host run passed.
- Both-variable synthetic report and saved-bundle rescoring passed. Open
  [the demonstration](results/demo/report/report.html); its scores are synthetic.
- Real temperature checkpoint/data smoke passed: four candidate patches,
  two members, two DDIM steps, GPU 1; **237 scored values, seven profiles,
  three bootstrap blocks**. This short chain is a functional check,
  not an accuracy experiment. Evidence is under `results/smoke_real_attempt04/`.
- Earlier smoke attempts caught an incorrect sampler-override path and mixed
  fractional-second timestamp parsing; both are fixed and regression-tested.
  One selected patch had no wet GLORYS reference support and was excluded.
  Failed-attempt logs remain under `results/verification/`.

The configured 2016 temperature, SST, profile store, mask, checkpoint, and
climatology paths are present. A raster read and the checkpoint smoke succeeded.
This does not verify every data chunk or constitute a complete regional evaluation.

## Italian data recovered

Raw PNRA files exist inside `evaluations/focus_evaluation.tar.gz`; the extracted
Italian directory is empty. The companion screened identity table has 2,137
profiles. The 2016 import matched all 102 requested profiles and retained 108,428
corrected `QF=1` measurements from two cruises. Evidence:
`results/xbt_import/{profiles.csv,manifest.json}`.

Reference-salinity preparation retained 79,367 rows across all 102 profiles;
29,061 rows lacked a permitted vertical bracket or wet salinity support.
The default 100 m model-level-gap cap causes most exclusions. The preparation
manifest records counts and every used raster hash. Evidence:
`results/xbt_reference/{profiles.csv,manifest.json}`.

GSW 3.6.19 was installed with `--no-deps --target val/vendor`, leaving the shared
Python environment unchanged. Using `PYTHONPATH=val/vendor:src`, real TEOS-10
conversion passed for all 79,367 prepared rows; the maximum correction from
in-situ to potential temperature was approximately 0.05268°C. Reference salinity
comes from GLORYS and remains an observation-operator assumption requiring a
sensitivity analysis for small gains.

## Scientific results still pending

The complete 20-member regional benchmark has not been run. The selected QC
checkpoint is temperature-only; salinity and joint capability is implemented and
tested using distinct physical outputs, but requires trained compatible weights.
The defaults now pin all five built-in baselines in `val/baselines.yaml`.
Strict loading passed for all eight archived neural checkpoints (four
architectures, temperature and salinity); the full baseline campaign remains
pending. Tests verify method propagation through the scientific scorecards and
conditioning budgets, including explicit unsupported-budget reports.
Italian screening is not verified non-assimilation evidence. The compact store
lacks platform/cruise identities for complete conditioning exclusion, and 2016
was used in prior model selection. Confirmation gates therefore remain false.
No accuracy, uncertainty, global-superiority, or efficiency benefit is claimed
from these software checks.
