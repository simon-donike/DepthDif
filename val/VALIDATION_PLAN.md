# Validation plan for the frozen QC-finetuned residual model

This is the analysis contract for the new validation branch. It is deliberately
data-path agnostic: the data can be copied back later and the paths then filled
in in the `val` configuration. The plan describes two target variables from the
start, temperature (T) and salinity (S), but it must refuse to claim S skill
until a checkpoint with a salinity output is supplied. The locally documented
QC-finetuning residual checkpoint is temperature-only (`include_salinity: false`);
that is a capability gate, not evidence about salinity.

Implementation: `val/validation.yaml` runs the complete campaign with one
`python -m val.run validate` command. `paper_defaults.yaml` freezes the calibration
partition, conditioning budgets, fixed OI parameters, named stress-test regions
and extreme thresholds. Ten modules under `val/techniques/` execute sequentially,
with CSV/LaTeX tables, PDF/SVG/PNG figures and an HTML index. See `README.md` for
the command, prerequisite gates and saved-bundle rescoring. Crossing-depth
physical diagnostics explicitly retain their conditional completeness limitation;
the primary profile comparisons remain on prediction-independent common support.

The primary scientific question is: **does the method improve on GLORYS12V1 at
withheld observations, and in which regions?** The plan has a cheap global scan
to discover regional heterogeneity, followed by a frozen confirmation set. The
scan is useful for plots and hypotheses; it cannot be used to choose a favourable
region and then report that region as independent confirmation.

## What the validation must establish

Ruben's three possible paper arguments become three predeclared result tracks:

1. **Held-out observations:** DepthDif has lower paired T/S error than GLORYS12V1
   in the all-region result or a region selected before opening confirmation
   labels. Italian PNRA Southern Ocean XBT profiles are a named temperature-only
   candidate confirmation cohort, pending independence and split checks, because
   XBTs do not measure salinity.
2. **AI baselines:** the same held-out cases compare the frozen model with simple
   observation-based baselines and published reconstruction settings whose inputs
   and target are actually matched. A forecasting model is not a fair baseline
   for this reconstruction task.
3. **Probabilistic benefit:** the DepthDif ensemble has lower CRPS and better
   coverage/sharpness than deterministic GLORYS and the deterministic baselines.
   This requires ensemble samples at evaluation locations; it does not require
   sampling the whole world. Probabilistic superiority is a separate claim from
   lower ensemble-mean RMSE.

An efficiency result supports the probabilistic argument. Record wall time,
GPU-hours, peak memory, number of generated profiles, ensemble size, and energy
measurement method for each method under the same hardware and batch contract.
Report cost per profile and cost per 50-member profile set with uncertainty from
at least five repeated timed batches. Do not compare training cost with inference
cost, or compare a full GLORYS download with a local tensor lookup without saying
so. A method's useful efficiency claim is “at the same evaluation workload and
hardware, N samples cost …”; it is not a claim that its scientific error is lower.

## Frozen checkpoints and splits

The default checkpoint is the QC fine-tuning residual run selected by the user.
Before inference, record its absolute path, SHA-256, source run, raw/EMA choice,
resolved configuration, climatology hash, model output variables, and inference
seed. Use the same checkpoint and sampler for the exploratory scan and the
confirmation run. Do not choose a later checkpoint from confirmation metrics.

The repository's 2016 validation loader is intentionally shuffled and remains so;
its metrics are a development monitor only. The old 2016 validation metrics and
the previous EN4 evaluation are useful diagnostics, but checkpoint selection
against them means they are not an untouched final test.

Use three labels in every output:

| Label | Use |
| --- | --- |
| `development_2016` | Existing loader/validation metrics, checkpoint sanity, and debugging |
| `exploratory_scan` | Multi-region map and strata search; may suggest hypotheses, never select confirmation after labels are opened |
| `confirmatory` | Frozen held-out profile/campaign units, opened once after the checkpoint and regions are frozen |

The preferred confirmation split is a complete held-out calendar year (including
surface windows touching that year) or a prospective period. For campaigns, hold
out the whole cruise, deployment, or platform-year. Remove the held-out unit from
every conditioning window, including overlapping patches. If a profile or its
duplicate was used for checkpoint training, label it `assimilated_or_training`
and exclude it from the primary GLORYS comparison. If only a same-date diagnostic
is possible, say so in the report and do not call it independent validation.

## Reference data and independence audit

GLORYS12V1 is a 1/12 degree, 50-level global reanalysis. Its product page says
that altimetry, satellite SST, sea ice concentration, and in situ vertical T/S
profiles are jointly assimilated. Therefore EN4, Argo, and other archive names
are not sufficient evidence of independence. Store for each profile its source,
original identifier, platform/cruise, date, QC, duplicate matches, and an
`independence`: `verified_not_assimilated`, `archive_screened`, `unknown`, or
`assimilated`. Primary GLORYS claims use `verified_not_assimilated` only; report
the other classes as sensitivity strata.

Primary references:

- [Copernicus GLORYS12V1 product page](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
  (variables, 1/12 degree grid, 50 levels, assimilation description, and links to
  the User Manual and Quality Information Document).
- [Lellouche et al. 2021, GLORYS12 evaluation](https://doi.org/10.3389/feart.2021.698876)
  (published regional and depth-dependent product limitations to use as diagnostic
  context, not as new test labels).
- [Verezemskaya et al. 2021, 59.5°N section assessment](https://doi.org/10.1029/2020JC016317)
  (a useful independent section design; its velocity findings cannot be claimed
  from a T/S-only model).

For every matchup, sample GLORYS and DepthDif at the same native observation
longitude, latitude, time, and depth. Retain observation time minus target-week
time. Use potential temperature versus in-situ temperature explicitly; GLORYS
`thetao` is potential temperature. Convert only in a documented TEOS-10 pipeline
when derived density or heat content is computed. Never call an XBT temperature
observation a salinity observation.

## Italian PNRA XBT confirmation

Search the local workspace first for the archive and preserve the original files
and metadata. If absent, configure (but do not download automatically) the
[Zenodo record 14848849](https://doi.org/10.5281/zenodo.14848849) associated with
[Aulicino et al. 2025](https://doi.org/10.5194/essd-17-2625-2025). The paper
describes Sippican T7 casts, roughly 20 km sampling, approximately 65 cm vertical
resolution, nominal depth to 760 m, and quality control including malfunction,
spike, adjacent-profile, regional-feature, altimetry, and operator checks. Retain
the source QC and discard flags; do not silently repair values.

The XBT result is a **T-only** campaign result. It is particularly useful for
Southern Ocean sectors and frontal structure, but it cannot support a salinity,
density, or joint T/S headline. Because the campaign spans repeated cruises,
cluster all uncertainty by cruise occupation and report cruise-level maps before
pooling. Check whether each occupation and target week was available to GLORYS
assimilation; if unknown, keep it out of the primary independent claim.

## Baselines and information settings

### What GLORYS uncertainty is available

The [GLORYS12V1 catalogue](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
lists deterministic physical fields rather than a delivered member ensemble or
per-profile predictive temperature/salinity intervals. Its assimilation error
covariances should not be treated as downloaded calibrated prediction intervals.
The [separate global ensemble reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_ENS_001_031/description)
provides a multi-reanalysis uncertainty context at 0.25° and 75 levels; it combines
GLORYS2V4, ORAS5, and C-GLORSv7. It is not a GLORYS12 posterior ensemble. A comparison
would need matched monthly means, grids, depths, and observation support, and
cannot substitute for the primary weekly GLORYS12 comparison.

### Current implementation and subsequent experiments

The implemented runner produces point/profile-balanced errors, CRPS, coverage,
width, cluster-bootstrap comparisons, and direct HTML/CSV reports from matched
point arrays (`points.csv` plus `members.npz`). It uses containing raster cells
for horizontal matching and model-to-observation depth interpolation, rather than
bilinear horizontal matching. The default is a 20-member development evaluation,
2,000 bootstrap replicates, and 30 profiles/20 clusters. Larger ensembles and
replicate counts below are final-report targets to freeze before confirmation.
Training-only probabilistic climatology, energy scores, area-weighted dense field
maps, sections, and repeat-timing experiments below remain separate experiments;
they are not automatically completed by this first validation command.

Every baseline must receive the same allowed information as DepthDif and be
evaluated at the same withheld observations.

Required point baselines are:

- GLORYS12V1 sampled by the common observation operator (the principal reference).
- Training-only monthly/depth climatology, including the residual model's frozen
  climatology. The climatology must be fitted on training data only.
- Local sparse-observation interpolation or nearest/linear profile baseline, with
  a declared radius and no access to the withheld profile.
- A simple training-only linear or ridge regression using the same surface and
  coordinate inputs, if those inputs are available to the residual model.

Use an existing published reconstruction method only when its input data,
prediction task, target depths, and evaluation split can be matched. Relevant
primary examples include [Buongiorno Nardelli 2020](https://doi.org/10.3390/rs12193151),
which retrieves hydrographic profiles from combined satellite and in-situ
measurements, and [Meng et al. 2021](https://doi.org/10.1029/2021JC017605),
which reconstructs 3-D T/S from satellite observations. These papers motivate
LSTM/CNN or EOF-style observation-to-profile baselines; their reported numbers
must not be copied as comparisons because domains, inputs, and splits differ.
Do not use a forecast-only model as an AI baseline for this reconstruction test.

The most direct published probabilistic comparator is Buongiorno Nardelli's
stacked LSTM with Monte Carlo dropout; [the author's 3Drec repository](https://github.com/bbuong/3Drec)
provides both LSTM and feed-forward implementations. It already estimates
hydrographic uncertainty, so the existence of a DepthDif ensemble is not by itself
a new advantage. Demonstrate better proper scores, calibration, spatial/profile
dependence, or matched inference cost. Its satellite predictors include more
modalities than the frozen SST-only QC model; use a declared matched-input variant
or report this information difference explicitly. The repository's existing
pointwise LSTM is a toy baseline, not an automatic reproduction of 3Drec.

For probabilistic scoring, a deterministic method is a point-mass distribution:
its CRPS is absolute error. Add a training-only empirical residual distribution or
calibrated baseline only when its distribution can be constructed without test
labels. A normal distribution fitted from training residuals is acceptable as a
diagnostic, not as a claim that the ocean errors are normal.

## Matchup table and common operator

Write one immutable long-form table (plus a Zarr/NetCDF ensemble array) with at
least: `case_id`, `profile_id`, `source_id`, platform/cruise, observation and
target times, lon/lat, observed and matched depths, T/S observations, GLORYS T/S,
ensemble member references, conditioning count and nearest distance, source/QC
flags, independence class, region, season, and model version hashes.

Use bilinear horizontal interpolation only when all required source cells are wet;
otherwise use the nearest wet cell and flag it. Vertically interpolate within the
valid water column and never extrapolate below the local seafloor. Keep station
metrics alongside any objectively mapped section figures. Use native observation
values for point scores. The operator and all flags must be identical for GLORYS,
DepthDif, and baselines.

## Metrics and uncertainty

For T and S separately, report bias, MAE, RMSE, median absolute error, 90th/95th
absolute-error quantiles, and ensemble-mean versus median error. Report depth bands
`0–10`, `10–50`, `50–100`, `100–200`, `200–500`, `500–700`, and `700–2000 m`, with
layer-thickness weighting for integrated quantities. Include profile shape RMSE
after subtracting each profile mean and vertical-gradient error. Do not pool T and
S into an undocumented scalar.

The current validation runs one generation per batch and conditioning budget.
Point/profile scores remain available; ensemble calibration and dependence
claims are deferred. The following larger-ensemble audit is a future protocol,
not the current inference setting.

For 50 fixed-seed members (and 100 members on a fixed 10% audit subset), report
CRPS, central 50/80/90/95% coverage and width, rank/PIT histograms, spread-skill,
and energy score for complete standardized T/S profiles when both variables exist.
Use the empirical ensemble CRPS for descriptive scoring and the finite-ensemble
fair correction as a sensitivity. [Ferro 2014](https://doi.org/10.1002/qj.2270)
shows why finite ensembles make the usual CRPS unfair when members represent a
sample from the predictive distribution. [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)
establishes proper scoring rules; use lower-is-better conventions consistently.

Pair every DepthDif result with GLORYS on the same case/profile. Bootstrap the
paired difference by independent `case_id`, not depth pixels: platform-year for
floats and complete cruise for sections/XBT.
Use 10,000 reproducible replicates for the final report, report the mean and
median difference, 95% interval, and win fraction, and show the number of clusters.
Use a minimum gate of 30 independent cases for a general regional comparison and
five complete XBT cruises for a descriptive campaign result. Below a gate, show
the estimate and interval as underpowered without a comparative claim.

## Region design and multiple testing

The exploratory scan covers global wet cells and fixed broad regions: North
Atlantic, Southern Ocean (with Atlantic/Indian/Pacific sectors), Beaufort, and
optional Weddell. Use wet-cell area weights and publish the
region registry, masks, depth bands, seasonal bins, and counts with the report.
The Italian XBT line is a separately identified Southern Ocean campaign, not a
global-region replacement.

The confirmatory region list and primary endpoint must be frozen before reading
confirmatory labels. The primary endpoint is paired MAE difference in `50–100 m`
for T and S where available, with secondary endpoints of `700–2000 m` error,
interval coverage, and the named campaign T result. Control multiplicity with
Holm correction across the predeclared endpoints within each experiment. Treat
all other region/depth/season cells as descriptive. Do not select a region because
its scan interval favours DepthDif and then call it independent confirmation.

## Outputs that are easy to inspect

The runner should produce a self-contained `report.html` and machine-readable
files under a run directory:

- `run_manifest.json`: checkpoint/config/data/operator hashes, seeds, hardware,
  split labels, and capability gates;
- `profile_matchups.parquet` and `ensemble.zarr` (or equivalent);
- `metrics_point.csv`, `metrics_probabilistic.csv`, `bootstrap_deltas.csv`, and
  `counts.csv`;
- `regional_scan.html` and `confirmatory.html`, each with a one-page summary,
  region/depth table, paired-delta plot, coverage plot, and map/section plot;
- `report.html` with a top-line decision table: `better`, `no evidence`,
  `worse`, or `underpowered`, separately for T, S, point skill, and uncertainty;
- `timings.csv` and `efficiency.html` with matched hardware/workload timings.

The top-line table must show raw GLORYS and DepthDif values, paired delta with
95% interval, corrected p-value where applicable, cluster count, and whether the
result is exploratory or confirmatory. It must state “S not evaluated” when the
checkpoint lacks salinity output. No result should require opening a notebook to
understand whether the method helped.

## Execution order and stop gates

Before publication-grade comparison, apply the native-GLORYS, daily-target timing,
precision, and estimand checks in [Scientific Opportunities](SCIENTIFIC_OPPORTUNITIES.md).
The repository's GLORYS targets are daily means sampled every seven days, not
weekly averages. The initial packaged-data scores are development diagnostics.

1. Discover and hash local data; validate profile schema, units, QC, duplicate
   IDs, and temperature type. Stop if no independent cases remain.
2. Validate checkpoint capabilities for T and S, run the common operator on a
   tiny non-test fixture, and verify deterministic seeds and ensemble count.
3. Run development 2016 checks, preserving the shuffled loader, then a broad
   exploratory scan over all available regions.
4. Freeze the confirmation registry, endpoint list, and manifest. Generate the
   held-out EN4-style evaluation and the Italian XBT T-only evaluation with
   complete profile/cruise removal.
5. Run paired metrics, cluster bootstrap, multiplicity correction, timings, and
   report generation. Archive the manifest before opening confirmation results.

Stop and report a data limitation when archives are missing, a profile's GLORYS
assimilation status is unknown for a claimed independent result, or the residual
checkpoint cannot load the requested variable. Missing data is a pending gate;
it is not a failed model result.

