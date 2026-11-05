# Scientific opportunities beyond profile RMSE

Research assessment: 3 November 2026. These are proposed experiments, not new
performance results. The frozen reference remains QC residual run `1419568d`,
best step 2,000, EMA. Its output is temperature and its dense surface input is
SST; salinity and joint tests require compatible trained weights. The experiments
below now have separate sequential implementations in `val/techniques/`, with
campaign configuration in `val/validation.yaml`. Availability and evidence gates
remain part of every report: implementation does not establish a performance
benefit. No new training or production-code changes were made.

## Recommended scientific claim

The most plausible contribution is an inexpensive conditional reconstruction
of a GLORYS-learned prior that gives useful distributions and responds to sparse
observations. A stronger claim, to earn through testing, is improved independent
hydrography in a small number of predefined regimes. Replacing GLORYS as a
globally superior dynamical ocean model is a much harder and unnecessary claim.

There are two distinct potential gains:

- **Better state estimates:** lower observation error, including a physically
  useful quantity such as thermocline depth or layer temperature.
- **Better information about uncertainty:** calibrated probabilities and useful
  vertical/spatial dependence, even when the mean is only competitive.

Either can be a scientific benefit. Neither follows from realistic-looking
samples, broader intervals, a favorable isolated cruise, or agreement with the
training reanalysis.

The QC configuration disables ambient occlusion, sparse-observation, and increment
losses. Its dense supervisory target is GLORYS. There is therefore no explicit
loss teaching this checkpoint how to remove GLORYS's observational biases.
Any such correction is a hypothesis to test, not an expected property of residual
learning. The climatology-residual parameterization alone does not add independent
information about the true ocean.

## Fix the reference comparison before an expensive final run

### Native product and time semantics

The download helper explicitly selects one daily GLORYS mean every seven days:
[`download_glorys_weekly.sh`](../src/depth_recon/data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh).
The targets are not weekly means. Surface EO is generally a centered seven-day
mean, and profile matching permits a 3.5-day offset. These operators can create
different errors near moving fronts, rapid mixing, and coastal upwelling.

Use the source NetCDF time coordinate and bounds, including the daily-mean
reference time; do not assume midnight from a filename. Publish separate scores
for within 12 hours, within 24 hours, and the full matching window. If daily
observations are averaged, use the same averaging operator on every method.
These are different estimands and should remain labelled separately.

The current comparison reads GLORYS from 0.1-degree uint8 exports. The original
product is 1/12 degree. The exported temperature quantization step is approximately
0.1496 degrees C, with maximum rounding error about 0.0748 degrees C; the salinity
step is about 0.0394 with maximum rounding error about 0.0197. See
[Data Export](../docs/data-export.md). These bounds do not prohibit estimating a
smaller aggregate effect, but they make a native-data sensitivity essential.

Keep two named references:

1. **Packaged GLORYS:** identical grid, precision, and target dates as training;
   useful for diagnosing the learned representation.
2. **Native GLORYS12:** floating-point product, sampled at observation position,
   source time, and depth using a frozen, wet-cell-aware operator; required for a
   claim about improvement over the delivered GLORYS product.

Show whether conclusions survive nearest-cell versus valid bilinear matching,
position uncertainty, temperature conversion, and time-offset sensitivity.
The published GLORYS system assimilates surface and in-situ observations, so its
native analysis contains substantially more information than the small set of
inputs supplied to this network. [Lellouche et al. 2021](https://doi.org/10.3389/feart.2021.698876)
describes that system and reports residual upper-ocean seasonal stratification
errors. It does not establish that our model corrects them.

### Split and independence have different meanings

Separate three questions: was the verification target used to optimize/select
the network; was its profile available in network conditioning; and was it
assimilated into GLORYS? A profile can pass one check and fail another.

2016 is discovery data because checkpoint selection used that year's validation
behavior. Withholding a profile now does not create an untouched final year.
Use new, frozen campaigns or a prospective period for confirmation where the
checkpoint's training coverage permits it. A geographic holdout also requires
excluding dense training target patches in that geography; masking observations
after training is a conditioning experiment, not geographic generalization.

Score documented non-assimilated profiles separately from archive-screened or
unknown profiles and likely-assimilated profiles. Unknown status still permits
development and carefully labelled operational matchups; it does not support an
unbiased independent-reanalysis superiority claim. Do not discard the valid
independent stratum merely because other rows in the bundle have weaker provenance.

## Experiment 1: beat useful GLORYS post-processing

**Question:** does the model provide more than an inexpensive correction or an
uncertainty wrapper around GLORYS?

Use a separate calibration cohort to build this comparison ladder:

| Comparator | Information and role |
| --- | --- |
| Native GLORYS | The delivered point estimate |
| Bias-corrected GLORYS | Depth/season/region correction fitted without final labels |
| Probabilistic GLORYS | GLORYS plus calibrated joint residuals or a simple heteroscedastic distribution |
| GLORYS plus local optimal interpolation | Same permitted sparse context profiles as DepthDif |
| Probabilistic climatology | Training-only background and anomaly distribution |
| Published probabilistic reconstruction | A reproducible method with declared input/domain differences |

For the probabilistic GLORYS baseline, estimate residuals `observation - GLORYS`
from calibration units. Start with coarse, regularized depth/season bins;
complex region-specific fitting is inappropriate when only a few cruises exist.
Fit both location and dispersion. For joint tests, resample complete residual
profiles rather than independent noise at each depth. A simple covariance model
or empirical profile residuals is a stronger control than diagonal Gaussian noise.

Evaluate ensemble-mean RMSE, ensemble-median MAE, and profile-balanced CRPS on
the same independent final observations. Report bias and practical effect size,
not just significance. GLORYS-as-point-mass has CRPS equal to its absolute error;
that is a valid reference but a weak test of the need for a learned distribution.

**Convincing result:** lower CRPS than calibrated GLORYS and probabilistic
climatology, with useful conditional coverage/width, plus competitive point error.
**Falsification:** a regional bias shift or the calibrated GLORYS baseline removes
the claimed advantage. That supports a narrower post-processing result instead.

Keep operational and same-information comparisons separate. If GLORYS cannot be
rerun with observations removed, its observing budget has not been matched.
If DepthDif uses new profiles that GLORYS never saw, the gain can still be valuable,
but attribute it to a conditional update using new information.

## Experiment 2: prove that the learned joint distribution matters

**Question:** does a diffusion member describe a plausible whole profile or field,
rather than a collection of calibrated marginal values?

Use a particularly informative negative control: permute ensemble member labels
independently at each depth of a fixed case. Every depth retains exactly the same
ensemble mean, spread, quantiles, and univariate CRPS, but cross-depth dependence
is disrupted. Keep three diagnostic controls distinct:

1. Vertical dependence: independently permute members at each depth within a
   variable.
2. T/S dependence: preserve every complete vertical T and S profile, then permute
   whole S-member profiles relative to T members.
3. Spatial dependence: preserve each location's complete vertical T/S member,
   then independently permute whole-member labels between locations.

Use several pre-frozen permutation seeds and average their diagnostic results;
one finite-ensemble permutation adds Monte Carlo noise. These are ablations,
while the correlated GLORYS-error model remains a substantive comparator.

Compare intact and disrupted ensembles against observed:

- vertically integrated temperature anomalies and layer means;
- thermocline or isotherm depths and temperature-gradient structure;
- with joint T/S, density, freshwater content, and stratification;
- with coherent section observations, along-track differences at frozen lags.

Calculate a derived quantity for **each member**, then score its distribution.
For nonlinear quantities, the quantity of the ensemble mean is generally not the
mean of the member-wise quantities. Use a complete-profile energy score together
with a variogram score at physically meaningful vertical/horizontal separations.
Standardize using training-derived scales, not test variances. Variogram scores
are useful because the energy score can have limited sensitivity to erroneous
dependence. [Scheuerer and Hamill 2015](https://doi.org/10.1175/MWR-D-14-00269.1)
and [Pic et al. 2025](https://ascmo.copernicus.org/articles/11/23/2025/)
motivate dependence- and functional-sensitive proper verification.

**Convincing result:** the intact ensemble improves observed functional/variogram
scores against both the dependence-destroyed control and a correlated GLORYS-error
baseline. Better smoothness alone is not sufficient.

The existing point ensemble bundle supports within-profile tests where the depth
support is complete. Horizontal covariance requires points from the same
generated patch and realization. Independent samples from overlapping patches
must not be presented as a dynamically coherent global realization.

## Experiment 3: quantify the information supplied by sparse profiles

**Question:** can a few allowed observations improve predictions at other,
completely withheld profiles, and does uncertainty contract appropriately?

Freeze verification platforms/cruises first. Reveal 0, 1, 2, 4, and 8 permitted
context profiles at predeclared distance bands, or use retained fractions where
counts are low. Use the same targets and paired random seeds in each setting.
Remove complete test platforms from all overlapping input patches/windows; hide
both fields at all depths even when only T is scored.

Compare the frozen conditional model with local interpolation, climatology plus
interpolated innovations, and GLORYS plus an objective-analysis update using the
same context information. Fit covariance/radius parameters on calibration data.
Do not score at the newly revealed input profile itself.

Plot MAE/RMSE and CRPS against profile count and nearest-context distance;
also plot uncertainty width and empirical coverage. A no-profile pass on a model
not trained for zero input is an attribution stress test, not automatically a
competitive surface-only baseline. Use a separately matched surface-only model
for that performance claim. Profile/location/date shuffles are negative controls;
interpret them carefully because arbitrary shuffles can be out of distribution.

**Convincing result:** context observations reduce error on other profiles and
improve sharpness at maintained calibration, beyond the simple local update.
**Falsification:** changes occur only where observations are directly supplied,
or disappear once context distances and whole-platform exclusion are enforced.

This supports fast conditional reconstruction. It does not prove GLORYS would
require more observations to reach the same accuracy; that requires an actual
observing-system/assimilation rerun, not an offline mask applied only to DepthDif.

## Experiment 4: test physically useful vertical structure now

For the current temperature model, prioritize thermocline depth, preselected
isotherm depth, maximum resolved temperature gradient, thermal-layer depth, and
0–200 m layer-temperature anomaly. Report errors in metres or degrees C with
paired profile/block intervals. Also score the member-wise distributions of
these quantities with CRPS and coverage.

A temperature threshold layer is not density mixed-layer depth. A transparent
starting definition uses a 0.2 degrees C departure from a 10 m reference; examine
inversions and a fixed alternative definition. The established MLD literature
provides both threshold and hybrid approaches, and definition sensitivity must
be assessed. [de Boyer Montegut et al. 2004](https://doi.org/10.1029/2004JC002378),
[Holte and Talley 2009](https://doi.org/10.1175/2009JTECHO543.1).

Use identical vertical operators for every method. Native XBT resolves much
finer structure than the model's 50 levels; report the raw observation diagnostic
and a model-resolution sensitivity. Mark unresolved or multiply crossing
thermoclines rather than selecting the crossing closest to the prediction.
Incomplete layer coverage produces missing/censored quantities, not an integral
over a shorter layer mislabeled as 0–200 or 0–700 m.

For XBT-only heat diagnostics, use layer-average temperature or explicitly label
the constant-density/heat-capacity approximation `rho0*cp0*integral(T_anomaly dz)`.
It is not a full TEOS-10 heat-content estimate without salinity and a thermodynamic
convention. The prepared GLORYS salinity used to convert XBT temperature is not
independent observed salinity.

**Convincing result:** a materially better thermocline or layer estimate in a
predefined regime, even if global average profile error is merely competitive.

## Experiment 5: assess extremes without rewarding false alarms

Freeze local seasonal/depth 90th and 95th percentile anomaly thresholds from
training/calibration data with adequate support. Score exceedance probability
using Brier score and reliability on **all** verification cases, including
non-events. Report misses and false alarms together, and compare with both
climatological event probability and calibrated probabilistic GLORYS.

Outcome-filtered CRPS computed only on observed extreme cases is not a valid
overall probabilistic ranking. A threshold-weighted CRPS with fixed nonnegative
weights is a proper alternative for emphasizing a tail, alongside ordinary CRPS;
it should not be confused with multiplying a score by an observed-event indicator.
See [Lerch et al. 2017](https://doi.org/10.1214/16-STS588) and
[Gneiting and Ranjan 2011](https://doi.org/10.1198/jbes.2010.08110).

Use subsurface threshold-exceedance language for isolated weekly-stride profiles.
The standard marine-heatwave definition includes a daily duration requirement;
the current sampling cannot identify five-day persistence or onset timing.
That needs daily coherent fields and sustained independent observations.
[Hobday et al. 2016](https://doi.org/10.1016/j.pocean.2015.12.014).

**Convincing result:** better event probabilities with lower false-alarm-adjusted
loss or predeclared decision cost. A collection of spectacular extreme members
or increased event recall from an overly broad ensemble is not enough.

## Experiment 6: show uncertainty can guide a useful decision

Choose an operational decision before opening final labels: request an additional
profile, flag a questionable reconstruction, or abstain from using it downstream.
Use predictive quantities available at inference time to rank cases. Evaluate
risk-versus-retained-coverage curves and errors at fixed retained fractions.

Compare model spread/interval width with simple predictors: nearest-profile
distance, observation count, season/depth, SST-gradient strength, and a calibrated
GLORYS-error model. Select thresholds only on calibration data. Ranking by the
actual GLORYS error on the test set is an oracle diagnostic, not a deployable
selection policy. Report how much coastline, region, and season coverage is lost
when rejecting difficult cases.

An even more direct observing experiment simulates a fixed budget for additional
context profiles, chosen without seeing their values. Compare random, spatial
coverage, simple baseline uncertainty, and model-guided policies on a distinct
verification set. Merely sampling at the eventual verification locations leaks
the answer and is not an observing-value result.

**Convincing result:** lower error at a fixed retained coverage, or more error
reduction per added observation than simple policies. Spread-error correlation
alone is only a diagnostic.

## Where to look, without selecting regions from favorable test errors

| Regime/data | Scientific motivation | Primary proposed endpoint | Priority |
| --- | --- | --- | --- |
| PNRA Southern Ocean XBT | Data already recovered; repeated crossings of strong hydrographic regimes | T 50–100 m, thermal-layer/isotherm depth; coarse-scale front position | High development value; only two cruises in the recovered 2016 subset |
| North Atlantic 59.5 N | Published overflow/deep-water discrepancies | Predefined 700–2000 m T/S and water-mass properties | Secondary; current surface-only signal may be weak |
| Beaufort / East Greenland | Freshwater and halocline hypotheses | Layer salinity and freshwater anomaly, density stratification | Future S/joint model |
| Weddell repeat hydrography | Recent deep-water bias/trend evidence | Water-mass T/S properties | Future joint model; low priority for this checkpoint |

The 59.5 N study motivates a defined overflow/deep-water test, not arbitrary
North Atlantic box selection. [Verezemskaya et al. 2021](https://doi.org/10.1029/2020JC016317).
Recent Weddell work identifies water-mass bias and trend concerns, but a small
same-date sample cannot test a multidecadal trend. [Noro et al. 2025](https://nora.nerc.ac.uk/id/eprint/540478/).
East Greenland freshwater behavior supplies a separate salinity hypothesis.
[Schiller-Weiss et al. 2023](https://doi.org/10.1029/2023JC019890).

## Future joint T/S: make covariance a real scientific target

Once compatible weights exist, prioritize density-based MLD, pycnocline depth,
freshwater content over fixed complete layers, barrier-layer thickness, and
density-compensated temperature/salinity anomalies. Compute density using TEOS-10
with explicit salinity and temperature conventions. Apply native layer thickness
and partial-layer weights for integrals.

Compare intact joint members with whole-profile T/S member-pairing permutations
that preserve each variable's internal vertical dependence.
Evaluate density and functional CRPS plus T/S vertical variogram scores. Check
unphysical values and resolved static-instability frequencies against QC data
and native GLORYS; genuine ocean inversions and sensor noise mean zero inversion
is not a universal physical objective. Never sort each sampled profile into a
stable state only for scoring without declaring and evaluating that transformation.

This could distinguish a conditional field model from independently reconstructed
profiles. It cannot establish better velocity, transport, overturning, turbulence,
or sea-ice dynamics from the current output variables.

## New literature changes the novelty argument

Probabilistic ocean-profile reconstruction already exists. The 2020 stacked
LSTM/Monte Carlo dropout method has [author code](https://github.com/bbuong/3Drec).
A July 2026 paper reconstructs 5–200 m temperature ensembles from ERA5 surface
forcing/history with CRPS training and released inference code.
[Cavaiola 2026](https://doi.org/10.1016/j.apor.2026.105121),
[author repository](https://github.com/Mattiads1/AI-Sea-Temperature-Profile-reconstruction).
Neither is automatically a same-input, same-domain baseline. Do not compare
published aggregate scores to our numbers directly. The contribution must be
measured conditional information use, spatial/vertical joint structure, calibrated
functional probabilities, or a demonstrated cost-quality advantage.

A finite issued ensemble has a valid empirical-CDF CRPS. The fair CRPS correction
targets an underlying distribution when members are random draws under the
appropriate assumptions; it is not automatically appropriate to deterministic
quantile/output heads. Use member-count sensitivity and clearly distinguish the
two interpretations. [Ferro 2014](https://doi.org/10.1002/qj.2270).

## Decision rules, sample size, and the paper package

Choose one primary profile-balanced endpoint per experiment. Within a profile,
use a frozen depth grid/band weighting or vertical quadrature; the 0.65 m XBT
samples must not overwhelm sparsely sampled profiles. The existing runner's pooled
probabilistic/reliability tables and profile-balanced comparisons are different
estimands and need a consistent primary convention before publication.

Bootstrap whole independent deployment/cruise/platform-year units, with sensitivity
to residual space/time correlation. Two XBT cruises remain two independent units
regardless of their row count. There is no universal magic threshold of 20 or 30
clusters: choose sample size using a desired detectable practical effect and
pilot block variability. More clusters cannot repair biased matching or leakage.

For region-specific claims, freeze one to three mechanistically motivated regions
on discovery data and confirm on separate units. Multiplicity correction of a
searched map is useful screening, not a substitute for confirmation. Pair a
regional superiority claim with a predeclared aggregate non-inferiority margin;
do not silently exchange broad degradation for one local win. An observation-
sampled global mean is not an area-weighted ocean mean.

Suggested compact figures:

1. Paired profile error/CRPS differences against raw, corrected, updated, and
   probabilistic GLORYS, with intervals and independent-unit counts.
2. Skill versus retained profile count and nearest-context distance.
3. Intact-versus-permuted member scores for layer temperature and thermocline depth;
   joint T/S density/freshwater later.
4. Conditional calibration, event reliability, and a risk-coverage curve.
5. Accuracy/probabilistic-score versus total wall/GPU cost for 10/20/50/100 members.

Include preprocessing, data loading, and amortized training/storage in cost
reporting where relevant. Timing the network against a GLORYS file lookup does not
measure the cost of producing GLORYS; timing it against an entire reanalysis is
not a matched-workload comparison either. Current independent target dates do not
support trajectory-based temporal or climate-trend claims.

### Execution priority

1. Native GLORYS/clock/precision audit and consistent profile-level estimand.
2. Calibrated GLORYS plus probabilistic climatology; frozen-model correction and
   conditioning diagnostics on discovery cases.
3. Temperature thermocline/layer and dependence-permutation tests.
4. Predefine the final region, meaningful margin, primary metric, and independent
   cohort; only then run the larger confirmatory ensemble.
5. Extremes and decision utility once calibration is credible; joint T/S physical
   diagnostics when compatible weights arrive.

If the model cannot beat a calibrated/simple updated GLORYS reference, report
that outcome. A scientifically useful efficiency or conditional-sampling result
may still be possible, but the evidence must identify that narrower benefit.
