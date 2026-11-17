# Cavaiola 2026 comparison

The local paper is [instructions/2026_lstm.pdf](../instructions/2026_lstm.pdf):
*Surface-driven AI reconstruction of global subsurface ocean temperature profiles
from ERA5 forcing*, [Applied Ocean Research, DOI 10.1016/j.apor.2026.105121](https://doi.org/10.1016/j.apor.2026.105121).
The [author's repository](https://github.com/Mattiads1/AI-Sea-Temperature-Profile-reconstruction)
includes the TensorFlow architecture, inference pipeline, trained weights and
scalers. Source inspected at commit `31005802c24085ed6b831103c87d21455482285a`.
Our existing pointwise vertical LSTM is a different model.

## How the paper evaluates

- Training: 371,634 Argo profiles from 2004–2020. Testing: 86,291 profiles from
  2021–2023 (Table 1). The internal validation fraction is 10%.
- Observations: delayed-mode/adjusted temperature where available, filtering
  profiles with more than 20% missing upper-ocean data. Observations are linearly
  interpolated to 40 depths, 5–200 m every 5 m (section 2.1).
- Inputs: 25 ERA5 variables at 25 times spanning six preceding days at six-hour
  intervals, plus latitude and cyclic longitude/month. No subsurface observations
  enter inference. The released code uses `t0, t0-6h, ..., t0-144h`.
- Model: temporal BiLSTM, attention, static dense branch, and vertical convolutions.
  The released implementation uses 128 LSTM units per direction. Its 100 outputs
  are deterministic channels jointly trained with empirical CRPS, rather than
  independent stochastic model runs (section 2.4 and `utils/model.py`).
- Reference: daily GLORYS12V1 at the nearest horizontal grid point and observation
  date, linearly interpolated to the same 40 depths (section 2.3). Argo may have
  been assimilated by GLORYS; this is an observation-constrained reference.
- Point scores: ensemble-mean profile NMSE, vertical Pearson correlation, and
  depth-dependent signed bias. NMSE is `sum(error**2) / sum(observation**2)`,
  rather than normalization by variance (Appendix B). It depends on temperature
  origin/scale; retain our physical MAE/RMSE as primary comparison metrics.
- Summaries: six ocean basins, profile-metric distributions and maps, depth bands
  5–50, 50–100 and 100–200 m, and bootstrap 95% confidence intervals. Their
  March–August "summer" applies globally; it is not hemisphere-adjusted.
- Uncertainty: empirical CRPS, rank/PIT histograms, a histogram-uniformity Delta
  index, ensemble variance and spread/error deciles (section 3.2).
- Roughly 70% of test floats also occur in training. They additionally compare
  seen/unseen float IDs, but do not perform a strict spatial block holdout
  (section 3.1, Table 3). Their claim of negligible differences should be read
  with the table: North Atlantic NMSE is 8.2% for seen versus 4.0% for unseen.
- Ridge and XGBoost baselines are mentioned, but their results are not shown.

The paper's prose reverses the interpretation of bias signs in places;
Appendix B defines prediction minus observation. The release uses Argo `TEMP`
targets while GLORYS supplies `thetao`; an explicit temperature conversion is
needed for our benchmark. The paper does not document that conversion.
The claim that deterministic GLORYS cannot have CRPS is too broad: its point-mass
CRPS equals absolute error. Our scorecard includes that reference.

## Optional diagnostic integration

The released weights include **2016 in their training period**. Their exact
training-profile registry is unavailable, so profile overlap is not established
or ruled out. This adapter always labels results as a training-era diagnostic
and sets `confirmation_eligible: false`. It does not change training, the frozen
2016 cohort, or the main full-depth scores.

`val.cavaiola` calls the original preparation and inference functions in a
separate CLI process. It retains ERA5 inputs and records hashes of source,
checkpoint, scalers, and predictions. The scoring stage uses safe numeric NPZ
files, matches cast ID/time/coordinates, interpolates every head to the existing
observation depths, and restricts **every method** to temperature at 5–200 m.
It converts the predicted in-situ temperature to potential temperature with
TEOS-10 and explicit practical reference salinity. It never searches for another
Argo profile or downloads another GLORYS reference.

The diagnostic exports equal-profile-weight MAE, RMSE, bias and empirical CRPS
for Cavaiola, DepthDif and every available full-conditioning method: packaged
GLORYS, climatology, IDW, LSTM, CNN, U-Net 3D, U-Net 2D, and calibrated references.
It includes paired cluster-bootstrap differences against each method. During
the full suite this scorecard is produced after the calibration-baseline stage.
Its heads remain separate from stages that require independent
draws or validated joint dependence; the fair CRPS correction is not applied.
This adapter does not reproduce the paper's complete basin/season/PIT analysis.
ERA5 uses the observation timestamp; our stored DepthDif/GLORYS predictions retain
their original weekly target times. The matched table preserves that difference.

1. Obtain the author's checkout and dependencies. For example:

   ```bash
   git clone https://github.com/Mattiads1/AI-Sea-Temperature-Profile-reconstruction /tmp/cavaiola2026
   git -C /tmp/cavaiola2026 checkout 31005802c24085ed6b831103c87d21455482285a
   ```

   The inference process needs the upstream TensorFlow/legacy Keras,
   scikit-learn (released scalers record 1.6.1), and ERA5 dependencies. See the
   author's `requirements.txt` and README. CDS requires the user's existing
   credentials and accepted dataset terms. These dependencies and the ERA5 inputs
   are not bundled or installed by this integration. Use only
   `/work/envs/depth/bin/python`; any optional package target must be added to
   `PYTHONPATH` explicitly. The existing `val/vendor` supplies optional GSW.

2. After generating a normal benchmark bundle, run the released model:

   ```bash
   PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.cavaiola \
     --points val/results/en4/bundle/points.csv \
     --repository /tmp/cavaiola2026 \
     --output val/results/cavaiola2026 --source cds
   ```

   Choose a new output directory. This makes ERA5 requests for each eligible
   cast; source `gcs` is also available but may transfer large global chunks.
   The command prints a `paper.comparators.cavaiola2026` configuration block
   containing the manifest path and SHA256. Add it to a copy of the cohort config.

3. Supply practical salinity for the common temperature conversion. The XBT
   bundle retains its documented `reference_salinity` column. Otherwise provide
   a CSV with `point_id,reference_salinity` for every eligible point and add:

   ```yaml
   paper:
     comparators:
       cavaiola2026:
         manifest: val/results/cavaiola2026/manifest.json
         manifest_sha256: <printed SHA256>
         reference_salinity:
           path: val/inputs/cavaiola_reference_salinity.csv
           sha256: <CSV SHA256>
   ```

   Use the same documented salinity assumption as the other temperature
   conversions. Missing, duplicate, or nonfinite salinity is an explicit error.

4. Rescore the saved bundle using that copied config:

   ```bash
   PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.run suite \
     --bundle val/results/en4/bundle --config val/config_cavaiola_local.yaml \
     --output val/results/cavaiola_comparison
   ```

   The comparator stage contains `cavaiola_scores.csv`,
   `cavaiola_paired_comparisons.csv`, `cavaiola_matched_points.csv` and the
   aligned member matrices. No reported result implies held-out 2016 skill.

Tests cover matching, no extrapolation, per-head temperature conversion,
artifact integrity, diagnostic labels, and keeping the main benchmark intact.
They use synthetic upstream outputs; no real ERA5/TensorFlow inference has been
run as part of adding this adapter.
