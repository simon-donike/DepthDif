# Spectral Wavenumber Validation

This page documents the post-inference validation procedure used by
`depth_recon.inference.export_wavenumber_spectra` and covered by
`tests/test_wavenumber_spectra.py`. The diagnostic compares the spatial
structure of DepthDif predictions against GLORYS reference fields, and when
available against surface observations, by estimating isotropic 2D wavenumber
spectra on the same selected patch windows.

## Scope

The analysis is a spatial spectrum diagnostic, not a replacement for pixelwise
MAE or profile error validation. It answers whether generated fields preserve
energy at comparable horizontal scales:

- high wavenumber / short wavelength bins test small-scale texture
- low wavenumber / long wavelength bins test basin-scale smoothness
- prediction-to-GLORYS spectral ratios show scale-dependent smoothing or excess
  variance

Version 1 analyzes temperature and salinity fields directly in physical units.
It does not estimate buoyancy spectra, internal-wave partitions, or
frequency-wavenumber spectra.

## Inputs

For each discovered single-variable run, the exporter reads:

- `selected_patches.csv` for patch footprints and grid offsets
- prediction GeoTIFF depth exports
- GLORYS GeoTIFF depth exports, or packaged GLORYS source rasters as a fallback
- surface observation rasters for the variable when packaged source data exists
  (`OSTIA analysed_sst` for temperature, `SSS sos` for salinity)
- `world_oceans.geojson` regions for basin-level aggregation

Integer source rasters with stretch tags are decoded before spectral analysis.
For temperature rasters stored in Kelvin, decoded values are shifted to Celsius:

$$
T_C = T_K - 273.15
$$

By default, patch windows with nodata or `NaN` are included. The exporter fits
the detrending plane on finite pixels, then zero-fills missing residuals before
the Hann-windowed FFT so prediction and GLORYS use the same land/coast support.
Pass `--require-complete-patches` to keep only complete finite patch windows.

## Patch Geometry

Each selected patch row defines lon-lat bounds
`(lon0, lon1, lat0, lat1)` and a raster window. Pixel sizes are approximated in
kilometers from the patch center latitude:

$$
\Delta x =
\frac{|lon_1 - lon_0|}{W} \, 111.32 \, \max(|\cos(\phi_c)|, 10^{-6})
$$

$$
\Delta y =
\frac{|lat_1 - lat_0|}{H} \, 111.32
$$

where `W` and `H` are the patch width and height in pixels and
`phi_c = (lat_0 + lat_1) / 2`.

## Detrending

For one patch field `X(i, j)`, with row index `i` and column index `j`, a planar
trend is fitted by least squares:

$$
(\hat{a}, \hat{b}, \hat{c})
=
\arg\min_{a,b,c}
\sum_{(i,j) \in \Omega}
\left[X(i,j) - (a j + b i + c)\right]^2
$$

`Omega` is the finite-pixel support. The detrended residual is:

$$
R(i,j) = X(i,j) - (\hat{a}j + \hat{b}i + \hat{c})
$$

This removes patch-scale mean and linear gradients before measuring spectral
power, so the diagnostic emphasizes resolved spatial variability rather than a
tilted local background.

## Windowed FFT

A separable 2D Hann window is applied:

$$
w(i,j) = w_y(i) w_x(j)
$$

$$
w_x(j) = \frac{1}{2}\left(1 - \cos\frac{2\pi j}{W-1}\right),
\quad
w_y(i) = \frac{1}{2}\left(1 - \cos\frac{2\pi i}{H-1}\right)
$$

The windowed field is:

$$
Y(i,j) = R(i,j) w(i,j)
$$

The 2D discrete Fourier transform follows `np.fft.fft2`:

$$
F(m,n) =
\sum_{i=0}^{H-1}\sum_{j=0}^{W-1}
Y(i,j)
\exp\left[-2\pi \mathrm{i}\left(\frac{mi}{H}+\frac{nj}{W}\right)\right]
$$

Power is normalized by Hann-window energy:

$$
P(m,n) =
\frac{|F(m,n)|^2}{\sum_{i,j} w(i,j)^2}
$$

This normalization keeps spectra from same-sized windows comparable after
tapering.

## Radial Wavelength Binning

FFT frequencies are converted to cycles per kilometer:

$$
k_x(n) = \mathrm{fftfreq}(W, \Delta x)_n,
\quad
k_y(m) = \mathrm{fftfreq}(H, \Delta y)_m
$$

The isotropic radial wavenumber and wavelength are:

$$
k_r(m,n) = \sqrt{k_x(n)^2 + k_y(m)^2}
$$

$$
\lambda(m,n) = \frac{1}{k_r(m,n)}
$$

The zero-frequency term, where `k_r = 0`, is excluded. Wavelength-bin edges are
log-spaced:

$$
e_q =
\lambda_{\min} \,
\left(\frac{\lambda_{\max}}{\lambda_{\min}}\right)^{q/Q}
$$

with default `lambda_min = 30 km`, `lambda_max = 1000 km`, and `Q = 32`. The
displayed bin center is the geometric mean:

$$
\lambda_q^\ast = \sqrt{e_q e_{q+1}}
$$

Patch spectral power for bin `q` is first computed as the mean Fourier power
over all coefficients whose wavelengths fall inside the bin:

$$
S_q =
\frac{1}{|B_q|}
\sum_{(m,n)\in B_q} P(m,n),
\quad
B_q = \{(m,n): e_q \le \lambda(m,n) < e_{q+1}\}
$$

The exporter also converts each wavelength bin to its horizontal-wavenumber
width in cycles per kilometer:

$$
\Delta k_q = \frac{1}{e_q} - \frac{1}{e_{q+1}}
$$

and reports a one-dimensional radial PSD estimate:

$$
PSD_q = \frac{S_q}{\Delta k_q}
$$

Bins with no FFT coefficients are recorded as missing.

## Basin Assignment

Each patch footprint is intersected with the ocean-region polygons. A patch is
assigned to the basin with maximum overlap only when:

$$
\frac{\mathrm{area}(A_{\mathrm{patch}} \cap A_{\mathrm{basin}})}
{\mathrm{area}(A_{\mathrm{patch}})}
\ge 0.30
$$

The default threshold is `0.30` and is controlled by `--basin-overlap-threshold`.
Patches below the threshold remain in `All Oceans` but are excluded from named-basin aggregates.

## Aggregation

Each accepted patch spectrum is stored with variable, layer, date, season, basin,
and depth metadata. Aggregation groups spectra by:

- variable
- layer: prediction, GLORYS, or surface observation
- depth
- basin scope: `All Oceans` plus named basins
- period: year, season, or month

For group `G`, the mean spectrum is:

$$
\bar{S}_{G,q} =
\frac{1}{N_{G,q}}
\sum_{p \in G_q} S_{p,q}
$$

where `G_q` contains spectra with finite power in bin `q` and `N_{G,q}` is the
finite count. The exported `spectrum_count` reports how many patch spectra
contributed at least one finite bin to the group.

The dashboard derives prediction-vs-GLORYS scale diagnostics from the aggregated
spectra:

$$
R_q = \frac{\bar{S}^{pred}_q}{\bar{S}^{glorys}_q}
$$

$$
B_q = \bar{S}^{pred}_q - \bar{S}^{glorys}_q
$$

$$
B^{rel}_q =
\frac{\bar{S}^{pred}_q - \bar{S}^{glorys}_q}{\bar{S}^{glorys}_q}
$$

Values below 1 in `R_q` indicate that predictions contain less variance than
GLORYS at that wavelength. Values above 1 indicate excess variance.

## Interpretation

The dashboard displays horizontal wavenumber in cycles per kilometer (`cpkm`)
on a log axis that increases left-to-right by default. Use the X Axis selector
to switch the display to wavelength in kilometers (`km`), also on an increasing
left-to-right log axis. Hover labels report the equivalent wavelength. Because
each patch is detrended before the FFT, the spectra describe residual spatial
variability within patches, not the patch mean or a local linear gradient.

The dashboard plots `psd_mean`, which is `power_mean` divided by the
horizontal-wavenumber bin width. The paper-style velocity unit
`m^2/(s^4 cpkm)` still does not apply because our fields are temperature and
salinity rather than that velocity-derived quantity. The dashboard labels the
y-axis as `PSD [degC^2/cpkm]` for temperature and `PSD [salinity^2/cpkm]` for
salinity. A good prediction spectrum tracks the GLORYS spectrum with a similar
level and slope across the wavelength range. A clearly lower prediction curve
means the model is too smooth at those scales. A clearly higher curve means it
is adding too much variance, often visible as noisy or speckled texture at
short wavelengths.

The prediction/GLORYS ratio is the most direct quality signal:

- `R_q = 1` is ideal agreement at that wavelength.
- roughly `0.8` to `1.25` is usually a close scale match.
- values around `0.5` to `0.8` indicate moderate smoothing.
- values below `0.5` indicate severe loss of variance at that scale.
- values around `1.25` to `2` indicate excess variance.
- values above `2` indicate strong over-energizing or likely artifacts.

These ranges are rules of thumb, not pass/fail limits. They should be judged
more strictly where many spectra contribute and more cautiously where
`spectrum_count` is small or bins contain few FFT coefficients.

For dashboard metrics, `relative_bias = 0` is ideal, negative values mean
smoothing, and positive values mean excess variance. Ratio and relative bias are
unitless; relative bias is plotted as a fraction, so `0.25` means `+25%`. The
`difference` metric is in PSD units and is best used to see where the
absolute mismatch is largest. The magnitude is not directly comparable across
variables or depths. The summary cards use the currently selected metric:
`High freq` averages the short-wavelength range up to about
100 km, and `Large scale` averages the long-wavelength range from about 300 km
upward. `Slope diff` is the fitted log-power-vs-log-wavelength slope for
prediction minus GLORYS. Values near zero are best. Positive values usually
mean the prediction is too dominated by large scales; negative values usually
mean relatively too much small-scale power.

Good diagnostics therefore look like:

- prediction, GLORYS, and OSTIA/SSS curves nearly parallel and close together
- prediction/GLORYS ratio near 1 across both short and long wavelengths
- relative bias near 0 without a persistent sign across wavelengths
- high `spectrum_count` for the selected basin and period

Bad diagnostics commonly look like:

- low short-wavelength ratio, showing over-smoothed fronts or missing eddies
- high short-wavelength ratio, showing noisy or speckled predictions
- low long-wavelength ratio, showing basin-scale structure is muted
- high long-wavelength ratio, showing broad false gradients or large blobs
- large slope differences, showing the model has the wrong scale balance

## Test Coverage

`tests/test_wavenumber_spectra.py` validates the scientific and export
assumptions with small deterministic fixtures:

- exact planar fields detrend to numerical zero
- a sinusoidal field has its Hann-windowed FFT peak in the expected wavelength
  bin
- incomplete patches are skipped by default
- raster nodata is converted to `NaN`
- stretched uint8 temperature rasters decode to Celsius
- basin assignment respects the overlap threshold
- paired and temporal run discovery expands the expected run roots
- the full exporter writes patch spectra, aggregated spectra, summary JSON,
  plots, basin JSON, and static dashboard assets
- dashboard generation can be disabled with `--no-dashboard`
- the hosted spectral dashboard exposes the expected controls and links from the
  analysis landing page

## Main Outputs

The procedure writes:

- `patch_spectra.npz`: dense patch-by-wavelength spectrum matrix and wavelength
  edges/centers
- `patch_spectra_records.csv`: one metadata row per accepted patch spectrum
- `aggregated_spectra.csv`: grouped spectra with `power_mean`, `psd_mean`, and wavenumber-bin metadata used by plots and dashboards
- `summary.json`: run, binning, skip-count, and artifact metadata
- `plots/*.png`: default log-log seasonal/yearly spectra
- `spectral-config.json`, `basin-map.geojson`, `basins/*.json`, and copied
  dashboard static files when dashboard export is enabled

Typical command:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_wavenumber_spectra \
  --run-dir inference/outputs/global_variables_2018_W25_v2 \
  --include-temporal-runs \
  --variables temperature salinity \
  --output-dir inference/outputs/global_variables_2018_W25_v2/wavenumber_spectra
```
