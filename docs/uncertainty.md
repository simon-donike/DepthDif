# Uncertainty

DepthDif estimates predictive uncertainty with an ensemble of stochastic reverse
diffusion samples. For the same input batch and conditioning tensor, inference is
run \(N\) times:

\[
\hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(N)}
\]

The prediction used for a pixel can be summarized by the ensemble mean:

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}\hat{y}^{(i)}
\]

Uncertainty is the pixel-wise ensemble standard deviation:

\[
\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\left(\hat{y}^{(i)} - \mu\right)^2}
\]

This is computed after denormalization, so temperature and salinity uncertainty
maps are reported in physical units. Temperature uncertainty is in degrees
Celsius, and salinity uncertainty is in PSU.

For multi-depth outputs, the implementation computes the per-channel standard
deviation as a depth-resolved \(B \times D \times H \times W\) tensor. The
production global exporter keeps that tensor by default and writes uncertainty
GeoTIFFs for the same depth levels used by the globe. Callers that need the older
single-map behavior can request channel collapse, which averages the depth
channels into a \(B \times 1 \times H \times W\) raster. Joint
temperature/salinity runs keep field-specific uncertainty maps before producing
their normalized display rasters.

Empirically, the observed reconstruction error lines up quite well with the
estimated uncertainty. This is the expected behavior: regions where the ensemble
samples disagree more should also be regions where the model is more likely to
make larger errors.
