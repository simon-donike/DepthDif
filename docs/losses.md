# Diffusion and Auxiliary Losses

DepthDif always computes a base diffusion loss. Optional auxiliary terms can add
sparse-anchor, profile-shape, structure-function, or spectral constraints. All
optional terms are disabled in the maintained presets.

## Combined objective

For enabled terms, the implementation forms

\[
\mathcal{L} = w_{base}\mathcal{L}_{base} +
g(t)\left(
w_{obs}\mathcal{L}_{obs} +
w_{inc}\mathcal{L}_{inc} +
w_{str}\mathcal{L}_{str} +
w_{spec}\mathcal{L}_{spec}
\right),
\]

where `g(t)` is the optional auxiliary-only timestep weight. It never rescales
the base diffusion term.

Current configured weights are:

| Term | Enabled | Weight |
| --- | --- | ---: |
| Base diffusion/ambient | Yes | 1.0 |
| Sparse observation consistency | No | 0.25 |
| Sparse profile increments | No | 0.1 |
| Structure-function prior | No | 0.1 |
| Spectral energy floor | No | 0.05 |
| Feature Gram prior | No, unimplemented | 0.01 reserved |

## Base supervision

Standard dense training compares the configured diffusion target with paired
GLORYS or the deterministic synthetic target over valid target/ocean support.
Ambient training instead uses the original sparse observation support described
in [Ambient occlusion](ambient-occlusion-objective.md).

The optional coastal weighting multiplies valid ocean loss near land according
to its radius, maximum weight, and ramp. It is disabled in all current pixel
presets.

## Auxiliary timestep weighting

The local preset enables SNR weighting with minimum 0.1, maximum 1.0, and
`snr_gamma: 5.0`. Since no auxiliary term is enabled, this currently has no
numerical effect. It becomes active only when an auxiliary loss is opted in.

The supported modes are:

- `snr`: derives a normalized, clamped weight from diffusion SNR.
- `linear`: interpolates between configured noisy- and clean-timestep weights.

## Sparse observation consistency

This term applies a Charbonnier penalty between the predicted clean field and
observed EN4/ARGO values on valid sparse support:

\[
\rho(r)=\sqrt{r^2+\epsilon^2}.
\]

It is an optional extra anchor; ambient base loss already uses sparse observed
support in the maintained local objective.

## Sparse profile increments

The increment term compares differences between observed pairs rather than
absolute values. Vertical adjacent-depth pairs are enabled in its configuration;
horizontal pairs are optional. Pair sampling is capped per sample to bound cost.

## Structure-function prior

For sampled same-depth pixel pairs separated into distance bins, the second-order
structure statistic is based on

\[
S_2(r)=\mathbb{E}[(x(p+r)-x(p))^2].
\]

The loss compares log statistics against either a precomputed reference `.pt`
file (`target: reference`) or the paired GLORYS target
(`target: paired_glorys`). Reference mode requires `reference_path`.

## Spectral energy floor

This term computes radial spatial-frequency energy and penalizes predicted bands
that fall below a configured reference or paired-GLORYS energy floor. It skips
the DC band by default and uses a hinge in log-energy space. It is distinct from
the post-inference validation exporter in [Spectral validation](spectral-wavenumber-validation.md).

## Unsupported feature prior

`feature_gram_prior` reserves configuration for a frozen encoder and reference
feature statistics, but the loss is not implemented. Enabling it raises
`NotImplementedError`; documentation and configs must not present it as usable.

## Scenario limits

The optional auxiliary implementations operate on scalar temperature or salinity
fields. Keep them disabled for `--scenario joint` unless the implementation and
tests are extended to define joint-field semantics.
