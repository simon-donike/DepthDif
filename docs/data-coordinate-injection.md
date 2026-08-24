# Coordinate and Date Conditioning

The maintained pixel model encodes one patch location and target date, projects
them through a shared MLP, and injects the resulting embedding into ConvNeXt
blocks with feature-wise linear modulation (FiLM).

## Configuration

```yaml
model:
  coord_conditioning:
    enabled: true
    encoding: unit_sphere
    include_date: true
    date_encoding: day_of_year_sincos
    embed_dim: null
data:
  dataset:
    output:
      return_coords: true
```

When coordinate conditioning is enabled, missing `coords` raises an error. When
date conditioning is also enabled, missing `date` raises an error.

## Coordinate encodings

- `unit_sphere` converts latitude/longitude to a three-component point on the
  unit sphere. It is the current default and is continuous across the dateline.
- `sincos` returns sine/cosine pairs for latitude and longitude.
- `raw` returns latitude/90 and longitude/180; it is compact but discontinuous at
  the dateline.


## Date encoding

`day_of_year_sincos` parses integer `YYYYMMDD`, validates the calendar fields,
computes day of year, and returns

\[
\left(\sin(2\pi d/365),\cos(2\pi d/365)\right).
\]

This represents annual periodicity without treating the end and start of a year
as distant scalar values.


## FiLM path

The coordinate and optional date vectors are concatenated and projected once by
`coord_mlp`. The U-Net passes the same embedding to conditioned blocks. Each block
produces channel-wise scale and shift values and applies

\[
h' = h\,(1+s) + b,
\]

with spatial broadcasting. Diffusion timestep conditioning remains a separate
additive path. Near-zero FiLM output therefore leaves the block close to its
unconditioned behavior.

The encoding and dispatch live in the conditional diffusion process; FiLM is
applied in the ConvNeXt U-Net backbone. Tests cover supported encodings, required
batch fields, and conditioned model dry runs.
