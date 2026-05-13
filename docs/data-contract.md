# Data Contract

This page defines the model-facing sample produced from the GeoTIFF workflow.
The files on disk are byte-encoded rasters and a preprocessed ARGO profile
store; the loader decodes them into physical units, normalizes temperature, and
returns PyTorch tensors with the shapes below.

## Axes

Symbols used in this contract:

| Symbol | Meaning | Default |
| --- | --- | --- |
| `B` | batch size after `DataLoader` collation | configured by training |
| `D` | GLORYS depth levels | `50` |
| `H` | patch height in raster rows | `128` |
| `W` | patch width in raster columns | `128` |

The default horizontal resolution is `0.1` degrees. With `H = W = 128`, one
training patch covers `12.8 x 12.8` degrees. The default GeoTIFF patch stride is
`32` pixels, so neighboring patches overlap by 75% of a tile.

## Sample Keys

Each dataset item is a dictionary. After collation, tensor keys gain a leading
batch dimension.

| Key | Item shape | Batch shape | Dtype | Meaning |
| --- | ---: | ---: | --- | --- |
| `eo` | `(1, H, W)` | `(B, 1, H, W)` | `float32` | Dense OSTIA surface temperature context, normalized. |
| `x` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Sparse ARGO temperature observations, normalized and zero-filled where missing. |
| `y` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Dense GLORYS `thetao` target, normalized and zero-filled where invalid. |
| `x_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | True where `x` contains an observed ARGO temperature value. |
| `y_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | True where the GLORYS target is valid ocean data. |
| `x_valid_mask_1d` | `(1, H, W)` | `(B, 1, H, W)` | `bool` | True where any ARGO depth is present in that horizontal pixel. |
| `land_mask` | `(1, H, W)` | `(B, 1, H, W)` | `float32` | Horizontal target support mask; `1` where the first target depth is valid and `0` elsewhere. |
| `date` | scalar | `(B,)` | integer | GLORYS target date as `YYYYMMDD`. |
| `coords` | `(2,)` | `(B, 2)` | `float32` | Optional patch-center latitude and longitude. |
| `info` | dictionary | list-like | metadata | Optional debugging metadata, not part of the training model input. |

Despite the historical name, `land_mask` is a model support mask derived from
target validity. Training code should use `y_valid_mask` for supervised losses
and should not infer missing values from zeros in `x`, `y`, or `eo`.

## Loading Steps

For each selected `(patch, date)` row, the GeoTIFF loader should:

1. Build a rasterio window from the shared land-mask grid.
2. Read `rasters/glorys/thetao/thetao_YYYYMMDD.tif` as `(D, H, W)`.
3. Read `rasters/ostia/analysed_sst/analysed_sst_YYYYMMDD.tif` as `(H, W)` and
   add the leading channel dimension.
4. Query preprocessed ARGO profiles assigned to the same target date and patch
   window.
5. Rasterize ARGO temperature onto `(D, H, W)` using precomputed `grid_row` and
   `grid_col`; average duplicate observations in the same depth/pixel cell.
6. Build validity masks from GeoTIFF nodata codes and ARGO valid flags.
7. Normalize temperature tensors and replace NaN or infinite normalized values
   with `0.0`.

GLORYS salinity, ARGO salinity, and sea-level `adt` are exported on the same
grid for auxiliary experiments. They should only be added to model inputs with
an explicit model/config change that defines their channels and normalization.

## Decoding

All exported rasters use `uint8` with `255` reserved for nodata:

```text
0..254 = valid stretched values
255    = nodata
decoded = minimum + code / 254 * (maximum - minimum)
```

Temperature rasters decode to Kelvin:

| Product | Variable | Decoded units | Stretch |
| --- | --- | --- | --- |
| GLORYS | `thetao` | Kelvin | `[270.15, 308.15]` |
| OSTIA | `analysed_sst` | Kelvin | `[270.15, 308.15]` |
| ARGO | temperature | Kelvin | `[270.15, 308.15]` |

Other exported variables decode to their physical units:

| Product | Variable | Decoded units | Stretch |
| --- | --- | --- | --- |
| GLORYS | `so` | PSU | `[30, 40]` |
| ARGO | salinity | PSU | `[30, 40]` |
| Sea Level L4 | `adt` | meters | `[-2, 2]` |

The loader must treat code `255` as missing before normalization. Valid code
`0` is a real clipped value, not missing.

## Temperature Normalization

Training temperature tensors use the existing project normalization:

```text
normalized = (temperature_kelvin - 289.74267177946783) / 10.933397487585731
```

This is equivalent to converting decoded Kelvin to Celsius and calling
`temperature_normalize(mode="norm", ...)`, because that helper adds `273.15`
internally. A GeoTIFF loader can therefore either normalize directly from
Kelvin with the formula above or convert to Celsius first and reuse the helper.

After normalization:

- Missing `x`, `y`, and `eo` values are filled with `0.0`.
- `x_valid_mask`, `y_valid_mask`, and `x_valid_mask_1d` preserve which values
  were physically observed or supervised.
- Losses and metrics should use `y_valid_mask` so zero-filled invalid target
  pixels do not contribute.

## Temporal Contract

The GLORYS weekly date is the sample date. Dense rasters are expected to exist
for every exported date:

- `thetao` is the GLORYS weekly target for that date.
- `analysed_sst` is the centered 7-day OSTIA mean around that date.
- `adt` is the centered 7-day sea-level mean around that date.
- ARGO profiles are assigned to the nearest GLORYS weekly date inside the same
  temporal window.

The default validation split uses calendar year `2018`; all other years are
training rows. When patches overlap, keep a date-based split such as this to
avoid spatial train/validation leakage.
