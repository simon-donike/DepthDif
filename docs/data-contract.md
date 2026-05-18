# Data Contract  

This page defines the model-facing sample produced from the GeoTIFF workflow.  
The files on disk are byte-encoded rasters and a preprocessed ARGO profile  
store; the loader decodes them into physical units, normalizes temperature,  
and returns PyTorch tensors with the shapes below. Salinity tensors are  
returned only when `dataset.output.include_salinity=true`.  

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
batch dimension. Salinity side-channel keys are opt-in for joint  
temperature/salinity training.  

| Key | Item shape | Batch shape | Dtype | Meaning |  
| --- | ---: | ---: | --- | --- |  
| `eo` | `(1, H, W)` | `(B, 1, H, W)` | `float32` | Dense OSTIA surface temperature context, normalized. |  
| `x` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Sparse ARGO temperature observations, normalized and zero-filled where missing. |  
| `y` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Dense GLORYS `thetao` target, normalized and zero-filled where invalid. |  
| `x_salinity` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Opt-in sparse ARGO salinity observations, normalized and zero-filled where missing. |  
| `y_salinity` | `(D, H, W)` | `(B, D, H, W)` | `float32` | Opt-in dense GLORYS `so` salinity target, normalized and zero-filled where invalid. |  
| `x_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | True where `x` contains an observed ARGO temperature value. |  
| `y_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | True where the GLORYS target is valid ocean data. |  
| `x_salinity_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | Opt-in mask where `x_salinity` contains an observed ARGO salinity value. |  
| `y_salinity_valid_mask` | `(D, H, W)` | `(B, D, H, W)` | `bool` | Opt-in mask where the GLORYS salinity target is valid ocean data. |  
| `x_valid_mask_1d` | `(1, H, W)` | `(B, 1, H, W)` | `bool` | True where any ARGO temperature depth is present in that horizontal pixel. |  
| `x_salinity_valid_mask_1d` | `(1, H, W)` | `(B, 1, H, W)` | `bool` | Opt-in mask where any ARGO salinity depth is present in that horizontal pixel. |  
| `land_mask` | `(1, H, W)` | `(B, 1, H, W)` | `float32` | GLORYS spatial ocean/domain support; `1` where any temperature target depth is finite and `0` elsewhere. |  
| `date` | scalar | `(B,)` | integer | GLORYS target date as `YYYYMMDD`. |  
| `coords` | `(2,)` | `(B, 2)` | `float32` | Optional patch-center latitude and longitude. |  
| `info` | dictionary | list-like | metadata | Optional debugging metadata, not part of the training model input. |  

`x_valid_mask` is ARGO observation support, collapsed to one channel only when it is used as conditioning. `land_mask` is GLORYS-derived spatial ocean/domain support and gates the diffusion loss together with the task-valid mask; if GLORYS support is unavailable for mask construction, the loader falls back to finite OSTIA support and then the configured on-disk mask. Train/validation dataloaders do not return the common on-disk mask; callers may pass an optional `output_land_mask` directly to `predict_step` for final cleanup overlays. Training code should not infer missing values from zeros in `x`, `y`, optional `x_salinity`, optional `y_salinity`, or `eo`.  

## Salinity Opt-In  

Salinity is disabled by default in the active GeoTIFF data configs:  

```yaml
dataset:
  output:
    include_salinity: false
```

When this remains `false`, samples do not include `x_salinity`, `y_salinity`, or  
their salinity masks, and the GeoTIFF loader does not require/open `so` rasters  
or `argo_psal_*` zarr variables. Set it to `true` only for workflows that need  
salinity tensors in the batch:  

```yaml
dataset:
  output:
    include_salinity: true
```

The data flag only controls the sample contract. Joint training also requires  
`model.output_fields=["temperature", "salinity"]` and matching 100-output /  
103-condition channel counts on the model side.  

## Loading Steps  

For each selected `(patch, date)` row, the GeoTIFF loader should:  

1. Build a rasterio window from the shared land-mask grid.  
2. Read `rasters/glorys/thetao/thetao_YYYYMMDD.tif` as `(D, H, W)`.  
3. Read `rasters/ostia/analysed_sst/analysed_sst_YYYYMMDD.tif` as `(H, W)` and  
   add the leading channel dimension.  
4. If `dataset.output.include_salinity=true`, read  
   `rasters/glorys/so/so_YYYYMMDD.tif` as `(D, H, W)`.  
5. Query preprocessed ARGO profiles assigned to the same target date and patch  
   window.  
6. Rasterize ARGO temperature, plus salinity when enabled, onto `(D, H, W)`  
   using precomputed `grid_row` and `grid_col`; average duplicate observations  
   in the same depth/pixel cell.  
7. Build validity masks from GeoTIFF nodata codes and ARGO valid flags.  
8. Derive `land_mask` from finite GLORYS temperature support, with fallback to finite OSTIA support and then the configured on-disk mask.  
9. Normalize temperature, plus salinity when enabled, and replace NaN or  
   infinite normalized values with `0.0`.  

Sea-level `adt` and SSS `sos`/`dos` are exported on the same grid for
auxiliary experiments.
They should only be added to model inputs with an explicit model/config change
that defines channels and normalization.

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
| SSS | `sos` | PSU | `[30, 40]` |
| SSS | `dos` | kg/m3 | `[1000, 1035]` |

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

## Salinity Normalization

Training salinity tensors use the project salinity target statistics:

```text
normalized = (salinity_psu - 34.54260282159372) / 1.158266487751096
```

This is equivalent to calling `salinity_normalize(mode="norm", ...)`; use  
`salinity_normalize(mode="denorm", ...)` to recover physical PSU values.  
After normalization, missing `x_salinity` and `y_salinity` values are filled  
with `0.0`, while `x_salinity_valid_mask`, `y_salinity_valid_mask`, and  
`x_salinity_valid_mask_1d` preserve the physical support.

## Temporal Contract  

The GLORYS weekly date is the sample date. Dense rasters are expected to exist  
for every exported date:  

- `thetao` is the GLORYS weekly target for that date.  
- `analysed_sst` is the centered 7-day OSTIA mean around that date.  
- `adt` is the centered 7-day sea-level mean around that date.  
- SSS `sos` and `dos` are centered 7-day means around that date.
- ARGO profiles are assigned to the nearest GLORYS weekly date inside the same  
  temporal window.  

The default validation split uses calendar year `2018`; all other years are  
training rows. When patches overlap, keep a date-based split such as this to  
avoid spatial train/validation leakage.  
