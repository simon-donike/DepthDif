# NetCDF Patch Dataset

DepthDif now uses one model-facing dataset path: `argo_netcdf_gridded`.
It builds patch samples lazily from source NetCDF files and writes no patch
tensors, NumPy exports, or dataset GeoTIFF manifests.

Active config:

- `configs/px_space/data_ostia_argo_netcdf.yaml`

Active implementation:

- `data/dataset_argo_netcdf_gridded.py`
- `ArgoNetCDFGriddedPatchDataset`

## Sources

The dataset reads directly from local NetCDF roots configured under
`dataset.core`:

- `argo_dir`: EN4 / ARGO profile NetCDF files
- `glorys_dir`: GLORYS target NetCDF files
- `ostia_dir`: OSTIA surface SST NetCDF files
- `sealevel_dir`: sea-level NetCDF files, indexed for auxiliary metadata and
  diagnostics

Only compact cache files are allowed under `metadata_cache_dir`. These caches
store source file indexes, patch rows, split labels, land fractions, and ARGO
support flags. They do not store model-ready patch tensors.

## Sample Contract

Each dataset item keeps the current training and inference batch contract:

- `eo`: `(1, H, W)` normalized OSTIA `analysed_sst`
- `x`: `(D, H, W)` normalized sparse ARGO temperature projected onto GLORYS depth
- `y`: `(D, H, W)` normalized GLORYS `thetao`
- `x_valid_mask`: `(D, H, W)` ARGO support mask
- `y_valid_mask`: `(D, H, W)` GLORYS finite-value support mask
- `x_valid_mask_1d`: `(1, H, W)` horizontal ARGO support mask
- `land_mask`: `(1, H, W)` horizontal land/ocean mask
- `date`: integer `YYYYMMDD`
- optional `coords`
- optional `info`

Public dataset metadata used by inference:

- `dataset.rows`
- `dataset.depth_axis_m`
- `ArgoNetCDFGriddedPatchDataset.from_config(...)`

Rows expose stable fields: `patch_id`, `date`, `lat0`, `lat1`, `lon0`, `lon1`,
and `split`.

## Patch Assembly

For each `(patch, date)` row, the loader:

1. Reads GLORYS `thetao` over the patch and depth axis to form `y`.
2. Reads OSTIA `analysed_sst`, converts Kelvin values to Celsius when needed,
   and interpolates to the patch grid to form `eo`.
3. Finds ARGO profiles inside the configured spatial bounds and temporal window.
4. Projects each ARGO `TEMP` profile from `DEPH_CORRECTED` depths onto the
   GLORYS depth axis.
5. Rasterizes profile hits into the patch; duplicate hits in the same
   pixel/depth cell are averaged.
6. Builds validity masks from finite support and returns normalized tensors.

Empty-ARGO rows are kept only when the split selection allows them, such as
global inference with `require_argo_for_all: false`.

## Synthetic GLORYS Input Option

For controlled ablations, the same NetCDF dataset can build sparse `x` from
GLORYS instead of ARGO:

```yaml
dataset:
  synthetic:
    enabled: true
    pixel_count: 250
```

When enabled, `x` is copied from `y` at `pixel_count` randomly selected
horizontal pixels per patch. The full valid GLORYS depth profile is kept at each
selected pixel, and every other `x` location is zeroed after normalization with
`x_valid_mask=false`. Sampling is deterministic per `(random_seed, patch_id,
date, row index)`.

Synthetic mode does not require ARGO support for split filtering because the
sparse input is produced from GLORYS.

## Normalization And Masks

Temperature tensors are loaded in degrees Celsius and normalized through
`utils.normalizations.temperature_normalize`.

Mask semantics:

- `x_valid_mask` marks observed ARGO support after profile-depth alignment.
- `y_valid_mask` marks valid GLORYS target support.
- `x_valid_mask_1d` collapses ARGO support across depth for conditioning and
  visualization.
- `land_mask` is horizontal and derived from target support.

## Split Behavior

Patch split labels are deterministic and stored in the compact metadata cache.
Selection flags control whether rows without ARGO support are retained:

- `dataset.selection.require_argo_for_train`
- `dataset.selection.require_argo_for_val`
- `dataset.selection.require_argo_for_all`
- `dataset.synthetic.enabled`
- `dataset.synthetic.pixel_count`

The default keeps ARGO-supported rows for train/val and permits no-ARGO rows for
full-grid inference.
