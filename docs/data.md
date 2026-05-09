# NetCDF Patch Dataset

DepthDif supports the raw NetCDF path `argo_netcdf_gridded` and a compact zarr
path `argo_zarr_gridded`. Both build patch samples lazily and keep the same
model-facing training keys.

Active config:

- `configs/px_space/data_ostia_argo_netcdf.yaml`
- `configs/px_space/data_ostia_argo_zarr.yaml`

Active implementation:

- `data/dataset_argo_netcdf_gridded.py`
- `ArgoNetCDFGriddedPatchDataset`
- `data/dataset_argo_zarr_gridded.py`
- `ArgoZarrGriddedPatchDataset`

## Sources

The dataset reads directly from local NetCDF roots configured under
`dataset.core`:

- `argo_dir`: EN4 / ARGO profile NetCDF files
- `glorys_dir`: GLORYS target NetCDF files
- `ostia_dir`: OSTIA surface SST NetCDF files
- `sealevel_dir`: sea-level NetCDF files, indexed for auxiliary metadata and
  diagnostics

The zarr variant reads compact stores exported by
`data/dataset_creation/export_dataset_zarr/export_dataset_zarr.py`:

- `ostia.zarr`: `analysed_sst` and optional `mask`
- `argo.zarr`: `TEMP`, `PSAL_CORRECTED`, GLORYS `depth`, and profile helpers
- `glorys.zarr`: `thetao`, `so`, `zos`
- `sealevel.zarr`: `adt` by default

By default, the exporter interpolates GLORYS raster stores to 0.1 degrees and
reprojects OSTIA plus sea-level rasters onto the exact same GLORYS
latitude/longitude grid before writing. OSTIA and sea-level daily files are
saved as centered 7-day aggregates around the GLORYS timesteps, matching the
weekly training cadence. This matches the default patch grid and lets the zarr
loader use exact grid selection for those rasters. Continuous fields are packed
to int16, masks are stored as int8, and ARGO profile variables are pre-projected
onto the GLORYS depth axis so the loader does not interpolate profiles at
training time.

Only compact cache files are allowed under `metadata_cache_dir`. These caches
store patch rows, split labels, land fractions, and ARGO support flags. They do
not store model-ready patch tensors.

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

When `dataset.output.return_modalities: true` is enabled for the zarr dataset,
additional raw modality tensors are returned in `sample["modalities"]` with
matching masks in `sample["modality_valid_masks"]`. Variable names are resolved
from the zarr data variables and config aliases instead of fixed file paths.

Public dataset metadata used by inference:

- `dataset.rows`
- `dataset.depth_axis_m`
- `ArgoNetCDFGriddedPatchDataset.from_config(...)`
- `ArgoZarrGriddedPatchDataset.from_config(...)`

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

The active config uses `split.val_year: 2018`, so every row dated in 2018 is
assigned to `val` and every other year is assigned to `train`.

When `split.val_year` is null, patch split labels are deterministic and use
`split.val_fraction` instead. Split labels are stored in the compact metadata
cache.

Selection flags control whether rows without ARGO support are retained:

- `split.val_year`
- `split.val_fraction`
- `dataset.selection.require_argo_for_train`
- `dataset.selection.require_argo_for_val`
- `dataset.selection.require_argo_for_all`
- `dataset.synthetic.enabled`
- `dataset.synthetic.pixel_count`

The default keeps ARGO-supported rows for train/val and permits no-ARGO rows for
full-grid inference.
