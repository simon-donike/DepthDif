# Dataset Creation

This folder contains source-data download helpers and shared NetCDF source
metadata used by the active patch dataset.

Folder layout:

- `data_download_raw/`: source-specific scripts for downloading upstream
  EN4/ARGO, GLORYS, OSTIA, SSS, and sea-level NetCDF files.
- `data_download_packaged/`: packaged dataset download and extraction helpers.
- `export_aligned_argo/`: aligned ARGO export workflow scripts, source variable
  names, and NetCDF source-file utilities used by the GeoTIFF export pipeline.
- `export_dataset_geotiff/`: aligned uint8 GeoTIFF export workflow for dense
  GLORYS, OSTIA, sea-level, and SSS rasters plus a compact grid-indexed ARGO
  profile zarr.

The current default source root is:

```bash
./data/raw
```

Use the project Python environment explicitly:

```bash
/work/envs/depth/bin/python
```

## Download Source Data

Download OSTIA daily surface fields:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_ostia/download_ostia.sh \
  ./data/raw/ostia
```

Download EN4 / ARGO profile archives:

```bash
START_YEAR=2010 END_YEAR=2025 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_argo/download_en4_profiles.sh \
  ./data/raw/en4_profiles
```

Download GLORYS files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 STEP_DAYS=7 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh \
  ./data/raw/glorys
```

Download daily sea-level files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_sealevel/download_sealevel_daily.sh \
  ./data/raw/sealevel_daily
```

Download daily sea-surface salinity files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_sss/download_sss_daily.sh \
  ./data/raw/sss_daily
```

The current pixel training path first exports these sources into the GeoTIFF store used by `training_super_config.yaml`.

## Export GeoTIFF Raster Training Stores

The GeoTIFF workflow writes dense gridded fields as one uint8 raster per
variable/date on the land-mask grid, and writes ARGO profiles as a compact
profile-indexed zarr with precomputed target date, grid row/column, temperature,
salinity, and validity masks. Temperature stretches decode to Kelvin.
The GeoTIFF dataloader keeps temperature in the existing `x`/`y` keys. The pixel
scenario resolver sets `output.fields`; `--scenario salinity` skips temperature and
returns `x_salinity`, `y_salinity`, and their validity masks, while `--scenario joint`
returns both field groups. Use `salinity_normalize(..., mode="denorm")` to recover
physical PSU values. The dataloader does not concatenate variables;
The selected baseline stacks fields according to the resolved scenario.
The authoritative land-mask GeoTIFF is copied into `masks/` in the export root
and recorded in `manifest.yaml`.
Dense raster dates are exported with process workers by default; use `--workers`
to tune CPU and RAM use for the machine.

Raster values use the full unsigned byte range for accuracy: valid codes are
`0..254`, `255` is nodata, and decoding is
`minimum + code / 254 * (maximum - minimum)`. The same transform metadata is
stored in GeoTIFF tags and `manifest.yaml`, including quantization step and
worst-case rounding error.

| Variable family | Stretch | uint8 step | uint8 max error | int8 nonnegative step | int8 nonnegative max error |
| --- | --- | ---: | ---: | ---: | ---: |
| Temperature | `[270.15, 308.15] K` | `0.1496 K` | `0.0748 K` | `0.3016 K` | `0.1508 K` |
| Salinity | `[30, 40] PSU` | `0.0394 PSU` | `0.0197 PSU` | `0.0794 PSU` | `0.0397 PSU` |
| Density | `[1000, 1035] kg/m3` | `0.1378 kg/m3` | `0.0689 kg/m3` | `0.2778 kg/m3` | `0.1389 kg/m3` |
| Sea height `adt` | `[-2, 2] m` | `0.0157 m` | `0.0079 m` | `0.0317 m` | `0.0159 m` |

The int8 comparison assumes a signed-byte layout that only uses nonnegative
codes `0..126` plus nodata `127`. A signed int8 remapped across all 255
non-nodata codes would have the same precision as uint8, but the unsigned layout
keeps the transform simpler and interoperates better with raster tooling.

By default, the working dataset root is `./data/ocean_depth_reconstruction`. The ARGO
exporter owns both ARGO Zarr products: the full enriched profile-level Zarr and
the compact grid-indexed Zarr consumed by the GeoTIFF dataloader:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles \
  --argo-dir ./data/raw/en4_profiles \
  --glorys-dir ./data/raw/glorys_weekly \
  --ostia-dir ./data/raw/ostia \
  --sealevel-dir ./data/raw/sealevel_daily \
  --sss-dir ./data/raw/sss_daily \
  --output-zarr ./data/ocean_depth_reconstruction/enriched_argo_profiles.zarr \
  --compact-output-zarr ./data/ocean_depth_reconstruction/argo/argo_profiles_on_grid.zarr \
  --compact-land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
  --start-date 20100101 \
  --end-date 20240731 \
  --workers 4 \
  --overwrite
```

The GeoTIFF exporter owns dense raster products. Use `--rasters-only` when the
compact ARGO Zarr has already been written by the ARGO exporter:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff \
  --glorys-dir ./data/raw/glorys_weekly \
  --ostia-dir ./data/raw/ostia \
  --sealevel-dir ./data/raw/sealevel_daily \
  --sss-dir ./data/raw/sss_daily \
  --land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
  --output-dir ./data/ocean_depth_reconstruction \
  --start-date 20100101 \
  --end-date 20240731 \
  --surface-aggregate-days 7 \
  --workers 4 \
  --rasters-only \
  --overwrite
```

Use `--skip-existing` instead of `--overwrite` to resume a partial GeoTIFF export
without rewriting existing modality/date rasters.

## Package Ocean Depth Reconstruction Dataset for Hugging Face

After ARGO and raster exports complete, assemble a self-contained upload folder
with root-level `rasters/` using:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo \
  --input-zarr ./data/ocean_depth_reconstruction/enriched_argo_profiles.zarr \
  --raster-root ./data/ocean_depth_reconstruction/rasters \
  --compact-argo-zarr ./data/ocean_depth_reconstruction/argo/argo_profiles_on_grid.zarr \
  --manifest-path ./data/ocean_depth_reconstruction/manifest.yaml \
  --masks-dir ./data/ocean_depth_reconstruction/masks \
  --output-dir /work/data/OceanDepthReconstruction \
  --zarr-name argo_glors_ostia_ssh.zarr \
  --file-mode copy \
  --overwrite
```

The upload root contains `rasters/`, `argo/argo_profiles_on_grid.zarr`,
`data/argo_glors_ostia_ssh.zarr`, `manifest.yaml`, `masks/`, Parquet indices,
examples, metadata, local `assets/` for the Hugging Face dataset card,
`README.md`, and `LICENSE`. SSS variables are included in the enriched Zarr as
`sss_sos`, `sss_dos`, `sss_sea_ice_fraction`, and `sss_temporal_status`.
