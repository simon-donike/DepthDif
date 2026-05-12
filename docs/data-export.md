# Data Export

This page describes the GeoTIFF training dataset exported from the downloaded
raw products. Use [Raw Data Download](data-download.md) for acquisition and
[Depth Alignment](depth-alignment.md) for the ARGO-to-GLORYS vertical
projection.

The export goal is a directory that can be shared or copied as one training
dataset: dense fields are stored as aligned GeoTIFF rasters, and sparse ARGO
profiles are stored in a compact grid-indexed profile store.
Training can select this exported dataset with
`configs/px_space/data_ostia_argo_geotiff.yaml`, which sets
`dataset.core.dataset_variant: argo_geotiff_gridded`.

## Output Layout

Default export root:

```text
/work/data/depthdif
```

Output files:

```text
/work/data/depthdif/
  manifest.yaml
  rasters/
    glorys/
      thetao/thetao_YYYYMMDD.tif
      so/so_YYYYMMDD.tif
    ostia/
      analysed_sst/analysed_sst_YYYYMMDD.tif
    sealevel/
      adt/adt_YYYYMMDD.tif
  argo/
    argo_profiles_on_grid.zarr
```

Each `YYYYMMDD` is a GLORYS weekly target date. Files for the same date share
the same CRS, transform, width, height, pixel centers, and nodata convention.

## Spatial Grid

The land-mask GeoTIFF is the authoritative grid:

```text
data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

The export uses that file for:

- CRS and affine transform
- raster width and height
- 0.1 degree pixel centers
- patch-grid compatibility with training and inference

GLORYS, OSTIA, and sea-level data are read onto this exact grid before
quantization. If source coordinates already match the requested pixel centers,
the exporter uses exact nearest-coordinate selection; otherwise it interpolates
onto the target axes.

## Dense Raster Products

### GLORYS Temperature

Path:

```text
rasters/glorys/thetao/thetao_YYYYMMDD.tif
```

Contents:

- Source variable: `thetao`
- Source units: Celsius in GLORYS source files
- Stored physical units after decoding: Kelvin
- File structure: one multiband GeoTIFF per weekly date
- Bands: one band per GLORYS depth level
- Band metadata: `depth_m`, stretch information, clipped counts, nodata counts

GLORYS weekly files are exported as-is for their target date. They are not
averaged across the week.

### GLORYS Salinity

Path:

```text
rasters/glorys/so/so_YYYYMMDD.tif
```

Contents:

- Source variable: `so`
- Stored physical units after decoding: PSU
- File structure: one multiband GeoTIFF per weekly date
- Bands: one band per GLORYS depth level
- Band metadata: `depth_m`, stretch information, clipped counts, nodata counts

### OSTIA Surface Temperature

Path:

```text
rasters/ostia/analysed_sst/analysed_sst_YYYYMMDD.tif
```

Contents:

- Source variable: `analysed_sst`
- Stored physical units after decoding: Kelvin
- File structure: one single-band GeoTIFF per GLORYS weekly date
- Temporal aggregation: centered 7-day mean around the GLORYS date by default

OSTIA values are normally Kelvin. The exporter preserves Kelvin values and only
adds `273.15` when the source values look Celsius-like.

### Sea Level

Path:

```text
rasters/sealevel/adt/adt_YYYYMMDD.tif
```

Contents:

- Source variable: `adt`
- Stored physical units after decoding: meters
- File structure: one single-band GeoTIFF per GLORYS weekly date
- Temporal aggregation: centered 7-day mean around the GLORYS date by default

## ARGO Profile Store

Path:

```text
argo/argo_profiles_on_grid.zarr
```

The exporter reads the enriched ARGO profile input by default:

```text
/work/data/depthdif/aligned_argo/enriched_argo_profiles.zarr
```

The saved profile store contains the profile information needed by a GeoTIFF
loader without redoing expensive profile preprocessing:

- `profile_date`
- `target_date`
- `latitude`
- `longitude`
- `grid_row`
- `grid_col`
- `argo_temp_kelvin_uint8`
- `argo_psal_uint8`
- `argo_temp_valid`
- `argo_psal_valid`
- optional source profile identifiers

Profiles are assigned to the nearest GLORYS target date inside the centered
weekly window and to the nearest raster grid cell. Temperature is converted to
Kelvin before quantization.

## Quantization

All dense rasters are written as tiled `uint8` GeoTIFFs with ZSTD compression
when available, otherwise DEFLATE. `BIGTIFF=IF_SAFER` is enabled.

Encoding:

```text
0..254 = valid stretched values
255    = nodata
decoded = minimum + code / 254 * (maximum - minimum)
```

The transform, units, valid range, clipped-low count, clipped-high count,
nodata count, quantization step, and worst-case rounding error are written to
GeoTIFF tags and to `manifest.yaml`.

| Variable family | Stretch | uint8 step | uint8 max error | int8 nonnegative step | int8 nonnegative max error |
| --- | --- | ---: | ---: | ---: | ---: |
| Temperature | `[270.15, 308.15] K` | `0.1496 K` | `0.0748 K` | `0.3016 K` | `0.1508 K` |
| Salinity | `[30, 40] PSU` | `0.0394 PSU` | `0.0197 PSU` | `0.0794 PSU` | `0.0397 PSU` |
| Sea height `adt` | `[-2, 2] m` | `0.0157 m` | `0.0079 m` | `0.0317 m` | `0.0159 m` |

The int8 comparison assumes a signed-byte encoding that uses only `0..126` as
valid values and `127` as nodata. A signed int8 layout remapped across all 255
non-nodata codes would have the same quantization error as `uint8`, but `uint8`
keeps byte values and nodata handling explicit for raster readers.

## Manifest

`manifest.yaml` records the export configuration and output metadata:

- creation script and UTC timestamp
- requested date range
- land-mask grid source, CRS, transform, shape, and resolution
- target weekly dates
- GLORYS depth axis
- per-variable stretch metadata
- per-file paths, source files, compression, and encode statistics
- ARGO profile count and ARGO stretch metadata

The manifest is the stable entry point for downstream loaders: it tells the
loader which dates exist, how to decode each byte raster, and which grid/depth
axes the exported arrays use.

## Command

Run from the repository root:

```bash
/work/envs/depth/bin/python \
  data/dataset_creation/export_dataset_geotiff/export_dataset_geotiff.py \
  --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
  --enriched-argo-zarr /work/data/depthdif/aligned_argo/enriched_argo_profiles.zarr \
  --land-mask-path data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
  --output-dir /work/data/depthdif \
  --start-date 20100101 \
  --end-date 20240731 \
  --surface-aggregate-days 7 \
  --workers 4 \
  --overwrite
```

`--workers` controls process-level parallelism for dense raster dates. Lower it
if source-disk contention or memory pressure becomes the bottleneck.
