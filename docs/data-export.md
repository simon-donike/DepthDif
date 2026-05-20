# Data Export

This page describes the GeoTIFF training dataset exported from the downloaded
raw products. Use [Dataset Downloads](data-download.md) for acquisition and
[Depth Alignment](depth-alignment.md) for the ARGO-to-GLORYS vertical
projection.

The export goal is a directory that can be shared or copied as one training
dataset: dense fields are stored as aligned GeoTIFF rasters, and sparse ARGO
profiles are stored in a compact grid-indexed profile store. The aligned ARGO
profile zarr is also packageable as a Hugging Face dataset folder before it is
used by the GeoTIFF export.
Training can select this exported dataset with
`src/depth_recon/configs/px_space/training_super_config.yaml`, which sets
`data.dataset.core.dataset_variant: argo_geotiff_gridded`.

## Aligned ARGO and Hugging Face Layout

`b_export_enriched_argo_profiles.py` creates the enriched profile-level zarr by
aligning EN4/ARGO profiles with GLORYS depth levels and sampling GLORYS, OSTIA,
sea-level, and SSS context at each profile location. By default, it writes:

```text
/data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr
```

`c_package_huggingface_aligned_argo.py` repackages that same zarr into a
Hugging Face-ready folder without changing the zarr schema:

```text
/data1/datasets/depth_v2/aligned_argo/
  enriched_argo_profiles.zarr
  hf_argo_glors_ostia_ssh/
    README.md
    LICENSE
    data/
      argo_glors_ostia_ssh.zarr/
    indices/
      profiles.parquet
      variables.parquet
    examples/
      open_with_xarray.py
      subset_by_region_time.py
    metadata/
      dataset_description.json
      citation.cff
      stac-item.json
```

The GeoTIFF exporter can read either the direct zarr path or the packaged zarr
path:

```text
/data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
```

## Output Layout

Default GeoTIFF export root:

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
    sss/
      sos/sos_YYYYMMDD.tif
      dos/dos_YYYYMMDD.tif
  argo/
    argo_profiles_on_grid.zarr
```

Each `YYYYMMDD` is a GLORYS weekly target date. Files for the same date share
the same CRS, transform, width, height, pixel centers, and nodata convention.

The export includes GLORYS `so` rasters and ARGO `argo_psal_*` variables so the
same on-disk dataset can support salinity-only and joint temperature/salinity
experiments. The GeoTIFF dataloader reads and returns those salinity fields when
the resolved scenario is `salinity` or `joint`. Daily SSS `sos` and `dos`
rasters are also exported for auxiliary experiments.

## Spatial Grid

The land-mask GeoTIFF is the authoritative grid:

```text
src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

The export uses that file for:

- CRS and affine transform
- raster width and height
- 0.1 degree pixel centers
- patch-grid compatibility with training and inference

GLORYS, OSTIA, sea-level, and SSS data are read onto this exact grid before
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

### SSS Surface Salinity and Density

Path:

```text
rasters/sss/<variable>/<variable>_YYYYMMDD.tif
```

Contents:

- Source variables: `sos`, `dos`
- Stored physical units after decoding: PSU for `sos`, kg/m3 for `dos`
- File structure: one single-band GeoTIFF per variable and GLORYS weekly date
- Temporal aggregation: centered 7-day mean around the GLORYS date by default

## Variables in the Export Products

The enriched ARGO zarr and its Hugging Face package contain the same variables.
The package adds Parquet indices for discovery only; those indices do not
replace the zarr arrays.

| Product | Variables |
| --- | --- |
| ARGO profile coordinates and provenance | `latitude`, `longitude`, `profile_date`, `profile_juld`, `profile_idx`, `profile_source_file`, `valid_observed_depth_count` |
| ARGO profiles on GLORYS depths | `argo_temp_on_glorys_depth`, `argo_potm_on_glorys_depth`, `argo_psal_on_glorys_depth`, matching `*_valid_on_glorys_depth` masks, matching `*_qc_on_glorys_depth` codes, `argo_depth_qc_on_glorys_depth`, whole-profile QC fields |
| GLORYS profile context | `glorys_thetao`, `glorys_so`, `glorys_uo`, `glorys_vo`, `glorys_zos`, `glorys_mlotst`, `glorys_bottomT`, `glorys_sithick`, `glorys_siconc`, `glorys_usi`, `glorys_vsi`, `glorys_temporal_status` |
| OSTIA profile context | `ostia_analysed_sst`, `ostia_analysis_error`, `ostia_sea_ice_fraction`, `ostia_mask`, `ostia_temporal_status` |
| Sea-level profile context | `sealevel_sla`, `sealevel_err_sla`, `sealevel_adt`, `sealevel_ugosa`, `sealevel_err_ugosa`, `sealevel_vgosa`, `sealevel_err_vgosa`, `sealevel_ugos`, `sealevel_vgos`, `sealevel_flag_ice`, `sealevel_tpa_correction`, `sealevel_temporal_status` |
| SSS profile context | `sss_sos`, `sss_dos`, `sss_sea_ice_fraction`, `sss_temporal_status` |
| HF Parquet indices | `indices/profiles.parquet` stores scalar profile metadata and valid-depth counts; `indices/variables.parquet` stores variable names, dimensions, dtypes, and descriptions. |
| GeoTIFF dense rasters | `thetao`, `so`, `analysed_sst`, `adt`, `sos`, `dos` |
| GeoTIFF compact ARGO zarr | `argo_temp_kelvin_uint8`, `argo_psal_uint8`, `argo_temp_valid`, `argo_psal_valid`, `profile_date`, `target_date`, `latitude`, `longitude`, `grid_row`, `grid_col`, optional source identifiers |

The raw SSS error variables `sos_error` and `dos_error` are not exported to the
enriched ARGO zarr or the GeoTIFF training store. Daily SSS `sos` and `dos` are
included as profile context in the aligned ARGO zarr and as dense surface
rasters in the GeoTIFF export.

## ARGO Profile Store

Path:

```text
argo/argo_profiles_on_grid.zarr
```

The exporter reads the enriched ARGO profile input by default:

```text
/data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr
```

The packaged Hugging Face zarr can be used instead:

```text
/data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
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
| Density | `[1000, 1035] kg/m3` | `0.1378 kg/m3` | `0.0689 kg/m3` | `0.2778 kg/m3` | `0.1389 kg/m3` |
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
loader which dates exist, how to decode each byte raster, which SSS rasters are
present, and which grid/depth axes the exported arrays use.

## Command

Run from the repository root:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff \
  --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
  --sss-dir /data1/datasets/depth_v2/sss_daily \
  --enriched-argo-zarr /data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr \
  --land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
  --output-dir /work/data/depthdif \
  --start-date 20100101 \
  --end-date 20240731 \
  --surface-aggregate-days 7 \
  --workers 4 \
  --overwrite
```

`--workers` controls process-level parallelism for dense raster dates. Lower it
if source-disk contention or memory pressure becomes the bottleneck. Use
`--skip-existing` instead of `--overwrite` to resume a partial export: existing
modality/date rasters are validated and reused, and missing rasters are written.
