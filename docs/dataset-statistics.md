# Dataset Statistics

These statistics were measured from the local exported GeoTIFF dataset at
`/work/data/depthdif` with the active GeoTIFF data config:
`src/depth_recon/configs/px_space/data_ostia_argo_geotiff.yaml`.

The export manifest was created at `2026-05-12T11:28:47+00:00` and covers
weekly target dates from `20100101` to `20240726`.

## Summary

| Item | Value |
| --- | ---: |
| Target dates (Weeks) | 761 |
| Grid size | 3600 x 1800 pixels |
| Grid resolution | 0.1 degrees |
| CRS | EPSG:4326 |
| GLORYS depth levels | 50 |
| ARGO profiles in exported profile store | 6,608,517 |
| ARGO profiles with valid temperature | 6,608,321 |
| ARGO profiles with valid salinity | 5,393,686 |
| Valid ARGO temperature depth points | 156,290,112 |
| Valid ARGO salinity depth points | 133,218,721 |

## ARGO Profile Store

Path:

```text
/work/data/depthdif/argo/argo_profiles_on_grid.zarr
```

The profile store has dimensions:

| Dimension | Count |
| --- | ---: |
| `profile` | 6,608,517 |
| `glorys_depth` | 50 |

Stored variables:

| Variable | Explanation | Shape | Type |
| --- | --- | --- | --- |
| `argo_temp_kelvin_uint8` | ARGO temperature values, quantized in Kelvin. | `profile x glorys_depth` | `uint8` |
| `argo_temp_valid` | True where the temperature value is an observed valid point. | `profile x glorys_depth` | `bool` |
| `argo_psal_uint8` | ARGO practical salinity values, quantized. | `profile x glorys_depth` | `uint8` |
| `argo_psal_valid` | True where the salinity value is an observed valid point. | `profile x glorys_depth` | `bool` |
| `profile_date` | Date of the original ARGO profile observation. | `profile` | `int32` |
| `target_date` | Weekly GLORYS target date the profile was assigned to. | `profile` | `int32` |
| `latitude` | Profile latitude in degrees north. | `profile` | `float32` |
| `longitude` | Profile longitude in degrees east. | `profile` | `float32` |
| `grid_row` | Row index of the nearest raster grid cell. | `profile` | `int32` |
| `grid_col` | Column index of the nearest raster grid cell. | `profile` | `int32` |
| `profile_source_file` | Source EN4/ARGO file name for traceability. | `profile` | `<U33` |
| `source_profile_idx` | Profile index inside the source file. | `profile` | `int32` |

Date coverage:

| Field | First | Last | Unique target dates |
| --- | ---: | ---: | ---: |
| ARGO profile dates | 20100101 | 20240729 | - |
| Assigned target dates | 20100101 | 20240726 | 736 |

## GeoTIFF Rasters

All raster files share the same grid: 3600 x 1800 pixels, EPSG:4326, stored as
`uint8`.

| Modality | Variable | Files | Bands per file | First date | Last date |
| --- | --- | ---: | ---: | ---: | ---: |
| GLORYS | `thetao` | 761 | 50 | 20100101 | 20240726 |
| GLORYS | `so` | 761 | 50 | 20100101 | 20240726 |
| OSTIA | `analysed_sst` | 761 | 1 | 20100101 | 20240726 |
| Sea level | `adt` | 761 | 1 | 20100101 | 20240726 |

## Patch Dataset

The active GeoTIFF loader uses 128 x 128 pixel patches with a 32-pixel stride.
At 0.1 degrees per pixel, each patch is 12.8 x 12.8 degrees and neighboring
patch starts are 3.2 degrees apart.

| Split | Rows | Unique spatial patches | Dates | Rows with ARGO |
| --- | ---: | ---: | ---: | ---: |
| `all` | 2,699,267 | 3,547 | 761 | 2,384,535 |
| `train` | 2,214,599 | 3,544 | 684 | 2,214,599 |
| `val` | 169,936 | 3,391 | 52 | 169,936 |

The configured split uses `2018` as validation year. Training and validation
rows require at least one valid ARGO temperature profile; the `all` split keeps
patch/date rows even when no ARGO profile is present.

**ARGO profile support per patch/date row**

Each value is the number of ARGO profiles inside one spatial patch for one
weekly date. `all` has a minimum of `0` because it keeps patch/date rows without
ARGO support; `train` and `val` require at least one valid ARGO temperature
profile, so their minimum is `1`.

| Split | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0 | 13 | 25.32 | 8,064 |
| `train` | 1 | 15 | 28.24 | 8,064 |
| `val` | 1 | 16 | 34.19 | 4,580 |

Land fraction per selected patch/date row:

| Split | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.0000 | 0.0001 | 0.0469 | 0.5994 |
| `train` | 0.0000 | 0.0001 | 0.0420 | 0.5994 |
| `val` | 0.0000 | 0.0001 | 0.0430 | 0.5994 |

## Overlap

| Statistic | Value |
| --- | ---: |
| Patch size | 128 px / 12.8 degrees |
| Patch stride | 32 px / 3.2 degrees |
| Nominal overlap per axis | 96 px / 9.6 degrees |
| Nominal overlap fraction per axis | 75% |
| Selected spatial patches | 3,547 |