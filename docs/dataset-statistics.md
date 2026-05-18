# Dataset Statistics

These statistics were measured from the local dataset root at
`/work/data/depthdif`. The page is split into two parts:

1. the exported enriched ARGO Zarr profile dataset
2. the saved GeoTIFF ML dataset used by the patch dataloader

Note: the exporter now also writes SSS fields. The measured counts below
predate a full local re-export with those fields unless the corresponding Zarr
or manifest timestamp has been regenerated after the SSS change. Schema tables
list the current expected outputs.

## 1. Exported ARGO Zarr Dataset

Path:

```text
/work/data/depthdif/enriched_argo_profiles.zarr
```

This Zarr is the profile-level export. Each row is one EN4/ARGO profile that
passed the date and coordinate filters. ARGO values are projected onto the
GLORYS depth axis, and GLORYS, OSTIA, sea-level, and SSS fields are sampled at
the same profile location and date.

The export was created at `2026-05-10T17:54:01+00:00` for the requested date
range `20100101` to `20240731`.

### Summary

| Item | Value |
| --- | ---: |
| Profile rows | 6,637,381 |
| GLORYS depth levels | 50 |
| Profile dates | 20100101 to 20240731 |
| Unique profile dates | 5,142 |
| Source EN4/ARGO files | 169 |
| Profiles with valid temperature | 6,609,159 |
| Profiles with valid potential temperature | 6,609,159 |
| Profiles with valid salinity | 5,394,516 |
| Valid temperature depth points | 156,319,955 |
| Valid potential-temperature depth points | 156,319,955 |
| Valid salinity depth points | 133,248,365 |

### Source Products

| Source | Files | First file/date | Last file/date |
| --- | ---: | --- | --- |
| EN4/ARGO profiles | 169 | `EN.4.2.2.f.profiles.g10.201001.nc` | `EN.4.2.2.f.profiles.g10.202407.nc` |
| GLORYS | 843 | 20100101 | 20260220 |
| OSTIA | 5,326 | 20100101 | 20240731 |
| Sea level | 5,326 | 20100101 | 20240731 |
| SSS | 5,326 expected after full re-export | 20100101 | 20240731 |

### Dimensions

| Dimension | Meaning | Count |
| --- | --- | ---: |
| `profile` | One ARGO profile location/date. | 6,637,381 |
| `glorys_depth` | Native GLORYS depth axis used for vertical alignment. | 50 |

### Variables

Variables with shape `profile x glorys_depth` store one value per ARGO profile
and GLORYS depth level. Variables with shape `profile` store one value per
profile.

| Variable | Explanation | Shape | Type |
| --- | --- | --- | --- |
| `latitude` | Profile latitude in degrees north. | `profile` | `float64` |
| `longitude` | Profile longitude in degrees east. | `profile` | `float64` |
| `profile_date` | Profile observation date as `YYYYMMDD`. | `profile` | `int64` |
| `profile_juld` | Original EN4/ARGO Julian day timestamp. | `profile` | `float64` |
| `profile_idx` | Profile index in the exported Zarr. | `profile` | `int64` |
| `profile_source_file` | Source EN4/ARGO file name. | `profile` | `<U33` |
| `valid_observed_depth_count` | Number of valid observed depth levels in the source profile. | `profile` | `int64` |
| `argo_temp_on_glorys_depth` | ARGO in-situ temperature interpolated to GLORYS depth levels. | `profile x glorys_depth` | `float32` |
| `argo_temp_valid_on_glorys_depth` | True where interpolated ARGO temperature is valid. | `profile x glorys_depth` | `bool` |
| `argo_temp_qc_on_glorys_depth` | ARGO temperature QC code on GLORYS depth levels. | `profile x glorys_depth` | `int8` |
| `argo_potm_on_glorys_depth` | ARGO potential temperature interpolated to GLORYS depth levels. | `profile x glorys_depth` | `float32` |
| `argo_potm_valid_on_glorys_depth` | True where interpolated ARGO potential temperature is valid. | `profile x glorys_depth` | `bool` |
| `argo_potm_qc_on_glorys_depth` | ARGO potential-temperature QC code on GLORYS depth levels. | `profile x glorys_depth` | `int8` |
| `argo_psal_on_glorys_depth` | ARGO practical salinity interpolated to GLORYS depth levels. | `profile x glorys_depth` | `float32` |
| `argo_psal_valid_on_glorys_depth` | True where interpolated ARGO salinity is valid. | `profile x glorys_depth` | `bool` |
| `argo_psal_qc_on_glorys_depth` | ARGO salinity QC code on GLORYS depth levels. | `profile x glorys_depth` | `int8` |
| `argo_depth_qc_on_glorys_depth` | ARGO depth QC code projected to GLORYS depth levels. | `profile x glorys_depth` | `int8` |
| `argo_juld_qc` | Whole-profile date/time QC code. | `profile` | `int8` |
| `argo_position_qc` | Whole-profile position QC code. | `profile` | `int8` |
| `argo_profile_depth_qc` | Whole-profile depth QC code. | `profile` | `int8` |
| `argo_profile_potm_qc` | Whole-profile potential-temperature QC code. | `profile` | `int8` |
| `argo_profile_psal_qc` | Whole-profile salinity QC code. | `profile` | `int8` |
| `glorys_thetao` | GLORYS sea-water potential temperature sampled at the profile point. | `profile x glorys_depth` | `float32` |
| `glorys_so` | GLORYS salinity sampled at the profile point. | `profile x glorys_depth` | `float32` |
| `glorys_uo` | GLORYS eastward sea-water velocity sampled at the profile point. | `profile x glorys_depth` | `float32` |
| `glorys_vo` | GLORYS northward sea-water velocity sampled at the profile point. | `profile x glorys_depth` | `float32` |
| `glorys_zos` | GLORYS sea-surface height sampled at the profile point. | `profile` | `float32` |
| `glorys_mlotst` | GLORYS mixed-layer thickness sampled at the profile point. | `profile` | `float32` |
| `glorys_bottomT` | GLORYS sea-floor potential temperature sampled at the profile point. | `profile` | `float32` |
| `glorys_sithick` | GLORYS sea-ice thickness sampled at the profile point. | `profile` | `float32` |
| `glorys_siconc` | GLORYS sea-ice area fraction sampled at the profile point. | `profile` | `float32` |
| `glorys_usi` | GLORYS eastward sea-ice velocity sampled at the profile point. | `profile` | `float32` |
| `glorys_vsi` | GLORYS northward sea-ice velocity sampled at the profile point. | `profile` | `float32` |
| `glorys_temporal_status` | Status of GLORYS temporal matching. | `profile` | `int8` |
| `ostia_analysed_sst` | OSTIA analysed sea-surface temperature sampled at the profile point. | `profile` | `float32` |
| `ostia_analysis_error` | OSTIA SST analysis error sampled at the profile point. | `profile` | `float32` |
| `ostia_sea_ice_fraction` | OSTIA sea-ice fraction sampled at the profile point. | `profile` | `float32` |
| `ostia_mask` | OSTIA categorical mask sampled at the profile point. | `profile` | `float32` |
| `ostia_temporal_status` | Status of OSTIA temporal matching. | `profile` | `int8` |
| `sealevel_sla` | Sea-level anomaly sampled at the profile point. | `profile` | `float32` |
| `sealevel_err_sla` | Sea-level anomaly formal mapping error. | `profile` | `float32` |
| `sealevel_adt` | Absolute dynamic topography sampled at the profile point. | `profile` | `float32` |
| `sealevel_ugosa` | Eastward geostrophic velocity anomaly. | `profile` | `float32` |
| `sealevel_err_ugosa` | Formal mapping error for eastward velocity anomaly. | `profile` | `float32` |
| `sealevel_vgosa` | Northward geostrophic velocity anomaly. | `profile` | `float32` |
| `sealevel_err_vgosa` | Formal mapping error for northward velocity anomaly. | `profile` | `float32` |
| `sealevel_ugos` | Absolute eastward geostrophic velocity. | `profile` | `float32` |
| `sealevel_vgos` | Absolute northward geostrophic velocity. | `profile` | `float32` |
| `sealevel_flag_ice` | Sea-level product ice flag. | `profile` | `float32` |
| `sealevel_tpa_correction` | TOPEX-A instrumental drift correction field. | `profile` | `float32` |
| `sealevel_temporal_status` | Status of sea-level temporal matching. | `profile` | `int8` |
| `sss_sos` | SSS analysed sea-surface salinity sampled at the profile point. | `profile` | `float32` |
| `sss_dos` | SSS analysed sea-surface density sampled at the profile point. | `profile` | `float32` |
| `sss_sea_ice_fraction` | SSS sea-ice fraction sampled at the profile point. | `profile` | `float32` |
| `sss_temporal_status` | Status of SSS temporal matching. | `profile` | `int8` |

The raw SSS error variables `sos_error` and `dos_error` are intentionally not
exported to the enriched ARGO Zarr. Temporal status fields describe how the
exporter chose the source file for each profile date; they do not describe
spatial missingness, land masks, ice masks, or source QC flags.

| Code | Meaning | Interpretation |
| ---: | --- | --- |
| `0` | `nearest_or_exact` | The exporter found an exact source date, or the target date was inside the source time range and the nearest available file was used. This is the normal status. |
| `1` | `nearest_edge` | The target date was outside the available source time range, so the exporter used the first or last available source file. Treat this as an edge extrapolation warning. |
| `2` | `missing` | No usable source file existed for that modality, so the sampled values for that source were written as missing values. |

## 2. Saved GeoTIFF ML Dataset

Path:

```text
/work/data/depthdif
```

This is the model-ready dataset. It stores dense fields as aligned GeoTIFF
rasters and uses a compact, grid-indexed ARGO Zarr store for the patch
dataloader. The active config is
`src/depth_recon/configs/px_space/training_super_config.yaml`.

The GeoTIFF export manifest was created at `2026-05-12T11:28:47+00:00` and
covers weekly target dates from `20100101` to `20240726`.

### Summary

| Item | Value |
| --- | ---: |
| Target dates (weeks) | 761 |
| Grid size | 3600 x 1800 pixels |
| Grid resolution | 0.1 degrees |
| CRS | EPSG:4326 |
| GLORYS depth levels | 50 |
| Compact ARGO profiles | 6,608,517 |
| Compact ARGO profiles with valid temperature | 6,608,321 |
| Compact ARGO profiles with valid salinity | 5,393,686 |
| Valid compact ARGO temperature depth points | 156,290,112 |
| Valid compact ARGO salinity depth points | 133,218,721 |

### Compact ARGO Profile Store

Path:

```text
/work/data/depthdif/argo/argo_profiles_on_grid.zarr
```

This store is smaller than the enriched ARGO export because it keeps only the
profile fields needed by the GeoTIFF patch loader.

| Dimension | Meaning | Count |
| --- | --- | ---: |
| `profile` | ARGO profile assigned to the GeoTIFF grid. | 6,608,517 |
| `glorys_depth` | GLORYS depth level. | 50 |

| Variable | Explanation | Shape | Type |
| --- | --- | --- | --- |
| `argo_temp_kelvin_uint8` | ARGO temperature values, quantized in Kelvin. | `profile x glorys_depth` | `uint8` |
| `argo_temp_valid` | True where the temperature value is valid. | `profile x glorys_depth` | `bool` |
| `argo_psal_uint8` | ARGO practical salinity values, quantized. | `profile x glorys_depth` | `uint8` |
| `argo_psal_valid` | True where the salinity value is valid. | `profile x glorys_depth` | `bool` |
| `profile_date` | Date of the original ARGO profile observation. | `profile` | `int32` |
| `target_date` | Weekly GLORYS target date assigned to the profile. | `profile` | `int32` |
| `latitude` | Profile latitude in degrees north. | `profile` | `float32` |
| `longitude` | Profile longitude in degrees east. | `profile` | `float32` |
| `grid_row` | Row index of the nearest GeoTIFF grid cell. | `profile` | `int32` |
| `grid_col` | Column index of the nearest GeoTIFF grid cell. | `profile` | `int32` |
| `profile_source_file` | Source EN4/ARGO file name. | `profile` | `<U33` |
| `source_profile_idx` | Profile index inside the source file. | `profile` | `int32` |

| Field | First | Last | Unique target dates |
| --- | ---: | ---: | ---: |
| ARGO profile dates | 20100101 | 20240729 | - |
| Assigned target dates | 20100101 | 20240726 | 736 |

### GeoTIFF Rasters

All raster files share the same grid: 3600 x 1800 pixels, EPSG:4326, stored as
`uint8`.

| Modality | Variable | Files | Bands per file | First date | Last date |
| --- | --- | ---: | ---: | ---: | ---: |
| GLORYS | `thetao` | 761 | 50 | 20100101 | 20240726 |
| GLORYS | `so` | 761 | 50 | 20100101 | 20240726 |
| OSTIA | `analysed_sst` | 761 | 1 | 20100101 | 20240726 |
| Sea level | `adt` | 761 | 1 | 20100101 | 20240726 |
| SSS | `sos` | 761 expected after full re-export | 1 | 20100101 | 20240726 |
| SSS | `dos` | 761 expected after full re-export | 1 | 20100101 | 20240726 |

### Patch Dataset

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

### Overlap

Coverage multiplicity is the number of selected spatial patches covering a
grid pixel before the date dimension is applied.

| Statistic | Value |
| --- | ---: |
| Patch size | 128 px / 12.8 degrees |
| Patch stride | 32 px / 3.2 degrees |
| Nominal overlap per axis | 96 px / 9.6 degrees |
| Nominal overlap fraction per axis | 75% |
| Selected spatial patches | 3,547 |
| Covered grid pixels | 4,883,456 |
| Coverage multiplicity min | 1 |
| Coverage multiplicity median | 15 |
| Coverage multiplicity mean | 11.90 |
| Coverage multiplicity max | 20 |
