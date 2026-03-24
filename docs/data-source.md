# Data Sources
This page documents the native properties of the upstream products used in the project.

Use [Depth Alignment](depth-alignment.md) for ARGO-to-GLORYS vertical resampling and [Production Dataset](production-datasets.md) for spatial and temporal dataset assembly.

## Overview
| Source | Role In Project | Native Sampling | Key Variables |
|---|---|---|---|
| GLORYS | 3D ocean reanalysis field | global gridded field, fixed 50 depth levels | `thetao`, `depth` |
| OSTIA | daily surface temperature field | daily global 2D grid | `analysed_sst` |
| EN4 / ARGO profiles | in-situ temperature observations | profile-specific corrected depths | `TEMP`, `DEPH_CORRECTED` |

## Product A: GLORYS Reanalysis
- Provider: Copernicus Marine Service
- Product family: Global Ocean Physics Reanalysis
- Source model used here: `MERCATOR GLORYS12V1`
- Native variables used here: `thetao`, `depth`
- Vertical coordinate: 50 fixed depth levels from roughly `0.494 m` to `5727.917 m`

Relevant helper scripts:
- `data/get_glorys/download_glorys_monthly.sh`
- `data/get_glorys/download_glorys_weekly.sh`

GLORYS depth distribution:
![img](assets/depth_levels.png)


## Product B: OSTIA Surface EO
- Provider: Copernicus Marine Service / UKMO OSTIA stream
- Dataset used here: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`
- Native variable used here: `analysed_sst`
- Temporal sampling: one daily SST snapshot at `12:00 UTC`
- Geometry: global 2D grid

OSTIA example tile:
![img](assets/dataset_ostia.png)

## Product C: EN4 / ARGO Profiles
- Provider: UK Met Office Hadley Centre EN4
- Dataset family: EN4.2.2 profile archives
- Raw variables used here: `TEMP`, `DEPH_CORRECTED`
- Profile storage: rectangular arrays with capacity for up to `400` profile samples
- Vertical sampling: corrected depths vary profile-by-profile and are irregular in physical depth

Relevant helper script:
- `data/get_argo/download_en4_profiles.sh`

Archive-wide corrected-depth histogram with GLORYS reference levels:
![img](assets/argo_corrected_depth_histogram.png)

- The histogram aggregates finite `DEPH_CORRECTED` values over the scanned EN4 archive.
- Sampling density is concentrated in the upper ocean and decreases with depth.
- The dotted GLORYS depth markers show the discrete reference levels used elsewhere in the project.

Example ARGO profile in 3D:
![img](assets/argo_profile_3D.gif)

## Raw Product Notes
Representative raw GLORYS variables:
| Variable | Dimensions | Description | Units |
|---|---|---|---|
| `bottomT` | `(time, latitude, longitude)` | Sea floor potential temperature | `degrees_C` |
| `mlotst` | `(time, latitude, longitude)` | Mixed layer thickness | `m` |
| `zos` | `(time, latitude, longitude)` | Sea surface height | `m` |
| `so` | `(time, depth, latitude, longitude)` | Salinity | `Practical Salinity Unit` |
| `thetao` | `(time, depth, latitude, longitude)` | Temperature | `degrees_C` |
| `uo` | `(time, depth, latitude, longitude)` | Eastward velocity | `m s-1` |
| `vo` | `(time, depth, latitude, longitude)` | Northward velocity | `m s-1` |
