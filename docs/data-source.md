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

### GLORYS Depth Levels
| Level | Depth (m) |
|---:|---:|
| 0 | 0.494 |
| 1 | 1.541 |
| 2 | 2.646 |
| 3 | 3.819 |
| 4 | 5.078 |
| 5 | 6.441 |
| 6 | 7.930 |
| 7 | 9.573 |
| 8 | 11.405 |
| 9 | 13.467 |
| 10 | 15.810 |
| 11 | 18.496 |
| 12 | 21.599 |
| 13 | 25.211 |
| 14 | 29.445 |
| 15 | 34.434 |
| 16 | 40.344 |
| 17 | 47.374 |
| 18 | 55.764 |
| 19 | 65.807 |
| 20 | 77.854 |
| 21 | 92.326 |
| 22 | 109.729 |
| 23 | 130.666 |
| 24 | 155.851 |
| 25 | 186.126 |
| 26 | 222.475 |
| 27 | 266.040 |
| 28 | 318.127 |
| 29 | 380.213 |
| 30 | 453.938 |
| 31 | 541.089 |
| 32 | 643.567 |
| 33 | 763.333 |
| 34 | 902.339 |
| 35 | 1062.440 |
| 36 | 1245.291 |
| 37 | 1452.251 |
| 38 | 1684.284 |
| 39 | 1941.893 |
| 40 | 2225.078 |
| 41 | 2533.336 |
| 42 | 2865.703 |
| 43 | 3220.820 |
| 44 | 3597.032 |
| 45 | 3992.484 |
| 46 | 4405.224 |
| 47 | 4833.291 |
| 48 | 5274.784 |
| 49 | 5727.917 |

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
