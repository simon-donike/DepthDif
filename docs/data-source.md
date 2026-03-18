# Data Source  
DepthDif now uses two raw upstream sources:  
- Copernicus Marine monthly reanalysis for sub-surface temperatures (targets and sparse inputs)  
- OSTIA L4 SST for sea-surface EO conditioning  
  
Use this page for raw-source provenance and download details. See [Synthetic Dataset](data.md) for project-specific preprocessing and transformations.  
  
  
## Product A: Sub-Surface Reanalysis (Copernicus)  
- Provider: Copernicus Marine Service  
- Product family: Global Ocean Physics Reanalysis  
- Dataset used in this project: `global-reanalysis-001-030-monthly`  
- Source model: `MERCATOR GLORYS12V1`  
- Typical files: monthly NetCDF (`*.nc`)  
  
Reference dataset link:  
[Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1M-m_202311&path=GLOBAL_MULTIYEAR_PHY_001_030%2Fcmems_mod_glo_phy_my_0.083deg_P1M-m_202311%2F2024%2F)  
  
The Depth levels range from 0.4m to >5km, rigins almost exponentially due to the larger gradients existing in the upper alyers.  
![img](assets/depth_levels.png)  
  
  
## Download  
Example CLI from this project:  
`copernicusmarine get -i cmems_mod_glo_phy_my_0.083deg_P1M-m --filter "*YYYY/*"`  
  
Project helper scripts:  
- `data/get_glorys/download_glorys_monthly.sh` (monthly GLORYS downloads from `2010-01-01` to today by default)  
- `data/get_glorys/download_glorys_weekly.sh` (weekly-cadence archive built by downloading every 7th GLORYS daily file from `2010-01-01` to today by default into `/data1/datasets/depth_v2/glorys_weekly`)  
- `data/EDA_glorys_argo_alignment.py` (opens representative ARGO and GLORYS files, then saves the nearest ARGO-level -> GLORYS-level channel mapping into `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json`)  

Saved alignment artifact:  
- `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json`  
- this JSON stores, for each ARGO depth level index, the representative ARGO depth, the closest GLORYS level index, the matched GLORYS depth, and the absolute depth difference  
- intended use: downstream preprocessing can load this file and map GLORYS channels to ARGO-style target channels without recomputing the depth comparison every run  

Important EN4 depth-layout note:  
- EN4 stores profile depths in a rectangular `(N_PROF, N_LEVELS)` array where `N_LEVELS` is often `400`  
- that `400` value is the storage width of the file, not a promise that each individual profile measures 400 real depths  
- each profile only fills the depth slots it actually observed; the remaining slots are missing/fill values  
- the saved ARGO-to-GLORYS mapping therefore uses a representative depth per ARGO level index aggregated across all profiles in one representative EN4 file, rather than relying on one single observed profile  
  
## Product B: Surface EO SST (OSTIA).  
- Provider: Copernicus Marine Service / UKMO OSTIA stream  
- Dataset used in this project: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`  
- Files used here: daily OSTIA snapshots (`YYYYMMDD120000`) from `2010-01-01` onward  
- Variable used for conditioning: `analysed_sst` (sea-surface temperature)  
  
OSTIA dataset sample used in this project:  
![img](assets/dataset_ostia.png).  
  
Temporal note for current OSTIA overlap workflow:  
- previous depth workflow was monthly reanalysis composites  
- EO now comes from daily OSTIA snapshots (12:00 UTC), not monthly mean composites  
- overlap dataset keeps only months where both sources are available  
  
## Product C: EN4 Profile Observations (Argo/Other In-Situ)  
- Provider: UK Met Office Hadley Centre (EN4)  
- Dataset family: EN4.2.2 profile ZIP archives  
- Typical file naming: `EN.4.2.2.profiles.g10.YYYY.zip`  
- Example direct link:  
  `https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2/EN.4.2.2.profiles.g10.2019.zip`  
  
Project helper script:  
- `data/get_argo/download_en4_profiles.sh`  
- checks each year individually (dry-run HEAD check), logs CSV, and downloads immediately per available year  

Depth-grid interpretation note:  
- EN4 profile depths are stored in `DEPH_CORRECTED`, which is the corrected physical depth assigned to each temperature sample after EN4 processing  
- these are not one shared, evenly spaced ARGO depth levels that apply identically to every profile  
- instead, each profile contains its own valid observed depths, and different profiles can have different depth values and different numbers of valid depth samples  
- the rectangular EN4 storage shape `(N_PROF, 400)` should therefore be interpreted as file layout, not as a universal physical 400-level ocean grid  
- this is why a representative per-slot median depth curve can show local dips or non-monotonic sections: the slot index is a storage coordinate, while the physical depth varies across profiles  

Practical alignment consequence:  
- if ARGO should be the ground-truth source for model targets, the physically clean way to align ARGO and GLORYS is to interpolate each ARGO profile independently onto one shared target depth grid  
- in this project, the recommended common grid is the fixed GLORYS `depth` axis, because it is monotonic, physically meaningful, and already stable across files  
- this yields `ARGO-on-GLORYS-grid` as the ground-truth target and avoids treating raw EN4 storage-slot indices as if they were a true physical depth coordinate  

Depth-alignment figures generated from the saved mapping artifact:  
![img](assets/argo_glorys_depth_vs_index.png)  
![img](assets/argo_glorys_absolute_difference.png)  
![img](assets/argo_glorys_depth_scatter.png)  
![img](assets/argo_level_valid_profile_count.png)  

Example 3D ARGO profile visualization:  
![img](assets/argo_profile_3D.gif)  
  
## Data Contents (`data/data_info.txt`, Reanalysis Sample)  
The raw source file contains the following data variables (NetCDF variables):  
  
| Variable | Dimensions | Description | Units |  
|---|---|---|---|  
| `bottomT` | `(time, latitude, longitude)` | Sea floor potential temperature | `degrees_C` |  
| `mlotst` | `(time, latitude, longitude)` | Density ocean mixed layer thickness | `m` |  
| `zos` | `(time, latitude, longitude)` | Sea surface height | `m` |  
| `sithick` | `(time, latitude, longitude)` | Sea ice thickness | `m` |  
| `siconc` | `(time, latitude, longitude)` | Ice concentration | `1` |  
| `usi` | `(time, latitude, longitude)` | Sea ice eastward velocity | `m s-1` |  
| `vsi` | `(time, latitude, longitude)` | Sea ice northward velocity | `m s-1` |  
| `so` | `(time, depth, latitude, longitude)` | Salinity | `Practical Salinity Unit` |  
| `thetao` | `(time, depth, latitude, longitude)` | Temperature | `degrees_C` |  
| `uo` | `(time, depth, latitude, longitude)` | Eastward velocity | `m s-1` |  
| `vo` | `(time, depth, latitude, longitude)` | Northward velocity | `m s-1` |  
  
## Core Input Axes (Reanalysis Product)  
- `time`: monthly timestamp (single monthly slice per file in the inspected sample)  
- `latitude`: from `-80` to `90`  
- `longitude`: from `-180` to `179.9167`  
- `depth`: 50 vertical levels  
  
  
## Depth Levels (Meters)  
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
