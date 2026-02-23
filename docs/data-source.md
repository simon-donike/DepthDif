# Data Source
DepthDif uses Copernicus Marine monthly ocean reanalysis as its raw input source.

## Product
- Provider: Copernicus Marine Service
- Product family: Global Ocean Physics Reanalysis
- Dataset used in this project: `global-reanalysis-001-030-monthly`
- Source model: `MERCATOR GLORYS12V1`
- Typical files: monthly NetCDF (`*.nc`)

Reference dataset link:  
[Global Ocean Physics Reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1M-m_202311&path=GLOBAL_MULTIYEAR_PHY_001_030%2Fcmems_mod_glo_phy_my_0.083deg_P1M-m_202311%2F2024%2F)

## Download
Example CLI from this project:
`copernicusmarine get -i cmems_mod_glo_phy_my_0.083deg_P1M-m --filter "*YYYY/*"`

## Data Contents (`data/data_info.txt`)
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
| `so` | `(time, depth, latitude, longitude)` | Salinity | `1e-3` |
| `thetao` | `(time, depth, latitude, longitude)` | Temperature | `degrees_C` |
| `uo` | `(time, depth, latitude, longitude)` | Eastward velocity | `m s-1` |
| `vo` | `(time, depth, latitude, longitude)` | Northward velocity | `m s-1` |

## Core Input Axes
- `time`: monthly timestamp (single monthly slice per file in the inspected sample)
- `latitude`: from `-80` to `90`
- `longitude`: from `-180` to `179.9167`
- `depth`: 50 vertical levels

Use this page for raw-source provenance and download details. See [Synthetic Dataset](data.md) for project-specific preprocessing and transformations.
