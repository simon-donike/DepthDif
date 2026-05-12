# Raw Data Download

This page documents how DepthDif downloads the upstream raw datasets before any
alignment, packing, or training export is applied. Use
[Data Sources](data-source.md) for native product details and
[Data Export](data-export.md) for conversion into trainable stores.

All examples assume the project environment is available at
`/work/envs/depth` and raw NetCDF data is stored under:

```bash
/data1/datasets/depth_v2
```

Download scripts write one CSV log in each output directory. Set
`DRY_RUN_ONLY=1` to query availability without downloading files.

## Copernicus Marine Inputs

GLORYS, OSTIA, and sea-level files are downloaded with the
`copernicusmarine get` CLI. The scripts first run a dry query for each date and
then download only matching NetCDF files.

### GLORYS

Role: weekly 3D reanalysis target and salinity/source fields.

Provider: Copernicus Marine Service.

Product family: Global Ocean Physics Reanalysis / GLORYS12V1. The weekly helper
downloads one daily file every 7 days because the catalogue exposes daily and
monthly streams, not a dedicated weekly stream.

Dataset candidates used by the script:

- `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- `cmems_mod_glo_phy_my_0.083deg_P1D-m_202311`
- `global-reanalysis-phy-001-030-daily`

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 STEP_DAYS=7 \
  data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh \
  /data1/datasets/depth_v2/glorys_weekly
```

Daily and monthly helpers also exist for diagnostics or alternate exports:

- `data/dataset_creation/data_download_raw/get_glorys/download_glorys_daily.sh`
- `data/dataset_creation/data_download_raw/get_glorys/download_glorys_monthly.sh`

### OSTIA

Role: daily sea-surface temperature input.

Provider: Copernicus Marine Service / UKMO OSTIA stream.

Product ID: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`.

Dataset candidates used by the script:

- `METOFFICE-GLO-SST-L4-REP-OBS-SST`
- `METOFFICE-GLO-SST-L4-REP-OBS-SST-V2`
- `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  data/dataset_creation/data_download_raw/get_ostia/download_ostia.sh \
  /data1/datasets/depth_v2/ostia
```

### Sea Level

Role: daily auxiliary sea-surface height field, currently exported as `adt`.

Provider: Copernicus Marine Service.

Product ID: `SEALEVEL_GLO_PHY_L4_MY_008_047`.

Dataset ID:
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`.

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  data/dataset_creation/data_download_raw/get_sealevel/download_sealevel_daily.sh \
  /data1/datasets/depth_v2/sealevel_daily
```

## EN4 / ARGO Profiles

Role: sparse in-situ temperature and salinity profiles.

Provider: UK Met Office Hadley Centre EN4.

The helper downloads yearly EN4 profile ZIP archives from:

```text
https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1
```

Expected archive names follow:

```text
EN.4.2.2.profiles.g10.YYYY.zip
```

```bash
START_YEAR=2010 END_YEAR=2025 \
  data/dataset_creation/data_download_raw/get_argo/download_en4_profiles.sh \
  /data1/datasets/depth_v2/en4_profiles
```

## Land Mask

The patch grid and GeoTIFF raster export use a global 0.1 degree land mask:

```text
data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

It is derived by downloading a world GeoJSON file and rasterizing it to the
GLORYS-style global grid where `1=land` and `0=water`:

```bash
/work/envs/depth/bin/python \
  data/dataset_creation/data_download_raw/get_world/download_manipulate_world_file.py \
  --overwrite
```

The default GeoJSON source URL is stored in the script and can be overridden
with `--source-url`.
