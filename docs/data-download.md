# Dataset Downloads  

This page is split between two download paths:  

- **Raw upstream data**: original GLORYS, OSTIA, sea-level, SSS, EN4/ARGO, and land
  mask inputs used to build DepthDif training stores.  
- **Packaged DepthDif datasets**: prebuilt folders or archives that can be
  downloaded without rebuilding every intermediate from the upstream sources.

Use [Data Sources](data-source.md) for native product details and  
[Data Export](data-export.md) for conversion into trainable stores.  

All examples assume the project environment is available at  
`/work/envs/depth`. Raw NetCDF data is stored under:  

```bash
/data1/datasets/depth_v2
```

Raw and aligned intermediate products are normally stored under
`/data1/datasets/depth_v2`. The model-ready packaged dataset used by the GeoTIFF dataloader is normally stored
under the Hugging Face repo name:

```bash
/work/data/OceanVariableReconstruction
```

## Raw Upstream Data  

Raw download scripts write one CSV log in each output directory. Set  
`DRY_RUN_ONLY=1` to query availability without downloading files.  

### Copernicus Marine Inputs  

GLORYS, OSTIA, sea-level, and SSS files are downloaded with the
`copernicusmarine get` CLI. The scripts first run a dry query for each date and  
then download only matching NetCDF files.  

#### GLORYS  

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
  src/depth_recon/data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh \
  /data1/datasets/depth_v2/glorys_weekly
```

Daily and monthly helpers also exist for diagnostics or alternate exports:  

- `src/depth_recon/data/dataset_creation/data_download_raw/get_glorys/download_glorys_daily.sh`  
- `src/depth_recon/data/dataset_creation/data_download_raw/get_glorys/download_glorys_monthly.sh`  

#### OSTIA  

Role: daily sea-surface temperature input.  

Provider: Copernicus Marine Service / UKMO OSTIA stream.  

Product ID: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`.  

Dataset candidates used by the script:  

- `METOFFICE-GLO-SST-L4-REP-OBS-SST`  
- `METOFFICE-GLO-SST-L4-REP-OBS-SST-V2`  
- `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`  

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_ostia/download_ostia.sh \
  /data1/datasets/depth_v2/ostia
```

#### Sea Level  

Role: daily auxiliary sea-surface height field, currently exported as `adt`.  

Provider: Copernicus Marine Service.  

Product ID: `SEALEVEL_GLO_PHY_L4_MY_008_047`.  

Dataset ID:  
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`.  

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_sealevel/download_sealevel_daily.sh \
  /data1/datasets/depth_v2/sealevel_daily
```

#### Sea-Surface Salinity

Role: daily auxiliary sea-surface salinity, density, and ice fields.

Provider: Copernicus Marine Service / CNR.

Product ID: `MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013`.

Default dataset ID: `cmems_obs-mob_glo_phy-sss_my_multi_P1D`.

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  src/depth_recon/data/dataset_creation/data_download_raw/get_sss/download_sss_daily.sh \
  /data1/datasets/depth_v2/sss_daily
```

### EN4 / ARGO Profiles  

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
  src/depth_recon/data/dataset_creation/data_download_raw/get_argo/download_en4_profiles.sh \
  /data1/datasets/depth_v2/en4_profiles
```

### Land Mask  

The patch grid and GeoTIFF raster export use a global 0.1 degree land mask:  

```text
src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

It is derived by downloading a world GeoJSON file and rasterizing it to the  
GLORYS-style global grid where `1=land` and `0=water`:  

```bash
/work/envs/depth/bin/python \
  src/depth_recon/data/dataset_creation/data_download_raw/get_world/download_manipulate_world_file.py \
  --overwrite
```

The default GeoJSON source URL is stored in the script and can be overridden  
with `--source-url`.  

## Packaged DepthDif Datasets  

Packaged dataset downloaders live under  
`src/depth_recon/data/dataset_creation/data_download_packaged/`. Hosted dataset  
URLs are configured in  
`src/depth_recon/data/dataset_creation/data_download_packaged/dataset_links.yaml`.  

Use packaged downloads when you want the prepared DepthDif artifacts directly
instead of reconstructing them from the raw upstream products. The downloaders
mirror files from the official Hugging Face dataset repository into
`--output-dir`, reusing existing files unless `--force-download` is passed. Pass
`--overwrite` when existing package files should be replaced.

### Aligned ARGO Zarr  

Role: prealigned sparse ARGO/EN4 profile store, packaged as a Hugging Face
dataset folder with `data/argo_glors_ostia_ssh.zarr` and Parquet indices. This
package includes GLORYS, OSTIA, sea-level, and SSS profile-context variables.

Configured link key: `argo_aligned`.

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
  --output-dir /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh
```

The zarr path produced by this download is:

```text
/data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
```

Pass that path to `--enriched-argo-zarr` when exporting the GeoTIFF training
dataset from the packaged copy.

### Exported GeoTIFF Training Dataset  

Role: exported DepthDif training dataset with aligned dense rasters and gridded  
ARGO profile data, packaged in the official Hugging Face repository.  

Configured link key: `depthdif_training`.  

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
  --output-dir /work/data/OceanVariableReconstruction
```
