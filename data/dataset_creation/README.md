# Dataset Creation

This folder contains source-data download helpers and shared NetCDF source
metadata used by the active patch dataset.

Folder layout:

- `data_download_raw/`: source-specific scripts for downloading upstream
  EN4/ARGO, GLORYS, OSTIA, and sea-level NetCDF files.
- `data_download_packaged/`: placeholder for future packaged dataset downloads.
- `export_aligned_argo/`: aligned ARGO export workflow scripts, source variable
  names, and NetCDF source-file utilities used by
  `ArgoNetCDFGriddedPatchDataset`.
- `export_dataset_zarr/`: compact zarr export workflow for ML-friendly
  training sources containing only OSTIA SST, ARGO
  temperature/salinity, GLORYS temperature/salinity/SSH, and altimetry SSH.

The current default source root is:

```bash
/data1/datasets/depth_v2
```

Use the project Python environment explicitly:

```bash
/work/envs/depth/bin/python
```

## Download Source Data

Download OSTIA daily surface fields:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  data/dataset_creation/data_download_raw/get_ostia/download_ostia.sh \
  /data1/datasets/depth_v2/ostia
```

Download EN4 / ARGO profile archives:

```bash
START_YEAR=2010 END_YEAR=2025 \
  data/dataset_creation/data_download_raw/get_argo/download_en4_profiles.sh \
  /data1/datasets/depth_v2/en4_profiles
```

Download GLORYS files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 STEP_DAYS=7 \
  data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh \
  /data1/datasets/depth_v2/glorys
```

Download daily sea-level files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  data/dataset_creation/data_download_raw/get_sealevel/download_sealevel_daily.sh \
  /data1/datasets/depth_v2/sealevel_daily
```

The active dataset reads these NetCDF files directly and creates only compact
metadata caches under `dataset.core.metadata_cache_dir`.

## Export Compact Zarr Training Stores

To reduce disk footprint and speed loader access, export only the training
modalities configured in `export_dataset_zarr/source_variables.yaml` into zarr.
OSTIA, GLORYS, and sea-level rasters are resampled to 0.1 degrees by default:

```bash
/work/envs/depth/bin/python data/dataset_creation/export_dataset_zarr/export_dataset_zarr.py \
  --argo-dir /data1/datasets/depth_v2/en4_profiles \
  --glorys-dir /data1/datasets/depth_v2/glorys \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
  --output-dir /data1/datasets/depth_v2/zarr_training \
  --start-date 20100101 \
  --end-date 20240731 \
  --target-resolution-deg 0.1 \
  --overwrite
```

Then train with `configs/px_space/data_ostia_argo_zarr.yaml`, which selects
`dataset.core.dataset_variant: argo_zarr_gridded`.
