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
