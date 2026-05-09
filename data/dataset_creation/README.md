# Dataset Creation

This folder contains the source-data acquisition, on-disk file validity checks,
and enriched ARGO profile export tools used before model-facing dataset loaders
consume the data.

Folder layout:

- `data_download_raw/`: source-specific scripts for downloading raw upstream
  EN4/ARGO, GLORYS, OSTIA, and sea-level files
- `data_download_packaged/`: placeholder for future packaged dataset downloads
  from Hugging Face or similar distribution channels
- `source_variables.yaml`: shared source variable names for EN4/ARGO, GLORYS,
  OSTIA, and sea-level inputs
- `source_files.py`: shared source-variable loading, source-file date parsing,
  and NetCDF open/index helpers used by the check and export scripts
- `a_check_export_sourcefiles.py`: raw source file validity checks and optional
  redownload/overwrite repair
- `b_export_enriched_argo_profiles.py`: enriched ARGO profile Zarr export

The current default source root is:

```bash
/data1/datasets/depth_v2
```

All commands should use the project Python environment explicitly:

```bash
/work/envs/depth/bin/python
```

## Shared Configuration

`source_variables.yaml` is the single editable list of upstream NetCDF
variables used by the dataset-creation tools. It groups source variable names
by product (`argo`, `glorys`, `ostia`, `sealevel`) and includes comments with
the expected physical meaning and units from representative source metadata.
Those comments are for humans; the scripts read only the variable-name lists.

`source_files.py` loads `source_variables.yaml` and exposes the ordered tuples
used by both scripts, including `ARGO_PROFILE_VARS`, `GLORYS_3D_VARS`,
`GLORYS_2D_VARS`, `OSTIA_VARS`, `SEALEVEL_VARS`, and `SOURCE_VARIABLES`. It
also owns the shared source-file mechanics: `TimedFile`, source date parsing,
ARGO month filtering, EN4/ARGO NetCDF opening, and gridded-file time indexing.

This keeps `a_check_export_sourcefiles.py` and
`b_export_enriched_argo_profiles.py` as sibling consumers of the same source
configuration instead of making the checker import implementation details from
the exporter.

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

Download weekly-stride GLORYS files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 STEP_DAYS=7 \
  data/dataset_creation/data_download_raw/get_glorys/download_glorys_weekly.sh \
  /data1/datasets/depth_v2/glorys_weekly
```

Download daily sea-level files:

```bash
START_DATE=2010-01-01 END_DATE=2024-07-31 \
  data/dataset_creation/data_download_raw/get_sealevel/download_sealevel_daily.sh \
  /data1/datasets/depth_v2/sealevel_daily
```

These shell scripts use `copernicusmarine` for Copernicus products when needed
and `/work/envs/depth/bin/python` for their helper parsing.

## Check File Validity

`a_check_export_sourcefiles.py` performs a general file validity check for
the source files that an enriched ARGO export would read. It opens every
selected NetCDF file, verifies required variables exist, and reads a small
sample from each required variable so truncated or unreadable files are
reported before the expensive export starts. The check uses `tqdm` progress
bars with file counts, processing rate, ETA, source kind, and broken-file
counts. The file header contains copy-pasteable CLI examples for the default
check, explicit source paths, source-group filtering, and repair mode. Required
variables come from `source_variables.yaml` through `source_files.py`.

Default check for the overlap window:

```bash
/work/envs/depth/bin/python data/dataset_creation/a_check_export_sourcefiles.py
```

Limit to a source group:

```bash
/work/envs/depth/bin/python data/dataset_creation/a_check_export_sourcefiles.py \
  --include argo \
  --start-date 20100101 \
  --end-date 20240731
```

Repair mode re-runs the same validity check, prints broken files, then asks for
confirmation before replacing anything:

```bash
/work/envs/depth/bin/python data/dataset_creation/a_check_export_sourcefiles.py \
  --repair
```

After typing `yes`, each broken file is redownloaded into a temporary directory,
validated there, and only then moved over the existing broken file.

## Export Enriched ARGO Profiles

`b_export_enriched_argo_profiles.py` builds a profile-level Zarr from raw EN4,
GLORYS, OSTIA, and sea-level NetCDF folders. It does not depend on the older
CSV manifest pipeline. It uses `tqdm` progress bars while scanning source
directories, walking ARGO source months, and collocating profiles so long runs
show throughput, queue/written counts, and ETA. The file header contains
copy-pasteable CLI examples for a production-range export and a small smoke
export.

For each EN4 profile, the exporter:

- projects `TEMP`, `POTM_CORRECTED`, and `PSAL_CORRECTED` onto the GLORYS depth
  axis
- carries optional EN4/ARGO QC flags into the output as `int8` codes, with
  profile-level QC copied per profile and depth-level QC aligned to the GLORYS
  depth axis
- samples GLORYS, OSTIA, and sea-level variables at the profile latitude,
  longitude, and time
- interpolates continuous time-varying fields between bracketing source files
- uses nearest temporal samples for categorical fields such as masks and flags
- records per-source temporal status values in the output Zarr

The output Zarr includes global and per-variable metadata extracted from the
source NetCDF files where available: product/provider notes, file counts and
date coverage, representative source filenames, source dimensions, source
variable dtype/dims/attrs, collocation rules, and processing notes. It stores
filenames only for source files; absolute local filesystem paths are
intentionally sanitized out of the metadata.

Source variable names are loaded from `source_variables.yaml` through
`source_files.py`, so the source-file checker and the enriched-profile exporter
validate and read the same configured fields without importing one script from
the other. Units in the YAML comments are orientation notes; the exporter
extracts the authoritative units and other attributes from the representative
source NetCDF files and stores them in the output Zarr metadata.

Recommended production-range export:

```bash
/work/envs/depth/bin/python data/dataset_creation/b_export_enriched_argo_profiles.py \
  --start-date 20100101 \
  --end-date 20240731 \
  --output-zarr /data1/datasets/depth_v2/enriched_argo_profiles.zarr
```

Use an explicit `--end-date` that matches the available OSTIA and sea-level
range. Running without a bounded end date may edge-fill later ARGO profiles with
the final available OSTIA and sea-level files.

Small smoke export:

```bash
/work/envs/depth/bin/python data/dataset_creation/b_export_enriched_argo_profiles.py \
  --start-date 20100101 \
  --end-date 20100101 \
  --max-profiles 4 \
  --batch-size 2 \
  --output-zarr /tmp/depthdif_enriched_argo_profiles_smoke.zarr \
  --overwrite
```
