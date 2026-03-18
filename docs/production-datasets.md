# Production Dataset
This page documents the production dataset assembly pipeline: how the raw products are sampled in space and time, how the indexing CSVs are built, and what the current dataset versions look like.

Use [Data Sources](data-source.md) for raw-product details and [Depth Alignment](depth-alignment.md) for vertical-grid handling and ARGO-to-GLORYS alignment logic.

## Scope
The production workflow has three jobs:
1. build a fixed spatial patch grid from OSTIA coverage
2. expand that grid into a daily or multi-day temporal index
3. attach ARGO availability and profile linkage metadata to each `(patch, time)` row

## Source Inputs Required
Expected source trees:
- `/data1/datasets/depth_v2/ostia`
- `/data1/datasets/depth_v2/en4_profiles`
- optional aligned GLORYS helpers in `/data1/datasets/depth_v2/glorys_weekly`

Main build scripts:
- `data/get_ostia/build_ostia_patch_time_index.py`
- `data/get_argo/build_argo_datetime_match_index.py`
- `data/get_argo/merge_argo_into_ostia_daily_index.py`

## 1) Spatial Sampling
The fixed patch geometry is derived from OSTIA because OSTIA defines the surface EO coverage.

Patch-grid logic:
- infer or set the pixel resolution
- choose a tile size
- classify each spatial tile as valid or invalid from OSTIA valid-pixel coverage
- split valid water tiles into `train` and `val`

Current commonly used setup:
- pixel resolution: `0.1°`
- tile size: `128 x 128`
- physical span: `12.8° x 12.8°`

Recommended native-grid OSTIA patch-index build:

```bash
/work/envs/depth/bin/python data/get_ostia/build_ostia_patch_time_index.py \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --output-spatial-csv /data1/datasets/depth_v2/ostia_patch_index_spatial.csv \
  --output-daily-csv /data1/datasets/depth_v2/ostia_patch_index_daily.csv \
  --tile-size 128 \
  --invalid-threshold 0.2 \
  --val-fraction 0.15 \
  --split-seed 7 \
  --valid-mask-values 1
```

Spatial split visualization:
![OSTIA train val land patch map](assets/train_val_split_0p1.png)

## 2) Temporal Sampling
After the spatial patch grid exists, the production dataset expands it across time.

### Daily Sampling
Daily OSTIA index rows are:
- one row per `(patch, day)`
- each row contains:
  - patch bounds
  - split label
  - OSTIA file path for that day

ARGO is then attached date-wise:
- EN4 `JULD` is converted to `YYYYMMDD`
- each ARGO profile day is matched to the nearest OSTIA day in the same month
- that match metadata is merged into the daily patch CSV

Datetime match command:

```bash
/work/envs/depth/bin/python data/get_argo/build_argo_datetime_match_index.py \
  --argo-dir /data1/datasets/depth_v2/en4_profiles \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --output-csv /data1/datasets/depth_v2/argo_profile_datetime_match.csv
```

Merge command:

```bash
/work/envs/depth/bin/python data/get_argo/merge_argo_into_ostia_daily_index.py \
  --daily-csv /data1/datasets/depth_v2/ostia_patch_index_daily.csv \
  --argo-match-csv /data1/datasets/depth_v2/argo_profile_datetime_match.csv \
  --output-csv /data1/datasets/depth_v2/ostia_patch_index_daily.csv
```

### Multi-Day Temporal Windows
The dataset layer can also aggregate a centered time window instead of one single day.

Current behavior in the production loader:
- `days=1` keeps single-day behavior
- `days=7` aggregates a seven-day temporal window around the target row date
- the output stays one spatial sample with aggregated observations, not `7x` stacked temporal channels

Interpretation:
- single observations may contribute to multiple neighboring daily rows when they fall inside the configured temporal window
- this increases observation density without changing the fixed output tensor shape

## 3) Production Index Files
Core CSV artifacts:
- `ostia_patch_index_spatial.csv`
- `ostia_patch_index_daily.csv`
- `argo_profile_datetime_match.csv`

Important path-format rule:
- stored paths are anchored at `depth_v2/...` rather than absolute machine-specific paths
- this keeps the dataset tree relocatable

Columns typically added during ARGO merge:
- `argo_valid`
- `argo_profile_count`
- `argo_month_key`
- `argo_file_path`

## 4) Current Dataset Versions
Snapshot numbers below were measured/recomputed on **March 10, 2026**.

Shared source inventory:
- OSTIA daily files (`ostia/*.nc`): `5,326`
- EN4 monthly profile files (`en4_profiles/*.nc`): `186`
- daily-index date range: `2010-01-01` to `2024-07-31`

### Version A: 0.05 Degree, Daily
Files:
- `ostia_patch_index_spatial.csv`
- `ostia_patch_index_daily.csv`

Counts:
- spatial patches: `751`
- spatial split: `638 train`, `113 val`
- daily rows: `3,999,826`
- daily split rows: `3,397,988 train`, `601,838 val`
- rows with `argo_valid=1`: `3,153,449` (`78.84%`)
- rows with `argo_valid=0`: `846,377` (`21.16%`)

### Version B: 0.1 Degree, Daily
Files:
- `ostia_patch_index_spatial_0p1_recomputed.csv`
- `ostia_patch_index_daily_0p1_recomputed_merged.csv`

Counts:
- spatial patches: `175`
- spatial split: `149 train`, `26 val`
- daily rows: `932,050`
- daily split rows: `793,574 train`, `138,476 val`
- rows with `argo_valid=1`: `734,825` (`78.84%`)
- rows with `argo_valid=0`: `197,225` (`21.16%`)

Sample Image:
![OSTIA 0.1 deg sample](assets/argo_ostia_sample_5.png)

Train-set valid-fraction histogram:
![OSTIA 0.1 deg hist](assets/argo_observations_histogram.png)

Train-set patch map:
![OSTIA 0.1 deg map](assets/argo_observations_map.png)

### Version C: 0.1 Degree, 7-Day Aggregate
Interpretation:
- the same spatial grid as Version B
- temporal support expanded to a centered 7-day window
- observations contribute to multiple neighboring rows when they overlap in time

Counts:
- patch-day rows: `932,050`
- spatial split: `149 train`, `26 val`
- daily split rows: `793,574 train`, `138,476 val`
- rows without any ARGO observations after aggregation: about `1.5%`
- average observations for valid tiles: about `20.5`

Sample Image:
![OSTIA 0.1 deg sample](assets/argo_ostia_sample_0p1_7days.png)

Train-set valid-fraction histogram:
![OSTIA 0.1 deg hist](assets/argo_observations_histogram_7days.png)

Train-set patch map:
![OSTIA 0.1 deg map](assets/argo_observations_map_7days.png)

## 5) End-To-End Build Order
1. Download OSTIA daily files with `data/get_ostia/download_ostia.sh`.
2. Download and extract EN4 profile data with `data/get_argo/download_en4_profiles.sh`.
3. Build the OSTIA spatial and daily patch index.
4. Build the ARGO <-> OSTIA datetime match table.
5. Merge ARGO validity and linkage columns into the daily patch CSV.
6. Choose the dataset version:
   - daily rows
   - or multi-day temporal windows at load time

## 6) Practical Interpretation
This page intentionally focuses on:
- where the patches come from
- how often they are sampled
- how train/val splits are defined
- how ARGO and OSTIA are linked in time

It intentionally does not define the physical vertical alignment between ARGO and GLORYS. That logic lives in [Depth Alignment](depth-alignment.md), because it is a raw-source and resampling question rather than a spatial/temporal sampling question.
