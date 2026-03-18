# Production Datasets  
This page documents the non-toy datasets currently used in DepthDif:  
- OSTIA L4 reprocessed daily sea-surface temperature (EO condition)  
- EN4.2.2 profile archives (submarine/in-situ profile observations)  
- GLORYS weekly profile archive plus the saved ARGO-to-GLORYS depth-channel mapping used to align vertical levels  
  
## 1) OSTIA L4 Reprocessed (EO Surface Condition)  
Source:  
- Copernicus Marine product: `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`  
- Dataset ID used for download: `METOFFICE-GLO-SST-L4-REP-OBS-SST`  
  
Coverage used:  
- Daily files from `2010-01-01` to `2024-07-31`  
- Global grid at 0.05 degree  
  
Filename structure:  
- `YYYYMMDD120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc`  
- Example: `20100206120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc`  
  
Logical structure in each NetCDF:  
- One daily time slice (12:00 UTC snapshot)  
- 2D global fields on latitude/longitude grid  
- Main variable used here: `analysed_sst`  
  
Download workflow in this repo:  
- Script: `data/get_ostia/download_ostia.sh`  
- Behavior:  
  - Checks each day (dry-run availability)  
  - Immediately downloads that day if available  
  - Writes CSV log for each day (`filename,path,datetime,status`)  
  - Prints progress and ETA  
  
Current folder size (snapshot on March 10, 2026):  
- `/data1/datasets/depth_v2/ostia`: `86,859,298,331` bytes (`~80.89 GiB`)  
  
### Building the OSTIA Patch Dataset (0.05 degree grid)  
Patch-level train/val indexing is built from the downloaded OSTIA files using:  
- Script: `data/get_ostia/build_ostia_patch_time_index.py`  
- Input: daily OSTIA NetCDF files in `ostia_dir`  
- Logic:  
  - infer native OSTIA resolution (0.05 degree when using default inference)  
  - build fixed-size spatial patches on that grid  
  - classify each patch as `invalid`/`train`/`val` from invalid-pixel fraction  
  - expand spatial patches to daily `(patch, day)` rows  
- Outputs:  
  - spatial patch metadata CSV (`ostia_patch_index_spatial.csv`)  
  - daily expanded CSV (`ostia_patch_index_daily.csv`)  
  
Grid and patch setup used in current runs:  
- pixel resolution: `0.05°`  
- patch size: `128 x 128` pixels  
- physical patch span: `128 * 0.05° = 6.4°` per axis  
- temporal aggregation level: daily (`one row per (patch, day)` in `ostia_patch_index_daily.csv`)  
  
Recommended command (native OSTIA resolution):  
  
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
  
Notes for this command:  
- if `--resolution-deg` is omitted, native OSTIA spacing is inferred from file coordinates (`~0.05°`)  
- patch geographic span is `tile_size * resolution_deg` per axis  
- invalid patches are excluded by default (`train`/`val` only); use `--include-invalid` to keep them  
  
Patch split visualization (blue = train, red = val, number show fraction of land pixels) for 0.1 deg resolution and 128x128 patch size:  
  
![OSTIA train val land patch map](assets/train_val_split_0p1.png)  
  
Source portal:  
- <https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011>  
  
## 2) EN4.2.2 Profiles (Argo + Other In-Situ)  
Source:  
- UK Met Office Hadley Centre EN4 page:  
  <https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html>  
  
Coverage used:  
- Yearly profile archives from `2010` onward  
- Files are annual ZIPs  
  
Filename structure:  
- `EN.4.2.2.profiles.g10.YYYY.zip`  
- Example: `EN.4.2.2.profiles.g10.2022.zip`  
  
Direct URL structure:  
- `https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2.profiles.g10.YYYY.zip`  
  
Archive content structure (high-level):  
- One ZIP per year containing profile observation files for that year  
- Includes Argo and other in-situ profile sources used in EN4  
  
Download workflow in this repo:  
- Script: `data/get_argo/download_en4_profiles.sh`  
- Behavior:  
  - Checks each year URL availability  
  - Immediately downloads when available  
  - Writes CSV log including transfer stats  
    (`filename,path,datetime,status,expected_bytes,downloaded_bytes,duration_seconds,avg_mb_per_s`)  
  - Prints per-file live progress (size/speed/ETA), plus run progress/ETA  
  
Current folder size (snapshot on March 10, 2026):  
- `/data1/datasets/depth_v2/en4_profiles`: `25,074,283,765` bytes (`~23.35 GiB`)  

Shared vertical-grid note:  
- in the current EN4 files used here, profile depth levels are treated as consistent enough across files for a representative-file alignment check  
- that lets us derive one fixed channel mapping from ARGO level index -> nearest GLORYS depth level index and save it once for downstream reuse  
- important EN4 storage detail: `N_LEVELS=400` is the rectangular storage capacity in the file, not a guarantee that every single profile contains 400 real observed depths  
- each EN4 profile stores its actual observed depths in `DEPH_CORRECTED`; unobserved slots remain missing/fill-value entries inside that 400-slot row  
- therefore, a single ARGO profile can contain far fewer valid depths than 400, even though the source file still has shape `(N_PROF, 400)`  

## 3) GLORYS Weekly Archive And ARGO Depth Alignment  
Source:  
- Copernicus Marine GLORYS global physics reanalysis daily stream, subsampled every 7 days into a weekly archive  

Download workflow in this repo:  
- Script: `data/get_glorys/download_glorys_weekly.sh`  
- Output directory by default: `/data1/datasets/depth_v2/glorys_weekly`  
- Behavior:  
  - scans from `2010-01-01` to today by default  
  - downloads one GLORYS daily file every `7` days  
  - writes a CSV log per sampled date (`filename,path,datetime,status`)  

Depth alignment workflow in this repo:  
- Script: `data/EDA_glorys_argo_alignment.py`  
- Purpose:  
  - open one representative ARGO EN4 monthly file  
  - read `DEPH_CORRECTED` across all profiles in that file, not just one sampled observation, to recover a representative depth for each of the 400 EN4 level indices  
  - open one representative weekly GLORYS file and read its fixed `depth` coordinate  
  - compute, for each ARGO level index, the nearest GLORYS depth level index using the median ARGO depth at that level across all valid profiles in the representative EN4 file  
  - save that mapping as a JSON artifact for downstream dataset/export code  

Recommended command:  

```bash
/work/envs/depth/bin/python data/EDA_glorys_argo_alignment.py \
  --output-dir data/glorys_argo_alignment
```

Saved outputs in this repo:  
- `data/glorys_argo_alignment/glorys_argo_alignment_report.txt`  
- `data/glorys_argo_alignment/argo_glorys_depth_alignment.csv`  
- `data/glorys_argo_alignment/argo_depth_level_summary.csv`  
- `data/glorys_argo_alignment/glorys_depth_coverage_summary.csv`  
- `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json`  

Most important persisted artifact:  
- `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json`  
- this file stores the nearest-neighbor mapping from each ARGO level index to a GLORYS level index together with the representative ARGO depth, the matched GLORYS depth, and the absolute depth mismatch  
- the mapping is based on all valid profiles in one representative EN4 monthly file, so it describes the full 400-slot EN4 level layout rather than one specific observation profile  
- example meaning: if one mapping entry says `argo_level_index=17`, `argo_depth_m=146.89`, `glorys_level_index=24`, `glorys_depth_m=155.85`, then ARGO level `17` is represented by about `146.89 m` and uses GLORYS level `24` as its closest 1-to-1 match, with the stored absolute depth difference documenting the mismatch  

Operational guidance:  
- regenerate the JSON if you switch to a different GLORYS product, a different ARGO depth variable, or a different EN4 family/version  
- if later analysis shows ARGO depth grids are not stable across files, this mapping should be recomputed from a broader sample instead of one representative profile  
## 4) Argo <-> OSTIA Datetime Matching  
After EN4 monthly NetCDF profile files are available in  
`/data1/datasets/depth_v2/en4_profiles`, build the datetime matching table:  
  
```bash  
/work/envs/depth/bin/python data/get_argo/build_argo_datetime_match_index.py \  
  --argo-dir /data1/datasets/depth_v2/en4_profiles \  
  --ostia-dir /data1/datasets/depth_v2/ostia \  
  --output-csv /data1/datasets/depth_v2/argo_profile_datetime_match.csv  
```  
  
Script:  
- `data/get_argo/build_argo_datetime_match_index.py`  
  
Output columns:  
- `argo_row_id`  
- `argo_month_key`  
- `argo_profile_date`  
- `profile_idx`  
- `argo_file_path`  
- `matched_ostia_date`  
- `matched_ostia_file_path`  
  
Path format note:  
- path columns are stored from `depth_v2` onward (for example `depth_v2/en4_profiles/...nc`) instead of absolute filesystem paths  
  
Matching behavior:  
- convert EN4 `JULD` to `YYYYMMDD`  
- for each profile date, choose nearest OSTIA day within the same month  
- unreadable/corrupted EN4 files are skipped  
  
## 5) Single Source-Of-Truth Daily CSV  
Merge Argo validity into the OSTIA daily patch CSV:  
  
```bash  
/work/envs/depth/bin/python data/get_argo/merge_argo_into_ostia_daily_index.py \  
  --daily-csv /data1/datasets/depth_v2/ostia_patch_index_daily.csv \  
  --argo-match-csv /data1/datasets/depth_v2/argo_profile_datetime_match.csv \  
  --output-csv /data1/datasets/depth_v2/ostia_patch_index_daily.csv  
```  
  
Script:  
- `data/get_argo/merge_argo_into_ostia_daily_index.py`  
  
Added columns in final daily CSV:  
- `argo_valid` (`1` if day has matched Argo profiles, else `0`)  
- `argo_profile_count` (matched profile count for that day)  
- `argo_month_key`  
- `argo_file_path`  
  
Path format note:  
- `ostia_file_path` and `argo_file_path` are written as `depth_v2/...` anchored paths so the dataset tree can be moved without regenerating absolute paths  
  
## End-To-End Build Order  
Run these steps in order:  
1. Download OSTIA daily files (`data/get_ostia/download_ostia.sh`).  
2. Download EN4 profile data (`data/get_argo/download_en4_profiles.sh`) and extract `.nc` files, for example:  
  
```bash  
mkdir -p /data1/datasets/depth_v2/en4_profiles  
for z in /data1/datasets/depth_v2/en4_profiles/*.zip; do  
  unzip -o "$z" -d /data1/datasets/depth_v2/en4_profiles  
done  
```  
  
Then ensure monthly files like `EN.4.2.2.f.profiles.g10.YYYYMM.nc` exist in that folder.  
3. Build OSTIA patch index (`data/get_ostia/build_ostia_patch_time_index.py`).  
4. Build and save the ARGO-to-GLORYS depth mapping (`data/EDA_glorys_argo_alignment.py`) if downstream code needs 1-to-1 GLORYS channel alignment.  
5. Build Argo datetime match table (`data/get_argo/build_argo_datetime_match_index.py`).  
6. Merge into one final daily CSV (`data/get_argo/merge_argo_into_ostia_daily_index.py`).  
  
## Parameter Tuning Guide (OSTIA Patch Index)  
Most relevant knobs in `build_ostia_patch_time_index.py`:  
- `--tile-size`: patch size in pixels (`128` means 128x128)  
- `--resolution-deg`: pixel size in degrees; omit for native OSTIA  
- `--invalid-threshold`: max invalid fraction before patch is marked invalid  
- `--val-fraction`: fraction of valid water patches assigned to `val`  
- `--split-seed`: deterministic train/val split seed  
- `--valid-mask-values`: OSTIA mask classes counted as valid water  
- `--include-invalid`: include invalid/land patches in outputs  
  
Examples:  
- native 0.05° OSTIA patches: omit `--resolution-deg`  
- custom 0.1°/pixel patches: set `--resolution-deg 0.1`  
  
## Operational Notes  
- Both download scripts support `DRY_RUN_ONLY=1` for availability checks without downloading.  
- Both download scripts append tracking CSV logs in the output directory by default.  
- For EN4, `404` means the specific year/file is not present at the current published path.  
  
## Current Dataset Footprint (Snapshot)  
Snapshot measured/recomputed on **March 10, 2026**.  
  
Shared source data (used by both index versions):  
- OSTIA daily files (`ostia/*.nc`): `5,326`  
- EN4 monthly profile files (`en4_profiles/*.nc`): `186`  
- date range in daily indices: `2010-01-01` to `2024-07-31`  
  
### Index Version A (0.05 degree, daily)  
Files:  
- spatial index: `ostia_patch_index_spatial.csv`  
- daily index: `ostia_patch_index_daily.csv`  
  
Counts:  
- spatial patches: `751` total  
- spatial split: `638 train`, `113 val`  
- daily rows: `3,999,826` total  
- daily split rows: `3,397,988 train`, `601,838 val`  
  
Argo coverage in final daily CSV:  
- rows with `argo_valid=1`: `3,153,449` (`78.84%`)  
- rows with `argo_valid=0`: `846,377` (`21.16%`)  
  
### Index Version B (0.1 degree, daily, recomputed)  
Files:  
- spatial index: `ostia_patch_index_spatial_0p1_recomputed.csv`  
- daily index: `ostia_patch_index_daily_0p1_recomputed_merged.csv`  
  
Counts:  
- spatial patches: `175` total  
- spatial split: `149 train`, `26 val`  
- daily rows: `932,050` total  
- daily split rows: `793,574 train`, `138,476 val`  
  
Argo coverage in final daily CSV:  
- rows with `argo_valid=1`: `734,825` (`78.84%`)  
- rows with `argo_valid=0`: `197,225` (`21.16%`)  
  
  
## Dataset Reduction (Spatial and Temporal)  
  
Resolution progression in this project: initial experiments started at `0.05°`, and the current dataset version is `0.1°` for better observation density.  
  
### V1: 0.05 Deg, Daily  
Recomputed production numbers:  
- `3,999,826` patch-day samples  
- `638 train` / `113 val` spatial patches  
- `3,397,988 train` / `601,838 val` daily rows  
- `846,377` rows (`21.16%`) without Argo observations (`argo_valid=0`)  
  
Estimate:  
`~3M` samples, `~55%` samples without a single observation, `~1.6` average observations for valid tiles.  
  
### V2: 0.1 Deg, Daily  
Recomputed numbers:  
- `932,050` patch-day samples  
- `149 train` / `26 val` spatial patches  
- `793,574 train` / `138,476 val` daily rows  
- `197,225` rows (`21.16%`) without Argo observations (`argo_valid=0`)  
  
Estimate:  
`~600k` samples, `~15%` samples without a single observation, `~4.05` average observations for valid tiles.  
  
Sample Image:  
![OSTIA 0.1 deg sample](assets/argo_ostia_sample_5.png)  
  
Train Set Histogram of valid pixel fractions for the years 2019 and 2020:  
![OSTIA 0.1 deg hist](assets/argo_observations_histogram.png)  
  
Train Set Map of patches, color-coded by valid pixel fraction for the years 2019 and 2020:  
![OSTIA 0.1 deg map](assets/argo_observations_map.png)  

### V3: 0.1 Deg, 7-day aggregate
Recomputed numbers:  
- `932,050` patch-day samples  
- `149 train` / `26 val` spatial patches  
- `793,574 train` / `138,476 val` daily rows  
- `197,225` rows (`21.16%`) without Argo observations (`argo_valid=0`)  

Each step is now sampled +/- N days, so single observations are loaded multiple times when they intersect with the timespan in question. The number of days is adjustable, it's at 7 for now.
  
Estimate:  
`~600k` samples, `~1.5%` samples without a single observation, `~20.5` average observations for valid tiles.  
  
Sample Image:  
![OSTIA 0.1 deg sample](assets/argo_ostia_sample_0p1_7days.png)  

Train Set Histogram of valid pixel fractions for the year 2019 (random, 10% subset):  
![OSTIA 0.1 deg hist](assets/argo_observations_histogram_7days.png)  
  
Train Set Map of patches, color-coded by valid pixel fraction for the year 2019 (random, 10% subset):  
![OSTIA 0.1 deg map](assets/argo_observations_map_7days.png)  

## ARGO To GLORYS Depth Mapping Table
The table below is generated from `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json` and lists all `400` ARGO level indices from the representative EN4 file together with their representative ARGO depth, the nearest GLORYS level, the GLORYS depth, and the absolute depth difference.

| ARGO Level Index | ARGO Depth (m) | GLORYS Level Index | GLORYS Depth (m) | Absolute Difference (m) |
|---:|---:|---:|---:|---:|
| 0 | 3.596 | 3 | 3.819 | 0.223 |
| 1 | 9.117 | 7 | 9.573 | 0.456 |
| 2 | 14.904 | 10 | 15.810 | 0.907 |
| 3 | 20.000 | 11 | 18.496 | 1.504 |
| 4 | 26.000 | 13 | 25.211 | 0.789 |
| 5 | 31.000 | 14 | 29.445 | 1.555 |
| 6 | 38.000 | 16 | 40.344 | 2.344 |
| 7 | 42.529 | 16 | 40.344 | 2.185 |
| 8 | 49.000 | 17 | 47.374 | 1.626 |
| 9 | 52.586 | 18 | 55.764 | 3.179 |
| 10 | 55.088 | 18 | 55.764 | 0.677 |
| 11 | 59.606 | 18 | 55.764 | 3.842 |
| 12 | 64.549 | 19 | 65.807 | 1.258 |
| 13 | 69.387 | 19 | 65.807 | 3.580 |
| 14 | 74.319 | 20 | 77.854 | 3.535 |
| 15 | 79.248 | 20 | 77.854 | 1.395 |
| 16 | 84.186 | 20 | 77.854 | 6.332 |
| 17 | 87.495 | 21 | 92.326 | 4.831 |
| 18 | 91.000 | 21 | 92.326 | 1.326 |
| 19 | 98.194 | 21 | 92.326 | 5.868 |
| 20 | 103.021 | 22 | 109.729 | 6.708 |
| 21 | 108.267 | 22 | 109.729 | 1.462 |
| 22 | 113.352 | 22 | 109.729 | 3.623 |
| 23 | 118.222 | 22 | 109.729 | 8.493 |
| 24 | 123.216 | 23 | 130.666 | 7.450 |
| 25 | 128.168 | 23 | 130.666 | 2.498 |
| 26 | 133.125 | 23 | 130.666 | 2.459 |
| 27 | 138.077 | 23 | 130.666 | 7.411 |
| 28 | 143.037 | 23 | 130.666 | 12.371 |
| 29 | 148.000 | 24 | 155.851 | 7.851 |
| 30 | 153.381 | 24 | 155.851 | 2.469 |
| 31 | 158.372 | 24 | 155.851 | 2.522 |
| 32 | 163.362 | 24 | 155.851 | 7.511 |
| 33 | 168.355 | 24 | 155.851 | 12.504 |
| 34 | 173.321 | 25 | 186.126 | 12.805 |
| 35 | 178.275 | 25 | 186.126 | 7.850 |
| 36 | 183.240 | 25 | 186.126 | 2.886 |
| 37 | 188.250 | 25 | 186.126 | 2.124 |
| 38 | 193.225 | 25 | 186.126 | 7.099 |
| 39 | 198.222 | 25 | 186.126 | 12.096 |
| 40 | 208.575 | 26 | 222.475 | 13.900 |
| 41 | 218.577 | 26 | 222.475 | 3.898 |
| 42 | 228.556 | 26 | 222.475 | 6.080 |
| 43 | 238.609 | 26 | 222.475 | 16.134 |
| 44 | 248.463 | 27 | 266.040 | 17.577 |
| 45 | 258.405 | 27 | 266.040 | 7.636 |
| 46 | 267.913 | 27 | 266.040 | 1.873 |
| 47 | 277.472 | 27 | 266.040 | 11.431 |
| 48 | 287.256 | 27 | 266.040 | 21.215 |
| 49 | 297.105 | 28 | 318.127 | 21.023 |
| 50 | 306.865 | 28 | 318.127 | 11.263 |
| 51 | 316.377 | 28 | 318.127 | 1.751 |
| 52 | 326.108 | 28 | 318.127 | 7.980 |
| 53 | 336.013 | 28 | 318.127 | 17.885 |
| 54 | 345.720 | 28 | 318.127 | 27.593 |
| 55 | 344.000 | 28 | 318.127 | 25.873 |
| 56 | 178.485 | 25 | 186.126 | 7.640 |
| 57 | 183.997 | 25 | 186.126 | 2.129 |
| 58 | 167.800 | 24 | 155.851 | 11.949 |
| 59 | 176.750 | 25 | 186.126 | 9.376 |
| 60 | 188.854 | 25 | 186.126 | 2.728 |
| 61 | 179.000 | 25 | 186.126 | 7.126 |
| 62 | 154.696 | 24 | 155.851 | 1.155 |
| 63 | 159.000 | 24 | 155.851 | 3.149 |
| 64 | 154.913 | 24 | 155.851 | 0.938 |
| 65 | 158.681 | 24 | 155.851 | 2.831 |
| 66 | 159.340 | 24 | 155.851 | 3.489 |
| 67 | 160.754 | 24 | 155.851 | 4.903 |
| 68 | 162.828 | 24 | 155.851 | 6.978 |
| 69 | 145.990 | 24 | 155.851 | 9.861 |
| 70 | 142.805 | 23 | 130.666 | 12.139 |
| 71 | 140.960 | 23 | 130.666 | 10.294 |
| 72 | 93.624 | 21 | 92.326 | 1.298 |
| 73 | 91.755 | 21 | 92.326 | 0.571 |
| 74 | 92.210 | 21 | 92.326 | 0.116 |
| 75 | 91.986 | 21 | 92.326 | 0.340 |
| 76 | 92.180 | 21 | 92.326 | 0.146 |
| 77 | 92.715 | 21 | 92.326 | 0.389 |
| 78 | 92.178 | 21 | 92.326 | 0.148 |
| 79 | 92.118 | 21 | 92.326 | 0.208 |
| 80 | 92.905 | 21 | 92.326 | 0.579 |
| 81 | 93.723 | 21 | 92.326 | 1.397 |
| 82 | 94.446 | 21 | 92.326 | 2.120 |
| 83 | 95.105 | 21 | 92.326 | 2.779 |
| 84 | 96.168 | 21 | 92.326 | 3.842 |
| 85 | 97.000 | 21 | 92.326 | 4.674 |
| 86 | 98.060 | 21 | 92.326 | 5.734 |
| 87 | 99.400 | 21 | 92.326 | 7.074 |
| 88 | 100.227 | 21 | 92.326 | 7.901 |
| 89 | 101.134 | 22 | 109.729 | 8.595 |
| 90 | 101.970 | 22 | 109.729 | 7.759 |
| 91 | 101.374 | 22 | 109.729 | 8.355 |
| 92 | 101.890 | 22 | 109.729 | 7.839 |
| 93 | 102.900 | 22 | 109.729 | 6.829 |
| 94 | 103.900 | 22 | 109.729 | 5.829 |
| 95 | 105.090 | 22 | 109.729 | 4.639 |
| 96 | 106.219 | 22 | 109.729 | 3.510 |
| 97 | 107.310 | 22 | 109.729 | 2.419 |
| 98 | 108.178 | 22 | 109.729 | 1.552 |
| 99 | 109.400 | 22 | 109.729 | 0.329 |
| 100 | 111.415 | 22 | 109.729 | 1.686 |
| 101 | 112.870 | 22 | 109.729 | 3.141 |
| 102 | 114.184 | 22 | 109.729 | 4.455 |
| 103 | 115.350 | 22 | 109.729 | 5.621 |
| 104 | 116.770 | 22 | 109.729 | 7.041 |
| 105 | 117.460 | 22 | 109.729 | 7.731 |
| 106 | 114.800 | 22 | 109.729 | 5.071 |
| 107 | 116.345 | 22 | 109.729 | 6.616 |
| 108 | 117.600 | 22 | 109.729 | 7.871 |
| 109 | 117.325 | 22 | 109.729 | 7.596 |
| 110 | 117.210 | 22 | 109.729 | 7.481 |
| 111 | 118.169 | 22 | 109.729 | 8.440 |
| 112 | 119.271 | 22 | 109.729 | 9.542 |
| 113 | 121.410 | 23 | 130.666 | 9.256 |
| 114 | 121.940 | 23 | 130.666 | 8.726 |
| 115 | 119.700 | 22 | 109.729 | 9.971 |
| 116 | 120.700 | 23 | 130.666 | 9.966 |
| 117 | 121.700 | 23 | 130.666 | 8.966 |
| 118 | 122.700 | 23 | 130.666 | 7.966 |
| 119 | 123.700 | 23 | 130.666 | 6.966 |
| 120 | 124.600 | 23 | 130.666 | 6.066 |
| 121 | 125.600 | 23 | 130.666 | 5.066 |
| 122 | 126.600 | 23 | 130.666 | 4.066 |
| 123 | 127.600 | 23 | 130.666 | 3.066 |
| 124 | 128.600 | 23 | 130.666 | 2.066 |
| 125 | 129.600 | 23 | 130.666 | 1.066 |
| 126 | 130.600 | 23 | 130.666 | 0.066 |
| 127 | 131.600 | 23 | 130.666 | 0.934 |
| 128 | 132.600 | 23 | 130.666 | 1.934 |
| 129 | 133.500 | 23 | 130.666 | 2.834 |
| 130 | 134.500 | 23 | 130.666 | 3.834 |
| 131 | 135.500 | 23 | 130.666 | 4.834 |
| 132 | 136.500 | 23 | 130.666 | 5.834 |
| 133 | 137.500 | 23 | 130.666 | 6.834 |
| 134 | 138.500 | 23 | 130.666 | 7.834 |
| 135 | 139.500 | 23 | 130.666 | 8.834 |
| 136 | 140.500 | 23 | 130.666 | 9.834 |
| 137 | 141.500 | 23 | 130.666 | 10.834 |
| 138 | 142.400 | 23 | 130.666 | 11.734 |
| 139 | 143.400 | 24 | 155.851 | 12.451 |
| 140 | 144.400 | 24 | 155.851 | 11.451 |
| 141 | 145.400 | 24 | 155.851 | 10.451 |
| 142 | 146.400 | 24 | 155.851 | 9.451 |
| 143 | 147.400 | 24 | 155.851 | 8.451 |
| 144 | 148.400 | 24 | 155.851 | 7.451 |
| 145 | 149.300 | 24 | 155.851 | 6.551 |
| 146 | 150.300 | 24 | 155.851 | 5.551 |
| 147 | 151.300 | 24 | 155.851 | 4.551 |
| 148 | 152.300 | 24 | 155.851 | 3.551 |
| 149 | 153.300 | 24 | 155.851 | 2.551 |
| 150 | 154.300 | 24 | 155.851 | 1.551 |
| 151 | 155.300 | 24 | 155.851 | 0.551 |
| 152 | 156.300 | 24 | 155.851 | 0.449 |
| 153 | 157.300 | 24 | 155.851 | 1.449 |
| 154 | 158.300 | 24 | 155.851 | 2.449 |
| 155 | 159.300 | 24 | 155.851 | 3.449 |
| 156 | 160.200 | 24 | 155.851 | 4.349 |
| 157 | 161.200 | 24 | 155.851 | 5.349 |
| 158 | 162.200 | 24 | 155.851 | 6.349 |
| 159 | 163.200 | 24 | 155.851 | 7.349 |
| 160 | 164.200 | 24 | 155.851 | 8.349 |
| 161 | 165.200 | 24 | 155.851 | 9.349 |
| 162 | 166.200 | 24 | 155.851 | 10.349 |
| 163 | 167.200 | 24 | 155.851 | 11.349 |
| 164 | 168.200 | 24 | 155.851 | 12.349 |
| 165 | 169.200 | 24 | 155.851 | 13.349 |
| 166 | 170.200 | 24 | 155.851 | 14.349 |
| 167 | 171.100 | 25 | 186.126 | 15.026 |
| 168 | 172.100 | 25 | 186.126 | 14.026 |
| 169 | 173.100 | 25 | 186.126 | 13.026 |
| 170 | 174.100 | 25 | 186.126 | 12.026 |
| 171 | 175.100 | 25 | 186.126 | 11.026 |
| 172 | 176.100 | 25 | 186.126 | 10.026 |
| 173 | 177.100 | 25 | 186.126 | 9.026 |
| 174 | 178.100 | 25 | 186.126 | 8.026 |
| 175 | 179.100 | 25 | 186.126 | 7.026 |
| 176 | 180.000 | 25 | 186.126 | 6.126 |
| 177 | 181.100 | 25 | 186.126 | 5.026 |
| 178 | 182.000 | 25 | 186.126 | 4.126 |
| 179 | 183.000 | 25 | 186.126 | 3.126 |
| 180 | 184.000 | 25 | 186.126 | 2.126 |
| 181 | 185.000 | 25 | 186.126 | 1.126 |
| 182 | 186.000 | 25 | 186.126 | 0.126 |
| 183 | 187.000 | 25 | 186.126 | 0.874 |
| 184 | 188.000 | 25 | 186.126 | 1.874 |
| 185 | 189.000 | 25 | 186.126 | 2.874 |
| 186 | 189.900 | 25 | 186.126 | 3.774 |
| 187 | 190.900 | 25 | 186.126 | 4.774 |
| 188 | 191.900 | 25 | 186.126 | 5.774 |
| 189 | 192.900 | 25 | 186.126 | 6.774 |
| 190 | 193.900 | 25 | 186.126 | 7.774 |
| 191 | 194.900 | 25 | 186.126 | 8.774 |
| 192 | 195.900 | 25 | 186.126 | 9.774 |
| 193 | 196.900 | 25 | 186.126 | 10.774 |
| 194 | 197.900 | 25 | 186.126 | 11.774 |
| 195 | 198.900 | 25 | 186.126 | 12.774 |
| 196 | 199.800 | 25 | 186.126 | 13.674 |
| 197 | 200.800 | 25 | 186.126 | 14.674 |
| 198 | 201.800 | 25 | 186.126 | 15.674 |
| 199 | 202.800 | 25 | 186.126 | 16.674 |
| 200 | 203.800 | 25 | 186.126 | 17.674 |
| 201 | 204.800 | 26 | 222.475 | 17.675 |
| 202 | 205.800 | 26 | 222.475 | 16.675 |
| 203 | 206.800 | 26 | 222.475 | 15.675 |
| 204 | 207.700 | 26 | 222.475 | 14.775 |
| 205 | 208.800 | 26 | 222.475 | 13.675 |
| 206 | 209.700 | 26 | 222.475 | 12.775 |
| 207 | 210.800 | 26 | 222.475 | 11.675 |
| 208 | 211.700 | 26 | 222.475 | 10.775 |
| 209 | 212.700 | 26 | 222.475 | 9.775 |
| 210 | 213.700 | 26 | 222.475 | 8.775 |
| 211 | 214.700 | 26 | 222.475 | 7.775 |
| 212 | 215.700 | 26 | 222.475 | 6.775 |
| 213 | 216.700 | 26 | 222.475 | 5.775 |
| 214 | 217.700 | 26 | 222.475 | 4.775 |
| 215 | 218.700 | 26 | 222.475 | 3.775 |
| 216 | 219.700 | 26 | 222.475 | 2.775 |
| 217 | 220.600 | 26 | 222.475 | 1.875 |
| 218 | 221.600 | 26 | 222.475 | 0.875 |
| 219 | 222.600 | 26 | 222.475 | 0.125 |
| 220 | 223.600 | 26 | 222.475 | 1.125 |
| 221 | 224.600 | 26 | 222.475 | 2.125 |
| 222 | 225.600 | 26 | 222.475 | 3.125 |
| 223 | 226.600 | 26 | 222.475 | 4.125 |
| 224 | 227.600 | 26 | 222.475 | 5.125 |
| 225 | 228.600 | 26 | 222.475 | 6.125 |
| 226 | 229.600 | 26 | 222.475 | 7.125 |
| 227 | 230.600 | 26 | 222.475 | 8.125 |
| 228 | 231.600 | 26 | 222.475 | 9.125 |
| 229 | 232.500 | 26 | 222.475 | 10.025 |
| 230 | 233.500 | 26 | 222.475 | 11.025 |
| 231 | 234.500 | 26 | 222.475 | 12.025 |
| 232 | 235.500 | 26 | 222.475 | 13.025 |
| 233 | 236.500 | 26 | 222.475 | 14.025 |
| 234 | 237.500 | 26 | 222.475 | 15.025 |
| 235 | 238.500 | 26 | 222.475 | 16.025 |
| 236 | 239.750 | 26 | 222.475 | 17.275 |
| 237 | 240.780 | 26 | 222.475 | 18.305 |
| 238 | 242.000 | 26 | 222.475 | 19.525 |
| 239 | 243.000 | 26 | 222.475 | 20.525 |
| 240 | 244.000 | 26 | 222.475 | 21.525 |
| 241 | 245.000 | 27 | 266.040 | 21.040 |
| 242 | 246.100 | 27 | 266.040 | 19.940 |
| 243 | 247.100 | 27 | 266.040 | 18.940 |
| 244 | 248.100 | 27 | 266.040 | 17.940 |
| 245 | 249.100 | 27 | 266.040 | 16.940 |
| 246 | 250.000 | 27 | 266.040 | 16.040 |
| 247 | 251.100 | 27 | 266.040 | 14.940 |
| 248 | 252.000 | 27 | 266.040 | 14.040 |
| 249 | 253.000 | 27 | 266.040 | 13.040 |
| 250 | 254.000 | 27 | 266.040 | 12.040 |
| 251 | 255.000 | 27 | 266.040 | 11.040 |
| 252 | 256.000 | 27 | 266.040 | 10.040 |
| 253 | 257.000 | 27 | 266.040 | 9.040 |
| 254 | 257.850 | 27 | 266.040 | 8.190 |
| 255 | 258.700 | 27 | 266.040 | 7.340 |
| 256 | 259.820 | 27 | 266.040 | 6.220 |
| 257 | 260.700 | 27 | 266.040 | 5.340 |
| 258 | 261.700 | 27 | 266.040 | 4.340 |
| 259 | 262.900 | 27 | 266.040 | 3.140 |
| 260 | 263.900 | 27 | 266.040 | 2.140 |
| 261 | 264.900 | 27 | 266.040 | 1.140 |
| 262 | 265.800 | 27 | 266.040 | 0.240 |
| 263 | 266.750 | 27 | 266.040 | 0.710 |
| 264 | 267.550 | 27 | 266.040 | 1.510 |
| 265 | 268.100 | 27 | 266.040 | 2.060 |
| 266 | 269.100 | 27 | 266.040 | 3.060 |
| 267 | 270.145 | 27 | 266.040 | 4.105 |
| 268 | 271.400 | 27 | 266.040 | 5.360 |
| 269 | 272.700 | 27 | 266.040 | 6.660 |
| 270 | 273.750 | 27 | 266.040 | 7.710 |
| 271 | 274.700 | 27 | 266.040 | 8.660 |
| 272 | 275.700 | 27 | 266.040 | 9.660 |
| 273 | 276.800 | 27 | 266.040 | 10.760 |
| 274 | 277.750 | 27 | 266.040 | 11.710 |
| 275 | 278.750 | 27 | 266.040 | 12.710 |
| 276 | 279.700 | 27 | 266.040 | 13.660 |
| 277 | 280.800 | 27 | 266.040 | 14.760 |
| 278 | 281.800 | 27 | 266.040 | 15.760 |
| 279 | 282.800 | 27 | 266.040 | 16.760 |
| 280 | 283.800 | 27 | 266.040 | 17.760 |
| 281 | 284.800 | 27 | 266.040 | 18.760 |
| 282 | 285.800 | 27 | 266.040 | 19.760 |
| 283 | 286.900 | 27 | 266.040 | 20.860 |
| 284 | 288.800 | 27 | 266.040 | 22.760 |
| 285 | 291.000 | 27 | 266.040 | 24.960 |
| 286 | 296.600 | 28 | 318.127 | 21.527 |
| 287 | 300.700 | 28 | 318.127 | 17.427 |
| 288 | 311.400 | 28 | 318.127 | 6.727 |
| 289 | 322.050 | 28 | 318.127 | 3.923 |
| 290 | 333.715 | 28 | 318.127 | 15.588 |
| 291 | 345.360 | 28 | 318.127 | 27.233 |
| 292 | 349.660 | 29 | 380.213 | 30.553 |
| 293 | 359.310 | 29 | 380.213 | 20.903 |
| 294 | 370.750 | 29 | 380.213 | 9.463 |
| 295 | 382.015 | 29 | 380.213 | 1.802 |
| 296 | 390.450 | 29 | 380.213 | 10.237 |
| 297 | 395.640 | 29 | 380.213 | 15.427 |
| 298 | 401.480 | 29 | 380.213 | 21.267 |
| 299 | 404.785 | 29 | 380.213 | 24.572 |
| 300 | 409.120 | 29 | 380.213 | 28.907 |
| 301 | 411.105 | 29 | 380.213 | 30.892 |
| 302 | 413.700 | 29 | 380.213 | 33.487 |
| 303 | 415.940 | 29 | 380.213 | 35.727 |
| 304 | 418.110 | 30 | 453.938 | 35.828 |
| 305 | 420.430 | 30 | 453.938 | 33.508 |
| 306 | 422.985 | 30 | 453.938 | 30.953 |
| 307 | 425.935 | 30 | 453.938 | 28.003 |
| 308 | 428.020 | 30 | 453.938 | 25.918 |
| 309 | 429.700 | 30 | 453.938 | 24.238 |
| 310 | 432.210 | 30 | 453.938 | 21.728 |
| 311 | 434.600 | 30 | 453.938 | 19.338 |
| 312 | 436.930 | 30 | 453.938 | 17.008 |
| 313 | 440.600 | 30 | 453.938 | 13.338 |
| 314 | 442.650 | 30 | 453.938 | 11.288 |
| 315 | 444.830 | 30 | 453.938 | 9.108 |
| 316 | 447.390 | 30 | 453.938 | 6.548 |
| 317 | 448.990 | 30 | 453.938 | 4.948 |
| 318 | 451.180 | 30 | 453.938 | 2.758 |
| 319 | 453.800 | 30 | 453.938 | 0.138 |
| 320 | 455.920 | 30 | 453.938 | 1.982 |
| 321 | 456.960 | 30 | 453.938 | 3.022 |
| 322 | 462.080 | 30 | 453.938 | 8.142 |
| 323 | 463.720 | 30 | 453.938 | 9.782 |
| 324 | 465.600 | 30 | 453.938 | 11.662 |
| 325 | 467.550 | 30 | 453.938 | 13.612 |
| 326 | 469.275 | 30 | 453.938 | 15.337 |
| 327 | 471.200 | 30 | 453.938 | 17.262 |
| 328 | 473.270 | 30 | 453.938 | 19.332 |
| 329 | 474.875 | 30 | 453.938 | 20.937 |
| 330 | 476.470 | 30 | 453.938 | 22.532 |
| 331 | 478.270 | 30 | 453.938 | 24.332 |
| 332 | 480.040 | 30 | 453.938 | 26.102 |
| 333 | 481.680 | 30 | 453.938 | 27.742 |
| 334 | 483.840 | 30 | 453.938 | 29.902 |
| 335 | 485.180 | 30 | 453.938 | 31.242 |
| 336 | 485.920 | 30 | 453.938 | 31.982 |
| 337 | 488.380 | 30 | 453.938 | 34.442 |
| 338 | 486.800 | 30 | 453.938 | 32.862 |
| 339 | 488.260 | 30 | 453.938 | 34.322 |
| 340 | 489.255 | 30 | 453.938 | 35.317 |
| 341 | 490.040 | 30 | 453.938 | 36.102 |
| 342 | 489.390 | 30 | 453.938 | 35.452 |
| 343 | 489.890 | 30 | 453.938 | 35.952 |
| 344 | 491.450 | 30 | 453.938 | 37.512 |
| 345 | 494.765 | 30 | 453.938 | 40.827 |
| 346 | 497.200 | 30 | 453.938 | 43.262 |
| 347 | 503.200 | 31 | 541.089 | 37.889 |
| 348 | 513.339 | 31 | 541.089 | 27.750 |
| 349 | 523.868 | 31 | 541.089 | 17.221 |
| 350 | 534.343 | 31 | 541.089 | 6.746 |
| 351 | 544.747 | 31 | 541.089 | 3.658 |
| 352 | 555.131 | 31 | 541.089 | 14.042 |
| 353 | 565.459 | 31 | 541.089 | 24.370 |
| 354 | 575.823 | 31 | 541.089 | 34.734 |
| 355 | 586.177 | 31 | 541.089 | 45.088 |
| 356 | 596.571 | 32 | 643.567 | 46.996 |
| 357 | 606.836 | 32 | 643.567 | 36.730 |
| 358 | 617.152 | 32 | 643.567 | 26.415 |
| 359 | 627.996 | 32 | 643.567 | 15.570 |
| 360 | 638.762 | 32 | 643.567 | 4.804 |
| 361 | 649.453 | 32 | 643.567 | 5.886 |
| 362 | 660.110 | 32 | 643.567 | 16.543 |
| 363 | 670.682 | 32 | 643.567 | 27.115 |
| 364 | 681.235 | 32 | 643.567 | 37.668 |
| 365 | 691.750 | 32 | 643.567 | 48.183 |
| 366 | 702.132 | 32 | 643.567 | 58.565 |
| 367 | 710.965 | 33 | 763.333 | 52.369 |
| 368 | 712.828 | 33 | 763.333 | 50.505 |
| 369 | 714.692 | 33 | 763.333 | 48.641 |
| 370 | 716.560 | 33 | 763.333 | 46.773 |
| 371 | 718.426 | 33 | 763.333 | 44.907 |
| 372 | 720.293 | 33 | 763.333 | 43.040 |
| 373 | 722.159 | 33 | 763.333 | 41.174 |
| 374 | 724.025 | 33 | 763.333 | 39.308 |
| 375 | 725.860 | 33 | 763.333 | 37.473 |
| 376 | 736.560 | 33 | 763.333 | 26.773 |
| 377 | 747.536 | 33 | 763.333 | 15.797 |
| 378 | 753.485 | 33 | 763.333 | 9.848 |
| 379 | 760.785 | 33 | 763.333 | 2.548 |
| 380 | 772.720 | 33 | 763.333 | 9.387 |
| 381 | 784.900 | 33 | 763.333 | 21.567 |
| 382 | 796.419 | 33 | 763.333 | 33.086 |
| 383 | 807.014 | 33 | 763.333 | 43.681 |
| 384 | 817.608 | 33 | 763.333 | 54.275 |
| 385 | 828.107 | 33 | 763.333 | 64.773 |
| 386 | 838.922 | 34 | 902.339 | 63.418 |
| 387 | 851.440 | 34 | 902.339 | 50.899 |
| 388 | 863.480 | 34 | 902.339 | 38.859 |
| 389 | 876.075 | 34 | 902.339 | 26.264 |
| 390 | 892.400 | 34 | 902.339 | 9.939 |
| 391 | 908.268 | 34 | 902.339 | 5.929 |
| 392 | 912.340 | 34 | 902.339 | 10.001 |
| 393 | 926.815 | 34 | 902.339 | 24.476 |
| 394 | 944.000 | 34 | 902.339 | 41.661 |
| 395 | 953.285 | 34 | 902.339 | 50.946 |
| 396 | 960.450 | 34 | 902.339 | 58.111 |
| 397 | 919.797 | 34 | 902.339 | 17.458 |
| 398 | 931.355 | 34 | 902.339 | 29.016 |
| 399 | 994.100 | 35 | 1062.440 | 68.340 |
