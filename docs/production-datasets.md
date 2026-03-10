# Production Datasets
This page documents the non-toy datasets currently used in DepthDif:
- OSTIA L4 reprocessed daily sea-surface temperature (EO condition)
- EN4.2.2 profile archives (submarine/in-situ profile observations)

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

## Operational Notes  
- Both scripts support `DRY_RUN_ONLY=1` for availability checks without downloading.  
- Both scripts append tracking CSV logs in the output directory by default.  
- For EN4, `404` means the specific year/file is not present at the current published path.  
