# Production Results

This page tracks runs on the maintained pixel GeoTIFF dataset path:

- `src/depth_recon/configs/px_space/training_super_config.yaml`
- `src/depth_recon/data/dataset_argo_geotiff_gridded.py`

## Scope

Production runs use:

- real EN4 / ARGO profile NetCDF files for sparse inputs
- GLORYS NetCDF files for dense targets
- OSTIA SST, SSS `sos`, and sea-level ADT rasters in fixed `[sst, sss, adt]` conditioning order
- the exported GeoTIFF store plus compact metadata caches

## Current Status

New production results should record the scenario, original super-config, effective resolved configs, checkpoint path, selected date/week, and validation/export artifacts.
