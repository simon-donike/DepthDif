# Production Results

This page tracks runs on the maintained NetCDF dataset path:

- `src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml`
- `src/depth_recon/data/dataset_argo_netcdf_gridded.py`

## Scope

Production runs use:

- real EN4 / ARGO profile NetCDF files for sparse inputs
- GLORYS NetCDF files for dense targets
- OSTIA NetCDF files for EO conditioning
- compact metadata caches only

## Current Status

New production results should record the exact model, training, data config,
checkpoint path, selected date/week, and validation/export artifacts.
