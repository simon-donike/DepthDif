# Zarr Training Dataset Export

This folder contains the compact zarr export path. The zarr-backed training
loader lives in `data/dataset_argo_zarr_gridded.py`.

The exporter keeps only the model-relevant variables configured in
`source_variables.yaml`, writes raster sources on the GLORYS grid resampled to
0.1 degrees by default, and stores continuous fields as packed int16 arrays:

- OSTIA: weekly `analysed_sst` aggregates plus optional `mask` for patch filtering
- ARGO / EN4: `TEMP` and `PSAL_CORRECTED` projected onto the GLORYS depth axis
- GLORYS: `thetao`, `so`, and `zos`
- Sea Level L4: weekly `adt` aggregates by default

Run from the repository root:

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
  --surface-aggregate-days 7 \
  --chunk-time 1 \
  --chunk-profile 20000 \
  --chunk-lat 256 \
  --chunk-lon 256 \
  --dask-scheduler threads \
  --dask-num-workers 8 \
  --overwrite
```

The output folder contains:

- `ostia.zarr`
- `argo.zarr`
- `glorys.zarr`
- `sealevel.zarr` when `--sealevel-dir` has matching files
- `manifest.yaml`

Edit `source_variables.yaml` to change the default variables kept in each Zarr
store. By default, GLORYS is interpolated to 0.1 degrees and OSTIA plus
sea-level rasters are reprojected onto those exact GLORYS latitude/longitude
coordinates before writing. OSTIA and sea-level daily files are reduced to
centered 7-day windows around the GLORYS timesteps, so the saved surface stores
share the weekly GLORYS cadence and grid. ARGO profile variables are projected
from `DEPH_CORRECTED` onto the 50-level GLORYS depth coordinate. Continuous
variables are packed to int16 with variable-specific scale factors, and the
OSTIA mask is stored as int8. Set `--target-resolution-deg none` to keep the
native GLORYS raster grid as the shared output grid.
The CLI `--surface-aggregate-days`, `--ostia-vars`, `--argo-vars`,
`--argo-depth-var`, `--glorys-vars`, and `--sealevel-vars` flags still override
those defaults for one-off exports.
The exporter prints elapsed wall time for each major source scan/open/write
phase. Use `--dask-scheduler` and `--dask-num-workers` to control dask execution
while writing the zarr stores; leave them unset to keep dask's default scheduler.

Training can use `configs/px_space/data_ostia_argo_zarr.yaml` with
`dataset.core.dataset_variant: argo_zarr_gridded`.

The zarr loader preserves the current model-facing keys (`eo`, `x`, `y`,
valid masks, date, and optional coords/info). Set
`dataset.output.return_modalities: true` to also receive the saved salinity and
sea-surface-height tensors under `sample["modalities"]` with matching masks in
`sample["modality_valid_masks"]`.
