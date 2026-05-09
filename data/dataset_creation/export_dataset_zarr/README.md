# Zarr Training Dataset Export

This folder contains the compact zarr export path. The zarr-backed training
loader lives in `data/dataset_argo_zarr_gridded.py`.

The exporter keeps only the model-relevant variables configured in
`source_variables.yaml`, writes raster sources on a 0.1 degree grid by default,
and stores continuous fields as packed int16 arrays:

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
  --overwrite
```

The output folder contains:

- `ostia.zarr`
- `argo.zarr`
- `glorys.zarr`
- `sealevel.zarr` when `--sealevel-dir` has matching files
- `manifest.yaml`

Edit `source_variables.yaml` to change the default variables kept in each Zarr
store. By default, OSTIA, GLORYS, and sea-level rasters are interpolated to
0.1 degrees before writing. OSTIA and sea-level daily files are reduced to
centered 7-day windows around the GLORYS timesteps, so the saved surface stores
share the weekly GLORYS cadence. ARGO profile variables are projected from
`DEPH_CORRECTED` onto the 50-level GLORYS depth coordinate. Continuous variables
are packed to int16 with variable-specific scale factors, and the OSTIA mask is
stored as int8. Set `--target-resolution-deg none` to keep native raster grids.
The CLI `--surface-aggregate-days`, `--ostia-vars`, `--argo-vars`,
`--argo-depth-var`, `--glorys-vars`, and `--sealevel-vars` flags still override
those defaults for one-off exports.

Training can use `configs/px_space/data_ostia_argo_zarr.yaml` with
`dataset.core.dataset_variant: argo_zarr_gridded`.

The zarr loader preserves the current model-facing keys (`eo`, `x`, `y`,
valid masks, date, and optional coords/info). Set
`dataset.output.return_modalities: true` to also receive the saved salinity and
sea-surface-height tensors under `sample["modalities"]` with matching masks in
`sample["modality_valid_masks"]`.
