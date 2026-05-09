# Zarr Training Dataset Export

This folder contains the compact zarr export path. The zarr-backed training
loader lives in `data/dataset_argo_zarr_gridded.py`.

The exporter keeps only the model-relevant variables configured in
`source_variables.yaml` and writes raster sources on a 0.1 degree grid by
default:

- OSTIA: `analysed_sst` plus optional `mask` for patch filtering
- ARGO / EN4: `TEMP`, `PSAL_CORRECTED`, and `DEPH_CORRECTED`
- GLORYS: `thetao`, `so`, and `zos`
- Sea Level L4: `adt` by default

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
  --chunk-time 16 \
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
0.1 degrees before writing, while ARGO profiles stay at native profile
locations. Set `--target-resolution-deg none` to keep native raster grids. The
CLI `--ostia-vars`, `--argo-vars`, `--argo-depth-var`, `--glorys-vars`, and
`--sealevel-vars` flags still override those defaults for one-off exports.

Training can use `configs/px_space/data_ostia_argo_zarr.yaml` with
`dataset.core.dataset_variant: argo_zarr_gridded`.

The zarr loader preserves the current model-facing keys (`eo`, `x`, `y`,
valid masks, date, and optional coords/info). Set
`dataset.output.return_modalities: true` to also receive the saved salinity and
sea-surface-height tensors under `sample["modalities"]` with matching masks in
`sample["modality_valid_masks"]`.
