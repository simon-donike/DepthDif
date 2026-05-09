# Early Experiments

This page is kept as a short index for older qualitative notes. The active
training path is now the NetCDF patch dataset documented in
[NetCDF Patch Dataset](data.md), and new runs should use:

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/px_space/training_config.yaml \
  --model-config configs/px_space/model_config.yaml
```

Older notes that depended on removed dataset variants are no longer part of the
maintained repository workflow.
