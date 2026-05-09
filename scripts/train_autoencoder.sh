#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config configs/lat_space/ae_config.yaml \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/lat_space/training_config.yaml
