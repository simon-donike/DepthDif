#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/lat_space/training_config.yaml \
  --model-config configs/lat_space/model_config.yaml
