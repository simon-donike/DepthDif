#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config configs/lat_space/ae_config.yaml \
  --data-config configs/lat_space/data_config.yaml \
  --train-config configs/lat_space/training_config.yaml
