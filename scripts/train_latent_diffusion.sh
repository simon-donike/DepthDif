#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train.py \
  --data-config configs/lat_space/data_config.yaml \
  --train-config configs/lat_space/training_config.yaml \
  --model-config configs/lat_space/model_config.yaml
