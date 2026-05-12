#!/usr/bin/env bash
# Example:
#   src/depth_recon/scripts/train_autoencoder.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_ROOT="$REPO_ROOT/src/depth_recon/configs"

cd "$REPO_ROOT"

/work/envs/depth/bin/python "$REPO_ROOT/train_autoencoder.py" \
  --ae-config "$CONFIG_ROOT/lat_space/ae_config.yaml" \
  --data-config "$CONFIG_ROOT/px_space/data_ostia_argo_netcdf.yaml" \
  --train-config "$CONFIG_ROOT/lat_space/training_config.yaml"
