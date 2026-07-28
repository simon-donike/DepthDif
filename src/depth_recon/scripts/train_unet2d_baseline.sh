#!/usr/bin/env bash
# Example:
#   src/depth_recon/scripts/train_unet2d_baseline.sh --scenario temperature --config src/depth_recon/configs/px_space/training_super_config.yaml --run-name unet2d_baseline_temperature --fast-dev-run 0 --set training.trainer.max_epochs=100
set -euo pipefail

PYTHON_BIN="/work/envs/depth/bin/python"
REPO_ROOT="/work/code/DepthDif"
CONFIG="src/depth_recon/configs/px_space/training_super_config.yaml"
SCENARIO="temperature"
RUN_NAME=""
FAST_DEV_RUN="0"
EXTRA_SETS=()

usage() {
  cat <<'USAGE'
Usage: src/depth_recon/scripts/train_unet2d_baseline.sh [options]

Options:
  --scenario <temperature|salinity|joint>  Scenario passed to train.py.
  --config <path>                          Pixel training super-config path.
  --run-name <name>                        W&B run name override.
  --fast-dev-run <n>                       Lightning fast-dev-run batch count.
  --set <key=value>                        Extra train.py config override; repeatable.
  --help                                   Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --fast-dev-run|--fast_dev_run)
      FAST_DEV_RUN="$2"
      shift 2
      ;;
    --set)
      EXTRA_SETS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO_ROOT/$CONFIG"
fi
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="unet2d_baseline_${SCENARIO}"
fi

cd "$REPO_ROOT"
cmd=(
  "$PYTHON_BIN" train.py
  --config "$CONFIG"
  --scenario "$SCENARIO"
  --fast-dev-run "$FAST_DEV_RUN"
  --set model.model_type=unet2d_baseline
  --set "training.wandb.run_name=$RUN_NAME"
)
for override in "${EXTRA_SETS[@]}"; do
  cmd+=(--set "$override")
done

exec "${cmd[@]}"
