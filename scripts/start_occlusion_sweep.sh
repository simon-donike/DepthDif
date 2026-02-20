#!/usr/bin/env bash
set -euo pipefail

WAND_BIN="/work/envs/depth/bin/wandb"
DEFAULT_SWEEP_CFG="configs/sweeps/eo_occlusion_grid_no_eodrop.yaml"
SWEEP_CFG="${1:-$DEFAULT_SWEEP_CFG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$SWEEP_CFG" != /* ]]; then
  SWEEP_CFG="$REPO_ROOT/$SWEEP_CFG"
fi

if [[ ! -f "$SWEEP_CFG" ]]; then
  echo "Sweep config not found: $SWEEP_CFG" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "Creating sweep from: $SWEEP_CFG"
sweep_output="$("$WAND_BIN" sweep "$SWEEP_CFG" 2>&1)"
printf '%s\n' "$sweep_output"

# Parse the exact agent target emitted by wandb so users do not need to copy IDs.
agent_target="$(printf '%s\n' "$sweep_output" | sed -n 's/.*Run sweep agent with: *wandb agent \(.*\)$/\1/p' | tail -n 1)"
if [[ -z "$agent_target" ]]; then
  echo "Could not extract sweep agent target from wandb output." >&2
  exit 1
fi

echo "Starting agent: $agent_target"
exec "$WAND_BIN" agent "$agent_target"
