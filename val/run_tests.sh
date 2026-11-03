#!/usr/bin/env bash
# Example: bash val/run_tests.sh
# Run both suites; keep every log and temporary artifact inside val/ or /tmp.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p val/results/verification
export MPLCONFIGDIR=/tmp/depthdif-val-matplotlib
bash tests/run_tests.sh > val/results/verification/repository_tests.log 2>&1
PYTHONPATH="val/vendor:src${PYTHONPATH:+:$PYTHONPATH}" /work/envs/depth/bin/python -m unittest discover -s val -p 'test_*.py' -v > val/results/verification/validation_tests.log 2>&1
tail -5 val/results/verification/repository_tests.log
tail -5 val/results/verification/validation_tests.log
