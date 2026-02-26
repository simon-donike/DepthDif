#!/usr/bin/env bash
set -euo pipefail

DATASET="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"

# NRT window you observed is ~2024-01-17 .. 2026-02-25
START_YEAR=2024
END_YEAR=2026

for YEAR in $(seq $START_YEAR $END_YEAR); do
  for MONTH in {01..12}; do
    # skip months outside known availability window
    if [[ "$YEAR" == "2024" && "$MONTH" == "01" ]]; then
      continue  # 2024-01-15 not available
    fi
    if [[ "$YEAR" == "2026" && "$MONTH" > "02" ]]; then
      continue  # beyond latest month in your listing
    fi

    DAYTAG="${YEAR}${MONTH}15"
    echo "Downloading ${YEAR}-${MONTH}-15"

    copernicusmarine get \
      -i "$DATASET" \
      --filter "*/${YEAR}/${MONTH}/*${DAYTAG}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
  done
done