#!/usr/bin/env bash
set -euo pipefail

# Run command
# bash data/get_ostia/download_daily_ostia_reanalysis_last_year.sh /path/to/output
# NUM_DAYS=180 bash data/get_ostia/download_daily_ostia_reanalysis_last_year.sh /path/to/output

OSTIA_DATASET="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
REANALYSIS_DATASET="cmems_mod_glo_phy_my_0.083deg_P1D-m"
REANALYSIS_DATASET_VERSION="202311"
# Number of UTC days to process in the [START_DATE, END_DATE] window.
NUM_DAYS="${NUM_DAYS:-365}"

OUTPUT_DIR="${1:-./downloads/daily_ostia_reanalysis_last_year}"
mkdir -p "${OUTPUT_DIR}"

if command -v copernicusmarine >/dev/null 2>&1; then
  COPERNICUS_CMD="copernicusmarine"
elif [[ -x "/work/envs/depth/bin/copernicusmarine" ]]; then
  COPERNICUS_CMD="/work/envs/depth/bin/copernicusmarine"
else
  echo "Error: could not find copernicusmarine CLI in PATH or /work/envs/depth/bin." >&2
  exit 1
fi
PYTHON_CMD="/work/envs/depth/bin/python"
if [[ ! -x "${PYTHON_CMD}" ]]; then
  echo "Error: required python interpreter not found at ${PYTHON_CMD}." >&2
  exit 1
fi

# Default window is the last NUM_DAYS complete UTC days (yesterday included, today excluded).
END_DATE="${END_DATE:-$(date -u -d "yesterday" +%F)}"
if ! [[ "${NUM_DAYS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NUM_DAYS must be a positive integer, got '${NUM_DAYS}'." >&2
  exit 1
fi
START_DATE="${START_DATE:-$(date -u -d "${END_DATE} - $((NUM_DAYS - 1)) days" +%F)}"

echo "Using date range ${START_DATE} .. ${END_DATE}"
echo "Writing all files to ${OUTPUT_DIR}"
echo

current_date="${START_DATE}"
total_days_scanned=0
ostia_missing_days=0
reanalysis_missing_days=0
overlap_days=0
downloaded_pairs=0
download_error_days=0
estimated_ostia_mb=0
estimated_reanalysis_mb=0
matched_days=()

sum_float() {
  local lhs="$1"
  local rhs="$2"
  awk -v a="${lhs}" -v b="${rhs}" 'BEGIN { printf "%.6f", a + b }'
}

format_size_mb() {
  local mb="$1"
  awk -v value_mb="${mb}" 'BEGIN { printf "%.2f MB (%.2f GB)", value_mb, value_mb / 1024 }'
}

get_dry_run_stats() {
  local dataset_id="$1"
  local filter_pattern="$2"
  local dataset_version="${3:-}"
  local output
  if [[ -n "${dataset_version}" ]]; then
    output="$("${COPERNICUS_CMD}" get \
      -i "${dataset_id}" \
      --dataset-version "${dataset_version}" \
      --filter "${filter_pattern}" \
      --dry-run \
      --log-level ERROR 2>/dev/null)" || return 1
  else
    output="$("${COPERNICUS_CMD}" get \
      -i "${dataset_id}" \
      --filter "${filter_pattern}" \
      --dry-run \
      --log-level ERROR 2>/dev/null)" || return 1
  fi

  # Parse dry-run JSON once to extract file count and estimated total size in MB.
  printf "%s" "${output}" | "${PYTHON_CMD}" -c '
import json
import sys

payload = json.load(sys.stdin)
count = int(payload.get("number_of_files_to_download") or 0)
size_mb = float(payload.get("total_size") or 0.0)
print(f"{count} {size_mb}")
'
}

# Preflight scan: gather overlap, counts, and estimated transfer size before downloading.
while [[ "${current_date}" < "${END_DATE}" || "${current_date}" == "${END_DATE}" ]]; do
  total_days_scanned=$((total_days_scanned + 1))
  year="$(date -u -d "${current_date}" +%Y)"
  month="$(date -u -d "${current_date}" +%m)"
  day="$(date -u -d "${current_date}" +%d)"
  day_tag="${year}${month}${day}"

  # OSTIA daily files use YYYY/MM folders and the 12:00 UTC timestamp in filename.
  ostia_filter="*/${year}/${month}/*${day_tag}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
  # Daily GLORYS files are named like mercatorglorys12v1_gl12_mean_YYYYMMDD.nc.
  reanalysis_filter="*/${year}/${month}/*mercatorglorys12v1_gl12_mean_${day_tag}.nc"

  echo "[${current_date}] Preflight availability check"
  ostia_stats="$(get_dry_run_stats "${OSTIA_DATASET}" "${ostia_filter}" || true)"
  if [[ -z "${ostia_stats}" ]]; then
    echo "  -> OSTIA preflight query failed; skipping day."
    download_error_days=$((download_error_days + 1))
    current_date="$(date -u -d "${current_date} + 1 day" +%F)"
    continue
  fi
  ostia_count="${ostia_stats%% *}"
  ostia_size_mb="${ostia_stats#* }"
  if [[ "${ostia_count}" -eq 0 ]]; then
    echo "  -> OSTIA not available for ${current_date}; skipping day."
    ostia_missing_days=$((ostia_missing_days + 1))
    current_date="$(date -u -d "${current_date} + 1 day" +%F)"
    continue
  fi

  reanalysis_stats="$(
    get_dry_run_stats \
      "${REANALYSIS_DATASET}" \
      "${reanalysis_filter}" \
      "${REANALYSIS_DATASET_VERSION}" || true
  )"
  if [[ -z "${reanalysis_stats}" ]]; then
    echo "  -> Reanalysis preflight query failed; skipping day."
    download_error_days=$((download_error_days + 1))
    current_date="$(date -u -d "${current_date} + 1 day" +%F)"
    continue
  fi
  reanalysis_count="${reanalysis_stats%% *}"
  reanalysis_size_mb="${reanalysis_stats#* }"
  if [[ "${reanalysis_count}" -eq 0 ]]; then
    echo "  -> Reanalysis not available for ${current_date}; skipping day."
    reanalysis_missing_days=$((reanalysis_missing_days + 1))
    current_date="$(date -u -d "${current_date} + 1 day" +%F)"
    continue
  fi

  matched_days+=("${current_date}")
  overlap_days=$((overlap_days + 1))
  estimated_ostia_mb="$(sum_float "${estimated_ostia_mb}" "${ostia_size_mb}")"
  estimated_reanalysis_mb="$(sum_float "${estimated_reanalysis_mb}" "${reanalysis_size_mb}")"

  current_date="$(date -u -d "${current_date} + 1 day" +%F)"
done

estimated_total_mb="$(sum_float "${estimated_ostia_mb}" "${estimated_reanalysis_mb}")"

echo
echo "Preflight summary"
echo "- Timeframe: ${START_DATE} .. ${END_DATE} (NUM_DAYS=${NUM_DAYS})"
echo "- Days scanned: ${total_days_scanned}"
echo "- Overlap days available: ${overlap_days}"
echo "- Missing OSTIA days: ${ostia_missing_days}"
echo "- Missing reanalysis days: ${reanalysis_missing_days}"
echo "- Preflight query errors: ${download_error_days}"
echo "- Planned files to download: $((overlap_days * 2))"
echo "- Estimated OSTIA size: $(format_size_mb "${estimated_ostia_mb}")"
echo "- Estimated reanalysis size: $(format_size_mb "${estimated_reanalysis_mb}")"
echo "- Estimated total size: $(format_size_mb "${estimated_total_mb}")"
echo

if [[ "${overlap_days}" -eq 0 ]]; then
  echo "No overlap days found in timeframe. Exiting."
  exit 0
fi

read -r -p "Proceed with downloading ${overlap_days} overlap day pairs to ${OUTPUT_DIR}? [y/N]: " confirm_download
if [[ ! "${confirm_download}" =~ ^[Yy]$ ]]; then
  echo "Aborted by user after preflight."
  exit 0
fi

echo
echo "Starting downloads..."
echo

for current_date in "${matched_days[@]}"; do
  year="$(date -u -d "${current_date}" +%Y)"
  month="$(date -u -d "${current_date}" +%m)"
  day="$(date -u -d "${current_date}" +%d)"
  day_tag="${year}${month}${day}"
  ostia_filter="*/${year}/${month}/*${day_tag}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
  reanalysis_filter="*/${year}/${month}/*mercatorglorys12v1_gl12_mean_${day_tag}.nc"

  echo "[${current_date}] Downloading OSTIA + reanalysis overlap pair"
  # Keep final output overlap-only: move files into OUTPUT_DIR only if both downloads succeed.
  day_tmp_dir="$(mktemp -d "${OUTPUT_DIR}/.tmp_${day_tag}_XXXXXX")"
  if ! "${COPERNICUS_CMD}" get \
    -i "${OSTIA_DATASET}" \
    --filter "${ostia_filter}" \
    -o "${day_tmp_dir}" \
    -nd; then
    echo "  -> OSTIA download failed for ${current_date}; skipping day."
    download_error_days=$((download_error_days + 1))
    rm -rf "${day_tmp_dir}"
    continue
  fi

  if "${COPERNICUS_CMD}" get \
    -i "${REANALYSIS_DATASET}" \
    --dataset-version "${REANALYSIS_DATASET_VERSION}" \
    --filter "${reanalysis_filter}" \
    -o "${day_tmp_dir}" \
    -nd; then
    # Enforce true pairwise commits: only keep the day if both modalities are present.
    shopt -s nullglob
    ostia_files=("${day_tmp_dir}"/*"${day_tag}"120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc)
    reanalysis_files=("${day_tmp_dir}"/*mercatorglorys12v1_gl12_mean_"${day_tag}".nc)
    shopt -u nullglob
    if [[ ${#ostia_files[@]} -eq 0 || ${#reanalysis_files[@]} -eq 0 ]]; then
      echo "  -> Pair validation failed for ${current_date}; not committing partial files."
      download_error_days=$((download_error_days + 1))
      rm -rf "${day_tmp_dir}"
      continue
    fi

    shopt -s nullglob
    day_files=("${day_tmp_dir}"/*)
    shopt -u nullglob
    if [[ ${#day_files[@]} -gt 0 ]]; then
      mv -n "${day_files[@]}" "${OUTPUT_DIR}/"
    fi
    rm -rf "${day_tmp_dir}"
    downloaded_pairs=$((downloaded_pairs + 1))
  else
    echo "  -> Reanalysis download failed for ${current_date}; skipping day."
    download_error_days=$((download_error_days + 1))
    rm -rf "${day_tmp_dir}"
  fi
done

echo
echo "Done."
echo "Paired days downloaded: ${downloaded_pairs}/${overlap_days}"
echo "Days missing OSTIA: ${ostia_missing_days}"
echo "Days missing reanalysis: ${reanalysis_missing_days}"
echo "Days with query/download errors: ${download_error_days}"
