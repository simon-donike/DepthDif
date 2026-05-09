#!/usr/bin/env bash
set -euo pipefail

# Copernicus GLORYS 001_030 currently exposes daily/monthly streams, not a
# dedicated weekly one. This helper therefore downloads one daily file every
# 7 days to build a weekly-timescale archive aligned with the Argo-era range.
if [[ -n "${GLORYS_DATASET_ID:-}" ]]; then
  # When user provides an explicit dataset ID, use only that value.
  GLORYS_DATASET_CANDIDATES=("${GLORYS_DATASET_ID}")
else
  GLORYS_DATASET_CANDIDATES=(
    "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311"
    "global-reanalysis-phy-001-030-daily"
  )
fi
START_DATE="${START_DATE:-2010-01-01}"
END_DATE="${END_DATE:-$(date -u +%F)}"
OUTPUT_DIR="${1:-/data1/datasets/depth_v2/glorys_weekly}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
STEP_DAYS="${STEP_DAYS:-7}"

mkdir -p "${OUTPUT_DIR}"
CSV_LOG_PATH="${CSV_LOG_PATH:-${OUTPUT_DIR}/glorys_weekly_download_log.csv}"

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

if [[ "${END_DATE}" < "${START_DATE}" ]]; then
  echo "Error: END_DATE (${END_DATE}) must be >= START_DATE (${START_DATE})." >&2
  exit 1
fi
if ! [[ "${STEP_DAYS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: STEP_DAYS (${STEP_DAYS}) must be a positive integer." >&2
  exit 1
fi

echo "Using GLORYS dataset candidates: ${GLORYS_DATASET_CANDIDATES[*]}"
echo "Date range: ${START_DATE} .. ${END_DATE}"
echo "Weekly stride: every ${STEP_DAYS} days"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dry-run-only mode: ${DRY_RUN_ONLY} (must be 0 for real downloads)"
echo "CSV log: ${CSV_LOG_PATH}"
echo

if [[ ! -f "${CSV_LOG_PATH}" ]]; then
  echo "filename,path,datetime,status" > "${CSV_LOG_PATH}"
fi

append_csv_row() {
  local filename="$1"
  local path="$2"
  local day="$3"
  local status="$4"
  # Keep CSV append centralized to avoid inconsistent status formatting across branches.
  printf '"%s","%s","%s","%s"\n' "${filename}" "${path}" "${day}" "${status}" >> "${CSV_LOG_PATH}"
}

start_epoch="$(date -u -d "${START_DATE}" +%s)"
end_epoch="$(date -u -d "${END_DATE}" +%s)"
total_days="$(( (end_epoch - start_epoch) / 86400 ))"
total_steps="$(( total_days / STEP_DAYS + 1 ))"
current_date="${START_DATE}"
processed_steps=0
missing_steps=0
query_errors=0
selected_dataset_id=""
downloaded_steps=0
download_errors=0
run_start_epoch="$(date -u +%s)"

format_eta() {
  local total_seconds="$1"
  local hours="$(( total_seconds / 3600 ))"
  local minutes="$(( (total_seconds % 3600) / 60 ))"
  local seconds="$(( total_seconds % 60 ))"
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

while [[ "${current_date}" < "${END_DATE}" || "${current_date}" == "${END_DATE}" ]]; do
  processed_steps=$((processed_steps + 1))
  year="$(date -u -d "${current_date}" +%Y)"
  month="$(date -u -d "${current_date}" +%m)"
  day_tag="$(date -u -d "${current_date}" +%Y%m%d)"
  # Keep the filter broad to tolerate provider-side filename/version suffix changes.
  glorys_filter="*/${year}/${month}/*${day_tag}*.nc"

  dry_output=""
  dry_error=""
  used_dataset_id=""
  for candidate_id in "${GLORYS_DATASET_CANDIDATES[@]}"; do
    if [[ -z "${candidate_id}" ]]; then
      continue
    fi
    err_tmp_file="$(mktemp)"
    cmd_output="$("${COPERNICUS_CMD}" get \
      -i "${candidate_id}" \
      --filter "${glorys_filter}" \
      --dry-run \
      --log-level ERROR 2>"${err_tmp_file}" || true)"
    cmd_error="$(cat "${err_tmp_file}")"
    rm -f "${err_tmp_file}"
    if file_count_candidate="$(printf "%s" "${cmd_output}" | "${PYTHON_CMD}" -c 'import json,sys; print(int((json.load(sys.stdin).get("number_of_files_to_download") or 0)))' 2>/dev/null)"; then
      dry_output="${cmd_output}"
      used_dataset_id="${candidate_id}"
      break
    fi
    dry_error="${cmd_error}"
  done
  if [[ -z "${dry_output}" ]]; then
    echo "[${current_date}] query_failed"
    if [[ -n "${dry_error}" ]]; then
      echo "  error: ${dry_error}"
    fi
    append_csv_row "" "" "${current_date}" "query_failed"
    query_errors=$((query_errors + 1))
  else
    if [[ -n "${used_dataset_id}" && -z "${selected_dataset_id}" ]]; then
      selected_dataset_id="${used_dataset_id}"
      echo "  -> Selected dataset ID: ${selected_dataset_id}"
    fi
    file_count="$(printf "%s" "${dry_output}" | "${PYTHON_CMD}" -c 'import json,sys; print(int((json.load(sys.stdin).get("number_of_files_to_download") or 0)))' 2>/dev/null || echo 0)"
    if [[ "${file_count}" -eq 0 ]]; then
      echo "[${current_date}] missing"
      append_csv_row "" "" "${current_date}" "missing"
      missing_steps=$((missing_steps + 1))
    elif [[ "${DRY_RUN_ONLY}" == "1" ]]; then
      echo "[${current_date}] available_dry_run"
      append_csv_row "" "" "${current_date}" "available_dry_run"
    else
      download_err_tmp="$(mktemp)"
      if "${COPERNICUS_CMD}" get \
        -i "${used_dataset_id}" \
        --filter "${glorys_filter}" \
        -o "${OUTPUT_DIR}" \
        -nd \
        --log-level ERROR >/dev/null 2>"${download_err_tmp}"; then
        shopt -s nullglob
        downloaded_files=("${OUTPUT_DIR}"/*"${day_tag}"*.nc)
        shopt -u nullglob
        if [[ ${#downloaded_files[@]} -gt 0 ]]; then
          downloaded_file="${downloaded_files[0]}"
          append_csv_row "$(basename "${downloaded_file}")" "${downloaded_file}" "${current_date}" "downloaded"
          echo "[${current_date}] downloaded"
        else
          append_csv_row "" "" "${current_date}" "downloaded_but_file_not_found"
          echo "[${current_date}] downloaded_but_file_not_found"
        fi
        downloaded_steps=$((downloaded_steps + 1))
      else
        echo "[${current_date}] download_failed"
        download_err_msg="$(cat "${download_err_tmp}")"
        if [[ -n "${download_err_msg}" ]]; then
          echo "  error: ${download_err_msg}"
        fi
        append_csv_row "" "" "${current_date}" "download_failed"
        download_errors=$((download_errors + 1))
      fi
      rm -f "${download_err_tmp}"
    fi
  fi

  elapsed_seconds="$(( $(date -u +%s) - run_start_epoch ))"
  remaining_steps="$(( total_steps - processed_steps ))"
  if [[ "${processed_steps}" -gt 0 ]]; then
    eta_seconds="$(( elapsed_seconds * remaining_steps / processed_steps ))"
  else
    eta_seconds=0
  fi
  echo "  -> Progress: ${processed_steps}/${total_steps} weekly steps processed, ${downloaded_steps} downloaded | ETA: $(format_eta "${eta_seconds}")"

  current_date="$(date -u -d "${current_date} + ${STEP_DAYS} day" +%F)"
done

echo
echo "Summary"
echo "- Weekly steps scanned: ${total_steps}"
echo "- Missing weekly steps: ${missing_steps}"
echo "- Query errors: ${query_errors}"
echo "- Weekly files downloaded: ${downloaded_steps}"
echo "- Download errors: ${download_errors}"
echo "- Dry-run-only mode: ${DRY_RUN_ONLY}"

echo
echo "Done."
