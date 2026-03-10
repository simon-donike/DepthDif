#!/usr/bin/env bash
set -euo pipefail

# Monthly GLORYS reanalysis candidates for GLOBAL_MULTIYEAR_PHY_001_030.
# `copernicusmarine get -i` expects a dataset ID.
if [[ -n "${GLORYS_DATASET_ID:-}" ]]; then
  # When user provides an explicit dataset ID, use only that value.
  GLORYS_DATASET_CANDIDATES=("${GLORYS_DATASET_ID}")
else
  GLORYS_DATASET_CANDIDATES=(
    "cmems_mod_glo_phy_my_0.083deg_P1M-m"
    "cmems_mod_glo_phy_my_0.083deg_P1M-m_202311"
    "global-reanalysis-001-030-monthly"
    "global-reanalysis-phy-001-030-monthly"
  )
fi
START_DATE="${START_DATE:-2010-01-01}"
END_DATE="${END_DATE:-$(date -u +%F)}"
OUTPUT_DIR="${1:-/data1/datasets/depth_v2/glorys_monthly}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

mkdir -p "${OUTPUT_DIR}"
CSV_LOG_PATH="${CSV_LOG_PATH:-${OUTPUT_DIR}/glorys_monthly_download_log.csv}"

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

echo "Using GLORYS dataset candidates: ${GLORYS_DATASET_CANDIDATES[*]}"
echo "Date range: ${START_DATE} .. ${END_DATE}"
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

start_month="$(date -u -d "${START_DATE}" +%Y-%m-01)"
end_month="$(date -u -d "${END_DATE}" +%Y-%m-01)"
# Count inclusive months between start/end anchors (YYYY-MM-01) for progress/ETA.
total_months="$(( ($(date -u -d "${end_month}" +%Y) - $(date -u -d "${start_month}" +%Y)) * 12 + $(date -u -d "${end_month}" +%m) - $(date -u -d "${start_month}" +%m) + 1 ))"
current_month="${start_month}"
processed_months=0
missing_months=0
query_errors=0
selected_dataset_id=""
selected_filter=""
downloaded_months=0
download_errors=0
run_start_epoch="$(date -u +%s)"

format_eta() {
  local total_seconds="$1"
  local hours="$(( total_seconds / 3600 ))"
  local minutes="$(( (total_seconds % 3600) / 60 ))"
  local seconds="$(( total_seconds % 60 ))"
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

while [[ "${current_month}" < "${end_month}" || "${current_month}" == "${end_month}" ]]; do
  processed_months=$((processed_months + 1))
  year="$(date -u -d "${current_month}" +%Y)"
  month="$(date -u -d "${current_month}" +%m)"
  # Monthly files embed YYYYMM in filename for GLORYS monthly streams.
  month_tag="$(date -u -d "${current_month}" +%Y%m)"
  # Try several path layouts because dataset hosting changed across catalog versions.
  glorys_filters=(
    "*/${year}/${month}/*${month_tag}*.nc"
    "*/${year}/*${month_tag}*.nc"
    "*${month_tag}*.nc"
  )

  dry_output=""
  dry_error=""
  used_dataset_id=""
  used_filter=""
  fallback_output=""
  fallback_dataset_id=""
  fallback_filter=""
  for candidate_id in "${GLORYS_DATASET_CANDIDATES[@]}"; do
    if [[ -z "${candidate_id}" ]]; then
      continue
    fi
    for candidate_filter in "${glorys_filters[@]}"; do
      err_tmp_file="$(mktemp)"
      cmd_output="$("${COPERNICUS_CMD}" get \
        -i "${candidate_id}" \
        --filter "${candidate_filter}" \
        --dry-run \
        --log-level ERROR 2>"${err_tmp_file}" || true)"
      cmd_error="$(cat "${err_tmp_file}")"
      rm -f "${err_tmp_file}"
      if file_count_candidate="$(printf "%s" "${cmd_output}" | "${PYTHON_CMD}" -c 'import json,sys; print(int((json.load(sys.stdin).get("number_of_files_to_download") or 0)))' 2>/dev/null)"; then
        # Keep a parsed fallback so we can still report "missing" instead of "query_failed".
        if [[ -z "${fallback_output}" ]]; then
          fallback_output="${cmd_output}"
          fallback_dataset_id="${candidate_id}"
          fallback_filter="${candidate_filter}"
        fi
        # Stop early only when this candidate/filter actually reports data.
        if [[ "${file_count_candidate}" -gt 0 ]]; then
          dry_output="${cmd_output}"
          used_dataset_id="${candidate_id}"
          used_filter="${candidate_filter}"
          break 2
        fi
      else
        dry_error="${cmd_error}"
      fi
    done
  done
  if [[ -z "${dry_output}" && -n "${fallback_output}" ]]; then
    dry_output="${fallback_output}"
    used_dataset_id="${fallback_dataset_id}"
    used_filter="${fallback_filter}"
  fi
  if [[ -z "${dry_output}" ]]; then
    echo "[${month_tag}] query_failed"
    if [[ -n "${dry_error}" ]]; then
      echo "  error: ${dry_error}"
    fi
    append_csv_row "" "" "${month_tag}" "query_failed"
    query_errors=$((query_errors + 1))
  else
    if [[ -n "${used_dataset_id}" && -z "${selected_dataset_id}" ]]; then
      selected_dataset_id="${used_dataset_id}"
      echo "  -> Selected dataset ID: ${selected_dataset_id}"
    fi
    if [[ -n "${used_filter}" && -z "${selected_filter}" ]]; then
      selected_filter="${used_filter}"
      echo "  -> Selected filter: ${selected_filter}"
    fi
    file_count="$(printf "%s" "${dry_output}" | "${PYTHON_CMD}" -c 'import json,sys; print(int((json.load(sys.stdin).get("number_of_files_to_download") or 0)))' 2>/dev/null || echo 0)"
    if [[ "${file_count}" -eq 0 ]]; then
      echo "[${month_tag}] missing"
      append_csv_row "" "" "${month_tag}" "missing"
      missing_months=$((missing_months + 1))
    elif [[ "${DRY_RUN_ONLY}" == "1" ]]; then
      echo "[${month_tag}] available_dry_run"
      append_csv_row "" "" "${month_tag}" "available_dry_run"
    else
      download_err_tmp="$(mktemp)"
      if "${COPERNICUS_CMD}" get \
        -i "${used_dataset_id}" \
        --filter "${used_filter}" \
        -o "${OUTPUT_DIR}" \
        -nd \
        --log-level ERROR >/dev/null 2>"${download_err_tmp}"; then
        shopt -s nullglob
        downloaded_files=("${OUTPUT_DIR}"/*"${month_tag}"*.nc)
        shopt -u nullglob
        if [[ ${#downloaded_files[@]} -gt 0 ]]; then
          downloaded_file="${downloaded_files[0]}"
          append_csv_row "$(basename "${downloaded_file}")" "${downloaded_file}" "${month_tag}" "downloaded"
          echo "[${month_tag}] downloaded"
        else
          append_csv_row "" "" "${month_tag}" "downloaded_but_file_not_found"
          echo "[${month_tag}] downloaded_but_file_not_found"
        fi
        downloaded_months=$((downloaded_months + 1))
      else
        echo "[${month_tag}] download_failed"
        download_err_msg="$(cat "${download_err_tmp}")"
        if [[ -n "${download_err_msg}" ]]; then
          echo "  error: ${download_err_msg}"
        fi
        append_csv_row "" "" "${month_tag}" "download_failed"
        download_errors=$((download_errors + 1))
      fi
      rm -f "${download_err_tmp}"
    fi
  fi

  elapsed_seconds="$(( $(date -u +%s) - run_start_epoch ))"
  remaining_months="$(( total_months - processed_months ))"
  if [[ "${processed_months}" -gt 0 ]]; then
    eta_seconds="$(( elapsed_seconds * remaining_months / processed_months ))"
  else
    eta_seconds=0
  fi
  echo "  -> Progress: ${processed_months}/${total_months} months processed, ${downloaded_months} downloaded | ETA: $(format_eta "${eta_seconds}")"

  current_month="$(date -u -d "${current_month} + 1 month" +%Y-%m-01)"
done

echo
echo "Summary"
echo "- Months scanned: ${total_months}"
echo "- Missing months: ${missing_months}"
echo "- Query errors: ${query_errors}"
echo "- Months downloaded: ${downloaded_months}"
echo "- Download errors: ${download_errors}"
echo "- Dry-run-only mode: ${DRY_RUN_ONLY}"

echo
echo "Done."
