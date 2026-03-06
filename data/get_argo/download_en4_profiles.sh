#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1}"
START_YEAR="${START_YEAR:-2010}"
END_YEAR="${END_YEAR:-2025}"
OUTPUT_DIR="${1:-./downloads/en4_profiles}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
CSV_LOG_PATH="${CSV_LOG_PATH:-${OUTPUT_DIR}/en4_profiles_download_log.csv}"

mkdir -p "${OUTPUT_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required but not found in PATH." >&2
  exit 1
fi

if ! [[ "${START_YEAR}" =~ ^[0-9]{4}$ && "${END_YEAR}" =~ ^[0-9]{4}$ ]]; then
  echo "Error: START_YEAR and END_YEAR must be 4-digit years." >&2
  exit 1
fi
if [[ "${END_YEAR}" -lt "${START_YEAR}" ]]; then
  echo "Error: END_YEAR (${END_YEAR}) must be >= START_YEAR (${START_YEAR})." >&2
  exit 1
fi

echo "EN4 base URL: ${BASE_URL}"
echo "Year range: ${START_YEAR} .. ${END_YEAR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dry-run-only mode: ${DRY_RUN_ONLY} (must be 0 for real downloads)"
echo "CSV log: ${CSV_LOG_PATH}"
echo

if [[ ! -f "${CSV_LOG_PATH}" ]]; then
  echo "filename,path,datetime,status,expected_bytes,downloaded_bytes,duration_seconds,avg_mb_per_s" > "${CSV_LOG_PATH}"
fi

append_csv_row() {
  local filename="$1"
  local path="$2"
  local day="$3"
  local status="$4"
  local expected_bytes="$5"
  local downloaded_bytes="$6"
  local duration_seconds="$7"
  local avg_mb_per_s="$8"
  # Keep CSV appends centralized so status strings are consistent.
  printf '"%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "${filename}" "${path}" "${day}" "${status}" "${expected_bytes}" "${downloaded_bytes}" "${duration_seconds}" "${avg_mb_per_s}" >> "${CSV_LOG_PATH}"
}

format_eta() {
  local total_seconds="$1"
  local hours="$(( total_seconds / 3600 ))"
  local minutes="$(( (total_seconds % 3600) / 60 ))"
  local seconds="$(( total_seconds % 60 ))"
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

total_items="$(( END_YEAR - START_YEAR + 1 ))"
processed_items=0
available_items=0
downloaded_items=0
missing_items=0
query_errors=0
download_errors=0
run_start_epoch="$(date -u +%s)"

for year in $(seq "${START_YEAR}" "${END_YEAR}"); do
  processed_items=$((processed_items + 1))
  filename="EN.4.2.2.profiles.g10.${year}.zip"
  item_url="${BASE_URL}/${filename}"
  item_date="${year}-01-01"
  output_path="${OUTPUT_DIR}/${filename}"

  # Dry-run check for current item only (HEAD); download immediately if available.
  head_tmp="$(mktemp)"
  if curl --silent --show-error --fail --location --head "${item_url}" >"${head_tmp}" 2>/dev/null; then
    expected_bytes="$(awk 'tolower($1)=="content-length:"{size=$2} END{gsub("\r","",size); print size}' "${head_tmp}")"
    if ! [[ "${expected_bytes}" =~ ^[0-9]+$ ]]; then
      expected_bytes=""
    fi
    available_items=$((available_items + 1))
    if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
      echo "[${year}] available_dry_run"
      append_csv_row "${filename}" "" "${item_date}" "available_dry_run" "${expected_bytes}" "" "" ""
    else
      tmp_output="${output_path}.part"
      download_err_tmp="$(mktemp)"
      download_start_epoch="$(date -u +%s)"
      # Run download in background so we can print custom size/speed/ETA updates.
      curl --silent --show-error --fail --location "${item_url}" -o "${tmp_output}" 2>"${download_err_tmp}" &
      download_pid="$!"
      while kill -0 "${download_pid}" >/dev/null 2>&1; do
        sleep 1
        downloaded_bytes="$(stat -c%s "${tmp_output}" 2>/dev/null || echo 0)"
        elapsed="$(( $(date -u +%s) - download_start_epoch ))"
        if [[ "${elapsed}" -lt 1 ]]; then
          elapsed=1
        fi
        avg_mb_per_s="$(awk -v b="${downloaded_bytes}" -v t="${elapsed}" 'BEGIN { printf "%.2f", (b/1048576)/t }')"
        if [[ -n "${expected_bytes}" && "${expected_bytes}" -gt 0 ]]; then
          eta_seconds="$(awk -v total="${expected_bytes}" -v done="${downloaded_bytes}" -v t="${elapsed}" 'BEGIN { sp=done/t; if (sp <= 0) print 0; else { rem=total-done; if (rem < 0) rem=0; print int(rem/sp); } }')"
          eta_text="$(format_eta "${eta_seconds}")"
          done_mb="$(awk -v b="${downloaded_bytes}" 'BEGIN { printf "%.2f", b/1048576 }')"
          total_mb="$(awk -v b="${expected_bytes}" 'BEGIN { printf "%.2f", b/1048576 }')"
          printf "\r[%s] downloading %s/%s MB | %s MB/s | ETA %s" "${year}" "${done_mb}" "${total_mb}" "${avg_mb_per_s}" "${eta_text}"
        else
          done_mb="$(awk -v b="${downloaded_bytes}" 'BEGIN { printf "%.2f", b/1048576 }')"
          printf "\r[%s] downloading %s MB | %s MB/s | ETA unknown" "${year}" "${done_mb}" "${avg_mb_per_s}"
        fi
      done
      if wait "${download_pid}"; then
        printf "\n"
        mv -f "${tmp_output}" "${output_path}"
        echo "[${year}] downloaded"
        final_bytes="$(stat -c%s "${output_path}" 2>/dev/null || echo 0)"
        duration_seconds="$(( $(date -u +%s) - download_start_epoch ))"
        if [[ "${duration_seconds}" -lt 1 ]]; then
          duration_seconds=1
        fi
        avg_mb_per_s="$(awk -v b="${final_bytes}" -v t="${duration_seconds}" 'BEGIN { printf "%.2f", (b/1048576)/t }')"
        append_csv_row "${filename}" "${output_path}" "${item_date}" "downloaded" "${expected_bytes}" "${final_bytes}" "${duration_seconds}" "${avg_mb_per_s}"
        downloaded_items=$((downloaded_items + 1))
      else
        printf "\n"
        rm -f "${tmp_output}"
        echo "[${year}] download_failed"
        append_csv_row "${filename}" "" "${item_date}" "download_failed" "${expected_bytes}" "" "" ""
        err_msg="$(cat "${download_err_tmp}")"
        if [[ -n "${err_msg}" ]]; then
          echo "  error: ${err_msg}"
        fi
        download_errors=$((download_errors + 1))
      fi
      rm -f "${download_err_tmp}"
    fi
  else
    echo "[${year}] missing_or_query_failed"
    append_csv_row "${filename}" "" "${item_date}" "missing_or_query_failed" "" "" "" ""
    missing_items=$((missing_items + 1))
    query_errors=$((query_errors + 1))
  fi
  rm -f "${head_tmp}"

  elapsed_seconds="$(( $(date -u +%s) - run_start_epoch ))"
  remaining_items="$(( total_items - processed_items ))"
  if [[ "${processed_items}" -gt 0 ]]; then
    eta_seconds="$(( elapsed_seconds * remaining_items / processed_items ))"
  else
    eta_seconds=0
  fi
  echo "  -> Progress: ${processed_items}/${total_items}, ${downloaded_items} downloaded | ETA: $(format_eta "${eta_seconds}")"
done

echo
echo "Summary"
echo "- Items scanned: ${total_items}"
echo "- Items available: ${available_items}"
echo "- Items downloaded: ${downloaded_items}"
echo "- Missing/query_failed: ${missing_items}"
echo "- Query errors: ${query_errors}"
echo "- Download errors: ${download_errors}"
echo "- Dry-run-only mode: ${DRY_RUN_ONLY}"
echo "Done."
