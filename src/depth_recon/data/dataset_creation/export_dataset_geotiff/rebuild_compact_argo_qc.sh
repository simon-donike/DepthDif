#!/usr/bin/env bash
# Example:
# src/depth_recon/data/dataset_creation/export_dataset_geotiff/rebuild_compact_argo_qc.sh \
#   --dataset-root /work/data/OceanVariableReconstruction \
#   --enriched-argo-zarr /work/data/depthdif/enriched_argo_profiles.zarr \
#   --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
#   --ostia-dir /data1/datasets/depth_v2/ostia \
#   --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
#   --sss-dir /data1/datasets/depth_v2/sss_daily \
#   --argo-dir /data1/datasets/depth_v2/en4_profiles \
#   --land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
#   --start-date 20100101 \
#   --end-date 20240731 \
#   --surface-aggregate-days 7 \
#   --chunk-profile 100000 \
#   --workers 12 \
#   --backup-suffix pre_qc_backup

set -euo pipefail

PYTHON_BIN="/work/envs/depth/bin/python"
DATASET_ROOT="/work/data/OceanVariableReconstruction"
ENRICHED_ARGO_ZARR="/work/data/depthdif/enriched_argo_profiles.zarr"
GLORYS_DIR="/data1/datasets/depth_v2/glorys_weekly"
OSTIA_DIR="/data1/datasets/depth_v2/ostia"
SEALEVEL_DIR="/data1/datasets/depth_v2/sealevel_daily"
SSS_DIR="/data1/datasets/depth_v2/sss_daily"
ARGO_DIR="/data1/datasets/depth_v2/en4_profiles"
LAND_MASK_PATH="src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif"
START_DATE="20100101"
END_DATE="20240731"
SURFACE_AGGREGATE_DAYS="7"
CHUNK_PROFILE="100000"
WORKERS="12"
BACKUP_SUFFIX="pre_qc_backup"

usage() {
  sed -n '2,18p' "$0" | sed 's/^# //'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --enriched-argo-zarr) ENRICHED_ARGO_ZARR="$2"; shift 2 ;;
    --glorys-dir) GLORYS_DIR="$2"; shift 2 ;;
    --ostia-dir) OSTIA_DIR="$2"; shift 2 ;;
    --sealevel-dir) SEALEVEL_DIR="$2"; shift 2 ;;
    --sss-dir) SSS_DIR="$2"; shift 2 ;;
    --argo-dir) ARGO_DIR="$2"; shift 2 ;;
    --land-mask-path) LAND_MASK_PATH="$2"; shift 2 ;;
    --start-date) START_DATE="$2"; shift 2 ;;
    --end-date) END_DATE="$2"; shift 2 ;;
    --surface-aggregate-days) SURFACE_AGGREGATE_DAYS="$2"; shift 2 ;;
    --chunk-profile) CHUNK_PROFILE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --backup-suffix) BACKUP_SUFFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

COMPACT_ARGO_ZARR="${DATASET_ROOT}/argo/argo_profiles_on_grid.zarr"
BACKUP_ARGO_ZARR="${DATASET_ROOT}/argo/argo_profiles_on_grid.${BACKUP_SUFFIX}.zarr"

restore_backup() {
  local failed_path
  failed_path="${DATASET_ROOT}/argo/argo_profiles_on_grid.failed_$(date +%Y%m%d_%H%M%S).zarr"
  echo
  echo "Rebuild failed; restoring previous compact ARGO store."
  if [[ -e "${COMPACT_ARGO_ZARR}" ]]; then
    echo "Moving failed partial output to: ${failed_path}"
    mv "${COMPACT_ARGO_ZARR}" "${failed_path}"
  fi
  if [[ -e "${BACKUP_ARGO_ZARR}" ]]; then
    mv "${BACKUP_ARGO_ZARR}" "${COMPACT_ARGO_ZARR}"
    echo "Restored: ${COMPACT_ARGO_ZARR}"
  fi
}

echo "Checking enriched ARGO QC fields: ${ENRICHED_ARGO_ZARR}"
"${PYTHON_BIN}" -c "import xarray as xr; p='${ENRICHED_ARGO_ZARR}'; ds=xr.open_zarr(p, consolidated=None); qc=[k for k in ds.data_vars if k.endswith('_qc') or '_qc_on_glorys_depth' in k]; print('QC field count:', len(qc)); print('QC fields:', qc[:20]); print('profiles:', ds.sizes.get('profile')); ds.close(); assert qc, 'No QC fields found in enriched ARGO zarr'"

if [[ ! -d "${COMPACT_ARGO_ZARR}" ]]; then
  echo "Compact ARGO store does not exist: ${COMPACT_ARGO_ZARR}" >&2
  exit 1
fi
if [[ -e "${BACKUP_ARGO_ZARR}" ]]; then
  echo "Backup already exists: ${BACKUP_ARGO_ZARR}" >&2
  echo "Use --backup-suffix with a new suffix, or move/remove the existing backup." >&2
  exit 1
fi

echo "Moving existing compact ARGO store to backup:"
echo "  ${COMPACT_ARGO_ZARR}"
echo "  ${BACKUP_ARGO_ZARR}"
mv "${COMPACT_ARGO_ZARR}" "${BACKUP_ARGO_ZARR}"
trap restore_backup ERR

echo "Regenerating compact ARGO store while skipping existing rasters."
"${PYTHON_BIN}" -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff \
  --glorys-dir "${GLORYS_DIR}" \
  --ostia-dir "${OSTIA_DIR}" \
  --sealevel-dir "${SEALEVEL_DIR}" \
  --sss-dir "${SSS_DIR}" \
  --enriched-argo-zarr "${ENRICHED_ARGO_ZARR}" \
  --argo-dir "${ARGO_DIR}" \
  --land-mask-path "${LAND_MASK_PATH}" \
  --output-dir "${DATASET_ROOT}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --surface-aggregate-days "${SURFACE_AGGREGATE_DAYS}" \
  --argo-source enriched \
  --chunk-profile "${CHUNK_PROFILE}" \
  --workers "${WORKERS}" \
  --skip-existing

echo "Verifying regenerated compact ARGO store."
"${PYTHON_BIN}" -c "import xarray as xr; p='${COMPACT_ARGO_ZARR}'; ds=xr.open_zarr(p, consolidated=None); required=['target_date','grid_row','grid_col','argo_temp_kelvin_uint8','argo_temp_valid','argo_psal_uint8','argo_psal_valid']; missing=[k for k in required if k not in ds]; qc=[k for k in ds.data_vars if k.endswith('_qc') or '_qc_on_glorys_depth' in k]; print('profiles:', ds.sizes.get('profile')); print('depths:', ds.sizes.get('glorys_depth')); print('missing required:', missing); print('QC field count:', len(qc)); print('QC fields:', qc[:20]); ds.close(); assert not missing; assert qc"

echo "Verifying dataloader QC settings and store opening."
"${PYTHON_BIN}" -c "from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset; ds=ArgoGeoTIFFGriddedPatchDataset.from_config('src/depth_recon/configs/px_space/training_super_config.yaml', split='val'); print('rows:', len(ds)); print('filter_bad_argo_quality:', ds.filter_bad_argo_quality); print('accepted_argo_qc_flags:', ds.accepted_argo_qc_flags); print('argo profiles indexed:', ds.argo_store.target_date.size); ds.argo_store.close(); ds.raster_cache.close(); assert ds.filter_bad_argo_quality; assert ds.accepted_argo_qc_flags == (1, 2)"

trap - ERR
echo
echo "Done. Backup retained at: ${BACKUP_ARGO_ZARR}"
