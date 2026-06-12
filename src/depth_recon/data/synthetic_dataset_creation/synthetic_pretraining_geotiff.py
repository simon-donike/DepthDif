"""
Example:
 /work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.synthetic_pretraining_geotiff \
   --geotiff-root-dir /work/data/OceanVariableReconstruction \
   --workers 4 \
   --overwrite-synthetic

Create SST/SSS-guided synthetic pretraining target rasters beside a packaged
GeoTIFF training dataset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from tqdm import tqdm
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import (
    ArgoGeoTIFFProfileStore,
    _resolve_manifest_path,
)
from depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff import (
    SALINITY_STRETCH,
    STRETCH_SPECS,
    TEMPERATURE_KELVIN_STRETCH,
    EncodeStats,
    TargetGrid,
    _existing_raster_metadata,
    _load_target_grid,
    _merge_stats,
    _output_relative,
    _set_common_tags,
    _stretch_manifest,
    _write_geotiff_with_fallback,
    _yaml_safe,
    decode_stretched_uint8,
    encode_stretched_uint8,
)

NODATA_CODE = 255
DEFAULT_IDW_K = 8
DEFAULT_IDW_POWER = 2.0
DEFAULT_IDW_BLOCK_ROWS = 128
DEFAULT_SALINITY_SURFACE_SMOOTHING_SIGMA = 0.0
STRATEGY_NAME = "vertical_delta_sss_v4"


@dataclass(frozen=True)
class SyntheticDateResult:
    """Manifest-ready output summary for one generated date."""

    date: int
    thetao: dict[str, Any] | None
    so: dict[str, Any] | None
    skipped: dict[str, Any] | None


def _records_by_date(
    entries: Sequence[dict[str, Any]], root_dir: Path
) -> dict[int, Path]:
    """Return a date-to-path mapping for manifest raster entries."""
    records: dict[int, Path] = {}
    for entry in entries:
        records[int(entry["date"])] = _resolve_manifest_path(root_dir, entry["path"])
    return records


def _read_raster_band(
    path: Path, stretch_name: str, *, band_index: int = 1
) -> np.ndarray:
    """Read one encoded raster band into physical units."""
    with rasterio.open(path) as src:
        encoded = src.read(int(band_index))
    return decode_stretched_uint8(encoded, STRETCH_SPECS[stretch_name]).astype(
        np.float32,
        copy=False,
    )


def _read_ocean_mask(path: Path) -> np.ndarray:
    """Read a land-mask GeoTIFF as a boolean ocean mask."""
    with rasterio.open(path) as src:
        values = src.read(1)
    return np.asarray(values <= 0, dtype=bool)


def _nan_gaussian_filter(values: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter while ignoring NaN values."""
    arr = np.asarray(values, dtype=np.float32)
    if float(sigma) <= 0.0:
        return arr.copy()
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0).astype(np.float32, copy=False)
    weights = valid.astype(np.float32, copy=False)
    smooth_values = ndi.gaussian_filter(filled, sigma=float(sigma), mode="nearest")
    smooth_weights = ndi.gaussian_filter(weights, sigma=float(sigma), mode="nearest")
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    keep = smooth_weights > np.float32(1.0e-6)
    out[keep] = smooth_values[keep] / smooth_weights[keep]
    return out


def _depth_smoothing_sigma(depth_m: float) -> float:
    """Return final synthetic-field smoothing sigma for one depth."""
    depth = float(depth_m)
    if depth <= 200.0:
        return 0.0
    if depth <= 700.0:
        return 0.5
    if depth <= 1500.0:
        return 1.0
    return 1.5


def _read_glorys_valid_mask(
    path: Path, *, band_count: int, grid: TargetGrid
) -> np.ndarray:
    """Read the GLORYS uint8 nodata layout as a per-band valid mask."""
    with rasterio.open(path) as src:
        if src.count != int(band_count):
            raise RuntimeError(
                f"Expected {band_count} GLORYS bands in {path}, got {src.count}."
            )
        if src.height != int(grid.height) or src.width != int(grid.width):
            raise RuntimeError(f"GLORYS mask grid does not match target grid: {path}")
        encoded = src.read()
    return np.asarray(encoded != NODATA_CODE, dtype=bool)


def _fill_missing_nearest(values: np.ndarray, fill_mask: np.ndarray) -> np.ndarray:
    """Fill missing guide-raster pixels inside a target mask from nearest valid pixels."""
    arr = np.asarray(values, dtype=np.float32).copy()
    target = np.asarray(fill_mask, dtype=bool)
    valid = np.isfinite(arr)
    missing = target & ~valid
    if not np.any(missing) or not np.any(valid):
        return arr
    # Distance transform returns indices of the nearest finite guide pixel for each gap.
    nearest = ndi.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    arr[missing] = arr[tuple(index[missing] for index in nearest)]
    return arr


def _profile_surface_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each profile's shallowest valid value and profile-valid mask."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected profile values shaped (N,D), got {arr.shape}.")
    valid = np.isfinite(arr)
    has_valid = valid.any(axis=1)
    first_valid = np.argmax(valid, axis=1)
    surface = np.full((arr.shape[0],), np.nan, dtype=np.float32)
    if np.any(has_valid):
        profile_indices = np.arange(arr.shape[0], dtype=np.int64)[has_valid]
        surface[has_valid] = arr[profile_indices, first_valid[has_valid]]
    return surface, has_valid


def _idw_interpolate_grid(
    *,
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    point_values: np.ndarray,
    shape: tuple[int, int],
    k: int,
    power: float,
    block_rows: int,
) -> np.ndarray:
    """Interpolate point values to a full raster grid with inverse-distance weights."""
    rows = np.asarray(point_rows, dtype=np.float64).reshape(-1)
    cols = np.asarray(point_cols, dtype=np.float64).reshape(-1)
    values = np.asarray(point_values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(rows) & np.isfinite(cols) & np.isfinite(values)
    if not np.any(valid):
        return np.full(shape, np.nan, dtype=np.float32)

    rows = rows[valid]
    cols = cols[valid]
    values = values[valid]
    tree = cKDTree(np.column_stack([rows, cols]))
    k_eff = max(1, min(int(k), int(values.size)))
    out = np.full(shape, np.nan, dtype=np.float32)
    height, width = int(shape[0]), int(shape[1])
    block_size = max(1, int(block_rows))
    eps = np.finfo(np.float32).eps

    for y0 in range(0, height, block_size):
        y1 = min(height, y0 + block_size)
        yy, xx = np.indices((y1 - y0, width), dtype=np.float64)
        query = np.column_stack([(yy.reshape(-1) + y0), xx.reshape(-1)])
        distances, indices = tree.query(query, k=k_eff)
        if k_eff == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        neighbor_values = values[indices]
        zero_distance = distances <= eps
        weights = np.power(np.maximum(distances, eps), -float(power))
        weighted = np.sum(weights * neighbor_values, axis=1) / np.sum(weights, axis=1)
        exact_rows = np.any(zero_distance, axis=1)
        if np.any(exact_rows):
            exact_indices = np.argmax(zero_distance[exact_rows], axis=1)
            weighted[exact_rows] = neighbor_values[exact_rows, exact_indices]
        out[y0:y1, :] = weighted.reshape(y1 - y0, width).astype(
            np.float32,
            copy=False,
        )
    return out


def _profile_indices_for_date(
    store: ArgoGeoTIFFProfileStore,
    date_value: int,
) -> np.ndarray:
    """Return compact ARGO profile indices assigned to one target date."""
    bounds = store._profile_index_bounds_by_date.get(int(date_value))
    if bounds is None:
        return np.zeros((0,), dtype=np.int64)
    start, stop = bounds
    return store._valid_profile_indices_by_date[int(start) : int(stop)].astype(
        np.int64,
        copy=False,
    )


def _write_synthetic_multiband_raster(
    *,
    path: Path,
    output_dir: Path,
    values: np.ndarray,
    date_value: int,
    grid: TargetGrid,
    variable: str,
    stretch_name: str,
    depth_axis_m: np.ndarray,
    overwrite_synthetic: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    """Write one synthetic multiband raster with the same uint8 contract as GLORYS."""
    stretch = STRETCH_SPECS[stretch_name]
    depth_axis = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
    if path.exists() and not overwrite_synthetic:
        if skip_existing:
            compression, stats = _existing_raster_metadata(
                path,
                count=int(depth_axis.size),
                grid=grid,
            )
            return {
                "date": int(date_value),
                "path": _output_relative(path, output_dir),
                "source": "synthetic_pretraining",
                "compression": compression,
                "band_count": int(depth_axis.size),
                "stats": stats,
                "skipped_existing": True,
            }
        raise FileExistsError(f"Synthetic raster already exists: {path}")

    per_band_stats: list[EncodeStats] = []

    def writer(dst: rasterio.io.DatasetWriter) -> None:
        """Write encoded depth bands into an opened GeoTIFF."""
        per_band_stats.clear()
        _set_common_tags(
            dst,
            source_product="synthetic",
            variable=variable,
            stretch=stretch,
        )
        dst.update_tags(
            date=int(date_value),
            source_formula=STRATEGY_NAME,
        )
        for band_idx, depth_value in enumerate(depth_axis.tolist(), start=1):
            encoded, stats = encode_stretched_uint8(values[band_idx - 1], stretch)
            per_band_stats.append(stats)
            dst.write(encoded, band_idx)
            dst.set_band_description(band_idx, f"depth_{float(depth_value):g}_m")
            dst.update_tags(
                band_idx,
                depth_m=float(depth_value),
                **_stretch_manifest(stretch),
                **{
                    "valid_count": stats.valid_count,
                    "nodata_count": stats.nodata_count,
                    "clipped_low_count": stats.clipped_low_count,
                    "clipped_high_count": stats.clipped_high_count,
                },
            )

    compression = _write_geotiff_with_fallback(
        path,
        count=int(depth_axis.size),
        grid=grid,
        writer=writer,
    )
    return {
        "date": int(date_value),
        "path": _output_relative(path, output_dir),
        "source": "synthetic_pretraining",
        "compression": compression,
        "band_count": int(depth_axis.size),
        "stats": _merge_stats(per_band_stats),
    }


def _export_synthetic_date(task: dict[str, Any]) -> dict[str, Any]:
    """Generate and write synthetic temperature/salinity rasters for one date."""
    root_dir = Path(task["geotiff_root_dir"])
    manifest = task["manifest"]
    date_value = int(task["date"])
    grid = _load_target_grid(Path(task["land_mask_path"]))
    ocean_mask = _read_ocean_mask(Path(task["land_mask_path"]))
    depth_axis = np.asarray(task["depth_axis_m"], dtype=np.float32).reshape(-1)
    argo_path = _resolve_manifest_path(root_dir, manifest["argo"]["path"])
    profile_store = ArgoGeoTIFFProfileStore(argo_path, include_salinity=True)
    try:
        indices = _profile_indices_for_date(profile_store, date_value)
        if indices.size == 0:
            return SyntheticDateResult(
                date=date_value,
                thetao=None,
                so=None,
                skipped={"reason": "no_argo_profiles", "argo_profile_count": 0},
            ).__dict__

        rows = profile_store.grid_row[indices].astype(np.int64, copy=False)
        cols = profile_store.grid_col[indices].astype(np.int64, copy=False)
        temp_kelvin = profile_store.load_temperature_profiles(indices) + np.float32(
            273.15
        )
        salinity = profile_store.load_salinity_profiles(indices)

        sst = _read_raster_band(
            Path(task["ostia_path"]),
            TEMPERATURE_KELVIN_STRETCH,
        )
        sss = _read_raster_band(Path(task["sss_path"]), SALINITY_STRETCH)
        thetao_valid_mask = _read_glorys_valid_mask(
            Path(task["glorys_thetao_path"]),
            band_count=int(depth_axis.size),
            grid=grid,
        )
        so_valid_mask = _read_glorys_valid_mask(
            Path(task["glorys_so_path"]),
            band_count=int(depth_axis.size),
            grid=grid,
        )
        sst = _fill_missing_nearest(sst, ocean_mask & thetao_valid_mask.any(axis=0))
        sss = _fill_missing_nearest(sss, ocean_mask & so_valid_mask.any(axis=0))

        temperature = np.full(
            (int(depth_axis.size), int(grid.height), int(grid.width)),
            np.nan,
            dtype=np.float32,
        )
        salinity_out = np.full_like(temperature, np.nan)

        temp_surface, temp_profile_valid = _profile_surface_values(temp_kelvin)
        sal_surface, sal_profile_valid = _profile_surface_values(salinity)
        _ = temp_profile_valid
        _ = sal_profile_valid

        sal_surface_prior = _nan_gaussian_filter(
            sss,
            float(task["salinity_surface_smoothing_sigma"]),
        )

        for depth_idx, depth_value in enumerate(depth_axis.tolist()):
            temp_delta = temp_kelvin[:, depth_idx] - temp_surface
            if np.any(np.isfinite(temp_delta)):
                temp_delta_field = _idw_interpolate_grid(
                    point_rows=rows,
                    point_cols=cols,
                    point_values=temp_delta,
                    shape=(int(grid.height), int(grid.width)),
                    k=int(task["idw_k"]),
                    power=float(task["idw_power"]),
                    block_rows=int(task["idw_block_rows"]),
                )
            else:
                temp_delta_field = np.zeros(
                    (int(grid.height), int(grid.width)), dtype=np.float32
                )
            temp_field = sst + temp_delta_field
            temp_field = _nan_gaussian_filter(
                temp_field,
                _depth_smoothing_sigma(float(depth_value)),
            )
            temp_mask = thetao_valid_mask[depth_idx]
            temperature[depth_idx][temp_mask & np.isfinite(temp_field)] = temp_field[
                temp_mask & np.isfinite(temp_field)
            ]

            sal_delta = salinity[:, depth_idx] - sal_surface
            if np.any(np.isfinite(sal_delta)):
                sal_delta_field = _idw_interpolate_grid(
                    point_rows=rows,
                    point_cols=cols,
                    point_values=sal_delta,
                    shape=(int(grid.height), int(grid.width)),
                    k=int(task["idw_k"]),
                    power=float(task["idw_power"]),
                    block_rows=int(task["idw_block_rows"]),
                )
            else:
                sal_delta_field = np.zeros(
                    (int(grid.height), int(grid.width)), dtype=np.float32
                )
            sal_field = sal_surface_prior + sal_delta_field
            sal_field = _nan_gaussian_filter(
                sal_field,
                _depth_smoothing_sigma(float(depth_value)),
            )
            sal_mask = so_valid_mask[depth_idx]
            salinity_out[depth_idx][sal_mask & np.isfinite(sal_field)] = sal_field[
                sal_mask & np.isfinite(sal_field)
            ]

        thetao_path = (
            root_dir / "rasters" / "synthetic" / "thetao" / f"thetao_{date_value}.tif"
        )
        so_path = root_dir / "rasters" / "synthetic" / "so" / f"so_{date_value}.tif"
        thetao_result = _write_synthetic_multiband_raster(
            path=thetao_path,
            output_dir=root_dir,
            values=temperature,
            date_value=date_value,
            grid=grid,
            variable="thetao",
            stretch_name=TEMPERATURE_KELVIN_STRETCH,
            depth_axis_m=depth_axis,
            overwrite_synthetic=bool(task["overwrite_synthetic"]),
            skip_existing=bool(task["skip_existing"]),
        )
        so_result = _write_synthetic_multiband_raster(
            path=so_path,
            output_dir=root_dir,
            values=salinity_out,
            date_value=date_value,
            grid=grid,
            variable="so",
            stretch_name=SALINITY_STRETCH,
            depth_axis_m=depth_axis,
            overwrite_synthetic=bool(task["overwrite_synthetic"]),
            skip_existing=bool(task["skip_existing"]),
        )
        result = SyntheticDateResult(
            date=date_value,
            thetao=thetao_result,
            so=so_result,
            skipped=None,
        ).__dict__
        result["profile_count"] = int(indices.size)
        return result
    finally:
        profile_store.close()


def _copy_manifest_backup(manifest_path: Path) -> Path:
    """Create a timestamped manifest backup before in-place updates."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = manifest_path.with_name(
        f"{manifest_path.name}.synthetic_backup_{stamp}"
    )
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def export_synthetic_pretraining_geotiff_dataset(
    *,
    geotiff_root_dir: str | Path,
    target_dates: Sequence[int] | None = None,
    idw_k: int = DEFAULT_IDW_K,
    idw_power: float = DEFAULT_IDW_POWER,
    idw_block_rows: int = DEFAULT_IDW_BLOCK_ROWS,
    salinity_surface_smoothing_sigma: float = DEFAULT_SALINITY_SURFACE_SMOOTHING_SIGMA,
    workers: int = 1,
    overwrite_synthetic: bool = False,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> Path:
    """Write synthetic pretraining rasters into an existing GeoTIFF dataset root."""
    root_dir = Path(geotiff_root_dir)
    manifest_path = root_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"GeoTIFF manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    argo_info = manifest.get("argo", {})
    if not argo_info.get("path"):
        raise RuntimeError("Manifest has no compact ARGO profile store path.")
    depth_axis = np.asarray(manifest.get("depth_axis_m", ()), dtype=np.float32)
    if depth_axis.size == 0:
        raise RuntimeError("Manifest is missing depth_axis_m.")

    rasters = manifest.get("rasters", {})
    ostia_paths = _records_by_date(
        rasters.get("ostia", {}).get("analysed_sst", []),
        root_dir,
    )
    sss_paths = _records_by_date(rasters.get("sss", {}).get("sos", []), root_dir)
    glorys_thetao_paths = _records_by_date(
        rasters.get("glorys", {}).get("thetao", []), root_dir
    )
    glorys_so_paths = _records_by_date(
        rasters.get("glorys", {}).get("so", []), root_dir
    )
    requested_dates = (
        [int(value) for value in target_dates]
        if target_dates is not None
        else [int(value) for value in manifest.get("target_dates", [])]
    )
    if not requested_dates:
        raise RuntimeError("No target dates were provided or found in manifest.")

    land_mask_path = _resolve_manifest_path(root_dir, manifest["grid"]["source"])
    runnable_dates = [
        date_value
        for date_value in requested_dates
        if date_value in ostia_paths
        and date_value in sss_paths
        and date_value in glorys_thetao_paths
        and date_value in glorys_so_paths
    ]
    missing_guides = [
        int(date_value)
        for date_value in requested_dates
        if (date_value not in ostia_paths or date_value not in sss_paths)
    ]
    if not runnable_dates:
        raise RuntimeError(
            "No requested dates have OSTIA SST, SSS, and GLORYS target mask rasters."
        )

    worker_count = max(1, int(workers))
    skip_existing = bool(skip_existing and not overwrite_synthetic)
    tasks = [
        {
            "geotiff_root_dir": str(root_dir),
            "manifest": manifest,
            "date": int(date_value),
            "land_mask_path": str(land_mask_path),
            "depth_axis_m": depth_axis.tolist(),
            "ostia_path": str(ostia_paths[int(date_value)]),
            "sss_path": str(sss_paths[int(date_value)]),
            "glorys_thetao_path": str(glorys_thetao_paths[int(date_value)]),
            "glorys_so_path": str(glorys_so_paths[int(date_value)]),
            "idw_k": int(idw_k),
            "idw_power": float(idw_power),
            "idw_block_rows": int(idw_block_rows),
            "salinity_surface_smoothing_sigma": float(salinity_surface_smoothing_sigma),
            "overwrite_synthetic": bool(overwrite_synthetic),
            "skip_existing": bool(skip_existing),
        }
        for date_value in runnable_dates
    ]

    results: list[dict[str, Any]] = []
    if worker_count == 1:
        iterator = tqdm(
            tasks,
            desc="Generating synthetic GeoTIFF dates",
            unit="date",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for task in iterator:
            results.append(_export_synthetic_date(task))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_date = {
                executor.submit(_export_synthetic_date, task): int(task["date"])
                for task in tasks
            }
            for future in tqdm(
                as_completed(future_to_date),
                total=len(future_to_date),
                desc=f"Generating synthetic GeoTIFF dates ({worker_count} workers)",
                unit="date",
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                results.append(future.result())

    results_by_date = {int(result["date"]): result for result in results}
    thetao_entries = []
    so_entries = []
    skipped_dates = [
        {"date": int(date), "reason": "missing_guiding_raster"}
        for date in missing_guides
    ]
    argo_profile_counts: dict[int, int] = {}
    for date_value in runnable_dates:
        result = results_by_date[int(date_value)]
        skipped = result.get("skipped")
        if skipped is not None:
            skipped_dates.append({"date": int(date_value), **dict(skipped)})
            argo_profile_counts[int(date_value)] = 0
            continue
        thetao_entries.append(result["thetao"])
        so_entries.append(result["so"])
        argo_profile_counts[int(date_value)] = int(result.get("profile_count", 0))

    backup_path = _copy_manifest_backup(manifest_path)
    manifest.setdefault("rasters", {}).setdefault("synthetic", {})
    manifest["rasters"]["synthetic"]["thetao"] = thetao_entries
    manifest["rasters"]["synthetic"]["so"] = so_entries
    manifest["synthetic_pretraining"] = {
        "created_by": "depth_recon.data.synthetic_dataset_creation.synthetic_pretraining_geotiff",
        "strategy": STRATEGY_NAME,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "manifest_backup": _output_relative(backup_path, root_dir),
        "source_groups": {
            "profiles": manifest["argo"]["path"],
            "sst": "rasters.ostia.analysed_sst",
            "sss": "rasters.sss.sos",
            "target_masks": "rasters.glorys.thetao and rasters.glorys.so",
        },
        "parameters": {
            "idw_k": int(idw_k),
            "idw_power": float(idw_power),
            "idw_block_rows": int(idw_block_rows),
            "strategy": STRATEGY_NAME,
            "salinity_surface_smoothing_sigma": float(salinity_surface_smoothing_sigma),
            "depth_smoothing_schedule": {
                "0-200m": 0.0,
                "200-700m": 0.5,
                "700-1500m": 1.0,
                ">1500m": 1.5,
            },
        },
        "generated_dates": [int(entry["date"]) for entry in thetao_entries],
        "skipped_dates": skipped_dates,
        "argo_support": argo_profile_counts,
        "storage_contract": {
            "temperature": _stretch_manifest(STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH]),
            "salinity": _stretch_manifest(STRETCH_SPECS[SALINITY_STRETCH]),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_safe(manifest), f, sort_keys=False)
    return root_dir


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create SST/SSS-guided synthetic GeoTIFF pretraining targets."
    )
    parser.add_argument("--geotiff-root-dir", type=Path, required=True)
    parser.add_argument("--target-date", type=int, action="append", default=None)
    parser.add_argument("--idw-k", type=int, default=DEFAULT_IDW_K)
    parser.add_argument("--idw-power", type=float, default=DEFAULT_IDW_POWER)
    parser.add_argument("--idw-block-rows", type=int, default=DEFAULT_IDW_BLOCK_ROWS)
    parser.add_argument(
        "--salinity-surface-smoothing-sigma",
        type=float,
        default=DEFAULT_SALINITY_SURFACE_SMOOTHING_SIGMA,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite-synthetic", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    """Run the synthetic pretraining GeoTIFF exporter from the CLI."""
    args = _build_parser().parse_args()
    export_synthetic_pretraining_geotiff_dataset(
        geotiff_root_dir=args.geotiff_root_dir,
        target_dates=args.target_date,
        idw_k=args.idw_k,
        idw_power=args.idw_power,
        idw_block_rows=args.idw_block_rows,
        salinity_surface_smoothing_sigma=args.salinity_surface_smoothing_sigma,
        workers=args.workers,
        overwrite_synthetic=args.overwrite_synthetic,
        skip_existing=args.skip_existing,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
