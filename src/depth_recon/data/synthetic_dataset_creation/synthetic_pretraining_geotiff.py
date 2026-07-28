"""
Example:
 /work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.synthetic_pretraining_geotiff \
   --geotiff-root-dir /work/data/OceanVariableReconstruction \
   --workers 1 \
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
import torch
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
DEFAULT_IDW_QUERY_CHUNK_SIZE = 16384
DELTA_OUTLIER_TRIM_FRACTION = 0.25
MIN_DELTA_TRIM_SUPPORT = 4
STRATEGY_NAME = "robust_vertical_delta_sss_v6"


@dataclass(frozen=True)
class SyntheticDateResult:
    """Manifest-ready output summary for one generated date."""

    date: int
    thetao: dict[str, Any] | None
    so: dict[str, Any] | None
    skipped: dict[str, Any] | None


@dataclass(frozen=True)
class SyntheticFieldResult:
    """Physical synthetic fields and masks for one date before GeoTIFF writing."""

    date: int
    temperature: np.ndarray | None
    salinity: np.ndarray | None
    thetao_valid_mask: np.ndarray | None
    so_valid_mask: np.ndarray | None
    depth_axis_m: np.ndarray
    grid: TargetGrid
    profile_count: int
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


def _profile_vertical_deltas(
    values: np.ndarray,
    depth_axis_m: np.ndarray,
) -> np.ndarray:
    """Build profile deltas from the shallowest value and fill missing depths."""
    arr = np.asarray(values, dtype=np.float32)
    depth_axis = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
    surface, has_valid = _profile_surface_values(arr)
    deltas = arr - surface[:, None]
    filled = np.full_like(deltas, np.nan, dtype=np.float32)

    for profile_idx in np.flatnonzero(has_valid).tolist():
        valid = np.isfinite(deltas[profile_idx])
        if int(valid.sum()) == 1:
            filled[profile_idx, :] = deltas[profile_idx, valid][0]
            continue
        if int(valid.sum()) > 1:
            # Use the observed vertical delta shape for the whole 50-level axis.
            # np.interp holds the nearest edge value beyond observed depths, which
            # avoids losing deep levels when ARGO ends above the target depth.
            filled[profile_idx, :] = np.interp(
                depth_axis,
                depth_axis[valid],
                deltas[profile_idx, valid],
            ).astype(np.float32, copy=False)
    return filled


def _trim_delta_outliers(
    values: np.ndarray,
    *,
    trim_fraction: float,
    min_support: int,
) -> np.ndarray:
    """Drop the most extreme profile deltas before IDW interpolation."""
    arr = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    support = int(finite.sum())
    fraction = max(0.0, min(float(trim_fraction), 0.95))
    if fraction <= 0.0 or support < max(1, int(min_support)):
        return arr

    tail_percent = 50.0 * fraction
    low, high = np.nanpercentile(arr[finite], [tail_percent, 100.0 - tail_percent])
    keep = finite & (arr >= np.float32(low)) & (arr <= np.float32(high))
    if not np.any(keep):
        return arr
    arr[finite & ~keep] = np.nan
    return arr


def _idw_candidate_count(k: int, point_count: int) -> int:
    """Return the shared neighbor count used for stacked IDW."""
    return max(1, min(max(int(k) * 8, 64), int(point_count)))


def _idw_interpolate_stack_cpu(
    *,
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    point_values: np.ndarray,
    shape: tuple[int, int],
    k: int,
    power: float,
    block_rows: int,
) -> np.ndarray:
    """Interpolate multiple depth columns on CPU with one neighbor search."""
    rows = np.asarray(point_rows, dtype=np.float64).reshape(-1)
    cols = np.asarray(point_cols, dtype=np.float64).reshape(-1)
    values = np.asarray(point_values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected point_values shaped (N,D), got {values.shape}.")
    valid = np.isfinite(rows) & np.isfinite(cols) & np.any(np.isfinite(values), axis=1)
    depth_count = int(values.shape[1])
    if not np.any(valid):
        return np.full((depth_count, *shape), np.nan, dtype=np.float32)

    rows = rows[valid]
    cols = cols[valid]
    values = values[valid]
    tree = cKDTree(np.column_stack([rows, cols]))
    k_eff = _idw_candidate_count(k, int(values.shape[0]))
    height, width = int(shape[0]), int(shape[1])
    total = height * width
    out = np.empty((depth_count, total), dtype=np.float32)
    max_rows_by_chunk = max(1, DEFAULT_IDW_QUERY_CHUNK_SIZE // max(1, width))
    block_size = max(1, min(int(block_rows), max_rows_by_chunk))
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
        finite = np.isfinite(neighbor_values)
        weights = np.power(np.maximum(distances, eps), -float(power))[:, :, None]
        safe_values = np.where(finite, neighbor_values, 0.0)
        safe_weights = np.where(finite, weights, 0.0)
        numerator = np.sum(safe_weights * safe_values, axis=1)
        denominator = np.sum(safe_weights, axis=1)
        weighted = np.full(numerator.shape, np.nan, dtype=np.float32)
        keep = denominator > eps
        weighted[keep] = (numerator[keep] / denominator[keep]).astype(
            np.float32, copy=False
        )
        start = y0 * width
        stop = y1 * width
        out[:, start:stop] = weighted.T
    return out.reshape(depth_count, height, width)


def _idw_interpolate_stack_cuda(
    *,
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    point_values: np.ndarray,
    shape: tuple[int, int],
    k: int,
    power: float,
    query_chunk_size: int,
) -> np.ndarray:
    """Interpolate multiple depth columns on CUDA with one neighbor search."""
    rows = np.asarray(point_rows, dtype=np.float32).reshape(-1)
    cols = np.asarray(point_cols, dtype=np.float32).reshape(-1)
    values = np.asarray(point_values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected point_values shaped (N,D), got {values.shape}.")
    valid = np.isfinite(rows) & np.isfinite(cols) & np.any(np.isfinite(values), axis=1)
    depth_count = int(values.shape[1])
    if not np.any(valid):
        return np.full((depth_count, *shape), np.nan, dtype=np.float32)

    device = torch.device("cuda")
    points = torch.as_tensor(
        np.column_stack([rows[valid], cols[valid]]),
        dtype=torch.float32,
        device=device,
    )
    point_values_t = torch.as_tensor(values[valid], dtype=torch.float32, device=device)
    k_eff = _idw_candidate_count(k, int(point_values_t.shape[0]))
    height, width = int(shape[0]), int(shape[1])
    total = height * width
    out = np.empty((depth_count, total), dtype=np.float32)
    chunk_size = max(1024, int(query_chunk_size))
    eps = torch.finfo(torch.float32).eps

    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        linear = torch.arange(start, stop, dtype=torch.float32, device=device)
        query = torch.stack(
            [torch.floor(linear / float(width)), torch.remainder(linear, float(width))],
            dim=1,
        )
        diff = query[:, None, :] - points[None, :, :]
        dist2 = torch.sum(diff * diff, dim=2)
        nearest_dist2, nearest_idx = torch.topk(
            dist2, k=k_eff, dim=1, largest=False, sorted=False
        )
        neighbor_values = point_values_t[nearest_idx]
        finite = torch.isfinite(neighbor_values)
        weights = torch.pow(torch.clamp(nearest_dist2, min=eps), -0.5 * float(power))
        safe_values = torch.where(
            finite, neighbor_values, torch.zeros_like(neighbor_values)
        )
        safe_weights = weights[:, :, None] * finite.to(dtype=torch.float32)
        numerator = torch.sum(safe_weights * safe_values, dim=1)
        denominator = torch.sum(safe_weights, dim=1)
        weighted = numerator / denominator.clamp_min(eps)
        weighted = torch.where(
            denominator > eps, weighted, torch.full_like(weighted, torch.nan)
        )
        out[:, start:stop] = weighted.detach().cpu().numpy().T
    return out.reshape(depth_count, height, width)


def _idw_interpolate_stack(
    *,
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    point_values: np.ndarray,
    shape: tuple[int, int],
    k: int,
    power: float,
    block_rows: int,
) -> np.ndarray:
    """Interpolate a full depth stack with shared IDW neighbor geometry."""
    if torch.cuda.is_available():
        try:
            return _idw_interpolate_stack_cuda(
                point_rows=point_rows,
                point_cols=point_cols,
                point_values=point_values,
                shape=shape,
                k=k,
                power=power,
                query_chunk_size=DEFAULT_IDW_QUERY_CHUNK_SIZE,
            )
        except RuntimeError:
            torch.cuda.empty_cache()
    return _idw_interpolate_stack_cpu(
        point_rows=point_rows,
        point_cols=point_cols,
        point_values=point_values,
        shape=shape,
        k=k,
        power=power,
        block_rows=block_rows,
    )


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


def _build_synthetic_fields(task: dict[str, Any]) -> SyntheticFieldResult:
    """Generate physical synthetic fields for one date without writing rasters."""
    root_dir = Path(task["geotiff_root_dir"])
    manifest = task["manifest"]
    date_value = int(task["date"])
    grid = _load_target_grid(Path(task["land_mask_path"]))
    depth_axis = np.asarray(task["depth_axis_m"], dtype=np.float32).reshape(-1)

    argo_path = _resolve_manifest_path(root_dir, manifest["argo"]["path"])
    profile_store = ArgoGeoTIFFProfileStore(argo_path, include_salinity=True)
    try:
        indices = _profile_indices_for_date(profile_store, date_value)
        if indices.size == 0:
            return SyntheticFieldResult(
                date=date_value,
                temperature=None,
                salinity=None,
                thetao_valid_mask=None,
                so_valid_mask=None,
                depth_axis_m=depth_axis,
                grid=grid,
                profile_count=0,
                skipped={"reason": "no_argo_profiles", "argo_profile_count": 0},
            )

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
        sst = _fill_missing_nearest(sst, thetao_valid_mask.any(axis=0))
        sss = _fill_missing_nearest(sss, so_valid_mask.any(axis=0))

        temp_delta_profiles = _profile_vertical_deltas(temp_kelvin, depth_axis)
        sal_delta_profiles = _profile_vertical_deltas(salinity, depth_axis)
        sal_surface_prior = sss
        trim_fraction = float(
            task.get(
                "delta_outlier_trim_fraction",
                DELTA_OUTLIER_TRIM_FRACTION,
            )
        )
        min_trim_support = int(
            task.get("min_delta_trim_support", MIN_DELTA_TRIM_SUPPORT)
        )

        for depth_idx in range(int(depth_axis.size)):
            temp_delta_profiles[:, depth_idx] = _trim_delta_outliers(
                temp_delta_profiles[:, depth_idx],
                trim_fraction=trim_fraction,
                min_support=min_trim_support,
            )
            sal_delta_profiles[:, depth_idx] = _trim_delta_outliers(
                sal_delta_profiles[:, depth_idx],
                trim_fraction=trim_fraction,
                min_support=min_trim_support,
            )

        combined_delta_profiles = np.concatenate(
            [temp_delta_profiles, sal_delta_profiles], axis=1
        )
        del temp_delta_profiles, sal_delta_profiles
        combined_delta_fields = _idw_interpolate_stack(
            point_rows=rows,
            point_cols=cols,
            point_values=combined_delta_profiles,
            shape=(int(grid.height), int(grid.width)),
            k=int(task["idw_k"]),
            power=float(task["idw_power"]),
            block_rows=int(task["idw_block_rows"]),
        )
        del combined_delta_profiles
        depth_count = int(depth_axis.size)
        temperature = combined_delta_fields[:depth_count].copy()
        salinity_out = combined_delta_fields[depth_count:].copy()
        del combined_delta_fields

        temperature += sst[None, :, :]
        salinity_out += sal_surface_prior[None, :, :]
        temperature[~thetao_valid_mask | ~np.isfinite(temperature)] = np.nan
        salinity_out[~so_valid_mask | ~np.isfinite(salinity_out)] = np.nan

        return SyntheticFieldResult(
            date=date_value,
            temperature=temperature,
            salinity=salinity_out,
            thetao_valid_mask=thetao_valid_mask,
            so_valid_mask=so_valid_mask,
            depth_axis_m=depth_axis,
            grid=grid,
            profile_count=int(indices.size),
            skipped=None,
        )
    finally:
        profile_store.close()


def _export_synthetic_date(task: dict[str, Any]) -> dict[str, Any]:
    """Generate and write synthetic temperature/salinity rasters for one date."""
    root_dir = Path(task["geotiff_root_dir"])
    field_result = _build_synthetic_fields(task)
    if field_result.skipped is not None:
        return SyntheticDateResult(
            date=field_result.date,
            thetao=None,
            so=None,
            skipped=field_result.skipped,
        ).__dict__

    thetao_path = (
        root_dir
        / "rasters"
        / "synthetic"
        / "thetao"
        / f"thetao_{field_result.date}.tif"
    )
    so_path = root_dir / "rasters" / "synthetic" / "so" / f"so_{field_result.date}.tif"
    thetao_result = _write_synthetic_multiband_raster(
        path=thetao_path,
        output_dir=root_dir,
        values=field_result.temperature,
        date_value=field_result.date,
        grid=field_result.grid,
        variable="thetao",
        stretch_name=TEMPERATURE_KELVIN_STRETCH,
        depth_axis_m=field_result.depth_axis_m,
        overwrite_synthetic=bool(task["overwrite_synthetic"]),
        skip_existing=bool(task["skip_existing"]),
    )
    so_result = _write_synthetic_multiband_raster(
        path=so_path,
        output_dir=root_dir,
        values=field_result.salinity,
        date_value=field_result.date,
        grid=field_result.grid,
        variable="so",
        stretch_name=SALINITY_STRETCH,
        depth_axis_m=field_result.depth_axis_m,
        overwrite_synthetic=bool(task["overwrite_synthetic"]),
        skip_existing=bool(task["skip_existing"]),
    )
    result = SyntheticDateResult(
        date=field_result.date,
        thetao=thetao_result,
        so=so_result,
        skipped=None,
    ).__dict__
    result["profile_count"] = int(field_result.profile_count)
    return result


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
        if (
            date_value not in ostia_paths
            or date_value not in sss_paths
            or date_value not in glorys_thetao_paths
            or date_value not in glorys_so_paths
        )
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
            "delta_outlier_trim_fraction": float(DELTA_OUTLIER_TRIM_FRACTION),
            "min_delta_trim_support": int(MIN_DELTA_TRIM_SUPPORT),
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
            "delta_outlier_trim_fraction": float(DELTA_OUTLIER_TRIM_FRACTION),
            "min_delta_trim_support": int(MIN_DELTA_TRIM_SUPPORT),
            "smoothing_applies_to": None,
            "vertical_delta_gap_fill": "linear_depth_interpolation_with_edge_hold",
            "idw_backend": "single_pass_stack_cuda_if_available_else_cpu",
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
        workers=args.workers,
        overwrite_synthetic=args.overwrite_synthetic,
        skip_existing=args.skip_existing,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
