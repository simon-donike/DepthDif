# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_error_analysis_dashboard --run-dir inference/outputs/global_top_band_20150615 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe
"""Export absolute-error analysis data for one inference run."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import (
    _resolve_depth_export_artifacts,
    _resolve_land_mask_path,
    _resolve_layer_url,
    _resolve_run_artifacts,
    _run_variable_metadata,
)

DEFAULT_DASHBOARD_DIR_NAME = "error_analysis"
DEFAULT_ANALYSIS_JSON_NAME = "error-analysis.json"
DEFAULT_ANALYSIS_GRID_GEOJSON_NAME = "analysis-grid.geojson"
DEFAULT_GRID_SIZE_DEGREES = 5.0
DEFAULT_GEOJSON_COORD_PRECISION = 4
METRIC_KEYS = ("median", "mean", "p90", "p95")
BASIN_NAMES = ("Pacific", "Atlantic", "Indian", "Southern", "Arctic", "Other")


def normalize_longitude(lon: float) -> float:
    """Normalize longitude to the `[-180, 180)` display interval."""
    normalized = ((float(lon) + 180.0) % 360.0) - 180.0
    return 180.0 if np.isclose(normalized, -180.0) and lon > 0.0 else normalized


def _normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    """Normalize longitudes with vectorized NumPy operations."""
    lon_values = np.asarray(lons, dtype=np.float64)
    normalized = ((lon_values + 180.0) % 360.0) - 180.0
    dateline_mask = np.isclose(normalized, -180.0) & (lon_values > 0.0)
    return np.where(dateline_mask, 180.0, normalized)


def _basin_label_array(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Assign basin labels with explicit marginal-sea and land fallback regions."""
    lon = _normalize_longitudes(lons)
    lat = np.asarray(lats, dtype=np.float64)
    labels = np.full(lon.shape, "Other", dtype="<U8")
    finite_coords = np.isfinite(lon) & np.isfinite(lat)

    arctic = finite_coords & (lat >= 66.0)
    southern = finite_coords & (lat <= -60.0)
    labels[arctic] = "Arctic"
    labels[southern] = "Southern"

    open_ocean = finite_coords & ~arctic & ~southern & (lat > -60.0) & (lat < 66.0)
    atlantic = open_ocean & (
        ((lon >= -70.0) & (lon < 20.0))
        | ((lon >= -10.0) & (lon < 42.0) & (lat >= 30.0) & (lat < 48.0))
        | ((lon >= -25.0) & (lon < 32.0) & (lat >= 48.0))
    )
    labels[atlantic] = "Atlantic"

    indian = (
        open_ocean
        & (labels == "Other")
        & (
            ((lon >= 20.0) & (lon < 120.0) & (lat < 32.0))
            | ((lon >= 120.0) & (lon < 147.0) & (lat < 0.0))
        )
    )
    labels[indian] = "Indian"

    pacific = (
        open_ocean
        & (labels == "Other")
        & (
            (lon < -70.0)
            | (lon >= 120.0)
            | ((lon >= 100.0) & (lon < 120.0) & (lat > -15.0) & (lat < 32.0))
        )
    )
    labels[pacific] = "Pacific"
    return labels


def assign_ocean_basin(lon: float, lat: float) -> str:
    """Assign an approximate ocean basin for diagnostic regional summaries."""
    labels = _basin_label_array(
        np.asarray([lon], dtype=np.float64),
        np.asarray([lat], dtype=np.float64),
    )
    return str(labels[0])


def _filter_ocean_points(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    land_mask_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop points that fall on the Natural Earth/GLORYS land mask."""
    if land_mask_path is None:
        return values, lons, lats

    with rasterio.open(land_mask_path) as mask_ds:
        land_mask = mask_ds.read(1, masked=False)
        rows, cols = rasterio.transform.rowcol(
            mask_ds.transform,
            _normalize_longitudes(lons),
            lats,
        )

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    inside = (
        (rows >= 0)
        & (rows < land_mask.shape[0])
        & (cols >= 0)
        & (cols < land_mask.shape[1])
    )
    ocean = np.zeros_like(inside, dtype=bool)
    if np.any(inside):
        # The packaged world mask stores land as 1 and ocean as 0.
        land = np.asarray(land_mask[rows[inside], cols[inside]], dtype=np.float32) > 0.5
        ocean[inside] = ~land
    return values[ocean], lons[ocean], lats[ocean]


def _valid_raster_arrays(
    path: Path,
    *,
    land_mask_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite ocean raster values and their lon/lat pixel-center coordinates."""
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=False).astype(np.float64, copy=False)
        valid = np.isfinite(data)
        if ds.nodata is not None and np.isfinite(float(ds.nodata)):
            valid &= ~np.isclose(data, float(ds.nodata), atol=0.0, rtol=0.0)
        row_idx, col_idx = np.nonzero(valid)
        if row_idx.size == 0:
            return (
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64),
            )
        xs, ys = rasterio.transform.xy(
            ds.transform,
            row_idx,
            col_idx,
            offset="center",
        )

    values = data[row_idx, col_idx].astype(np.float64, copy=False)
    lons = np.asarray(xs, dtype=np.float64)
    lats = np.asarray(ys, dtype=np.float64)
    return _filter_ocean_points(values, lons, lats, land_mask_path=land_mask_path)


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    """Compute compact error statistics for finite values."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "p90": float(np.nanpercentile(finite, 90.0)),
        "p95": float(np.nanpercentile(finite, 95.0)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def _empty_group_stats(name: str) -> dict[str, Any]:
    stats = summarize_values(np.asarray([], dtype=np.float64))
    stats["name"] = name
    return stats


def aggregate_by_basin(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    basin_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Aggregate absolute-error values into approximate ocean basins."""
    labels = (
        _basin_label_array(lons, lats)
        if basin_labels is None
        else np.asarray(basin_labels, dtype="<U8")
    )
    if labels.shape != values.shape:
        raise ValueError("basin_labels must match values shape.")

    summaries: list[dict[str, Any]] = []
    for basin in BASIN_NAMES:
        row = summarize_values(values[labels == basin])
        row["name"] = basin
        summaries.append(row)
    return summaries


def _dominant_basin(labels: np.ndarray) -> str:
    """Return the most common basin label in a grouped grid cell."""
    if labels.size == 0:
        return "Other"
    unique, counts = np.unique(labels, return_counts=True)
    counts_by_name = {str(name): int(count) for name, count in zip(unique, counts)}
    best_name = "Other"
    best_count = -1
    for name in BASIN_NAMES:
        count = counts_by_name.get(name, 0)
        if count > best_count:
            best_name = name
            best_count = count
    return best_name


def _grid_cell_id(lat_bin: int, lon_bin: int) -> str:
    """Return the stable dashboard id for one analysis grid cell."""
    return f"cell_{int(lat_bin)}_{int(lon_bin)}"


def _grid_cell_bounds(
    lat_bin: int,
    lon_bin: int,
    *,
    grid_size_degrees: float,
) -> tuple[float, float, float, float]:
    """Return west/south/east/north bounds for one dashboard grid cell."""
    grid_size = float(grid_size_degrees)
    west = -180.0 + float(lon_bin) * grid_size
    south = -90.0 + float(lat_bin) * grid_size
    east = min(180.0, west + grid_size)
    north = min(90.0, south + grid_size)
    return float(west), float(south), float(east), float(north)


def _round_geojson_coordinates(value: Any, *, decimals: int) -> Any:
    """Round nested GeoJSON coordinates for compact dashboard geometry."""
    if isinstance(value, (list, tuple)):
        return [_round_geojson_coordinates(item, decimals=decimals) for item in value]
    if isinstance(value, float):
        return round(value, decimals)
    return value


def _cell_window_from_bounds(
    dataset: rasterio.DatasetReader,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> rasterio.windows.Window | None:
    """Return a raster window for the cell bounds clipped to the mask extent."""
    bounds = dataset.bounds
    clipped_west = max(float(west), float(bounds.left))
    clipped_south = max(float(south), float(bounds.bottom))
    clipped_east = min(float(east), float(bounds.right))
    clipped_north = min(float(north), float(bounds.top))
    if clipped_east <= clipped_west or clipped_north <= clipped_south:
        return None

    raw_window = rasterio.windows.from_bounds(
        clipped_west,
        clipped_south,
        clipped_east,
        clipped_north,
        transform=dataset.transform,
    )
    row_off = max(0, int(np.floor(float(raw_window.row_off) + 1.0e-9)))
    col_off = max(0, int(np.floor(float(raw_window.col_off) + 1.0e-9)))
    row_stop = min(
        int(dataset.height),
        int(np.ceil(float(raw_window.row_off + raw_window.height) - 1.0e-9)),
    )
    col_stop = min(
        int(dataset.width),
        int(np.ceil(float(raw_window.col_off + raw_window.width) - 1.0e-9)),
    )
    if row_stop <= row_off or col_stop <= col_off:
        return None
    return rasterio.windows.Window(
        col_off=col_off,
        row_off=row_off,
        width=col_stop - col_off,
        height=row_stop - row_off,
    )


def _ocean_mask_from_land_window(
    dataset: rasterio.DatasetReader,
    window: rasterio.windows.Window,
) -> np.ndarray:
    """Read one land-mask window and return pixels that should draw as ocean."""
    data = dataset.read(1, window=window, masked=True)
    values = np.asarray(data.filled(1), dtype=np.float32)
    valid = ~np.ma.getmaskarray(data)
    if dataset.nodata is not None and np.isfinite(float(dataset.nodata)):
        valid &= ~np.isclose(values, float(dataset.nodata), atol=0.0, rtol=0.0)
    # The packaged world mask stores land as 1 and ocean as 0.
    return (valid & (values <= 0.5)).astype(np.uint8, copy=False)


def build_analysis_grid_geojson_payload(
    *,
    land_mask_path: Path,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    coordinate_precision: int = DEFAULT_GEOJSON_COORD_PRECISION,
) -> dict[str, Any]:
    """Build coast-clipped dashboard grid-cell geometries from a land mask."""
    grid_size = float(grid_size_degrees)
    if grid_size <= 0.0:
        raise ValueError("grid_size_degrees must be positive.")

    lon_bin_count = int(np.ceil(360.0 / grid_size))
    lat_bin_count = int(np.ceil(180.0 / grid_size))
    features: list[dict[str, Any]] = []
    with rasterio.open(land_mask_path) as dataset:
        for lat_bin in range(lat_bin_count):
            for lon_bin in range(lon_bin_count):
                west, south, east, north = _grid_cell_bounds(
                    lat_bin,
                    lon_bin,
                    grid_size_degrees=grid_size,
                )
                window = _cell_window_from_bounds(
                    dataset,
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                )
                if window is None:
                    continue

                ocean_mask = _ocean_mask_from_land_window(dataset, window)
                ocean_pixel_count = int(np.count_nonzero(ocean_mask))
                if ocean_pixel_count <= 0:
                    continue

                transform = rasterio.windows.transform(window, dataset.transform)
                polygons: list[Any] = []
                for geometry, value in rasterio.features.shapes(
                    ocean_mask,
                    mask=ocean_mask.astype(bool),
                    transform=transform,
                ):
                    if int(value) != 1 or geometry.get("type") != "Polygon":
                        continue
                    polygons.append(
                        _round_geojson_coordinates(
                            geometry.get("coordinates", []),
                            decimals=int(coordinate_precision),
                        )
                    )
                if not polygons:
                    continue

                total_pixel_count = int(ocean_mask.size)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": polygons,
                        },
                        "properties": {
                            "id": _grid_cell_id(lat_bin, lon_bin),
                            "west": float(west),
                            "south": float(south),
                            "east": float(east),
                            "north": float(north),
                            "ocean_pixel_count": ocean_pixel_count,
                            "total_pixel_count": total_pixel_count,
                            "ocean_fraction": float(
                                ocean_pixel_count / float(total_pixel_count)
                            ),
                        },
                    }
                )

    return {
        "type": "FeatureCollection",
        "name": "DepthDif analysis ocean grid",
        "grid_size_degrees": grid_size,
        "features": features,
    }


def write_analysis_grid_geojson(
    *,
    output_path: Path,
    land_mask_path: Path,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    coordinate_precision: int = DEFAULT_GEOJSON_COORD_PRECISION,
) -> Path:
    """Write coast-clipped dashboard grid-cell geometries as GeoJSON."""
    payload = build_analysis_grid_geojson_payload(
        land_mask_path=Path(land_mask_path),
        grid_size_degrees=grid_size_degrees,
        coordinate_precision=coordinate_precision,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    return output_path


def aggregate_by_grid(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    basin_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Aggregate absolute-error values into fixed lat-lon grid cells."""
    grid_size = float(grid_size_degrees)
    if grid_size <= 0.0:
        raise ValueError("grid_size_degrees must be positive.")
    normalized_lons = _normalize_longitudes(lons)
    lon_bins = np.floor((normalized_lons + 180.0) / grid_size).astype(np.int64)
    lat_bins = np.floor((np.asarray(lats) + 90.0) / grid_size).astype(np.int64)
    lon_bins = np.clip(lon_bins, 0, int(np.ceil(360.0 / grid_size)) - 1)
    lat_bins = np.clip(lat_bins, 0, int(np.ceil(180.0 / grid_size)) - 1)
    labels = (
        _basin_label_array(lons, lats)
        if basin_labels is None
        else np.asarray(basin_labels, dtype="<U8")
    )
    if labels.shape != values.shape:
        raise ValueError("basin_labels must match values shape.")

    if values.size == 0:
        return []
    order = np.lexsort((lon_bins, lat_bins))
    sorted_lat_bins = lat_bins[order]
    sorted_lon_bins = lon_bins[order]
    sorted_values = values[order]
    sorted_labels = labels[order]
    key_changes = np.flatnonzero(
        (np.diff(sorted_lat_bins) != 0) | (np.diff(sorted_lon_bins) != 0)
    )
    starts = np.concatenate(([0], key_changes + 1))
    stops = np.concatenate((key_changes + 1, [sorted_values.size]))

    cells: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        lat_bin = int(sorted_lat_bins[start])
        lon_bin = int(sorted_lon_bins[start])
        west, south, east, north = _grid_cell_bounds(
            lat_bin,
            lon_bin,
            grid_size_degrees=grid_size,
        )
        cell_values = sorted_values[start:stop]
        stats = summarize_values(cell_values)
        if int(stats["count"]) <= 0:
            continue
        stats.update(
            {
                "id": _grid_cell_id(lat_bin, lon_bin),
                "label": f"{south:.0f} to {north:.0f} lat, {west:.0f} to {east:.0f} lon",
                "basin": _dominant_basin(sorted_labels[start:stop]),
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north),
                "center_lon": float((west + east) / 2.0),
                "center_lat": float((south + north) / 2.0),
            }
        )
        cells.append(stats)
    return cells


def _top_cells(
    cells: list[dict[str, Any]], *, metric: str, limit: int
) -> list[dict[str, Any]]:
    """Return the highest-error grid cells by one metric."""
    ranked = [
        cell
        for cell in cells
        if cell.get(metric) is not None and int(cell.get("count", 0)) > 0
    ]
    ranked.sort(key=lambda cell: float(cell[metric]), reverse=True)
    return ranked[: int(limit)]


def _aggregate_summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average per-depth metric summaries and sum their valid pixel counts."""
    total_count = sum(int(row.get("count", 0) or 0) for row in rows)
    summary: dict[str, Any] = {"count": int(total_count)}
    for metric in METRIC_KEYS:
        weighted_sum = 0.0
        weight_sum = 0
        for row in rows:
            value = row.get(metric)
            count = int(row.get("count", 0) or 0)
            if value is None or count <= 0:
                continue
            weighted_sum += float(value) * float(count)
            weight_sum += count
        summary[metric] = (
            None if weight_sum <= 0 else float(weighted_sum / float(weight_sum))
        )

    min_values = [float(row["min"]) for row in rows if row.get("min") is not None]
    max_values = [float(row["max"]) for row in rows if row.get("max") is not None]
    summary["min"] = min(min_values) if min_values else None
    summary["max"] = max(max_values) if max_values else None
    return summary


def _aggregate_basin_rows(depth_levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build all-depth basin summaries from per-depth basin rows."""
    basins: list[dict[str, Any]] = []
    for basin in BASIN_NAMES:
        rows = [
            row
            for depth in depth_levels
            for row in depth.get("basins", [])
            if row.get("name") == basin
        ]
        basin_row = _aggregate_summary_rows(rows)
        basin_row["name"] = basin
        basins.append(basin_row)
    return basins


def _aggregate_grid_cell_rows(
    depth_levels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build all-depth grid-cell summaries from matching per-depth cell ids."""
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for depth in depth_levels:
        for cell in depth.get("grid_cells", []):
            rows_by_cell.setdefault(str(cell["id"]), []).append(cell)

    cells: list[dict[str, Any]] = []
    for rows in rows_by_cell.values():
        template = rows[0]
        cell = _aggregate_summary_rows(rows)
        cell.update(
            {
                "id": template["id"],
                "label": template["label"],
                "basin": template.get("basin", "Other"),
                "west": template["west"],
                "south": template["south"],
                "east": template["east"],
                "north": template["north"],
                "center_lon": template["center_lon"],
                "center_lat": template["center_lat"],
            }
        )
        cells.append(cell)

    cells.sort(key=lambda cell: (float(cell["south"]), float(cell["west"])))
    return cells


def _build_all_depths_analysis(
    depth_levels: list[dict[str, Any]],
    *,
    top_cell_count: int,
) -> dict[str, Any]:
    """Build the first dashboard level aggregating all exported depths."""
    grid_cells = _aggregate_grid_cell_rows(depth_levels)
    return {
        "index": -1,
        "suffix": "all_depths",
        "label": "All Depths",
        "requested_depth_m": None,
        "actual_depth_m": None,
        "channel_index": None,
        "is_aggregate": True,
        "depth_count": int(len(depth_levels)),
        "aggregation_method": "Count-weighted average of per-depth regional metrics; counts are summed across depths.",
        "global": _aggregate_summary_rows([depth["global"] for depth in depth_levels]),
        "basins": _aggregate_basin_rows(depth_levels),
        "grid_cells": grid_cells,
        "top_cells": {
            metric: _top_cells(
                grid_cells,
                metric=metric,
                limit=int(top_cell_count),
            )
            for metric in METRIC_KEYS
        },
    }


def _valid_array_points(
    data: np.ndarray,
    *,
    transform: Any,
    land_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite ocean array values and their lon/lat pixel-center coordinates."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D analysis array, got shape {tuple(array.shape)}."
        )

    valid = np.isfinite(array)
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != array.shape:
            raise ValueError(
                "land_mask shape must match analysis array shape: "
                f"{tuple(land.shape)} != {tuple(array.shape)}."
            )
        valid &= ~land

    row_idx, col_idx = np.nonzero(valid)
    if row_idx.size == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )

    xs, ys = rasterio.transform.xy(transform, row_idx, col_idx, offset="center")
    return (
        array[row_idx, col_idx].astype(np.float64, copy=False),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    )


def _build_depth_level_analysis_from_values(
    depth_index: int,
    depth_metadata: dict[str, Any],
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    grid_size_degrees: float,
    top_cell_count: int,
) -> dict[str, Any]:
    """Build exact error-analysis summaries for one depth from point values."""
    basin_labels = _basin_label_array(lons, lats)
    grid_cells = aggregate_by_grid(
        values,
        lons,
        lats,
        grid_size_degrees=grid_size_degrees,
        basin_labels=basin_labels,
    )
    global_stats = summarize_values(values)
    basin_stats = aggregate_by_basin(values, lons, lats, basin_labels=basin_labels)
    top_cells = {
        metric: _top_cells(
            grid_cells,
            metric=metric,
            limit=int(top_cell_count),
        )
        for metric in METRIC_KEYS
    }
    return {
        "index": int(depth_index),
        "suffix": str(depth_metadata["suffix"]),
        "label": str(depth_metadata["label"]),
        "requested_depth_m": float(depth_metadata["requested_depth_m"]),
        "actual_depth_m": float(depth_metadata["actual_depth_m"]),
        "channel_index": int(depth_metadata["channel_index"]),
        "global": global_stats,
        "basins": basin_stats,
        "grid_cells": grid_cells,
        "top_cells": top_cells,
    }


def _build_depth_level_analysis(
    depth_index: int,
    depth_export: dict[str, Any],
    *,
    grid_size_degrees: float,
    top_cell_count: int,
    land_mask_path: Path | None,
) -> dict[str, Any]:
    """Build exact error-analysis summaries for one depth export."""
    values, lons, lats = _valid_raster_arrays(
        Path(depth_export["absolute_error_path"]),
        land_mask_path=land_mask_path,
    )
    return _build_depth_level_analysis_from_values(
        depth_index,
        depth_export,
        values,
        lons,
        lats,
        grid_size_degrees=grid_size_degrees,
        top_cell_count=top_cell_count,
    )


def _build_depth_level_analysis_worker(
    args: tuple[int, dict[str, Any], float, int, Path | None],
) -> dict[str, Any]:
    """Unpack process-pool arguments for one depth export."""
    depth_index, depth_export, grid_size_degrees, top_cell_count, land_mask_path = args
    return _build_depth_level_analysis(
        depth_index,
        depth_export,
        grid_size_degrees=grid_size_degrees,
        top_cell_count=top_cell_count,
        land_mask_path=land_mask_path,
    )


def build_error_analysis_payload(
    *,
    run_dir: Path,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
    analysis_workers: int = 1,
) -> dict[str, Any]:
    """Build the JSON-serializable dashboard analysis payload."""
    run_dir = Path(run_dir).resolve()
    (
        prediction_path,
        ground_truth_path,
        _absolute_error_path,
        _points_path,
        _patch_splits_path,
        _full_sample_points_path,
        _graphs_dir_path,
        _uncertainty_path,
        run_summary,
    ) = _resolve_run_artifacts(run_dir)
    variable_metadata = _run_variable_metadata(run_summary)
    land_mask_path = _resolve_land_mask_path(run_summary, run_dir=run_dir)
    depth_exports = _resolve_depth_export_artifacts(
        run_dir=run_dir,
        run_summary=run_summary,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
    )
    depth_exports = [
        depth_export
        for depth_export in depth_exports
        if depth_export.get("absolute_error_path") is not None
    ]
    if not depth_exports:
        raise FileNotFoundError(
            "No absolute-error GeoTIFFs were found in the run summary."
        )

    worker_count = max(1, int(analysis_workers))
    worker_count = min(worker_count, len(depth_exports))
    depth_levels: list[dict[str, Any] | None] = [None] * len(depth_exports)
    progress = tqdm(
        total=len(depth_exports),
        desc=f"{variable_metadata['label']} error analysis",
        unit="depth",
    )
    if worker_count == 1:
        for depth_index, depth_export in enumerate(depth_exports):
            suffix = str(depth_export["suffix"])
            progress.set_postfix_str(suffix)
            depth_levels[depth_index] = _build_depth_level_analysis(
                depth_index,
                depth_export,
                grid_size_degrees=grid_size_degrees,
                top_cell_count=top_cell_count,
                land_mask_path=land_mask_path,
            )
            progress.update(1)
    else:
        tasks = [
            (
                depth_index,
                depth_export,
                float(grid_size_degrees),
                int(top_cell_count),
                land_mask_path,
            )
            for depth_index, depth_export in enumerate(depth_exports)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_build_depth_level_analysis_worker, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                depth_index = futures[future]
                depth_levels[depth_index] = future.result()
                progress.update(1)
    progress.close()
    completed_depth_levels = [
        depth_level for depth_level in depth_levels if depth_level is not None
    ]
    displayed_depth_levels = [
        _build_all_depths_analysis(
            completed_depth_levels,
            top_cell_count=top_cell_count,
        ),
        *completed_depth_levels,
    ]

    return {
        "schema_version": 1,
        "title": "DepthDif Error Analysis",
        "description": (
            "Aggregated prediction-vs-GLORYS absolute error diagnostics for one "
            "global inference export."
        ),
        "run": {
            "run_dir": str(run_dir),
            "selected_date": run_summary.get("selected_date"),
            "target_date": run_summary.get(
                "target_date", run_summary.get("selected_date")
            ),
            "iso_year": run_summary.get("iso_year"),
            "iso_week": run_summary.get("iso_week"),
        },
        "variable": {
            "name": str(variable_metadata["name"]),
            "label": str(variable_metadata["label"]),
            "value_units": str(variable_metadata["value_units"]),
            "value_unit_label": str(variable_metadata["value_unit_label"]),
        },
        "metrics": list(METRIC_KEYS),
        "grouping": {
            "basins": list(BASIN_NAMES),
            "grid_size_degrees": float(grid_size_degrees),
            "basin_method": "Land-filtered deterministic lon/lat basin buckets with dominant basin labels on grid cells.",
        },
        "depth_levels": displayed_depth_levels,
    }


def build_error_analysis_payload_from_depth_arrays(
    *,
    run_summary: dict[str, Any],
    variable_metadata: dict[str, Any],
    depth_levels_metadata: Sequence[dict[str, Any]],
    absolute_error_arrays: Iterable[np.ndarray],
    transform: Any,
    land_mask: np.ndarray | None = None,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
) -> dict[str, Any]:
    """Build dashboard analysis payload from stitched per-depth error arrays."""
    depth_levels: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(depth_levels_metadata),
        desc=f"{variable_metadata['label']} full-depth error analysis",
        unit="depth",
    )
    try:
        for depth_index, (depth_metadata, error_array) in enumerate(
            zip(depth_levels_metadata, absolute_error_arrays, strict=True)
        ):
            progress.set_postfix_str(str(depth_metadata["suffix"]))
            values, lons, lats = _valid_array_points(
                error_array,
                transform=transform,
                land_mask=land_mask,
            )
            depth_levels.append(
                _build_depth_level_analysis_from_values(
                    depth_index,
                    depth_metadata,
                    values,
                    lons,
                    lats,
                    grid_size_degrees=grid_size_degrees,
                    top_cell_count=top_cell_count,
                )
            )
            progress.update(1)
    finally:
        progress.close()

    displayed_depth_levels = [
        _build_all_depths_analysis(depth_levels, top_cell_count=top_cell_count),
        *depth_levels,
    ]
    return {
        "schema_version": 1,
        "title": "DepthDif Error Analysis",
        "description": (
            "Aggregated prediction-vs-GLORYS absolute error diagnostics for one "
            "global inference export."
        ),
        "run": {
            "run_dir": str(run_summary.get("run_dir", "")),
            "selected_date": run_summary.get("selected_date"),
            "target_date": run_summary.get(
                "target_date", run_summary.get("selected_date")
            ),
            "iso_year": run_summary.get("iso_year"),
            "iso_week": run_summary.get("iso_week"),
        },
        "variable": {
            "name": str(variable_metadata["name"]),
            "label": str(variable_metadata["label"]),
            "value_units": str(variable_metadata["value_units"]),
            "value_unit_label": str(variable_metadata["value_unit_label"]),
        },
        "metrics": list(METRIC_KEYS),
        "grouping": {
            "basins": list(BASIN_NAMES),
            "grid_size_degrees": float(grid_size_degrees),
            "basin_method": "Land-filtered deterministic lon/lat basin buckets with dominant basin labels on grid cells.",
        },
        "depth_levels": displayed_depth_levels,
    }


def export_error_analysis_dashboard(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    public_base_url: str | None = None,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
    analysis_workers: int = 1,
) -> dict[str, Any]:
    """Write `error-analysis.json` for one run."""
    run_dir = Path(run_dir).resolve()
    if output_dir is None:
        output_dir = run_dir / DEFAULT_DASHBOARD_DIR_NAME
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_error_analysis_payload(
        run_dir=run_dir,
        grid_size_degrees=grid_size_degrees,
        top_cell_count=top_cell_count,
        analysis_workers=analysis_workers,
    )
    json_path = output_dir / DEFAULT_ANALYSIS_JSON_NAME
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")

    (
        _prediction_path,
        _ground_truth_path,
        _absolute_error_path,
        _points_path,
        _patch_splits_path,
        _full_sample_points_path,
        _graphs_dir_path,
        _uncertainty_path,
        run_summary,
    ) = _resolve_run_artifacts(run_dir)
    land_mask_path = _resolve_land_mask_path(run_summary, run_dir=run_dir)
    grid_geojson_path: Path | None = None
    if land_mask_path is not None:
        grid_geojson_path = write_analysis_grid_geojson(
            output_path=output_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
            land_mask_path=land_mask_path,
            grid_size_degrees=grid_size_degrees,
        )

    result = {
        "output_dir": str(output_dir),
        "json_path": str(json_path),
        "json_url": _resolve_layer_url(json_path.name, public_base_url=public_base_url),
        "depth_level_count": int(len(payload["depth_levels"])),
    }
    if grid_geojson_path is not None:
        result["grid_geojson_path"] = str(grid_geojson_path)
        result["grid_geojson_url"] = _resolve_layer_url(
            grid_geojson_path.name,
            public_base_url=public_base_url,
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export absolute-error analysis data.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--public-base-url", type=str, default=None)
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=DEFAULT_GRID_SIZE_DEGREES,
    )
    parser.add_argument("--top-cell-count", type=int, default=24)
    parser.add_argument("--analysis-workers", type=int, default=1)
    return parser


def main() -> None:
    """Run the error-analysis data exporter from the command line."""
    args = _build_parser().parse_args()
    result = export_error_analysis_dashboard(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        public_base_url=args.public_base_url,
        grid_size_degrees=args.grid_size_degrees,
        top_cell_count=args.top_cell_count,
        analysis_workers=args.analysis_workers,
    )
    print(f"Wrote error analysis data to: {result['output_dir']}")
    print(f"- data: {result['json_path']}")
    if result.get("grid_geojson_path") is not None:
        print(f"- ocean grid: {result['grid_geojson_path']}")


if __name__ == "__main__":
    main()
