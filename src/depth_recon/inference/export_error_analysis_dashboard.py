# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_error_analysis_dashboard --run-dir inference/outputs/global_top_band_20150615 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe
"""Export a standalone absolute-error analysis dashboard for one inference run."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import rasterio
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
DEFAULT_ANALYSIS_HTML_NAME = "error-analysis.html"
DEFAULT_GRID_SIZE_DEGREES = 5.0
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
        west = -180.0 + float(lon_bin) * grid_size
        south = -90.0 + float(lat_bin) * grid_size
        east = min(180.0, west + grid_size)
        north = min(90.0, south + grid_size)
        cell_values = sorted_values[start:stop]
        stats = summarize_values(cell_values)
        if int(stats["count"]) <= 0:
            continue
        stats.update(
            {
                "id": f"cell_{lat_bin}_{lon_bin}",
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


def _build_depth_level_analysis(
    depth_index: int,
    depth_export: dict[str, Any],
    *,
    grid_size_degrees: float,
    top_cell_count: int,
    land_mask_path: Path | None,
) -> dict[str, Any]:
    """Build exact error-analysis summaries for one depth export."""
    suffix = str(depth_export["suffix"])
    values, lons, lats = _valid_raster_arrays(
        Path(depth_export["absolute_error_path"]),
        land_mask_path=land_mask_path,
    )
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
        "suffix": suffix,
        "label": str(depth_export["label"]),
        "requested_depth_m": float(depth_export["requested_depth_m"]),
        "actual_depth_m": float(depth_export["actual_depth_m"]),
        "channel_index": int(depth_export["channel_index"]),
        "global": global_stats,
        "basins": basin_stats,
        "grid_cells": grid_cells,
        "top_cells": top_cells,
    }


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


def _dashboard_html() -> str:
    """Return the standalone dashboard HTML document."""
    return r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DepthDif Error Analysis</title>
    <style>
      :root { color-scheme: dark; --bg: #071114; --panel: #101a1d; --line: #26363a; --text: #edf7f5; --muted: #9fb7b3; --cyan: #69d7d0; --red: #ef5b5b; --yellow: #f4c95d; --green: #3ddc84; }
      * { box-sizing: border-box; }
      body { margin: 0; min-height: 100vh; color: var(--text); background: var(--bg); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      button, select { font: inherit; }
      .app { display: grid; grid-template-rows: auto auto 1fr auto; gap: 12px; min-height: 100vh; padding: 14px; }
      header { display: flex; gap: 16px; align-items: end; justify-content: space-between; }
      h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
      .subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; }
      .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      .control { display: grid; gap: 4px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
      select, .segmented button { border: 1px solid var(--line); border-radius: 6px; background: #111f23; color: var(--text); padding: 8px 10px; }
      .segmented { display: flex; gap: 4px; }
      .segmented button[aria-pressed="true"] { border-color: var(--cyan); color: #051112; background: var(--cyan); }
      .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
      .kpi, .panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
      .kpi { padding: 12px; }
      .kpi-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
      .kpi-value { margin-top: 6px; font-size: 23px; font-weight: 700; }
      .main { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 12px; min-height: 0; }
      .panel { min-height: 0; padding: 12px; }
      .panel-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
      #map { width: 100%; height: min(58vh, 560px); border-radius: 6px; background: #071013; cursor: crosshair; }
      .rankings { display: grid; gap: 12px; overflow: auto; max-height: min(58vh, 560px); }
      .rank-list { display: grid; gap: 6px; }
      .rank-item { width: 100%; display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; padding: 8px; border: 1px solid var(--line); border-radius: 6px; color: var(--text); background: #0b1619; text-align: left; cursor: pointer; }
      .rank-item:hover, .rank-item.is-active { border-color: var(--cyan); }
      .rank-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .rank-meta { color: var(--muted); font-size: 12px; }
      .charts { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
      svg { width: 100%; height: 230px; display: block; overflow: visible; }
      .axis { stroke: #4a5d62; stroke-width: 1; }
      .bar { fill: var(--cyan); }
      .line { fill: none; stroke: var(--yellow); stroke-width: 2.5; }
      .point { fill: var(--yellow); }
      .tooltip { position: fixed; z-index: 5; display: none; max-width: 280px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; background: #071114; color: var(--text); font-size: 12px; pointer-events: none; box-shadow: 0 12px 30px rgba(0,0,0,.35); }
      @media (max-width: 980px) { .main, .charts, .kpis { grid-template-columns: 1fr; } header { align-items: start; flex-direction: column; } #map, .rankings { max-height: none; height: 420px; } }
    </style>
  </head>
  <body>
    <div class="app">
      <header>
        <div>
          <h1>DepthDif Error Analysis</h1>
          <div id="subtitle" class="subtitle">Loading absolute-error dashboard...</div>
        </div>
        <div class="controls">
          <label class="control">Depth <select id="depthSelect"></select></label>
          <div class="control">Metric <div class="segmented" id="metricButtons"></div></div>
        </div>
      </header>
      <section class="kpis" id="kpis"></section>
      <main class="main">
        <section class="panel">
          <div class="panel-title"><span>Geographic Hotspots</span><span id="selectionLabel">Global</span></div>
          <canvas id="map"></canvas>
        </section>
        <aside class="panel rankings">
          <div>
            <div class="panel-title">Worst Basins</div>
            <div id="basinRanking" class="rank-list"></div>
          </div>
          <div>
            <div class="panel-title">Worst Grid Cells</div>
            <div id="cellRanking" class="rank-list"></div>
          </div>
        </aside>
      </main>
      <section class="charts">
        <div class="panel">
          <div class="panel-title">Depth Profile</div>
          <svg id="profileChart" role="img" aria-label="Error by depth"></svg>
        </div>
        <div class="panel">
          <div class="panel-title">Basin Comparison</div>
          <svg id="basinChart" role="img" aria-label="Basin error comparison"></svg>
        </div>
      </section>
    </div>
    <div id="tooltip" class="tooltip"></div>
    <script>
      const state = { data: null, depthIndex: 0, metric: "median", selection: { type: "global", id: "global", label: "Global" }, hitCells: [] };
      const fmt = (value) => value === null || value === undefined || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(2);
      const countFmt = (value) => Number(value || 0).toLocaleString();
      function metricLabel(metric) { return metric === "p90" ? "P90" : metric === "p95" ? "P95" : metric[0].toUpperCase() + metric.slice(1); }
      function unit() { return state.data.variable.value_unit_label || ""; }
      function activeDepth() { return state.data.depth_levels[state.depthIndex]; }
      function chartDepths() {
        const depths = state.data.depth_levels.filter((depth) => !depth.is_aggregate);
        return depths.length ? depths : state.data.depth_levels;
      }
      async function init() {
        const response = await fetch(new URL("error-analysis.json", window.location.href));
        state.data = await response.json();
        document.getElementById("subtitle").textContent = `${state.data.variable.label} absolute error | ${state.data.run.iso_year || ""} W${state.data.run.iso_week || ""}`;
        setupControls();
        window.addEventListener("resize", render);
        render();
      }
      function setupControls() {
        const select = document.getElementById("depthSelect");
        select.innerHTML = state.data.depth_levels.map((d, i) => `<option value="${i}">${d.label}</option>`).join("");
        select.addEventListener("change", () => { state.depthIndex = Number(select.value); render(); });
        const buttons = document.getElementById("metricButtons");
        buttons.innerHTML = state.data.metrics.map((metric) => `<button type="button" data-metric="${metric}" aria-pressed="${metric === state.metric}">${metricLabel(metric)}</button>`).join("");
        buttons.addEventListener("click", (event) => {
          const button = event.target.closest("button[data-metric]");
          if (!button) return;
          state.metric = button.dataset.metric;
          document.querySelectorAll("#metricButtons button").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
          render();
        });
      }
      function render() {
        renderKpis();
        renderRankings();
        renderMap();
        renderProfileChart();
        renderBasinChart();
        document.getElementById("selectionLabel").textContent = state.selection.label;
      }
      function renderKpis() {
        const g = activeDepth().global;
        const cards = [
          ["Median", `${fmt(g.median)} ${unit()}`],
          ["Mean", `${fmt(g.mean)} ${unit()}`],
          ["P95", `${fmt(g.p95)} ${unit()}`],
          ["Valid Pixels", countFmt(g.count)],
        ];
        document.getElementById("kpis").innerHTML = cards.map(([label, value]) => `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div></div>`).join("");
      }
      function itemButton(item, type, label) {
        const active = state.selection.type === type && state.selection.id === (item.id || item.name);
        return `<button class="rank-item ${active ? "is-active" : ""}" data-type="${type}" data-id="${item.id || item.name}" data-label="${label}"><span><span class="rank-name">${label}</span><span class="rank-meta">${countFmt(item.count)} px</span></span><strong>${fmt(item[state.metric])}</strong></button>`;
      }
      function renderRankings() {
        const depth = activeDepth();
        const basins = [...depth.basins].filter((b) => b[state.metric] !== null).sort((a, b) => b[state.metric] - a[state.metric]);
        document.getElementById("basinRanking").innerHTML = basins.map((b) => itemButton(b, "basin", b.name)).join("");
        document.getElementById("cellRanking").innerHTML = (depth.top_cells[state.metric] || []).map((c) => itemButton(c, "cell", c.label)).join("");
        document.querySelectorAll(".rank-item").forEach((button) => {
          button.addEventListener("click", () => {
            state.selection = { type: button.dataset.type, id: button.dataset.id, label: button.dataset.label };
            render();
          });
        });
      }
      function quantile(sortedValues, fraction) {
        if (sortedValues.length === 0) return null;
        const index = (sortedValues.length - 1) * Math.max(0, Math.min(1, fraction));
        const lower = Math.floor(index), upper = Math.ceil(index);
        if (lower === upper) return sortedValues[lower];
        const weight = index - lower;
        return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
      }
      function colorDomain(cells) {
        const values = cells.map((cell) => Number(cell[state.metric])).filter(Number.isFinite).sort((a, b) => a - b);
        if (values.length === 0) return { lower: 0, upper: 1 };
        const lower = Math.max(0, quantile(values, 0.05) ?? values[0]);
        let upper = quantile(values, 0.95) ?? values[values.length - 1];
        if (!Number.isFinite(upper) || upper <= lower) upper = values[values.length - 1] > lower ? values[values.length - 1] : lower + 1;
        return { lower, upper };
      }
      function colorFor(value, domain) {
        const normalized = Math.max(0, Math.min(1, (Number(value || 0) - domain.lower) / Math.max(1e-9, domain.upper - domain.lower)));
        const t = Math.pow(normalized, 0.72);
        const stops = [[44,123,182], [0,166,202], [127,211,78], [253,174,97], [215,25,28]];
        const scaled = t * (stops.length - 1);
        const lowerIndex = Math.min(stops.length - 2, Math.floor(scaled));
        const upperIndex = lowerIndex + 1;
        const local = scaled - lowerIndex;
        const lower = stops[lowerIndex], upper = stops[upperIndex];
        const rgb = lower.map((channel, index) => Math.round(channel + (upper[index] - channel) * local));
        return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      }
      function renderMap() {
        const canvas = document.getElementById("map");
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.fillStyle = "#071013";
        ctx.fillRect(0, 0, rect.width, rect.height);
        ctx.strokeStyle = "#213137";
        ctx.lineWidth = 1;
        for (let lon = -180; lon <= 180; lon += 30) { const x = (lon + 180) / 360 * rect.width; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke(); }
        for (let lat = -60; lat <= 60; lat += 30) { const y = (90 - lat) / 180 * rect.height; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke(); }
        const cells = activeDepth().grid_cells.filter((cell) => cell[state.metric] !== null);
        const domain = colorDomain(cells);
        state.hitCells = [];
        for (const cell of cells) {
          const x = (cell.west + 180) / 360 * rect.width;
          const y = (90 - cell.north) / 180 * rect.height;
          const w = (cell.east - cell.west) / 360 * rect.width;
          const h = (cell.north - cell.south) / 180 * rect.height;
          ctx.fillStyle = colorFor(cell[state.metric], domain);
          ctx.globalAlpha = state.selection.type === "cell" && state.selection.id !== cell.id ? 0.45 : 0.88;
          ctx.fillRect(x, y, Math.max(1, w), Math.max(1, h));
          if (state.selection.type === "cell" && state.selection.id === cell.id) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h); }
          state.hitCells.push({ x, y, w, h, cell });
        }
        ctx.globalAlpha = 1;
        canvas.onmousemove = (event) => {
          const box = canvas.getBoundingClientRect();
          const x = event.clientX - box.left;
          const y = event.clientY - box.top;
          const hit = state.hitCells.find((item) => x >= item.x && x <= item.x + item.w && y >= item.y && y <= item.y + item.h);
          showTooltip(event, hit ? `${hit.cell.label}<br>${metricLabel(state.metric)}: ${fmt(hit.cell[state.metric])} ${unit()}<br>Count: ${countFmt(hit.cell.count)}` : "");
        };
        canvas.onclick = (event) => {
          const box = canvas.getBoundingClientRect();
          const x = event.clientX - box.left;
          const y = event.clientY - box.top;
          const hit = state.hitCells.find((item) => x >= item.x && x <= item.x + item.w && y >= item.y && y <= item.y + item.h);
          if (hit) { state.selection = { type: "cell", id: hit.cell.id, label: hit.cell.label }; render(); }
        };
        canvas.onmouseleave = () => showTooltip(null, "");
      }
      function selectedSeries(depths = chartDepths()) {
        return depths.map((depth) => {
          if (state.selection.type === "basin") return (depth.basins.find((b) => b.name === state.selection.id) || {})[state.metric] ?? null;
          if (state.selection.type === "cell") return (depth.grid_cells.find((c) => c.id === state.selection.id) || {})[state.metric] ?? null;
          return depth.global[state.metric] ?? null;
        });
      }
      function renderProfileChart() {
        const svg = document.getElementById("profileChart");
        const depths = chartDepths();
        const values = selectedSeries(depths);
        const labels = depths.map((d) => d.label);
        lineChart(svg, values, labels);
      }
      function renderBasinChart() {
        const svg = document.getElementById("basinChart");
        const basins = activeDepth().basins.filter((b) => b[state.metric] !== null);
        barChart(svg, basins.map((b) => b[state.metric]), basins.map((b) => b.name));
      }
      function lineChart(svg, values, labels) {
        const width = svg.clientWidth || 500, height = 230, pad = 34;
        const nums = values.filter((v) => v !== null).map(Number);
        const max = Math.max(...nums, 1);
        const points = values.map((v, i) => v === null ? null : [pad + i * ((width - pad * 2) / Math.max(1, values.length - 1)), height - pad - (Number(v) / max) * (height - pad * 2)]);
        svg.innerHTML = `<line class="axis" x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}"/><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height-pad}"/>`;
        const path = points.filter(Boolean).map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
        svg.insertAdjacentHTML("beforeend", `<path class="line" d="${path}"/>`);
        points.forEach((p, i) => { if (p) svg.insertAdjacentHTML("beforeend", `<circle class="point" cx="${p[0]}" cy="${p[1]}" r="4"><title>${labels[i]}: ${fmt(values[i])} ${unit()}</title></circle>`); });
      }
      function barChart(svg, values, labels) {
        const width = svg.clientWidth || 500, height = 230, pad = 34;
        const max = Math.max(...values.map(Number), 1);
        const barW = (width - pad * 2) / Math.max(1, values.length);
        svg.innerHTML = `<line class="axis" x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}"/>`;
        values.forEach((v, i) => {
          const h = (Number(v) / max) * (height - pad * 2);
          const x = pad + i * barW + 4;
          const y = height - pad - h;
          svg.insertAdjacentHTML("beforeend", `<rect class="bar" x="${x}" y="${y}" width="${Math.max(4, barW - 8)}" height="${h}"><title>${labels[i]}: ${fmt(v)} ${unit()}</title></rect><text x="${x}" y="${height-8}" fill="#9fb7b3" font-size="10">${labels[i].slice(0, 3)}</text>`);
        });
      }
      function showTooltip(event, html) {
        const tip = document.getElementById("tooltip");
        if (!html) { tip.style.display = "none"; return; }
        tip.innerHTML = html;
        tip.style.display = "block";
        tip.style.left = `${event.clientX + 14}px`;
        tip.style.top = `${event.clientY + 14}px`;
      }
      init().catch((error) => {
        document.getElementById("subtitle").textContent = `Failed to load dashboard data: ${error.message}`;
      });
    </script>
  </body>
</html>
"""


def export_error_analysis_dashboard(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    public_base_url: str | None = None,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
    analysis_workers: int = 1,
) -> dict[str, Any]:
    """Write `error-analysis.json` and `error-analysis.html` for one run."""
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
    html_path = output_dir / DEFAULT_ANALYSIS_HTML_NAME
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    html_path.write_text(_dashboard_html(), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "json_url": _resolve_layer_url(json_path.name, public_base_url=public_base_url),
        "html_url": _resolve_layer_url(html_path.name, public_base_url=public_base_url),
        "depth_level_count": int(len(payload["depth_levels"])),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a standalone absolute-error analysis dashboard."
    )
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
    """Run the error-analysis dashboard exporter from the command line."""
    args = _build_parser().parse_args()
    result = export_error_analysis_dashboard(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        public_base_url=args.public_base_url,
        grid_size_degrees=args.grid_size_degrees,
        top_cell_count=args.top_cell_count,
        analysis_workers=args.analysis_workers,
    )
    print(f"Wrote error analysis dashboard to: {result['output_dir']}")
    print(f"- data: {result['json_path']}")
    print(f"- html: {result['html_path']}")


if __name__ == "__main__":
    main()
