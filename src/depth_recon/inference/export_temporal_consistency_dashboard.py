# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_temporal_consistency_dashboard --temperature-run-dir inference/outputs/temporal_variables_2018_W22_W28/runs/temperature/2018_W22 --temperature-run-dir inference/outputs/temporal_variables_2018_W22_W28/runs/temperature/2018_W23 --salinity-run-dir inference/outputs/temporal_variables_2018_W22_W28/runs/salinity/2018_W22 --salinity-run-dir inference/outputs/temporal_variables_2018_W22_W28/runs/salinity/2018_W23 --output-dir inference/outputs/temporal_variables_2018_W22_W28/temporal --public-base-url https://globe-assets.hyperalislabs.com/inference_production/temporal
"""Export temporal consistency dashboard data from weekly inference runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import (
    _resolve_depth_export_artifacts,
    _resolve_land_mask_path,
    _resolve_layer_url,
    _resolve_run_artifacts,
    _run_variable_metadata,
    _sync_with_rclone,
)
from depth_recon.inference.export_error_analysis_dashboard import (
    BASIN_NAMES,
    DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
    DEFAULT_GRID_SIZE_DEGREES,
    METRIC_KEYS,
    _aggregate_summary_rows,
    _basin_label_array,
    _top_cells,
    _valid_array_points,
    aggregate_by_basin,
    aggregate_by_grid,
    summarize_values,
    write_analysis_grid_geojson,
)

DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME = "temporal"
DEFAULT_TEMPORAL_CONFIG_NAME = "temporal-config.json"
DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME = "temporal-analysis.json"
TEMPORAL_FIELD_KEYS = (
    "change_error",
    "prediction_change",
    "glorys_change",
    "prediction_flicker",
)
TEMPORAL_FIELD_LABELS = {
    "change_error": "Change Error",
    "prediction_change": "Prediction Change",
    "glorys_change": "GLORYS Change",
    "prediction_flicker": "Prediction Flicker",
}
TEMPORAL_FIELD_DESCRIPTIONS = {
    "change_error": "Absolute error between model and GLORYS week-to-week change.",
    "prediction_change": "Absolute week-to-week model prediction change.",
    "glorys_change": "Absolute week-to-week GLORYS change.",
    "prediction_flicker": "Absolute second temporal difference of model predictions over a 3-week window.",
}


@dataclass(frozen=True)
class TemporalRun:
    """Resolved metadata and depth artifacts for one weekly inference run."""

    run_dir: Path
    run_summary: dict[str, Any]
    variable_metadata: dict[str, Any]
    depth_exports: list[dict[str, Any]]
    selected_date: int
    iso_year: int | None
    iso_week: int | None
    land_mask_path: Path | None


def _period_url(
    name: str,
    *,
    variable: str | None,
    public_base_url: str | None,
) -> str:
    """Return a temporal-dashboard asset URL from the output-root perspective."""
    if variable is None:
        return _resolve_layer_url(name, public_base_url=public_base_url)
    clean_name = str(name).lstrip("/")
    if public_base_url is None:
        return f"{variable}/{clean_name}"
    return f"{public_base_url.rstrip('/')}/{variable}/{clean_name}"


def _resolve_selected_date(run_summary: dict[str, Any], run_dir: Path) -> int:
    """Resolve a sortable YYYYMMDD date from one run summary."""
    date_value = run_summary.get("target_date", run_summary.get("selected_date"))
    if date_value is None:
        raise ValueError(f"Run summary has no selected_date or target_date: {run_dir}")
    return int(date_value)


def _resolve_temporal_run(run_dir: Path) -> TemporalRun:
    """Resolve one weekly run directory into temporal aggregation metadata."""
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
    if prediction_path is None:
        raise FileNotFoundError(f"No prediction GeoTIFF was found in {run_dir}.")
    if ground_truth_path is None and not run_summary.get("depth_exports"):
        raise FileNotFoundError(f"No GLORYS GeoTIFF was found in {run_dir}.")

    depth_exports = _resolve_depth_export_artifacts(
        run_dir=run_dir,
        run_summary=run_summary,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
    )
    missing_truth = [
        str(depth_export.get("label", depth_export.get("suffix", "unknown")))
        for depth_export in depth_exports
        if depth_export.get("ground_truth_path") is None
    ]
    if missing_truth:
        raise FileNotFoundError(
            "Temporal consistency requires prediction and GLORYS rasters for every "
            f"depth export. Missing GLORYS for {', '.join(missing_truth)} in {run_dir}."
        )

    return TemporalRun(
        run_dir=run_dir,
        run_summary=run_summary,
        variable_metadata=_run_variable_metadata(run_summary),
        depth_exports=depth_exports,
        selected_date=_resolve_selected_date(run_summary, run_dir),
        iso_year=(
            None
            if run_summary.get("iso_year") is None
            else int(run_summary["iso_year"])
        ),
        iso_week=(
            None
            if run_summary.get("iso_week") is None
            else int(run_summary["iso_week"])
        ),
        land_mask_path=_resolve_land_mask_path(run_summary, run_dir=run_dir),
    )


def _depth_signature(depth_export: dict[str, Any]) -> tuple[str, int, float, float]:
    """Return the comparable identity for one exported depth raster."""
    return (
        str(depth_export["suffix"]),
        int(depth_export["channel_index"]),
        float(depth_export["requested_depth_m"]),
        float(depth_export["actual_depth_m"]),
    )


def _validate_run_series(runs: Sequence[TemporalRun]) -> None:
    """Validate that a temporal series can be compared without resampling."""
    if len(runs) < 2:
        raise ValueError("At least two weekly runs are required for temporal analysis.")

    variable = runs[0].variable_metadata["name"]
    depth_signature = [_depth_signature(depth) for depth in runs[0].depth_exports]
    for run in runs[1:]:
        if run.variable_metadata["name"] != variable:
            raise ValueError(
                "All runs in one temporal series must export the same variable: "
                f"{variable!r} != {run.variable_metadata['name']!r}."
            )
        other_signature = [_depth_signature(depth) for depth in run.depth_exports]
        if other_signature != depth_signature:
            raise ValueError(
                "Temporal runs must contain matching exported depth rasters. "
                f"Expected {depth_signature}, got {other_signature} in {run.run_dir}."
            )


def _read_float_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one raster band as float64 with nodata values converted to NaN."""
    with rasterio.open(path) as dataset:
        data = dataset.read(1, masked=False).astype(np.float64, copy=False)
        if dataset.nodata is not None and np.isfinite(float(dataset.nodata)):
            data = data.copy()
            data[np.isclose(data, float(dataset.nodata), atol=0.0, rtol=0.0)] = np.nan
        profile = {
            "path": str(path),
            "shape": tuple(int(value) for value in data.shape),
            "crs": None if dataset.crs is None else dataset.crs.to_string(),
            "transform": dataset.transform,
        }
    return data, profile


def _assert_matching_raster_profiles(
    profiles: Sequence[dict[str, Any]],
    *,
    context: str,
) -> None:
    """Raise when temporal rasters do not share an exact grid contract."""
    if not profiles:
        return
    first = profiles[0]
    for profile in profiles[1:]:
        if profile["shape"] != first["shape"]:
            raise ValueError(
                f"Raster shape mismatch for {context}: "
                f"{first['shape']} != {profile['shape']} ({profile['path']})."
            )
        if profile["crs"] != first["crs"]:
            raise ValueError(
                f"Raster CRS mismatch for {context}: "
                f"{first['crs']} != {profile['crs']} ({profile['path']})."
            )
        if profile["transform"] != first["transform"]:
            raise ValueError(
                f"Raster transform mismatch for {context}: {profile['path']}."
            )


def _nan_metric_array(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a float32 metric raster with invalid positions set to NaN."""
    metric = np.full(values.shape, np.nan, dtype=np.float32)
    metric[valid] = values[valid].astype(np.float32, copy=False)
    return metric


def _period_label(left_run: TemporalRun, right_run: TemporalRun) -> str:
    """Return a compact date label for one temporal interval."""
    return f"{left_run.selected_date} to {right_run.selected_date}"


def _flicker_label(
    previous_run: TemporalRun,
    center_run: TemporalRun,
    next_run: TemporalRun,
) -> str:
    """Return a compact date label for one 3-week flicker window."""
    return (
        f"{previous_run.selected_date} to {center_run.selected_date} "
        f"to {next_run.selected_date}"
    )


def _interval_period_metadata(
    index: int, left_run: TemporalRun, right_run: TemporalRun
) -> dict[str, Any]:
    """Build period metadata for one week-to-week temporal interval."""
    return {
        "index": int(index),
        "period_key": f"interval_{int(index):03d}",
        "kind": "interval",
        "label": _period_label(left_run, right_run),
        "start_date": int(left_run.selected_date),
        "end_date": int(right_run.selected_date),
        "start_iso_year": left_run.iso_year,
        "start_iso_week": left_run.iso_week,
        "end_iso_year": right_run.iso_year,
        "end_iso_week": right_run.iso_week,
    }


def _flicker_period_metadata(
    index: int,
    previous_run: TemporalRun,
    center_run: TemporalRun,
    next_run: TemporalRun,
) -> dict[str, Any]:
    """Build period metadata for one 3-week flicker window."""
    return {
        "index": int(index),
        "period_key": f"flicker_{int(index):03d}",
        "kind": "window",
        "label": _flicker_label(previous_run, center_run, next_run),
        "previous_date": int(previous_run.selected_date),
        "center_date": int(center_run.selected_date),
        "next_date": int(next_run.selected_date),
        "previous_iso_year": previous_run.iso_year,
        "previous_iso_week": previous_run.iso_week,
        "center_iso_year": center_run.iso_year,
        "center_iso_week": center_run.iso_week,
        "next_iso_year": next_run.iso_year,
        "next_iso_week": next_run.iso_week,
    }


def _build_period_stats(
    values: np.ndarray,
    *,
    transform: Any,
    period_metadata: dict[str, Any],
    grid_size_degrees: float,
    top_cell_count: int,
) -> dict[str, Any]:
    """Aggregate one temporal metric raster into global, basin, and grid rows."""
    metric_values, lons, lats = _valid_array_points(values, transform=transform)
    basin_labels = _basin_label_array(lons, lats)
    grid_cells = aggregate_by_grid(
        metric_values,
        lons,
        lats,
        grid_size_degrees=grid_size_degrees,
        basin_labels=basin_labels,
    )
    return {
        **period_metadata,
        "global": summarize_values(metric_values),
        "basins": aggregate_by_basin(
            metric_values,
            lons,
            lats,
            basin_labels=basin_labels,
        ),
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


def _build_depth_temporal_fields(
    runs: Sequence[TemporalRun],
    *,
    depth_index: int,
    grid_size_degrees: float,
    top_cell_count: int,
) -> dict[str, dict[str, Any]]:
    """Build temporal field summaries for one exported depth across all runs."""
    fields = {
        key: {
            "key": key,
            "label": TEMPORAL_FIELD_LABELS[key],
            "description": TEMPORAL_FIELD_DESCRIPTIONS[key],
            "periods": [],
        }
        for key in TEMPORAL_FIELD_KEYS
    }

    predictions: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    profiles: list[dict[str, Any]] = []
    for run in runs:
        depth_export = run.depth_exports[depth_index]
        prediction, prediction_profile = _read_float_raster(
            Path(depth_export["prediction_path"])
        )
        ground_truth, ground_truth_profile = _read_float_raster(
            Path(depth_export["ground_truth_path"])
        )
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        profiles.extend([prediction_profile, ground_truth_profile])

    _assert_matching_raster_profiles(
        profiles,
        context=f"depth {runs[0].depth_exports[depth_index]['label']}",
    )
    transform = profiles[0]["transform"]

    for interval_index in range(len(runs) - 1):
        left_run = runs[interval_index]
        right_run = runs[interval_index + 1]
        pred_delta = predictions[interval_index + 1] - predictions[interval_index]
        truth_delta = ground_truths[interval_index + 1] - ground_truths[interval_index]
        valid = (
            np.isfinite(predictions[interval_index])
            & np.isfinite(predictions[interval_index + 1])
            & np.isfinite(ground_truths[interval_index])
            & np.isfinite(ground_truths[interval_index + 1])
        )
        period_metadata = _interval_period_metadata(
            interval_index,
            left_run,
            right_run,
        )
        metrics = {
            "change_error": np.abs(pred_delta - truth_delta),
            "prediction_change": np.abs(pred_delta),
            "glorys_change": np.abs(truth_delta),
        }
        for field_key, metric_values in metrics.items():
            fields[field_key]["periods"].append(
                _build_period_stats(
                    _nan_metric_array(metric_values, valid),
                    transform=transform,
                    period_metadata=period_metadata,
                    grid_size_degrees=grid_size_degrees,
                    top_cell_count=top_cell_count,
                )
            )

    for flicker_index in range(len(runs) - 2):
        previous_run = runs[flicker_index]
        center_run = runs[flicker_index + 1]
        next_run = runs[flicker_index + 2]
        flicker = np.abs(
            predictions[flicker_index + 2]
            - 2.0 * predictions[flicker_index + 1]
            + predictions[flicker_index]
        )
        valid = (
            np.isfinite(predictions[flicker_index])
            & np.isfinite(predictions[flicker_index + 1])
            & np.isfinite(predictions[flicker_index + 2])
        )
        fields["prediction_flicker"]["periods"].append(
            _build_period_stats(
                _nan_metric_array(flicker, valid),
                transform=transform,
                period_metadata=_flicker_period_metadata(
                    flicker_index,
                    previous_run,
                    center_run,
                    next_run,
                ),
                grid_size_degrees=grid_size_degrees,
                top_cell_count=top_cell_count,
            )
        )

    return fields


def _period_by_key(periods: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index temporal periods by their stable period key."""
    return {str(period["period_key"]): period for period in periods}


def _aggregate_basin_period_rows(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-depth basin rows for an all-depth temporal period."""
    basins: list[dict[str, Any]] = []
    for basin in BASIN_NAMES:
        rows = [
            row
            for period in periods
            for row in period.get("basins", [])
            if row.get("name") == basin
        ]
        basin_row = _aggregate_summary_rows(rows)
        basin_row["name"] = basin
        basins.append(basin_row)
    return basins


def _aggregate_grid_period_rows(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-depth grid rows for an all-depth temporal period."""
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for period in periods:
        for cell in period.get("grid_cells", []):
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


def _aggregate_all_depth_period(
    period_template: dict[str, Any],
    depth_periods: list[dict[str, Any]],
    *,
    top_cell_count: int,
) -> dict[str, Any]:
    """Build one all-depth period from matching per-depth period rows."""
    grid_cells = _aggregate_grid_period_rows(depth_periods)
    return {
        **{
            key: period_template[key]
            for key in period_template
            if key
            not in {
                "global",
                "basins",
                "grid_cells",
                "top_cells",
            }
        },
        "global": _aggregate_summary_rows(
            [period["global"] for period in depth_periods]
        ),
        "basins": _aggregate_basin_period_rows(depth_periods),
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


def _build_all_depth_temporal_level(
    depth_levels: list[dict[str, Any]],
    *,
    top_cell_count: int,
) -> dict[str, Any]:
    """Build an all-depth aggregate temporal level."""
    fields: dict[str, dict[str, Any]] = {}
    for field_key in TEMPORAL_FIELD_KEYS:
        periods_by_depth = [
            _period_by_key(depth["fields"][field_key]["periods"])
            for depth in depth_levels
        ]
        all_periods: list[dict[str, Any]] = []
        for period_key, period_template in periods_by_depth[0].items():
            matching_periods = [
                periods[period_key]
                for periods in periods_by_depth
                if period_key in periods
            ]
            if len(matching_periods) != len(periods_by_depth):
                continue
            all_periods.append(
                _aggregate_all_depth_period(
                    period_template,
                    matching_periods,
                    top_cell_count=top_cell_count,
                )
            )
        fields[field_key] = {
            "key": field_key,
            "label": TEMPORAL_FIELD_LABELS[field_key],
            "description": TEMPORAL_FIELD_DESCRIPTIONS[field_key],
            "periods": all_periods,
        }

    return {
        "index": -1,
        "suffix": "all_depths",
        "label": "All Depths",
        "requested_depth_m": None,
        "actual_depth_m": None,
        "channel_index": None,
        "is_aggregate": True,
        "depth_count": int(len(depth_levels)),
        "aggregation_method": "Count-weighted average of per-depth temporal metrics; counts are summed across depths.",
        "fields": fields,
    }


def _run_rows(runs: Sequence[TemporalRun]) -> list[dict[str, Any]]:
    """Return compact run metadata for the temporal payload."""
    return [
        {
            "index": int(index),
            "run_dir": str(run.run_dir),
            "selected_date": int(run.selected_date),
            "target_date": run.run_summary.get(
                "target_date",
                run.run_summary.get("selected_date"),
            ),
            "iso_year": run.iso_year,
            "iso_week": run.iso_week,
        }
        for index, run in enumerate(runs)
    ]


def build_temporal_analysis_payload(
    *,
    run_dirs: Sequence[Path],
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
) -> dict[str, Any]:
    """Build the JSON-serializable temporal consistency payload."""
    runs = sorted(
        [_resolve_temporal_run(Path(run_dir)) for run_dir in run_dirs],
        key=lambda run: run.selected_date,
    )
    _validate_run_series(runs)

    variable_metadata = runs[0].variable_metadata
    depth_levels: list[dict[str, Any]] = []
    for depth_index, depth_export in enumerate(runs[0].depth_exports):
        depth_levels.append(
            {
                "index": int(depth_index),
                "suffix": str(depth_export["suffix"]),
                "label": str(depth_export["label"]),
                "requested_depth_m": float(depth_export["requested_depth_m"]),
                "actual_depth_m": float(depth_export["actual_depth_m"]),
                "channel_index": int(depth_export["channel_index"]),
                "is_aggregate": False,
                "fields": _build_depth_temporal_fields(
                    runs,
                    depth_index=depth_index,
                    grid_size_degrees=grid_size_degrees,
                    top_cell_count=top_cell_count,
                ),
            }
        )

    displayed_depth_levels = [
        _build_all_depth_temporal_level(
            depth_levels,
            top_cell_count=top_cell_count,
        ),
        *depth_levels,
    ]
    return {
        "schema_version": 1,
        "title": "DepthDif Temporal Consistency",
        "description": (
            "Aggregated temporal consistency diagnostics comparing model and "
            "GLORYS changes across consecutive weekly global exports."
        ),
        "run": {
            "run_count": int(len(runs)),
            "start_date": int(runs[0].selected_date),
            "end_date": int(runs[-1].selected_date),
            "start_iso_year": runs[0].iso_year,
            "start_iso_week": runs[0].iso_week,
            "end_iso_year": runs[-1].iso_year,
            "end_iso_week": runs[-1].iso_week,
            "runs": _run_rows(runs),
        },
        "variable": {
            "name": str(variable_metadata["name"]),
            "label": str(variable_metadata["label"]),
            "value_units": str(variable_metadata["value_units"]),
            "value_unit_label": str(variable_metadata["value_unit_label"]),
        },
        "metrics": list(METRIC_KEYS),
        "temporal_fields": [
            {
                "key": key,
                "label": TEMPORAL_FIELD_LABELS[key],
                "description": TEMPORAL_FIELD_DESCRIPTIONS[key],
            }
            for key in TEMPORAL_FIELD_KEYS
        ],
        "grouping": {
            "basins": list(BASIN_NAMES),
            "grid_size_degrees": float(grid_size_degrees),
            "basin_method": "Land-filtered deterministic lon/lat basin buckets with dominant basin labels on grid cells.",
        },
        "depth_levels": displayed_depth_levels,
    }


def _write_json(
    path: Path, payload: dict[str, Any], *, indent: int | None = None
) -> Path:
    """Write a JSON payload with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, separators=None if indent else (",", ":"))
        f.write("\n")
    return path


def _ordered_variable_items(
    variable_run_dirs: dict[str, Sequence[Path]],
) -> list[tuple[str, Sequence[Path]]]:
    """Return temporal variable run directories in viewer order."""
    items = {
        str(key).strip().lower(): value for key, value in variable_run_dirs.items()
    }
    ordered: list[tuple[str, Sequence[Path]]] = []
    for key in ("temperature", "salinity"):
        if key in items:
            ordered.append((key, items.pop(key)))
    ordered.extend(sorted(items.items(), key=lambda item: item[0]))
    return ordered


def _default_variable_key(variables: dict[str, Any]) -> str:
    """Return the preferred default variable key for dashboard controls."""
    if "temperature" in variables:
        return "temperature"
    return next(iter(variables))


def _copy_dashboard_pages(output_dir: Path) -> None:
    """Copy standalone temporal dashboard files beside the generated config."""
    repo_root = Path(__file__).resolve().parents[3]
    source_files = {
        repo_root / "docs" / "temporal" / "index.html": output_dir / "index.html",
        repo_root
        / "docs"
        / "javascripts"
        / "temporal-dashboard.js": output_dir
        / "javascripts"
        / "temporal-dashboard.js",
        repo_root
        / "docs"
        / "stylesheets"
        / "temporal-dashboard.css": output_dir
        / "stylesheets"
        / "temporal-dashboard.css",
    }
    for source_path, destination_path in source_files.items():
        if not source_path.exists():
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def export_temporal_dashboard_assets(
    *,
    variable_run_dirs: dict[str, Sequence[Path]],
    output_dir: Path,
    public_base_url: str | None = None,
    grid_size_degrees: float = DEFAULT_GRID_SIZE_DEGREES,
    top_cell_count: int = 24,
    rclone_remote: str | None = None,
    copy_dashboard: bool = True,
) -> dict[str, Any]:
    """Write temporal dashboard JSON assets for one or more variables."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variables: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for variable, raw_run_dirs in _ordered_variable_items(variable_run_dirs):
        run_dirs = [Path(run_dir) for run_dir in raw_run_dirs]
        if not run_dirs:
            continue
        variable_dir = output_dir / str(variable)
        variable_dir.mkdir(parents=True, exist_ok=True)
        payload = build_temporal_analysis_payload(
            run_dirs=run_dirs,
            grid_size_degrees=grid_size_degrees,
            top_cell_count=top_cell_count,
        )
        payload_variable = str(payload["variable"]["name"])
        if payload_variable != str(variable):
            raise ValueError(
                f"Variable key {variable!r} does not match run payload variable "
                f"{payload_variable!r}."
            )

        analysis_path = _write_json(
            variable_dir / DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME,
            payload,
        )
        first_run = _resolve_temporal_run(sorted(run_dirs)[0])
        grid_path: Path | None = None
        if first_run.land_mask_path is not None:
            grid_path = write_analysis_grid_geojson(
                output_path=variable_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
                land_mask_path=first_run.land_mask_path,
                grid_size_degrees=grid_size_degrees,
            )
        variables[variable] = {
            "variable": payload_variable,
            "variable_label": payload["variable"]["label"],
            "value_units": payload["variable"]["value_units"],
            "value_unit_label": payload["variable"]["value_unit_label"],
            "run_count": payload["run"]["run_count"],
            "start_date": payload["run"]["start_date"],
            "end_date": payload["run"]["end_date"],
            "temporal_analysis_data_url": _period_url(
                DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME,
                variable=variable,
                public_base_url=public_base_url,
            ),
            "analysis_grid_geojson_url": (
                None
                if grid_path is None
                else _period_url(
                    DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
                    variable=variable,
                    public_base_url=public_base_url,
                )
            ),
        }
        results[variable] = {
            "analysis_json_path": str(analysis_path),
            "analysis_grid_geojson_path": None if grid_path is None else str(grid_path),
            "run_count": int(payload["run"]["run_count"]),
            "depth_level_count": int(len(payload["depth_levels"])),
        }

    if not variables:
        raise ValueError("No temporal variable run directories were provided.")

    default_variable = _default_variable_key(variables)
    config = {
        "schema_version": 1,
        "title": "DepthDif Temporal Consistency",
        "default_variable": default_variable,
        "available_variables": list(variables.keys()),
        "variables": variables,
    }
    config_path = _write_json(
        output_dir / DEFAULT_TEMPORAL_CONFIG_NAME,
        config,
        indent=2,
    )
    run_summary = {
        "temporal_dashboard": {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "public_base_url": public_base_url,
            "default_variable": default_variable,
            "variables": variables,
            "grid_size_degrees": float(grid_size_degrees),
            "top_cell_count": int(top_cell_count),
        },
        "variable_results": results,
    }
    summary_path = output_dir / "run_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    if copy_dashboard:
        _copy_dashboard_pages(output_dir)

    upload_ok: bool | None = None
    upload_message: str | None = None
    if rclone_remote is not None:
        ok, message = _sync_with_rclone(output_dir, rclone_remote)
        upload_ok = bool(ok)
        upload_message = str(message)

    return {
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "variables": list(variables.keys()),
        "default_variable": default_variable,
        "upload_requested": rclone_remote is not None,
        "upload_ok": upload_ok,
        "upload_message": upload_message,
        "upload_remote": rclone_remote,
        "variable_results": results,
    }


def _collect_variable_run_dirs(args: argparse.Namespace) -> dict[str, list[Path]]:
    """Collect CLI run-dir arguments into a variable-keyed mapping."""
    variable_run_dirs: dict[str, list[Path]] = {}
    if args.temperature_run_dir:
        variable_run_dirs["temperature"] = [
            Path(path) for path in args.temperature_run_dir
        ]
    if args.salinity_run_dir:
        variable_run_dirs["salinity"] = [Path(path) for path in args.salinity_run_dir]
    for run_dir in args.run_dir or []:
        run = _resolve_temporal_run(Path(run_dir))
        variable = str(run.variable_metadata["name"])
        variable_run_dirs.setdefault(variable, []).append(Path(run_dir))
    return variable_run_dirs


def _build_parser() -> argparse.ArgumentParser:
    """Build the temporal dashboard aggregation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate weekly inference runs into temporal dashboard JSON assets."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="Single-variable run directory. Can be repeated; variable is read from run_summary.yaml.",
    )
    parser.add_argument(
        "--temperature-run-dir",
        action="append",
        default=[],
        type=Path,
        help="Temperature weekly run directory. Repeat in any order.",
    )
    parser.add_argument(
        "--salinity-run-dir",
        action="append",
        default=[],
        type=Path,
        help="Salinity weekly run directory. Repeat in any order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference/outputs") / DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME,
    )
    parser.add_argument("--public-base-url", type=str, default=None)
    parser.add_argument("--rclone-remote", type=str, default=None)
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=DEFAULT_GRID_SIZE_DEGREES,
    )
    parser.add_argument("--top-cell-count", type=int, default=24)
    parser.add_argument(
        "--copy-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the standalone dashboard HTML/CSS/JS beside the generated config.",
    )
    return parser


def main() -> None:
    """Run the temporal dashboard aggregation CLI."""
    args = _build_parser().parse_args()
    result = export_temporal_dashboard_assets(
        variable_run_dirs=_collect_variable_run_dirs(args),
        output_dir=args.output_dir,
        public_base_url=args.public_base_url,
        grid_size_degrees=args.grid_size_degrees,
        top_cell_count=args.top_cell_count,
        rclone_remote=args.rclone_remote,
        copy_dashboard=bool(args.copy_dashboard),
    )
    print(f"Wrote temporal dashboard assets to: {result['output_dir']}")
    print(f"- config: {result['config_path']}")
    for variable, variable_result in result["variable_results"].items():
        print(f"- {variable}: {variable_result['analysis_json_path']}")
    if result.get("upload_requested"):
        print(f"- upload: {result.get('upload_message')}")


if __name__ == "__main__":
    main()
