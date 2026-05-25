# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_temporal_consistency_dashboard --temperature-run-dir inference/outputs/temporal_variables_2018/runs/temperature/2018_W01 --salinity-run-dir inference/outputs/temporal_variables_2018/runs/salinity/2018_W01 --output-dir inference/outputs/temporal_variables_2018/temporal --validation-year 2018 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/temporal
"""Export compact temporal dashboard data from weekly validation-year runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import (
    _coerce_existing_path,
    _resolve_depth_export_artifacts,
    _resolve_layer_url,
    _resolve_run_artifacts,
    _run_variable_metadata,
    _sync_with_rclone,
)
from depth_recon.inference.export_error_analysis_dashboard import (
    BASIN_BUTTON_NAMES,
    BASIN_NAMES,
    world_ocean_region_geojson_features,
)

DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME = "temporal"
DEFAULT_TEMPORAL_CONFIG_NAME = "temporal-config.json"
DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME = "temporal-analysis.json"
DEFAULT_TEMPORAL_BASIN_DATA_DIR_NAME = "basins"
DEFAULT_TEMPORAL_BASIN_MAP_GEOJSON_NAME = "basin-map.geojson"
DEFAULT_TEMPORAL_VALIDATION_YEAR = 2018
DEFAULT_TEMPORAL_YEAR_WEEK_COUNT = 52
BASIN_DISPLAY_NAMES = {basin: basin for basin in BASIN_NAMES}


def _default_basin_name() -> str:
    """Return the preferred initial basin for temporal dashboard controls."""
    if "North Pacific Ocean" in BASIN_BUTTON_NAMES:
        return "North Pacific Ocean"
    return BASIN_BUTTON_NAMES[0]


@dataclass(frozen=True)
class TemporalRun:
    """Resolved weekly run metadata needed for temporal aggregation."""

    run_dir: Path
    run_summary: dict[str, Any]
    compact_summary: dict[str, Any]
    variable_metadata: dict[str, Any]
    selected_date: int
    iso_year: int | None
    iso_week: int | None
    depth_exports: list[dict[str, Any]]
    ten_meter_artifact: dict[str, Any]


def _temporal_url(
    relative_path: str,
    *,
    public_base_url: str | None,
) -> str:
    """Return a temporal-dashboard asset URL from the output-root perspective."""
    return _resolve_layer_url(relative_path, public_base_url=public_base_url)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    indent: int | None = None,
) -> Path:
    """Write a JSON payload with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, separators=None if indent else (",", ":"))
        f.write("\n")
    return path


def _resolve_selected_date(run_summary: dict[str, Any], run_dir: Path) -> int:
    """Resolve a sortable YYYYMMDD date from one run summary."""
    date_value = run_summary.get("target_date", run_summary.get("selected_date"))
    if date_value is None:
        raise ValueError(f"Run summary has no selected_date or target_date: {run_dir}")
    return int(date_value)


def _resolve_compact_summary_path(run_summary: dict[str, Any], run_dir: Path) -> Path:
    """Resolve the compact basin-depth summary emitted by temporal exports."""
    path = _coerce_existing_path(
        run_summary.get("temporal_basin_depth_error_json_path"),
        run_dir=run_dir,
    )
    if path is None:
        raise FileNotFoundError(
            "Temporal dashboard v2 requires temporal_basin_depth_error_json_path "
            f"in {run_dir / 'run_summary.yaml'}."
        )
    return path


def _ten_meter_artifact(
    depth_exports: Sequence[dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    """Return the retained 10m prediction and absolute-error artifacts for one run."""
    for depth_export in depth_exports:
        if str(depth_export.get("suffix")) != "10m":
            continue
        prediction_path = depth_export.get("prediction_path")
        absolute_error_path = depth_export.get("absolute_error_path")
        if prediction_path is None or absolute_error_path is None:
            raise FileNotFoundError(
                f"10m prediction and absolute-error rasters are required in {run_dir}."
            )
        return {
            "label": str(depth_export.get("label", "10m")),
            "suffix": "10m",
            "requested_depth_m": float(depth_export.get("requested_depth_m", 10.0)),
            "actual_depth_m": float(depth_export.get("actual_depth_m", 10.0)),
            "channel_index": int(depth_export.get("channel_index", 0)),
            "prediction_path": Path(prediction_path),
            "absolute_error_path": Path(absolute_error_path),
        }
    raise FileNotFoundError(f"No retained 10m depth export was found in {run_dir}.")


def _resolve_temporal_run(run_dir: Path) -> TemporalRun:
    """Resolve one weekly run directory into compact temporal metadata."""
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

    depth_exports = _resolve_depth_export_artifacts(
        run_dir=run_dir,
        run_summary=run_summary,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
    )
    compact_summary = _read_json(_resolve_compact_summary_path(run_summary, run_dir))
    variable_metadata = _run_variable_metadata(run_summary)
    compact_variable = compact_summary.get("variable", {})
    if str(compact_variable.get("name", variable_metadata["name"])) != str(
        variable_metadata["name"]
    ):
        raise ValueError(
            f"Compact summary variable does not match run summary in {run_dir}."
        )

    return TemporalRun(
        run_dir=run_dir,
        run_summary=run_summary,
        compact_summary=compact_summary,
        variable_metadata=variable_metadata,
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
        depth_exports=depth_exports,
        ten_meter_artifact=_ten_meter_artifact(depth_exports, run_dir),
    )


def _depth_signature(depth: dict[str, Any]) -> tuple[str, int, float, float]:
    """Return the comparable identity for one compact depth row."""
    return (
        str(depth["suffix"]),
        int(depth["channel_index"]),
        float(depth["requested_depth_m"]),
        float(depth["actual_depth_m"]),
    )


def _sorted_runs(run_dirs: Sequence[Path]) -> list[TemporalRun]:
    """Resolve and sort weekly runs by selected date."""
    runs = sorted(
        [_resolve_temporal_run(Path(run_dir)) for run_dir in run_dirs],
        key=lambda run: run.selected_date,
    )
    if not runs:
        raise ValueError("At least one weekly run is required for temporal analysis.")
    variable = runs[0].variable_metadata["name"]
    signature = [
        _depth_signature(depth) for depth in runs[0].compact_summary["depth_levels"]
    ]
    for run in runs[1:]:
        if run.variable_metadata["name"] != variable:
            raise ValueError(
                "All runs in one temporal series must export the same variable: "
                f"{variable!r} != {run.variable_metadata['name']!r}."
            )
        other_signature = [
            _depth_signature(depth) for depth in run.compact_summary["depth_levels"]
        ]
        if other_signature != signature:
            raise ValueError(
                "Temporal runs must contain matching compact depth metadata. "
                f"Expected {signature}, got {other_signature} in {run.run_dir}."
            )
    return runs


def _empty_depth_accumulators(
    depth_levels: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Create zeroed per-basin accumulators matching compact depth metadata."""
    return {
        basin: [
            {
                "index": int(depth["index"]),
                "suffix": str(depth["suffix"]),
                "label": str(depth["label"]),
                "requested_depth_m": float(depth["requested_depth_m"]),
                "actual_depth_m": float(depth["actual_depth_m"]),
                "channel_index": int(depth["channel_index"]),
                "count": 0,
                "sum_absolute_error": 0.0,
            }
            for depth in depth_levels
        ]
        for basin in BASIN_NAMES
    }


def _add_depth_rows(
    accumulators: dict[str, list[dict[str, Any]]],
    depth_levels: Sequence[dict[str, Any]],
) -> None:
    """Add one weekly compact summary into basin-depth accumulators."""
    for depth_index, depth in enumerate(depth_levels):
        for row in depth.get("basins", []):
            basin = str(row.get("name", "Other"))
            if basin not in accumulators:
                continue
            # The weekly summary stores sums so the yearly mean is exact across
            # weeks with different valid-pixel counts.
            accumulators[basin][depth_index]["count"] += int(row.get("count", 0) or 0)
            accumulators[basin][depth_index]["sum_absolute_error"] += float(
                row.get("sum_absolute_error", 0.0) or 0.0
            )


def _finalize_depth_errors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert accumulated sums and counts into dashboard depth-error rows."""
    finalized: list[dict[str, Any]] = []
    for row in rows:
        count = int(row["count"])
        total = float(row["sum_absolute_error"])
        finalized.append(
            {
                **row,
                "count": count,
                "sum_absolute_error": total,
                "mean_absolute_error": (
                    None if count <= 0 else float(total / float(count))
                ),
            }
        )
    return finalized


def _run_rows(runs: Sequence[TemporalRun]) -> list[dict[str, Any]]:
    """Return compact run metadata for the temporal payload."""
    return [
        {
            "index": int(index),
            "run_dir": str(run.run_dir),
            "selected_date": int(run.selected_date),
            "target_date": run.run_summary.get(
                "target_date", run.run_summary.get("selected_date")
            ),
            "iso_year": run.iso_year,
            "iso_week": run.iso_week,
        }
        for index, run in enumerate(runs)
    ]


def build_temporal_analysis_payload(
    *,
    run_dirs: Sequence[Path],
    validation_year: int = DEFAULT_TEMPORAL_VALIDATION_YEAR,
    grid_size_degrees: float | None = None,
    top_cell_count: int | None = None,
) -> dict[str, Any]:
    """Build one variable's compact validation-year temporal summary."""
    _ = grid_size_degrees, top_cell_count
    runs = _sorted_runs(run_dirs)
    first_depth_levels = runs[0].compact_summary["depth_levels"]
    accumulators = _empty_depth_accumulators(first_depth_levels)
    for run in runs:
        _add_depth_rows(accumulators, run.compact_summary["depth_levels"])

    basin_depth_errors = {
        basin: _finalize_depth_errors(rows) for basin, rows in accumulators.items()
    }
    variable_metadata = runs[0].variable_metadata
    return {
        "schema_version": 2,
        "title": "DepthDif Temporal Validation Error",
        "description": "Validation-year mean absolute prediction-vs-GLORYS error by basin and native depth.",
        "validation_year": int(validation_year),
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
        "grouping": {
            "basins": list(BASIN_NAMES),
            "basin_method": "Land-filtered world_oceans.geojson region polygons.",
        },
        "depth_levels": [
            {
                "index": int(depth["index"]),
                "suffix": str(depth["suffix"]),
                "label": str(depth["label"]),
                "requested_depth_m": float(depth["requested_depth_m"]),
                "actual_depth_m": float(depth["actual_depth_m"]),
                "channel_index": int(depth["channel_index"]),
            }
            for depth in first_depth_levels
        ],
        "basin_depth_errors": basin_depth_errors,
        "weekly_10m_artifacts": [run.ten_meter_artifact for run in runs],
    }


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


def _basin_map_payload() -> dict[str, Any]:
    """Return authoritative GeoJSON basin polygons for the dashboard selector."""
    return {
        "type": "FeatureCollection",
        "name": "DepthDif world_oceans.geojson temporal basins",
        "features": world_ocean_region_geojson_features(),
    }


def _safe_basin_file_name(basin: str) -> str:
    """Return a stable JSON filename for one basin."""
    stem = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(basin).strip()
    ).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return f"{stem or 'basin'}.json"


def _copy_weekly_10m_artifacts(
    *,
    output_dir: Path,
    variable: str,
    variable_payload: dict[str, Any],
    public_base_url: str | None,
) -> list[dict[str, Any]]:
    """Copy retained weekly 10m rasters into the temporal bundle."""
    copied: list[dict[str, Any]] = []
    for artifact in variable_payload["weekly_10m_artifacts"]:
        iso_year = variable_payload["run"]["runs"][len(copied)].get("iso_year")
        iso_week = variable_payload["run"]["runs"][len(copied)].get("iso_week")
        week_name = (
            f"{int(iso_year)}_W{int(iso_week):02d}"
            if iso_year and iso_week
            else f"week_{len(copied):03d}"
        )
        destination_dir = output_dir / "weekly" / variable / week_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        prediction_source = Path(artifact["prediction_path"])
        error_source = Path(artifact["absolute_error_path"])
        prediction_dest = destination_dir / prediction_source.name
        error_dest = destination_dir / error_source.name
        if prediction_source.resolve() != prediction_dest.resolve():
            shutil.copy2(prediction_source, prediction_dest)
        if error_source.resolve() != error_dest.resolve():
            shutil.copy2(error_source, error_dest)
        prediction_relative = prediction_dest.relative_to(output_dir).as_posix()
        error_relative = error_dest.relative_to(output_dir).as_posix()
        copied.append(
            {
                "iso_year": iso_year,
                "iso_week": iso_week,
                "selected_date": variable_payload["run"]["runs"][len(copied)][
                    "selected_date"
                ],
                "depth_label": artifact["label"],
                "requested_depth_m": artifact["requested_depth_m"],
                "actual_depth_m": artifact["actual_depth_m"],
                "prediction_url": _temporal_url(
                    prediction_relative, public_base_url=public_base_url
                ),
                "absolute_error_url": _temporal_url(
                    error_relative, public_base_url=public_base_url
                ),
            }
        )
    return copied


def _payload_week_signature(
    payload: dict[str, Any],
) -> list[tuple[int | None, int | None, int]]:
    """Return comparable weekly coverage metadata for one variable payload."""
    return [
        (run.get("iso_year"), run.get("iso_week"), int(run["selected_date"]))
        for run in payload["run"]["runs"]
    ]


def _payload_depth_signature(
    payload: dict[str, Any],
) -> list[tuple[str, int, float, float]]:
    """Return comparable depth metadata for one variable payload."""
    return [_depth_signature(depth) for depth in payload["depth_levels"]]


def _validate_variable_payloads(variable_payloads: dict[str, dict[str, Any]]) -> None:
    """Require variables to cover the same weeks and native depths."""
    if len(variable_payloads) <= 1:
        return
    first_variable, first_payload = next(iter(variable_payloads.items()))
    week_signature = _payload_week_signature(first_payload)
    depth_signature = _payload_depth_signature(first_payload)
    for variable, payload in list(variable_payloads.items())[1:]:
        if _payload_week_signature(payload) != week_signature:
            raise ValueError(
                "Temporal variables must cover matching weekly runs: "
                f"{first_variable!r} != {variable!r}."
            )
        if _payload_depth_signature(payload) != depth_signature:
            raise ValueError(
                "Temporal variables must contain matching compact depth metadata: "
                f"{first_variable!r} != {variable!r}."
            )


def _basin_payload(
    *,
    basin: str,
    validation_year: int,
    variable_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON loaded when one basin is active in the dashboard."""
    variables: dict[str, Any] = {}
    for variable, payload in variable_payloads.items():
        variables[variable] = {
            "variable": payload["variable"],
            "run": payload["run"],
            "depth_errors": payload["basin_depth_errors"][basin],
        }
    return {
        "schema_version": 2,
        "basin": basin,
        "basin_label": BASIN_DISPLAY_NAMES.get(basin, basin),
        "validation_year": int(validation_year),
        "variables": variables,
    }


def export_temporal_dashboard_assets(
    *,
    variable_run_dirs: dict[str, Sequence[Path]],
    output_dir: Path,
    public_base_url: str | None = None,
    grid_size_degrees: float | None = None,
    top_cell_count: int | None = None,
    rclone_remote: str | None = None,
    copy_dashboard: bool = True,
    validation_year: int = DEFAULT_TEMPORAL_VALIDATION_YEAR,
) -> dict[str, Any]:
    """Write compact temporal dashboard JSON and retained 10m assets."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variable_payloads: dict[str, dict[str, Any]] = {}
    variables: dict[str, Any] = {}
    results: dict[str, Any] = {}

    for variable, raw_run_dirs in _ordered_variable_items(variable_run_dirs):
        run_dirs = [Path(run_dir) for run_dir in raw_run_dirs]
        if not run_dirs:
            continue
        payload = build_temporal_analysis_payload(
            run_dirs=run_dirs,
            validation_year=validation_year,
            grid_size_degrees=grid_size_degrees,
            top_cell_count=top_cell_count,
        )
        payload_variable = str(payload["variable"]["name"])
        if payload_variable != str(variable):
            raise ValueError(
                f"Variable key {variable!r} does not match run payload variable "
                f"{payload_variable!r}."
            )
        variable_payloads[variable] = payload
        weekly_artifacts = _copy_weekly_10m_artifacts(
            output_dir=output_dir,
            variable=variable,
            variable_payload=payload,
            public_base_url=public_base_url,
        )
        variables[variable] = {
            "variable": payload_variable,
            "variable_label": payload["variable"]["label"],
            "value_units": payload["variable"]["value_units"],
            "value_unit_label": payload["variable"]["value_unit_label"],
            "run_count": payload["run"]["run_count"],
            "start_date": payload["run"]["start_date"],
            "end_date": payload["run"]["end_date"],
            "depth_level_count": len(payload["depth_levels"]),
            "weekly_10m_artifacts": weekly_artifacts,
        }
        results[variable] = {
            "run_count": int(payload["run"]["run_count"]),
            "depth_level_count": int(len(payload["depth_levels"])),
            "weekly_10m_artifact_count": int(len(weekly_artifacts)),
        }

    if not variables:
        raise ValueError("No temporal variable run directories were provided.")
    _validate_variable_payloads(variable_payloads)

    basin_dir = output_dir / DEFAULT_TEMPORAL_BASIN_DATA_DIR_NAME
    basin_data_urls: dict[str, str] = {}
    basin_json_paths: dict[str, str] = {}
    for basin in BASIN_BUTTON_NAMES:
        basin_relative = (
            f"{DEFAULT_TEMPORAL_BASIN_DATA_DIR_NAME}/{_safe_basin_file_name(basin)}"
        )
        basin_path = _write_json(
            output_dir / basin_relative,
            _basin_payload(
                basin=basin,
                validation_year=validation_year,
                variable_payloads=variable_payloads,
            ),
        )
        basin_data_urls[basin] = _temporal_url(
            basin_relative, public_base_url=public_base_url
        )
        basin_json_paths[basin] = str(basin_path)
    basin_dir.mkdir(parents=True, exist_ok=True)

    basin_map_path = _write_json(
        output_dir / DEFAULT_TEMPORAL_BASIN_MAP_GEOJSON_NAME, _basin_map_payload()
    )
    default_variable = _default_variable_key(variables)
    default_basin = _default_basin_name()
    config = {
        "schema_version": 2,
        "title": "DepthDif Temporal Validation Error",
        "validation_year": int(validation_year),
        "default_variable": default_variable,
        "default_basin": default_basin,
        "available_variables": list(variables.keys()),
        "basins": [
            {"name": basin, "label": BASIN_DISPLAY_NAMES.get(basin, basin)}
            for basin in BASIN_BUTTON_NAMES
        ],
        "basin_map_geojson_url": _temporal_url(
            DEFAULT_TEMPORAL_BASIN_MAP_GEOJSON_NAME,
            public_base_url=public_base_url,
        ),
        "basin_data_urls": basin_data_urls,
        "variables": variables,
    }
    config_path = _write_json(
        output_dir / DEFAULT_TEMPORAL_CONFIG_NAME,
        config,
        indent=2,
    )
    run_summary = {
        "temporal_dashboard": {
            "schema_version": 2,
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "basin_map_geojson_path": str(basin_map_path),
            "basin_json_paths": basin_json_paths,
            "public_base_url": public_base_url,
            "default_variable": default_variable,
            "default_basin": default_basin,
            "validation_year": int(validation_year),
            "variables": variables,
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
        "basin_map_geojson_path": str(basin_map_path),
        "basin_json_paths": basin_json_paths,
        "variables": list(variables.keys()),
        "default_variable": default_variable,
        "default_basin": default_basin,
        "validation_year": int(validation_year),
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
        description="Aggregate weekly inference runs into compact temporal dashboard assets."
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
        "--validation-year",
        type=int,
        default=DEFAULT_TEMPORAL_VALIDATION_YEAR,
        help="Validation year represented by the temporal dashboard.",
    )
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; schema v2 no longer stores grid cells.",
    )
    parser.add_argument(
        "--top-cell-count",
        type=int,
        default=None,
        help="Accepted for CLI compatibility; schema v2 no longer stores top cells.",
    )
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
        validation_year=int(args.validation_year),
    )
    print(f"Wrote temporal dashboard assets to: {result['output_dir']}")
    print(f"- config: {result['config_path']}")
    print(f"- basin map: {result['basin_map_geojson_path']}")
    for basin, basin_path in result["basin_json_paths"].items():
        print(f"- {basin}: {basin_path}")
    if result.get("upload_requested"):
        print(f"- upload: {result.get('upload_message')}")


if __name__ == "__main__":
    main()
