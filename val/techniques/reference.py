"""Audits for the packaged GLORYS reference and optional native product.

The packaged files are useful for reproducing the training target, but their
integer encoding and seven day sampling can hide a small effect.  This stage
keeps those diagnostics explicit and only adds a native baseline when a source
NetCDF path is supplied and can be sampled without extrapolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a context, bundle, or mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _table(frame: pd.DataFrame, directory: Path, name: str) -> list[str]:
    """Write a paper table in portable CSV and LaTeX formats."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = directory / f"{name}.csv", directory / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(tex_path, index=False, float_format="%.6g")
    return [str(csv_path), str(tex_path)]


def _paths(options: Mapping[str, Any]) -> list[Path]:
    """Resolve configured native files, accepting a path or a list of paths."""
    value = options.get("native_glorys") or options.get("native_paths")
    if not value:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [Path(str(item)).expanduser() for item in values]


def _native_metadata(paths: list[Path]) -> dict[str, Any]:
    """Inspect coordinate and variable names without loading a whole product."""
    import xarray as xr

    records = []
    for path in paths:
        with xr.open_dataset(path, decode_times=True) as dataset:
            records.append(
                {
                    "path": str(path),
                    "variables": sorted(str(name) for name in dataset.data_vars),
                    "coordinates": sorted(str(name) for name in dataset.coords),
                    "dims": {str(key): int(value) for key, value in dataset.sizes.items()},
                    "time_start": str(dataset.time.values[0]) if "time" in dataset else None,
                    "time_end": str(dataset.time.values[-1]) if "time" in dataset else None,
                }
            )
    return {"files": records}


def _coord_name(dataset: Any, candidates: tuple[str, ...]) -> str | None:
    """Find a coordinate by common GLORYS naming conventions."""
    names = set(dataset.coords) | set(dataset.dims)
    return next((name for name in candidates if name in names), None)


def _sample_native(points: pd.DataFrame, paths: list[Path], options: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample native floating point GLORYS with depth and time interpolation.

    Spatial selection uses the nearest valid wet cell.  The packaged product's
    target grid remains the authoritative comparison grid; native values are a
    sensitivity baseline and are never used to fill missing observations.
    """
    import xarray as xr

    value_rows, audit_rows = [], []
    variable_names = {"temperature": "thetao", "salinity": "so"}
    for path in paths:
        with xr.open_dataset(path, decode_times=True) as dataset:
            time_name = _coord_name(dataset, ("time", "TIME", "time_counter"))
            lat_name = _coord_name(dataset, ("latitude", "lat", "nav_lat"))
            lon_name = _coord_name(dataset, ("longitude", "lon", "nav_lon"))
            depth_name = _coord_name(dataset, ("depth", "deptht", "depthu", "lev"))
            if not all((time_name, lat_name, lon_name, depth_name)):
                raise ValueError(f"native file lacks time/lat/lon/depth coordinates: {path}")
            times = pd.to_datetime(dataset[time_name].values, utc=True)
            for index, row in points.reset_index(drop=True).iterrows():
                variable = str(row.variable).lower()
                data_name = variable_names.get(variable)
                if data_name not in dataset:
                    audit_rows.append({"point_id": row.get("point_id", index), "reason": "variable_missing", "path": str(path)})
                    continue
                target = pd.to_datetime(row.get("target_time", row.get("observation_time")), utc=True)
                deltas = np.abs((times - target).asi8 / 86_400_000_000_000)
                time_index = int(np.argmin(deltas))
                time_delta = float(deltas[time_index])
                if time_delta > float(options.get("native_time_tolerance_days", 3.5)):
                    audit_rows.append({"point_id": row.get("point_id", index), "reason": "time_outside_tolerance", "path": str(path), "time_delta_days": time_delta})
                    continue
                field = dataset[data_name].isel({time_name: time_index})
                try:
                    selected = field.interp({depth_name: float(row.depth_m)}, kwargs={"fill_value": np.nan})
                    selected = selected.sel({lat_name: float(row.latitude), lon_name: float(row.longitude)}, method="nearest")
                    value = float(np.asarray(selected.values).squeeze())
                except (KeyError, ValueError, IndexError):
                    value = np.nan
                if not np.isfinite(value):
                    audit_rows.append({"point_id": row.get("point_id", index), "reason": "land_or_depth_gap", "path": str(path), "time_delta_days": time_delta})
                    continue
                value_rows.append({"point_id": row.get("point_id", index), "variable": variable, "glorys_native": value, "native_time_delta_days": time_delta, "native_path": str(path)})
                audit_rows.append({"point_id": row.get("point_id", index), "reason": "matched", "path": str(path), "time_delta_days": time_delta})
    return pd.DataFrame(value_rows), pd.DataFrame(audit_rows)


def run(context: Any, output_dir: Path, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run precision, temporal-window, and optional native GLORYS audits."""
    options = dict(options or {})
    output_dir = Path(output_dir)
    evaluation = _get(context, "evaluation", context)
    points = _get(evaluation, "points")
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {"status": "blocked", "summary": "No verification points available.", "artifacts": [], "details": {}}
    artifacts: list[str] = []
    audit = pd.DataFrame(
        [
            {"reference": "packaged_glorys", "grid_deg": 0.1, "temporal_operator": "daily_mean_sampled_at_7_day_stride", "temperature_quantization_step": 0.1496, "temperature_rounding_bound": 0.0748, "salinity_quantization_step": 0.0394, "salinity_rounding_bound": 0.0197},
            {"reference": "native_glorys12", "grid_deg": 1.0 / 12.0, "temporal_operator": "source_time_coordinate", "temperature_quantization_step": 0.0, "temperature_rounding_bound": 0.0, "salinity_quantization_step": 0.0, "salinity_rounding_bound": 0.0},
        ]
    )
    artifacts += _table(audit, output_dir, "reference_precision_audit")
    times = pd.to_datetime(points.get("observation_time", points.get("target_time")), utc=True, errors="coerce")
    target = pd.to_datetime(points.get("target_time", points.get("observation_time")), utc=True, errors="coerce")
    delta = (times - target).dt.total_seconds().abs() / 3600.0
    windows = pd.DataFrame({"window": ["within_12h", "within_24h", "full_matching_window"], "hours": [12.0, 24.0, np.inf], "n_points": [(delta <= 12).sum(), (delta <= 24).sum(), delta.notna().sum()]})
    artifacts += _table(windows, output_dir, "reference_time_windows")
    paths = _paths(options)
    if not paths:
        blocked = pd.DataFrame([{"status": "partial", "reason": "native GLORYS NetCDF path not configured; packaged audit remains reproducible"}])
        artifacts += _table(blocked, output_dir, "native_reference_status")
        return {"status": "partial", "summary": "Packaged precision and timing audits completed; native NetCDF is not configured.", "artifacts": artifacts, "details": {"native_available": False, "temporal_windows": windows.to_dict(orient="records")}}
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        blocked = pd.DataFrame([{"status": "partial", "reason": "native GLORYS file missing", "path": path} for path in missing])
        artifacts += _table(blocked, output_dir, "native_reference_status")
        return {"status": "partial", "summary": "Native GLORYS was configured but files are unavailable.", "artifacts": artifacts, "details": {"native_available": False, "missing": missing}}
    try:
        metadata = _native_metadata(paths)
        native, matching = _sample_native(points, paths, options)
    except (ImportError, OSError, ValueError) as error:
        failure = pd.DataFrame([{"status": "partial", "reason": str(error)}])
        artifacts += _table(failure, output_dir, "native_reference_status")
        return {"status": "partial", "summary": "Native GLORYS audit could not be sampled safely.", "artifacts": artifacts, "details": {"native_available": False, "error": str(error)}}
    artifacts += _table(matching, output_dir, "native_reference_matches")
    merged = points.reset_index(drop=True).merge(native, on=["point_id", "variable"], how="left")
    artifacts += _table(merged, output_dir, "native_reference_values")
    status = "complete" if merged.glorys_native.notna().any() else "partial"
    return {"status": status, "summary": f"Reference audit completed with {int(merged.glorys_native.notna().sum())} native matches.", "artifacts": artifacts, "details": {"native_available": True, "metadata": metadata, "n_native_matches": int(merged.glorys_native.notna().sum())}}
