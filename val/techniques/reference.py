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
import hashlib


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
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with xr.open_dataset(path, decode_times=True) as dataset:
            records.append(
                {
                    "path": str(path),
                    "variables": sorted(str(name) for name in dataset.data_vars),
                    "coordinates": sorted(str(name) for name in dataset.coords),
                    "dims": {
                        str(key): int(value) for key, value in dataset.sizes.items()
                    },
                    "time_start": (
                        str(dataset.time.values[0]) if "time" in dataset else None
                    ),
                    "time_end": (
                        str(dataset.time.values[-1]) if "time" in dataset else None
                    ),
                    "sha256": digest,
                }
            )
    return {"files": records}


def _coord_name(dataset: Any, candidates: tuple[str, ...]) -> str | None:
    """Find a coordinate by common GLORYS naming conventions."""
    names = set(dataset.coords) | set(dataset.dims)
    return next((name for name in candidates if name in names), None)


def _sample_native(
    points: pd.DataFrame, paths: list[Path], options: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                raise ValueError(
                    f"native file lacks time/lat/lon/depth coordinates: {path}"
                )
            times = pd.to_datetime(dataset[time_name].values, utc=True)
            bounds_name = dataset[time_name].attrs.get("bounds")
            if bounds_name not in dataset:
                bounds_name = _coord_name(dataset, ("time_bnds", "time_bounds"))
            time_bounds = None
            if bounds_name in dataset and dataset[bounds_name].ndim == 2:
                # Keep bounds timezone-naive as numpy datetime64 values.  Calling
                # astype on a timezone-aware DatetimeIndex fails on recent pandas.
                time_bounds = np.asarray(
                    pd.to_datetime(dataset[bounds_name].values.reshape(-1), utc=True),
                    dtype="datetime64[ns]",
                ).reshape(dataset[bounds_name].shape)
            latitude_values = np.asarray(dataset[lat_name].values, dtype=float).reshape(
                -1
            )
            longitude_values = np.asarray(
                dataset[lon_name].values, dtype=float
            ).reshape(-1)

            def spatial_value(
                field: Any, latitude: float, longitude: float
            ) -> tuple[float, str, float]:
                """Sample nearest or valid bilinear, rejecting distant/dry cells."""
                method = str(options.get("native_spatial_operator", "nearest")).lower()
                max_distance = float(
                    options.get("native_max_spatial_distance_deg", 0.15)
                )
                lat_index = int(np.argmin(np.abs(latitude_values - latitude)))
                # GLORYS commonly stores longitudes in [0, 360), while profiles
                # are often expressed in [-180, 180].
                wrapped_longitude = (
                    longitude % 360.0 if longitude_values.min() >= 0 else longitude
                )
                lon_index = int(np.argmin(np.abs(longitude_values - wrapped_longitude)))
                nearest_distance = float(
                    np.hypot(
                        latitude_values[lat_index] - latitude,
                        ((longitude_values[lon_index] - wrapped_longitude + 180) % 360)
                        - 180,
                    )
                )
                if nearest_distance > max_distance:
                    return np.nan, "spatial_distance_exceeded", nearest_distance
                if (
                    method != "valid_bilinear"
                    or latitude_values.size < 2
                    or longitude_values.size < 2
                ):
                    try:
                        value = float(
                            np.asarray(
                                field.isel(
                                    {lat_name: lat_index, lon_name: lon_index}
                                ).values
                            ).squeeze()
                        )
                    except (KeyError, IndexError, ValueError):
                        value = np.nan
                    return value, "nearest", nearest_distance
                lat_order = np.argsort(latitude_values)
                lon_order = np.argsort(longitude_values)
                lat_sorted, lon_sorted = (
                    latitude_values[lat_order],
                    longitude_values[lon_order],
                )
                lat_hi, lon_hi = int(np.searchsorted(lat_sorted, latitude)), int(
                    np.searchsorted(lon_sorted, longitude)
                )
                lat_lo, lon_lo = lat_hi - 1, lon_hi - 1
                if (
                    min(lat_lo, lon_lo) < 0
                    or lat_hi >= len(lat_sorted)
                    or lon_hi >= len(lon_sorted)
                ):
                    return np.nan, "outside_bilinear_cell", nearest_distance
                corners = []
                weights = []
                lat_weight = (latitude - lat_sorted[lat_lo]) / (
                    lat_sorted[lat_hi] - lat_sorted[lat_lo]
                )
                lon_weight = (longitude - lon_sorted[lon_lo]) / (
                    lon_sorted[lon_hi] - lon_sorted[lon_lo]
                )
                for lat_position, lat_weight_value in (
                    (lat_lo, 1 - lat_weight),
                    (lat_hi, lat_weight),
                ):
                    for lon_position, lon_weight_value in (
                        (lon_lo, 1 - lon_weight),
                        (lon_hi, lon_weight),
                    ):
                        value = float(
                            np.asarray(
                                field.isel(
                                    {
                                        lat_name: int(lat_order[lat_position]),
                                        lon_name: int(lon_order[lon_position]),
                                    }
                                ).values
                            ).squeeze()
                        )
                        corners.append(value)
                        weights.append(lat_weight_value * lon_weight_value)
                if not np.isfinite(corners).all():
                    return np.nan, "dry_or_invalid_bilinear_corner", nearest_distance
                return (
                    float(np.dot(corners, weights)),
                    "valid_bilinear",
                    nearest_distance,
                )

            for index, row in points.reset_index(drop=True).iterrows():
                variable = str(row.variable).lower()
                data_name = variable_names.get(variable)
                if data_name not in dataset:
                    audit_rows.append(
                        {
                            "point_id": row.get("point_id", index),
                            "reason": "variable_missing",
                            "path": str(path),
                        }
                    )
                    continue
                target = pd.to_datetime(
                    row.get("target_time", row.get("observation_time")), utc=True
                )
                deltas = np.abs((times - target).asi8 / 86_400_000_000_000)
                time_index = int(np.argmin(deltas))
                time_delta = float(deltas[time_index])
                if time_delta > float(options.get("native_time_tolerance_days", 3.5)):
                    audit_rows.append(
                        {
                            "point_id": row.get("point_id", index),
                            "reason": "time_outside_tolerance",
                            "path": str(path),
                            "time_delta_days": time_delta,
                        }
                    )
                    continue
                field = dataset[data_name].isel({time_name: time_index})
                try:
                    depth_value = float(row.depth_m)
                    depth_values = np.asarray(
                        dataset[depth_name].values, dtype=float
                    ).reshape(-1)
                    exact = np.flatnonzero(
                        np.isclose(depth_values, depth_value, rtol=0.0, atol=1e-8)
                    )
                    selected = (
                        field.isel({depth_name: int(exact[0])})
                        if exact.size
                        else field.interp(
                            {depth_name: depth_value}, kwargs={"fill_value": np.nan}
                        )
                    )
                    value, spatial_operator, spatial_distance = spatial_value(
                        selected, float(row.latitude), float(row.longitude)
                    )
                except (KeyError, ValueError, IndexError):
                    value, spatial_operator, spatial_distance = (
                        np.nan,
                        "sampling_error",
                        np.nan,
                    )
                if not np.isfinite(value):
                    audit_rows.append(
                        {
                            "point_id": row.get("point_id", index),
                            "reason": (
                                spatial_operator
                                if spatial_operator != "sampling_error"
                                else "land_or_depth_gap"
                            ),
                            "path": str(path),
                            "time_delta_days": time_delta,
                            "spatial_distance_deg": spatial_distance,
                        }
                    )
                    continue
                bound_start = bound_end = None
                if time_bounds is not None:
                    bound_start, bound_end = time_bounds[time_index]
                value_rows.append(
                    {
                        "point_id": row.get("point_id", index),
                        "variable": variable,
                        "glorys_native": value,
                        "native_time_delta_days": time_delta,
                        "native_time_bound_start": bound_start,
                        "native_time_bound_end": bound_end,
                        "native_spatial_operator": spatial_operator,
                        "native_spatial_distance_deg": spatial_distance,
                        "native_path": str(path),
                    }
                )
                audit_rows.append(
                    {
                        "point_id": row.get("point_id", index),
                        "reason": "matched",
                        "path": str(path),
                        "time_delta_days": time_delta,
                        "time_bound_start": bound_start,
                        "time_bound_end": bound_end,
                        "spatial_operator": spatial_operator,
                        "spatial_distance_deg": spatial_distance,
                    }
                )
    value_columns = [
        "point_id",
        "variable",
        "glorys_native",
        "native_time_delta_days",
        "native_time_bound_start",
        "native_time_bound_end",
        "native_spatial_operator",
        "native_spatial_distance_deg",
        "native_path",
    ]
    audit_columns = [
        "point_id",
        "reason",
        "path",
        "time_delta_days",
        "time_bound_start",
        "time_bound_end",
        "spatial_operator",
        "spatial_distance_deg",
    ]
    values = pd.DataFrame(value_rows, columns=value_columns)
    if not values.empty:
        values["_time_sort"] = pd.to_numeric(
            values["native_time_delta_days"], errors="coerce"
        )
        values = (
            values.sort_values(["point_id", "variable", "_time_sort", "native_path"])
            .drop_duplicates(["point_id", "variable"], keep="first")
            .drop(columns="_time_sort")
        )
    return values, pd.DataFrame(audit_rows, columns=audit_columns)


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run precision, temporal-window, and optional native GLORYS audits."""
    options = dict(options or {})
    output_dir = Path(output_dir)
    evaluation = _get(context, "evaluation", context)
    points = _get(evaluation, "points")
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No verification points available.",
            "artifacts": [],
            "details": {},
        }
    artifacts: list[str] = []
    evaluation_metadata = _get(evaluation, "metadata", {}) or {}
    declared_temperature_kind = (
        evaluation_metadata.get(
            "temperature_kind_compared",
            evaluation_metadata.get("model_temperature_kind"),
        )
        if isinstance(evaluation_metadata, Mapping)
        else None
    )
    audit = pd.DataFrame(
        [
            {
                "reference": "packaged_glorys",
                "grid_deg": 0.1,
                "temporal_operator": "daily_mean_sampled_at_7_day_stride",
                "temperature_quantization_step": 0.1496,
                "temperature_rounding_bound": 0.0748,
                "salinity_quantization_step": 0.0394,
                "salinity_rounding_bound": 0.0197,
            },
            {
                "reference": "native_glorys12",
                "grid_deg": 1.0 / 12.0,
                "temporal_operator": "source_time_coordinate",
                "temperature_quantization_step": 0.0,
                "temperature_rounding_bound": 0.0,
                "salinity_quantization_step": 0.0,
                "salinity_rounding_bound": 0.0,
            },
        ]
    )
    artifacts += _table(audit, output_dir, "reference_precision_audit")
    times = pd.to_datetime(
        points.get("observation_time", points.get("target_time")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    target = pd.to_datetime(
        points.get("target_time", points.get("observation_time")),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    delta = (times - target).dt.total_seconds().abs() / 3600.0
    windows = pd.DataFrame(
        {
            "window": ["within_12h", "within_24h", "full_matching_window"],
            "hours": [12.0, 24.0, np.inf],
            "n_points": [(delta <= 12).sum(), (delta <= 24).sum(), delta.notna().sum()],
        }
    )
    artifacts += _table(windows, output_dir, "reference_time_windows")
    members = _get(evaluation, "members", {})
    residual = members.get("residual") if isinstance(members, Mapping) else None
    time_scores = []
    for label, hours in (
        ("within_12h", 12),
        ("within_24h", 24),
        ("full_matching_window", np.inf),
    ):
        for variable in points.variable.unique():
            selected = delta.le(hours) & points.variable.eq(variable)
            subset = points.loc[selected]
            if subset.empty:
                continue
            methods = {"packaged_glorys": subset.glorys.to_numpy(float)}
            if residual is not None:
                methods["depthdif_mean"] = np.asarray(residual)[
                    selected.to_numpy()
                ].mean(axis=1)
            for method, prediction in methods.items():
                error = prediction - subset.observation.to_numpy(float)
                profile = (
                    pd.DataFrame(
                        {
                            "profile_id": subset.profile_id,
                            "mae": np.abs(error),
                            "mse": error**2,
                            "bias": error,
                        }
                    )
                    .groupby("profile_id")
                    .mean()
                )
                time_scores.append(
                    {
                        "window": label,
                        "variable": variable,
                        "method": method,
                        "profiles": len(profile),
                        "points": len(subset),
                        "mae": profile.mae.mean(),
                        "rmse": np.sqrt(profile.mse.mean()),
                        "bias": profile.bias.mean(),
                        "reference_time": "packaged filename daily date; source bounds unverified",
                    }
                )
    if time_scores:
        artifacts += _table(
            pd.DataFrame(time_scores), output_dir, "packaged_time_window_scores"
        )
    paths = _paths(options)
    if not paths:
        blocked = pd.DataFrame(
            [
                {
                    "status": "partial",
                    "reason": "native GLORYS NetCDF path not configured; packaged audit remains reproducible",
                }
            ]
        )
        artifacts += _table(blocked, output_dir, "native_reference_status")
        return {
            "status": "partial",
            "summary": "Packaged precision and timing audits completed; native NetCDF is not configured.",
            "artifacts": artifacts,
            "details": {
                "native_available": False,
                "temporal_windows": windows.to_dict(orient="records"),
            },
        }
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        blocked = pd.DataFrame(
            [
                {
                    "status": "partial",
                    "reason": "native GLORYS file missing",
                    "path": path,
                }
                for path in missing
            ]
        )
        artifacts += _table(blocked, output_dir, "native_reference_status")
        return {
            "status": "partial",
            "summary": "Native GLORYS was configured but files are unavailable.",
            "artifacts": artifacts,
            "details": {"native_available": False, "missing": missing},
        }
    try:
        metadata = _native_metadata(paths)
        native, matching = _sample_native(points, paths, options)
    except (ImportError, OSError, ValueError) as error:
        failure = pd.DataFrame([{"status": "partial", "reason": str(error)}])
        artifacts += _table(failure, output_dir, "native_reference_status")
        return {
            "status": "partial",
            "summary": "Native GLORYS audit could not be sampled safely.",
            "artifacts": artifacts,
            "details": {"native_available": False, "error": str(error)},
        }
    artifacts += _table(matching, output_dir, "native_reference_matches")
    merged = points.reset_index(drop=True).merge(
        native, on=["point_id", "variable"], how="left"
    )
    # Keep partial native support explicit; regional comparisons select the same
    # finite cases before computing paired errors and independent-cluster CIs.
    points["baseline_glorys_native"] = merged.glorys_native.to_numpy(float)
    artifacts += _table(merged, output_dir, "native_reference_values")
    # Keep the native comparison on exactly the rows that were sampled by both
    # references.  This avoids presenting different support as an improvement.
    common = merged[merged.glorys_native.notna()].copy()
    comparison_rows = []
    members = _get(evaluation, "members", {})
    model_kind = declared_temperature_kind or "potential"
    common["time_window"] = np.where(
        common.native_time_delta_days * 24 <= 12,
        "within_12h",
        np.where(
            common.native_time_delta_days * 24 <= 24,
            "within_24h",
            "full_matching_window",
        ),
    )
    for variable, frame in common.groupby("variable", sort=True):
        # Residual inference may expose the ensemble under a generic key.  The
        # comparison is still labelled by the point variable, never by that key.
        array = None
        if isinstance(members, Mapping):
            candidate = members.get(variable, members.get("residual"))
            if candidate is not None:
                candidate = np.asarray(candidate, dtype=float)
                if candidate.ndim == 2 and candidate.shape[0] == len(points):
                    array = candidate
        for window_name, window_frame in frame.groupby("time_window", sort=False):
            truth = window_frame.observation.to_numpy(float)
            for method, values in (
                ("packaged_glorys", window_frame.glorys.to_numpy(float)),
                ("native_glorys", window_frame.glorys_native.to_numpy(float)),
            ):
                comparison_rows.append(
                    {
                        "variable": variable,
                        "time_window": window_name,
                        "method": method,
                        "n_points": len(window_frame),
                        "mae": float(np.mean(np.abs(values - truth))),
                        "rmse": float(np.sqrt(np.mean((values - truth) ** 2))),
                        "bias": float(np.mean(values - truth)),
                    }
                )
            if array is not None:
                positions = window_frame.index.to_numpy()
                prediction = array[positions].mean(axis=1)
                comparison_rows.append(
                    {
                        "variable": variable,
                        "time_window": window_name,
                        "method": "depthdif_mean",
                        "n_points": len(window_frame),
                        "mae": float(np.mean(np.abs(prediction - truth))),
                        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                        "bias": float(np.mean(prediction - truth)),
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        artifacts += _table(comparison, output_dir, "native_vs_packaged_scores")
        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(6.5, 4.0))
            for variable, frame in comparison.groupby("variable", sort=True):
                axis.plot(frame.method, frame.rmse, marker="o", label=variable)
            axis.set_ylabel("RMSE on common native support")
            axis.set_title("Native versus packaged GLORYS")
            axis.legend()
            figure.tight_layout()
            for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
                path = output_dir / f"native_vs_packaged_scores.{suffix}"
                figure.savefig(path, **kwargs)
                artifacts.append(str(path))
            plt.close(figure)
        except ImportError:
            pass
    status = "complete" if merged.glorys_native.notna().any() else "partial"
    if (
        declared_temperature_kind
        and str(declared_temperature_kind).lower() != "potential"
        and (common.variable == "temperature").any()
    ):
        status = "partial"
    return {
        "status": status,
        "summary": f"Reference audit completed with {int(merged.glorys_native.notna().sum())} native matches.",
        "artifacts": artifacts,
        "details": {
            "native_available": True,
            "metadata": metadata,
            "n_native_matches": int(merged.glorys_native.notna().sum()),
            "model_temperature_kind": model_kind,
            "common_support_scores": comparison.to_dict(orient="records"),
        },
    }
