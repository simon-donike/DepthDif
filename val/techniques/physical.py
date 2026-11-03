"""Physically meaningful vertical structure diagnostics.

The module deliberately computes nonlinear diagnostics for every ensemble member
before scoring them.  Incomplete profiles are recorded as censored diagnostics;
they are never silently extrapolated to a requested layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gsw
except ImportError:  # pragma: no cover - optional scientific dependency
    gsw = None

from val.paper_common import save_figure, save_table


def _crps(observation: float, members: np.ndarray) -> float:
    """Return empirical CRPS for one finite ensemble."""
    values = np.asarray(members, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size or not np.isfinite(observation):
        return np.nan
    return float(
        np.abs(values - observation).mean()
        - np.abs(values[:, None] - values[None, :]).mean() / 2.0
    )


def _crossing(depth: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Interpolate the first crossing, returning NaN when it is unresolved."""
    valid = np.isfinite(depth) & np.isfinite(values)
    depth, values = depth[valid], values[valid]
    if depth.size < 2:
        return np.nan
    order = np.argsort(depth)
    depth, values = depth[order], values[order]
    delta = values - threshold
    for index in range(len(depth) - 1):
        if delta[index] == 0:
            return float(depth[index])
        if delta[index] * delta[index + 1] <= 0 and delta[index] != delta[index + 1]:
            return float(
                depth[index]
                + (threshold - values[index])
                * (depth[index + 1] - depth[index])
                / (values[index + 1] - values[index])
            )
    return np.nan


def _profile_diagnostics(
    depth: np.ndarray, values: np.ndarray, options: dict[str, Any]
) -> dict[str, float]:
    """Compute frozen temperature structure quantities for one profile/member."""
    result: dict[str, float] = {}
    order = np.argsort(depth)
    depth, values = np.asarray(depth)[order], np.asarray(values)[order]
    finite = np.isfinite(depth) & np.isfinite(values)
    depth, values = depth[finite], values[finite]
    if depth.size < 2:
        return result
    low, high = options.get("thermocline_range_m", (10.0, 300.0))
    in_range = (depth >= low) & (depth <= high)
    if in_range.sum() >= 2:
        gradient = np.gradient(values[in_range], depth[in_range])
        result["max_gradient"] = float(np.max(np.abs(gradient)))
        index = int(np.argmax(np.abs(gradient)))
        result["thermocline_depth"] = float(depth[in_range][index])
    reference_depth = float(options.get("thermal_reference_depth_m", 10.0))
    reference_value = (
        float(np.interp(reference_depth, depth, values))
        if depth[0] <= reference_depth <= depth[-1]
        else np.nan
    )
    # A reference outside the observed/modelled water column is censored;
    # np.interp must never be used to invent a shallow endpoint.
    result["thermal_layer_depth"] = np.nan
    departure = float(options.get("thermal_departure_deg_c", 0.2))
    for index in range(len(depth) - 1):
        if depth[index + 1] <= reference_depth:
            continue
        left = abs(values[index] - reference_value) - departure
        right = abs(values[index + 1] - reference_value) - departure
        if left < 0 <= right and right != left:
            result["thermal_layer_depth"] = float(
                depth[index]
                + (-left) * (depth[index + 1] - depth[index]) / (right - left)
            )
            break
    for threshold in options.get("isotherms_deg_c", (5.0, 10.0, 15.0, 20.0)):
        result[f"isotherm_{threshold:g}c_depth"] = _crossing(
            depth, values, float(threshold)
        )
    for lower, upper in options.get("layers_m", ((10.0, 200.0), (10.0, 700.0))):
        key = f"layer_mean_{int(lower)}_{int(upper)}m"
        layer_mask = (depth >= lower) & (depth <= upper)
        internal_gaps = (
            np.diff(depth[layer_mask]) if layer_mask.sum() > 1 else np.array([])
        )
        if (
            depth[0] > lower
            or depth[-1] < upper
            or (
                internal_gaps.size
                and np.max(internal_gaps) > options.get("max_model_gap_m", 100.0)
            )
        ):
            result[key] = np.nan
            continue
        grid = np.unique(np.r_[lower, depth[(depth > lower) & (depth < upper)], upper])
        result[key] = float(
            np.trapezoid(np.interp(grid, depth, values), grid) / (upper - lower)
        )
    return result


def _joint_diagnostics(
    depth: np.ndarray,
    temperature: np.ndarray,
    salinity: np.ndarray,
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> dict[str, float]:
    """Compute TEOS-10 density structure when optional gsw is installed."""
    if gsw is None:
        return {}
    depth = np.asarray(depth, float)
    if depth.size < 2 or depth[0] > 10.0 or depth[-1] < 10.0:
        return {
            "density_mld_0.03kgm3": np.nan,
            "pycnocline_depth": np.nan,
            "stability_max_kgm4": np.nan,
            "freshwater_content_10_200m": np.nan,
            "barrier_layer_depth": np.nan,
        }
    pressure = gsw.p_from_z(-depth, latitude)
    absolute = gsw.SA_from_SP(
        np.asarray(salinity, float), pressure, longitude, latitude
    )
    # The model target is potential temperature, so use the TEOS-10 potential
    # temperature conversion instead of treating it as in-situ temperature.
    conservative = gsw.CT_from_pt(absolute, np.asarray(temperature, float))
    sigma = gsw.sigma0(absolute, conservative)
    reference = float(np.interp(10.0, depth, sigma))
    departure = sigma - reference
    mld = np.nan
    crossing = np.flatnonzero((depth >= 10.0) & (departure >= 0.03))
    if crossing.size:
        i = int(crossing[0])
        if i and departure[i] != departure[i - 1]:
            mld = float(
                depth[i - 1]
                + (0.03 - departure[i - 1])
                * (depth[i] - depth[i - 1])
                / (departure[i] - departure[i - 1])
            )
        else:
            mld = float(depth[i])
    gradient = np.gradient(sigma, depth)
    thermal = _profile_diagnostics(depth, temperature, {})
    result = {
        "density_mld_0.03kgm3": mld,
        "pycnocline_depth": float(depth[np.nanargmax(np.abs(gradient))]),
        "stability_max_kgm4": float(np.nanmax(gradient)),
        "stability_violation_fraction": float(np.mean(gradient < 0.0)),
        "barrier_layer_depth": (
            float(max(0.0, thermal.get("thermal_layer_depth", np.nan) - mld))
            if np.isfinite(mld)
            and np.isfinite(thermal.get("thermal_layer_depth", np.nan))
            else np.nan
        ),
    }
    if (
        depth[0] <= 10.0
        and depth[-1] >= 200.0
        and np.max(np.diff(depth[depth <= 200.0])) <= 100.0
    ):
        grid = np.unique(np.r_[10.0, depth[(depth > 10.0) & (depth < 200.0)], 200.0])
        result["freshwater_content_10_200m"] = float(
            np.trapezoid(
                (34.8 - np.interp(grid, depth, salinity)) / 34.8,
                grid,
            )
        )
    else:
        result["freshwater_content_10_200m"] = np.nan
    return result


def _table(frame: pd.DataFrame, path: Path) -> list[str]:
    """Write through the shared publication exporter."""
    return save_table(frame, path.parent, path.stem)


def _get(context: Any, name: str, default: Any = None) -> Any:
    """Read a field from either the shared context dataclass or a mapping."""
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run member-wise physical diagnostics for all available variables."""
    options = dict(options or {})
    evaluation = _get(context, "evaluation")
    points = (
        _get(evaluation, "points")
        if evaluation is not None
        else _get(context, "points")
    )
    members = (
        _get(evaluation, "members", {})
        if evaluation is not None
        else _get(context, "members", {})
    )
    output_dir = Path(output_dir)
    if points is None or not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No matched points available.",
            "artifacts": [],
            "details": {},
        }
    # Bundle arrays are point aligned; reset labels so filtered source frames
    # cannot accidentally be interpreted as positional array offsets.
    points = points.reset_index(drop=True)
    rows, reasons = [], []

    def reason_summary(values):
        counts = {}
        for value in values:
            key = str(value).split(":")[0:2]
            label = ":".join(key)
            counts[label] = counts.get(label, 0) + 1
        return {"counts": counts, "examples": values[:10]}

    variables_present = set(points.variable.dropna().unique())
    if not {"temperature", "salinity"}.issubset(variables_present):
        reasons.append(
            "joint density/pycnocline/freshwater diagnostics blocked: paired T/S observations are unavailable"
        )
    elif gsw is None:
        reasons.append(
            "joint density/pycnocline/freshwater diagnostics blocked: optional gsw is unavailable"
        )
    method_items = list(members.items()) if isinstance(members, dict) else []
    # Accept the old direct variable mapping only for backwards-compatible
    # unit tests; production bundles use method names and a variable column.
    if method_items and {key for key, _ in method_items}.issubset(
        {"temperature", "salinity"}
    ):
        method_items = [("residual", np.asarray(method_items[0][1]))]
    for method, array in method_items:
        for variable, variable_frame in points.groupby("variable", sort=True):
            if array is None:
                reasons.append(f"{variable}: no ensemble members")
                continue
            array = np.asarray(array, dtype=float)
            indices = variable_frame.index.to_numpy()
            if array.shape[0] != len(points):
                if array.shape[0] != len(variable_frame):
                    reasons.append(
                        f"{variable}: member array is not aligned with points"
                    )
                    continue
                values = array
            else:
                values = array[indices]
            for profile_id, frame in variable_frame.groupby("profile_id", sort=False):
                local = frame.index.to_numpy()
                order = np.argsort(frame.depth_m.to_numpy(float))
                depths = frame.depth_m.to_numpy(float)[order]
                observation = frame.observation.to_numpy(float)[order]
                # Preserve the row-to-point identity even when a filtered frame has
                # non-contiguous indices from the shared all-variable table.
                row_for_point = {int(point): row for row, point in enumerate(indices)}
                model_values = values[[row_for_point[int(point)] for point in local]]
                model_values = model_values[order]
                diagnostic_options = (
                    options
                    if variable == "temperature"
                    else {
                        **options,
                        "isotherms_deg_c": (),
                        "thermal_departure_deg_c": 1e9,
                    }
                )
                diagnostics = [
                    _profile_diagnostics(
                        depths, model_values[:, member], diagnostic_options
                    )
                    for member in range(model_values.shape[1])
                ]
                if variable == "salinity":
                    for item in diagnostics:
                        item.pop("thermal_layer_depth", None)
                        if "thermocline_depth" in item:
                            item["halocline_depth"] = item.pop("thermocline_depth")
                observed_diag = _profile_diagnostics(
                    depths, observation, diagnostic_options
                )
                if variable == "salinity":
                    observed_diag.pop("thermal_layer_depth", None)
                    if "thermocline_depth" in observed_diag:
                        observed_diag["halocline_depth"] = observed_diag.pop(
                            "thermocline_depth"
                        )
                for metric in sorted(observed_diag):
                    ensemble = np.array(
                        [item.get(metric, np.nan) for item in diagnostics], dtype=float
                    )
                    if (
                        not np.isfinite(observed_diag[metric])
                        or not np.isfinite(ensemble).all()
                    ):
                        reasons.append(f"{variable}:{metric}:incomplete:{profile_id}")
                        continue
                    rows.append(
                        {
                            "method": method,
                            "variable": variable,
                            "profile_id": profile_id,
                            "metric": metric,
                            "observation": observed_diag[metric],
                            "model_mean": np.nanmean(ensemble),
                            "model_error": np.nanmean(ensemble) - observed_diag[metric],
                            "model_crps": _crps(observed_diag[metric], ensemble),
                            "n_members": int(np.isfinite(ensemble).sum()),
                        }
                    )
        # Joint diagnostics are derived member by member from paired T/S
        # profiles; no density or freshwater quantity is emitted for T-only data.
        if {"temperature", "salinity"}.issubset(
            set(points.variable.unique())
        ) and gsw is not None:
            profiles = set(points[points.variable == "temperature"].profile_id) & set(
                points[points.variable == "salinity"].profile_id
            )
            for profile_id in profiles:
                tframe = points[
                    (points.profile_id == profile_id)
                    & (points.variable == "temperature")
                ].sort_values("depth_m")
                sframe = points[
                    (points.profile_id == profile_id) & (points.variable == "salinity")
                ].sort_values("depth_m")
                if not np.array_equal(
                    tframe.depth_m.to_numpy(), sframe.depth_m.to_numpy()
                ):
                    reasons.append(f"joint:unpaired_depths:{profile_id}")
                    continue
                tvalues = array[tframe.index.to_numpy()]
                svalues = array[sframe.index.to_numpy()]
                lat = float(tframe.latitude.iloc[0]) if "latitude" in tframe else 0.0
                lon = float(tframe.longitude.iloc[0]) if "longitude" in tframe else 0.0
                observed = _joint_diagnostics(
                    tframe.depth_m.to_numpy(float),
                    tframe.observation.to_numpy(float),
                    sframe.observation.to_numpy(float),
                    lat,
                    lon,
                )
                member_diagnostics = [
                    _joint_diagnostics(
                        tframe.depth_m.to_numpy(float),
                        tvalues[:, i],
                        svalues[:, i],
                        lat,
                        lon,
                    )
                    for i in range(tvalues.shape[1])
                ]
                for metric, truth in observed.items():
                    ensemble = np.asarray(
                        [item.get(metric, np.nan) for item in member_diagnostics], float
                    )
                    if not np.isfinite(truth) or not np.isfinite(ensemble).all():
                        reasons.append(f"joint:{metric}:incomplete:{profile_id}")
                        continue
                    rows.append(
                        {
                            "method": method,
                            "variable": "joint",
                            "profile_id": profile_id,
                            "metric": metric,
                            "observation": truth,
                            "model_mean": np.nanmean(ensemble),
                            "model_error": np.nanmean(ensemble) - truth,
                            "model_crps": _crps(truth, ensemble),
                            "n_members": int(np.isfinite(ensemble).sum()),
                        }
                    )
    # Deterministic references are evaluated once, independently of the
    # available stochastic method arrays.
    for variable, variable_frame in points.groupby("variable", sort=True):
        for profile_id, frame in variable_frame.groupby("profile_id", sort=False):
            order = np.argsort(frame.depth_m.to_numpy(float))
            depths = frame.depth_m.to_numpy(float)[order]
            observed_diag = _profile_diagnostics(
                depths, frame.observation.to_numpy(float)[order], options
            )
            for reference in ("glorys", "climatology"):
                if reference not in frame:
                    reasons.append(f"{reference}:missing_column")
                    continue
                diagnostic = _profile_diagnostics(
                    depths, frame[reference].to_numpy(float)[order], options
                )
                for metric, observed_value in observed_diag.items():
                    if (
                        metric in diagnostic
                        and np.isfinite(observed_value)
                        and np.isfinite(diagnostic[metric])
                    ):
                        rows.append(
                            {
                                "method": reference,
                                "variable": variable,
                                "profile_id": profile_id,
                                "metric": metric,
                                "observation": observed_value,
                                "model_mean": diagnostic[metric],
                                "model_error": diagnostic[metric] - observed_value,
                                "model_crps": abs(diagnostic[metric] - observed_value),
                                "n_members": 1,
                            }
                        )
    if {"temperature", "salinity"}.issubset(
        set(points.variable.unique())
    ) and gsw is not None:
        profiles = set(points[points.variable == "temperature"].profile_id) & set(
            points[points.variable == "salinity"].profile_id
        )
        for profile_id in profiles:
            tf = points[
                (points.profile_id == profile_id) & (points.variable == "temperature")
            ].sort_values("depth_m")
            sf = points[
                (points.profile_id == profile_id) & (points.variable == "salinity")
            ].sort_values("depth_m")
            if not np.array_equal(tf.depth_m.to_numpy(), sf.depth_m.to_numpy()):
                continue
            depth = tf.depth_m.to_numpy(float)
            latitude = float(tf.latitude.iloc[0]) if "latitude" in tf else 0.0
            longitude = float(tf.longitude.iloc[0]) if "longitude" in tf else 0.0
            truth = _joint_diagnostics(
                depth,
                tf.observation.to_numpy(float),
                sf.observation.to_numpy(float),
                latitude,
                longitude,
            )
            for reference in ("glorys", "climatology"):
                if reference not in tf or reference not in sf:
                    continue
                predicted = _joint_diagnostics(
                    depth,
                    tf[reference].to_numpy(float),
                    sf[reference].to_numpy(float),
                    latitude,
                    longitude,
                )
                for metric, value in truth.items():
                    if np.isfinite(value) and np.isfinite(
                        predicted.get(metric, np.nan)
                    ):
                        rows.append(
                            {
                                "method": reference,
                                "variable": "joint",
                                "profile_id": profile_id,
                                "metric": metric,
                                "observation": value,
                                "model_mean": predicted[metric],
                                "model_error": predicted[metric] - value,
                                "model_crps": abs(predicted[metric] - value),
                                "n_members": 1,
                            }
                        )
    if not rows:
        return {
            "status": "partial",
            "summary": "No complete vertical diagnostics were available.",
            "artifacts": [],
            "details": {"reasons": reason_summary(reasons)},
        }
    result = pd.DataFrame(rows)
    if "method" not in result:
        result["method"] = "model"
    else:
        result["method"] = result["method"].fillna("model")
    summary = (
        result.groupby(["variable", "metric", "method"], dropna=False)
        .agg(
            n_profiles=("profile_id", "nunique"),
            mean_abs_error=("model_error", lambda x: np.mean(np.abs(x))),
            mean_crps=("model_crps", "mean"),
            mean_error=("model_error", "mean"),
        )
        .reset_index()
    )
    artifacts = _table(result, output_dir / "physical_member_diagnostics")
    artifacts += _table(summary, output_dir / "physical_summary")
    # A compact error figure is exported in all formats for paper assembly.
    import matplotlib.pyplot as plt

    metrics = list(
        result[["variable", "metric"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for page, start in enumerate(range(0, len(metrics), 6)):
        page_metrics = metrics[start : start + 6]
        figure, axes = plt.subplots(
            len(page_metrics), 1, figsize=(8, 3 * len(page_metrics)), squeeze=False
        )
        for axis, (variable, metric) in zip(axes[:, 0], page_metrics):
            subset = result[(result.metric == metric) & (result.variable == variable)]
            subset.groupby("method").model_error.apply(
                lambda values: np.mean(np.abs(values))
            ).plot.bar(ax=axis)
            axis.set_title(f"{variable}: {metric} (resolved profiles)")
            axis.set_ylabel("Mean absolute error (native units)")
            axis.set_xlabel("Method")
        artifacts += save_figure(
            figure, output_dir, f"physical_error_summary_{page + 1}"
        )
    return {
        "status": "complete" if not reasons else "partial",
        "summary": f"Computed {len(summary)} physical diagnostic groups.",
        "artifacts": artifacts,
        "details": {
            "n_profiles": int(result.profile_id.nunique()),
            "reasons": reason_summary(reasons),
            "crossing_score_scope": "Crossing-depth scores require the observation and every member to resolve the crossing. Counts differ by method; these conditional diagnostics alone do not establish uncensored distribution superiority.",
        },
    }
