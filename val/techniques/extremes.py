"""Tail probability validation with thresholds frozen on a calibration cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _get(context: Any, name: str, default: Any = None) -> Any:
    """Read a field from the shared Context or its evaluation Bundle."""
    bundle = getattr(context, "evaluation", None)
    if bundle is not None and name == "points":
        return getattr(bundle, "points", default)
    if bundle is not None and name == "members":
        return getattr(bundle, "members", default)
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def _bundle(context: Any, name: str) -> Any:
    """Return a calibration/evaluation Bundle from either context shape."""
    if isinstance(context, dict):
        return context.get(name)
    return getattr(context, name, None)


def _crps(values: np.ndarray, truth: float, fair: bool = True) -> float:
    """Return empirical CRPS with the finite ensemble fair correction."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(truth):
        return np.nan
    denominator = len(x) * (len(x) - 1 if fair and len(x) > 1 else len(x))
    return float(
        np.mean(np.abs(x - truth)) - np.sum(np.abs(x[:, None] - x)) / (2 * denominator)
    )


def _write_table(frame: pd.DataFrame, output: Path, name: str) -> list[str]:
    """Write a CSV and a compact LaTeX table."""
    output.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = output / f"{name}.csv", output / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(tex_path, index=False, float_format="%.5g")
    return [str(csv_path), str(tex_path)]


def _figure(frame: pd.DataFrame, output: Path) -> list[str]:
    """Write a reliability figure when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for variable, group in frame.groupby("variable", sort=True):
        ax.plot(
            group.forecast_probability,
            group.observed_frequency,
            "o-",
            label=variable,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Forecast probability",
        ylabel="Observed frequency",
    )
    if not frame.empty:
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = output / f"extreme_reliability.{suffix}"
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _thresholds(calibration: Any, quantile: float) -> dict[tuple[str, str, str], float]:
    """Fit anomaly thresholds from calibration observations only."""
    points = getattr(calibration, "points", None)
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {}
    values = points.observation.to_numpy(float) - points.climatology.to_numpy(float)
    work = points[["variable", "depth_m"]].copy()
    work["anomaly"] = values
    work["depth_bin"] = pd.cut(
        work.depth_m, bins=[-np.inf, 50, 200, 700, np.inf], right=False
    )
    # Preserve original point values while assigning each profile equal total
    # mass within each variable/depth bin.
    work["profile_id"] = points.profile_id.to_numpy()
    result = {}
    for (variable, depth_bin), group in work.groupby(
        ["variable", "depth_bin"], observed=True
    ):
        finite = group[np.isfinite(group.anomaly)].copy()
        if not finite.empty:
            counts = finite.groupby("profile_id").size()
            n_profiles = len(counts)
            finite["weight"] = finite.profile_id.map(
                lambda profile: 1.0 / (n_profiles * counts[profile])
            )
            for tail, values in (
                ("upper", finite.anomaly.to_numpy(float)),
                ("lower", -finite.anomaly.to_numpy(float)),
            ):
                result[(str(variable), str(depth_bin), tail)] = _weighted_quantile(
                    values, finite.weight.to_numpy(float), quantile
                )
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Return a deterministic weighted quantile from finite weighted values."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    order = np.argsort(values[finite], kind="mergesort")
    ordered_values, ordered_weights = values[finite][order], weights[finite][order]
    cumulative = np.cumsum(ordered_weights) / np.sum(ordered_weights)
    return float(ordered_values[np.searchsorted(cumulative, quantile, side="left")])


def _threshold_for(
    row: pd.Series,
    thresholds: dict[tuple[str, str, str], float],
    tail: str,
    default: float = np.nan,
) -> float:
    """Look up a frozen variable/depth threshold, falling back safely."""
    depth_bin = pd.cut(
        [float(row.depth_m)], bins=[-np.inf, 50, 200, 700, np.inf], right=False
    )[0]
    return float(thresholds.get((str(row.variable), str(depth_bin), tail), default))


def _calibration_tail_probability(
    calibration: Any, row: pd.Series, upper_threshold: float, lower_threshold: float
) -> float:
    """Estimate the climatology distribution tail probability in a fixed bin."""
    points = calibration.points
    depth_bin = pd.cut(
        [float(row.depth_m)], [-np.inf, 50, 200, 700, np.inf], right=False
    )[0]
    work = points[points.variable.astype(str) == str(row.variable)].copy()
    work["depth_bin"] = pd.cut(
        work.depth_m, [-np.inf, 50, 200, 700, np.inf], right=False
    )
    work = work[work.depth_bin == depth_bin]
    if work.empty:
        return np.nan
    signed = work.observation.to_numpy(float) - work.climatology.to_numpy(float)
    return float(np.mean((signed >= upper_threshold) | (signed <= -lower_threshold)))


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate tail probabilities, reliability, and fixed weighted tail CRPS."""
    options = dict(options or {})
    output_dir = Path(output_dir)
    points = _get(context, "points")
    members = _get(context, "members", {})
    calibration = _bundle(context, "calibration")
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No evaluation points.",
            "artifacts": [],
            "details": {},
        }
    if calibration is None or not isinstance(
        getattr(calibration, "points", None), pd.DataFrame
    ):
        return {
            "status": "blocked",
            "summary": "Extreme thresholds require a separate calibration cohort.",
            "artifacts": [],
            "details": {},
        }
    overlap = set(points.profile_id.astype(str)) & set(
        calibration.points.profile_id.astype(str)
    )
    if overlap and not options.get("allow_calibration_overlap", False):
        return {
            "status": "blocked",
            "summary": "Calibration and evaluation profiles overlap; thresholds are not frozen independently.",
            "artifacts": [],
            "details": {"overlap_profiles": sorted(overlap)},
        }
    quantiles = options.get("quantiles", [options.get("quantile", 0.9)])
    quantiles = [float(value) for value in quantiles]
    threshold_sets = {q: _thresholds(calibration, q) for q in quantiles}
    if not any(threshold_sets.values()):
        return {
            "status": "blocked",
            "summary": "Calibration has no finite anomaly thresholds.",
            "artifacts": [],
            "details": {},
        }
    unsupported = sorted(
        variable
        for variable in set(points.variable.astype(str))
        if not any(
            key[0] == variable
            for thresholds in threshold_sets.values()
            for key in thresholds
        )
    )
    if unsupported:
        return {
            "status": "blocked",
            "summary": "Calibration has no threshold support for evaluation variables.",
            "artifacts": [],
            "details": {"unsupported_variables": unsupported},
        }
    missing_bins = []
    for _, row in points.iterrows():
        depth_bin = str(
            pd.cut([float(row.depth_m)], [-np.inf, 50, 200, 700, np.inf], right=False)[
                0
            ]
        )
        for quantile, thresholds in threshold_sets.items():
            for tail in ("upper", "lower"):
                if (str(row.variable), depth_bin, tail) not in thresholds:
                    missing_bins.append((str(row.variable), depth_bin, quantile, tail))
    if missing_bins:
        return {
            "status": "blocked",
            "summary": "Calibration has no support for an evaluation variable/depth bin.",
            "artifacts": [],
            "details": {"missing_variable_depth_bins": missing_bins},
        }
    rows, reliability_rows = [], []
    points = points.reset_index(drop=True)
    variable_positions = {
        variable: np.flatnonzero(points.variable.to_numpy() == variable)
        for variable in points.variable.unique()
    }
    baseline_arrays = {}
    if isinstance(members, dict):
        for name in (
            "glorys_calibrated",
            "bias_corrected",
            "glorys_bias_corrected",
        ):
            if members.get(name) is not None:
                baseline_arrays[name] = np.asarray(members[name], float)
    for index, row in points.iterrows():
        variable = str(row.variable)
        array = None
        if isinstance(members, dict):
            # Inference persists one all-point matrix under ``residual``.
            array = members.get(variable, members.get("residual"))
        if array is None:
            continue
        array = np.asarray(array, float)
        # Also accept a variable-only matrix while preserving row alignment.
        if array.shape[0] == len(points):
            local = index
        else:
            matches = np.flatnonzero(variable_positions[variable] == index)
            local = int(matches[0]) if len(matches) else -1
        if local < 0 or local >= array.shape[0]:
            continue
        ensemble = array[local]
        anomaly = ensemble - float(row.climatology)
        observed_anomaly = float(row.observation) - float(row.climatology)
        uncertainty = row.get(
            "obs_uncertainty", options.get("observation_uncertainty", 0.0)
        )
        uncertainty = float(uncertainty) if np.isfinite(float(uncertainty)) else 0.0
        # Integrate the event indicator over the declared observation-error law;
        # this keeps the uncertainty sensitivity separate from the point truth.
        draws = int(options.get("observation_uncertainty_draws", 200))
        if uncertainty > 0 and draws > 0:
            rng = np.random.default_rng(int(options.get("seed", 20261103)) + index)
            observed_draws = float(row.observation) + rng.normal(0, uncertainty, draws)
            uncertain_draws = observed_draws - float(row.climatology)
        else:
            uncertain_draws = np.array([observed_anomaly])
        raw_anomaly = float(row.glorys) - float(row.climatology)
        baseline_values = {}
        for name, baseline in baseline_arrays.items():
            if baseline.shape[0] == len(points):
                baseline_row = baseline[index]
            else:
                baseline_row = baseline[local]
            baseline_values[name] = np.asarray(baseline_row, float).reshape(-1)
        for quantile, thresholds in threshold_sets.items():
            upper_threshold = _threshold_for(row, thresholds, "upper")
            lower_threshold = _threshold_for(row, thresholds, "lower")
            if not np.isfinite(upper_threshold) or not np.isfinite(lower_threshold):
                continue
            threshold = min(upper_threshold, lower_threshold)
            event = float(
                observed_anomaly >= upper_threshold
                or observed_anomaly <= -lower_threshold
            )
            probability = float(
                np.mean((anomaly >= upper_threshold) | (anomaly <= -lower_threshold))
            )
            warm_event = float(observed_anomaly >= upper_threshold)
            cold_event = float(observed_anomaly <= -lower_threshold)
            warm_probability = float(np.mean(anomaly >= upper_threshold))
            cold_probability = float(np.mean(anomaly <= -lower_threshold))
            uncertain_event = float(
                np.mean(
                    (uncertain_draws >= upper_threshold)
                    | (uncertain_draws <= -lower_threshold)
                )
            )
            climate_probability = _calibration_tail_probability(
                calibration, row, upper_threshold, lower_threshold
            )
            upper_truth = max(
                float(row.observation), float(row.climatology) + upper_threshold
            )
            lower_truth = min(
                float(row.observation), float(row.climatology) - lower_threshold
            )
            upper = np.maximum(ensemble, float(row.climatology) + upper_threshold)
            lower = np.minimum(ensemble, float(row.climatology) - lower_threshold)
            row_result = {
                "point_id": row.point_id,
                "profile_id": row.profile_id,
                "variable": variable,
                "quantile": quantile,
                "depth_bin": str(
                    pd.cut(
                        [float(row.depth_m)],
                        [-np.inf, 50, 200, 700, np.inf],
                        right=False,
                    )[0]
                ),
                "threshold": threshold,
                "threshold_upper": upper_threshold,
                "threshold_lower": lower_threshold,
                "event": event,
                "probability": probability,
                "warm_event": warm_event,
                "cold_event": cold_event,
                "warm_probability": warm_probability,
                "cold_probability": cold_probability,
                "raw_glorys_event": float(
                    raw_anomaly >= upper_threshold or raw_anomaly <= -lower_threshold
                ),
                "raw_glorys_warm_event": float(raw_anomaly >= upper_threshold),
                "raw_glorys_cold_event": float(raw_anomaly <= -lower_threshold),
                "climatology_probability": climate_probability,
                "observation_uncertainty": uncertainty,
                "event_probability_with_observation_error": uncertain_event,
                "tail_crps_upper": _crps(upper, upper_truth),
                "tail_crps_lower": _crps(lower, lower_truth),
                "tail_crps_upper_empirical": _crps(upper, upper_truth, fair=False),
                "tail_crps_lower_empirical": _crps(lower, lower_truth, fair=False),
                "n_members": len(ensemble),
            }
            for name, baseline in baseline_values.items():
                anomaly_values = baseline - float(row.climatology)
                prefix = name.replace("glorys_", "glorys_")
                row_result[f"{prefix}_probability"] = float(
                    np.mean(
                        (anomaly_values >= upper_threshold)
                        | (anomaly_values <= -lower_threshold)
                    )
                )
                row_result[f"{prefix}_tail_crps_upper"] = _crps(
                    np.maximum(baseline, float(row.climatology) + upper_threshold),
                    upper_truth,
                )
                row_result[f"{prefix}_tail_crps_lower"] = _crps(
                    np.minimum(baseline, float(row.climatology) - lower_threshold),
                    lower_truth,
                )
            rows.append(row_result)
            reliability_rows.append(
                {
                    "probability": probability,
                    "event": event,
                    "variable": variable,
                    "quantile": quantile,
                }
            )
    if not rows:
        return {
            "status": "partial",
            "summary": "No ensemble rows available.",
            "artifacts": [],
            "details": {},
        }
    detail = pd.DataFrame(rows)
    rel = pd.DataFrame(reliability_rows)
    bins = np.linspace(0, 1, int(options.get("reliability_bins", 11)))
    rel["bin"] = pd.cut(rel.probability, bins=bins, include_lowest=True)
    reliability = (
        rel.groupby(["variable", "quantile", "bin"], observed=True)
        .agg(
            forecast_probability=("probability", "mean"),
            observed_frequency=("event", "mean"),
            n=("event", "size"),
        )
        .reset_index()
    )
    detail["brier_point"] = (detail.probability - detail.event) ** 2
    detail["brier_warm"] = (detail.warm_probability - detail.warm_event) ** 2
    detail["brier_cold"] = (detail.cold_probability - detail.cold_event) ** 2
    detail["brier_observation_error"] = (
        detail.probability - detail.event_probability_with_observation_error
    ) ** 2
    detail["brier_observation_error_expected"] = (
        detail.probability**2
        - 2 * detail.probability * detail.event_probability_with_observation_error
        + detail.event_probability_with_observation_error
    )
    detail["brier_raw_glorys"] = (detail.raw_glorys_event - detail.event) ** 2
    detail["brier_climatology"] = (detail.climatology_probability - detail.event) ** 2
    for baseline_name in baseline_arrays:
        probability_column = f"{baseline_name}_probability"
        detail[f"brier_{baseline_name}"] = (
            detail[probability_column] - detail.event
        ) ** 2
    profile_scores = detail.groupby(
        ["variable", "quantile", "profile_id"], as_index=False
    ).agg(
        event=("event", "mean"),
        brier_point=("brier_point", "mean"),
        brier_warm=("brier_warm", "mean"),
        brier_cold=("brier_cold", "mean"),
        brier_observation_error=("brier_observation_error", "mean"),
        brier_observation_error_expected=("brier_observation_error_expected", "mean"),
        brier_raw_glorys=("brier_raw_glorys", "mean"),
        brier_climatology=("brier_climatology", "mean"),
        tail_crps_upper=("tail_crps_upper", "mean"),
        tail_crps_lower=("tail_crps_lower", "mean"),
    )
    for name in baseline_arrays:
        key = f"brier_{name}"
        extra = detail.groupby(["variable", "quantile", "profile_id"], as_index=False)[
            key
        ].mean()
        profile_scores = profile_scores.merge(
            extra, on=["variable", "quantile", "profile_id"], how="left"
        )
    summary = profile_scores.groupby(["variable", "quantile"], as_index=False).agg(
        n_profiles=("profile_id", "nunique"),
        event_rate=("event", "mean"),
        brier=("brier_point", "mean"),
        brier_warm=("brier_warm", "mean"),
        brier_cold=("brier_cold", "mean"),
        brier_observation_error=("brier_observation_error", "mean"),
        brier_observation_error_expected=("brier_observation_error_expected", "mean"),
        brier_raw_glorys=("brier_raw_glorys", "mean"),
        brier_climatology=("brier_climatology", "mean"),
        tail_crps_upper=("tail_crps_upper", "mean"),
        tail_crps_lower=("tail_crps_lower", "mean"),
    )
    for name in baseline_arrays:
        key = f"brier_{name}"
        summary[key] = (
            profile_scores.groupby(["variable", "quantile"])[key].mean().to_numpy()
        )
    artifacts = _write_table(detail, output_dir, "extreme_point_scores")
    artifacts += _write_table(reliability, output_dir, "extreme_reliability")
    artifacts += _write_table(summary, output_dir, "extreme_summary")
    artifacts += _write_table(profile_scores, output_dir, "extreme_profile_scores")
    artifacts += _figure(reliability, output_dir)
    return {
        "status": "complete",
        "summary": "Scored all available extreme cases with calibration-frozen thresholds.",
        "artifacts": artifacts,
        "details": {
            "n_points": len(detail),
            "threshold_quantile": options.get("quantile", 0.9),
            "threshold_quantiles": quantiles,
            "n_profiles": int(profile_scores.profile_id.nunique()),
            "thresholds_fit_on": "profile_balanced_calibration_observations",
            "tail_score": "fair_empirical_fixed_threshold_transform",
            "observation_error_brier": "expected_binary_brier_with_fractional_event_probability",
        },
    }
