"""Ensemble size, uncertainty quality, and measured runtime diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import method_arrays, point_crps, profile_mean


def _crps(x: np.ndarray, y: np.ndarray) -> float:
    """Return fair empirical CRPS for a single finite ensemble."""
    if len(x) < 2:
        return float(np.mean(np.abs(x - y))) if len(x) else np.nan
    return float(
        np.mean(np.abs(x - y))
        - np.sum(np.abs(x[:, None] - x[None, :])) / (2 * len(x) * (len(x) - 1))
    )


def _get(
    context: Any,
) -> tuple[pd.DataFrame | None, dict[str, np.ndarray], dict[str, Any]]:
    """Extract evaluation data and metadata from a Context or mapping."""
    bundle = getattr(context, "evaluation", None)
    if bundle is not None:
        return bundle.points, bundle.members, getattr(bundle, "metadata", {}) or {}
    if isinstance(context, dict):
        return (
            context.get("points"),
            context.get("members", {}),
            context.get("metadata", {}),
        )
    return (
        getattr(context, "points", None),
        getattr(context, "members", {}),
        getattr(context, "metadata", {}) or {},
    )


def _write(frame: pd.DataFrame, output: Path, name: str) -> list[str]:
    """Write CSV and LaTeX table artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = output / f"{name}.csv", output / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(tex_path, index=False, float_format="%.5g")
    return [str(csv_path), str(tex_path)]


def _plot(frame: pd.DataFrame, output: Path) -> list[str]:
    """Write the member count curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fig, ax = plt.subplots(figsize=(5, 3.8))
    metric = "crps_fair" if frame.crps_fair.notna().any() else "crps_empirical"
    for variable, group in frame.groupby("variable"):
        ax.plot(group.n_members, group[metric], marker="o", label=variable)
    ax.set(
        xlabel="Generations",
        ylabel=(
            "Fair CRPS" if metric == "crps_fair" else "Point-forecast absolute error"
        ),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = output / f"efficiency_member_curve.{suffix}"
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score deterministic prefixes of the same ensemble draws."""
    options = dict(options or {})
    output_dir = Path(output_dir)
    points, members, metadata = _get(context)
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No evaluation points.",
            "artifacts": [],
            "details": {},
        }
    requested = options.get("member_counts", (1, 2, 5, 10, 20, 50, 100))
    missing_prefixes = {}
    rows, reasons = [], []
    for variable, frame in points.reset_index(drop=True).groupby("variable", sort=True):
        array = None
        if isinstance(members, dict):
            # The inference bundle uses one all-point matrix named residual.
            array = members.get(variable, members.get("residual"))
        if array is None:
            reasons.append(f"{variable}: no ensemble")
            continue
        array = np.asarray(array, float)
        variable_indices = np.flatnonzero(points.variable.to_numpy() == variable)
        if array.shape[0] not in {len(points), len(variable_indices)}:
            reasons.append(f"{variable}: members are not point aligned")
            continue
        truth = frame.observation.to_numpy(float)
        missing_prefixes[str(variable)] = [
            int(count) for count in requested if count > array.shape[1]
        ]
        for count in requested:
            if count > array.shape[1]:
                continue
            indices = frame.index.to_numpy()
            if array.shape[0] != len(points):
                indices = np.searchsorted(variable_indices, indices)
            prefix = array[indices, :count]
            crps = [_crps(prefix[index], truth[index]) for index in range(len(frame))]
            lower, upper = np.quantile(prefix, [0.05, 0.95], axis=1)
            coverage = np.mean((truth >= lower) & (truth <= upper))
            diversity = np.mean(np.abs(prefix[:, :, None] - prefix[:, None, :]))
            rows.append(
                {
                    "variable": variable,
                    "n_members": int(count),
                    "n_points": len(frame),
                    "crps_fair": np.nanmean(crps) if count > 1 else np.nan,
                    "crps_empirical": point_crps(prefix, truth).mean(),
                    "coverage_90": coverage if count > 1 else np.nan,
                    "mean_width_90": np.mean(upper - lower) if count > 1 else np.nan,
                    "mean_pairwise_difference": diversity if count > 1 else np.nan,
                    "uncertainty_available": count > 1,
                }
            )
    if not rows:
        return {
            "status": "partial",
            "summary": "No requested member prefix is available.",
            "artifacts": [],
            "details": {"reasons": reasons},
        }
    result = pd.DataFrame(rows)
    timing = {}
    if isinstance(metadata, dict):
        # Inference currently calls this ``timing``; accept the explicit trace
        # name as well so saved bundles remain forward compatible.
        timing = metadata.get("timing", metadata.get("timing_trace", {}))
        if not timing and metadata.get("member_timings"):
            timing = {"member_timings": metadata["member_timings"]}
    measured = bool(timing.get("measured", False))
    timing_rows = []
    per_member = (
        timing.get("per_member_seconds", {}) if isinstance(timing, dict) else {}
    )
    if isinstance(timing, dict) and timing.get("member_timings"):
        # Sum measured batch traces by member index; prefix cost is the cost of
        # all members included in that prefix, with no invented GLORYS timing.
        measured_rows = pd.DataFrame(timing["member_timings"])
        if {"member", "elapsed_seconds"}.issubset(measured_rows.columns):
            # The quality ablations are separate inference runs.  The prefix
            # curve below describes the full residual ensemble only.
            if "budget" in measured_rows.columns:
                full_rows = measured_rows[
                    measured_rows.budget.astype(str).str.lower() == "full"
                ]
                if not full_rows.empty:
                    measured_rows = full_rows
            per_member = measured_rows.groupby("member").elapsed_seconds.sum().to_dict()
            measured = True
    for count in sorted(result.n_members.unique()):
        if isinstance(per_member, dict):
            member_values = [
                value
                for member, value in per_member.items()
                if int(member) < int(count)
            ]
            value = float(np.sum(member_values)) if member_values else np.nan
        else:
            value = np.nan
        timing_rows.append(
            {
                "n_members": int(count),
                "seconds": value,
                "timing_status": (
                    "measured"
                    if measured and np.isfinite(value)
                    else "unavailable_or_estimated"
                ),
                "elapsed_measured": bool(measured and np.isfinite(value)),
            }
        )
    timing_frame = pd.DataFrame(timing_rows)
    artifacts = _write(result, output_dir, "efficiency_member_counts")
    artifacts += _write(timing_frame, output_dir, "efficiency_timing")
    artifacts += _plot(result, output_dir)
    method_rows = []
    for name, values in method_arrays(points, members).items():
        for variable, frame in points.groupby("variable", sort=True):
            matrix = values[frame.index.to_numpy()]
            truth = frame.observation.to_numpy(float)
            error = matrix.mean(axis=1) - truth
            timing_info = metadata.get("baselines", {}).get(name, {})
            seconds = timing_info.get("elapsed_seconds", np.nan)
            method_rows.append(
                {
                    "method": name,
                    "variable": variable,
                    "n_members": matrix.shape[1],
                    "mae": profile_mean(frame, np.abs(error)),
                    "rmse": np.sqrt(profile_mean(frame, error**2)),
                    "empirical_crps": profile_mean(frame, point_crps(matrix, truth)),
                    "seconds": seconds,
                    "timing_status": (
                        "measured" if np.isfinite(seconds) else "unavailable"
                    ),
                    "cost_scope": "complete cohort inference, shared across variables",
                }
            )
    artifacts += _write(pd.DataFrame(method_rows), output_dir, "efficiency_all_methods")
    status = (
        "complete" if measured and not any(missing_prefixes.values()) else "partial"
    )
    summary = (
        "Scored member prefixes; runtime timing is measured."
        if measured
        else "Scored member prefixes; no measured runtime trace was supplied."
    )
    return {
        "status": status,
        "summary": summary,
        "artifacts": artifacts,
        "details": {
            "available_members": {
                key: int(np.asarray(value).shape[1]) for key, value in members.items()
            },
            "timing_status": "measured" if measured else "unavailable_or_estimated",
            "unavailable_requested_prefixes": missing_prefixes,
            "glorys_cost_comparison": "not_estimated",
            "uncertainty_available": bool(result.uncertainty_available.any()),
        },
    }
