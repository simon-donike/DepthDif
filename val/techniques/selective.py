"""Selective risk and coverage curves using forecast covariates only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import method_arrays


def _points(context: Any) -> tuple[pd.DataFrame | None, dict[str, np.ndarray]]:
    """Extract point and member data from either shared context representation."""
    bundle = getattr(context, "evaluation", None)
    if bundle is not None:
        return bundle.points, bundle.members
    if isinstance(context, dict):
        return context.get("points"), context.get("members", {})
    return getattr(context, "points", None), getattr(context, "members", {})


def _write(frame: pd.DataFrame, output: Path, name: str) -> list[str]:
    """Write a paper table in CSV and LaTeX formats."""
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{name}.csv", output / f"{name}.tex"]
    frame.to_csv(paths[0], index=False)
    frame.to_latex(paths[1], index=False, float_format="%.5g")
    return [str(path) for path in paths]


def _plot(frame: pd.DataFrame, output: Path) -> list[str]:
    """Write the risk coverage plot in publication friendly formats."""
    if frame.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    variables = sorted(frame.variable.dropna().unique())
    fig, axes = plt.subplots(
        1, len(variables), figsize=(5 * len(variables), 3.8), squeeze=False
    )
    for axis, variable in zip(axes.flat, variables):
        subset = frame[frame.variable == variable]
        for method, group in subset.groupby("method"):
            axis.plot(group.coverage, group.risk, marker="o", label=method)
        axis.set(
            title=variable, xlabel="Coverage retained", ylabel="Profile balanced MAE"
        )
        axis.legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        path = output / f"selective_risk_coverage.{suffix}"
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Rank cases by deployment available uncertainty and report risk coverage."""
    options = dict(options or {})
    output_dir = Path(output_dir)
    points, members = _points(context)
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No evaluation points.",
            "artifacts": [],
            "details": {},
        }
    rows = []
    work = points.reset_index(drop=True)
    baselines = {
        name: values
        for name, values in method_arrays(work, members).items()
        if name not in {"residual", "temperature", "salinity", "glorys"}
    }
    calibrated_members = None
    if isinstance(members, dict):
        calibrated_members = members.get("calibrated_glorys")
        if calibrated_members is None:
            calibrated_members = members.get("glorys_calibrated")
        if calibrated_members is None:
            calibrated_members = members.get("probabilistic_glorys")
    for variable, frame in work.groupby("variable", sort=True):
        array = None
        if isinstance(members, dict):
            # Inference stores the aligned physical ensemble as ``residual``.
            array = members.get(variable, members.get("residual"))
        if array is None:
            continue
        array = np.asarray(array, float)
        variable_indices = np.flatnonzero(work.variable.to_numpy() == variable)
        if array.shape[0] not in {len(work), len(variable_indices)}:
            continue
        for profile, group in frame.groupby("profile_id", sort=False):
            idx = group.index.to_numpy()
            local_idx = (
                idx
                if array.shape[0] == len(work)
                else np.searchsorted(variable_indices, idx)
            )
            prediction = np.mean(array[local_idx], axis=1)
            spread = (
                np.mean(np.std(array[local_idx], axis=1))
                if array.shape[1] > 1
                else np.nan
            )
            model_error = float(
                np.mean(np.abs(prediction - group.observation.to_numpy(float)))
            )
            glorys_error = float(
                np.mean(
                    np.abs(
                        group.glorys.to_numpy(float) - group.observation.to_numpy(float)
                    )
                )
            )
            distance = float(
                pd.to_numeric(
                    group.get("nearest_context_distance_km", pd.Series([np.nan])).iloc[
                        0
                    ]
                )
            )
            density = float(
                pd.to_numeric(group.get("context_density", pd.Series([np.nan])).iloc[0])
            )
            calibrated_spread = float(
                pd.to_numeric(
                    group.get("calibrated_glorys_spread", pd.Series([np.nan])).iloc[0]
                )
            )
            if calibrated_members is not None and not np.isfinite(calibrated_spread):
                calibrated_array = np.asarray(calibrated_members, float)
                if calibrated_array.shape[0] == len(work):
                    calibrated_spread = float(
                        np.mean(np.std(calibrated_array[idx], axis=1))
                    )
            rows.append(
                {
                    "profile_id": profile,
                    "variable": variable,
                    "model_risk": model_error,
                    "glorys_risk": glorys_error,
                    "spread": spread,
                    "calibrated_glorys_spread": calibrated_spread,
                    "distance": distance,
                    "density": density,
                    **{
                        f"risk_{name}": float(
                            np.mean(
                                np.abs(
                                    values[idx].mean(axis=1)
                                    - group.observation.to_numpy(float)
                                )
                            )
                        )
                        for name, values in baselines.items()
                    },
                }
            )
    if not rows:
        return {
            "status": "partial",
            "summary": "No profile aligned member arrays available.",
            "artifacts": [],
            "details": {},
        }
    profile = pd.DataFrame(rows)
    curves = []
    covariates = ["spread", "calibrated_glorys_spread", "distance", "density"]
    if (
        not profile.get("calibrated_glorys_spread", pd.Series(dtype=float))
        .notna()
        .any()
    ):
        covariates.remove("calibrated_glorys_spread")
    missing_covariates = []
    for variable, variable_profile in profile.groupby("variable", sort=True):
        for covariate in covariates:
            usable = variable_profile[np.isfinite(variable_profile[covariate])]
            if usable.empty:
                missing_covariates.append(f"{variable}:{covariate}")
                continue
            # Larger density and smaller uncertainty/distance are preferred.
            ascending = covariate != "density"
            ranked = usable.sort_values(
                covariate, ascending=ascending, na_position="last"
            )
            for coverage in np.linspace(0.1, 1.0, 10):
                count = max(1, int(np.ceil(coverage * len(ranked))))
                selected = ranked.iloc[:count]
                for method, risk in [
                    ("DepthDif", selected.model_risk.mean()),
                    ("GLORYS", selected.glorys_risk.mean()),
                    *[(name, selected[f"risk_{name}"].mean()) for name in baselines],
                ]:
                    curves.append(
                        {
                            "variable": variable,
                            "covariate": covariate,
                            "coverage": count / len(ranked),
                            "method": method,
                            "risk": risk,
                            "n_profiles": count,
                        }
                    )
            # Random ordering is the expected reference, averaged analytically.
            curves.append(
                {
                    "variable": variable,
                    "covariate": covariate,
                    "coverage": 1.0,
                    "method": "random",
                    "risk": variable_profile.model_risk.mean(),
                    "n_profiles": len(variable_profile),
                }
            )
    curve = pd.DataFrame(
        curves,
        columns=("variable", "covariate", "coverage", "method", "risk", "n_profiles"),
    )
    artifacts = _write(profile, output_dir, "selective_profile_risk")
    artifacts += _write(curve, output_dir, "selective_risk_coverage")
    artifacts += _plot(curve[curve.covariate == "spread"], output_dir)
    return {
        "status": "complete" if not missing_covariates else "partial",
        "summary": "Computed deployment covariate risk coverage curves without using outcomes for ranking.",
        "artifacts": artifacts,
        "details": {
            "n_profiles": len(profile),
            "ranking_covariates": covariates,
            "missing_covariates": missing_covariates,
            "oracle_free": True,
        },
    }
