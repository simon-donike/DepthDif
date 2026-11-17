"""Paper diagnostics for the value of sparse profile conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import pandas as pd

from val.metrics import _crps
from val.paper_common import save_figure, save_table


def _evaluation_parts(context: Any) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Extract point assignments and named ensembles from the shared context."""
    evaluation = getattr(context, "evaluation", context)
    points = getattr(evaluation, "points", None)
    members = getattr(evaluation, "members", None)
    if points is None and isinstance(evaluation, Mapping):
        points, members = evaluation["points"], evaluation["members"]
    if members is None and getattr(evaluation, "path", None):
        with np.load(Path(evaluation.path) / "members.npz") as saved:
            members = {name: saved[name] for name in saved.files}
    if points is None or members is None:
        raise ValueError("conditioning stage requires evaluation points and ensembles")
    if not isinstance(members, Mapping):
        raise ValueError("evaluation.members must be a mapping of ensemble names")
    return points.copy(), {
        str(key): np.asarray(value) for key, value in members.items()
    }


def _scores(points: pd.DataFrame, values: np.ndarray, name: str) -> pd.DataFrame:
    """Compute profile-balanced point and ensemble scores for one budget."""
    if values.ndim != 2 or values.shape[0] != len(points) or values.shape[1] < 1:
        raise ValueError(f"{name} has incompatible ensemble shape {values.shape}")
    frame = points[
        ["point_id", "profile_id", "variable", "region", "observation", "glorys"]
    ].copy()
    frame["method"] = name
    frame["mae"] = np.abs(values.mean(axis=1) - frame.observation.to_numpy(float))
    frame["rmse"] = (values.mean(axis=1) - frame.observation.to_numpy(float)) ** 2
    frame["crps"] = [
        (
            _crps(float(observation), ensemble)[1]
            if len(ensemble) > 1
            else abs(float(observation) - float(ensemble[0]))
        )
        for observation, ensemble in zip(frame.observation, values)
    ]
    grouped = frame.groupby(["profile_id", "variable"], sort=False, observed=True)
    return (
        grouped[["mae", "rmse", "crps"]]
        .mean()
        .assign(rmse=lambda item: np.sqrt(item.rmse), method=name)
        .reset_index()
        .merge(
            frame[["profile_id", "variable", "region"]].drop_duplicates(),
            on=["profile_id", "variable"],
        )
    )


def _reference_scores(points: pd.DataFrame) -> pd.DataFrame:
    """Score deterministic GLORYS and climatology on the identical rows."""
    rows = []
    for method in ("glorys", "climatology"):
        if method not in points:
            continue
        frame = points[
            ["profile_id", "variable", "region", "observation", method]
        ].copy()
        error = frame[method].to_numpy(float) - frame.observation.to_numpy(float)
        frame["mae"], frame["rmse"], frame["crps"] = (
            np.abs(error),
            error**2,
            np.abs(error),
        )
        rows.append(
            frame.groupby(["profile_id", "variable"], sort=False, observed=True)[
                ["mae", "rmse", "crps"]
            ]
            .mean()
            .assign(rmse=lambda item: np.sqrt(item.rmse), method=method)
            .reset_index()
            .merge(
                frame[["profile_id", "variable", "region"]].drop_duplicates(),
                on=["profile_id", "variable"],
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(
    context: Any, output_dir: Path, options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write conditioning budget tables and figures from one captured ensemble run."""
    del options
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        points, members = _evaluation_parts(context)
    except (KeyError, ValueError, FileNotFoundError) as error:
        (output_dir / "status.json").write_text(
            json.dumps({"status": "blocked", "reason": str(error)}) + "\n"
        )
        return {"status": "blocked", "summary": str(error), "artifacts": []}
    rows = []
    has_budget_arrays = False
    metadata = getattr(getattr(context, "evaluation", None), "metadata", {})
    baseline_names = set(metadata.get("baselines", {}))
    for name, values in members.items():
        if (
            name != "residual"
            and not name.startswith("conditioning_")
            and not name.startswith("baseline_oi_")
            and not name.startswith("baseline_model_")
            and name not in baseline_names
        ):
            continue
        if name.startswith("baseline_model_"):
            model_name, _, value = name.removeprefix("baseline_model_").rpartition(
                "_budget_"
            )
            budget = int(value)
            score_name = model_name
        elif name in baseline_names:
            budget = "full"
            score_name = name
        elif name.startswith("baseline_oi_"):
            budget = name.rsplit("_", 1)[-1]
            score_name = name
        else:
            budget = "full" if name == "residual" else int(name.split("_", 1)[1])
            score_name = str(budget)
        has_budget_arrays |= name.startswith("conditioning_")
        scored = _scores(points, values, score_name)
        scored["budget"] = budget
        rows.append(scored)
    if not rows or not has_budget_arrays:
        reason = "No conditioning_* arrays were captured; enable paper.conditioning."
        (output_dir / "status.json").write_text('{"status": "blocked"}\n')
        return {"status": "blocked", "summary": reason, "artifacts": []}
    scores = pd.concat(rows, ignore_index=True)
    references = _reference_scores(points)
    summary = (
        scores.groupby(["method", "budget", "variable"], sort=True, observed=True)[
            ["mae", "rmse", "crps"]
        ]
        .mean()
        .reset_index()
    )
    table_artifacts = save_table(
        summary,
        output_dir,
        "conditioning_scores",
        "Profile balanced scores by retained sparse profile conditioning budget",
    )
    if metadata.get("baseline_availability"):
        table_artifacts += save_table(
            pd.DataFrame(metadata["baseline_availability"]),
            output_dir,
            "conditioning_baseline_availability",
            "Support and measured runtimes for every baseline and conditioning budget",
        )
    pivot = summary.pivot_table(
        index=["budget", "method"], columns="variable", values="mae"
    )
    pivot_artifacts = save_table(
        pivot.reset_index(),
        output_dir,
        "conditioning_mae_by_budget",
        "Profile balanced MAE across retained sparse profile budgets and methods",
    )
    reference_artifacts = []
    comparison_artifacts = []
    if not references.empty:
        reference_summary = (
            references.groupby(["method", "variable"], observed=True)[
                ["mae", "rmse", "crps"]
            ]
            .mean()
            .reset_index()
        )
        reference_artifacts = save_table(
            reference_summary,
            output_dir,
            "conditioning_reference_scores",
            "Deterministic reference scores on the identical validation rows",
        )
        comparison = summary.pivot_table(
            index=["method", "budget", "variable"],
            values=["mae", "rmse", "crps"],
        ).reset_index()
        reference_wide = reference_summary.pivot(
            index="variable", columns="method", values="mae"
        ).reset_index()
        comparison = comparison.merge(reference_wide, on="variable", how="left")
        comparison_artifacts = save_table(
            comparison,
            output_dir,
            "conditioning_comparisons",
            "Model budget scores paired with deterministic GLORYS and climatology scores",
        )
    figure_paths = []
    try:
        import matplotlib.pyplot as plt

        variables = list(summary.variable.astype(str).unique())
        figure, axes = plt.subplots(
            1,
            len(variables),
            figsize=(6.5 * max(1, len(variables)), 4.0),
            squeeze=False,
        )
        for axis, variable in zip(axes.flat, variables):
            group = summary[summary.variable.astype(str).eq(variable)]
            numeric = group[group.budget.astype(str) != "full"].copy()
            numeric["budget_number"] = pd.to_numeric(numeric.budget)
            numeric["curve"] = numeric.method.map(
                lambda name: (
                    "GLORYS + local OI"
                    if str(name).startswith("baseline_oi_glorys")
                    else (
                        "climatology + local OI"
                        if str(name).startswith("baseline_oi_climatology")
                        else ("DepthDif" if str(name).isdigit() else str(name))
                    )
                )
            )
            for label, curve in numeric.groupby("curve", sort=True):
                curve = curve.sort_values("budget_number")
                axis.plot(curve.budget_number, curve.mae, marker="o", label=label)
            full = group[
                group.budget.astype(str).eq("full")
                & group.method.astype(str).eq("full")
            ]
            if not full.empty:
                axis.axhline(
                    float(full.mae.iloc[0]),
                    color="black",
                    linestyle="--",
                    label="full context",
                )
            axis.set(
                title=variable,
                xlabel="Number of retained input profiles",
                ylabel="Profile-balanced MAE",
            )
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure_paths.extend(save_figure(figure, output_dir, "conditioning_curve"))
    except ImportError:
        pass
    return {
        "status": "complete",
        "summary": "Profile-conditioning budget curve computed.",
        "artifacts": [
            *table_artifacts,
            *pivot_artifacts,
            *reference_artifacts,
            *comparison_artifacts,
            *figure_paths,
        ],
        "details": {"budgets": sorted(str(value) for value in scores.budget.unique())},
    }
