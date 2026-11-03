"""Calibration-only comparator ladder for the frozen reconstruction.

The routines in this module deliberately keep calibration and verification data
separate.  They provide useful controls for the claim that the network adds
information beyond a GLORYS point estimate: a regularised bias correction,
correlated empirical GLORYS errors, and a climatological distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import save_figure, save_table
from val.metrics import _cluster_bootstrap, _profile_losses


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a context, bundle, or mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _bundle(context: Any, name: str) -> Any:
    """Return the named bundle while supporting the shared Context contract."""
    bundle = _field(context, name)
    if bundle is not None:
        return bundle
    # The small fallback also makes the stage convenient in notebooks/tests.
    return context


def _crps(truth: float, members: np.ndarray, fair: bool = False) -> float:
    """Compute empirical CRPS, with the fair finite ensemble correction."""
    x = np.asarray(members, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(truth):
        return np.nan
    first = np.mean(np.abs(x - truth))
    if x.size < 2:
        return float(first)
    pair = np.abs(x[:, None] - x[None, :])
    denominator = x.size * (x.size - 1) if fair else x.size * x.size
    return float(first - 0.5 * pair.sum() / denominator)


def _write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> list[str]:
    """Write stable CSV and LaTeX artefacts for a paper table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = output_dir / f"{name}.csv", output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    frame.to_latex(tex_path, index=False, float_format="%.5g")
    return [str(csv_path), str(tex_path)]


def _paired_cluster_table(
    points: pd.DataFrame, arrays: dict[str, np.ndarray], options: dict[str, Any]
) -> pd.DataFrame:
    """Bootstrap paired profile losses by cluster for direct comparator CIs."""
    methods = [
        name for name in ("glorys", "glorys_calibrated", "residual") if name in arrays
    ]
    if len(methods) < 2 or "cluster_id" not in points:
        return pd.DataFrame()
    profile_rows = []
    for (variable, profile), frame in points.groupby(
        ["variable", "profile_id"], sort=False
    ):
        indices = frame.index.to_numpy()
        cluster = (
            str(frame.cluster_id.iloc[0])
            if frame.cluster_id.notna().any()
            else str(profile)
        )
        row = {"variable": variable, "profile_id": profile, "cluster_id": cluster}
        truth = frame.observation.to_numpy(float)
        for method in methods:
            forecast = arrays[method][indices]
            forecast = forecast if forecast.ndim == 2 else forecast[:, None]
            row[method] = float(np.nanmean((np.nanmean(forecast, axis=1) - truth) ** 2))
        profile_rows.append(row)
    detail = pd.DataFrame(profile_rows)
    rng = np.random.default_rng(int(options.get("seed", 20261103)) + 107)
    output = []
    for variable, variable_detail in detail.groupby("variable", sort=False):
        clusters = variable_detail.cluster_id.unique()
        if len(clusters) < int(options.get("min_clusters", 3)):
            continue
        for challenger in methods[1:]:
            # _cluster_bootstrap first averages every profile, then resamples
            # independent clusters with profile-count denominators.  This is
            # the same profile-balanced estimand as the main validation score.
            profile = variable_detail[
                ["profile_id", "cluster_id", challenger, methods[0]]
            ].rename(columns={challenger: "model_loss", methods[0]: "baseline_loss"})
            result = _cluster_bootstrap(
                profile,
                int(options.get("bootstrap_replicates", 1000)),
                int(options.get("seed", 20261103)) + 107,
                int(options.get("min_profiles", 3)),
                int(options.get("min_clusters", 3)),
                lambda values: values,
            )
            output.append(
                {
                    "variable": variable,
                    "reference": methods[0],
                    "challenger": challenger,
                    "n_clusters": result["clusters"],
                    "n_profiles": result["profiles"],
                    "mean_loss_difference": float(result["difference"]),
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "p_one_sided": (
                        float(result["p_one_sided"])
                        if np.isfinite(result["p_one_sided"])
                        else np.nan
                    ),
                    "difference_is_challenger_minus_reference": True,
                }
            )
    return pd.DataFrame(output)


def _rank_and_spread_figures(
    points: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    options: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Write PIT/rank histograms and spread-skill figures for each ensemble."""
    import matplotlib.pyplot as plt

    rank_rows, artifacts = [], []
    ensemble_names = [
        name
        for name, values in arrays.items()
        if np.asarray(values).ndim == 2 and np.asarray(values).shape[1] > 1
    ]
    variables = list(points.variable.dropna().astype(str).unique())
    panels = [(variable, name) for variable in variables for name in ensemble_names][:6]
    if not panels:
        return artifacts, pd.DataFrame()
    rank_figure, rank_axes = plt.subplots(
        len(panels),
        1,
        squeeze=False,
        figsize=(6, max(2.5, 2.2 * len(panels))),
    )
    spread_figure, spread_axes = plt.subplots(
        len(panels),
        1,
        squeeze=False,
        figsize=(6, max(2.5, 2.2 * len(panels))),
    )
    for rank_axis, spread_axis, (variable, name) in zip(
        rank_axes[:, 0], spread_axes[:, 0], panels
    ):
        values = np.asarray(arrays[name], dtype=float)
        selected = points.variable.astype(str).to_numpy() == variable
        ranks = []
        spreads, absolute_errors = [], []
        for index, truth in enumerate(points.observation.to_numpy(float)):
            if not selected[index]:
                continue
            members = values[index]
            if not np.isfinite(truth) or not np.isfinite(members).all():
                continue
            # Randomized tie breaking is fixed so PIT results are reproducible.
            below = np.sum(members < truth)
            ties = np.sum(members == truth)
            rank = below + int(
                np.random.default_rng(
                    int(options.get("seed", 20261103)) + index
                ).integers(ties + 1)
            )
            ranks.append(rank)
            spreads.append(float(np.std(members)))
            absolute_errors.append(float(abs(np.mean(members) - truth)))
        if ranks:
            rank_rows.extend(
                {
                    "variable": variable,
                    "method": name,
                    "rank": rank,
                    "count": ranks.count(rank),
                }
                for rank in range(values.shape[1] + 1)
            )
            rank_axis.hist(
                ranks, bins=np.arange(values.shape[1] + 2) - 0.5, edgecolor="black"
            )
            rank_axis.set_title(f"{variable}/{name}: rank histogram (PIT proxy)")
            rank_axis.set_xlabel("rank")
            rank_axis.set_ylabel("count")
            spread_axis.scatter(spreads, absolute_errors, s=10, alpha=0.45)
            spread_axis.set_title(f"{variable}/{name}: spread-skill diagnostic")
            spread_axis.set_xlabel("ensemble standard deviation")
            spread_axis.set_ylabel("absolute mean error")
    artifacts += save_figure(rank_figure, output_dir, "baseline_rank_histograms")
    artifacts += save_figure(spread_figure, output_dir, "baseline_spread_skill")
    return artifacts, pd.DataFrame(rank_rows)


def _fit_calibration(
    calibration: pd.DataFrame, options: dict[str, Any]
) -> dict[str, Any]:
    """Fit depth/season residual means and a library of complete profiles."""
    frame = calibration.copy()
    frame["depth_bin"] = (
        (frame["depth_m"].astype(float) / options["depth_bin_m"]).round().astype(int)
    )
    times = frame.get("target_time", frame.get("observation_time"))
    if times is None:
        times = pd.Series(pd.NaT, index=frame.index)
    frame["month"] = pd.to_datetime(
        times, errors="coerce", utc=True, format="mixed"
    ).dt.month
    frame["month"] = frame["month"].fillna(0).astype(int)
    frame["residual"] = frame["observation"].astype(float) - frame["glorys"].astype(
        float
    )
    if "climatology" in frame:
        frame["residual_climatology"] = frame["observation"].astype(float) - frame[
            "climatology"
        ].astype(float)
    else:
        frame["residual_climatology"] = frame["residual"]
    frame = frame[np.isfinite(frame.residual)]
    # The shrinkage denominator is deliberately fixed from calibration only.
    profile_residual = (
        frame.groupby(["variable", "depth_bin", "month", "profile_id"], dropna=False)
        .residual.mean()
        .reset_index()
    )
    grouped = profile_residual.groupby(
        ["variable", "depth_bin", "month"], dropna=False
    ).residual.agg(["mean", "count"])
    global_mean = frame.groupby("variable").residual.mean().to_dict()
    profile_climate = (
        frame.groupby(["variable", "depth_bin", "month", "profile_id"], dropna=False)
        .residual_climatology.mean()
        .reset_index()
    )
    climate_grouped = profile_climate.groupby(
        ["variable", "depth_bin", "month"], dropna=False
    ).residual_climatology.agg(["mean", "count"])
    climate_global = frame.groupby("variable").residual_climatology.mean().to_dict()
    return {
        "rows": frame,
        "groups": grouped,
        "global": global_mean,
        "climate_groups": climate_grouped,
        "climate_global": climate_global,
    }


def _bias(fit: dict[str, Any], row: pd.Series, options: dict[str, Any]) -> float:
    """Return a regularised calibration bias for one verification point."""
    variable = row.variable
    depth_bin = int(round(float(row.depth_m) / options["depth_bin_m"]))
    month = int(row.month)
    grouped = fit["groups"]
    value = (
        grouped.loc[(variable, depth_bin, month)]
        if (variable, depth_bin, month) in grouped.index
        else None
    )
    prior = float(fit["global"].get(variable, 0.0))
    if value is None:
        return prior
    strength = float(options["shrinkage_count"])
    return float(
        (value["count"] * value["mean"] + strength * prior)
        / (value["count"] + strength)
    )


def _climate_bias(
    fit: dict[str, Any], row: pd.Series, options: dict[str, Any]
) -> float:
    """Return the fitted observation-minus-climatology location correction."""
    key = (
        row.variable,
        int(round(float(row.depth_m) / options["depth_bin_m"])),
        int(row.month),
    )
    value = (
        fit["climate_groups"].loc[key] if key in fit["climate_groups"].index else None
    )
    prior = float(fit["climate_global"].get(row.variable, 0.0))
    if value is None:
        return prior
    strength = float(options["shrinkage_count"])
    return float(
        (value["count"] * value["mean"] + strength * prior)
        / (value["count"] + strength)
    )


def _profile_ensemble(
    fit: dict[str, Any],
    points: pd.DataFrame,
    options: dict[str, Any],
    rng: np.random.Generator,
    residual_column: str = "residual",
) -> np.ndarray:
    """Sample complete calibration residual profiles and interpolate in depth."""
    calibration = fit["rows"]
    members = int(options["members"])
    output = np.full((len(points), members), np.nan, dtype=float)
    # When profile identifiers are shared by temperature and salinity, select
    # one source profile for both variables and every depth.  This preserves
    # the empirical cross-variable residual coupling of the calibration set.
    variables = list(points.variable.dropna().unique())
    grouped_source = {
        str(profile): group
        for profile, group in calibration.groupby("profile_id", sort=False)
    }
    joint_ids = [
        profile
        for profile, group in grouped_source.items()
        if set(variables).issubset(set(group.variable))
    ]
    selected: dict[tuple[str, int], str] = {}
    for profile in points.profile_id.dropna().astype(str).unique():
        for member in range(members):
            choices = joint_ids or list(grouped_source)
            if choices:
                selected[(profile, member)] = str(
                    choices[int(rng.integers(len(choices)))]
                )
    # One profile is selected per member and variable.  Vertical interpolation
    # preserves coherent profile errors and avoids diagonal noise masquerading
    # as a probabilistic GLORYS comparator.
    for variable, target in points.groupby("variable", sort=False):
        source = calibration[calibration.variable == variable]
        profiles = list(source.groupby("profile_id", sort=False))
        if not profiles:
            continue
        for member in range(members):
            # Select a complete residual profile per verification profile.  This
            # retains vertical coherence without making every location share
            # the same calibration error draw.
            for _, target_profile in target.groupby("profile_id", sort=False):
                selected_id = selected.get(
                    (str(target_profile.profile_id.iloc[0]), member)
                )
                if selected_id is not None and selected_id in grouped_source:
                    profile = grouped_source[selected_id]
                    profile = profile[profile.variable == variable]
                else:
                    _, profile = profiles[int(rng.integers(len(profiles)))]
                profile = profile.sort_values("depth_m")
                x = profile.depth_m.to_numpy(float)
                y = profile[residual_column].to_numpy(float)
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                for index, row in target_profile.iterrows():
                    if x.size and x.min() <= float(row.depth_m) <= x.max():
                        output[index, member] = np.interp(float(row.depth_m), x, y)
    return output


def _summary(
    points: pd.DataFrame, arrays: dict[str, np.ndarray], options: dict[str, Any]
) -> pd.DataFrame:
    """Calculate profile-balanced scores for all comparator distributions."""
    rows = []
    for name, values in arrays.items():
        for variable, frame in points.groupby("variable", sort=True):
            indices = frame.index.to_numpy()
            matrix = values[indices] if values.ndim == 2 else values[indices, None]
            truth = frame.observation.to_numpy(float)
            glorys = frame.glorys.to_numpy(float)
            errors = np.nanmean(matrix, axis=1) - truth
            for profile, local in frame.assign(_row=np.arange(len(frame))).groupby(
                "profile_id", sort=False
            ):
                pos = local._row.to_numpy()
                if not len(pos):
                    continue
                rows.append(
                    {
                        "method": name,
                        "variable": variable,
                        "profile_id": profile,
                        "rmse": float(np.sqrt(np.nanmean(errors[pos] ** 2))),
                        "mae": float(np.nanmean(np.abs(errors[pos]))),
                        "crps": float(
                            np.nanmean(
                                [
                                    _crps(t, matrix[p], fair=False)
                                    for p, t in zip(pos, truth[pos])
                                ]
                            )
                        ),
                        "fair_crps": float(
                            np.nanmean(
                                [
                                    _crps(t, matrix[p], fair=True)
                                    for p, t in zip(pos, truth[pos])
                                ]
                            )
                        ),
                        "coverage80": float(
                            np.mean(
                                (
                                    truth[pos]
                                    >= np.nanpercentile(matrix[pos], 10, axis=1)
                                )
                                & (
                                    truth[pos]
                                    <= np.nanpercentile(matrix[pos], 90, axis=1)
                                )
                            )
                        ),
                        "width80": float(
                            np.nanmean(
                                np.nanpercentile(matrix[pos], 90, axis=1)
                                - np.nanpercentile(matrix[pos], 10, axis=1)
                            )
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["method", "variable"], as_index=False)
        .agg(
            n_profiles=("profile_id", "nunique"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            crps=("crps", "mean"),
            fair_crps=("fair_crps", "mean"),
            coverage80=("coverage80", "mean"),
            width80=("width80", "mean"),
        )
    )


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run the calibration-only GLORYS comparator ladder."""
    options = {
        "depth_bin_m": 50.0,
        "shrinkage_count": 5.0,
        "members": 50,
        "seed": 20261103,
        "min_profiles": 3,
        **(options or {}),
    }
    output_dir = Path(output_dir)
    evaluation, calibration = _bundle(context, "evaluation"), _bundle(
        context, "calibration"
    )
    points = _field(evaluation, "points")
    calibration_points = _field(calibration, "points")
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No verification points available.",
            "artifacts": [],
            "details": {},
        }
    if not isinstance(calibration_points, pd.DataFrame) or calibration_points.empty:
        return {
            "status": "blocked",
            "summary": "Calibration cohort is required; no test labels were used.",
            "artifacts": [],
            "details": {"calibration_required": True},
        }
    required = {"observation", "glorys", "variable", "depth_m", "profile_id"}
    if not required.issubset(calibration_points.columns):
        return {
            "status": "blocked",
            "summary": f"Calibration cohort lacks {sorted(required - set(calibration_points.columns))}.",
            "artifacts": [],
            "details": {},
        }
    fit = _fit_calibration(calibration_points, options)
    eval_points = points.reset_index(drop=True).copy()
    times = eval_points.get("target_time", eval_points.get("observation_time"))
    if times is None:
        times = pd.Series(pd.NaT, index=eval_points.index)
    eval_points["month"] = (
        pd.to_datetime(times, errors="coerce", utc=True, format="mixed")
        .dt.month.fillna(0)
        .astype(int)
    )
    bias = np.array([_bias(fit, row, options) for _, row in eval_points.iterrows()])
    climate_bias = np.array(
        [_climate_bias(fit, row, options) for _, row in eval_points.iterrows()]
    )
    raw = eval_points.glorys.to_numpy(float)
    corrected = raw + bias
    residuals = _profile_ensemble(
        fit, eval_points, options, np.random.default_rng(int(options["seed"]))
    )
    residuals_climatology = _profile_ensemble(
        fit,
        eval_points,
        options,
        np.random.default_rng(int(options["seed"]) + 1),
        "residual_climatology",
    )
    # Centre using the fitted calibration mean, rather than the finite sample
    # mean.  The latter makes members dependent and invalidates fair CRPS.
    residuals = residuals - bias[:, None]
    residuals_climatology = residuals_climatology - climate_bias[:, None]
    calibrated = corrected[:, None] + residuals
    climatology = (
        eval_points.climatology.to_numpy(float)[:, None]
        + climate_bias[:, None]
        + residuals_climatology
    )
    arrays = {
        "glorys": raw[:, None],
        "glorys_bias_corrected": corrected[:, None],
        "glorys_calibrated": calibrated,
        "climatology_calibrated": climatology,
    }
    existing = _field(evaluation, "members", {})
    if isinstance(existing, dict):
        for name, values in existing.items():
            if np.asarray(values).ndim == 2 and np.asarray(values).shape[0] == len(
                eval_points
            ):
                arrays[name] = np.asarray(values, dtype=float)
    invalid = {
        name: int((~np.isfinite(values)).any(axis=1).sum())
        for name, values in arrays.items()
        if values.ndim == 2
    }
    invalid_ensemble = {name: count for name, count in invalid.items() if count}
    # A method with missing vertical support is excluded as a whole.  Quiet
    # row-wise nanmean scoring would give it a different target population.
    scorable_arrays = {
        name: values for name, values in arrays.items() if np.isfinite(values).all()
    }
    summary = _summary(eval_points, scorable_arrays, options)
    artifacts = save_table(
        summary,
        output_dir,
        "baseline_scores",
        "Profile-balanced comparator scores; calibration uses a disjoint cohort.",
    )
    detail = pd.DataFrame(
        {
            "point_id": eval_points.get("point_id", np.arange(len(eval_points))),
            "baseline_glorys_bias": corrected,
        }
    )
    artifacts += save_table(
        detail,
        output_dir,
        "baseline_predictions",
        "Deterministic GLORYS bias correction fitted on calibration profiles.",
    )
    paired = _paired_cluster_table(eval_points, scorable_arrays, options)
    if not paired.empty:
        artifacts += save_table(
            paired,
            output_dir,
            "baseline_paired_cluster_comparisons",
            "Paired cluster bootstrap differences; negative values favour the challenger.",
        )
    figure_artifacts, ranks = _rank_and_spread_figures(
        eval_points, scorable_arrays, output_dir, options
    )
    artifacts += figure_artifacts
    if not ranks.empty:
        artifacts += save_table(
            ranks,
            output_dir,
            "baseline_rank_histograms",
            "Finite ensemble rank counts with fixed randomized tie handling.",
        )
    # Add outputs to the shared evaluation bundle for downstream stages.
    members = _field(evaluation, "members")
    if isinstance(members, dict):
        if not invalid.get("glorys_calibrated"):
            members["glorys_calibrated"] = calibrated
        if not invalid.get("climatology_calibrated"):
            members["climatology_calibrated"] = climatology
    calibration_members = _field(calibration, "members")
    if isinstance(calibration_members, dict):
        # These are explicitly labelled in-sample calibration distributions;
        # they are suitable for ranking diagnostics, never final skill claims.
        calibration_eval = calibration_points.reset_index(drop=True).copy()
        calibration_residuals = _profile_ensemble(
            fit,
            calibration_eval,
            options,
            np.random.default_rng(int(options["seed"]) + 2),
        )
        calibration_distribution = (
            calibration_eval.glorys.to_numpy(float)[:, None] + calibration_residuals
        )
        if np.isfinite(calibration_distribution).all():
            calibration_members["glorys_calibrated_in_sample"] = (
                calibration_distribution
            )
    if isinstance(_field(evaluation, "points"), pd.DataFrame):
        _field(evaluation, "points")["baseline_glorys_bias"] = corrected
    min_profiles = int(options["min_profiles"])
    status = (
        "complete"
        if len(fit["rows"].profile_id.unique()) >= min_profiles and not invalid_ensemble
        else "partial"
    )
    return {
        "status": status,
        "summary": f"Scored {len(summary)} calibrated comparator groups from {fit['rows'].profile_id.nunique()} calibration profiles.",
        "artifacts": artifacts,
        "details": {
            "calibration_profiles": int(fit["rows"].profile_id.nunique()),
            "test_profiles": int(eval_points.profile_id.nunique()),
            "calibration_only": True,
            "fair_crps_is_iid_only": True,
            "invalid_ensemble_rows": invalid_ensemble,
            "in_sample_calibration_members": isinstance(calibration_members, dict),
        },
    }
