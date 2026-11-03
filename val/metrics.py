"""Independent, auditable validation scoring for ensemble ocean profiles."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "point_id",
    "profile_id",
    "cluster_id",
    "source",
    "region",
    "latitude",
    "longitude",
    "observation_time",
    "target_time",
    "depth_m",
    "variable",
    "observation",
    "glorys",
    "climatology",
    "independence",
    "split_role",
}
VARIABLES = {"temperature", "salinity"}
DEPTH_BINS = [0, 10, 50, 100, 200, 500, 700, 2000, np.inf]
DEPTH_LABELS = [
    "0-10m",
    "10-50m",
    "50-100m",
    "100-200m",
    "200-500m",
    "500-700m",
    "700-2000m",
    "2000m+",
]


def _crps(y: float, x: np.ndarray) -> tuple[float, float]:
    """Return empirical and finite-ensemble fair CRPS for one observation."""
    ensemble = np.sort(np.asarray(x, dtype=float))
    if ensemble.ndim != 1 or len(ensemble) < 2:
        raise ValueError(
            "CRPS requires a one-dimensional ensemble with at least two members"
        )
    first = float(np.abs(ensemble - y).mean())
    ranks = 2 * np.arange(1, len(ensemble) + 1) - len(ensemble) - 1
    pair_sum = float(2 * np.sum(ranks * ensemble))
    return (
        first - pair_sum / (2 * len(ensemble) ** 2),
        first - pair_sum / (2 * len(ensemble) * (len(ensemble) - 1)),
    )


def validate_bundle(
    points: pd.DataFrame,
    members: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Validate identities, coordinates, predictions, and threshold provenance."""
    if not isinstance(points, pd.DataFrame) or points.empty:
        raise ValueError("points must be a non-empty DataFrame")
    missing = sorted(REQUIRED_COLUMNS - set(points.columns))
    if missing:
        raise ValueError(f"points missing required columns: {missing}")
    members = np.asarray(members)
    if members.ndim != 2 or members.shape[0] != len(points) or members.shape[1] < 2:
        raise ValueError("members must have shape (len(points), M), with M >= 2")
    if not np.isfinite(members).all():
        raise ValueError("members contains non-finite values")
    if points["point_id"].duplicated().any() or points["point_id"].isna().any():
        raise ValueError("point_id must be unique and populated")
    categorical = (
        "profile_id",
        "cluster_id",
        "source",
        "region",
        "variable",
        "independence",
        "split_role",
        "observation_time",
        "target_time",
    )
    for column in categorical:
        if points[column].isna().any():
            raise ValueError(f"{column} must be populated")
    if not points["variable"].isin(VARIABLES).all():
        raise ValueError("variable must be one of temperature or salinity")
    if points.groupby("profile_id", dropna=False)["cluster_id"].nunique().gt(1).any():
        raise ValueError("each profile_id must map to exactly one cluster_id")
    numeric = [
        "latitude",
        "longitude",
        "depth_m",
        "observation",
        "glorys",
        "climatology",
    ]
    numeric += [column for column in points if column.startswith("baseline_")]
    for column in numeric:
        values = pd.to_numeric(points[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
    if (points["depth_m"] < 0).any():
        raise ValueError("depth_m must be non-negative")
    if not points["latitude"].between(-90, 90).all():
        raise ValueError("latitude must be in [-90, 90]")
    if not points["longitude"].between(-180, 180).all():
        raise ValueError("longitude must be in [-180, 180]")
    metadata = metadata or {}
    if {"threshold_low", "threshold_high"}.intersection(points.columns):
        if not {"threshold_low", "threshold_high"}.issubset(points.columns):
            raise ValueError(
                "threshold_low and threshold_high must be supplied together"
            )
        if not metadata.get("thresholds_external") or not metadata.get(
            "thresholds_provenance"
        ):
            raise ValueError(
                "activated thresholds require external provenance metadata"
            )
        low = pd.to_numeric(points["threshold_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(points["threshold_high"], errors="coerce").to_numpy(float)
        if (
            not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or (low >= high).any()
        ):
            raise ValueError("thresholds must be finite and low < high")


def _safe_skill(score: float, reference: float) -> float:
    """Return skill without turning a perfect reference score into infinity."""
    if not np.isfinite(reference) or np.isclose(reference, 0):
        return np.nan
    return float(1 - score / reference)


def _holm(values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving missing values."""
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(np.asarray(values)[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * values[index]))
        result[index] = running
    return result.tolist()


def _rank_mass(y: float, ensemble: np.ndarray) -> np.ndarray:
    """Distribute a tied truth rank uniformly over all compatible rank bins."""
    less = int(np.count_nonzero(ensemble < y))
    equal = int(np.count_nonzero(ensemble == y))
    mass = np.zeros(len(ensemble) + 1)
    mass[less : less + equal + 1] = 1 / (equal + 1)
    return mass


def _groups(points: pd.DataFrame) -> Iterator[tuple[dict[str, str], pd.DataFrame]]:
    """Yield variable-separated groups at useful spatial resolutions."""
    levels = (
        ("all", ["variable", "split_role"]),
        ("source_region", ["source", "region", "variable", "split_role"]),
        ("depth_band", ["variable", "split_role", "depth_band"]),
        (
            "source_region_depth_band",
            ["source", "region", "variable", "split_role", "depth_band"],
        ),
    )
    for level, columns in levels:
        for keys, frame in points.groupby(
            columns, sort=True, observed=True, dropna=False
        ):
            keys = keys if isinstance(keys, tuple) else (keys,)
            values = dict(zip(columns, map(str, keys)))
            yield {
                "group_level": level,
                "source": values.get("source", "ALL"),
                "region": values.get("region", "ALL"),
                "variable": values["variable"],
                "split_role": values["split_role"],
                "depth_band": values.get("depth_band", "ALL"),
            }, frame


def _profile_losses(
    frame: pd.DataFrame,
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
) -> pd.DataFrame:
    """Reduce paired point losses to equal-weight profile summaries."""
    work = frame[["profile_id", "cluster_id"]].copy()
    work["model_loss"] = model_loss
    work["baseline_loss"] = baseline_loss
    # Depth-rich profiles must not silently dominate the paired comparison.
    return work.groupby(["cluster_id", "profile_id"], sort=False, as_index=False).mean()


def _cluster_bootstrap(
    profile: pd.DataFrame,
    replicates: int,
    seed: int,
    min_profiles: int,
    min_clusters: int,
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Bootstrap independent clusters with efficient profile-count weighting."""
    grouped = profile.groupby("cluster_id", sort=False)
    model_sum = grouped["model_loss"].sum().to_numpy(float)
    baseline_sum = grouped["baseline_loss"].sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    n_clusters, n_profiles = len(counts), len(profile)
    model_score = float(transform(np.array([profile["model_loss"].mean()]))[0])
    baseline_score = float(transform(np.array([profile["baseline_loss"].mean()]))[0])
    difference = model_score - baseline_score
    result = {
        "model_score": model_score,
        "baseline_score": baseline_score,
        "difference": difference,
        "median_profile_difference": float(
            np.median(
                transform(profile.model_loss.to_numpy(float))
                - transform(profile.baseline_loss.to_numpy(float))
            )
        ),
        "win_fraction": float(
            np.mean(
                transform(profile.model_loss.to_numpy(float))
                < transform(profile.baseline_loss.to_numpy(float))
            )
        ),
        "clusters": n_clusters,
        "profiles": n_profiles,
    }
    if n_clusters < min_clusters or n_profiles < min_profiles:
        return result | {
            "inference_status": "insufficient_support",
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_one_sided": np.nan,
            "p_two_sided": np.nan,
        }
    if replicates < 2:
        raise ValueError("bootstrap_replicates must be at least 2")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    # Bounded chunks avoid allocating a replicate-by-cluster matrix for the whole run.
    # Keep the temporary integer matrix below roughly two million elements.
    chunk_size = max(1, min(256, 2_000_000 // n_clusters))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        multiplicity = rng.multinomial(
            n_clusters, np.full(n_clusters, 1 / n_clusters), size=stop - start
        )
        sampled_counts = multiplicity @ counts
        sampled_model = transform((multiplicity @ model_sum) / sampled_counts)
        sampled_baseline = transform((multiplicity @ baseline_sum) / sampled_counts)
        draws[start:stop] = sampled_model - sampled_baseline
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    # A null-centered distribution tests the observed magnitude against delta=0.
    centered = draws - difference
    lower_tail = (np.count_nonzero(centered <= difference) + 1) / (replicates + 1)
    upper_tail = (np.count_nonzero(centered >= difference) + 1) / (replicates + 1)
    return result | {
        "inference_status": "estimated",
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_one_sided": float(lower_tail),
        "p_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
    }


def _method_row(
    labels: dict[str, str],
    frame: pd.DataFrame,
    method: str,
    prediction: np.ndarray,
    glorys_error: np.ndarray,
) -> dict[str, Any]:
    """Calculate deterministic and profile-balanced scores for one method."""
    y = frame.observation.to_numpy(float)
    error = prediction - y
    work = frame[["profile_id"]].copy()
    work["absolute"] = np.abs(error)
    work["squared"] = error**2
    profile = work.groupby("profile_id", sort=False).mean()
    mae = float(np.abs(error).mean())
    row = {
        **labels,
        "method": method,
        "metric_type": "deterministic",
        "n_points": len(frame),
        "n_profiles": frame.profile_id.nunique(),
        "n_clusters": frame.cluster_id.nunique(),
        "mae": mae,
        "profile_balanced_mae": float(profile.absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "profile_balanced_rmse": float(np.sqrt(profile.squared.mean())),
        "bias": float(error.mean()),
        "median_absolute_error": float(np.median(np.abs(error))),
        "absolute_error_q90": float(np.quantile(np.abs(error), 0.9)),
        "absolute_error_q95": float(np.quantile(np.abs(error), 0.95)),
        "crps": mae,
        "fair_crps": np.nan,
        "skill_vs_glorys": _safe_skill(mae, float(glorys_error.mean())),
    }
    if {"threshold_low", "threshold_high"}.issubset(frame.columns):
        low, high = frame.threshold_low.to_numpy(float), frame.threshold_high.to_numpy(
            float
        )
        row["brier_low"] = float(
            np.mean(((prediction <= low).astype(float) - (y <= low)) ** 2)
        )
        row["brier_high"] = float(
            np.mean(((prediction >= high).astype(float) - (y >= high)) ** 2)
        )
    return row


def _score_rows(
    labels: dict[str, str],
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    crps: np.ndarray,
    fair_crps: np.ndarray,
    members: np.ndarray,
) -> list[dict[str, Any]]:
    """Build deterministic and probabilistic score rows for one group."""
    y = frame.observation.to_numpy(float)
    indices = frame.index.to_numpy()
    glorys_error = np.abs(predictions["glorys"] - y)
    rows = [
        _method_row(labels, frame, method, prediction, glorys_error)
        for method, prediction in predictions.items()
    ]
    prediction = members[indices].mean(axis=1)
    row = _method_row(labels, frame, "ensemble", prediction, glorys_error)
    row.update(
        metric_type="probabilistic",
        crps=float(crps[indices].mean()),
        fair_crps=float(fair_crps[indices].mean()),
        skill_vs_glorys=_safe_skill(
            float(crps[indices].mean()), float(glorys_error.mean())
        ),
    )
    if {"threshold_low", "threshold_high"}.issubset(frame.columns):
        low, high = frame.threshold_low.to_numpy(float), frame.threshold_high.to_numpy(
            float
        )
        probability_low = (members[indices] <= low[:, None]).mean(axis=1)
        probability_high = (members[indices] >= high[:, None]).mean(axis=1)
        row["brier_low"] = float(np.mean((probability_low - (y <= low)) ** 2))
        row["brier_high"] = float(np.mean((probability_high - (y >= high)) ** 2))
    rows.append(row)
    return rows


def _reliability_rows(
    labels: dict[str, str],
    frame: pd.DataFrame,
    members: np.ndarray,
) -> list[dict[str, Any]]:
    """Build interval, spread-skill, and normalized rank-histogram rows."""
    indices = frame.index.to_numpy()
    ensemble = members[indices]
    y = frame.observation.to_numpy(float)
    common = {
        **labels,
        "n_points": len(frame),
        "n_profiles": frame.profile_id.nunique(),
        "n_clusters": frame.cluster_id.nunique(),
    }
    rows = []
    for level in (0.5, 0.8, 0.9, 0.95):
        lower, upper = np.quantile(
            ensemble, [(1 - level) / 2, 1 - (1 - level) / 2], axis=1
        )
        rows += [
            {
                **common,
                "metric": f"coverage_{int(level * 100)}",
                "score": float(np.mean((y >= lower) & (y <= upper))),
            },
            {
                **common,
                "metric": f"width_{int(level * 100)}",
                "score": float(np.mean(upper - lower)),
            },
        ]
    rows += [
        {
            **common,
            "metric": "mean_spread",
            "score": float(ensemble.std(axis=1, ddof=1).mean()),
        },
        {
            **common,
            "metric": "rmse_ensemble_mean",
            "score": float(np.sqrt(np.mean((ensemble.mean(axis=1) - y) ** 2))),
        },
    ]
    rank = np.mean(
        [_rank_mass(value, sample) for value, sample in zip(y, ensemble)], axis=0
    )
    rows += [
        {**common, "metric": f"rank_histogram_{index}", "score": float(score)}
        for index, score in enumerate(rank)
    ]
    return rows


def _comparison_rows(
    labels: dict[str, str],
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    crps: np.ndarray,
    replicates: int,
    seed: int,
    min_profiles: int,
    min_clusters: int,
) -> list[dict[str, Any]]:
    """Run paired median-MAE, mean-RMSE, and ensemble-CRPS comparisons."""
    y = frame.observation.to_numpy(float)
    baselines = [
        name for name in predictions if name not in {"model_mean", "model_median"}
    ]
    endpoints = (
        (
            "mae_model_median",
            np.abs(predictions["model_median"] - y),
            np.abs,
            lambda x: x,
        ),
        (
            "rmse_model_mean",
            (predictions["model_mean"] - y) ** 2,
            lambda x: x**2,
            np.sqrt,
        ),
        ("crps_ensemble", crps[frame.index.to_numpy()], np.abs, lambda x: x),
    )
    rows = []
    for baseline in baselines:
        for endpoint, model_loss, baseline_loss_function, transform in endpoints:
            baseline_loss = baseline_loss_function(predictions[baseline] - y)
            profile = _profile_losses(frame, model_loss, baseline_loss)
            result = _cluster_bootstrap(
                profile, replicates, seed, min_profiles, min_clusters, transform
            )
            comparison_id = "|".join([*labels.values(), endpoint, baseline])
            rows.append(
                {
                    **labels,
                    "comparison_id": comparison_id,
                    "endpoint": endpoint,
                    "baseline": baseline,
                    **result,
                }
            )
    return rows


def _confirmation_gates(
    points: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, bool]:
    """Return gates required before confirmatory wording is allowed."""
    return {
        "checkpoint_verified": bool(metadata.get("checkpoint_verified")),
        "training_split_verified": bool(metadata.get("training_split_verified")),
        "conditioning_exclusion_verified": bool(
            metadata.get("conditioning_exclusion_verified")
        ),
        "confirmation_untouched": bool(metadata.get("confirmation_untouched")),
        "all_verified_not_assimilated": bool(
            (points.independence == "verified_not_assimilated").all()
        ),
        "confirmation_rows": bool((points.split_role == "confirmation").all()),
        "demo_false": not bool(metadata.get("demo")),
    }


def _classify(
    comparisons: pd.DataFrame, confirmation_eligible: bool, metadata: dict[str, Any]
) -> pd.DataFrame:
    """Apply endpoint gates, Holm correction, and cautious decisions."""
    declared = set(map(str, metadata.get("confirmatory_endpoints", [])))
    comparisons["evidence_class"] = "exploratory"
    if confirmation_eligible and declared:
        mask = comparisons.comparison_id.isin(declared)
        comparisons.loc[mask, "evidence_class"] = "confirmatory"
    comparisons["p_holm"] = np.nan
    comparisons["p_two_sided_holm"] = np.nan
    family_columns = [
        "evidence_class",
        "variable",
        "split_role",
        "endpoint",
        "baseline",
    ]
    # Each variable/endpoint experiment controls its complete spatial search family.
    for _, family in comparisons.groupby(family_columns, sort=False, dropna=False):
        comparisons.loc[family.index, "p_holm"] = _holm(family.p_one_sided.tolist())
        comparisons.loc[family.index, "p_two_sided_holm"] = _holm(
            family.p_two_sided.tolist()
        )
    decisions = []
    for row in comparisons.itertuples(index=False):
        if row.inference_status == "insufficient_support":
            decision = "underpowered"
        elif row.evidence_class == "confirmatory":
            if row.p_holm < 0.05 and row.difference < 0:
                decision = "better"
            elif row.p_two_sided_holm < 0.05 and row.difference > 0:
                decision = "worse"
            else:
                decision = "no evidence"
        elif row.p_holm < 0.05 and row.ci_high < 0:
            decision = "exploratory better"
        elif row.p_two_sided_holm < 0.05 and row.ci_low > 0:
            decision = "exploratory worse"
        else:
            decision = "inconclusive"
        decisions.append(decision)
    comparisons["decision"] = decisions
    return comparisons


def _regional_decisions(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Summarize regional evidence as winner, loser, mixed, or unresolved."""
    selected = comparisons[
        (comparisons.group_level == "source_region")
        & (comparisons.baseline == "glorys")
    ]
    rows = []
    columns = ["source", "region", "variable", "split_role"]
    for keys, frame in selected.groupby(columns, sort=True, dropna=False):
        decisions = set(frame.decision)
        better = bool(decisions & {"better", "exploratory better"})
        worse = bool(decisions & {"worse", "exploratory worse"})
        if better and worse:
            verdict = "mixed"
        elif better:
            verdict = "winner" if "better" in decisions else "exploratory winner"
        elif worse:
            verdict = "loser" if "worse" in decisions else "exploratory loser"
        elif decisions == {"underpowered"}:
            verdict = "underpowered"
        else:
            verdict = "no evidence"
        rows.append(dict(zip(columns, keys)) | {"verdict": verdict})
    return pd.DataFrame(rows, columns=columns + ["verdict"])


def _by_depth(
    points: pd.DataFrame, mean: np.ndarray, median: np.ndarray
) -> pd.DataFrame:
    """Aggregate depth scores at canonical exact depths or declared bands."""
    columns = [
        "source",
        "region",
        "variable",
        "split_role",
        "profile_id",
        "cluster_id",
        "depth_m",
        "depth_band",
    ]
    work = points[columns].copy()
    y = points.observation.to_numpy(float)
    work["model_mean_absolute_error"] = np.abs(mean - y)
    work["model_median_absolute_error"] = np.abs(median - y)
    work["glorys_absolute_error"] = np.abs(points.glorys.to_numpy(float) - y)
    outputs = []
    for _, frame in work.groupby(["variable", "split_role"], sort=True):
        frame = frame.copy()
        exact = frame.depth_m.nunique() <= 256
        frame["depth_resolution"] = "exact" if exact else "band"
        frame["depth"] = frame.depth_m.astype(str) if exact else frame.depth_band
        group = [
            "source",
            "region",
            "variable",
            "split_role",
            "depth_resolution",
            "depth",
        ]
        outputs.append(
            frame.groupby(group, sort=True, observed=True)
            .agg(
                n_points=("profile_id", "size"),
                n_profiles=("profile_id", "nunique"),
                n_clusters=("cluster_id", "nunique"),
                model_mean_mae=("model_mean_absolute_error", "mean"),
                model_median_mae=("model_median_absolute_error", "mean"),
                glorys_mae=("glorys_absolute_error", "mean"),
            )
            .reset_index()
        )
    return pd.concat(outputs, ignore_index=True)


def score_bundle(
    points: pd.DataFrame,
    members: np.ndarray,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20261103,
    min_clusters: int = 20,
    min_profiles: int = 30,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a validation bundle and write machine-readable and static reports."""
    metadata = dict(metadata or {})
    if bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least 2")
    if min_clusters < 1 or min_profiles < 1:
        raise ValueError("min_clusters and min_profiles must be positive")
    if "variable_units" in metadata and not isinstance(
        metadata["variable_units"], dict
    ):
        raise ValueError("metadata variable_units must be a mapping")
    if "confirmatory_endpoints" in metadata and not isinstance(
        metadata["confirmatory_endpoints"], (list, tuple, set)
    ):
        raise ValueError("metadata confirmatory_endpoints must be a sequence")
    validate_bundle(points, members, metadata)
    points = points.reset_index(drop=True).copy()
    members = np.asarray(members, dtype=float)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points["depth_band"] = pd.cut(
        points.depth_m,
        DEPTH_BINS,
        labels=DEPTH_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    y = points.observation.to_numpy(float)
    model_mean = members.mean(axis=1)
    model_median = np.median(members, axis=1)
    sorted_members = np.sort(members, axis=1)
    first = np.abs(sorted_members - y[:, None]).mean(axis=1)
    ranks = 2 * np.arange(1, members.shape[1] + 1) - members.shape[1] - 1
    pair_sum = 2 * (sorted_members * ranks).sum(axis=1)
    crps = first - pair_sum / (2 * members.shape[1] ** 2)
    fair_crps = first - pair_sum / (2 * members.shape[1] * (members.shape[1] - 1))
    baseline_columns = ["glorys", "climatology"] + sorted(
        column for column in points if column.startswith("baseline_")
    )
    all_predictions = {
        "model_mean": model_mean,
        "model_median": model_median,
        **{column: points[column].to_numpy(float) for column in baseline_columns},
    }
    score_rows, reliability_rows, comparison_rows = [], [], []
    for group_number, (labels, frame) in enumerate(_groups(points)):
        indices = frame.index.to_numpy()
        predictions = {
            name: values[indices] for name, values in all_predictions.items()
        }
        score_rows += _score_rows(labels, frame, predictions, crps, fair_crps, members)
        reliability_rows += _reliability_rows(labels, frame, members)
        comparison_rows += _comparison_rows(
            labels,
            frame,
            predictions,
            crps,
            bootstrap_replicates,
            seed + group_number,
            min_profiles,
            min_clusters,
        )
    scores = pd.DataFrame(score_rows)
    reliability = pd.DataFrame(reliability_rows)
    comparisons = pd.DataFrame(comparison_rows)
    gates = _confirmation_gates(points, metadata)
    confirmation_eligible = all(gates.values())
    comparisons = _classify(comparisons, confirmation_eligible, metadata)
    regional = _regional_decisions(comparisons)
    count_columns = [
        "group_level",
        "source",
        "region",
        "variable",
        "split_role",
        "depth_band",
        "n_points",
        "n_profiles",
        "n_clusters",
    ]
    counts = scores[count_columns].drop_duplicates()
    scores.to_csv(output_dir / "scores.csv", index=False)
    reliability.to_csv(output_dir / "reliability.csv", index=False)
    comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    counts.to_csv(output_dir / "counts.csv", index=False)
    regional.to_csv(output_dir / "regional_decisions.csv", index=False)
    _by_depth(points, model_mean, model_median).to_csv(
        output_dir / "by_depth.csv", index=False
    )
    summary = {
        "metadata": metadata,
        "demo": bool(metadata.get("demo")),
        "n_points": len(points),
        "n_profiles": int(points.profile_id.nunique()),
        "n_clusters": int(points.cluster_id.nunique()),
        "variables": sorted(points.variable.unique().tolist()),
        "variable_units": metadata.get(
            "variable_units", {"temperature": "degC", "salinity": "PSU"}
        ),
        "baselines": baseline_columns,
        "confirmation_gates": gates,
        "confirmation_eligible": confirmation_eligible,
        "confirmatory_comparisons": int(
            (comparisons.evidence_class == "confirmatory").sum()
        ),
        "regional_decisions": regional.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _report(output_dir, scores, comparisons, regional, summary)
    return summary


def _delta_svg(comparisons: pd.DataFrame) -> str:
    """Render a self-contained paired-delta interval plot with per-variable scales."""
    frame = comparisons[
        np.isfinite(comparisons.ci_low) & np.isfinite(comparisons.ci_high)
    ].copy()
    if frame.empty:
        return "<p>No interval plot: every comparison is underpowered.</p>"
    width, left, right, row_height = 1000, 300, 30, 26
    rows = []
    y_position = 28
    for variable, variable_frame in frame.groupby("variable", sort=True):
        extent = max(
            abs(variable_frame.ci_low.min()),
            abs(variable_frame.ci_high.max()),
            1e-12,
        )
        scale = (width - left - right) / (2 * extent)
        zero = left + (width - left - right) / 2
        unit = "°C" if variable == "temperature" else "salinity units"
        rows.append(
            f"<text x='4' y='{y_position}' font-weight='bold'>{html.escape(variable)} "
            f"(delta units: {html.escape(unit)})</text>"
        )
        y_position += row_height
        start = y_position - 18
        for result in variable_frame.itertuples(index=False):
            low = zero + result.ci_low * scale
            high = zero + result.ci_high * scale
            point = zero + result.difference * scale
            color = (
                "#087f23"
                if "better" in result.decision
                else "#b00020" if "worse" in result.decision else "#555"
            )
            significance = (
                " *corrected p<0.05"
                if result.evidence_class == "confirmatory"
                and result.decision in {"better", "worse"}
                else ""
            )
            label = f"{result.region} · {result.endpoint} · {result.evidence_class}{significance}"
            rows += [
                f"<text x='4' y='{y_position}' font-size='11'>{html.escape(label, quote=False)}</text>",
                f"<line x1='{zero:.1f}' y1='{start:.1f}' x2='{zero:.1f}' y2='{y_position + 6:.1f}' stroke='#aaa'/>",
                f"<line x1='{low:.1f}' y1='{y_position - 4}' x2='{high:.1f}' y2='{y_position - 4}' stroke='{color}' stroke-width='3'/>",
                f"<circle cx='{point:.1f}' cy='{y_position - 4}' r='4' fill='{color}'/>",
            ]
            y_position += row_height
        rows.append(
            f"<text x='{left}' y='{y_position}' font-size='10'>{-extent:.3g}</text>"
            f"<text x='{zero - 4}' y='{y_position}' font-size='10'>0</text>"
            f"<text x='{width - right - 35}' y='{y_position}' font-size='10'>{extent:.3g}</text>"
        )
        y_position += row_height
    return (
        f"<svg viewBox='0 0 {width} {y_position}' width='100%' role='img' "
        "aria-label='Paired score differences and 95 percent bootstrap intervals'>"
        + "".join(rows)
        + "</svg>"
    )


def _report(
    directory: Path,
    scores: pd.DataFrame,
    comparisons: pd.DataFrame,
    regional: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write directly readable Markdown and standalone HTML reports."""
    if summary["demo"]:
        banner = "DEMO DATA — RESULTS ARE NOT SCIENTIFIC EVIDENCE"
    elif not summary["confirmation_eligible"]:
        banner = "CONFIRMATION EVIDENCE MISSING — RESULTS ARE EXPLORATORY"
    else:
        banner = "Confirmation gates pass; only predeclared endpoints are confirmatory"
    gates = pd.DataFrame([summary["confirmation_gates"]])
    top_line = comparisons[
        comparisons.group_level.isin(["all", "source_region"])
        & (comparisons.baseline == "glorys")
    ]
    overview = (
        comparisons.groupby(["evidence_class", "decision"], dropna=False)
        .size()
        .rename("comparisons")
        .reset_index()
    )
    variable_note = (
        "Salinity not evaluated: this bundle contains temperature only."
        if "salinity" not in summary["variables"]
        else "Temperature and salinity were evaluated separately."
    )
    units_note = ", ".join(
        f"{variable}: {unit}"
        for variable, unit in summary["variable_units"].items()
        if variable in summary["variables"]
    )
    markdown = (
        f"# Validation report\n\n## {banner}\n\n"
        f"Points: {summary['n_points']}; profiles: {summary['n_profiles']}; "
        f"independent clusters: {summary['n_clusters']}. {variable_note} "
        f"Units: {units_note}.\n\n"
        f"## Decision counts\n\n{overview.to_string(index=False)}\n\n"
        f"## Confirmation gates\n\n{gates.to_string(index=False)}\n\n"
        f"## Regional verdicts\n\n{regional.to_string(index=False)}\n\n"
        f"## Top-line comparisons\n\n{top_line.to_string(index=False)}\n\n"
        f"## Scores\n\n{scores.to_string(index=False)}\n"
    )
    (directory / "report.md").write_text(markdown, encoding="utf-8")
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Validation report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222}"
        "table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}"
        "th,td{border:1px solid #ccc;padding:5px 8px;text-align:right}"
        "th{background:#f2f2f2;position:sticky;top:0}h2{margin-top:28px}</style></head><body>"
        f"<h1>Validation report</h1><div style='background:#8b0000;color:white;padding:16px;font-weight:bold'>{html.escape(banner)}</div>"
        f"<p>{summary['n_points']} points; {summary['n_profiles']} profiles; "
        f"{summary['n_clusters']} independent clusters. {html.escape(variable_note)} "
        f"Units: {html.escape(units_note)}.</p>"
        f"<h2>Decision overview</h2>{overview.to_html(index=False, border=0)}"
        f"<h2>Paired differences (lower is better)</h2>{_delta_svg(top_line)}"
        f"<h2>Confirmation gates</h2>{gates.to_html(index=False, border=0)}"
        f"<h2>Regional verdicts</h2>{regional.to_html(index=False, border=0)}"
        f"<h2>Top-line comparisons against GLORYS</h2>{top_line.to_html(index=False, border=0)}"
        f"<h2>Scores</h2>{scores.to_html(index=False, border=0)}</body></html>"
    )
    (directory / "report.html").write_text(document, encoding="utf-8")
