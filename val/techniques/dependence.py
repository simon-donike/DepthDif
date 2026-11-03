"""Functional and dependence sensitive ensemble validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import save_figure, save_table


def _crps(y: float, ensemble: np.ndarray) -> float:
    """Compute the empirical CRPS for a scalar ensemble."""
    x = np.asarray(ensemble, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or not np.isfinite(y):
        return np.nan
    return float(np.abs(x - y).mean() - np.abs(x[:, None] - x[None, :]).mean() / 2)


def energy_score(observation: np.ndarray, ensemble: np.ndarray) -> float:
    """Return the multivariate energy score with supplied training scaling."""
    y, x = np.asarray(observation, float), np.asarray(ensemble, float)
    if (
        x.ndim != 2
        or y.ndim != 1
        or x.shape[1] != y.size
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        return np.nan
    return float(
        np.mean(np.linalg.norm(x - y, axis=1))
        - 0.5 * np.mean(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2))
    )


def variogram_score(
    observation: np.ndarray, ensemble: np.ndarray, power: float = 0.5
) -> float:
    """Return a pairwise variogram score, sensitive to vertical dependence."""
    y, x = np.asarray(observation, float), np.asarray(ensemble, float)
    if (
        x.ndim != 2
        or y.ndim != 1
        or x.shape[1] != y.size
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        return np.nan
    score = 0.0
    for left in range(y.size):
        for right in range(left + 1, y.size):
            observed = abs(y[left] - y[right]) ** power
            predicted = np.mean(np.abs(x[:, left] - x[:, right]) ** power)
            score += (predicted - observed) ** 2
    return float(score)


def _weighted_profile_mean(values: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Compute normalized trapezoidal means for nonuniform vertical grids."""
    values, depths = np.asarray(values, float), np.asarray(depths, float)
    if values.ndim != 2 or depths.size != values.shape[1] or depths[-1] <= depths[0]:
        return np.nanmean(values, axis=1)
    return np.trapezoid(values, depths, axis=1) / (depths[-1] - depths[0])


def _field(context: Any, name: str, default: Any = None) -> Any:
    """Read a shared context field from a dataclass or dictionary."""
    return (
        context.get(name, default)
        if isinstance(context, dict)
        else getattr(context, name, default)
    )


def _profile_matrix(
    points: pd.DataFrame, values: np.ndarray, variable: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build complete, depth-sorted profile matrices from point aligned values."""
    result = []
    frame = points[points.variable == variable]
    for profile, group in frame.groupby("profile_id", sort=False):
        indices = group.index.to_numpy()
        order = np.argsort(group.depth_m.to_numpy(float))
        truth = group.observation.to_numpy(float)[order]
        ensemble = values[indices][order].T
        if np.isfinite(truth).all() and np.isfinite(ensemble).all() and len(truth) >= 2:
            result.append((str(profile), truth, ensemble))
    return result


def _permute_vertical(ensemble: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute member identity independently at each depth."""
    result = np.empty_like(ensemble)
    for depth in range(ensemble.shape[1]):
        result[:, depth] = ensemble[rng.permutation(ensemble.shape[0]), depth]
    return result


def _joint_profile(
    points: pd.DataFrame, array: np.ndarray, profile: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return paired T/S truth and member vectors on common profile depths."""
    frames = {}
    for variable in ("temperature", "salinity"):
        frame = points[(points.profile_id == profile) & (points.variable == variable)]
        if frame.empty:
            return None
        frame = frame.sort_values("depth_m")
        frames[variable] = frame
    if not np.array_equal(frames["temperature"].depth_m, frames["salinity"].depth_m):
        return None
    vectors, ensembles = [], []
    for variable in ("temperature", "salinity"):
        frame = frames[variable]
        array_for_profile = np.asarray(array, dtype=float)[frame.index.to_numpy()]
        vectors.append(frame.observation.to_numpy(float))
        ensembles.append(array_for_profile.T)
    return (
        np.concatenate(vectors),
        np.concatenate(ensembles, axis=1),
        frames["temperature"].depth_m.to_numpy(float),
    )


def run(
    context: Any, output_dir: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compare intact ensembles with fixed seeded dependence destroying controls."""
    options = dict(options or {})
    evaluation = _field(context, "evaluation")
    points = (
        _field(evaluation, "points")
        if evaluation is not None
        else _field(context, "points")
    )
    members = (
        _field(evaluation, "members", {})
        if evaluation is not None
        else _field(context, "members", {})
    )
    calibration = _field(context, "calibration")
    scales = dict(options.get("scales", {}))
    if not scales and calibration is not None:
        calibration_points = _field(calibration, "points")
        calibration_members = _field(calibration, "members", {})
        if isinstance(calibration_points, pd.DataFrame) and isinstance(
            calibration_members, dict
        ):
            for variable in calibration_points.variable.dropna().unique():
                values = calibration_members.get("residual")
                if values is None:
                    continue
                mask = calibration_points.variable.to_numpy() == variable
                residual = (
                    np.asarray(values, float)[mask]
                    - calibration_points.observation.to_numpy(float)[mask, None]
                )
                scale = float(np.nanstd(residual))
                if np.isfinite(scale) and scale > 0:
                    scales[str(variable)] = scale
    options["scales"] = scales
    output_dir = Path(output_dir)
    if not isinstance(points, pd.DataFrame) or points.empty:
        return {
            "status": "blocked",
            "summary": "No matched points available.",
            "artifacts": [],
            "details": {},
        }
    points = points.reset_index(drop=True)
    seed = int(options.get("seed", 20261103))
    permutations = int(options.get("permutations", 5))
    rows, reasons = [], []
    method_items = list(members.items()) if isinstance(members, dict) else []
    if method_items and {key for key, _ in method_items}.issubset(
        {"temperature", "salinity"}
    ):
        method_items = [("residual", method_items[0][1])]
    for method, array in method_items:
        array = np.asarray(array, dtype=float)
        if array.shape[0] != len(points):
            reasons.append(f"{method}: member array is not point aligned")
            continue
        for variable in sorted(points.variable.dropna().unique()):
            for profile, truth, intact in _profile_matrix(points, array, variable):
                scale = options.get("scales", {}).get(variable, 1.0)
                scaled_truth, scaled_intact = truth / scale, intact / scale
                controls = [("intact", scaled_intact)]
                for repeat in range(permutations):
                    controls.append(
                        (
                            f"vertical_permutation_{repeat}",
                            _permute_vertical(
                                scaled_intact, np.random.default_rng(seed + repeat)
                            ),
                        )
                    )
                for label, ensemble in controls:
                    profile_depths = np.sort(
                        points[
                            (points.profile_id.astype(str) == profile)
                            & (points.variable == variable)
                        ].depth_m.to_numpy(float)
                    )
                    layer = _weighted_profile_mean(ensemble, profile_depths)
                    rows.append(
                        {
                            "method": method,
                            "variable": variable,
                            "profile_id": profile,
                            "control": label,
                            "energy_score": energy_score(scaled_truth, ensemble),
                            "variogram_score": variogram_score(scaled_truth, ensemble),
                            "layer_mean_crps": _crps(
                                float(
                                    np.trapezoid(scaled_truth, profile_depths)
                                    / (profile_depths[-1] - profile_depths[0])
                                ),
                                layer,
                            ),
                            "n_depths": truth.size,
                            "n_members": ensemble.shape[0],
                        }
                    )
        # Spatial controls require complete profiles sharing an explicit
        # patch/time identity; global permutation would invent coherence.
        for variable in sorted(points.variable.dropna().unique()):
            profiles = _profile_matrix(points, array, variable)
            groups = {}
            for item in profiles:
                rows_for_profile = points[points.profile_id.astype(str) == item[0]]
                if {"patch_index", "target_time"}.issubset(
                    points.columns
                ) and not rows_for_profile.empty:
                    key = (
                        rows_for_profile.iloc[0].patch_index,
                        rows_for_profile.iloc[0].target_time,
                    )
                    groups.setdefault(key, []).append(item)
            for key, grouped in groups.items():
                if (
                    len(grouped) < 2
                    or len({len(item[1]) for item in grouped}) != 1
                    or len(
                        {
                            tuple(
                                np.asarray(
                                    points[
                                        (points.profile_id.astype(str) == item[0])
                                    ].depth_m,
                                    float,
                                )
                            )
                            for item in grouped
                        }
                    )
                    != 1
                ):
                    continue
                truth_joint = np.concatenate([item[1] for item in grouped])
                intact_joint = np.concatenate([item[2] for item in grouped], axis=1)
                scale = options.get("scales", {}).get(variable, 1.0)
                rows.append(
                    {
                        "method": method,
                        "variable": variable,
                        "profile_id": f"spatial:{key}",
                        "control": "spatial_intact",
                        "energy_score": energy_score(
                            truth_joint / scale, intact_joint / scale
                        ),
                        "variogram_score": variogram_score(
                            truth_joint / scale, intact_joint / scale
                        ),
                        "layer_mean_crps": np.nan,
                        "n_depths": truth_joint.size,
                        "n_members": intact_joint.shape[0],
                    }
                )
                for repeat in range(permutations):
                    # Keep each physical location fixed and permute only its
                    # member axis. This preserves every site's marginal profile
                    # distribution while disrupting cross-site member pairing.
                    permuted_blocks = []
                    for site_index, item in enumerate(grouped):
                        site_rng = np.random.default_rng(
                            seed + 3000 + repeat + site_index
                        )
                        permuted_blocks.append(
                            item[2][site_rng.permutation(item[2].shape[0])]
                        )
                    permuted_joint = np.concatenate(permuted_blocks, axis=1)
                    rows.append(
                        {
                            "method": method,
                            "variable": variable,
                            "profile_id": f"spatial:{key}",
                            "control": f"spatial_case_{repeat}",
                            "energy_score": energy_score(
                                truth_joint / scale, permuted_joint / scale
                            ),
                            "variogram_score": variogram_score(
                                truth_joint / scale, permuted_joint / scale
                            ),
                            "layer_mean_crps": np.nan,
                            "n_depths": truth_joint.size,
                            "n_members": intact_joint.shape[0],
                        }
                    )
    # Preserve complete T profiles while breaking the T/S member pairing. This
    # is only activated when both variables share exact depth support.
    joint_available = {"temperature", "salinity"}.issubset(
        set(points.variable.dropna().unique())
    )
    if joint_available and not options.get("scales"):
        reasons.append(
            "joint T/S dependence blocked: calibration-derived scales are missing"
        )
    for method, array in method_items:
        if joint_available and options.get("scales"):
            profiles = set(points[points.variable == "temperature"].profile_id) & set(
                points[points.variable == "salinity"].profile_id
            )
            for profile in sorted(profiles):
                joint = _joint_profile(points, np.asarray(array, dtype=float), profile)
                if joint is None:
                    continue
                truth, intact, _ = joint
                scaled_truth = truth.copy()
                scaled_intact = intact.copy()
                scales = options.get("scales", {})
                scaled_truth[: len(truth) // 2] /= scales.get("temperature", 1.0)
                scaled_truth[len(truth) // 2 :] /= scales.get("salinity", 1.0)
                scaled_intact[:, : len(truth) // 2] /= scales.get("temperature", 1.0)
                scaled_intact[:, len(truth) // 2 :] /= scales.get("salinity", 1.0)
                shuffled = scaled_intact.copy()
                shuffled[:, len(truth) // 2 :] = shuffled[
                    np.random.default_rng(seed).permutation(shuffled.shape[0]),
                    len(truth) // 2 :,
                ]
                for label, ensemble in (
                    ("joint_intact", scaled_intact),
                    ("joint_ts_permutation", shuffled),
                ):
                    rows.append(
                        {
                            "method": method,
                            "variable": "temperature_salinity",
                            "profile_id": str(profile),
                            "control": label,
                            "energy_score": energy_score(scaled_truth, ensemble),
                            "variogram_score": variogram_score(scaled_truth, ensemble),
                            "layer_mean_crps": np.nan,
                            "n_depths": len(truth),
                            "n_members": ensemble.shape[0],
                        }
                    )
    if not rows:
        return {
            "status": "partial",
            "summary": "No complete profiles available for dependence scores.",
            "artifacts": [],
            "details": {"reasons": reasons},
        }
    detail = pd.DataFrame(rows)
    signatures = points.groupby("profile_id").depth_m.apply(
        lambda values: ",".join(f"{value:.3f}" for value in sorted(values))
    )
    detail["depth_signature"] = [
        signatures.get(row.profile_id, f"spatial:{row.n_depths}values:{row.profile_id}")
        for row in detail.itertuples()
    ]
    summary = detail.groupby(
        ["method", "variable", "control", "depth_signature"], as_index=False
    ).agg(
        n_profiles=("profile_id", "nunique"),
        energy_score=("energy_score", "mean"),
        variogram_score=("variogram_score", "mean"),
        layer_mean_crps=("layer_mean_crps", "mean"),
    )
    artifacts = save_table(detail, output_dir, "dependence_scores")
    artifacts += save_table(summary, output_dir, "dependence_summary")
    import matplotlib.pyplot as plt

    facets = list(
        detail.groupby(["variable", "depth_signature"], as_index=False)
        .profile_id.nunique()
        .sort_values("profile_id", ascending=False)
        .head(6)[["variable", "depth_signature"]]
        .itertuples(index=False, name=None)
    )
    figure, axes = plt.subplots(
        len(facets), 1, figsize=(8, 4 * len(facets)), squeeze=False
    )
    for axis, (variable, signature) in zip(axes[:, 0], facets):
        subset = summary[
            (summary.depth_signature == signature) & (summary.variable == variable)
        ]
        subset.pivot_table(
            index="control", columns="method", values="energy_score", aggfunc="mean"
        ).plot.bar(ax=axis)
        axis.set_title(f"{variable}; depth support: {signature}")
        axis.set_ylabel("Energy score (lower is better)")
    artifacts += save_figure(figure, output_dir, "dependence_energy_scores")
    return {
        "status": "complete" if not reasons else "partial",
        "summary": "Scored intact and vertical dependence controls.",
        "artifacts": artifacts,
        "details": {"n_profiles": int(detail.profile_id.nunique()), "reasons": reasons},
    }
