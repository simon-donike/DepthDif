"""Paired, profile-balanced evidence by region, depth and provenance stratum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from val.metrics import _cluster_bootstrap, _holm, _profile_losses
from val.paper_common import point_crps, save_figure, save_table, stable_seed


def _strata(points: pd.DataFrame, options: dict):
    """Yield predefined geography and descriptive transfer/regime diagnostics."""
    work = points.copy()
    work["depth_band"] = pd.cut(
        work.depth_m,
        [-0.001, 50, 200, 700, 2000, np.inf],
        labels=["0-50m", "50-200m", "200-700m", "700-2000m", "2000m+"],
    ).astype(str)
    work["season"] = pd.to_datetime(
        work.observation_time, utc=True, format="mixed"
    ).dt.month.map(lambda month: ("DJF", "MAM", "JJA", "SON")[(month % 12) // 3])
    work["year"] = pd.to_datetime(
        work.observation_time, utc=True, format="mixed"
    ).dt.year.astype(str)
    for variable, group in work.groupby("variable", sort=True):
        yield {"variable": variable, "stratum": "all", "group": "all"}, group
        for column in (
            "source",
            "region",
            "depth_band",
            "season",
            "year",
            "independence",
            "case_id",
            "cruise",
            "deployment_id",
            "context_distance_band",
        ):
            if column in group:
                for label, subset in group.groupby(column, observed=True, sort=True):
                    yield {
                        "variable": variable,
                        "stratum": column,
                        "group": str(label),
                    }, subset
        for (region, depth), subset in group.groupby(
            ["region", "depth_band"], observed=True, sort=True
        ):
            yield {
                "variable": variable,
                "stratum": "region_depth",
                "group": f"{region}/{depth}",
            }, subset
        # Geographic polygons are frozen in the configuration before scoring.
        for region in options.get("regions", []):
            south, north = region["latitude"]
            west, east = region["longitude"]
            longitude = group.longitude
            longitude_match = (
                longitude.between(west, east)
                if west <= east
                else ((longitude >= west) | (longitude <= east))
            )
            subset = group[group.latitude.between(south, north) & longitude_match]
            if not subset.empty:
                yield {
                    "variable": variable,
                    "stratum": "predefined_region",
                    "group": region["name"],
                }, subset


def run(context: Any, output_dir: Path, options: dict | None = None) -> dict:
    """Estimate paired gains and retain all evidence gates next to the numbers."""
    options = dict(options or {})
    settings = {
        "bootstrap_replicates": 2000,
        "min_profiles": 30,
        "min_clusters": 20,
        **context.config.get("statistics", {}),
        **{
            key: options[key]
            for key in ("bootstrap_replicates", "min_profiles", "min_clusters")
            if key in options
        },
    }
    points = context.evaluation.points.reset_index(drop=True)
    arrays = context.evaluation.members
    truth = points.observation.to_numpy(float)
    deterministic = {
        "glorys_packaged": points.glorys.to_numpy(float),
        "climatology": points.climatology.to_numpy(float),
    }
    deterministic.update(
        {
            name.removeprefix("baseline_"): points[name].to_numpy(float)
            for name in points
            if name.startswith("baseline_")
        }
    )
    loss = {}
    for name, values in {**deterministic, **arrays}.items():
        values = np.asarray(values)
        mean = values.mean(axis=1) if values.ndim == 2 else values
        loss[name] = {
            "mae": np.abs(mean - truth),
            "rmse": (mean - truth) ** 2,
            "crps": (
                point_crps(values, truth) if values.ndim == 2 else np.abs(mean - truth)
            ),
        }
    rows = []
    for labels, frame in _strata(points, options):
        idx = frame.index.to_numpy()
        references = ["glorys_packaged"] + [
            name
            for name in ("glorys_bias", "glorys_calibrated", "glorys_native")
            if name in loss
        ]
        references += [
            name
            for name in arrays
            if name.startswith("baseline_oi_") and name.endswith("_full")
        ]
        for baseline in references:
            for metric in ("mae", "rmse", "crps"):
                model_loss, baseline_loss = (
                    loss["residual"][metric][idx],
                    loss[baseline][metric][idx],
                )
                valid = np.isfinite(model_loss) & np.isfinite(baseline_loss)
                if not valid.any():
                    continue
                profile = _profile_losses(
                    frame.loc[valid], model_loss[valid], baseline_loss[valid]
                )
                estimate = _cluster_bootstrap(
                    profile,
                    settings["bootstrap_replicates"],
                    stable_seed(str((labels, metric, baseline)), context.seed),
                    settings["min_profiles"],
                    settings["min_clusters"],
                    np.sqrt if metric == "rmse" else lambda value: value,
                )
                metadata = context.evaluation.metadata
                independent = (
                    frame.loc[valid].independence.eq("verified_not_assimilated").all()
                )
                gates = all(
                    bool(metadata.get(flag))
                    for flag in (
                        "checkpoint_verified",
                        "training_split_verified",
                        "confirmation_untouched",
                        "conditioning_exclusion_verified",
                    )
                )
                endpoint = {**labels, "metric": metric, "baseline": baseline}
                preregistered = endpoint in metadata.get("confirmatory_endpoints", [])
                eligible = (
                    gates
                    and independent
                    and frame.split_role.eq("confirmation").all()
                    and preregistered
                    and not metadata.get("demo", False)
                )
                rows.append(
                    {
                        **endpoint,
                        **estimate,
                        "confirmation_eligible": bool(eligible),
                        "verified_non_assimilated": bool(independent),
                        "predeclared_endpoint": preregistered,
                        "reference_scope": "native GLORYS required before delivered-product superiority claim",
                        "interpretation": "discovery diagnostic",
                    }
                )
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return {
            "status": "blocked",
            "summary": "No paired profile comparisons available",
            "artifacts": [],
            "details": {},
        }
    comparisons["holm_p_two_sided"] = np.nan
    # Correct the declared confirmation family separately from discovery scans.
    for eligible in (True, False):
        mask = (
            comparisons.confirmation_eligible.eq(eligible)
            & comparisons.p_two_sided.notna()
        )
        comparisons.loc[mask, "holm_p_two_sided"] = _holm(
            comparisons.loc[mask, "p_two_sided"].tolist()
        )
    benefit = (
        comparisons.confirmation_eligible
        & comparisons.holm_p_two_sided.lt(0.05)
        & comparisons.ci_high.lt(0)
    )
    comparisons.loc[benefit, "interpretation"] = (
        "predeclared independent improvement over packaged reference"
    )
    comparisons.loc[
        comparisons.inference_status.eq("insufficient_support"), "interpretation"
    ] = "insufficient independent clusters/profiles"
    artifacts = save_table(
        comparisons,
        output_dir,
        "paired_comparisons",
        "Paired profile-balanced differences: DepthDif minus reference; negative favors DepthDif. Confidence intervals resample whole clusters.",
    )
    coverage = []
    for region in options.get("regions", []):
        south, north = region["latitude"]
        west, east = region["longitude"]
        lon_mask = (
            points.longitude.between(west, east)
            if west <= east
            else ((points.longitude >= west) | (points.longitude <= east))
        )
        subset = points[points.latitude.between(south, north) & lon_mask]
        for variable in sorted(points.variable.unique()):
            selected = subset[subset.variable.eq(variable)]
            coverage.append(
                {
                    "region": region["name"],
                    "variable": variable,
                    "profiles": selected.profile_id.nunique(),
                    "clusters": selected.cluster_id.nunique(),
                    "points": len(selected),
                    "status": (
                        "no evaluation support"
                        if selected.empty
                        else "observation-sampled support"
                    ),
                }
            )
    if coverage:
        artifacts += save_table(
            pd.DataFrame(coverage),
            output_dir,
            "predefined_region_coverage",
            "All predefined regions, including regions without verification observations",
        )
    primary = comparisons[
        (comparisons.stratum == "all") & (comparisons.baseline == "glorys_packaged")
    ]
    artifacts += save_table(
        primary,
        output_dir,
        "primary_scorecard",
        "Whole-cohort profile scores, separated by variable",
    )
    fig, axes = plt.subplots(
        1,
        points.variable.nunique(),
        squeeze=False,
        figsize=(6 * points.variable.nunique(), 4),
    )
    for axis, (variable, frame) in zip(
        axes[0],
        comparisons[
            (comparisons.stratum == "region")
            & (comparisons.metric == "rmse")
            & (comparisons.baseline == "glorys_packaged")
        ].groupby("variable", sort=True),
    ):
        axis.axvline(0, color="black", linewidth=0.8)
        positions = np.arange(len(frame))
        axis.scatter(frame.difference, positions, color="tab:blue", zorder=3)
        for pos, row in zip(positions, frame.itertuples()):
            if np.isfinite(row.ci_low):
                axis.plot([row.ci_low, row.ci_high], [pos, pos], color="tab:blue")
        axis.set_yticks(positions, frame.group)
        axis.set(
            title=f"{variable}: regional RMSE", xlabel="DepthDif − packaged GLORYS"
        )
    fig.tight_layout()
    artifacts += save_figure(fig, output_dir, "regional_paired_improvement")
    correction = points[
        ["point_id", "profile_id", "cluster_id", "variable", "region", "depth_m"]
    ].copy()
    correction["glorys_error"] = points.glorys - points.observation
    correction["model_correction"] = arrays["residual"].mean(axis=1) - points.glorys
    correction["absolute_error_reduction"] = np.abs(correction.glorys_error) - np.abs(
        arrays["residual"].mean(axis=1) - truth
    )
    artifacts += save_table(
        correction,
        output_dir,
        "correction_diagnostics",
        "Diagnostic decomposition; verification-error strata are not deployable selection policies",
    )
    return {
        "status": "complete",
        "summary": "Paired regional, depth, seasonal and provenance comparisons with cluster uncertainty and evidence gates",
        "artifacts": artifacts,
        "details": {
            "comparisons": len(comparisons),
            "confirmatory_improvements": int(benefit.sum()),
            "area_weighting": "observation-sampled profiles, not global ocean area",
            "transfer_scope": "post-training cohort evaluation does not establish leave-region-out training generalization",
            "native_product_gate": "separate native-reference stage required",
        },
    }
