# Example: /work/envs/depth/bin/python -m val.export_dashboard --campaign val/results/temperature_parallel_20261117 --output docs/assets/validation
"""Export a portable, evaluation-only website snapshot without rerunning inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from val.paper import STAGES, load_bundle
from val.paper_common import method_arrays

COHORTS = {
    "italian_xbt_temperature": "Italian XBT · 2016",
    "en4_temperature": "EN4 · 2016",
}


def include_method(name: str) -> bool:
    """Exclude adjusted GLORYS references and climatology optimal interpolation."""
    return not str(name).startswith(
        (
            "glorys_bias",
            "glorys_calibrated",
            "baseline_glorys_bias",
            "baseline_oi_glorys",
            "baseline_oi_climatology",
        )
    )


def published_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested comparator selection without changing retained scores."""
    for column in ("method", "baseline"):
        if column in frame:
            frame = frame[frame[column].map(include_method)]
    return frame


def export_filtered_figure(source: Path, target: Path) -> None:
    """Rebuild comparison figures from filtered tables, preserving source evidence."""
    import matplotlib.pyplot as plt

    if source.stem == "depth_accuracy_temperature":
        frame = published_rows(pd.read_csv(source.parent / "depth_scores.csv"))
        figure, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 6))
        for method, group in frame.groupby("method"):
            group = group.sort_values("depth_m")
            for axis, metric in zip(axes, ("mae", "rmse", "bias")):
                axis.plot(group[metric], group.depth_m, label=method)
                axis.set_xlabel(f"{metric.upper()} (°C)")
                axis.grid(alpha=0.2)
        axes[0].set_ylabel("Depth (m)")
        axes[0].invert_yaxis()
        axes[-1].legend(fontsize=7)
    else:
        frame = published_rows(pd.read_csv(source.parent / "conditioning_scores.csv"))
        figure, axis = plt.subplots(figsize=(6.5, 4))
        # Numeric method names are the DepthDif retained-profile budgets.
        numeric = frame[frame.method.astype(str).str.isdigit()].copy()
        numeric["budget"] = pd.to_numeric(numeric.budget)
        numeric = numeric.sort_values("budget")
        axis.plot(numeric.budget, numeric.mae, marker="o", label="DepthDif")
        full = frame[frame.method.eq("full")]
        if not full.empty:
            axis.axhline(
                full.mae.iloc[0], linestyle="--", color="black", label="full context"
            )
        axis.set(
            xlabel="Number of retained input profiles", ylabel="Profile-balanced MAE"
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(target)
    plt.close(figure)


def records(frame: pd.DataFrame) -> list[dict]:
    """Convert pandas missing values to standards-compliant JSON nulls."""
    return json.loads(frame.to_json(orient="records", double_precision=12))


def evaluation_points(points: pd.DataFrame, partition: pd.DataFrame) -> pd.DataFrame:
    """Require a complete, disjoint partition before publishing any observations."""
    if partition.point_id.duplicated().any() or set(partition.point_id) != set(
        points.point_id
    ):
        raise ValueError("Partition must assign each bundle point exactly once")
    if not set(partition.partition).issubset({"evaluation", "calibration"}):
        raise ValueError("Unknown partition label")
    joined = points.merge(
        partition[["point_id", "partition"]], on="point_id", validate="one_to_one"
    )
    for key in ("profile_id", "cluster_id"):
        if joined.groupby(key).partition.nunique().gt(1).any():
            raise ValueError(f"Calibration/evaluation overlap in {key}")
    selected = points[
        points.point_id.isin(
            partition.loc[partition.partition.eq("evaluation"), "point_id"]
        )
    ]
    if selected.empty:
        raise ValueError("Evaluation partition is empty")
    return selected


def overall_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """Reuse the paper's profile-balanced aggregate scores without redefining RMSE."""
    whole = paired[paired.stratum.eq("all") & paired.group.eq("all")]
    baseline = whole.pivot(index="baseline", columns="metric", values="baseline_score")
    model = whole.groupby("metric").model_score.first()
    # Every comparator must use the same model support for an honest shared table.
    if whole.groupby("metric").model_score.nunique().gt(1).any():
        raise ValueError("Comparators have different model support")
    baseline.loc["residual"] = model
    baseline.index.name = "method"
    return baseline.reset_index().sort_values("mae")


def export_cohort(root: Path, destination: Path, run_status: str) -> dict:
    """Export completed tables, all evaluation profiles and a scientific artifact pack."""
    root = root.resolve()
    cohort = {"id": root.name, "label": COHORTS[root.name], "run_status": run_status}
    paper = root / "paper"
    all_profiles = root / "paper_all_profiles/all_profiles.json"
    policy = json.loads(all_profiles.read_text()) if all_profiles.exists() else None
    if policy and policy.get("complete"):
        paper = all_profiles.parent
        policy = {
            **policy,
            "policy": "All saved Italian profiles; fixed predictions. Original GLORYS is the reference.",
        }
    completed = (
        json.loads((paper / "stages.json").read_text())
        if (paper / "stages.json").exists()
        else []
    )
    by_name = {stage["technique"]: stage for stage in completed}
    cohort["stages"] = [
        {
            "name": name,
            "status": by_name.get(name, {}).get("status", "pending"),
            "summary": by_name.get(name, {})
            .get("summary", "Not available in this snapshot.")
            .replace("; no calibration is fitted", ""),
        }
        for name in STAGES
        if name != "baselines"
    ]
    membership_path = root / "profile_membership.json"
    if membership_path.exists():
        cohort["membership"] = json.loads(membership_path.read_text())
        cohort["membership"]["registry"] = Path(cohort["membership"]["registry"]).name
    if by_name.get("regional", {}).get("status") != "complete":
        cohort["available"] = False
        return cohort
    if cohort.get("membership", {}).get("status") != "passed":
        raise ValueError("Publication requires passed profile membership screening")
    bundle = load_bundle(root / "bundle")
    if policy and policy["bundle_sha256"] != bundle.metadata["bundle_sha256"]:
        raise ValueError("All-profile report differs from the saved inference bundle")
    if bundle.metadata.get("demo"):
        raise ValueError("Demo bundles cannot be published as campaign results")
    pilot = bundle.metadata.get("pilot")
    if pilot:
        # Describe the active evaluation without advertising unused analysis stages.
        pilot = {
            **pilot,
            "policy": "Preliminary coverage-selected EN4 subset; all retained points evaluated. All source holdouts remain masked. One generation, original DDPM steps and original residual batch seeds. Not an ocean-area representative sample or proof of non-assimilation.",
        }
    points = evaluation_points(bundle.points, pd.read_csv(paper / "partition.csv"))
    if policy and (
        len(points) != len(bundle.points) or policy["points"] != len(points)
    ):
        raise ValueError("All-profile report must include every saved observation")
    if set(points.variable) != {"temperature"}:
        raise ValueError("This dashboard expects temperature-only cohorts")
    destination.mkdir(parents=True, exist_ok=True)
    paired = published_rows(pd.read_csv(paper / "04_regional/paired_comparisons.csv"))
    overall = overall_scores(paired)
    depth = published_rows(pd.read_csv(paper / "04_regional/depth_scores.csv"))
    cohort.update(
        {
            "available": True,
            "counts": {
                "profiles": points.profile_id.nunique(),
                "points": len(points),
                "clusters": points.cluster_id.nunique(),
                "patches": (
                    int(points.patch_index.nunique())
                    if "patch_index" in points
                    else None
                ),
                "depths": int(points.depth_m.nunique()),
            },
            "partition": records(pd.read_csv(paper / "partition_summary.csv")),
            "evaluation_policy": policy,
            "pilot": pilot,
            "missions": records(
                points.groupby("cluster_id")
                .agg(
                    profiles=("profile_id", "nunique"),
                    points=("point_id", "size"),
                    start=("observation_time", "min"),
                    end=("observation_time", "max"),
                )
                .reset_index()
                .rename(columns={"cluster_id": "mission"})
            ),
            "mission_scores": (
                records(published_rows(pd.read_csv(paper / "mission_scores.csv")))
                if (paper / "mission_scores.csv").exists()
                else []
            ),
            "overall": records(overall),
            "depth": records(depth),
            "paired": records(
                paired[
                    paired.stratum.isin(
                        ["all", "depth_band", "region", "predefined_region"]
                    )
                ]
            ),
            "conditioning": (
                records(
                    published_rows(
                        pd.read_csv(paper / "05_conditioning/conditioning_scores.csv")
                    )
                )
                if (paper / "05_conditioning/conditioning_scores.csv").is_file()
                else []
            ),
            "model": {
                key: bundle.metadata.get(key)
                for key in (
                    "checkpoint_sha256",
                    "config_sha256",
                    "weight_variant",
                    "members",
                    "seed",
                    "bundle_sha256",
                )
            },
            "baselines": [
                {
                    "method": name,
                    **{
                        key: info.get(key)
                        for key in ("status", "invalid_points", "checkpoint_sha256")
                    },
                }
                for name, info in bundle.metadata.get("baselines", {}).items()
                if include_method(name)
            ],
        }
    )
    # Preserve the bundle row indices when selecting aligned member arrays.
    arrays = method_arrays(
        points, {name: values[points.index] for name, values in bundle.members.items()}
    )
    arrays = {name: values for name, values in arrays.items() if include_method(name)}
    examples = points.copy()
    for name, values in arrays.items():
        examples[name] = values.mean(axis=1) if values.ndim == 2 else values
    series = ["observation", *arrays]
    profiles = []
    profile_dir = destination / "profiles"
    profile_dir.mkdir(exist_ok=True)
    for number, (profile_id, frame) in enumerate(
        examples.groupby("profile_id", sort=True)
    ):
        frame = frame.sort_values("depth_m")
        first = frame.iloc[0]
        relative = f"profiles/{number:04d}.json"
        # Each profile is lazy-loaded at native depths; no display subsampling.
        (destination / relative).write_text(
            json.dumps(
                frame[["depth_m", *series]].round(10).to_dict(orient="list"),
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        profiles.append(
            {
                "id": profile_id,
                "file": relative,
                "latitude": float(first.latitude),
                "longitude": float(first.longitude),
                "time": str(first.observation_time),
                "points": len(frame),
                "cluster": str(first.cluster_id),
                "mission": str(first.get("cruise", first.cluster_id)),
            }
        )
    cohort["profiles"] = profiles
    if (destination / "spatial/manifest.json").is_file():
        cohort["spatial"] = "spatial/manifest.json"
    if (
        root.name.startswith("italian")
        and (destination / "ensemble/metrics.json").is_file()
    ):
        ensemble = json.loads((destination / "ensemble/metrics.json").read_text())
        if ensemble["profiles"] != len(profiles) or ensemble["points"] != len(points):
            raise ValueError("Ensemble publication support differs from cohort")
        if any(
            item["source_bundle_sha256"] != bundle.metadata["bundle_sha256"]
            for item in ensemble["provenance"]
        ):
            raise ValueError("Ensemble publication source differs from cohort")
        # Refresh saved metric exports without recomputing the ensemble.
        for key in ("overall", "depth", "missions", "spatial", "spectra"):
            ensemble[key] = [
                row for row in ensemble[key] if include_method(row["method"])
            ]
        (destination / "ensemble/metrics.json").write_text(
            json.dumps(ensemble, indent=2) + "\n"
        )
        for filename in ensemble["downloads"]:
            target = destination / "ensemble" / filename
            if target.suffix == ".csv":
                frame = pd.read_csv(target)
                filtered = published_rows(frame)
                if len(filtered) != len(frame):
                    filtered.to_csv(target, index=False)
        cohort["ensemble"] = "ensemble/metrics.json"
    overall.to_csv(destination / "overall_scores.csv", index=False)
    overall.to_latex(
        destination / "overall_scores.tex", index=False, float_format="%.4f"
    )
    cohort["downloads"] = []
    if cohort.get("ensemble"):
        for filename in ["metrics.json", *ensemble["downloads"]]:
            if Path(filename).name != filename:
                raise ValueError("Ensemble download must be a local filename")
            target = destination / "ensemble" / filename
            cohort["downloads"].append(
                {
                    "name": filename,
                    "file": str(Path("ensemble") / filename),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    # Only completed-stage artifacts are copied, so active stages cannot leak partial files.
    candidates = [paper / "partition_summary.csv", root / "profile_membership.csv"]
    candidates.extend(
        path
        for path in (paper / "mission_scores.csv", paper / "mission_scores.tex")
        if path.exists()
    )
    for stage in completed:
        candidates.extend(Path(value) for value in stage.get("artifacts", []))
    for source in sorted(set(candidates)):
        source = source.resolve()
        if not source.is_file() or source.suffix not in {
            ".csv",
            ".tex",
            ".pdf",
            ".svg",
            ".png",
        }:
            continue
        # Large pointwise diagnostic tables stay in the campaign; summaries are portable.
        if source.stat().st_size > 2_000_000:
            continue
        relative = Path("artifacts") / source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in {".csv", ".tex"} and source.with_suffix(".csv").exists():
            frame = published_rows(pd.read_csv(source.with_suffix(".csv")))
            if source.suffix == ".csv":
                frame.to_csv(target, index=False)
            else:
                frame.to_latex(target, index=False, float_format="%.4f")
        elif source.stem in {"depth_accuracy_temperature", "conditioning_curve"}:
            export_filtered_figure(source, target)
        else:
            shutil.copyfile(source, target)
        cohort["downloads"].append(
            {
                "name": source.name,
                "file": str(relative),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    # Retired public copies must not survive a refresh with a different report.
    published_files = {destination / item["file"] for item in cohort["downloads"]}
    for obsolete in (destination / "artifacts").rglob("*"):
        if obsolete.is_file() and obsolete not in published_files:
            obsolete.unlink()
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "model": cohort["model"],
                "membership": cohort["membership"],
                "baselines": cohort["baselines"],
                "stages": cohort["stages"],
                "pilot": cohort["pilot"],
                "artifacts": cohort["downloads"],
            },
            indent=2,
        )
        + "\n"
    )
    with zipfile.ZipFile(
        destination / "paper_artifacts.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for item in cohort["downloads"]:
            archive.write(destination / item["file"], item["file"])
        for name in ("overall_scores.csv", "overall_scores.tex", "provenance.json"):
            archive.write(destination / name, name)
    return cohort


def export(campaign: Path, output: Path) -> dict:
    """Refresh both cohorts and mark unfinished cohorts pending rather than invent scores."""
    output.mkdir(parents=True, exist_ok=True)
    statuses = json.loads((campaign / "status.json").read_text())
    by_name = {item["cohort"]: item["status"] for item in statuses["cohorts"]}
    result = {
        "campaign": campaign.name,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "cohorts": [
            export_cohort(campaign / name, output / name, by_name.get(name, "pending"))
            for name in COHORTS
        ],
    }
    # Swap the manifest last: readers never see links to files not yet written.
    temporary = output / "campaign.tmp"
    temporary.write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(output / "campaign.json")
    return result


def main() -> None:
    """Parse explicit campaign and website destination paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.campaign, args.output)
    print(f"Exported {len(result['cohorts'])} cohorts to {args.output}")


if __name__ == "__main__":
    main()
