# Example: /work/envs/depth/bin/python -m val.score_all_profiles --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/temperature_parallel_20261117/italian_xbt_temperature/paper_all_profiles
"""Score all saved Italian casts with fixed methods and no observation calibration."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd
import yaml

from val.export_dashboard import include_method, overall_scores
from val.paper import STAGES, _json, load_bundle, make_context
from val.run import _config


def score(cohort: Path, output: Path) -> None:
    """Create a separate all-cast report while preserving the original split report."""
    cohort, output = cohort.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Report already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(cohort / "bundle")
    config = _config(cohort / "config.yaml")
    if config["cohort"]["kind"] != "external":
        raise ValueError("All-cast report requires an external observation cohort")
    config["paper"]["calibration_fraction"] = 0
    config["paper"]["calibration_bundle"] = None
    # Saved predictions are reused exactly; no calibration stage may add fitted methods.
    bundle.members = {
        name: values for name, values in bundle.members.items() if include_method(name)
    }
    bundle.points = bundle.points[
        [name for name in bundle.points if include_method(name)]
    ]
    context = make_context(bundle, config, output)
    if context.calibration is not None or len(context.evaluation.points) != len(
        bundle.points
    ):
        raise ValueError("All-cast scoring must retain every point without calibration")
    (output / "analysis_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    records = []
    for order, name in enumerate(STAGES, 1):
        directory = output / f"{order:02d}_{name}"
        directory.mkdir()
        if name in {"reference", "comparators", "regional", "conditioning"}:
            print(f"Scoring all profiles: {name}", flush=True)
            options = dict(config["paper"].get(name, {}))
            options.setdefault("seed", context.seed)
            result = import_module(f"val.techniques.{name}").run(
                context, directory, options
            )
            if result["status"] == "failed":
                raise ValueError(f"All-cast {name} failed")
        else:
            result = {
                "status": "blocked",
                "summary": "Not included in this fixed-method comparison.",
                "artifacts": [],
                "details": {},
            }
        record = {"order": order, "technique": name, **result}
        records.append(record)
        _json(directory / "status.json", record)
        _json(output / "stages.json", records)
    paired = pd.read_csv(output / "04_regional/paired_comparisons.csv")
    missions = []
    for mission, rows in paired[paired.stratum.eq("cruise")].groupby("group"):
        scores = overall_scores(rows.assign(stratum="all", group="all"))
        scores.insert(0, "mission", mission)
        missions.append(scores)
    mission_scores = pd.concat(missions, ignore_index=True)
    mission_scores.to_csv(output / "mission_scores.csv", index=False)
    mission_scores.to_latex(
        output / "mission_scores.tex", index=False, float_format="%.4f"
    )
    _json(
        output / "all_profiles.json",
        {
            "complete": True,
            "policy": "All saved Italian profiles; fixed predictions. Original GLORYS is the reference.",
            "points": len(bundle.points),
            "profiles": int(bundle.points.profile_id.nunique()),
            "missions": int(bundle.points.cluster_id.nunique()),
            "bundle_sha256": bundle.metadata["bundle_sha256"],
        },
    )
    print(
        f"Scored {bundle.points.profile_id.nunique()} profiles without calibration",
        flush=True,
    )


def main() -> None:
    """Parse the saved cohort and a new report directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.cohort, args.output)


if __name__ == "__main__":
    main()
