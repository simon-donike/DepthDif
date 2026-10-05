# Example:
# /work/envs/depth/bin/python docs/experiments/2026-10-05-checkpoint-selection/log_to_wandb.py \
#   --record-dir docs/experiments/2026-10-05-checkpoint-selection \
#   --entity esa-phi-lab --project DepthDif_Simon
"""Log a completed, verified comparison and its small provenance files to W&B."""

import argparse
from pathlib import Path

import wandb
import yaml


def main():
    """Create one evaluation run, retaining its ID to prevent accidental duplicates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    root = args.record_dir
    report = yaml.safe_load((root / "results.yaml").read_text())
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    if not report["verified_identical_inputs_and_holdouts"]:
        raise ValueError("Only verified comparisons may be logged.")
    identity_path = root / "wandb_run.yaml"
    if identity_path.exists():
        raise FileExistsError(
            f"An evaluation run is already recorded in {identity_path}."
        )
    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=manifest["experiment_id"],
        job_type="checkpoint-evaluation",
        group="checkpoint-selection-2016",
        tags=["evaluation-only", "residual", "ddpm-1000", "profile-holdout"],
        config=manifest,
        notes=manifest["purpose"],
    ) as run:
        identity_path.write_text(
            yaml.safe_dump({"id": run.id, "url": run.url, "status": "uploading"})
        )
        for key, metrics in report["metrics"].items():
            for name, value in metrics.items():
                if isinstance(value, (float, int)):
                    run.summary[f"{key}/{name}"] = value
        run.summary["field_ranking"] = report["field_ranking_by_equal_depth_rmse"]
        run.summary["inputs_verified_identical"] = True
        run.summary["sample_count"] = manifest["sample_count"]
        run.summary["hpc_reproduction_status"] = "pending_checkpoint_unavailable"
        run.log({"depth_comparison": wandb.Image(str(root / "depth_comparison.png"))})
        # Checkpoints stay local; hashes and source IDs provide exact provenance.
        artifact = wandb.Artifact(
            manifest["experiment_id"], type="checkpoint-evaluation"
        )
        for pattern in ("*.yaml", "*.md", "*.py", "*.png"):
            for path in sorted(root.glob(pattern)):
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        url, run_id = run.url, run.id
    identity_path.write_text(
        yaml.safe_dump({"id": run_id, "url": url, "status": "finished"})
    )
    print(url)


if __name__ == "__main__":
    main()
