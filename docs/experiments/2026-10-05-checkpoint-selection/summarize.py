# Example:
# /work/envs/depth/bin/python docs/experiments/2026-10-05-checkpoint-selection/summarize.py \
#   --manifest docs/experiments/2026-10-05-checkpoint-selection/experiment.yaml \
#   --results outputs/checkpoint_selection_20261005/results_attempt02 \
#   --output-dir docs/experiments/2026-10-05-checkpoint-selection
"""Verify completed candidates and record their matched comparison results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from depth_recon.scripts.compare_checkpoints import file_sha256


def summarize(manifest_path, results, output_dir):
    """Reject mismatched/incomplete evidence before producing tables and figures."""
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest_sha = file_sha256(manifest_path)
    reference_selection, reference_batches = None, None
    summaries, statuses, evidence = {}, {}, {}
    for method in manifest["methods"]:
        folder = results / method
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "complete" or status["manifest_sha256"] != manifest_sha:
            raise ValueError(f"Incomplete or mismatched experiment: {method}")
        selection = json.loads((folder / "selection.json").read_text())
        batches = json.loads((folder / "batches.json").read_text())
        expected = (manifest["sample_count"] + manifest["batch_size"] - 1) // manifest[
            "batch_size"
        ]
        if (
            len(batches) != expected
            or len(selection["indices"]) != manifest["sample_count"]
        ):
            raise ValueError(f"Incomplete sample coverage: {method}")
        if reference_selection is None:
            reference_selection, reference_batches = selection, batches
        elif selection != reference_selection or any(
            first["input_sha256"] != second["input_sha256"]
            or first["profiles"] != second["profiles"]
            for first, second in zip(reference_batches, batches)
        ):
            raise ValueError(f"Different inputs/holdouts: {method}")
        summaries[method] = json.loads((folder / "summary.json").read_text())
        statuses[method] = status
        evidence[method] = {
            str(path): file_sha256(path) for path in sorted(folder.glob("*.json"))
        }
    first = next(iter(summaries.values()))
    baseline = {
        key: value
        for key, value in first.items()
        if key.startswith(("climatology/", "glorys/"))
    }
    for method, summary in summaries.items():
        if any(summary[key] != value for key, value in baseline.items()):
            raise ValueError(f"Different baseline references: {method}")
    table, all_metrics = {}, dict(baseline)
    for method, summary in summaries.items():
        for scope in ("field", "heldout_profiles"):
            all_metrics[f"{method}/{scope}"] = summary[f"{method}/{scope}"]
    for key, metrics in all_metrics.items():
        table[key] = {
            name: value for name, value in metrics.items() if name != "per_depth"
        }
    fields = {key: value for key, value in table.items() if key.endswith("/field")}
    ranking = sorted(fields, key=lambda key: fields[key]["equal_depth_rmse_c"])
    patch_wins, residual_diagnostics = {}, {}
    # Paired wins are descriptive; overlapping ocean patches are not independent trials.
    for method in manifest["methods"]:
        residual_sums = np.zeros(4, dtype=np.float64)
        wins, eligible = {"field": 0, "heldout_profiles": 0}, {
            "field": 0,
            "heldout_profiles": 0,
        }
        for path in sorted((results / method).glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                support = arrays["valid_mask"]
                correction = (
                    arrays["prediction"][support].astype(np.float64)
                    - arrays["climatology"][support]
                )
                anomaly = (
                    arrays["target"][support].astype(np.float64)
                    - arrays["climatology"][support]
                )
                residual_sums += [
                    np.square(correction).sum(),
                    np.square(anomaly).sum(),
                    (correction * anomaly).sum(),
                    int(support.sum()),
                ]
                for scope, target_key, mask_key in (
                    ("field", "target", "valid_mask"),
                    ("heldout_profiles", "profile_target", "heldout_mask"),
                ):
                    for prediction, climate, truth, mask in zip(
                        arrays["prediction"],
                        arrays["climatology"],
                        arrays[target_key],
                        arrays[mask_key],
                    ):
                        if mask.any():
                            eligible[scope] += 1
                            wins[scope] += int(
                                np.abs(prediction[mask] - truth[mask]).mean()
                                < np.abs(climate[mask] - truth[mask]).mean()
                            )
        patch_wins[method] = {
            scope: {
                "lower_pixel_mae_than_climatology": wins[scope],
                "eligible_patches": eligible[scope],
            }
            for scope in wins
        }
        correction_energy, anomaly_energy, alignment, count = residual_sums
        # Exact MSE decomposition describes the saved predictions; it does not
        # establish which training or sampling mechanism caused their errors.
        residual_diagnostics[method] = {
            "correction_rms_c": float(np.sqrt(correction_energy / count)),
            "true_anomaly_rms_c": float(np.sqrt(anomaly_energy / count)),
            "correction_energy_c2": float(correction_energy / count),
            "twice_correction_anomaly_product_c2": float(2 * alignment / count),
            "mse_increase_over_climatology_c2": float(
                (correction_energy - 2 * alignment) / count
            ),
        }
    hidden = sum(
        len(p["heldout_rows_cols"]) for b in reference_batches for p in b["profiles"]
    )
    report = {
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest_sha,
        "verified_identical_inputs_and_holdouts": True,
        "sample_count": manifest["sample_count"],
        "hidden_profile_locations_counted_per_patch": hidden,
        "field_ranking_by_equal_depth_rmse": ranking,
        "metrics": table,
        "paired_patch_wins": patch_wins,
        "residual_diagnostics": residual_diagnostics,
        "checkpoint_identity": manifest["methods"],
        "completion": {
            name: {
                key: status[key]
                for key in (
                    "started_at_utc",
                    "completed_at_utc",
                    "elapsed_seconds",
                    "weights",
                    "script_sha256",
                )
            }
            for name, status in statuses.items()
        },
        "evidence_sha256": evidence,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    colors = {
        "climatology": "black",
        "residual_early_ema": "#0072B2",
        "residual_early_standard": "#56B4E9",
        "residual_latest_best_ema": "#D55E00",
        "residual_latest_best_standard": "#E69F00",
    }
    for axis, scope in zip(axes, ("field", "heldout_profiles")):
        for name, color in colors.items():
            rows = all_metrics[f"{name}/{scope}"]["per_depth"]
            axis.plot(
                [r["mae_c"] for r in rows],
                [r["depth_m"] for r in rows],
                label=name,
                color=color,
            )
        axis.set_xlabel("MAE (°C)")
        axis.set_title(
            "GLORYS field" if scope == "field" else "Withheld gridded ARGO profiles"
        )
        axis.grid(alpha=0.2)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    axes[1].legend(fontsize=8)
    figure.suptitle("128 fixed 2016 patches · DDPM-1000 · 20% profile-location holdout")
    figure.tight_layout()
    figure.savefig(output_dir / "depth_comparison.png", dpi=160)
    plt.close(figure)
    print(yaml.safe_dump({"ranking": ranking, "metrics": table}, sort_keys=False))


def main():
    """Parse the recorded experiment and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.manifest, args.results, args.output_dir)


if __name__ == "__main__":
    main()
