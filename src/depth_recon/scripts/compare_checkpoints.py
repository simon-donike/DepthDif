# Example (run from the repository root; repeat with each declared method):
# /work/envs/depth/bin/python -m depth_recon.scripts.compare_checkpoints \
#   --manifest docs/experiments/2026-10-05-checkpoint-selection/experiment.yaml \
#   --method residual_early_ema --device cuda:0 \
#   --output-dir outputs/checkpoint_selection_20261005/results
"""Run a recorded, matched temperature checkpoint comparison without training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

from depth_recon.configs.config_resolver_pixel import load_pixel_training_config
from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    load_checkpoint_weights,
    to_device,
)
from depth_recon.scripts.evaluate_depth_baselines import _dataset_signature
from depth_recon.utils.normalizations import temperature_normalize
from depth_recon.utils.reconstruction_validation import _evaluation_rng


def file_sha256(path: str | Path) -> str:
    """Hash file contents without loading a checkpoint into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def holdout_profiles(batch: dict, indices: list[int], seed: int, fraction: float):
    """Hide entire observed grid columns, preserving at least one input column.

    These are gridded ARGO profile locations, not independent raw EN4 profiles.
    All depths at a chosen location are removed before inference. Patches with
    fewer than two observed locations remain in the field benchmark only.
    """
    if not 0 < fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one.")
    original = batch["x"].clone()
    support = batch["x_valid_mask"].bool()
    heldout = torch.zeros_like(support)
    locations = []
    for position, index in enumerate(indices):
        observed = torch.nonzero(support[position].any(dim=0), as_tuple=False)
        count = min(max(1, round(len(observed) * fraction)), len(observed) - 1)
        chosen = []
        if count > 0:
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            chosen = observed[rng.choice(len(observed), count, replace=False)].tolist()
            for row, col in chosen:
                heldout[position, :, row, col] = support[position, :, row, col]
        locations.append(
            dict(index=index, observed=len(observed), heldout_rows_cols=chosen)
        )
    # Clear values and both mask representations; no hidden input reaches the model.
    batch["x"] = batch["x"].masked_fill(heldout, 0)
    batch["x_valid_mask"] = support & ~heldout
    batch["x_valid_mask_1d"] = batch["x_valid_mask"].any(dim=1, keepdim=True)
    return original, heldout, locations


def error_statistics(prediction, target, support):
    """Pool per-depth error sums; reject invalid predictions on scored support."""
    support = np.broadcast_to(np.asarray(support, dtype=bool), target.shape)
    if not np.isfinite(target[support]).all():
        raise ValueError("Nonfinite reference on scored support.")
    if not np.isfinite(prediction[support]).all():
        raise ValueError("Nonfinite prediction on scored support.")
    error = np.where(support, prediction.astype(np.float64) - target, 0)
    return np.stack(
        [
            np.square(error).sum((0, 2, 3)),
            np.abs(error).sum((0, 2, 3)),
            error.sum((0, 2, 3)),
            support.sum((0, 2, 3)),
        ]
    )


def summarize_statistics(stats, depths):
    """Report pooled and equally weighted depth errors, without NaN JSON values."""
    squared, absolute, signed, count = stats
    valid = count > 0
    if not valid.any():
        return {"valid_values": 0, "supported_depths": 0}
    mae = np.divide(absolute, count, out=np.zeros_like(count), where=valid)
    rmse = np.sqrt(np.divide(squared, count, out=np.zeros_like(count), where=valid))
    result = {
        "equal_depth_rmse_c": float(rmse[valid].mean()),
        "equal_depth_mae_c": float(mae[valid].mean()),
        "pixel_mae_c": float(absolute.sum() / count.sum()),
        "pixel_rmse_c": float(np.sqrt(squared.sum() / count.sum())),
        "pixel_bias_c": float(signed.sum() / count.sum()),
        "valid_values": int(count.sum()),
        "supported_depths": int(valid.sum()),
        "per_depth": [
            {
                "depth_m": float(depth),
                "count": int(count[i]),
                "mae_c": float(mae[i]) if valid[i] else None,
                "rmse_c": float(rmse[i]) if valid[i] else None,
            }
            for i, depth in enumerate(depths)
        ],
    }
    return result


def _tensor_digest(batch):
    """Fingerprint all tensor inputs and references for cross-method verification."""
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_comparison(
    manifest_path: Path, method_name: str, device: str, output_dir: Path
):
    """Evaluate one declared weight variant, saving provenance and every batch."""
    manifest = yaml.safe_load(manifest_path.read_text())
    method = manifest["methods"][method_name]
    output = output_dir / method_name
    # Existing results are evidence: never silently replace or combine experiments.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {
        "method": method_name,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_sha256(manifest_path),
        "script_sha256": file_sha256(__file__),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": device,
        "torch_version": torch.__version__,
    }
    status_path = output / "status.json"
    status_path.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        for label, path, expected in (
            ("checkpoint", method["checkpoint"], method["checkpoint_sha256"]),
            ("config", method["config"], method["config_sha256"]),
            (
                "climatology",
                manifest["climatology"]["path"],
                manifest["climatology"]["sha256"],
            ),
        ):
            if file_sha256(path) != expected:
                raise ValueError(f"{label} contents no longer match the manifest.")
        torch.set_num_threads(4)
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        bundle = load_pixel_training_config(
            config_path_value=method["config"],
            runtime_config_dir=output / "effective",
            write_snapshots=True,
        )
        if (
            bundle.scenario != "temperature"
            or bundle.data_cfg["split"]["val_year"] != 2016
        ):
            raise ValueError(
                "This comparison requires temperature and the 2016 holdout."
            )
        sampling = bundle.training_cfg["training"]["validation_sampling"]
        if (
            sampling["sampler"] != "ddpm"
            or bundle.training_cfg["training"]["noise"]["num_timesteps"] != 1000
        ):
            raise ValueError("All methods must use DDPM-1000.")
        dataset = build_dataset(
            bundle.effective_data_config_path,
            bundle.data_cfg.get("dataset", {}),
            split="val",
        )
        seed, count, batch_size = (
            int(manifest[key]) for key in ("seed", "sample_count", "batch_size")
        )
        indices = (
            np.random.default_rng(seed)
            .choice(len(dataset), size=count, replace=False)
            .tolist()
        )
        rows = _dataset_signature(dataset, indices)
        if rows is None or any(int(row[0]) // 10000 != 2016 for row in rows):
            raise ValueError("Selected rows must have recorded identities in 2016.")
        selection = {
            "dataset_size": len(dataset),
            "indices": indices,
            "rows": rows,
            "seed": seed,
            "batch_size": batch_size,
        }
        (output / "selection.json").write_text(
            json.dumps(selection, indent=2, default=int) + "\n"
        )
        datamodule = build_datamodule(
            dataset=dataset, data_cfg=bundle.data_cfg, training_cfg=bundle.training_cfg
        )
        model = build_model(
            model_config_path=bundle.effective_model_config_path,
            data_config_path=bundle.effective_data_config_path,
            training_config_path=bundle.effective_training_config_path,
            model_cfg=bundle.model_cfg,
            datamodule=datamodule,
        )
        weights = load_checkpoint_weights(
            model,
            method["checkpoint"],
            strict=True,
            prefer_ema=method["weights"] == "ema",
        )
        if weights != method["weights"]:
            raise ValueError(
                f"Requested {method['weights']} weights, loaded {weights}."
            )
        model.to(resolved_device).eval()
        depths = np.asarray(dataset.depth_axis_m)
        totals, batches = {}, []
        for batch_index, start in enumerate(range(0, count, batch_size)):
            batch_started = time.monotonic()
            selected = indices[start : start + batch_size]
            with _evaluation_rng(resolved_device, seed + batch_index):
                batch = default_collate([dataset[index] for index in selected])
                observed, heldout, locations = holdout_profiles(
                    batch, selected, seed, manifest["holdout_fraction"]
                )
                denorm = lambda tensor: temperature_normalize("denorm", tensor).numpy()
                target = denorm(batch["y"])
                profile_target = denorm(observed)
                climate = denorm(batch["climatology"])
                support = batch["y_valid_mask"].numpy().astype(bool)
                ocean = batch["land_mask"].numpy() > 0.5
                support &= np.broadcast_to(ocean, support.shape)
                profile_support = heldout.numpy() & support
                digest = _tensor_digest(batch)
                with torch.inference_mode():
                    predicted = model.predict_step(
                        to_device(batch, resolved_device), batch_idx=batch_index
                    )
                prediction = (
                    predicted["y_hat_temperature_denorm"].detach().float().cpu().numpy()
                )
                del predicted
            scores = {}
            for name, values in (
                (method_name, prediction),
                ("climatology", climate),
                ("glorys", target),
            ):
                for scope, truth, mask in (
                    ("field", target, support),
                    ("heldout_profiles", profile_target, profile_support),
                ):
                    if name == "glorys" and scope == "field":
                        continue
                    key = name + "/" + scope
                    stats = error_statistics(values, truth, mask)
                    totals[key] = totals.get(key, np.zeros_like(stats)) + stats
                    scores[key] = summarize_statistics(stats, depths)
            np.savez_compressed(
                output / f"batch_{batch_index:03d}.npz",
                prediction=prediction,
                climatology=climate,
                target=target,
                valid_mask=support,
                profile_target=profile_target,
                heldout_mask=profile_support,
                sample_indices=selected,
                depth_axis_m=depths,
            )
            batch_record = {
                "batch": batch_index,
                "input_sha256": digest,
                "profiles": locations,
                "scores": scores,
            }
            batches.append(batch_record)
            (output / "batches.json").write_text(
                json.dumps(batches, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"{method_name}: batch {batch_index + 1}/{(count + batch_size - 1) // batch_size} in {time.monotonic() - batch_started:.1f}s",
                flush=True,
            )
        summary = {
            key: summarize_statistics(stats, depths) for key, stats in totals.items()
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        metadata.update(
            status="complete",
            weights=weights,
            checkpoint=method,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
        )
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    """Run one method from an explicit, checksum-pinned experiment manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_comparison(args.manifest, args.method, args.device, args.output_dir)


if __name__ == "__main__":
    main()
