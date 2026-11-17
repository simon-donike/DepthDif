# Example (run workers concurrently): CUDA_VISIBLE_DEVICES=0 PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/italian_ensemble_20261117 --shard 0 --members 20
# Second GPU: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --output val/results/italian_ensemble_20261117 --shard 1 --members 20
"""Generate durable Italian XBT ensembles with original conditioning and batch seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    load_checkpoint_weights,
    to_device,
)
from depth_recon.utils.normalizations import temperature_normalize
from depth_recon.utils.reconstruction_validation import _evaluation_rng
from val.baselines import check_artifact
from val.export_spatial import verify_predictions
from val.inference import (
    _batch_fingerprint,
    _clone_batch,
    _physical_output,
    _prediction_points,
)
from val.observations import ExternalObservationCohort, _grid_transform
from val.paper import _json, load_bundle
from val.run import _batches, _bundle, _config, _sha256


def worker(cohort_root: Path, output: Path, shard: int, members: int) -> None:
    """Split original batches across two GPUs; checkpoint each complete sample."""
    if members < 2 or shard not in (0, 1):
        raise ValueError("Require at least two members and shard 0 or 1")
    torch.set_num_threads(4)
    saved = load_bundle(cohort_root / "bundle")
    config = _config(cohort_root / "config.yaml")
    if (
        config["cohort"]["kind"] != "external"
        or not saved.points.profile_id.str.startswith("italian-xbt:").all()
    ):
        raise ValueError("This runner is restricted to the Italian XBT cohort")
    runtime = output / f"worker_{shard}"
    runtime.mkdir(parents=True, exist_ok=True)
    resolved = _bundle(config, runtime)
    dataset = build_dataset(
        resolved.effective_data_config_path, resolved.data_cfg["dataset"], split="val"
    )
    assignments = pd.read_csv(cohort_root / "cohort.csv")
    holdouts = json.loads((cohort_root / "holdouts.json").read_text())
    cohort = ExternalObservationCohort(
        dataset,
        assignments,
        sorted(assignments.patch_index.unique()),
        {tuple(item) for item in holdouts["locations"]},
        holdouts["metadata"],
        pd.DataFrame(),
    )
    batches = list(_batches(dataset, cohort, config, []))
    if len(batches) != len(saved.metadata["batch_records"]):
        raise ValueError("Original batch count changed")
    for number, (batch, _) in enumerate(batches):
        if (
            _batch_fingerprint(batch)
            != saved.metadata["batch_records"][number]["input_sha256"]
        ):
            raise ValueError(f"Conditioning changed in batch {number}")
    spec = config["model"]
    check_artifact(spec["checkpoint"], spec["checkpoint_sha256"])
    datamodule = build_datamodule(dataset, resolved.data_cfg, resolved.training_cfg)
    model = build_model(
        resolved.effective_model_config_path,
        resolved.effective_data_config_path,
        resolved.effective_training_config_path,
        resolved.model_cfg,
        datamodule,
    )
    loaded = load_checkpoint_weights(
        model,
        Path(spec["checkpoint"]),
        strict=True,
        prefer_ema=spec["weights"] == "ema",
    )
    if loaded != spec["weights"]:
        raise ValueError("Checkpoint weight selection changed")
    device = torch.device("cuda:0")
    model.to(device).eval()
    transform, _, _ = _grid_transform(dataset)
    for number, (batch, points) in enumerate(batches):
        if number % 2 != shard:
            continue
        directory = output / f"batch_{number:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = "y_glorys" if "y_glorys" in batch else "y"
        valid = (
            batch[target + "_valid_mask"].bool() & batch["land_mask"].bool()
        ).numpy()
        patches = [int(p) for p in points.patch_index.unique()]
        geography = []
        for patch in patches:
            row = dataset._rows.iloc[patch]
            _, latitude = transform * (
                int(row.grid_x0) + valid.shape[-1] / 2,
                int(row.grid_y0) + valid.shape[-2] / 2,
            )
            geography.append(
                {
                    "patch": patch,
                    "batch_index": int(
                        points.loc[points.patch_index.eq(patch), "batch_index"].iloc[0]
                    ),
                    "latitude": latitude,
                    "dx_km": abs(transform.a) * 111.32 * np.cos(np.deg2rad(latitude)),
                    "dy_km": abs(transform.e) * 111.32,
                }
            )
        manifest = {
            "batch": number,
            "members": members,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "config_sha256": _sha256(cohort_root / "config.yaml"),
            "source_bundle_sha256": saved.metadata["bundle_sha256"],
            "input_sha256": _batch_fingerprint(batch),
            "depths_m": np.asarray(dataset.depth_axis_m).tolist(),
            "geography": geography,
            "sampler": config["sampling"],
            "seed_rule": "original seed + original batch * 100000 + member",
            "gpu_shard": shard,
        }
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing ensemble has different provenance")
        _json(manifest_path, manifest)
        points.to_csv(directory / "assignments.csv", index=False)
        np.savez_compressed(
            directory / "reference.npz",
            glorys=temperature_normalize("denorm", batch[target]).numpy(),
            climatology=temperature_normalize("denorm", batch["climatology"]).numpy(),
            valid=valid,
        )
        for member in range(members):
            destination = directory / f"member_{member:03d}.npz"
            if destination.exists() and destination.with_suffix(".json").exists():
                record = json.loads(destination.with_suffix(".json").read_text())
                if record["sha256"] != _sha256(destination):
                    raise ValueError("Existing sample checksum differs")
                continue
            started = time.monotonic()
            seed = int(config["sampling"]["seed"]) + number * 100_000 + member
            with _evaluation_rng(device, seed), torch.inference_mode():
                result = model.predict_step(
                    to_device(_clone_batch(batch), device), batch_idx=number
                )
                values = _physical_output(result, "temperature").detach().float().cpu()
                del result
            if not np.isfinite(values.numpy()[valid]).all():
                raise ValueError("Nonfinite generated ocean values")
            if member == 0:
                # Exact replay anchors the new ensemble to the published deterministic run.
                verify_predictions(values, points, saved.points, "residual")
            predictions = _prediction_points(values, points, batch_offset=0)
            temporary = directory / "sample.tmp.npz"
            np.savez_compressed(
                temporary,
                fields=values.numpy(),
                point_ids=predictions.point_id.to_numpy(str),
                predictions=predictions.prediction.to_numpy(),
                seed=seed,
            )
            temporary.replace(destination)
            _json(
                destination.with_suffix(".json"),
                {
                    "sha256": _sha256(destination),
                    "seconds": time.monotonic() - started,
                    "seed": seed,
                },
            )
            print(
                f"GPU {shard}: batch {number + 1}/{len(batches)}, member {member + 1}/{members}, {time.monotonic() - started:.1f}s",
                flush=True,
            )
    _json(
        runtime / "complete.json",
        {"members": members, "shard": shard, "status": "complete"},
    )


def main() -> None:
    """Parse the explicit frozen-cohort and two-GPU worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=[0, 1], required=True)
    parser.add_argument("--members", type=int, default=20)
    args = parser.parse_args()
    worker(args.cohort, args.output, args.shard, args.members)


if __name__ == "__main__":
    main()
