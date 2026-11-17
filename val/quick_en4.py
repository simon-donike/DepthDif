# Examples: /work/envs/depth/bin/python -m val.quick_en4 prepare --source val/results/temperature_parallel_20261117/en4_temperature --output val/results/en4_quick/en4_temperature --patches 64 --seed 20261117
# CUDA_VISIBLE_DEVICES=0 /work/envs/depth/bin/python -m val.quick_en4 worker --output val/results/en4_quick/en4_temperature --shard 0
# CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -m val.quick_en4 worker --output val/results/en4_quick/en4_temperature --shard 1
# /work/envs/depth/bin/python -m val.quick_en4 merge --output val/results/en4_quick/en4_temperature
"""Run a frozen, coverage-selected EN4 pilot with durable two-patch bundles."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import yaml

from val.paper import STAGES, _json, load_bundle, run_suite
from val.run import _batches, _bundle, _config, _sha256
from val.screening import verify_screened_profiles


def choose_pairs(
    selection: dict, points: pd.DataFrame, patches: int, seed: int
) -> list[int]:
    """Choose original batches by month and region coverage, without using outcomes."""
    indices = [int(row["index"]) for row in selection["patches"]]
    if patches < 2 or patches % 2 or patches > len(indices) or len(indices) % 2:
        raise ValueError("Patch count must be even and within the frozen source cohort")
    features = {}
    for number in range(len(indices) // 2):
        rows = points[points.patch_index.isin(indices[2 * number : 2 * number + 2])]
        features[number] = (
            set(rows.region.astype(str)),
            set(pd.to_datetime(rows.target_time, utc=True).dt.month),
        )
    # Random tie ordering is fixed before predictions; prioritize all twelve months.
    remaining = list(np.random.default_rng(seed).permutation(len(features)))
    regions, months, chosen = set(), set(), []
    while len(chosen) < patches // 2:
        best = max(
            remaining,
            key=lambda n: (len(features[n][1] - months), len(features[n][0] - regions)),
        )
        chosen.append(int(best))
        remaining.remove(best)
        regions.update(features[best][0])
        months.update(features[best][1])
    return sorted(chosen)


def prepare(source: Path, output: Path, patches: int, seed: int) -> None:
    """Freeze source identities, original batch pairing, masks and two GPU shards."""
    output.mkdir(parents=True, exist_ok=False)
    source = source.resolve()
    config = _config(source / "config.yaml")
    selection = json.loads((source / "en4_selection.json").read_text())
    points = pd.read_csv(source / "cohort.csv")
    pairs = choose_pairs(selection, points, patches, seed)
    jobs = [
        {
            "original_batch": number,
            "patch_indices": [
                int(p["index"])
                for p in selection["patches"][2 * number : 2 * number + 2]
            ],
            "shard": i % 2,
        }
        for i, number in enumerate(pairs)
    ]
    selected = points[
        points.patch_index.isin([p for job in jobs for p in job["patch_indices"]])
    ]
    selected.to_csv(output / "cohort.csv", index=False)
    shutil.copy2(source / "holdouts.json", output / "holdouts.json")
    config["paper"]["calibration_fraction"] = 0
    config["paper"]["calibration_bundle"] = None
    config["paper"]["regional"]["depth_bin_m"] = None
    for stage in STAGES:
        config["paper"].setdefault(stage, {})["enabled"] = stage in {
            "reference",
            "comparators",
            "regional",
        }
    config["device"] = "cuda:0"
    config["cohort"]["max_patches"] = patches
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    screening = verify_screened_profiles(selected, config["cohort"], output)
    manifest = {
        "source": str(source),
        "seed": seed,
        "selection": "Greedy month then 20-degree region coverage, seeded tie order; original two-patch batches retained; no prediction-based selection.",
        "source_sha256": {
            name: _sha256(source / name)
            for name in (
                "cohort.csv",
                "en4_selection.json",
                "holdouts.json",
                "config.yaml",
            )
        },
        "frozen_sha256": {
            name: _sha256(output / name)
            for name in ("cohort.csv", "holdouts.json", "config.yaml")
        },
        "jobs": jobs,
        "patches": patches,
        "source_patches": len(selection["patches"]),
        "profiles": int(selected.profile_id.nunique()),
        "points_before_support": len(selected),
        "months": sorted(
            pd.to_datetime(selected.target_time, utc=True).dt.month.unique().tolist()
        ),
        "regions": sorted(selected.region.unique().tolist()),
        "screening": screening,
        "policy": "Preliminary coverage-selected EN4 subset; all retained points evaluated. All source holdouts remain masked. One generation, original DDPM steps and original residual batch seeds. Not an ocean-area representative sample or proof of non-assimilation.",
    }
    _json(output / "pilot.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("patches", "profiles", "points_before_support", "months")
            }
        ),
        flush=True,
    )


def worker(output: Path, shard: int) -> None:
    """Persist each original batch before moving on; resume only verified bundles."""
    import torch
    from depth_recon.inference.core import build_dataset
    from val.inference import run_model
    from val.observations import ExternalObservationCohort

    torch.set_num_threads(4)
    manifest = json.loads((output / "pilot.json").read_text())
    for name, digest in manifest["frozen_sha256"].items():
        if _sha256(output / name) != digest:
            raise ValueError(f"Frozen pilot input changed: {name}")
    config = _config(output / "config.yaml")
    workspace = output / f"worker_{shard}"
    workspace.mkdir(exist_ok=True)
    cache = workspace / "dataset_cache"
    if not cache.exists():
        shutil.copytree(Path(manifest["source"]) / "dataset_cache", cache)
    runtime = _bundle(config, workspace)
    dataset = build_dataset(
        runtime.effective_data_config_path, runtime.data_cfg["dataset"], split="val"
    )
    points = pd.read_csv(output / "cohort.csv")
    masks = json.loads((output / "holdouts.json").read_text())
    holdouts = {tuple(row) for row in masks["locations"]}
    source_selection = json.loads(
        (Path(manifest["source"]) / "en4_selection.json").read_text()
    )
    source_rows = {int(row["index"]): row for row in source_selection["patches"]}
    jobs = [job for job in manifest["jobs"] if job["shard"] == shard]
    for position, job in enumerate(jobs):
        number = job["original_batch"]
        target = output / "batches" / f"{number:03d}"
        if (target / "metadata.json").exists():
            load_bundle(target)
            continue
        for index in job["patch_indices"]:
            actual, expected = dataset._rows.iloc[index], source_rows[index]
            if any(
                int(actual[key]) != int(expected[key])
                for key in ("date", "grid_y0", "grid_x0")
            ):
                raise ValueError(
                    "Dataset patch indexing changed since the source selection"
                )
        cohort = ExternalObservationCohort(
            dataset,
            points[points.patch_index.isin(job["patch_indices"])].copy(),
            job["patch_indices"],
            holdouts,
            masks["metadata"],
            pd.DataFrame(),
        )
        exclusions = []
        batch_config = deepcopy(config)
        # A single-batch call otherwise resets its RNG to batch zero.
        batch_config["sampling"]["seed"] += number * 100000
        started = time.monotonic()
        print(
            f"SHARD {shard}: starting {position + 1}/{len(jobs)}, original batch {number}",
            flush=True,
        )
        metadata = run_model(
            batch_config,
            target,
            _batches(dataset, cohort, config, exclusions),
            bundle=runtime,
            dataset=dataset,
        )
        pd.DataFrame(exclusions, columns=["profile_id", "point_id", "reason"]).to_csv(
            target / "exclusions.csv", index=False
        )
        metadata.update(
            original_batch=number,
            patch_indices=job["patch_indices"],
            demo=False,
            checkpoint_verified=True,
        )
        metadata["bundle_sha256"] = {
            name: _sha256(target / name) for name in ("points.csv", "members.npz")
        }
        _json(target / "metadata.json", metadata)
        _json(
            workspace / "progress.json",
            {
                "completed": position + 1,
                "total": len(jobs),
                "last_batch": number,
                "last_seconds": time.monotonic() - started,
            },
        )
        print(
            f"SHARD {shard}: saved {position + 1}/{len(jobs)} in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    _json(workspace / "complete.json", {"complete": True, "batches": len(jobs)})


def merge(output: Path) -> None:
    """Combine all frozen batches on common method support and score without fitting."""
    manifest = json.loads((output / "pilot.json").read_text())
    for name, digest in manifest["frozen_sha256"].items():
        if _sha256(output / name) != digest:
            raise ValueError(f"Frozen pilot input changed: {name}")
    bundles = [
        load_bundle(output / "batches" / f'{job["original_batch"]:03d}')
        for job in manifest["jobs"]
    ]
    if len({b.metadata["checkpoint_sha256"] for b in bundles}) != 1:
        raise ValueError("Mixed residual checkpoints")
    for job, bundle in zip(manifest["jobs"], bundles):
        if (
            bundle.metadata["original_batch"] != job["original_batch"]
            or bundle.metadata["patch_indices"] != job["patch_indices"]
        ):
            raise ValueError("Saved batch differs from the frozen selection")
    for name in bundles[0].metadata["baselines"]:
        if (
            len(
                {
                    b.metadata["baselines"][name].get("checkpoint_sha256")
                    for b in bundles
                }
            )
            != 1
        ):
            raise ValueError(f"Mixed baseline checkpoints: {name}")
    methods = set.intersection(*(set(bundle.members) for bundle in bundles))
    points = pd.concat([bundle.points for bundle in bundles], ignore_index=True)
    if points.point_id.duplicated().any():
        raise ValueError("Duplicate evaluation point across batches")
    if not set(points.point_id).issubset(
        set(pd.read_csv(output / "cohort.csv").point_id)
    ):
        raise ValueError("Saved predictions include points outside the frozen cohort")
    for column in list(points):
        if (
            column.startswith("baseline_")
            and column.removeprefix("baseline_") not in methods
        ):
            points = points.drop(columns=column)
    destination = output / "bundle"
    destination.mkdir(exist_ok=False)
    points.to_csv(destination / "points.csv", index=False)
    np.savez_compressed(
        destination / "members.npz",
        **{
            name: np.concatenate([bundle.members[name] for bundle in bundles])
            for name in sorted(methods)
        },
    )
    metadata = deepcopy(bundles[0].metadata)
    metadata.update(
        point_count=len(points),
        seed=_config(output / "config.yaml")["sampling"]["seed"],
        pilot=manifest,
        cohort=json.loads((output / "holdouts.json").read_text())["metadata"],
        profile_screening=json.loads((output / "profile_membership.json").read_text()),
        temperature_kind_compared="potential",
        conditioning_exclusion_verified=False,
        training_split_verified=False,
        confirmation_untouched=False,
    )
    metadata.pop("original_batch", None)
    metadata.pop("patch_indices", None)
    metadata["batch_records"] = [
        {**row, "batch": bundle.metadata["original_batch"]}
        for bundle in bundles
        for row in bundle.metadata["batch_records"]
    ]
    metadata["batch_provenance"] = [
        {
            "original_batch": b.metadata["original_batch"],
            "metadata_sha256": _sha256(b.path / "metadata.json"),
            "bundle_sha256": b.metadata["bundle_sha256"],
        }
        for b in bundles
    ]
    for key in ("baseline_availability", "member_timings"):
        metadata[key] = [
            {**row, "batch": b.metadata["original_batch"]}
            for b in bundles
            for row in b.metadata[key]
        ]
    metadata["conditioning_count"] = sum(
        b.metadata["conditioning_count"] for b in bundles
    )
    metadata["peak_vram_bytes"] = max(b.metadata["peak_vram_bytes"] for b in bundles)
    metadata["workflow_sha256"] = {"quick_en4.py": _sha256(Path(__file__))}
    metadata["elapsed_seconds"] = sum(b.metadata["elapsed_seconds"] for b in bundles)
    metadata["elapsed_seconds_definition"] = (
        "Sum of batch worker elapsed times; two GPUs ran concurrently."
    )
    for name, record in metadata["baselines"].items():
        record["status"] = "available" if name in methods else "blocked"
        record["invalid_points"] = sum(
            b.metadata["baselines"][name].get("invalid_points", 0) for b in bundles
        )
        record["elapsed_seconds"] = sum(
            b.metadata["baselines"][name].get("elapsed_seconds", 0) for b in bundles
        )
    for filename in ("selection.csv", "exclusions.csv", "baseline_availability.csv"):
        pd.concat(
            [
                pd.read_csv(b.path / filename).assign(
                    original_batch=b.metadata["original_batch"]
                )
                for b in bundles
            ],
            ignore_index=True,
        ).to_csv(destination / filename, index=False)
    metadata["bundle_sha256"] = {
        name: _sha256(destination / name) for name in ("points.csv", "members.npz")
    }
    _json(destination / "metadata.json", metadata)
    config = _config(output / "config.yaml")
    result = run_suite(destination, output / "paper", config)
    _json(output / "result.json", result)
    print(
        f"Merged {len(points)} points, {points.profile_id.nunique()} profiles; methods {sorted(methods)}",
        flush=True,
    )


def main() -> None:
    """Parse preparation, one GPU worker, or final scoring commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--patches", type=int, default=64)
    prep.add_argument("--seed", type=int, default=20261117)
    work = commands.add_parser("worker")
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--shard", type=int, choices=(0, 1), required=True)
    finish = commands.add_parser("merge")
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "prepare":
        prepare(args.source, args.output, args.patches, args.seed)
    elif args.command == "worker":
        worker(args.output, args.shard)
    else:
        merge(args.output)


if __name__ == "__main__":
    main()
