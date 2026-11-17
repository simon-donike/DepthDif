# Example: CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -m val.export_spatial --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --work val/results/xbt_spatial_20261117 --output docs/assets/validation/italian_xbt_temperature/spatial --device cuda:0
# Support-only example (no inference): /work/envs/depth/bin/python -m val.export_spatial --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --work val/results/xbt_supports --output docs/assets/validation/italian_xbt_temperature/spatial --device cpu --supports-only
# All-cast example: CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -m val.export_spatial --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --work val/results/xbt_spatial_all_20261117 --output docs/assets/validation/italian_xbt_temperature/spatial --device cuda:0 --all-profiles
"""Replay frozen XBT batches, verify point predictions, and export depth image stacks."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import torch

from depth_recon.configs.config_resolver_pixel import load_pixel_training_config
from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    load_checkpoint_weights,
    to_device,
)
from depth_recon.utils.normalizations import temperature_normalize
from depth_recon.utils.reconstruction_validation import _evaluation_rng
from val.baselines import check_artifact, materialize_config
from val.export_dashboard import evaluation_points
from val.inference import (
    _batch_fingerprint,
    _clone_batch,
    _physical_output,
    _prediction_points,
)
from val.observations import ExternalObservationCohort, _grid_transform
from val.paper import load_bundle
from val.run import _batches, _bundle, _config


def verify_predictions(prediction, assignments, saved, method):
    """Reject spatial replays that disagree with the recorded point predictions."""
    column = {
        "residual": "prediction_mean",
        "glorys": "glorys",
        "climatology": "climatology",
    }.get(method, f"baseline_{method}")
    if column not in saved:
        return {"method": method, "status": "no_valid_point_comparator"}
    actual = _prediction_points(
        prediction, assignments, batch_offset=0
    ).prediction.to_numpy()
    expected = saved.set_index("point_id").loc[assignments.point_id, column].to_numpy()
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-6, equal_nan=False):
        difference = float(np.nanmax(np.abs(actual - expected)))
        raise ValueError(
            f"Spatial replay disagrees with saved {method} predictions: max difference {difference}"
        )
    return {
        "method": method,
        "status": "passed",
        "points": len(actual),
        "max_abs_difference": float(np.max(np.abs(actual - expected))),
    }


def write_stack(path, values, valid, limits):
    """Encode depth-first rasters vertically; transparent pixels denote missing support."""
    lower, upper = limits
    supported = valid & np.isfinite(values)
    normalized = (np.nan_to_num(values, nan=lower) - lower) / (upper - lower)
    rgba = matplotlib.colormaps["viridis"](np.clip(normalized, 0, 1), bytes=True)
    rgba[..., 3] = np.where(supported, 255, 0)
    Image.fromarray(rgba.reshape(-1, values.shape[-1], 4)).save(path)


def write_colour_bar(output):
    """Write the exact colour lookup used by the raster stack encoder."""
    colors = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256), bytes=True)
    Image.fromarray(colors[None, :, :]).save(output / "colour_bar.png")


class ChunkedLSTM(torch.nn.Module):
    """Bound workspace memory for independent per-pixel depth sequences."""

    def __init__(self, recurrent, chunk_size=1024):
        """Keep frozen recurrent weights and the original sequence length."""
        super().__init__()
        if not recurrent.batch_first:
            raise ValueError("Spatial replay requires batch-first LSTM sequences")
        self.recurrent = recurrent
        self.chunk_size = chunk_size

    def forward(self, sequence):
        """Concatenate independent outputs without splitting any depth sequence."""
        outputs, hidden, cells = [], [], []
        # Only the independent pixel dimension is chunked. Depth order,
        # conditioning features and the outer model batch remain unchanged.
        for part in sequence.split(self.chunk_size, dim=0):
            output, (state, cell) = self.recurrent(part)
            outputs.append(output)
            hidden.append(state)
            cells.append(cell)
        return torch.cat(outputs), (torch.cat(hidden, dim=1), torch.cat(cells, dim=1))


def support_locations(mask):
    """Describe retained temperature support cells and their valid depth indices."""
    valid = mask.detach().cpu().numpy().astype(bool)
    if valid.ndim != 3:
        raise ValueError("Expected a depth/row/column conditioning mask")
    # Project only the masked inference input, never the unmasked source archive.
    return [
        {
            "row": int(row),
            "col": int(col),
            "levels": np.flatnonzero(valid[:, row, col]).tolist(),
        }
        for row, col in np.argwhere(valid.any(axis=0))
    ]


def export(
    cohort_root, work, output, device_name, supports_only=False, all_profiles=False
):
    """Reconstruct all original batches, keeping only evaluation patches for display."""
    if work.exists() and any(path.name != "dataset_cache" for path in work.iterdir()):
        raise FileExistsError("Replay work directory already contains results")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    saved = load_bundle(cohort_root / "bundle")
    config = _config(cohort_root / "config.yaml")
    if config["cohort"]["kind"] != "external" or config["sampling"]["members"] != 1:
        raise ValueError(
            "Spatial replay currently requires a single-member external cohort"
        )
    config["device"] = device_name
    device = torch.device(device_name)
    bundle = _bundle(config, work)
    dataset = build_dataset(
        bundle.effective_data_config_path, bundle.data_cfg["dataset"], split="val"
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
        raise ValueError("Replay batch count differs from original inference")
    for number, (batch, points) in enumerate(batches):
        if (
            _batch_fingerprint(batch)
            != saved.metadata["batch_records"][number]["input_sha256"]
        ):
            raise ValueError(f"Replay conditioning differs in batch {number}")
    report = "paper_all_profiles" if all_profiles else "paper"
    if all_profiles:
        policy = json.loads((cohort_root / report / "all_profiles.json").read_text())
        if (
            not policy["complete"]
            or policy["bundle_sha256"] != saved.metadata["bundle_sha256"]
        ):
            raise ValueError("All-profile report differs from the frozen bundle")
    evaluation = evaluation_points(
        saved.points, pd.read_csv(cohort_root / report / "partition.csv")
    )
    if all_profiles and len(evaluation) != len(saved.points):
        raise ValueError("All-profile replay must include every observation")
    patches = sorted(evaluation.patch_index.unique())
    supports = {}
    context_patches = []
    transform, _, _ = _grid_transform(dataset)
    for batch, points in batches:
        for patch in points.patch_index.unique():
            index = int(points.loc[points.patch_index.eq(patch), "batch_index"].iloc[0])
            supports[int(patch)] = support_locations(batch["x_valid_mask"][index])
            row = dataset._rows.iloc[int(patch)]
            height, width = batch["x_valid_mask"].shape[-2:]
            west, north = transform * (int(row.grid_x0), int(row.grid_y0))
            east, south = transform * (
                int(row.grid_x0) + width,
                int(row.grid_y0) + height,
            )
            context_patches.append(
                {
                    "patch_index": int(patch),
                    "date": str(row.date),
                    "width": width,
                    "height": height,
                    "bounds": [west, south, east, north],
                    "en4_supports": supports[int(patch)],
                }
            )
    if supports_only:
        # Preserve verified image stacks; updating markers requires no model replay.
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest["checkpoint_sha256"] != saved.metadata["checkpoint_sha256"]:
            raise ValueError("Spatial manifest checkpoint differs from saved inference")
        if not np.array_equal(manifest["depths_m"], dataset.depth_axis_m):
            raise ValueError("Spatial manifest depth axis differs from saved inference")
        if {item["patch_index"] for item in manifest["patches"]} != set(patches):
            raise ValueError("Spatial manifest evaluation patches differ")
        for item in manifest["patches"]:
            item["en4_supports"] = supports[item["patch_index"]]
        manifest["context_patches"] = context_patches
        manifest["support_source"] = (
            "Temperature x_valid_mask after holdout masking; all batch input hashes verified against saved inference. Locations are model-grid cells, not unique source-profile identities."
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, allow_nan=False, indent=2) + "\n")
        temporary.replace(path)
        print(f"Exported verified EN4 supports for {len(patches)} patches", flush=True)
        return
    fields = {int(patch): {} for patch in patches}
    masks = {}
    checks = []
    for batch, points in batches:
        target = "y_glorys" if "y_glorys" in batch else "y"
        valid = batch[target + "_valid_mask"].bool() & batch["land_mask"].bool()
        for method, key in (("glorys", target), ("climatology", "climatology")):
            values = temperature_normalize("denorm", batch[key]).masked_fill(
                ~valid, float("nan")
            )
            checks.append(verify_predictions(values, points, saved.points, method))
            for patch in points.patch_index.unique():
                if patch in fields:
                    index = int(
                        points.loc[points.patch_index.eq(patch), "batch_index"].iloc[0]
                    )
                    fields[patch][method] = values[index].numpy().copy()
                    masks[patch] = valid[index].numpy().copy()
    specifications = [{"name": "residual", **config["model"]}, *config["baselines"]]
    specifications.sort(key=lambda spec: spec["name"] != "lstm")
    for spec in specifications:
        name = spec["name"]
        print(f"Replaying spatial fields: {name}", flush=True)
        runtime = work / name
        resolved = (
            bundle
            if name == "residual"
            else load_pixel_training_config(
                config_path_value=materialize_config(spec, runtime),
                runtime_config_dir=runtime,
                **(
                    {"scenario_override": spec["scenario"]}
                    if spec.get("scenario")
                    else {}
                ),
                **({"overrides": spec["overrides"]} if spec.get("overrides") else {}),
                write_snapshots=True,
            )
        )
        datamodule = build_datamodule(dataset, resolved.data_cfg, resolved.training_cfg)
        model = build_model(
            resolved.effective_model_config_path,
            resolved.effective_data_config_path,
            resolved.effective_training_config_path,
            resolved.model_cfg,
            datamodule,
        )
        if spec.get("checkpoint"):
            check_artifact(spec["checkpoint"], spec["checkpoint_sha256"])
            requested = spec.get("weights", "standard")
            loaded = load_checkpoint_weights(
                model,
                Path(spec["checkpoint"]),
                strict=True,
                prefer_ema=requested == "ema",
            )
            if loaded != requested:
                raise ValueError(f"Incorrect weights for {name}")
        if name == "lstm":
            for field, recurrent in list(model.field_lstms.items()):
                model.field_lstms[field] = ChunkedLSTM(recurrent)
        model.to(device).eval()
        for number, (batch, points) in enumerate(batches):
            # Preserve the original batch shape and RNG seed, even for calibration-only patches.
            seed = int(config["sampling"]["seed"]) + number * (
                100_000 if name == "residual" else 1
            )
            with _evaluation_rng(device, seed), torch.inference_mode():
                result = model.predict_step(
                    to_device(_clone_batch(batch), device), batch_idx=number
                )
                values = _physical_output(result, "temperature").detach().float().cpu()
                del result
            checks.append(
                {
                    "batch": number,
                    **verify_predictions(values, points, saved.points, name),
                }
            )
            for patch in points.patch_index.unique():
                if patch in fields:
                    index = int(
                        points.loc[points.patch_index.eq(patch), "batch_index"].iloc[0]
                    )
                    fields[patch][name] = values[index].numpy().copy()
            del values
            print(f"  batch {number + 1}/{len(batches)} verified", flush=True)
        # Load one model at a time to avoid competing with the running validation jobs.
        model.cpu()
        del model, datamodule
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    transform, _, _ = _grid_transform(dataset)
    write_colour_bar(output)
    exported = []
    for patch in patches:
        row = dataset._rows.iloc[int(patch)]
        values = fields[patch]
        valid = masks[patch]
        finite = np.concatenate(
            [value[valid & np.isfinite(value)] for value in values.values()]
        )
        limits = [float(finite.min()), float(finite.max())]
        if limits[0] == limits[1]:
            limits[1] += 1
        method_files = {}
        for method, field in values.items():
            filename = f"patch_{patch}_{method}.png"
            write_stack(output / filename, field, valid, limits)
            method_files[method] = filename
        # Preserve physical arrays locally; the website contains only labelled images.
        np.savez_compressed(work / f"patch_{patch}.npz", **values, valid=valid)
        height, width = valid.shape[-2:]
        west, north = transform * (int(row.grid_x0), int(row.grid_y0))
        east, south = transform * (int(row.grid_x0) + width, int(row.grid_y0) + height)
        selected = evaluation[evaluation.patch_index.eq(patch)].drop_duplicates(
            "profile_id"
        )
        exported.append(
            {
                "patch_index": int(patch),
                "date": str(row.date),
                "width": width,
                "height": height,
                "bounds": [west, south, east, north],
                "limits_c": limits,
                "methods": method_files,
                "en4_supports": supports[int(patch)],
                "profiles": selected[["profile_id", "row", "col"]].to_dict(
                    orient="records"
                ),
            }
        )
    manifest = {
        "execution_device": device_name,
        "lstm_execution": "Independent pixel sequences chunked on GPU; outer batches and full depth sequences preserved; saved point predictions verified",
        "depths_m": np.asarray(dataset.depth_axis_m).tolist(),
        "patches": exported,
        "context_patches": context_patches,
        "checkpoint_sha256": saved.metadata["checkpoint_sha256"],
        "point_checks": checks,
        "input_hashes_verified": True,
        "support_source": "Temperature x_valid_mask after holdout masking; all batch input hashes verified against saved inference. Locations are model-grid cells, not unique source-profile identities.",
        "color_scale": "viridis; common range across all methods and depths within each patch",
        "unsupported": "Transparent pixels indicate missing reference or model support. IDW is not scored because point support is incomplete.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, indent=2) + "\n"
    )
    print(
        f"Exported {len(exported)} evaluation patches, {len(dataset.depth_axis_m)} depth levels",
        flush=True,
    )


def main():
    """Parse replay inputs independently of the running campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Use the verified all-cast report, including both Italian missions",
    )
    parser.add_argument(
        "--supports-only",
        action="store_true",
        help="Update support markers in an existing manifest without running inference",
    )
    args = parser.parse_args()
    export(
        args.cohort,
        args.work,
        args.output,
        args.device,
        args.supports_only,
        args.all_profiles,
    )


if __name__ == "__main__":
    main()
