# Example:
# /work/envs/depth/bin/python -m depth_recon.scripts.evaluate_depth_baselines \
#   --target-npz validation_target.npz --target-key target \
#   --mask-npz validation_mask.npz --mask-key mask \
#   --method climatology=climatology_prediction.npz \
#   --method unet=unet_prediction.npz --output-json depth_metrics.json
# /work/envs/depth/bin/python -m depth_recon.scripts.evaluate_depth_baselines \
#   --configured-method 'best|training.yaml|best.ckpt' --sample-count 32 \
#   --seed 7 --batch-size 4 --device cuda --variable temperature \
#   --output-json depth_metrics.json --output-npz depth_predictions.npz
"""Compare fixed-sample depth predictions with mask-aware diagnostics.

Compare exported arrays or run configured checkpoints on reproducibly selected
validation samples without changing the shuffled training validation loader.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from depth_recon.configs.config_resolver_pixel import load_pixel_training_config
from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    choose_device,
    load_checkpoint_weights,
    to_device,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.validation_denoise import compute_depth_diagnostics


def _load_array(path: Path, *, key: str) -> np.ndarray:
    """Load one array from an .npy or .npz file."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    try:
        if key in loaded.files:
            return np.asarray(loaded[key])
        if len(loaded.files) == 1:
            return np.asarray(loaded[loaded.files[0]])
        raise ValueError(f"{path} contains multiple arrays; expected key {key!r}.")
    finally:
        loaded.close()


def evaluate_prediction_arrays(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Return JSON-compatible per-depth metrics for fixed prediction arrays."""
    output: dict[str, dict[str, Any]] = {}
    for method, prediction in predictions.items():
        diagnostics = compute_depth_diagnostics(
            prediction, target, valid_mask=valid_mask
        )
        output[str(method)] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in diagnostics.items()
        }
    return output


def _parse_configured_method(specification: str) -> tuple[str, Path, Path]:
    """Parse ``NAME|CONFIG|CHECKPOINT`` configured-method syntax."""
    pieces = specification.split("|", 2)
    if len(pieces) != 3 or not all(piece.strip() for piece in pieces):
        raise ValueError("--configured-method must use NAME|CONFIG|CHECKPOINT syntax.")
    return pieces[0].strip(), Path(pieces[1]), Path(pieces[2])


def _dataset_signature(
    dataset: Any, indices: list[int]
) -> list[tuple[Any, ...]] | None:
    """Return stable row identities when a dataset exposes validation rows."""
    rows = getattr(dataset, "rows", getattr(dataset, "_rows", None))
    if rows is None:
        return None
    fields = ("date", "grid_y0", "grid_x0", "patch_id")
    if hasattr(rows, "iloc"):
        rows = rows.iloc[indices].to_dict("records")
    else:
        rows = [rows[index] for index in indices]
    return [tuple(row.get(field) for field in fields) for row in rows]


def _evaluate_configured_methods(
    methods: list[tuple[str, Path, Path]],
    *,
    sample_count: int,
    seed: int,
    batch_size: int,
    device: str,
    output_npz: Path | None = None,
    variable: str = "temperature",
    runtime_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Evaluate checkpoints on one deterministic validation subset.

    Each method receives an independently constructed model, but all models
    consume the same dataset indices.  The dataset's normal validation loader
    is not modified, preserving its intentionally shuffled behavior.
    """
    if not methods:
        raise ValueError("At least one configured method is required.")
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive.")
    names = [name for name, _, _ in methods]
    if len(set(names)) != len(names) or "climatology" in names:
        raise ValueError("Method names must be unique and cannot be 'climatology'.")
    if variable not in {"temperature", "salinity"}:
        raise ValueError("variable must be 'temperature' or 'salinity'.")
    denormalize = (
        temperature_normalize if variable == "temperature" else salinity_normalize
    )
    first_name, first_config, _ = methods[0]
    del first_name
    first_bundle = load_pixel_training_config(
        config_path_value=first_config,
        runtime_config_dir=runtime_dir / "dataset",
        write_snapshots=False,
    )
    reference_dataset = build_dataset(
        first_bundle.effective_data_config_path,
        first_bundle.data_cfg.get("dataset", {}),
        split="val",
    )
    if len(reference_dataset) < int(sample_count):
        raise ValueError(
            f"Validation dataset has {len(reference_dataset)} samples; requested {sample_count}."
        )
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(
        len(reference_dataset), size=int(sample_count), replace=False
    ).tolist()
    loader = DataLoader(
        Subset(reference_dataset, indices), batch_size=int(batch_size), shuffle=False
    )
    resolved_device = choose_device(device)
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    climate: list[np.ndarray] = []
    for batch in loader:
        prefix = "y" if variable == "temperature" else "y_salinity"
        target_key = f"{prefix}_glorys" if f"{prefix}_glorys" in batch else prefix
        mask_key = f"{target_key}_valid_mask"
        targets.append(denormalize("denorm", batch[target_key]).numpy())
        valid_masks.append(batch[mask_key].numpy())
        climatology_key = (
            "climatology" if variable == "temperature" else "climatology_salinity"
        )
        if climatology_key in batch:
            climate.append(denormalize("denorm", batch[climatology_key]).numpy())
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(valid_masks, axis=0)
    report: dict[str, dict[str, Any]] = {}
    predictions_np: dict[str, np.ndarray] = {}
    if climate:
        predictions_np["climatology"] = np.concatenate(climate, axis=0)

    for method_index, (method_name, config_path, checkpoint_path) in enumerate(methods):
        bundle = load_pixel_training_config(
            config_path_value=config_path,
            runtime_config_dir=runtime_dir / f"method_{method_index}",
            write_snapshots=False,
        )
        dataset = build_dataset(
            bundle.effective_data_config_path,
            bundle.data_cfg.get("dataset", {}),
            split="val",
        )
        if len(dataset) != len(reference_dataset):
            raise ValueError(
                f"Method {method_name!r} uses a different validation dataset length."
            )
        reference_signature = _dataset_signature(reference_dataset, indices)
        method_signature = _dataset_signature(dataset, indices)
        if (
            reference_signature is not None
            and method_signature is not None
            and reference_signature != method_signature
        ):
            raise ValueError(f"Method {method_name!r} uses different validation rows.")
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
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        load_checkpoint_weights(model, checkpoint_path, strict=True)
        model.to(resolved_device).eval()
        predictions: list[np.ndarray] = []
        method_loader = DataLoader(
            Subset(dataset, indices), batch_size=int(batch_size), shuffle=False
        )
        start = 0
        for batch in method_loader:
            prefix = "y" if variable == "temperature" else "y_salinity"
            target_key = f"{prefix}_glorys" if f"{prefix}_glorys" in batch else prefix
            method_target = denormalize("denorm", batch[target_key]).numpy()
            stop = start + method_target.shape[0]
            if not np.array_equal(
                method_target, target_np[start:stop], equal_nan=True
            ) or not np.array_equal(
                batch[f"{target_key}_valid_mask"].numpy(), mask_np[start:stop]
            ):
                raise ValueError(
                    f"Method {method_name!r} has different target values or support."
                )
            start = stop
            prediction_batch = to_device(batch, resolved_device)
            with torch.no_grad():
                output = model.predict_step(prediction_batch, batch_idx=0)
            key = f"y_hat_{variable}_denorm"
            if key not in output:
                key = "y_hat_denorm"
            predictions.append(output[key].detach().cpu().numpy())
        predictions_np[method_name] = np.concatenate(predictions, axis=0)
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    for method_name, prediction in predictions_np.items():
        if np.any((mask_np > 0) & np.isfinite(target_np) & ~np.isfinite(prediction)):
            raise ValueError(
                f"Method {method_name!r} produced nonfinite predictions on valid targets."
            )
        diagnostics = compute_depth_diagnostics(prediction, target_np, mask_np)
        report[method_name] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in diagnostics.items()
        }
        report[method_name].update(
            depth_axis_m=np.asarray(reference_dataset.depth_axis_m).tolist(),
            sample_indices=indices,
            seed=int(seed),
            variable=variable,
        )
    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            target=target_np,
            valid_mask=mask_np,
            depth_axis_m=np.asarray(reference_dataset.depth_axis_m),
            sample_indices=np.asarray(indices),
            **predictions_np,
        )
    return report


def evaluate_configured_methods(
    methods: list[tuple[str, Path, Path]],
    *,
    sample_count: int,
    seed: int,
    batch_size: int,
    device: str,
    output_npz: Path | None = None,
    variable: str = "temperature",
) -> dict[str, dict[str, Any]]:
    """Compare checkpoints with isolated runtime configs and reproducible validation inputs."""
    with tempfile.TemporaryDirectory(prefix="depthdif_depth_evaluation_") as directory:
        return _evaluate_configured_methods(
            methods,
            sample_count=sample_count,
            seed=seed,
            batch_size=batch_size,
            device=device,
            output_npz=output_npz,
            variable=variable,
            runtime_dir=Path(directory),
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-npz", type=Path)
    parser.add_argument(
        "--target-key", default="target", help="Array key when target is an .npz file."
    )
    parser.add_argument(
        "--mask-npz", type=Path, help="Optional validity mask .npy/.npz file."
    )
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument(
        "--method",
        action="append",
        metavar="NAME=PATH",
        help="Prediction array to score; repeat for each method.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--configured-method",
        action="append",
        metavar="NAME|CONFIG|CHECKPOINT",
        help="Evaluate a configured checkpoint on a shared deterministic val subset.",
    )
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument(
        "--variable", choices=("temperature", "salinity"), default="temperature"
    )
    return parser


def main() -> None:
    """Parse inputs, score predictions, and write a JSON report."""
    args = _build_parser().parse_args()
    if args.configured_method:
        methods = [_parse_configured_method(item) for item in args.configured_method]
        report = evaluate_configured_methods(
            methods,
            sample_count=int(args.sample_count),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            device=args.device,
            output_npz=args.output_npz,
            variable=args.variable,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        return

    if args.target_npz is None or not args.method:
        raise ValueError("Array mode requires --target-npz and at least one --method.")
    target = _load_array(args.target_npz, key=args.target_key)
    valid_mask = (
        _load_array(args.mask_npz, key=args.mask_key)
        if args.mask_npz is not None
        else None
    )
    predictions: dict[str, np.ndarray] = {}
    for specification in args.method:
        if "=" not in specification:
            raise ValueError(f"--method must be NAME=PATH (got {specification!r}).")
        name, raw_path = specification.split("=", 1)
        if not name.strip():
            raise ValueError("--method names must not be empty.")
        predictions[name.strip()] = _load_array(Path(raw_path), key="prediction")
    report = evaluate_prediction_arrays(target, predictions, valid_mask)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
