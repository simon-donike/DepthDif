# Example:
# /work/envs/depth/bin/python experiments.py \
#   --model-config configs/px_space/model_config.yaml \
#   --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
#   --train-config configs/px_space/training_config.yaml \
#   --checkpoint logs/2026-02-25_12-32-00/last.ckpt \
#   --output-dir temp/experiments/conditioning_ablations \
#   --loader-split val \
#   --device auto \
#   --seed 7 \
#   --strict-load
"""Run manual qualitative conditioning ablations on one sample.

This script loads the configured model and checkpoint, creates a few fixed
conditioning cases, runs `predict_step` on each one, and saves comparison plots
plus compact metrics and tensor summaries for quick debugging.

Typical CLI:
    /work/envs/depth/bin/python experiments.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    choose_device,
    load_yaml,
    resolve_checkpoint_path,
    resolve_model_type,
    to_device,
)
from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch

MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"
DATA_CONFIG_PATH = "configs/px_space/data_ostia_argo_netcdf.yaml"
TRAIN_CONFIG_PATH = "configs/px_space/training_config.yaml"

LOADER_SPLIT = "val"
DEVICE = "auto"
SEED = 7
CHECKPOINT_PATH: str | None = None
# Optional explicit checkpoint override loaded right after model instantiation.
# Set to None to fall back to model.load_checkpoint then model.resume_checkpoint from model config.
CHECKPOINT_OVERRIDE_PATH: str | None = None
STRICT_LOAD = False
OUTPUT_DIR = Path("temp/experiments/conditioning_ablations")


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides while keeping constants editable for quick runs."""
    parser = argparse.ArgumentParser(
        description="Run conditioning ablation experiments on one dataloader sample."
    )
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--data-config", default=DATA_CONFIG_PATH)
    parser.add_argument("--train-config", default=TRAIN_CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_OVERRIDE_PATH or CHECKPOINT_PATH
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--loader-split", choices=["train", "val"], default=LOADER_SPLIT
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--strict-load", action="store_true", default=STRICT_LOAD)
    return parser.parse_args()


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Clone tensor fields so each experiment case can be edited independently."""
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _first_item_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep a single sample (batch size = 1) while preserving all batch keys."""
    one: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            one[key] = value[:1]
        elif isinstance(value, list):
            one[key] = value[:1]
        else:
            one[key] = value
    return one


def _build_case_batch(
    base_batch: dict[str, Any],
    *,
    disable_eo: bool,
    disable_x: bool,
    force_all_invalid: bool,
) -> dict[str, Any]:
    """Return one inference batch for a requested conditioning ablation."""
    case = _clone_batch(base_batch)

    if disable_eo and "eo" in case:
        case["eo"] = torch.zeros_like(case["eo"])
    if disable_x and "x" in case:
        case["x"] = torch.zeros_like(case["x"])
    if force_all_invalid and "x_valid_mask" in case:
        # x_valid_mask=0 everywhere means no observed x pixels are marked as known.
        case["x_valid_mask"] = torch.zeros_like(case["x_valid_mask"])
    if force_all_invalid and "x_valid_mask_1d" in case:
        case["x_valid_mask_1d"] = torch.zeros_like(case["x_valid_mask_1d"])

    return case


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    """Format compact summary statistics for one tensor."""
    t = tensor.detach().float()
    finite = t[torch.isfinite(t)]
    if int(finite.numel()) <= 0:
        return f"{name}: shape={tuple(t.shape)} finite=0"
    return (
        f"{name}: shape={tuple(t.shape)} "
        f"min={finite.min().item():.4f} max={finite.max().item():.4f} "
        f"mean={finite.mean().item():.4f} std={finite.std(unbiased=False).item():.4f} "
        f"finite={int(finite.numel())}"
    )


def _metric_support(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build the finite, valid support mask for image-space metric calculation."""
    support = torch.isfinite(predicted) & torch.isfinite(reference)
    if valid_mask is not None:
        # Masks in this codebase are float tensors where values above 0.5 are valid.
        support = support & (valid_mask > 0.5)
    return support


def _error_metrics(
    *,
    predicted: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> dict[str, float | int]:
    """Return MAE, RMSE, and bias over finite valid pixels."""
    support = _metric_support(predicted, reference, valid_mask)
    count = int(support.sum().item())
    if count <= 0:
        return {
            "count": 0,
            "mae_c": float("nan"),
            "rmse_c": float("nan"),
            "bias_c": float("nan"),
        }

    diff = (predicted - reference)[support].detach().float()
    return {
        "count": count,
        "mae_c": float(torch.mean(torch.abs(diff)).item()),
        "rmse_c": float(torch.sqrt(torch.mean(diff**2)).item()),
        "bias_c": float(torch.mean(diff).item()),
    }


def _case_metrics(
    *,
    case_name: str,
    source_batch: dict[str, Any],
    case_batch: dict[str, Any],
    pred: dict[str, Any],
    baseline_pred: torch.Tensor | None,
) -> dict[str, float | int | str]:
    """Compute target, observed-input, and baseline-delta metrics for one case."""
    y_hat_denorm = pred["y_hat_denorm"].detach().float()
    y_target_denorm = temperature_normalize(
        mode="denorm", tensor=source_batch["y"].detach().float()
    )
    x_denorm = temperature_normalize(
        mode="denorm", tensor=source_batch["x"].detach().float()
    )

    target_metrics = _error_metrics(
        predicted=y_hat_denorm,
        reference=y_target_denorm,
        valid_mask=source_batch.get("y_valid_mask"),
    )
    observed_metrics = _error_metrics(
        predicted=y_hat_denorm,
        reference=x_denorm,
        valid_mask=source_batch.get("x_valid_mask"),
    )
    row: dict[str, float | int | str] = {
        "case": case_name,
        "target_valid_count": target_metrics["count"],
        "target_mae_c": target_metrics["mae_c"],
        "target_rmse_c": target_metrics["rmse_c"],
        "target_bias_c": target_metrics["bias_c"],
        "observed_argo_count": observed_metrics["count"],
        "observed_argo_mae_c": observed_metrics["mae_c"],
        "observed_argo_rmse_c": observed_metrics["rmse_c"],
        "observed_argo_bias_c": observed_metrics["bias_c"],
        "input_x_valid_fraction": _mask_fraction(case_batch.get("x_valid_mask")),
        "input_eo_abs_mean": _abs_mean(case_batch.get("eo")),
    }

    if baseline_pred is not None:
        delta_metrics = _error_metrics(
            predicted=y_hat_denorm,
            reference=baseline_pred,
            valid_mask=source_batch.get("y_valid_mask"),
        )
        row["baseline_delta_mae_c"] = delta_metrics["mae_c"]
        row["baseline_delta_rmse_c"] = delta_metrics["rmse_c"]
    else:
        row["baseline_delta_mae_c"] = 0.0
        row["baseline_delta_rmse_c"] = 0.0
    return row


def _mask_fraction(mask: torch.Tensor | None) -> float:
    """Return the fraction of mask elements marked valid."""
    if mask is None:
        return float("nan")
    return float((mask.detach().float() > 0.5).float().mean().item())


def _abs_mean(tensor: torch.Tensor | None) -> float:
    """Return mean absolute value for a tensor, preserving missing tensors as NaN."""
    if tensor is None:
        return float("nan")
    return float(tensor.detach().float().abs().mean().item())


def _write_metrics_csv(
    *,
    output_path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Write experiment metrics to a stable CSV schema."""
    fieldnames = [
        "case",
        "target_valid_count",
        "target_mae_c",
        "target_rmse_c",
        "target_bias_c",
        "observed_argo_count",
        "observed_argo_mae_c",
        "observed_argo_rmse_c",
        "observed_argo_bias_c",
        "baseline_delta_mae_c",
        "baseline_delta_rmse_c",
        "input_x_valid_fraction",
        "input_eo_abs_mean",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mask_for_sample(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Return the first sample mask while preserving per-band masks when present."""
    if mask is None:
        return None
    if mask.ndim == 4:
        sample_mask = mask[0]
        if sample_mask.size(0) == 1:
            return sample_mask[0]
        return sample_mask
    if mask.ndim == 3:
        return mask[0]
    if mask.ndim == 2:
        return mask
    return None


def _plot_band_image(
    tensor: torch.Tensor,
    *,
    band_idx: int,
    mask: torch.Tensor | None = None,
) -> Any:
    """Render one band into [0,1] range with optional masking."""
    if tensor.ndim == 4:
        channel_idx = int(max(0, min(int(band_idx), int(tensor.size(1)) - 1)))
        image_t = tensor[0, channel_idx].detach().float()
    elif tensor.ndim == 3:
        image_t = tensor[0].detach().float()
    else:
        image_t = tensor.detach().float()

    image_t = torch.nan_to_num(image_t, nan=0.0, posinf=0.0, neginf=0.0)
    image_plot = minmax_stretch(image_t, mask=mask, nodata_value=None)
    return image_plot.cpu().numpy().astype("float32")


def _make_seen_withheld_rgb(
    seen_mask_1d: torch.Tensor,
    withheld_mask_1d: torch.Tensor,
) -> np.ndarray:
    """Render collapsed ambient visibility as green=seen, red=withheld, black=background."""
    if seen_mask_1d.ndim != 2 or withheld_mask_1d.ndim != 2:
        raise RuntimeError(
            "Expected 2D masks for RGB visualization, "
            f"got {tuple(seen_mask_1d.shape)} and {tuple(withheld_mask_1d.shape)}."
        )

    out = np.zeros((seen_mask_1d.shape[0], seen_mask_1d.shape[1], 3), dtype=np.float32)
    seen_np = seen_mask_1d.detach().cpu().numpy().astype(bool, copy=False)
    withheld_np = withheld_mask_1d.detach().cpu().numpy().astype(bool, copy=False)
    out[withheld_np, 0] = 1.0
    out[seen_np, 1] = 1.0
    return out


def _save_case_plot(
    *,
    case_name: str,
    case_batch: dict[str, Any],
    pred: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save one reconstruction grid image inspired by validation reconstruction logging."""
    x_denorm = temperature_normalize(
        mode="denorm", tensor=case_batch["x"].detach().float()
    )
    y_target_denorm = temperature_normalize(
        mode="denorm", tensor=case_batch["y"].detach().float()
    )
    y_hat_denorm = pred["y_hat_denorm"].detach().float()
    eo_denorm = None
    if "eo" in case_batch:
        eo_denorm = temperature_normalize(
            mode="denorm", tensor=case_batch["eo"].detach().float()
        )

    valid_mask_i = _mask_for_sample(case_batch.get("x_valid_mask"))
    land_mask_i = _mask_for_sample(case_batch.get("land_mask"))
    further_valid_mask_i = _mask_for_sample(pred.get("further_valid_mask"))
    ambient_valid_mask_rgb = None
    if valid_mask_i is not None and further_valid_mask_i is not None:
        valid_mask_1d = (
            valid_mask_i.any(dim=0) if valid_mask_i.ndim == 3 else valid_mask_i > 0.5
        )
        further_valid_mask_1d = (
            further_valid_mask_i.any(dim=0)
            if further_valid_mask_i.ndim == 3
            else further_valid_mask_i > 0.5
        )
        withheld_mask_1d = valid_mask_1d & ~further_valid_mask_1d
        ambient_valid_mask_rgb = _make_seen_withheld_rgb(
            further_valid_mask_1d, withheld_mask_1d
        )

    n_bands = int(y_hat_denorm.size(1)) if y_hat_denorm.ndim == 4 else 1
    ncols = 4 if eo_denorm is not None else 3
    if valid_mask_i is not None:
        ncols += 1
    if land_mask_i is not None:
        ncols += 1

    fig, axes = plt.subplots(
        n_bands, ncols, figsize=(4 * ncols, 2.8 * n_bands), squeeze=False
    )
    try:
        for band_idx in range(n_bands):
            valid_band = valid_mask_i
            if valid_band is not None and valid_band.ndim == 3:
                valid_band = valid_band[min(band_idx, int(valid_band.size(0)) - 1)]
            land_band = land_mask_i
            if land_band is not None and land_band.ndim == 3:
                land_band = land_band[min(band_idx, int(land_band.size(0)) - 1)]

            x_img = _plot_band_image(x_denorm, band_idx=band_idx, mask=land_band)
            y_hat_img = _plot_band_image(
                y_hat_denorm, band_idx=band_idx, mask=land_band
            )
            y_target_img = _plot_band_image(
                y_target_denorm, band_idx=band_idx, mask=land_band
            )
            if valid_band is not None:
                # Keep sparse observations visually sparse in the input panel.
                valid_np = valid_band.detach().cpu().numpy() > 0.5
                x_img[~valid_np] = 0.0
            if land_band is not None:
                ocean_np = land_band.detach().cpu().numpy() > 0.5
                x_img[~ocean_np] = 0.0
                y_hat_img[~ocean_np] = 0.0
                y_target_img[~ocean_np] = 0.0

            col = 0
            axes[band_idx, col].imshow(x_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[band_idx, col].set_axis_off()
            if band_idx == 0:
                axes[band_idx, col].set_title("Input")
            col += 1

            if eo_denorm is not None:
                eo_img = _plot_band_image(eo_denorm, band_idx=0, mask=land_band)
                if land_band is not None:
                    ocean_np = land_band.detach().cpu().numpy() > 0.5
                    eo_img[~ocean_np] = 0.0
                axes[band_idx, col].imshow(eo_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
                axes[band_idx, col].set_axis_off()
                if band_idx == 0:
                    axes[band_idx, col].set_title("EO condition")
                col += 1

            axes[band_idx, col].imshow(y_hat_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[band_idx, col].set_axis_off()
            if band_idx == 0:
                axes[band_idx, col].set_title("Reconstruction")
            col += 1

            axes[band_idx, col].imshow(y_target_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[band_idx, col].set_axis_off()
            if band_idx == 0:
                axes[band_idx, col].set_title("Target")
            col += 1

            if valid_mask_i is not None and valid_band is not None:
                if ambient_valid_mask_rgb is not None:
                    axes[band_idx, col].imshow(ambient_valid_mask_rgb)
                else:
                    axes[band_idx, col].imshow(
                        valid_band.detach().float().cpu().numpy(),
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                    )
                axes[band_idx, col].set_axis_off()
                if band_idx == 0:
                    axes[band_idx, col].set_title(
                        "Ambient valid mask"
                        if ambient_valid_mask_rgb is not None
                        else "Valid mask"
                    )
                col += 1

            if land_mask_i is not None and land_band is not None:
                axes[band_idx, col].imshow(
                    land_band.detach().float().cpu().numpy(),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[band_idx, col].set_axis_off()
                if band_idx == 0:
                    axes[band_idx, col].set_title("Land mask")

            axes[band_idx, 0].set_ylabel(f"b{band_idx}", rotation=90)

        fig.tight_layout()
        output_path = output_dir / f"{case_name}.png"
        fig.savefig(output_path, dpi=160)
        print(f"Saved plot: {output_path}")
    finally:
        plt.close(fig)


def main() -> None:
    """Run EO/X ablation experiments on one config-loaded dataloader sample."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    torch.manual_seed(int(args.seed))

    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    training_cfg = load_yaml(args.train_config)
    resolve_model_type(model_cfg)

    device = choose_device(args.device)

    dataset = build_dataset(args.data_config, data_cfg.get("dataset", {}))
    datamodule = build_datamodule(
        dataset=dataset, data_cfg=data_cfg, training_cfg=training_cfg
    )
    datamodule.setup("fit")

    model = build_model(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        training_config_path=args.train_config,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )

    ckpt_path = resolve_checkpoint_path(args.checkpoint, model_cfg)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict, strict=bool(args.strict_load))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    loader = (
        datamodule.train_dataloader()
        if args.loader_split == "train"
        else datamodule.val_dataloader()
    )
    batch = _first_item_batch(to_device(next(iter(loader)), device))

    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "eo_plus_x",
            _build_case_batch(
                batch,
                disable_eo=False,
                disable_x=False,
                force_all_invalid=False,
            ),
        ),
        (
            "x_only_no_eo",
            _build_case_batch(
                batch,
                disable_eo=True,
                disable_x=False,
                force_all_invalid=False,
            ),
        ),
        (
            "eo_only_no_x",
            _build_case_batch(
                batch,
                disable_eo=False,
                disable_x=True,
                force_all_invalid=True,
            ),
        ),
        (
            "coords_date_only_no_eo_no_x",
            _build_case_batch(
                batch,
                disable_eo=True,
                disable_x=True,
                force_all_invalid=True,
            ),
        ),
    ]

    hparams = getattr(model, "hparams", {})
    coord_conditioning_enabled = bool(
        getattr(
            getattr(model, "model", None),
            "coord_conditioning_enabled",
            (
                hparams.get("coord_conditioning_enabled", False)
                if hasattr(hparams, "get")
                else False
            ),
        )
    )
    date_conditioning_enabled = bool(
        getattr(
            getattr(model, "model", None),
            "date_conditioning_enabled",
            (
                hparams.get("date_conditioning_enabled", False)
                if hasattr(hparams, "get")
                else False
            ),
        )
    )

    if coord_conditioning_enabled and "coords" not in batch:
        raise RuntimeError(
            "Model has coord conditioning enabled, but batch has no 'coords'."
        )
    if date_conditioning_enabled and "date" not in batch:
        raise RuntimeError(
            "Model has date conditioning enabled, but batch has no 'date'."
        )

    run_manifest = {
        "model_config": str(args.model_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "loader_split": str(args.loader_split),
        "device": str(device),
        "seed": int(args.seed),
        "cases": [case_name for case_name, _case_batch in cases],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics_rows: list[dict[str, float | int | str]] = []
    summary_lines: list[str] = []
    baseline_pred: torch.Tensor | None = None
    for case_name, case_batch in cases:
        with torch.no_grad():
            pred = model.predict_step(case_batch, batch_idx=0)

        if baseline_pred is None:
            baseline_pred = pred["y_hat_denorm"].detach().float()

        _save_case_plot(
            case_name=case_name,
            case_batch=case_batch,
            pred=pred,
            output_dir=output_dir,
        )
        metrics_rows.append(
            _case_metrics(
                case_name=case_name,
                source_batch=batch,
                case_batch=case_batch,
                pred=pred,
                baseline_pred=baseline_pred,
            )
        )

        summary_lines.append(f"=== {case_name} ===")
        summary_lines.append(_summarize_tensor("input_x", case_batch["x"]))
        if "eo" in case_batch:
            summary_lines.append(_summarize_tensor("input_eo", case_batch["eo"]))
        if "x_valid_mask" in case_batch:
            summary_lines.append(
                _summarize_tensor("input_x_valid_mask", case_batch["x_valid_mask"])
            )
        if "coords" in case_batch:
            summary_lines.append(
                f"coords: {case_batch['coords'].detach().cpu().numpy().tolist()}"
            )
        if "date" in case_batch:
            summary_lines.append(
                f"date: {case_batch['date'].detach().cpu().numpy().tolist()}"
            )

        summary_lines.append(_summarize_tensor("y_hat", pred["y_hat"]))
        summary_lines.append(_summarize_tensor("y_hat_denorm", pred["y_hat_denorm"]))
        summary_lines.append("")

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "tensor_summary.txt"
    _write_metrics_csv(output_path=metrics_path, rows=metrics_rows)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote tensor summary: {summary_path}")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
