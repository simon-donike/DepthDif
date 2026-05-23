"""Export validation-set depthwise error summaries for one trained model checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.configs.config_resolver_pixel import (
    DEFAULT_PIXEL_INFERENCE_CONFIG_PATH,
    PIXEL_SCENARIOS,
    load_pixel_inference_config,
)
from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.inference.core import (
    build_dataset,
    build_model,
    choose_device,
    load_checkpoint_weights,
    load_yaml,
    resolve_checkpoint_path,
    to_device,
)
from depth_recon.inference.export_global import (
    _load_glorys_depth_axis_m,
    _parse_yyyymmdd,
)
from depth_recon.utils.normalizations import temperature_normalize
from depth_recon.utils.validation_denoise import (
    save_average_glorys_profile_and_error_plot,
    save_average_glorys_profile_error_plot,
)

DEFAULT_INFERENCE_CONFIG = DEFAULT_PIXEL_INFERENCE_CONFIG_PATH
DEFAULT_OUTPUT_ROOT = Path("inference/outputs")
DEFAULT_OUTPUT_NAME = "validation_error_summary"
DEFAULT_CSV_NAME = "validation_error_by_depth.csv"
DEFAULT_ERROR_PLOT_NAME = "validation_median_absolute_error_by_depth.png"
DEFAULT_PROFILE_ERROR_PLOT_NAME = "validation_median_profile_and_error_by_depth.png"


@dataclass
class ValidationErrorSummaryAccumulator:
    """Running pooled validation statistics for one full depth axis."""

    abs_error_prediction_vs_glorys_values: list[list[np.ndarray]]
    abs_error_prediction_vs_argo_values: list[list[np.ndarray]]
    prediction_values: list[list[np.ndarray]]
    glorys_values: list[list[np.ndarray]]
    argo_values: list[list[np.ndarray]]
    sample_count: int = 0
    batch_count: int = 0


def build_validation_summary_dataset(
    data_config_path: str,
    *,
    split: str,
) -> torch.utils.data.Dataset:
    """Build the explicit dataset split used for validation summary export."""
    data_cfg = load_yaml(data_config_path)
    ds_cfg = data_cfg.get("dataset", {})
    return build_dataset(
        str(data_config_path),
        ds_cfg,
        split=str(split),
    )


def _dataset_rows(dataset: torch.utils.data.Dataset) -> list[dict[str, Any]]:
    if hasattr(dataset, "rows"):
        return list(getattr(dataset, "rows"))
    if hasattr(dataset, "_rows"):
        return list(getattr(dataset, "_rows"))
    raise RuntimeError("Dataset does not expose rows metadata.")


def _set_dataset_rows(
    dataset: torch.utils.data.Dataset,
    rows: list[dict[str, Any]],
) -> None:
    if not hasattr(dataset, "_rows"):
        raise RuntimeError("Dataset rows cannot be replaced for ISO-week filtering.")
    dataset._rows = rows


def filter_validation_summary_dataset_by_iso_week(
    dataset: torch.utils.data.Dataset,
    *,
    iso_year: int | None = None,
    iso_week: int | None = None,
) -> tuple[int | None, int | None]:
    """Restrict the already split dataset rows to one requested ISO week."""
    if iso_year is None and iso_week is None:
        return None, None
    if (iso_year is None) ^ (iso_week is None):
        raise ValueError("--year and --iso-week must be provided together.")

    rows = _dataset_rows(dataset)
    filtered_rows = [
        row
        for row in rows
        if _parse_yyyymmdd(row["date"]).isocalendar()[:2]
        == (int(iso_year), int(iso_week))
    ]
    if not filtered_rows:
        raise RuntimeError(
            f"No dataset rows matched ISO week {int(iso_year)}-W{int(iso_week):02d} "
            f"within split '{getattr(dataset, 'split', '')}'."
        )
    _set_dataset_rows(dataset, filtered_rows)
    return int(iso_year), int(iso_week)


def create_validation_error_summary_accumulator(
    depth_size: int,
) -> ValidationErrorSummaryAccumulator:
    """Create pooled per-depth value buffers for the given depth axis."""
    empty_lists = [[] for _ in range(int(depth_size))]
    return ValidationErrorSummaryAccumulator(
        abs_error_prediction_vs_glorys_values=[bucket.copy() for bucket in empty_lists],
        abs_error_prediction_vs_argo_values=[bucket.copy() for bucket in empty_lists],
        prediction_values=[bucket.copy() for bucket in empty_lists],
        glorys_values=[bucket.copy() for bucket in empty_lists],
        argo_values=[bucket.copy() for bucket in empty_lists],
    )


def _masked_depthwise_values(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[np.ndarray]:
    """Extract valid flattened values per depth channel from a `(B,C,H,W)` tensor."""
    if values.ndim != 4 or valid_mask.ndim != 4:
        raise ValueError(
            "values and valid_mask must both have shape (B,C,H,W): "
            f"values={tuple(values.shape)}, valid_mask={tuple(valid_mask.shape)}"
        )
    if values.shape != valid_mask.shape:
        raise ValueError(
            "values and valid_mask must share the same shape: "
            f"values={tuple(values.shape)}, valid_mask={tuple(valid_mask.shape)}"
        )
    values_np = values.detach().cpu().numpy()
    valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    depthwise_values: list[np.ndarray] = []
    for depth_idx in range(int(values_np.shape[1])):
        selected = values_np[:, depth_idx][valid_mask_np[:, depth_idx]]
        depthwise_values.append(np.asarray(selected, dtype=np.float64).reshape(-1))
    return depthwise_values


def update_validation_error_summary_accumulator(
    accumulator: ValidationErrorSummaryAccumulator,
    *,
    x_denorm: torch.Tensor,
    y_denorm: torch.Tensor,
    y_hat_denorm: torch.Tensor,
    x_valid_mask: torch.Tensor,
    y_valid_mask: torch.Tensor,
) -> None:
    """Accumulate pooled per-depth values from one predicted batch."""
    y_support = (y_valid_mask > 0.5) & torch.isfinite(y_denorm)
    x_support = (x_valid_mask > 0.5) & torch.isfinite(x_denorm)
    prediction_support = y_support & torch.isfinite(y_hat_denorm)
    pred_vs_glorys_support = prediction_support & torch.isfinite(y_denorm)
    pred_vs_argo_support = x_support & torch.isfinite(y_hat_denorm)

    prediction_values = _masked_depthwise_values(
        y_hat_denorm,
        prediction_support,
    )
    glorys_values = _masked_depthwise_values(
        y_denorm,
        y_support,
    )
    argo_values = _masked_depthwise_values(
        x_denorm,
        x_support,
    )
    glorys_error_values = _masked_depthwise_values(
        torch.abs(y_hat_denorm - y_denorm),
        pred_vs_glorys_support,
    )
    argo_error_values = _masked_depthwise_values(
        torch.abs(y_hat_denorm - x_denorm),
        pred_vs_argo_support,
    )

    for depth_idx, values in enumerate(prediction_values):
        if values.size > 0:
            accumulator.prediction_values[depth_idx].append(values)
    for depth_idx, values in enumerate(glorys_values):
        if values.size > 0:
            accumulator.glorys_values[depth_idx].append(values)
    for depth_idx, values in enumerate(argo_values):
        if values.size > 0:
            accumulator.argo_values[depth_idx].append(values)
    for depth_idx, values in enumerate(glorys_error_values):
        if values.size > 0:
            accumulator.abs_error_prediction_vs_glorys_values[depth_idx].append(values)
    for depth_idx, values in enumerate(argo_error_values):
        if values.size > 0:
            accumulator.abs_error_prediction_vs_argo_values[depth_idx].append(values)
    accumulator.sample_count += int(x_denorm.shape[0])
    accumulator.batch_count += 1


def _pooled_nanmedian_and_count(
    values_by_depth: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one pooled median and count per depth while preserving empty levels as NaN."""
    medians = np.full((int(len(values_by_depth)),), np.nan, dtype=np.float64)
    counts = np.zeros((int(len(values_by_depth)),), dtype=np.int64)
    for depth_idx, depth_values in enumerate(values_by_depth):
        if not depth_values:
            continue
        concatenated = np.concatenate(depth_values, axis=0).astype(
            np.float64, copy=False
        )
        if concatenated.size <= 0:
            continue
        medians[depth_idx] = float(np.nanmedian(concatenated))
        counts[depth_idx] = int(concatenated.size)
    return medians, counts


def build_validation_error_summary_dataframe(
    accumulator: ValidationErrorSummaryAccumulator,
    *,
    depth_axis_m: np.ndarray,
) -> pd.DataFrame:
    """Convert pooled validation statistics into the exported per-depth CSV table."""
    depth_axis_np = np.asarray(depth_axis_m, dtype=np.float64).reshape(-1)
    if depth_axis_np.size != len(accumulator.abs_error_prediction_vs_glorys_values):
        raise ValueError(
            "depth_axis_m must match the accumulated depth dimension: "
            f"{int(depth_axis_np.size)} != "
            f"{int(len(accumulator.abs_error_prediction_vs_glorys_values))}"
        )
    median_abs_error_prediction_vs_glorys, count_prediction_vs_glorys = (
        _pooled_nanmedian_and_count(accumulator.abs_error_prediction_vs_glorys_values)
    )
    median_abs_error_prediction_vs_argo, count_prediction_vs_argo = (
        _pooled_nanmedian_and_count(accumulator.abs_error_prediction_vs_argo_values)
    )
    median_prediction_profile, count_prediction_profile = _pooled_nanmedian_and_count(
        accumulator.prediction_values
    )
    median_glorys_profile, count_glorys_profile = _pooled_nanmedian_and_count(
        accumulator.glorys_values
    )
    median_argo_profile, count_argo_profile = _pooled_nanmedian_and_count(
        accumulator.argo_values
    )
    return pd.DataFrame(
        {
            "depth_index": np.arange(int(depth_axis_np.size), dtype=np.int64),
            "depth_m": depth_axis_np,
            "median_abs_error_prediction_vs_glorys_c": median_abs_error_prediction_vs_glorys,
            "count_prediction_vs_glorys": count_prediction_vs_glorys,
            "median_abs_error_prediction_vs_argo_c": median_abs_error_prediction_vs_argo,
            "count_prediction_vs_argo": count_prediction_vs_argo,
            "median_prediction_profile_c": median_prediction_profile,
            "count_prediction_profile": count_prediction_profile,
            "median_glorys_profile_c": median_glorys_profile,
            "count_glorys_profile": count_glorys_profile,
            "median_argo_profile_c": median_argo_profile,
            "count_argo_profile": count_argo_profile,
        }
    )


def resolve_validation_summary_depth_axis_m(
    dataset: torch.utils.data.Dataset,
) -> np.ndarray:
    """Load the physical GLORYS depth axis from dataset metadata or a sample row."""
    if len(dataset) <= 0:
        raise RuntimeError("Validation summary export requires a non-empty dataset.")
    first_row = _dataset_rows(dataset)[0]
    sample = dataset[0]
    return _load_glorys_depth_axis_m(
        dataset,
        first_row,
        expected_size=int(sample["y"].shape[0]),
    )


def build_validation_summary_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    training_cfg: dict[str, Any],
    batch_size_override: int | None,
) -> DataLoader:
    """Create the deterministic dataloader used for the pooled validation export."""
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    batch_size = (
        int(batch_size_override)
        if batch_size_override is not None
        else int(dataloader_cfg.get("val_batch_size", 4))
    )
    num_workers = int(dataloader_cfg.get("val_num_workers", 0))
    persistent_workers = (
        bool(dataloader_cfg.get("val_persistent_workers", False)) and num_workers > 0
    )
    pin_memory = bool(dataloader_cfg.get("pin_memory", True))
    prefetch_factor = dataloader_cfg.get("prefetch_factor", 2)

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        # Keep row order stable so reruns over the same split are reproducible.
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def _figure_title_for_split(split: str) -> str:
    """Build the user-facing figure title for exported validation summary plots."""
    split_label = str(split).strip().lower()
    if split_label == "all":
        return "Median profile and absolute error across all dataset rows"
    return (
        f"Median profile and absolute error across " f"{split_label.capitalize()} split"
    )


def _figure_title_for_export(
    *,
    split: str,
    iso_year: int | None = None,
    iso_week: int | None = None,
) -> str:
    """Build the figure title including optional ISO-week context."""
    title = _figure_title_for_split(split)
    if iso_year is None or iso_week is None:
        return title
    return f"{title}\nWeek: ISO week {int(iso_year)}-W{int(iso_week):02d}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DepthDif inference across one dataset split and export pooled "
            "per-depth absolute error summaries against GLORYS and observed ARGO."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_INFERENCE_CONFIG,
        dest="config_path",
        help="Path to the pixel inference super-config yaml.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(PIXEL_SCENARIOS),
        default=None,
        help="High-level pixel inference scenario; derives data/model channel settings.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
        help=(
            "Override config values. Format: "
            "<data|training|model|inference>.<nested.path>=<yaml_value>. "
            "Repeat --set for multiple overrides."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        "--model-path-checkpoint",
        "--checkpoint",
        "--ckpt-path",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint override. Accepts the trained model .ckpt path "
            "and defaults to model config resolution."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "all"),
        help="Explicit dataset split to summarize.",
    )
    parser.add_argument("--year", type=int, default=None, help="ISO year filter.")
    parser.add_argument("--iso-week", type=int, default=None, help="ISO week filter.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to dataloader.val_batch_size then 4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Torch RNG seed so sampling stays reproducible.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory that receives the summary export folder.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Run directory name used under output-root.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Load checkpoint weights with strict=True instead of strict=False.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    config_bundle = load_pixel_inference_config(
        config_path_value=args.config_path,
        scenario_override=args.scenario,
        overrides=list(args.config_overrides or []),
        runtime_config_dir=Path("/tmp/depthdif_inference_configs")
        / f"validation_{os.getpid()}",
        write_snapshots=False,
    )
    model_cfg = config_bundle.model_cfg
    train_cfg = config_bundle.training_cfg
    args.model_config = config_bundle.effective_model_config_path
    args.data_config = config_bundle.effective_data_config_path
    args.train_config = config_bundle.effective_training_config_path

    dataset = build_validation_summary_dataset(args.data_config, split=str(args.split))
    selected_iso_year, selected_iso_week = (
        filter_validation_summary_dataset_by_iso_week(
            dataset,
            iso_year=args.year,
            iso_week=args.iso_week,
        )
    )
    depth_axis_m = resolve_validation_summary_depth_axis_m(dataset)
    accumulator = create_validation_error_summary_accumulator(int(depth_axis_m.size))

    dataloader = build_validation_summary_dataloader(
        dataset,
        training_cfg=train_cfg,
        batch_size_override=args.batch_size,
    )
    dataloader_cfg = dict(train_cfg.get("dataloader", {}))
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(dataloader_cfg.get("val_batch_size", 4))
    )

    datamodule = DepthTileDataModule(
        dataset=dataset,
        val_dataset=dataset,
        dataloader_cfg=dataloader_cfg,
    )
    model = build_model(
        model_config_path=str(args.model_config),
        data_config_path=str(args.data_config),
        training_config_path=str(args.train_config),
        model_cfg=model_cfg,
        datamodule=datamodule,
    )
    ckpt_path = resolve_checkpoint_path(args.checkpoint_path, model_cfg)
    if ckpt_path is None:
        raise RuntimeError("Validation summary export requires a resolved checkpoint.")
    weight_source = load_checkpoint_weights(
        model,
        ckpt_path,
        strict=bool(args.strict_load),
    )
    device = choose_device(args.device)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} ({weight_source} weights)")

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Validation summary", unit="batch")
        ):
            batch = to_device(batch, device)
            pred = model.predict_step(batch, batch_idx=batch_idx)
            y_hat_denorm = pred["y_hat_denorm"]
            x_denorm = temperature_normalize(mode="denorm", tensor=batch["x"])
            y_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
            update_validation_error_summary_accumulator(
                accumulator,
                x_denorm=x_denorm,
                y_denorm=y_denorm,
                y_hat_denorm=y_hat_denorm,
                x_valid_mask=batch["x_valid_mask"],
                y_valid_mask=batch["y_valid_mask"],
            )

    summary_df = build_validation_error_summary_dataframe(
        accumulator,
        depth_axis_m=depth_axis_m,
    )

    run_dir = Path(args.output_root) / str(args.output_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / DEFAULT_CSV_NAME
    summary_df.to_csv(csv_path, index=False)

    median_prediction_profile = summary_df["median_prediction_profile_c"].to_numpy(
        dtype=np.float64
    )
    median_glorys_profile = summary_df["median_glorys_profile_c"].to_numpy(
        dtype=np.float64
    )
    median_argo_profile = summary_df["median_argo_profile_c"].to_numpy(dtype=np.float64)
    median_abs_error_prediction_vs_glorys = summary_df[
        "median_abs_error_prediction_vs_glorys_c"
    ].to_numpy(dtype=np.float64)
    median_abs_error_prediction_vs_argo = summary_df[
        "median_abs_error_prediction_vs_argo_c"
    ].to_numpy(dtype=np.float64)
    figure_title = _figure_title_for_export(
        split=str(args.split),
        iso_year=selected_iso_year,
        iso_week=selected_iso_week,
    )

    error_plot_path = save_average_glorys_profile_error_plot(
        output_path=run_dir / DEFAULT_ERROR_PLOT_NAME,
        mean_abs_error_prediction_vs_glorys=median_abs_error_prediction_vs_glorys,
        mean_abs_error_prediction_vs_argo=median_abs_error_prediction_vs_argo,
        depth_axis=depth_axis_m,
        figure_title=f"{figure_title}\nError summary",
    )
    profile_error_plot_path = save_average_glorys_profile_and_error_plot(
        output_path=run_dir / DEFAULT_PROFILE_ERROR_PLOT_NAME,
        mean_argo_profile_c=median_argo_profile,
        mean_prediction_profile_c=median_prediction_profile,
        mean_glorys_profile_c=median_glorys_profile,
        mean_abs_error_prediction_vs_glorys=median_abs_error_prediction_vs_glorys,
        mean_abs_error_prediction_vs_argo=median_abs_error_prediction_vs_argo,
        depth_axis=depth_axis_m,
        figure_title=figure_title,
    )

    run_summary = {
        "checkpoint_path": str(ckpt_path),
        "model_config": str(args.model_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "device": str(device),
        "batch_size": int(batch_size),
        "split": str(args.split),
        "iso_year": selected_iso_year,
        "iso_week": selected_iso_week,
        "run_dir": str(run_dir),
        "dataset_row_count": int(len(dataset)),
        "processed_sample_count": int(accumulator.sample_count),
        "processed_batch_count": int(accumulator.batch_count),
        "depth_level_count": int(depth_axis_m.size),
        "summary_statistic": "median",
        "csv_path": str(csv_path.name),
        "error_plot_path": str(Path(error_plot_path).name),
        "profile_error_plot_path": str(Path(profile_error_plot_path).name),
    }
    with (run_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    print(f"Wrote validation summary CSV: {csv_path}")
    print(f"Wrote validation summary error plot: {error_plot_path}")
    print(f"Wrote validation summary profile/error plot: {profile_error_plot_path}")
    print(f"Wrote validation summary metadata: {run_dir / 'run_summary.yaml'}")


if __name__ == "__main__":
    main()
