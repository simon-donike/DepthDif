# Example:
# /work/envs/depth/bin/python src/depth_recon/experiments/compare_ddpm_ddim_step_grid.py \
#   --config src/depth_recon/configs/px_space/inference_super_config.yaml \
#   --checkpoint logs/<run>/best.ckpt \
#   --output temp/ddim_sampling/ddpm_ddim_step_grid.png \
#   --loader-split val \
#   --device auto \
#   --seed 7 \
#   --batch-size 10 \
#   --depth-level 0 \
#   --ddim-steps 1000,800,500,200,100,50 \
#   --ddim-eta 0.0 \
#   --ddim-temperature 1.0
"""Plot one temperature validation batch with DDPM and DDIM comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.configs.config_resolver_pixel import (
    DEFAULT_PIXEL_INFERENCE_CONFIG_PATH,
    load_pixel_inference_config,
)
from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    choose_device,
    load_checkpoint_weights,
    resolve_checkpoint_path,
    resolve_model_type,
    to_device,
)
from depth_recon.models.diffusion.DenoisingDiffusionProcess.samplers import (
    DDIM_Sampler,
)
from depth_recon.utils.normalizations import (
    PLOT_CMAP,
    PLOT_TEMP_MAX,
    PLOT_TEMP_MIN,
    temperature_normalize,
)

CONFIG_PATH = DEFAULT_PIXEL_INFERENCE_CONFIG_PATH
CHECKPOINT_PATH: str | None = None
OUTPUT_PATH = Path("temp/ddim_sampling/ddpm_ddim_step_grid.png")

LOADER_SPLIT = "val"
DEVICE = "auto"
SEED = 7
STRICT_LOAD = False
BATCH_SIZE = 10
MAX_BATCH_SIZE = 10
DEPTH_LEVEL = 0
DDIM_STEPS: tuple[int, ...] = (1000, 800, 500, 200, 100, 50)
DDIM_ETA = 0.0
DDIM_TEMPERATURE = 1.0


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the temperature sampler comparison script."""
    parser = argparse.ArgumentParser(
        description=(
            "Load one temperature validation batch and plot original, DDPM, "
            "and DDIM predictions for several DDIM step counts."
        )
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--loader-split", choices=["train", "val"], default=LOADER_SPLIT
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of batch rows to plot, capped at {MAX_BATCH_SIZE}.",
    )
    parser.add_argument("--depth-level", type=int, default=DEPTH_LEVEL)
    parser.add_argument(
        "--ddim-steps",
        default=",".join(str(step) for step in DDIM_STEPS),
        help="Comma-separated DDIM step counts.",
    )
    parser.add_argument("--ddim-eta", type=float, default=DDIM_ETA)
    parser.add_argument("--ddim-temperature", type=float, default=DDIM_TEMPERATURE)
    return parser.parse_args()


def _parse_step_counts(raw_value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated list of positive step counts."""
    steps: list[int] = []
    for raw_step in str(raw_value).split(","):
        step_text = raw_step.strip()
        if not step_text:
            continue
        step = int(step_text)
        if step < 1:
            raise ValueError("--ddim-steps values must be >= 1.")
        steps.append(step)
    if not steps:
        raise ValueError("--ddim-steps must contain at least one value.")
    return tuple(steps)


def _first_n_batch(batch: dict[str, Any], n_items: int) -> dict[str, Any]:
    """Return the first n batch items while preserving non-tensor metadata."""
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[:n_items]
        elif isinstance(value, list):
            sliced[key] = value[:n_items]
        else:
            sliced[key] = value
    return sliced


def _batch_with_sampler(
    batch: dict[str, Any], sampler: torch.nn.Module
) -> dict[str, Any]:
    """Attach one sampler to a shallow batch copy."""
    out = dict(batch)
    out["sampler"] = sampler
    return out


def _prediction_for_sampler(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sampler: torch.nn.Module,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run prediction with a fixed seed so sampler outputs are comparable."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        return model.predict_step(_batch_with_sampler(batch, sampler), batch_idx=0)


def _training_betas(model: torch.nn.Module) -> torch.Tensor:
    """Return the beta schedule used by the loaded diffusion process."""
    diffusion_process = getattr(model, "model", None)
    forward_process = getattr(diffusion_process, "forward_process", None)
    betas = getattr(forward_process, "betas", None)
    if betas is None:
        sampler = getattr(diffusion_process, "sampler", None)
        betas = getattr(sampler, "betas", None)
    if betas is None:
        raise RuntimeError("Could not locate the model training beta schedule.")
    return betas.detach().float().cpu()


def _build_ddim_sampler(
    *,
    model: torch.nn.Module,
    train_betas: torch.Tensor,
    ddim_steps: int,
    eta: float,
    temperature: float,
) -> DDIM_Sampler:
    """Build a DDIM sampler that matches the loaded model's training schedule."""
    diffusion_process = getattr(model, "model", None)
    parameterization = str(getattr(diffusion_process, "parameterization", "epsilon"))
    return DDIM_Sampler(
        num_timesteps=min(int(ddim_steps), int(train_betas.numel())),
        train_timesteps=int(train_betas.numel()),
        betas=train_betas,
        parameterization=parameterization,
        # The model predicts in standardized temperature space; do not clip physically.
        clip_sample=False,
        eta=float(eta),
        temperature=float(temperature),
    )


def _band_image(
    tensor: torch.Tensor, sample_idx: int, depth_level: int
) -> torch.Tensor:
    """Extract one image band, clamping depth to available channels."""
    if tensor.ndim == 4:
        band_idx = int(max(0, min(int(depth_level), int(tensor.size(1)) - 1)))
        return tensor[int(sample_idx), band_idx].detach().float().cpu()
    if tensor.ndim == 3:
        return tensor[int(sample_idx)].detach().float().cpu()
    return tensor.detach().float().cpu()


def _band_mask(
    mask: torch.Tensor | None,
    *,
    sample_idx: int,
    depth_level: int,
) -> torch.Tensor | None:
    """Extract a boolean mask band from 2D, 3D, or 4D masks."""
    if mask is None:
        return None
    if mask.ndim == 4:
        band_idx = 0 if int(mask.size(1)) == 1 else int(depth_level)
        band_idx = int(max(0, min(band_idx, int(mask.size(1)) - 1)))
        return mask[int(sample_idx), band_idx].detach().cpu() > 0.5
    if mask.ndim == 3:
        return mask[int(sample_idx)].detach().cpu() > 0.5
    if mask.ndim == 2:
        return mask.detach().cpu() > 0.5
    return None


def _masked_array(
    tensor: torch.Tensor,
    *,
    sample_idx: int,
    depth_level: int,
    masks: list[torch.Tensor | None],
) -> np.ndarray:
    """Convert one panel to a NumPy image with invalid pixels hidden as NaN."""
    arr = _band_image(tensor, sample_idx, depth_level).numpy().astype(np.float32)
    valid = np.isfinite(arr)
    for mask in masks:
        if mask is not None:
            valid &= mask.numpy().astype(bool, copy=False)
    arr[~valid] = np.nan
    return arr


def _comparison_rows(
    *,
    batch: dict[str, Any],
    pred_ddpm: dict[str, Any],
    pred_ddim_by_steps: dict[int, dict[str, Any]],
    depth_level: int,
) -> list[dict[str, np.ndarray]]:
    """Build plotted row dictionaries for every item in the loaded batch."""
    target_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
    batch_size = int(batch["y"].size(0))

    rows: list[dict[str, np.ndarray]] = []
    for sample_idx in range(batch_size):
        land_mask = _band_mask(
            batch.get("land_mask"), sample_idx=sample_idx, depth_level=depth_level
        )
        y_valid_mask = _band_mask(
            batch.get("y_valid_mask"), sample_idx=sample_idx, depth_level=depth_level
        )
        output_masks = [land_mask, y_valid_mask]
        row = {
            "sample_number": np.asarray(sample_idx, dtype=np.int64),
            "original": _masked_array(
                target_denorm,
                sample_idx=sample_idx,
                depth_level=depth_level,
                masks=output_masks,
            ),
            "ddpm": _masked_array(
                pred_ddpm["y_hat_denorm"],
                sample_idx=sample_idx,
                depth_level=depth_level,
                masks=output_masks,
            ),
        }
        for steps, pred in pred_ddim_by_steps.items():
            row[f"ddim_{steps}"] = _masked_array(
                pred["y_hat_denorm"],
                sample_idx=sample_idx,
                depth_level=depth_level,
                masks=output_masks,
            )
        rows.append(row)
    return rows


def _save_grid(
    *,
    rows: list[dict[str, np.ndarray]],
    output_path: Path,
    ddim_steps: tuple[int, ...],
    depth_level: int,
) -> None:
    """Save the comparison grid to a PNG file."""
    columns = [("original", "Original"), ("ddpm", "DDPM 1000")]
    columns.extend((f"ddim_{steps}", f"DDIM {steps}") for steps in ddim_steps)
    cmap = plt.get_cmap(PLOT_CMAP).copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(2.55 * len(columns), 2.35 * len(rows)),
        squeeze=False,
    )
    try:
        for row_idx, row in enumerate(rows):
            for col_idx, (key, title) in enumerate(columns):
                ax = axes[row_idx, col_idx]
                ax.imshow(
                    row[key],
                    cmap=cmap,
                    vmin=float(PLOT_TEMP_MIN),
                    vmax=float(PLOT_TEMP_MAX),
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(title, fontsize=9)
            axes[row_idx, 0].set_ylabel(
                f"sample {int(row['sample_number'])}",
                rotation=90,
                fontsize=8,
            )
        fig.suptitle(f"temperature, depth level {int(depth_level)}", fontsize=11)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def main() -> None:
    """Run DDPM/DDIM sampler comparisons on one temperature validation batch."""
    args = _parse_args()
    if float(args.ddim_temperature) < 0.0:
        raise ValueError("--ddim-temperature must be >= 0.0.")
    if float(args.ddim_eta) < 0.0:
        raise ValueError("--ddim-eta must be >= 0.0.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1.")

    torch.manual_seed(int(args.seed))
    ddim_steps = _parse_step_counts(args.ddim_steps)
    output_path = Path(args.output)

    setup_progress = tqdm(total=6, desc="Preparing comparison", unit="stage")
    config_bundle = load_pixel_inference_config(
        config_path_value=args.config,
        scenario_override="temperature",
        runtime_config_dir=output_path.parent / ".effective_configs",
        write_snapshots=False,
    )
    model_cfg = config_bundle.model_cfg
    data_cfg = config_bundle.data_cfg
    training_cfg = config_bundle.training_cfg
    resolve_model_type(model_cfg)
    setup_progress.set_postfix_str("loaded configs")
    setup_progress.update()

    batch_size = min(int(args.batch_size), MAX_BATCH_SIZE)
    if batch_size < int(args.batch_size):
        print(f"Capped --batch-size to {MAX_BATCH_SIZE} plotted rows.")
    dataloader_cfg = training_cfg.setdefault("dataloader", {})
    data_dataloader_cfg = data_cfg.setdefault("dataloader", {})
    batch_size_key = "val_batch_size" if args.loader_split == "val" else "batch_size"
    num_workers_key = "val_num_workers" if args.loader_split == "val" else "num_workers"
    dataloader_cfg[batch_size_key] = batch_size
    data_dataloader_cfg[batch_size_key] = batch_size
    dataloader_cfg[num_workers_key] = 0
    data_dataloader_cfg[num_workers_key] = 0
    # Do not override val_shuffle; the validation loader is intentionally shuffled.

    device = choose_device(str(args.device))
    dataset = build_dataset(
        config_bundle.effective_data_config_path, data_cfg.get("dataset", {})
    )
    datamodule = build_datamodule(
        dataset=dataset,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
    )
    datamodule.setup("fit")
    setup_progress.set_postfix_str("built datamodule")
    setup_progress.update()

    model = build_model(
        model_config_path=config_bundle.effective_model_config_path,
        data_config_path=config_bundle.effective_data_config_path,
        training_config_path=config_bundle.effective_training_config_path,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )
    setup_progress.set_postfix_str("built model")
    setup_progress.update()

    ckpt_path = resolve_checkpoint_path(args.checkpoint, model_cfg)
    if ckpt_path is not None:
        weight_source = load_checkpoint_weights(
            model,
            ckpt_path,
            strict=bool(STRICT_LOAD),
        )
        print(f"Loaded checkpoint: {ckpt_path} ({weight_source} weights)")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()
    setup_progress.set_postfix_str("loaded checkpoint")
    setup_progress.update()

    train_betas = _training_betas(model)
    diffusion_process = getattr(model, "model", None)
    ddpm_sampler = getattr(diffusion_process, "sampler", None)
    if ddpm_sampler is None:
        raise RuntimeError("Could not locate the model DDPM sampler.")

    ddim_samplers = {
        steps: _build_ddim_sampler(
            model=model,
            train_betas=train_betas,
            ddim_steps=steps,
            eta=float(args.ddim_eta),
            temperature=float(args.ddim_temperature),
        ).to(device)
        for steps in ddim_steps
    }
    setup_progress.set_postfix_str("built samplers")
    setup_progress.update()

    loader = (
        datamodule.train_dataloader()
        if args.loader_split == "train"
        else datamodule.val_dataloader()
    )
    batch = _first_n_batch(to_device(next(iter(loader)), device), batch_size)
    if int(batch["y"].size(0)) <= 0:
        raise RuntimeError("Loaded batch is empty.")
    setup_progress.set_postfix_str("loaded batch")
    setup_progress.update()
    setup_progress.close()

    sampler_jobs: list[tuple[str, int | None, torch.nn.Module]] = [
        ("DDPM 1000", None, ddpm_sampler)
    ]
    sampler_jobs.extend(
        (f"DDIM {steps}", steps, sampler) for steps, sampler in ddim_samplers.items()
    )

    pred_ddpm: dict[str, Any] | None = None
    pred_ddim_by_steps: dict[int, dict[str, Any]] = {}
    for label, steps, sampler in tqdm(
        sampler_jobs, desc="Running samplers", unit="sampler"
    ):
        tqdm.write(f"Running {label}")
        pred = _prediction_for_sampler(
            model,
            batch,
            sampler,
            seed=int(args.seed),
        )
        if steps is None:
            pred_ddpm = pred
        else:
            pred_ddim_by_steps[int(steps)] = pred
    if pred_ddpm is None:
        raise RuntimeError("DDPM prediction was not produced.")

    rows = _comparison_rows(
        batch=batch,
        pred_ddpm=pred_ddpm,
        pred_ddim_by_steps=pred_ddim_by_steps,
        depth_level=int(args.depth_level),
    )
    _save_grid(
        rows=rows,
        output_path=output_path,
        ddim_steps=ddim_steps,
        depth_level=int(args.depth_level),
    )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
