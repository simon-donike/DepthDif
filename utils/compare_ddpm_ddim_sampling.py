"""Compare DDPM and DDIM sampling on real Argo inputs for one checkpoint.

Typical CLI:
    /work/envs/depth/bin/python utils/compare_ddpm_ddim_sampling.py --depth-level 0   --num-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from models.difFF.DenoisingDiffusionProcess.samplers import DDIM_Sampler
from utils.normalizations import (
    PLOT_CMAP,
    PLOT_TEMP_MAX,
    PLOT_TEMP_MIN,
    temperature_normalize,
)

MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"
DATA_CONFIG_PATH = "configs/px_space/data_ostia_argo_disk_actual.yaml"
TRAIN_CONFIG_PATH = "configs/px_space/training_config.yaml"
CHECKPOINT_PATH: str | None = None

LOADER_SPLIT = "val"
DEVICE = "auto"
SEED = 7
STRICT_LOAD = False

NUM_SAMPLES = 100
BATCH_SIZE = 5
DEPTH_LEVEL = 0
DDIM_STEPS = 100
DDIM_TEMPERATURE = 0.5
ROWS_PER_FIGURE = 10
OUTPUT_DIR = Path("temp/ddpm_ddim_sampling_comparison_real_argo")


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides while keeping the script usable by editing constants."""
    parser = argparse.ArgumentParser(
        description="Plot input, DDPM, DDIM eta=0, DDIM eta=1, and target samples."
    )
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--data-config", default=DATA_CONFIG_PATH)
    parser.add_argument("--train-config", default=TRAIN_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--loader-split", choices=["train", "val"], default=LOADER_SPLIT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--depth-level", type=int, default=DEPTH_LEVEL)
    parser.add_argument("--ddim-steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--ddim-temperature", type=float, default=DDIM_TEMPERATURE)
    parser.add_argument("--rows-per-figure", type=int, default=ROWS_PER_FIGURE)
    return parser.parse_args()


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


def _batch_with_sampler(batch: dict[str, Any], sampler: torch.nn.Module) -> dict[str, Any]:
    """Attach one sampler without mutating the dataloader-owned batch."""
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
    """Run prediction with a fixed seed so all samplers start from comparable noise."""
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
        # Predictions are in the model's standardized temperature domain.
        clip_sample=False,
        eta=float(eta),
        temperature=float(temperature),
    )


def _band_image(tensor: torch.Tensor, sample_idx: int, depth_level: int) -> torch.Tensor:
    """Extract one image band, clamping depth when a condition has fewer channels."""
    if tensor.ndim == 4:
        band_idx = int(max(0, min(int(depth_level), int(tensor.size(1)) - 1)))
        return tensor[int(sample_idx), band_idx].detach().float().cpu()
    if tensor.ndim == 3:
        return tensor[int(sample_idx)].detach().float().cpu()
    return tensor.detach().float().cpu()


def _band_mask(
    mask: torch.Tensor | None,
    sample_idx: int,
    depth_level: int,
) -> torch.Tensor | None:
    """Extract a boolean mask band while accepting either 1-channel or per-depth masks."""
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
    sample_idx: int,
    depth_level: int,
    *,
    masks: list[torch.Tensor | None],
) -> np.ndarray:
    """Convert one plotted panel to Celsius with invalid pixels hidden as NaN."""
    arr = _band_image(tensor, sample_idx, depth_level).numpy().astype(np.float32)
    valid = np.ones(arr.shape, dtype=bool)
    for mask in masks:
        if mask is not None:
            valid &= mask.numpy().astype(bool, copy=False)
    arr[~valid] = np.nan
    return arr


def _page_path(output_dir: Path, page_idx: int) -> Path:
    """Return the output path for one comparison page."""
    return output_dir / f"ddpm_ddim_comparison_page_{int(page_idx):03d}.png"


def _save_page(
    *,
    rows: list[dict[str, np.ndarray]],
    output_dir: Path,
    page_idx: int,
    ddim_steps: int,
    ddim_temperature: float,
) -> Path:
    """Save one page of side-by-side sampler comparisons."""
    columns = [
        ("input", "Input"),
        ("ddpm", "DDPM"),
        (
            "ddim_eta0",
            f"DDIM eta=0, {int(ddim_steps)} steps, temp={float(ddim_temperature):g}",
        ),
        (
            "ddim_eta1",
            f"DDIM eta=1, {int(ddim_steps)} steps, temp={float(ddim_temperature):g}",
        ),
        ("target", "Ground truth"),
    ]
    cmap = plt.get_cmap(PLOT_CMAP).copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.2 * len(columns), 3.0 * len(rows)),
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
                    ax.set_title(title, fontsize=10)
            axes[row_idx, 0].set_ylabel(
                f"sample {int(row['sample_number'])}",
                rotation=90,
                fontsize=8,
            )
        fig.tight_layout()
        output_path = _page_path(output_dir, page_idx)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)
    return output_path


def _append_plot_rows(
    *,
    rows: list[dict[str, np.ndarray]],
    batch: dict[str, Any],
    pred_ddpm: dict[str, Any],
    pred_ddim_eta0: dict[str, Any],
    pred_ddim_eta1: dict[str, Any],
    sample_offset: int,
    depth_level: int,
) -> None:
    """Append plotted rows for every item in one processed batch."""
    x_denorm = temperature_normalize(mode="denorm", tensor=batch["x"].detach().float())
    y_denorm = temperature_normalize(mode="denorm", tensor=batch["y"].detach().float())
    batch_size = int(batch["x"].size(0))

    for sample_idx in range(batch_size):
        land_mask = _band_mask(batch.get("land_mask"), sample_idx, depth_level)
        x_valid_mask = _band_mask(batch.get("x_valid_mask"), sample_idx, depth_level)
        y_valid_mask = _band_mask(batch.get("y_valid_mask"), sample_idx, depth_level)
        output_masks = [land_mask, y_valid_mask]

        rows.append(
            {
                "sample_number": np.asarray(sample_offset + sample_idx, dtype=np.int64),
                "input": _masked_array(
                    x_denorm,
                    sample_idx,
                    depth_level,
                    masks=[land_mask, x_valid_mask],
                ),
                "ddpm": _masked_array(
                    pred_ddpm["y_hat_denorm"],
                    sample_idx,
                    depth_level,
                    masks=output_masks,
                ),
                "ddim_eta0": _masked_array(
                    pred_ddim_eta0["y_hat_denorm"],
                    sample_idx,
                    depth_level,
                    masks=output_masks,
                ),
                "ddim_eta1": _masked_array(
                    pred_ddim_eta1["y_hat_denorm"],
                    sample_idx,
                    depth_level,
                    masks=output_masks,
                ),
                "target": _masked_array(
                    y_denorm,
                    sample_idx,
                    depth_level,
                    masks=output_masks,
                ),
            }
        )


def main() -> None:
    """Run the sampler comparison script."""
    args = _parse_args()
    if float(args.ddim_temperature) < 0.0:
        raise ValueError("--ddim-temperature must be >= 0.0.")
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(args.model_config)
    data_cfg = load_yaml(args.data_config)
    training_cfg = load_yaml(args.train_config)
    resolve_model_type(model_cfg)

    dataloader_cfg = training_cfg.setdefault("dataloader", {})
    dataloader_cfg["val_batch_size" if args.loader_split == "val" else "batch_size"] = (
        int(args.batch_size)
    )
    dataloader_cfg["val_shuffle" if args.loader_split == "val" else "shuffle"] = False
    dataloader_cfg["val_num_workers" if args.loader_split == "val" else "num_workers"] = 0
    # Keep the first 100 samples reproducible even when the data config asks for shuffled validation.
    data_cfg.setdefault("dataloader", {})["val_shuffle"] = False

    device = choose_device(str(args.device))
    dataset = build_dataset(args.data_config, data_cfg.get("dataset", {}))
    datamodule = build_datamodule(
        dataset=dataset,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
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
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=bool(STRICT_LOAD))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    train_betas = _training_betas(model)
    diffusion_process = getattr(model, "model", None)
    ddpm_sampler = getattr(diffusion_process, "sampler", None)
    if ddpm_sampler is None:
        raise RuntimeError("Could not locate the model DDPM sampler.")
    ddim_eta0_sampler = _build_ddim_sampler(
        model=model,
        train_betas=train_betas,
        ddim_steps=int(args.ddim_steps),
        eta=0.0,
        temperature=float(args.ddim_temperature),
    )
    ddim_eta1_sampler = _build_ddim_sampler(
        model=model,
        train_betas=train_betas,
        ddim_steps=int(args.ddim_steps),
        eta=1.0,
        temperature=float(args.ddim_temperature),
    )

    loader = (
        datamodule.train_dataloader()
        if args.loader_split == "train"
        else datamodule.val_dataloader()
    )
    pending_rows: list[dict[str, np.ndarray]] = []
    saved_pages: list[Path] = []
    processed = 0
    page_idx = 0

    for batch in loader:
        if processed >= int(args.num_samples):
            break
        remaining = int(args.num_samples) - processed
        batch = _first_n_batch(to_device(batch, device), min(remaining, int(args.batch_size)))
        if int(batch["x"].size(0)) <= 0:
            continue

        pred_ddpm = _prediction_for_sampler(
            model,
            batch,
            ddpm_sampler,
            seed=int(args.seed) + processed,
        )
        pred_ddim_eta0 = _prediction_for_sampler(
            model,
            batch,
            ddim_eta0_sampler,
            seed=int(args.seed) + processed,
        )
        pred_ddim_eta1 = _prediction_for_sampler(
            model,
            batch,
            ddim_eta1_sampler,
            seed=int(args.seed) + processed,
        )

        _append_plot_rows(
            rows=pending_rows,
            batch=batch,
            pred_ddpm=pred_ddpm,
            pred_ddim_eta0=pred_ddim_eta0,
            pred_ddim_eta1=pred_ddim_eta1,
            sample_offset=processed,
            depth_level=int(args.depth_level),
        )
        processed += int(batch["x"].size(0))
        print(f"Processed {processed}/{int(args.num_samples)} samples")

        while len(pending_rows) >= int(args.rows_per_figure):
            page_rows = pending_rows[: int(args.rows_per_figure)]
            pending_rows = pending_rows[int(args.rows_per_figure) :]
            saved_path = _save_page(
                rows=page_rows,
                output_dir=output_dir,
                page_idx=page_idx,
                ddim_steps=int(args.ddim_steps),
                ddim_temperature=float(args.ddim_temperature),
            )
            saved_pages.append(saved_path)
            print(f"Saved plot: {saved_path}")
            page_idx += 1

    if pending_rows:
        saved_path = _save_page(
            rows=pending_rows,
            output_dir=output_dir,
            page_idx=page_idx,
            ddim_steps=int(args.ddim_steps),
            ddim_temperature=float(args.ddim_temperature),
        )
        saved_pages.append(saved_path)
        print(f"Saved plot: {saved_path}")

    print(f"Saved {len(saved_pages)} comparison page(s) under {output_dir}")


if __name__ == "__main__":
    main()
