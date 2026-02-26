from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from inference import (
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

MODEL_CONFIG_PATH = "configs/model_config.yaml"
DATA_CONFIG_PATH = "configs/data_config.yaml"
TRAIN_CONFIG_PATH = "configs/training_config.yaml"

LOADER_SPLIT = "val"
DEVICE = "auto"
SEED = 7
CHECKPOINT_PATH: str | None = None
# Optional explicit checkpoint override loaded right after model instantiation.
# Set to None to fall back to model.resume_checkpoint from model config.
CHECKPOINT_OVERRIDE_PATH: str | None = "logs/2026-02-25_12-32-00/last.ckpt"
STRICT_LOAD = False
OUTPUT_DIR = Path("temp/images")


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
    if force_all_invalid and "valid_mask" in case:
        # valid_mask=0 everywhere means no observed x pixels are marked as known.
        case["valid_mask"] = torch.zeros_like(case["valid_mask"])

    return case


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    """Format compact summary statistics for one tensor."""
    t = tensor.detach().float()
    return (
        f"{name}: shape={tuple(t.shape)} "
        f"min={t.min().item():.4f} max={t.max().item():.4f} "
        f"mean={t.mean().item():.4f} std={t.std(unbiased=False).item():.4f}"
    )


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


def _save_case_plot(
    *,
    case_name: str,
    case_batch: dict[str, Any],
    pred: dict[str, Any],
) -> None:
    """Save one reconstruction grid image inspired by validation reconstruction logging."""
    x_denorm = temperature_normalize(mode="denorm", tensor=case_batch["x"].detach().float())
    y_target_denorm = temperature_normalize(
        mode="denorm", tensor=case_batch["y"].detach().float()
    )
    y_hat_denorm = pred["y_hat_denorm"].detach().float()
    eo_denorm = None
    if "eo" in case_batch:
        eo_denorm = temperature_normalize(mode="denorm", tensor=case_batch["eo"].detach().float())

    valid_mask_i = _mask_for_sample(case_batch.get("valid_mask"))
    land_mask_i = _mask_for_sample(case_batch.get("land_mask"))

    n_bands = int(y_hat_denorm.size(1)) if y_hat_denorm.ndim == 4 else 1
    ncols = 4 if eo_denorm is not None else 3
    if valid_mask_i is not None:
        ncols += 1
    if land_mask_i is not None:
        ncols += 1

    fig, axes = plt.subplots(n_bands, ncols, figsize=(4 * ncols, 2.8 * n_bands), squeeze=False)
    try:
        for band_idx in range(n_bands):
            valid_band = valid_mask_i
            if valid_band is not None and valid_band.ndim == 3:
                valid_band = valid_band[min(band_idx, int(valid_band.size(0)) - 1)]
            land_band = land_mask_i
            if land_band is not None and land_band.ndim == 3:
                land_band = land_band[min(band_idx, int(land_band.size(0)) - 1)]

            x_img = _plot_band_image(x_denorm, band_idx=band_idx, mask=land_band)
            y_hat_img = _plot_band_image(y_hat_denorm, band_idx=band_idx, mask=land_band)
            y_target_img = _plot_band_image(y_target_denorm, band_idx=band_idx, mask=land_band)
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
                axes[band_idx, col].imshow(
                    valid_band.detach().float().cpu().numpy(),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[band_idx, col].set_axis_off()
                if band_idx == 0:
                    axes[band_idx, col].set_title("Valid mask")
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
        output_path = OUTPUT_DIR / f"{case_name}.png"
        fig.savefig(output_path, dpi=160)
        print(f"Saved plot: {output_path}")
    finally:
        plt.close(fig)


def main() -> None:
    """Run EO/X ablation experiments on one config-loaded dataloader sample."""
    torch.manual_seed(int(SEED))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    data_cfg = load_yaml(DATA_CONFIG_PATH)
    training_cfg = load_yaml(TRAIN_CONFIG_PATH)
    resolve_model_type(model_cfg)

    device = choose_device(DEVICE)

    dataset = build_dataset(DATA_CONFIG_PATH, data_cfg.get("dataset", {}))
    datamodule = build_datamodule(dataset=dataset, data_cfg=data_cfg, training_cfg=training_cfg)
    datamodule.setup("fit")

    model = build_model(
        model_config_path=MODEL_CONFIG_PATH,
        data_config_path=DATA_CONFIG_PATH,
        training_config_path=TRAIN_CONFIG_PATH,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )

    ckpt_path = resolve_checkpoint_path(CHECKPOINT_OVERRIDE_PATH or CHECKPOINT_PATH, model_cfg)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=bool(STRICT_LOAD))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    loader = datamodule.train_dataloader() if LOADER_SPLIT == "train" else datamodule.val_dataloader()
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
            hparams.get("coord_conditioning_enabled", False)
            if hasattr(hparams, "get")
            else False,
        )
    )
    date_conditioning_enabled = bool(
        getattr(
            getattr(model, "model", None),
            "date_conditioning_enabled",
            hparams.get("date_conditioning_enabled", False)
            if hasattr(hparams, "get")
            else False,
        )
    )

    if coord_conditioning_enabled and "coords" not in batch:
        raise RuntimeError("Model has coord conditioning enabled, but batch has no 'coords'.")
    if date_conditioning_enabled and "date" not in batch:
        raise RuntimeError("Model has date conditioning enabled, but batch has no 'date'.")

    for case_name, case_batch in cases:
        with torch.no_grad():
            pred = model.predict_step(case_batch, batch_idx=0)

        _save_case_plot(case_name=case_name, case_batch=case_batch, pred=pred)

        print(f"\n=== {case_name} ===")
        print(_summarize_tensor("input_x", case_batch["x"]))
        if "eo" in case_batch:
            print(_summarize_tensor("input_eo", case_batch["eo"]))
        if "valid_mask" in case_batch:
            print(_summarize_tensor("input_valid_mask", case_batch["valid_mask"]))
        if "coords" in case_batch:
            print(f"coords: {case_batch['coords'].detach().cpu().numpy().tolist()}")
        if "date" in case_batch:
            print(f"date: {case_batch['date'].detach().cpu().numpy().tolist()}")

        print(_summarize_tensor("y_hat", pred["y_hat"]))
        print(_summarize_tensor("y_hat_denorm", pred["y_hat_denorm"]))


if __name__ == "__main__":
    main()
