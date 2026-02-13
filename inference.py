from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from data.dataset_temp_v1 import SurfaceTempPatchLightDataset
from models.difFF import PixelDiffusionConditional

# ----------------------------
# In-script settings
# ----------------------------
MODEL_CONFIG_PATH = "configs/model_config.yaml"
DATA_CONFIG_PATH = "configs/data_config.yaml"
TRAIN_CONFIG_PATH = "configs/training_config.yaml"

# Optional explicit checkpoint path. If None, uses model.resume_checkpoint from model config.
CHECKPOINT_PATH: str | None = None

# "dataloader" or "random"
MODE = "dataloader" # or random

# Used when MODE == "dataloader": "train" or "val"
LOADER_SPLIT = "val"

# "auto", "cpu", or "cuda"
DEVICE = "auto"

SEED = 7
STRICT_LOAD = False
INCLUDE_INTERMEDIATES = False

# Used when MODE == "random"
RANDOM_BATCH_SIZE = 2
RANDOM_HEIGHT: int | None = None
RANDOM_WIDTH: int | None = None


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_variant(ds_cfg: dict[str, Any], data_config_path: str) -> str:
    variant = ds_cfg.get("dataset_variant", ds_cfg.get("variant", None))
    if variant is None:
        stem = Path(data_config_path).stem.lower()
        if "4band" in stem or "eo" in stem:
            return "eo_4band"
        return "temp_v1"
    return str(variant).strip().lower()


def build_dataset(data_config_path: str, ds_cfg: dict[str, Any]):
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant in {"temp_v1", "single_band", "1band", "default"}:
        return SurfaceTempPatchLightDataset.from_config(data_config_path, split="all")
    if dataset_variant in {"eo_4band", "4band_eo", "4bands"}:
        return SurfaceTempPatch4BandsLightDataset.from_config(data_config_path, split="all")
    raise ValueError(
        f"Unsupported dataset variant '{dataset_variant}'. Expected one of ['temp_v1', 'eo_4band']."
    )


def resolve_model_type(model_cfg: dict[str, Any]) -> str:
    model_type = str(model_cfg.get("model", {}).get("model_type", "cond_px_dif")).strip()
    if model_type == "cond_px_dif":
        return model_type
    raise ValueError(
        "Unsupported model.model_type value "
        f"'{model_type}'. Only 'cond_px_dif' is supported; legacy 'px_dif' was removed."
    )


def build_datamodule(
    dataset: torch.utils.data.Dataset,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> DepthTileDataModule:
    split_cfg = data_cfg.get("split", {})
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if "val_shuffle" in data_dataloader_cfg:
        dataloader_cfg["val_shuffle"] = bool(data_dataloader_cfg["val_shuffle"])

    return DepthTileDataModule(
        dataset=dataset,
        dataloader_cfg=dataloader_cfg,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        seed=int(data_cfg.get("dataset", {}).get("random_seed", 7)),
    )


def build_model(
    model_config_path: str,
    data_config_path: str,
    training_config_path: str,
    model_cfg: dict[str, Any],
    datamodule: DepthTileDataModule,
) -> PixelDiffusionConditional:
    resolve_model_type(model_cfg)
    return PixelDiffusionConditional.from_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )


def resolve_checkpoint_path(ckpt_override: str | None, model_cfg: dict[str, Any]) -> str | None:
    if ckpt_override:
        ckpt_path = Path(ckpt_override).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return str(ckpt_path)

    resume_cfg = model_cfg.get("model", {}).get("resume_checkpoint", False)
    if resume_cfg in (False, None):
        return None
    ckpt_path = Path(str(resume_cfg)).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint from config not found: {ckpt_path}")
    return str(ckpt_path)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def pretty_shape(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def build_random_batch(
    model: PixelDiffusionConditional,
    data_cfg: dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    m = model.hparams if hasattr(model, "hparams") else {}

    generated_channels = int(getattr(m, "generated_channels", 1))
    condition_channels = int(getattr(m, "condition_channels", generated_channels))
    condition_mask_channels = int(getattr(m, "condition_mask_channels", 0))
    include_eo = bool(getattr(m, "condition_include_eo", False))
    coord_enabled = bool(getattr(m, "coord_conditioning_enabled", False))
    date_enabled = bool(getattr(m, "date_conditioning_enabled", False))

    eo_channels = 1 if include_eo else 0
    x_channels = condition_channels - condition_mask_channels - eo_channels
    if x_channels <= 0:
        x_channels = generated_channels

    x = torch.randn(batch_size, x_channels, height, width, device=device)
    valid_mask = (torch.rand(batch_size, 1, height, width, device=device) > 0.25).float()
    land_mask = torch.ones(batch_size, 1, height, width, device=device)

    batch: dict[str, Any] = {
        "x": x,
        "valid_mask": valid_mask,
        "land_mask": land_mask,
    }

    if include_eo:
        batch["eo"] = torch.randn(batch_size, 1, height, width, device=device)

    # Provide coords if coordinate conditioning is enabled.
    if coord_enabled or bool(data_cfg.get("dataset", {}).get("return_coords", False)):
        lat = -90.0 + 180.0 * torch.rand(batch_size, 1, device=device)
        lon = -180.0 + 360.0 * torch.rand(batch_size, 1, device=device)
        batch["coords"] = torch.cat([lat, lon], dim=1)
        if date_enabled:
            # Match dataset convention for monthly tiles: YYYYMM15.
            months = torch.randint(1, 13, (batch_size,), device=device, dtype=torch.long)
            batch["date"] = torch.full_like(months, 2024) * 10000 + months * 100 + 15

    return batch


def run_predict_once(
    model: PixelDiffusionConditional,
    batch: dict[str, Any],
    include_intermediates: bool,
) -> dict[str, Any]:
    if include_intermediates:
        batch = dict(batch)
        batch["return_intermediates"] = True

    with torch.no_grad():
        return model.predict_step(batch, batch_idx=0)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def main() -> None:
    torch.manual_seed(int(SEED))

    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    data_cfg = load_yaml(DATA_CONFIG_PATH)
    training_cfg = load_yaml(TRAIN_CONFIG_PATH)
    resolve_model_type(model_cfg)

    device = choose_device(DEVICE)

    if MODE not in {"dataloader", "random"}:
        raise ValueError(f"MODE must be 'dataloader' or 'random' (got '{MODE}').")
    if LOADER_SPLIT not in {"train", "val"}:
        raise ValueError(
            f"LOADER_SPLIT must be 'train' or 'val' (got '{LOADER_SPLIT}')."
        )

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

    ckpt_path = resolve_checkpoint_path(CHECKPOINT_PATH, model_cfg)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=bool(STRICT_LOAD))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    if MODE == "dataloader":
        loader = datamodule.train_dataloader() if LOADER_SPLIT == "train" else datamodule.val_dataloader()
        batch = next(iter(loader))
        batch = to_device(batch, device)
    else:
        edge_size = int(data_cfg.get("dataset", {}).get("edge_size", 128))
        h = int(RANDOM_HEIGHT) if RANDOM_HEIGHT is not None else edge_size
        w = int(RANDOM_WIDTH) if RANDOM_WIDTH is not None else edge_size
        batch = build_random_batch(
            model=model,
            data_cfg=data_cfg,
            batch_size=int(RANDOM_BATCH_SIZE),
            height=h,
            width=w,
            device=device,
        )

    print("Input batch keys/shapes:")
    for k, v in batch.items():
        print(f"  - {k}: {pretty_shape(v)}")

    pred = run_predict_once(model, batch, include_intermediates=bool(INCLUDE_INTERMEDIATES))

    print("Output keys/shapes:")
    for k, v in pred.items():
        print(f"  - {k}: {pretty_shape(v)}")


if __name__ == "__main__":
    main()
