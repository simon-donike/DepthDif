"""Shared inference helpers used by scripts and export tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from data.dataset_ostia import SurfaceTempPatchOstiaLightDataset
from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset
from models.difFF import PixelDiffusionConditional
from models.latent import LatentDiffusionConditional


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and return yaml data."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ds_cfg_value(
    ds_cfg: dict[str, Any],
    nested_key: str,
    flat_key: str,
    *,
    default: Any,
) -> Any:
    """Read one dataset config field while preferring the nested schema."""
    node: Any = ds_cfg
    for part in nested_key.split("."):
        if not isinstance(node, dict) or part not in node:
            node = None
            break
        node = node[part]
    if node is not None:
        return node
    _ = flat_key
    return default


def resolve_dataset_variant(ds_cfg: dict[str, Any], data_config_path: str) -> str:
    """Resolve and validate dataset variant."""
    variant = ds_cfg_value(
        ds_cfg,
        "core.dataset_variant",
        "dataset_variant",
        default=None,
    )
    if variant is None:
        stem = Path(data_config_path).stem.lower()
        if "ostia" in stem:
            return "ostia"
        if "4band" in stem or "eo" in stem:
            return "eo_4band"
        return "eo_4band"
    return str(variant).strip().lower()


def build_dataset(
    data_config_path: str, ds_cfg: dict[str, Any]
) -> torch.utils.data.Dataset:
    """Build and return dataset."""
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant in {"eo_4band", "4band_eo", "4bands"}:
        return SurfaceTempPatch4BandsLightDataset.from_config(data_config_path, split="all")
    if dataset_variant in {"ostia", "ostia_4band", "4band_ostia"}:
        return SurfaceTempPatchOstiaLightDataset.from_config(data_config_path, split="all")
    if dataset_variant in {"ostia_argo_disk", "ostia_argo_tiff", "argo_ostia_tiff"}:
        return OstiaArgoTiffDataset(
            csv_path=str(
                ds_cfg_value(
                    ds_cfg,
                    "core.manifest_csv_path",
                    "manifest_csv_path",
                    default=OstiaArgoTiffDataset.DEFAULT_CSV_PATH,
                )
            ),
            split="all",
            return_info=bool(
                ds_cfg_value(ds_cfg, "output.return_info", "return_info", default=True)
            ),
            return_coords=bool(
                ds_cfg_value(ds_cfg, "output.return_coords", "return_coords", default=True)
            ),
            synthetic_mode=bool(
                ds_cfg_value(ds_cfg, "synthetic.enabled", "synthetic_enabled", default=False)
            ),
            synthetic_pixel_count=int(
                ds_cfg_value(ds_cfg, "synthetic.pixel_count", "synthetic_pixel_count", default=20)
            ),
            random_seed=int(
                ds_cfg_value(ds_cfg, "runtime.random_seed", "random_seed", default=7)
            ),
        )
    raise ValueError(
        "Unsupported dataset variant "
        f"'{dataset_variant}'. Expected one of ['eo_4band', 'ostia', 'ostia_argo_disk']."
    )


def resolve_model_type(model_cfg: dict[str, Any]) -> str:
    """Resolve and validate model type."""
    model_type = str(model_cfg.get("model", {}).get("model_type", "cond_px_dif")).strip()
    if model_type in {"cond_px_dif", "latent_cond_dif"}:
        return model_type
    raise ValueError(
        "Unsupported model.model_type value "
        f"'{model_type}'. Supported values: 'cond_px_dif', 'latent_cond_dif'."
    )


def build_datamodule(
    dataset: torch.utils.data.Dataset,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> DepthTileDataModule:
    """Build and return datamodule."""
    split_cfg = data_cfg.get("split", {})
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if "val_shuffle" in data_dataloader_cfg:
        dataloader_cfg["val_shuffle"] = bool(data_dataloader_cfg["val_shuffle"])

    return DepthTileDataModule(
        dataset=dataset,
        dataloader_cfg=dataloader_cfg,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        seed=int(
            ds_cfg_value(
                data_cfg.get("dataset", {}),
                "runtime.random_seed",
                "random_seed",
                default=7,
            )
        ),
    )


def build_model(
    model_config_path: str,
    data_config_path: str,
    training_config_path: str,
    model_cfg: dict[str, Any],
    datamodule: DepthTileDataModule,
) -> PixelDiffusionConditional | LatentDiffusionConditional:
    """Build and return model."""
    model_type = resolve_model_type(model_cfg)
    if model_type == "latent_cond_dif":
        return LatentDiffusionConditional.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_config_path=training_config_path,
            datamodule=datamodule,
        )
    return PixelDiffusionConditional.from_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )


def resolve_checkpoint_path(ckpt_override: str | None, model_cfg: dict[str, Any]) -> str | None:
    """Resolve and validate checkpoint path."""
    if ckpt_override:
        ckpt_path = Path(ckpt_override).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return str(ckpt_path)

    model_section = model_cfg.get("model", {})
    load_cfg = model_section.get("load_checkpoint", False)
    if load_cfg not in (False, None):
        ckpt_path = Path(str(load_cfg)).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint from config model.load_checkpoint not found: {ckpt_path}"
            )
        return str(ckpt_path)

    resume_cfg = model_section.get("resume_checkpoint", False)
    if resume_cfg in (False, None):
        return None
    ckpt_path = Path(str(resume_cfg)).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint from config not found: {ckpt_path}")
    return str(ckpt_path)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dictionary to the target device."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def pretty_shape(value: Any) -> str:
    """Return a compact human-readable shape/type description."""
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def build_random_batch(
    model: Any,
    data_cfg: dict[str, Any],
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build and return random batch."""
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
    mask_channels = max(1, generated_channels)
    x_valid_mask = torch.rand(
        batch_size, mask_channels, height, width, device=device
    ) > 0.25
    y_valid_mask = torch.ones(
        batch_size, generated_channels, height, width, device=device, dtype=torch.bool
    )
    x_valid_mask_1d = x_valid_mask.any(dim=1, keepdim=True)
    land_mask = torch.ones(
        batch_size, 1, height, width, device=device, dtype=torch.bool
    )

    batch: dict[str, Any] = {
        "x": x,
        "x_valid_mask": x_valid_mask,
        "y_valid_mask": y_valid_mask,
        "x_valid_mask_1d": x_valid_mask_1d,
        "land_mask": land_mask,
    }

    if include_eo:
        batch["eo"] = torch.randn(batch_size, 1, height, width, device=device)

    if coord_enabled or bool(
        ds_cfg_value(
            data_cfg.get("dataset", {}),
            "output.return_coords",
            "return_coords",
            default=False,
        )
    ):
        lat = -90.0 + 180.0 * torch.rand(batch_size, 1, device=device)
        lon = -180.0 + 360.0 * torch.rand(batch_size, 1, device=device)
        batch["coords"] = torch.cat([lat, lon], dim=1)
        if date_enabled:
            # Match dataset convention for monthly tiles so direct random runs still
            # exercise the date-conditioning path with realistic integer encodings.
            months = torch.randint(1, 13, (batch_size,), device=device, dtype=torch.long)
            batch["date"] = torch.full_like(months, 2024) * 10000 + months * 100 + 15

    return batch


def run_predict_once(
    model: Any,
    batch: dict[str, Any],
    include_intermediates: bool,
) -> dict[str, Any]:
    """Compute run predict once and return the result."""
    if include_intermediates:
        batch = dict(batch)
        batch["return_intermediates"] = True

    with torch.no_grad():
        return model.predict_step(batch, batch_idx=0)


def choose_device(device_arg: str) -> torch.device:
    """Choose and return device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)
