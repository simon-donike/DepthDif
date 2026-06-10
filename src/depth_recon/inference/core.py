"""Shared helpers for dataset-paper baseline training and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
import yaml

from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.models.baselines import (
    IDWInterpolationBaseline,
    PointwiseLSTMBaseline,
    UNetInfillingBaseline,
)
from depth_recon.paths import resolve_config_path

VARIABLE_SCENARIO_KEY = "variable_scenario"
MODEL_TYPES = ("idw_baseline", "lstm_baseline", "unet_baseline")
CHECKPOINT_FREE_MODEL_TYPES = {"idw_baseline"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and return YAML data from a package-relative or filesystem path."""
    with resolve_config_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    """Resolve and validate the configured dataset variant."""
    variant = ds_cfg_value(
        ds_cfg,
        "core.dataset_variant",
        "dataset_variant",
        default="argo_geotiff_gridded",
    )
    _ = data_config_path
    return str(variant).strip().lower()


def build_dataset(
    data_config_path: str,
    ds_cfg: dict[str, Any],
    *,
    split: str = "all",
    dataset_overrides: dict[str, Any] | None = None,
) -> torch.utils.data.Dataset:
    """Build and return the configured dataset."""
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant == "argo_geotiff_gridded":
        return ArgoGeoTIFFGriddedPatchDataset.from_config(
            data_config_path,
            split=split,
            dataset_overrides=dataset_overrides,
        )
    raise ValueError(
        "Unsupported dataset variant "
        f"{dataset_variant!r}. Expected 'argo_geotiff_gridded'."
    )


def resolve_model_type(model_cfg: dict[str, Any]) -> str:
    """Resolve and validate the configured baseline model type."""
    model_type = str(
        model_cfg.get("model", {}).get("model_type", "unet_baseline")
    ).strip()
    if model_type in MODEL_TYPES:
        return model_type
    supported = "', '".join(MODEL_TYPES)
    raise ValueError(
        "Unsupported model.model_type value "
        f"{model_type!r}. Supported values: '{supported}'."
    )


def model_requires_checkpoint(model_cfg: dict[str, Any]) -> bool:
    """Return whether the configured model type has trainable checkpoint weights."""
    return resolve_model_type(model_cfg) not in CHECKPOINT_FREE_MODEL_TYPES


def build_datamodule(
    dataset: torch.utils.data.Dataset,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> DepthTileDataModule:
    """Build and return a Lightning datamodule for one dataset."""
    split_cfg = data_cfg.get("split", {})
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if isinstance(data_dataloader_cfg, dict):
        dataloader_cfg.update(data_dataloader_cfg)

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
) -> IDWInterpolationBaseline | PointwiseLSTMBaseline | UNetInfillingBaseline:
    """Build and return the configured baseline model."""
    model_type = resolve_model_type(model_cfg)
    if model_type == "idw_baseline":
        return IDWInterpolationBaseline.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_config_path=training_config_path,
            datamodule=datamodule,
        )
    if model_type == "lstm_baseline":
        return PointwiseLSTMBaseline.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_config_path=training_config_path,
            datamodule=datamodule,
        )
    return UNetInfillingBaseline.from_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )


def resolve_checkpoint_path(
    ckpt_override: str | None, model_cfg: dict[str, Any]
) -> str | None:
    """Resolve and validate an optional checkpoint path."""
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


def extract_ema_state_dict(checkpoint: Any) -> dict[str, torch.Tensor] | None:
    """Extract EMA weights from a Lightning checkpoint payload when present."""
    if not isinstance(checkpoint, dict):
        return None

    direct_ema = checkpoint.get("ema_weights")
    if isinstance(direct_ema, dict):
        return {str(key): value for key, value in direct_ema.items()}

    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return None

    fallback_ema: dict[str, torch.Tensor] | None = None
    for callback_key, callback_state in callbacks.items():
        if not isinstance(callback_state, dict):
            continue
        ema_weights = callback_state.get("ema_weights")
        if not isinstance(ema_weights, dict):
            continue
        normalized = {str(key): value for key, value in ema_weights.items()}
        if "EMA" in str(callback_key):
            return normalized
        fallback_ema = normalized
    return fallback_ema


def _normalize_variable_scenario_value(value: Any) -> str | None:
    """Normalize optional variable scenario metadata."""
    if value is None or value is False:
        return None
    scenario = str(value).strip().lower()
    return scenario or None


def _model_variable_scenario(model: torch.nn.Module) -> str | None:
    """Return the scenario expected by a model when it exposes one."""
    scenario = _normalize_variable_scenario_value(
        getattr(model, VARIABLE_SCENARIO_KEY, None)
    )
    if scenario is not None:
        return scenario

    hparams = getattr(model, "hparams", None)
    if isinstance(hparams, dict):
        return _normalize_variable_scenario_value(hparams.get(VARIABLE_SCENARIO_KEY))
    return _normalize_variable_scenario_value(
        getattr(hparams, VARIABLE_SCENARIO_KEY, None)
    )


def _checkpoint_variable_scenario(checkpoint: Any) -> str | None:
    """Read scenario metadata from top-level checkpoint or Lightning hparams."""
    if not isinstance(checkpoint, dict):
        return None

    scenario = _normalize_variable_scenario_value(checkpoint.get(VARIABLE_SCENARIO_KEY))
    if scenario is not None:
        return scenario

    hparams = checkpoint.get("hyper_parameters")
    if isinstance(hparams, dict):
        return _normalize_variable_scenario_value(hparams.get(VARIABLE_SCENARIO_KEY))
    return None


def _validate_checkpoint_variable_scenario(
    model: torch.nn.Module, checkpoint: Any, checkpoint_path: str | Path
) -> None:
    """Validate that checkpoint scenario metadata matches the model."""
    expected_scenario = _model_variable_scenario(model)
    if expected_scenario is None:
        return

    checkpoint_scenario = _checkpoint_variable_scenario(checkpoint)
    if checkpoint_scenario is None:
        warnings.warn(
            "Checkpoint does not contain variable_scenario metadata; "
            f"loading checkpoint without scenario validation: {checkpoint_path}",
            stacklevel=2,
        )
        return

    if checkpoint_scenario != expected_scenario:
        raise ValueError(
            "Checkpoint variable_scenario mismatch: "
            f"checkpoint has {checkpoint_scenario!r}, "
            f"model expects {expected_scenario!r}. "
            "Use the checkpoint trained for the selected scenario."
        )


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    prefer_ema: bool = False,
) -> str:
    """Load Lightning checkpoint weights into a baseline model."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _validate_checkpoint_variable_scenario(model, checkpoint, checkpoint_path)
    if prefer_ema:
        ema_state_dict = extract_ema_state_dict(checkpoint)
        if ema_state_dict is not None:
            model.load_state_dict(ema_state_dict, strict=bool(strict))
            return "ema"

    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=bool(strict))
    return "standard"


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


def run_predict_once(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Run one model predict step without gradients."""
    with torch.no_grad():
        return model.predict_step(batch, batch_idx=0)


def choose_device(device_arg: str) -> torch.device:
    """Choose and return a torch device from a CLI argument."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)
