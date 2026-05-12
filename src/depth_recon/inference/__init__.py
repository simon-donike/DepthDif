"""Inference package for runtime helpers and hosted export workflows."""

from .core import (
    build_datamodule,
    build_dataset,
    build_model,
    build_random_batch,
    choose_device,
    ds_cfg_value,
    load_yaml,
    pretty_shape,
    resolve_checkpoint_path,
    resolve_dataset_variant,
    resolve_model_type,
    run_predict_once,
    to_device,
)

_API_EXPORTS = {
    "InferenceAssets",
    "PublicInferenceAssets",
    "download_argo_for_week",
    "resolve_hf_assets",
    "resolve_public_inference_assets",
    "run_week_inference",
}

__all__ = [
    "InferenceAssets",
    "PublicInferenceAssets",
    "build_datamodule",
    "build_dataset",
    "build_model",
    "build_random_batch",
    "choose_device",
    "download_argo_for_week",
    "ds_cfg_value",
    "load_yaml",
    "pretty_shape",
    "resolve_hf_assets",
    "resolve_public_inference_assets",
    "resolve_checkpoint_path",
    "resolve_dataset_variant",
    "resolve_model_type",
    "run_predict_once",
    "run_week_inference",
    "to_device",
]


def __getattr__(name: str):
    """Lazily expose public API helpers without preloading the CLI module."""
    if name in _API_EXPORTS:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
