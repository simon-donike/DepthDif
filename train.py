"""Train the main diffusion model from YAML configs.

This script builds the dataset/datamodule, instantiates the configured pixel-space
or latent diffusion model, restores checkpoints when requested, and launches the
PyTorch Lightning training run.

Typical CLI:
    /work/envs/depth/bin/python train.py --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml --train-config src/depth_recon/configs/px_space/training_config.yaml --model-config src/depth_recon/configs/px_space/model_config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __package__ in {None, ""}:
    # Keep the root-level training script runnable from a fresh src-layout checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.data.dataset_argo_netcdf_gridded import ArgoNetCDFGriddedPatchDataset
from depth_recon.inference.core import load_checkpoint_weights
from depth_recon.models.diffusion import EMA, PixelDiffusionConditional
from depth_recon.models.latent import LatentDiffusionConditional
from depth_recon.paths import config_path, resolve_config_path

PX_MODEL_CONFIG_PATH = str(config_path("px_space", "model_config.yaml"))
PX_DATA_CONFIG_PATH = str(config_path("px_space", "data_ostia_argo_netcdf.yaml"))
PX_TRAINING_CONFIG_PATH = str(config_path("px_space", "training_config.yaml"))


# Centralized YAML loader for config files.
def load_yaml(path: str) -> dict[str, Any]:
    """Load and return yaml data.

    Args:
        path (str): Path to an input or output file.

    Returns:
        dict[str, Any]: Dictionary containing computed outputs.
    """
    with resolve_config_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write yaml data to disk preserving key order."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_config_override(raw_override: str) -> tuple[str, list[str], Any]:
    """Parse one CLI config override expression.

    Expected format: <config_root>.<nested.path>=<yaml_value>
    where config_root is one of: data, training, model.
    """
    if "=" not in raw_override:
        raise ValueError(
            f"Invalid override '{raw_override}'. Expected format: "
            "<data|training|model>.<path>=<yaml_value>."
        )

    lhs, rhs = raw_override.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if "." not in lhs:
        raise ValueError(
            f"Invalid override '{raw_override}'. Missing nested key path after root."
        )

    root, *keys = [part.strip() for part in lhs.split(".")]
    if root not in {"data", "training", "model"}:
        raise ValueError(
            f"Invalid override root '{root}' in '{raw_override}'. "
            "Allowed roots: data, training, model."
        )
    if any(not key for key in keys):
        raise ValueError(
            f"Invalid override '{raw_override}'. Key path contains empty segment(s)."
        )

    return root, keys, yaml.safe_load(rhs)


def apply_config_overrides(
    overrides: list[str],
    configs_by_root: dict[str, dict[str, Any]],
) -> None:
    """Apply strict nested config overrides in-place.

    All path segments must already exist; this prevents silent typos.
    """
    for raw_override in overrides:
        root, keys, value = parse_config_override(raw_override)
        target: dict[str, Any] = configs_by_root[root]
        for key in keys[:-1]:
            if key not in target:
                raise KeyError(
                    f"Invalid override '{raw_override}': key '{key}' does not exist."
                )
            nested = target[key]
            if not isinstance(nested, dict):
                raise TypeError(
                    f"Invalid override '{raw_override}': '{key}' is not a mapping."
                )
            target = nested

        leaf_key = keys[-1]
        if leaf_key not in target:
            raise KeyError(
                f"Invalid override '{raw_override}': key '{leaf_key}' does not exist."
            )
        target[leaf_key] = value


# Resolve an optional resume checkpoint path and validate it early.
def resolve_resume_ckpt_path(model_cfg: dict[str, Any]) -> str | None:
    # Accept false/null to start fresh; otherwise require a valid checkpoint path string.
    """Resolve and validate resume ckpt path.

    Args:
        model_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        str | None: Computed output value.
    """
    resume_cfg = model_cfg.get("model", {}).get("resume_checkpoint", False)
    if resume_cfg is False or resume_cfg is None:
        return None
    if not isinstance(resume_cfg, str):
        raise ValueError(
            "model.resume_checkpoint must be false/null or a checkpoint path string."
        )

    ckpt_path = Path(resume_cfg).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path)


def resolve_load_checkpoint_only(model_cfg: dict[str, Any]) -> bool:
    """Resolve whether checkpoint loading should restore only model weights.

    Args:
        model_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        bool: True when only model weights should be loaded from resume checkpoint.
    """
    load_checkpoint_only = model_cfg.get("model", {}).get("load_checkpoint_only", False)
    if not isinstance(load_checkpoint_only, bool):
        raise ValueError("model.load_checkpoint_only must be true or false.")
    return load_checkpoint_only


def load_weights_only_checkpoint(model: torch.nn.Module, ckpt_path: str) -> str:
    """Load only model weights from a checkpoint, preferring EMA weights if present.

    Args:
        model (torch.nn.Module): Model receiving the loaded state dict.
        ckpt_path (str): Checkpoint path to load.

    Returns:
        str: Loaded weight source, either "ema" or "standard".
    """
    return load_checkpoint_weights(model, ckpt_path, strict=True, prefer_ema=True)


# Build process rank defensively across common launchers.
# Preference order avoids local-rank-only false positives in multi-node jobs.
def resolve_global_rank() -> int:
    """Resolve and validate global rank.

    Args:
        None: This callable takes no explicit input arguments.

    Returns:
        int: Computed scalar output.
    """
    rank_env_keys = ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK")
    for key in rank_env_keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


# Configure W&B logging and optional model watching.
def resolve_wandb_watch_mode(wandb_cfg: dict[str, Any]) -> str | None:
    # Map explicit booleans to wandb.watch(log=...):
    # - gradients + parameters -> "all"
    # - gradients only -> "gradients"
    # - parameters only -> "parameters"
    # - neither -> disable watch by returning None.
    # Explicit toggles take precedence when provided.
    """Resolve and validate wandb watch mode.

    Args:
        wandb_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        str | None: Computed output value.
    """
    has_explicit_toggles = (
        "watch_gradients" in wandb_cfg or "watch_parameters" in wandb_cfg
    )
    if has_explicit_toggles:
        watch_gradients = bool(wandb_cfg.get("watch_gradients", True))
        watch_parameters = bool(wandb_cfg.get("watch_parameters", True))
        if watch_gradients and watch_parameters:
            return "all"
        if watch_gradients:
            return "gradients"
        if watch_parameters:
            return "parameters"
        return None

    # Backward-compatible fallback for older configs using watch_log directly.
    watch_mode = wandb_cfg.get("watch_log", "all")
    if watch_mode is None or watch_mode is False:
        return None
    normalized = str(watch_mode).strip().lower()
    if normalized in {"none", "false", "off"}:
        return None
    return str(watch_mode)


def build_wandb_logger(
    training_cfg: dict[str, Any], model: pl.LightningModule
) -> WandbLogger:
    # Build logger from config first; watch settings are attached conditionally below.
    """Build and return wandb logger.

    Args:
        training_cfg (dict[str, Any]): Configuration dictionary or section.
        model (pl.LightningModule): Input value.

    Returns:
        WandbLogger: Computed output value.
    """
    wandb_cfg = training_cfg.get("wandb", {})
    logger = WandbLogger(
        project=wandb_cfg.get("project", "DepthDif"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        log_model=wandb_cfg.get("log_model", "all"),
    )

    # Only attach wandb.watch when watch_mode resolves to a valid mode.
    # Returning None from resolve_wandb_watch_mode disables watch entirely.
    watch_mode = resolve_wandb_watch_mode(wandb_cfg)
    if watch_mode is not None:
        logger.watch(
            model,
            log=watch_mode,
            log_freq=int(wandb_cfg.get("watch_log_freq", 25)),
            log_graph=bool(wandb_cfg.get("watch_log_graph", False)),
        )
    return logger


def upload_configs_to_wandb(logger: WandbLogger, config_paths: list[str]) -> None:
    # In offline/disabled logger modes experiment may be unavailable.
    """Upload configs to wandb to experiment tracking.

    Args:
        logger (WandbLogger): Logger instance used for experiment tracking.
        config_paths (list[str]): Path to an input or output file.

    Returns:
        None: No value is returned.
    """
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    # Upload exact local config files so each run can be reproduced from W&B artifacts.
    for cfg_path in config_paths:
        path = Path(cfg_path)
        if path.is_file():
            # Store configs as run files for reproducibility and easy download from UI.
            experiment.save(str(path.resolve()), policy="now")


def resolve_dataset_variant(ds_cfg: dict[str, Any], data_config_path: str) -> str:
    """Resolve and validate dataset variant.

    Args:
        ds_cfg (dict[str, Any]): Configuration dictionary or section.
        data_config_path (str): Path to an input or output file.

    Returns:
        str: Computed scalar output.
    """
    variant = ds_cfg_value(
        ds_cfg,
        "core.dataset_variant",
        "dataset_variant",
        default="argo_netcdf_gridded",
    )
    _ = data_config_path
    return str(variant).strip().lower()


def ds_cfg_value(
    ds_cfg: dict[str, Any],
    nested_key: str,
    flat_key: str,
    *,
    default: Any,
) -> Any:
    """Read nested dataset config."""
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


def build_dataset(
    data_config_path: str,
    ds_cfg: dict[str, Any],
    *,
    split: str,
) -> torch.utils.data.Dataset:
    # Keep one active dataset path so stale configs fail loudly.
    """Build and return dataset.

    Args:
        data_config_path (str): Path to an input or output file.
        ds_cfg (dict[str, Any]): Configuration dictionary or section.
        split (str): Dataset split label to instantiate.

    Returns:
        torch.utils.data.Dataset: Computed output value.
    """
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant == "argo_netcdf_gridded":
        return ArgoNetCDFGriddedPatchDataset.from_config(
            data_config_path,
            split=split,
        )
    if dataset_variant == "argo_geotiff_gridded":
        return ArgoGeoTIFFGriddedPatchDataset.from_config(
            data_config_path,
            split=split,
        )
    raise ValueError(
        "Unsupported dataset variant in data config. "
        f"Got '{dataset_variant}', expected one of "
        "{'argo_netcdf_gridded', 'argo_geotiff_gridded'}."
    )


def resolve_model_type(model_cfg: dict[str, Any]) -> str:
    """Resolve and validate model type.

    Args:
        model_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        str: Computed scalar output.
    """
    model_type = str(
        model_cfg.get("model", {}).get("model_type", "cond_px_dif")
    ).strip()
    if model_type in {"cond_px_dif", "latent_cond_dif"}:
        return model_type
    raise ValueError(
        "Unsupported model.model_type value "
        f"'{model_type}'. Supported values: 'cond_px_dif', 'latent_cond_dif'."
    )


def parse_config_bool(value: Any, *, key: str) -> bool:
    """Parse a strict boolean value from YAML config data."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{key} must be a boolean value.")


def build_ema_callback(model_cfg: dict[str, Any]) -> EMA | None:
    """Build the optional EMA callback from model config."""
    ema_cfg = model_cfg.get("model", {}).get("ema", {})
    if ema_cfg is None or ema_cfg is False:
        return None
    if not isinstance(ema_cfg, dict):
        raise ValueError("model.ema must be a mapping, false, or null.")
    if not parse_config_bool(ema_cfg.get("enabled", False), key="model.ema.enabled"):
        return None

    decay = float(ema_cfg.get("decay", 0.9999))
    if not (0.0 <= decay < 1.0):
        raise ValueError("model.ema.decay must be >= 0.0 and < 1.0.")

    apply_every_n_steps = int(
        ema_cfg.get(
            "apply_every_n_steps",
            ema_cfg.get("apply_ema_every_n_steps", 1),
        )
    )
    if apply_every_n_steps < 1:
        raise ValueError("model.ema.apply_every_n_steps must be >= 1.")

    start_step = int(ema_cfg.get("start_step", 0))
    if start_step < 0:
        raise ValueError("model.ema.start_step must be >= 0.")

    return EMA(
        decay=decay,
        apply_ema_every_n_steps=apply_every_n_steps,
        start_step=start_step,
        save_ema_weights_in_callback_state=parse_config_bool(
            ema_cfg.get("save_ema_weights_in_callback_state", True),
            key="model.ema.save_ema_weights_in_callback_state",
        ),
        evaluate_ema_weights_instead=parse_config_bool(
            ema_cfg.get("evaluate_ema_weights_instead", True),
            key="model.ema.evaluate_ema_weights_instead",
        ),
    )


def main(
    model_config_path: str = PX_MODEL_CONFIG_PATH,
    data_config_path: str = PX_DATA_CONFIG_PATH,
    training_config_path: str = PX_TRAINING_CONFIG_PATH,
    overrides: list[str] | None = None,
    fast_dev_run: int = 0,
) -> None:
    """Run the script entry point.

    Args:
        model_config_path (str): Path to an input or output file.
        data_config_path (str): Path to an input or output file.
        training_config_path (str): Path to an input or output file.
        overrides (list[str] | None): Optional config overrides from CLI.

    Returns:
        None: No value is returned.
    """
    model_config_path = str(resolve_config_path(model_config_path))
    data_config_path = str(resolve_config_path(data_config_path))
    training_config_path = str(resolve_config_path(training_config_path))

    # Determine rank before creating any run-scoped folders/files.
    global_rank = resolve_global_rank()
    is_global_zero = global_rank == 0

    # Create one run directory per launch; non-zero ranks reuse the resolved path.
    # Use one timestamped run directory; only global rank 0 creates it.
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("logs") / run_stamp
    if is_global_zero:
        suffix = 1
        while run_dir.exists():
            run_dir = Path("logs") / f"{run_stamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

        # Snapshot the exact configs used for reproducibility.
        shutil.copy2(model_config_path, run_dir / Path(model_config_path).name)
        shutil.copy2(data_config_path, run_dir / Path(data_config_path).name)
        shutil.copy2(training_config_path, run_dir / Path(training_config_path).name)

    # Load configuration and validate model settings.
    model_cfg = load_yaml(model_config_path)
    training_cfg = load_yaml(training_config_path)
    data_cfg = load_yaml(data_config_path)
    override_list = list(overrides or [])
    apply_config_overrides(
        overrides=override_list,
        configs_by_root={
            "model": model_cfg,
            "training": training_cfg,
            "data": data_cfg,
        },
    )
    latent_ae_config_path: str | None = None
    if (
        str(model_cfg.get("model", {}).get("model_type", "")).strip()
        == "latent_cond_dif"
    ):
        ae_cfg_value = (
            model_cfg.get("model", {}).get("latent", {}).get("ae_config_path", None)
        )
        if ae_cfg_value is None or str(ae_cfg_value).strip() == "":
            raise ValueError(
                "model.latent.ae_config_path is required for model.model_type='latent_cond_dif'."
            )
        ae_cfg_path = resolve_config_path(str(ae_cfg_value))
        if not ae_cfg_path.is_file():
            raise FileNotFoundError(f"AE config not found: {ae_cfg_path}")
        latent_ae_config_path = str(ae_cfg_path)
        if is_global_zero:
            shutil.copy2(ae_cfg_path, run_dir / ae_cfg_path.name)

    # Model/dataset builders load YAML files from paths, so write effective runtime
    # configs when overrides are present and route all path-based constructors there.
    effective_model_config_path = model_config_path
    effective_data_config_path = data_config_path
    effective_training_config_path = training_config_path
    if override_list:
        runtime_cfg_dir = (
            Path("/tmp/depthdif_runtime_configs")
            / f"{run_stamp}_{os.getpid()}_{global_rank}"
        )
        runtime_cfg_dir.mkdir(parents=True, exist_ok=True)
        effective_model_config_path = str(
            runtime_cfg_dir / Path(model_config_path).name
        )
        effective_data_config_path = str(runtime_cfg_dir / Path(data_config_path).name)
        effective_training_config_path = str(
            runtime_cfg_dir / Path(training_config_path).name
        )
        dump_yaml(effective_model_config_path, model_cfg)
        dump_yaml(effective_data_config_path, data_cfg)
        dump_yaml(effective_training_config_path, training_cfg)

    uploaded_config_paths = [model_config_path, data_config_path, training_config_path]
    if latent_ae_config_path is not None:
        uploaded_config_paths.append(latent_ae_config_path)
    if is_global_zero and override_list:
        model_effective_snapshot = (
            run_dir / f"{Path(model_config_path).stem}_effective.yaml"
        )
        data_effective_snapshot = (
            run_dir / f"{Path(data_config_path).stem}_effective.yaml"
        )
        training_effective_snapshot = (
            run_dir / f"{Path(training_config_path).stem}_effective.yaml"
        )
        dump_yaml(model_effective_snapshot, model_cfg)
        dump_yaml(data_effective_snapshot, data_cfg)
        dump_yaml(training_effective_snapshot, training_cfg)
        uploaded_config_paths = [
            str(model_effective_snapshot),
            str(data_effective_snapshot),
            str(training_effective_snapshot),
        ]
        if latent_ae_config_path is not None:
            uploaded_config_paths.append(latent_ae_config_path)

    # Resolve checkpoint paths once so failure happens early before trainer/model setup.
    resume_ckpt_path = resolve_resume_ckpt_path(model_cfg)
    load_checkpoint_only = resolve_load_checkpoint_only(model_cfg)
    trainer_cfg = training_cfg.get("trainer", model_cfg.get("trainer", {}))
    model_type = resolve_model_type(model_cfg)

    # Use Tensor Cores efficiently for fp16/bf16 mixed precision.
    torch.set_float32_matmul_precision(str(trainer_cfg.get("matmul_precision", "high")))

    # Reduce noisy framework warnings that are not actionable for this training loop.
    if bool(trainer_cfg.get("suppress_accumulate_grad_stream_mismatch_warning", True)):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    if bool(trainer_cfg.get("suppress_lightning_pytree_warning", True)):
        warnings.filterwarnings(
            "ignore",
            message=r".*LeafSpec.*deprecated.*",
            module=r"pytorch_lightning\.utilities\._pytree",
            category=Warning,
        )

    # Build EO-conditioned multiband dataset from config.
    ds_cfg = data_cfg.get("dataset", {})
    split_cfg = data_cfg.get("split", {})
    # Training config provides defaults; selected data configs can override
    # loader behavior because NetCDF and GeoTIFF backends have different I/O costs.
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if isinstance(data_dataloader_cfg, dict):
        dataloader_cfg.update(data_dataloader_cfg)
    dataloader_type = (
        str(
            ds_cfg_value(
                ds_cfg, "core.dataloader_type", "dataloader_type", default="light"
            )
        )
        .strip()
        .lower()
    )
    if dataloader_type != "light":
        raise ValueError(
            f"Only 'light' dataloader_type is supported in this runner; got '{dataloader_type}'."
        )
    # Instantiate dataset variant and inject EO dropout probability from data config.
    # Train/val datasets are instantiated separately so dataset split labels are respected.
    train_dataset = build_dataset(
        data_config_path=effective_data_config_path,
        ds_cfg=ds_cfg,
        split="train",
    )
    val_dataset = build_dataset(
        data_config_path=effective_data_config_path,
        ds_cfg=ds_cfg,
        split="val",
    )
    eo_dropout_prob = float(
        max(
            0.0,
            min(
                1.0,
                float(
                    ds_cfg_value(
                        ds_cfg,
                        "conditioning.eo_dropout_prob",
                        "eo_dropout_prob",
                        default=0.0,
                    )
                ),
            ),
        )
    )
    for dataset in (train_dataset, val_dataset):
        if hasattr(dataset, "eo_dropout_prob"):
            dataset.eo_dropout_prob = eo_dropout_prob
    datamodule = DepthTileDataModule(
        dataset=train_dataset,
        val_dataset=val_dataset,
        dataloader_cfg=dataloader_cfg,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        seed=int(ds_cfg_value(ds_cfg, "runtime.random_seed", "random_seed", default=7)),
    )

    # Instantiate the requested model type from config.
    if model_type == "latent_cond_dif":
        model = LatentDiffusionConditional.from_config(
            model_config_path=effective_model_config_path,
            data_config_path=effective_data_config_path,
            training_config_path=effective_training_config_path,
            datamodule=datamodule,
        )
    else:
        model = PixelDiffusionConditional.from_config(
            model_config_path=effective_model_config_path,
            data_config_path=effective_data_config_path,
            training_config_path=effective_training_config_path,
            datamodule=datamodule,
        )
    if resume_ckpt_path is not None and load_checkpoint_only:
        # Weight-only loading intentionally skips optimizer, scheduler, and trainer state.
        weight_source = load_weights_only_checkpoint(model, resume_ckpt_path)
        print(
            f"Loaded {weight_source} model weights from checkpoint: {resume_ckpt_path}"
        )

    # Set up experiment tracking and best-checkpoint saving.
    logger = build_wandb_logger(training_cfg, model)
    if is_global_zero:
        # Avoid duplicate uploads from DDP worker ranks.
        upload_configs_to_wandb(
            logger,
            uploaded_config_paths,
        )
    # Save the best checkpoint by monitored validation metric and always keep the latest checkpoint.
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-epoch{epoch:03d}",
        monitor=str(trainer_cfg.get("ckpt_monitor", "val/loss")),
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval=str(trainer_cfg.get("lr_logging_interval", "epoch"))
    )
    ema_callback = build_ema_callback(model_cfg)
    callbacks: list[pl.Callback] = [checkpoint_callback, lr_monitor_callback]
    if ema_callback is not None:
        # Restore raw training weights before ModelCheckpoint writes resume state.
        callbacks = [ema_callback, checkpoint_callback, lr_monitor_callback]

    # Build device settings from config
    num_gpus = trainer_cfg.get("num_gpus", None)
    # Keep backward compatibility with legacy num_gpus while supporting Lightning auto config.
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        accelerator = "gpu" if num_gpus > 0 else "cpu"
        devices = num_gpus if num_gpus > 0 else 1
    else:
        accelerator = trainer_cfg.get("accelerator", "auto")
        devices = trainer_cfg.get("devices", "auto")

    # Optional hard cap on number of validation batches per epoch.
    val_batches_per_epoch = trainer_cfg.get("val_batches_per_epoch", None)
    if val_batches_per_epoch is not None:
        limit_val_batches = int(val_batches_per_epoch)
        if limit_val_batches < 1:
            raise ValueError("trainer.val_batches_per_epoch must be >= 1 when set.")
    else:
        # Lightning-native value: float fraction (0-1] or int batch count.
        limit_val_batches = trainer_cfg.get("limit_val_batches", 1.0)

    # Trainer configuration is fully driven from training_config.yaml.
    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 100)),
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "32-true"),
        fast_dev_run=int(fast_dev_run) if int(fast_dev_run) > 0 else False,
        # Keep sanity checks enabled by default, but model-side logic keeps them lightweight.
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 2)),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        limit_val_batches=limit_val_batches,
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    if is_global_zero:
        # Print dataset sample counts explicitly because Lightning startup output
        # focuses on batch counts and device setup rather than total examples.
        print(
            "Dataset summary: "
            f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}"
        )

    # Start (or resume) training.
    fit_ckpt_path = None if load_checkpoint_only else resume_ckpt_path
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=fit_ckpt_path)


def parse_args() -> argparse.Namespace:
    # Keep CLI names stable and map aliases to canonical destination keys.
    """Parse command-line arguments for this script.

    Args:
        None: This callable takes no explicit input arguments.

    Returns:
        argparse.Namespace: Computed output value.
    """
    parser = argparse.ArgumentParser(description="Train DepthDif models.")
    parser.add_argument(
        "--data-config",
        default=PX_DATA_CONFIG_PATH,
        help="Path to data config yaml.",
    )
    parser.add_argument(
        "--train-config",
        "--training-config",
        default=PX_TRAINING_CONFIG_PATH,
        dest="training_config",
        help="Path to training config yaml.",
    )
    parser.add_argument(
        "--model-config",
        "--mdoel-config",
        default=PX_MODEL_CONFIG_PATH,
        dest="model_config",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
        help=(
            "Override config values. Format: "
            "<data|training|model>.<nested.path>=<yaml_value>. "
            "Repeat --set for multiple overrides."
        ),
    )
    parser.add_argument(
        "--fast_dev_run",
        "--fast-dev-run",
        type=int,
        default=0,
        dest="fast_dev_run",
        help="Run N batches for debugging (0 disables it, 1 behaves like true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        training_config_path=args.training_config,
        overrides=args.config_overrides,
        fast_dev_run=args.fast_dev_run,
    )

"""
# Training quick start (single command):
python train.py --model-config src/depth_recon/configs/px_space/model_config.yaml \
    --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
    --training-config src/depth_recon/configs/px_space/training_config.yaml

# Sweep quick start (single command):
./src/depth_recon/scripts/start_occlusion_sweep.sh
"""
