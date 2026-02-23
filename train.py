from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Any
import shutil
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from data.dataset_temp_v1 import SurfaceTempPatchLightDataset
from models.difFF import PixelDiffusionConditional


# Centralized YAML loader for config files.
def load_yaml(path: str) -> dict[str, Any]:
    """Load and return yaml data.

    Args:
        path (str): Path to an input or output file.

    Returns:
        dict[str, Any]: Dictionary containing computed outputs.
    """
    with Path(path).open("r", encoding="utf-8") as f:
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
    # Prefer explicit dataset variant from config, with filename-based fallback for old configs.
    """Resolve and validate dataset variant.

    Args:
        ds_cfg (dict[str, Any]): Configuration dictionary or section.
        data_config_path (str): Path to an input or output file.

    Returns:
        str: Computed scalar output.
    """
    variant = ds_cfg.get("dataset_variant", ds_cfg.get("variant", None))
    if variant is None:
        # Backward-compatible fallback: infer from config filename if explicit variant is absent.
        stem = Path(data_config_path).stem.lower()
        if "4band" in stem or "eo" in stem:
            return "eo_4band"
        return "temp_v1"
    return str(variant).strip().lower()


def build_dataset(
    data_config_path: str, ds_cfg: dict[str, Any]
) -> SurfaceTempPatchLightDataset | SurfaceTempPatch4BandsLightDataset:
    # Route to dataset implementation matching the requested training task.
    """Build and return dataset.

    Args:
        data_config_path (str): Path to an input or output file.
        ds_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        SurfaceTempPatchLightDataset | SurfaceTempPatch4BandsLightDataset: Computed output value.
    """
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant in {"temp_v1", "single_band", "1band", "default"}:
        return SurfaceTempPatchLightDataset.from_config(
            data_config_path,
            split="all",
        )
    if dataset_variant in {"eo_4band", "4band_eo", "4bands"}:
        return SurfaceTempPatch4BandsLightDataset.from_config(
            data_config_path,
            split="all",
        )
    raise ValueError(
        "Unsupported dataset variant in data config. "
        f"Got '{dataset_variant}', expected one of "
        "{'temp_v1', 'eo_4band'}."
    )


def resolve_model_type(model_cfg: dict[str, Any]) -> str:
    """Resolve and validate model type.

    Args:
        model_cfg (dict[str, Any]): Configuration dictionary or section.

    Returns:
        str: Computed scalar output.
    """
    model_type = str(model_cfg.get("model", {}).get("model_type", "cond_px_dif")).strip()
    if model_type == "cond_px_dif":
        return model_type
    raise ValueError(
        "Unsupported model.model_type value "
        f"'{model_type}'. Only 'cond_px_dif' is supported; legacy 'px_dif' was removed."
    )


def main(
    model_config_path: str = "configs/model_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
    overrides: list[str] | None = None,
) -> None:
    # Determine rank before creating any run-scoped folders/files.
    """Run the script entry point.

    Args:
        model_config_path (str): Path to an input or output file.
        data_config_path (str): Path to an input or output file.
        training_config_path (str): Path to an input or output file.
        overrides (list[str] | None): Optional config overrides from CLI.

    Returns:
        None: No value is returned.
    """
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
        effective_model_config_path = str(runtime_cfg_dir / Path(model_config_path).name)
        effective_data_config_path = str(runtime_cfg_dir / Path(data_config_path).name)
        effective_training_config_path = str(
            runtime_cfg_dir / Path(training_config_path).name
        )
        dump_yaml(effective_model_config_path, model_cfg)
        dump_yaml(effective_data_config_path, data_cfg)
        dump_yaml(effective_training_config_path, training_cfg)

    uploaded_config_paths = [model_config_path, data_config_path, training_config_path]
    if is_global_zero and override_list:
        model_effective_snapshot = run_dir / f"{Path(model_config_path).stem}_effective.yaml"
        data_effective_snapshot = run_dir / f"{Path(data_config_path).stem}_effective.yaml"
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

    # Resolve resume path once so failure happens early before trainer/model setup.
    resume_ckpt_path = resolve_resume_ckpt_path(model_cfg)
    trainer_cfg = training_cfg.get("trainer", model_cfg.get("trainer", {}))
    resolve_model_type(model_cfg)

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

    # Build dataset from config while preserving default temp_v1 behavior.
    ds_cfg = data_cfg.get("dataset", {})
    split_cfg = data_cfg.get("split", {})
    # Training config owns dataloader behavior; selected data config can still override val shuffle.
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if "val_shuffle" in data_dataloader_cfg:
        dataloader_cfg["val_shuffle"] = bool(data_dataloader_cfg["val_shuffle"])
    dataloader_type = str(ds_cfg.get("dataloader_type", "light")).strip().lower()
    if dataloader_type != "light":
        raise ValueError(
            f"Only 'light' dataloader_type is supported in this runner; got '{dataloader_type}'."
        )
    # Instantiate dataset variant and inject EO dropout probability from data config.
    # Train/val subsets share the same base dataset object, so this applies to both.
    dataset = build_dataset(data_config_path=effective_data_config_path, ds_cfg=ds_cfg)
    if hasattr(dataset, "eo_dropout_prob"):
        dataset.eo_dropout_prob = float(
            max(0.0, min(1.0, float(ds_cfg.get("eo_dropout_prob", 0.0))))
        )
    datamodule = DepthTileDataModule(
        dataset=dataset,
        dataloader_cfg=dataloader_cfg,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        seed=int(ds_cfg.get("random_seed", 7)),
    )

    # Instantiate the conditional model from config.
    model = PixelDiffusionConditional.from_config(
        model_config_path=effective_model_config_path,
        data_config_path=effective_data_config_path,
        training_config_path=effective_training_config_path,
        datamodule=datamodule,
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
        # Keep sanity checks enabled by default, but model-side logic keeps them lightweight.
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 2)),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        limit_val_batches=limit_val_batches,
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    # Start (or resume) training.
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


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
        default="configs/data_config.yaml",
        help="Path to data config yaml.",
    )
    parser.add_argument(
        "--train-config",
        "--training-config",
        default="configs/training_config.yaml",
        dest="training_config",
        help="Path to training config yaml.",
    )
    parser.add_argument(
        "--model-config",
        "--mdoel-config",
        default="configs/model_config.yaml",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        training_config_path=args.training_config,
        overrides=args.config_overrides,
    )

"""
# Training quick start (single command):
python train.py --model-config configs/model_config.yaml \
    --data-config configs/data_config.yaml \
    --training-config configs/training_config.yaml

# Sweep quick start (single command):
./scripts/start_occlusion_sweep.sh
"""