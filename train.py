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
from models.difFF import PixelDiffusion, PixelDiffusionConditional


# Centralized YAML loader for config files.
def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Resolve an optional resume checkpoint path and validate it early.
def resolve_resume_ckpt_path(model_cfg: dict[str, Any]) -> str | None:
    # Accept false/null to start fresh; otherwise require a valid checkpoint path string.
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
    variant = ds_cfg.get("dataset_variant", ds_cfg.get("variant", None))
    if variant is None:
        # Backward-compatible fallback: infer from config filename if explicit variant is absent.
        stem = Path(data_config_path).stem.lower()
        if "4band" in stem or "eo" in stem:
            return "eo_4band"
        return "temp_v1"
    return str(variant).strip().lower()


def build_dataset(data_config_path: str, ds_cfg: dict[str, Any]):
    # Route to dataset implementation matching the requested training task.
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


def main(
    model_config_path: str = "configs/model_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
) -> None:
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

    # Load configuration and choose model family.
    model_cfg = load_yaml(model_config_path)
    training_cfg = load_yaml(training_config_path)
    data_cfg = load_yaml(data_config_path)
    # Resolve resume path once so failure happens early before trainer/model setup.
    resume_ckpt_path = resolve_resume_ckpt_path(model_cfg)
    trainer_cfg = training_cfg.get("trainer", model_cfg.get("trainer", {}))
    model_type = model_cfg.get("model", {}).get("model_type", "cond_px_dif")

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
    dataset = build_dataset(data_config_path=data_config_path, ds_cfg=ds_cfg)
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

    # Instanciate appropriate model class from config.
    # Both model builders receive the datamodule because validation utilities query loaders from it.
    if model_type == "cond_px_dif":
        model = PixelDiffusionConditional.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_config_path=training_config_path,
            datamodule=datamodule,
        )
    elif model_type == "px_dif":
        model = PixelDiffusion.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_config_path=training_config_path,
            datamodule=datamodule,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Set up experiment tracking and best-checkpoint saving.
    logger = build_wandb_logger(training_cfg, model)
    if is_global_zero:
        # Avoid duplicate uploads from DDP worker ranks.
        upload_configs_to_wandb(
            logger,
            [model_config_path, data_config_path, training_config_path],
        )
    # Save only top-k checkpoints by monitored validation metric.
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor=str(trainer_cfg.get("ckpt_monitor", "val/loss")),
        mode="min",
        save_top_k=2,
        save_last=False,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        training_config_path=args.training_config,
    )
