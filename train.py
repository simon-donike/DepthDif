from __future__ import annotations

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
from models.difFF import PixelDiffusion, PixelDiffusionConditional


# Centralized YAML loader for config files.
def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Resolve an optional resume checkpoint path and validate it early.
def resolve_resume_ckpt_path(model_cfg: dict[str, Any]) -> str | None:
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
    has_explicit_toggles = "watch_gradients" in wandb_cfg or "watch_parameters" in wandb_cfg
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


def build_wandb_logger(training_cfg: dict[str, Any], model: pl.LightningModule) -> WandbLogger:
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
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return

    for cfg_path in config_paths:
        path = Path(cfg_path)
        if path.is_file():
            # Store configs as run files for reproducibility and easy download from UI.
            experiment.save(str(path.resolve()), policy="now")


def main(
    model_config_path: str = "configs/model_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
) -> None:
    # Determine rank before creating any run-scoped folders/files.
    global_rank = resolve_global_rank()
    is_global_zero = global_rank == 0

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

    # Data module encapsulates train/val dataset construction and loaders.
    datamodule = DepthTileDataModule(
        config_path=data_config_path,
        training_config_path=training_config_path,
    )

    # Instanciate appropriate model class from config.
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
        upload_configs_to_wandb(
            logger,
            [model_config_path, data_config_path, training_config_path],
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor=str(trainer_cfg.get("ckpt_monitor", "val/loss_ckpt")),
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval=str(trainer_cfg.get("lr_logging_interval", "step"))
    )

    # Build device settings from config
    num_gpus = trainer_cfg.get("num_gpus", None)
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        accelerator = "gpu" if num_gpus > 0 else "cpu"
        devices = num_gpus if num_gpus > 0 else 1
    else:
        accelerator = trainer_cfg.get("accelerator", "auto")
        devices = trainer_cfg.get("devices", "auto")

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
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    # Start (or resume) training.
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
