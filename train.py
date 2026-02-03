from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import DepthTileDataModule
from models.difFF import PixelDiffusion, PixelDiffusionConditional


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_wandb_logger(model_cfg: dict[str, Any], model: pl.LightningModule) -> WandbLogger:
    wandb_cfg = model_cfg.get("wandb", {})
    logger = WandbLogger(
        project=wandb_cfg.get("project", "DepthDif"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        log_model=wandb_cfg.get("log_model", "all"),
    )

    # Verbose tracking: gradients, parameters, and graph snapshots.
    logger.watch(
        model,
        log=wandb_cfg.get("watch_log", "all"),
        log_freq=int(wandb_cfg.get("watch_log_freq", 25)),
        log_graph=bool(wandb_cfg.get("watch_log_graph", True)),
    )
    return logger


def main(
    model_config_path: str = "configs/model_config.yaml",
    data_config_path: str = "configs/data_config.yaml",
) -> None:
    model_cfg = load_yaml(model_config_path)
    trainer_cfg = model_cfg.get("trainer", {})
    model_type = model_cfg.get("model", {}).get("model_type", "cond_px_dif")

    datamodule = DepthTileDataModule(
        config_path=data_config_path,
        val_fraction=0.2,
    )

    if model_type == "cond_px_dif":
        model = PixelDiffusionConditional.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            datamodule=datamodule,
        )
    elif model_type == "px_dif":
        model = PixelDiffusion.from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            datamodule=datamodule,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logger = build_wandb_logger(model_cfg, model)

    num_gpus = trainer_cfg.get("num_gpus", None)
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        accelerator = "gpu" if num_gpus > 0 else "cpu"
        devices = num_gpus if num_gpus > 0 else 1
    else:
        accelerator = trainer_cfg.get("accelerator", "auto")
        devices = trainer_cfg.get("devices", "auto")

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 100)),
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "32-true"),
        logger=logger,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
