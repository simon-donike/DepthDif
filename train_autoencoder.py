"""Train the depth autoencoder used by the latent workflow.

This script loads the configured dataset/datamodule, builds the autoencoder
Lightning module, restores checkpoints if configured, and runs the training job.

Typical CLI:
    /work/envs/depth/bin/python train_autoencoder.py --data-config configs/lat_space/data_config.yaml --train-config configs/lat_space/training_config.yaml --ae-config configs/lat_space/ae_config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from data.dataset_ostia import SurfaceTempPatchOstiaLightDataset
from models.latent import DepthBandAutoencoderLightning


def load_yaml(path: str) -> dict[str, Any]:
    """Load and return yaml data."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_global_rank() -> int:
    """Resolve and validate global rank."""
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
    data_config_path: str,
    ds_cfg: dict[str, Any],
) -> torch.utils.data.Dataset:
    """Build and return dataset."""
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant in {"eo_4band", "4band_eo", "4bands"}:
        return SurfaceTempPatch4BandsLightDataset.from_config(data_config_path, split="all")
    if dataset_variant in {"ostia", "ostia_4band", "4band_ostia"}:
        return SurfaceTempPatchOstiaLightDataset.from_config(data_config_path, split="all")
    raise ValueError(
        f"Unsupported dataset variant '{dataset_variant}'. Expected one of ['eo_4band', 'ostia']."
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


def build_wandb_logger(training_cfg: dict[str, Any]) -> WandbLogger:
    """Build and return wandb logger."""
    wandb_cfg = training_cfg.get("wandb", {})
    return WandbLogger(
        project=wandb_cfg.get("project", "DepthDif"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name", "autoencoder"),
        log_model=wandb_cfg.get("log_model", "all"),
    )


def upload_configs_to_wandb(logger: WandbLogger, config_paths: list[str]) -> None:
    """Upload configs to wandb to experiment tracking."""
    experiment = getattr(logger, "experiment", None)
    if experiment is None:
        return
    for cfg_path in config_paths:
        path = Path(cfg_path)
        if path.is_file():
            experiment.save(str(path.resolve()), policy="now")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for autoencoder training."""
    parser = argparse.ArgumentParser(description="Train depth-band autoencoder.")
    parser.add_argument(
        "--ae-config",
        default="configs/lat_space/ae_config.yaml",
        help="Path to autoencoder config yaml.",
    )
    parser.add_argument(
        "--data-config",
        default="configs/lat_space/data_config.yaml",
        help="Path to data config yaml.",
    )
    parser.add_argument(
        "--train-config",
        "--training-config",
        default="configs/lat_space/training_config.yaml",
        dest="training_config",
        help="Path to training config yaml.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path to resume full Lightning training state.",
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Optional checkpoint path to load model state_dict only.",
    )
    return parser.parse_args()


def main(
    *,
    ae_config_path: str,
    data_config_path: str,
    training_config_path: str,
    resume_checkpoint: str | None,
    load_checkpoint: str | None,
) -> None:
    """Run the script entry point."""
    global_rank = resolve_global_rank()
    is_global_zero = global_rank == 0

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("logs") / f"ae_{run_stamp}"
    if is_global_zero:
        suffix = 1
        while run_dir.exists():
            run_dir = Path("logs") / f"ae_{run_stamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(ae_config_path, run_dir / Path(ae_config_path).name)
        shutil.copy2(data_config_path, run_dir / Path(data_config_path).name)
        shutil.copy2(training_config_path, run_dir / Path(training_config_path).name)

    ae_cfg = load_yaml(ae_config_path)
    data_cfg = load_yaml(data_config_path)
    training_cfg = load_yaml(training_config_path)
    _ = ae_cfg

    trainer_cfg = training_cfg.get("trainer", {})

    dataset = build_dataset(data_config_path, data_cfg.get("dataset", {}))
    datamodule = build_datamodule(dataset=dataset, data_cfg=data_cfg, training_cfg=training_cfg)

    model = DepthBandAutoencoderLightning.from_configs(
        ae_config_path=ae_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from checkpoint: {load_checkpoint}")

    logger = build_wandb_logger(training_cfg)
    if is_global_zero:
        upload_configs_to_wandb(
            logger,
            [ae_config_path, data_config_path, training_config_path],
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-epoch{epoch:03d}",
        monitor=str(trainer_cfg.get("ckpt_monitor", "val/loss_ckpt")),
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor(
        logging_interval=str(trainer_cfg.get("lr_logging_interval", "epoch"))
    )

    num_gpus = trainer_cfg.get("num_gpus", None)
    if num_gpus is not None:
        num_gpus = int(num_gpus)
        accelerator = "gpu" if num_gpus > 0 else "cpu"
        devices = num_gpus if num_gpus > 0 else 1
    else:
        accelerator = trainer_cfg.get("accelerator", "auto")
        devices = trainer_cfg.get("devices", "auto")

    val_batches_per_epoch = trainer_cfg.get("val_batches_per_epoch", None)
    if val_batches_per_epoch is not None:
        limit_val_batches = int(val_batches_per_epoch)
        if limit_val_batches < 1:
            raise ValueError("trainer.val_batches_per_epoch must be >= 1 when set.")
    else:
        limit_val_batches = trainer_cfg.get("limit_val_batches", 1.0)

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 300)),
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "32-true"),
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 2)),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        limit_val_batches=limit_val_batches,
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    args = parse_args()
    main(
        ae_config_path=args.ae_config,
        data_config_path=args.data_config,
        training_config_path=args.training_config,
        resume_checkpoint=args.resume_checkpoint,
        load_checkpoint=args.load_checkpoint,
    )
