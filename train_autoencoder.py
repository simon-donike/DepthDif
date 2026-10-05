# Example: /work/envs/depth/bin/python train_autoencoder.py --ae-config src/depth_recon/configs/lat_space/ae_config.yaml --data-config src/depth_recon/configs/lat_space/training_super_config.yaml --train-config src/depth_recon/configs/lat_space/training_config.yaml --run-dir logs/ae_latent
# Resume alternative: append --resume-checkpoint logs/ae_latent/last.ckpt; weight-only alternative: append --load-checkpoint logs/ae_latent/final.ckpt.
"""Train the depth autoencoder used by the latent workflow.

This script loads the configured dataset/datamodule, builds the autoencoder
Lightning module, restores checkpoints if configured, and runs the training job.

Typical CLI:
    /work/envs/depth/bin/python train_autoencoder.py --data-config src/depth_recon/configs/lat_space/training_super_config.yaml --train-config src/depth_recon/configs/lat_space/training_config.yaml --ae-config src/depth_recon/configs/lat_space/ae_config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Any

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
from depth_recon.models.latent import DepthBandAutoencoderLightning
from depth_recon.paths import config_path, resolve_config_path
from depth_recon.configs.config_resolver_pixel import (
    FULL_RECONSTRUCTION_MONITOR,
    apply_reconstruction_checkpoint_contract,
)
from depth_recon.models.latent.workflow import (
    AutoencoderReconstructionValidation,
    export_calibrated_autoencoder,
)
from depth_recon.configs.config_resolver_pixel import load_pixel_training_config

LAT_AE_CONFIG_PATH = str(config_path("lat_space", "ae_config.yaml"))
PX_DATA_CONFIG_PATH = str(config_path("lat_space", "training_super_config.yaml"))
LAT_TRAINING_CONFIG_PATH = str(config_path("lat_space", "training_config.yaml"))


def load_yaml(path: str) -> dict[str, Any]:
    """Load and return yaml data."""
    with resolve_config_path(path).open("r", encoding="utf-8") as f:
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
        default="argo_geotiff_gridded",
    )
    _ = data_config_path
    return str(variant).strip().lower()


def build_dataset(
    data_config_path: str,
    ds_cfg: dict[str, Any],
    split: str = "train",
) -> torch.utils.data.Dataset:
    """Build and return dataset."""
    dataset_variant = resolve_dataset_variant(ds_cfg, data_config_path)
    if dataset_variant == "argo_geotiff_gridded":
        return ArgoGeoTIFFGriddedPatchDataset.from_config(data_config_path, split=split)
    raise ValueError(
        "Unsupported dataset variant "
        f"'{dataset_variant}'. Expected one of "
        "['argo_geotiff_gridded']."
    )


def build_datamodule(
    dataset: torch.utils.data.Dataset,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    val_dataset: torch.utils.data.Dataset | None = None,
) -> DepthTileDataModule:
    """Build and return datamodule."""
    split_cfg = data_cfg.get("split", {})
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    data_dataloader_cfg = data_cfg.get("dataloader", {})
    if "val_shuffle" in data_dataloader_cfg:
        dataloader_cfg["val_shuffle"] = bool(data_dataloader_cfg["val_shuffle"])

    return DepthTileDataModule(
        dataset=dataset,
        val_dataset=val_dataset,
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
        log_model=wandb_cfg.get("log_model", False),
        offline=bool(wandb_cfg.get("offline", True)),
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
        default=LAT_AE_CONFIG_PATH,
        help="Path to autoencoder config yaml.",
    )
    parser.add_argument(
        "--data-config",
        default=PX_DATA_CONFIG_PATH,
        help="Path to data config yaml.",
    )
    parser.add_argument(
        "--train-config",
        "--training-config",
        default=LAT_TRAINING_CONFIG_PATH,
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
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output directory; required for externally launched DDP.",
    )
    return parser.parse_args()


def main(
    *,
    ae_config_path: str,
    data_config_path: str,
    training_config_path: str,
    resume_checkpoint: str | None,
    load_checkpoint: str | None,
    run_dir_value: str | None = None,
) -> None:
    """Run the script entry point."""
    ae_config_path = str(resolve_config_path(ae_config_path))
    data_config_path = str(resolve_config_path(data_config_path))
    training_config_path = str(resolve_config_path(training_config_path))

    global_rank = resolve_global_rank()
    is_global_zero = global_rank == 0

    run_dir_value = run_dir_value or os.environ.get("DEPTHDIF_AE_RUN_DIR")
    if run_dir_value is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "Externally launched DDP requires --run-dir shared across ranks."
        )
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(run_dir_value) if run_dir_value else Path("logs") / f"ae_{run_stamp}"
    if run_dir_value is None:
        suffix = 1
        while run_dir.exists():
            run_dir = Path("logs") / f"ae_{run_stamp}_{suffix:02d}"
            suffix += 1
    # Lightning subprocess launch inherits the resolved path from the parent.
    os.environ["DEPTHDIF_AE_RUN_DIR"] = str(run_dir.resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_global_zero:
        shutil.copy2(ae_config_path, run_dir / Path(ae_config_path).name)
        shutil.copy2(data_config_path, run_dir / Path(data_config_path).name)
        shutil.copy2(training_config_path, run_dir / Path(training_config_path).name)

    ae_cfg = load_yaml(ae_config_path)
    data_cfg = load_yaml(data_config_path)
    if "data" in data_cfg and "dataset" not in data_cfg:
        # Pixel super-configs wrap the dataset config under top-level data.
        data_cfg = data_cfg["data"]
    training_cfg = load_yaml(training_config_path)
    apply_reconstruction_checkpoint_contract(training_cfg)
    if is_global_zero:
        with (run_dir / "training_config_effective.yaml").open(
            "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(training_cfg, f, sort_keys=False)
    bundle = load_pixel_training_config(
        config_path_value=data_config_path,
        runtime_config_dir=run_dir / f"effective_rank{global_rank}",
        write_snapshots=False,
    )
    data_cfg = bundle.data_cfg
    data_config_path = bundle.effective_data_config_path
    # AE optimization uses its own config, but data/scenario resolution matches diffusion.
    ae_training = ae_cfg.get("ae", {}).get("training", {})
    if "max_epochs" in ae_training:
        training_cfg.setdefault("trainer", {})["max_epochs"] = int(
            ae_training["max_epochs"]
        )
    if "batch_size" in ae_training:
        training_cfg.setdefault("dataloader", {})["batch_size"] = int(
            ae_training["batch_size"]
        )
    training_cfg.setdefault("scheduler", {}).setdefault("reduce_on_plateau", {})[
        "monitor"
    ] = FULL_RECONSTRUCTION_MONITOR
    training_config_path = str(
        run_dir / f"effective_rank{global_rank}" / "ae_training_effective.yaml"
    )
    with Path(training_config_path).open("w") as handle:
        yaml.safe_dump(training_cfg, handle, sort_keys=False)

    trainer_cfg = training_cfg.get("trainer", {})

    dataset = build_dataset(data_config_path, data_cfg.get("dataset", {}))
    val_dataset = build_dataset(
        data_config_path, data_cfg.get("dataset", {}), split="val"
    )
    datamodule = build_datamodule(
        dataset=dataset,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        val_dataset=val_dataset,
    )

    model = DepthBandAutoencoderLightning.from_configs(
        ae_config_path=ae_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )

    expected_fields = tuple(bundle.model_cfg["model"]["output_fields"])
    expected_channels = int(
        bundle.model_cfg["model"].get(
            "physical_channels", bundle.model_cfg["model"]["generated_channels"]
        )
    )
    if (
        model.output_fields != expected_fields
        or model.model.in_channels != expected_channels
    ):
        raise ValueError(
            "AE field order/channels must match the selected data scenario."
        )
    if model.model.climatology_residual != bool(
        bundle.model_cfg["model"].get("climatology_residual", False)
    ):
        raise ValueError("AE and data configuration residual modes must match.")
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location="cpu", weights_only=False)
        model.on_load_checkpoint(checkpoint)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model weights from checkpoint: {load_checkpoint}")

    logger = build_wandb_logger(training_cfg)
    if is_global_zero:
        upload_configs_to_wandb(
            logger,
            [ae_config_path, data_config_path, training_config_path],
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-epoch{epoch:03d}-step{step:09d}",
        monitor=FULL_RECONSTRUCTION_MONITOR,
        mode="min",
        save_top_k=3,
        save_last=True,
        save_on_train_epoch_end=False,
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
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
            AutoencoderReconstructionValidation(
                output_dir=run_dir,
                **training_cfg["training"]["reconstruction_eval"],
            ),
        ],
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        limit_val_batches=limit_val_batches,
        enable_model_summary=bool(trainer_cfg.get("enable_model_summary", True)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.0)),
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_checkpoint)
    trainer.save_checkpoint(str(run_dir / "final.ckpt"))
    if trainer.is_global_zero:
        if not checkpoint_callback.best_model_path:
            raise RuntimeError(
                "AE export requires a completed fixed-subset validation."
            )
        checkpoint = torch.load(
            checkpoint_callback.best_model_path,
            map_location=model.device,
            weights_only=False,
        )
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        export_calibrated_autoencoder(
            model,
            datamodule.train_dataloader(),
            run_dir / "autoencoder_calibrated.ckpt",
            max_batches=int(ae_training.get("calibration_batches", 32)),
        )
    trainer.strategy.barrier()


if __name__ == "__main__":
    args = parse_args()
    main(
        ae_config_path=args.ae_config,
        data_config_path=args.data_config,
        training_config_path=args.training_config,
        resume_checkpoint=args.resume_checkpoint,
        load_checkpoint=args.load_checkpoint,
        run_dir_value=args.run_dir,
    )
