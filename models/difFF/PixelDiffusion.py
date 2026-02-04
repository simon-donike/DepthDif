from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from .DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
    DenoisingDiffusionProcess,
)
from utils.normalizations import PLOT_CMAP, temperature_to_plot_unit


class PixelDiffusion(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule | None = None,
        generated_channels: int = 1,
        num_timesteps: int = 1000,
        unet_dim: int = 64,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        unet_with_time_emb: bool = True,
        unet_output_mean_scale: bool = False,
        unet_residual: bool = False,
        batch_size: int = 1,
        lr: float = 1e-3,
        wandb_verbose: bool = True,
        log_stats_every_n_steps: int = 1,
        log_images_every_n_steps: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])

        self.datamodule = datamodule
        self.lr = lr
        self.batch_size = batch_size
        self.wandb_verbose = wandb_verbose
        self.log_stats_every_n_steps = max(1, int(log_stats_every_n_steps))
        self.log_images_every_n_steps = max(1, int(log_images_every_n_steps))

        self.model = DenoisingDiffusionProcess(
            generated_channels=generated_channels,
            num_timesteps=num_timesteps,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            unet_with_time_emb=unet_with_time_emb,
            unet_output_mean_scale=unet_output_mean_scale,
            unet_residual=unet_residual,
        )

    @classmethod
    def from_config(
        cls,
        model_config_path: str = "configs/model_config.yaml",
        data_config_path: str = "configs/data_config.yaml",
        datamodule: pl.LightningDataModule | None = None,
    ) -> "PixelDiffusion":
        model_cfg = cls._load_yaml(model_config_path)
        data_cfg = cls._load_yaml(data_config_path)

        m = model_cfg.get("model", {})
        t = model_cfg.get("training", {})
        w = model_cfg.get("wandb", {})
        d = data_cfg.get("dataloader", {})
        unet_kwargs = cls._parse_unet_config(m)

        return cls(
            datamodule=datamodule,
            generated_channels=int(m.get("generated_channels", m.get("bands", 1))),
            num_timesteps=int(m.get("num_timesteps", 1000)),
            **unet_kwargs,
            batch_size=int(t.get("batch_size", d.get("batch_size", 1))),
            lr=float(t.get("lr", 1e-3)),
            wandb_verbose=bool(w.get("verbose", True)),
            log_stats_every_n_steps=int(w.get("log_stats_every_n_steps", 1)),
            log_images_every_n_steps=int(w.get("log_images_every_n_steps", 200)),
        )

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _parse_unet_dim_mults(value: Any) -> tuple[int, ...]:
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            value = parts if parts else [1, 2, 4, 8]
        elif isinstance(value, (int, float)):
            value = [int(value)]
        elif not isinstance(value, (list, tuple)):
            value = [1, 2, 4, 8]

        dim_mults = tuple(int(v) for v in value)
        if len(dim_mults) == 0:
            dim_mults = (1, 2, 4, 8)
        return dim_mults

    @classmethod
    def _parse_unet_config(cls, model_section: dict[str, Any]) -> dict[str, Any]:
        unet = model_section.get("unet", {})
        return {
            "unet_dim": int(unet.get("dim", 64)),
            "unet_dim_mults": cls._parse_unet_dim_mults(unet.get("dim_mults", [1, 2, 4, 8])),
            "unet_with_time_emb": bool(unet.get("with_time_emb", True)),
            "unet_output_mean_scale": bool(unet.get("output_mean_scale", False)),
            "unet_residual": bool(unet.get("residual", False)),
        }

    @torch.no_grad()
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.output_T(self.model(*args, **kwargs))

    def input_T(self, value: torch.Tensor) -> torch.Tensor:
        # Let the model consume [0, 1] range and internally map to [-1, 1].
        return (value.clamp(0, 1) * 2.0) - 1.0

    def output_T(self, value: torch.Tensor) -> torch.Tensor:
        # Inverse transform of model output from [-1, 1] to [0, 1].
        return (value + 1.0) / 2.0

    def _extract_unconditional_target(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch["y"]
        if isinstance(batch, (tuple, list)):
            return batch[-1]
        return batch

    @staticmethod
    def _should_sync_dist() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        target = self._extract_unconditional_target(batch)
        loss = self.model.p_loss(self.input_T(target))

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=target.size(0),
        )
        self._log_common_batch_stats(target, prefix="train", batch_size=int(target.size(0)))
        self._maybe_log_wandb_images(batch=batch, output=target, prefix="train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        target = self._extract_unconditional_target(batch)
        loss = self.model.p_loss(self.input_T(target))

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=target.size(0),
        )
        loss_ckpt = torch.nan_to_num(loss.detach(), nan=1e9, posinf=1e9, neginf=1e9)
        self.log(
            "val/loss_ckpt",
            loss_ckpt,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=target.size(0),
        )
        self._log_common_batch_stats(target, prefix="val", batch_size=int(target.size(0)))
        self._maybe_log_wandb_images(batch=batch, output=target, prefix="val")
        return loss

    def _log_common_batch_stats(self, tensor: torch.Tensor, prefix: str, batch_size: int | None = None) -> None:
        if not self.wandb_verbose:
            return
        if self.global_step % self.log_stats_every_n_steps != 0:
            return

        self.log(f"stats/{prefix}_batch_mean", tensor.mean(), on_step=True, on_epoch=False, logger=True, batch_size=batch_size)
        self.log(f"stats/{prefix}_batch_std", tensor.std(), on_step=True, on_epoch=False, logger=True, batch_size=batch_size)
        self.log(f"stats/{prefix}_batch_min", tensor.min(), on_step=True, on_epoch=False, logger=True, batch_size=batch_size)
        self.log(f"stats/{prefix}_batch_max", tensor.max(), on_step=True, on_epoch=False, logger=True, batch_size=batch_size)

    def _maybe_log_wandb_images(self, batch: Any, output: torch.Tensor, prefix: str) -> None:
        if not self.wandb_verbose:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        if self.global_step % self.log_images_every_n_steps != 0:
            return
        if not hasattr(self.logger, "experiment"):
            return

        experiment = self.logger.experiment
        if not hasattr(experiment, "log"):
            return

        try:
            import wandb
        except Exception:
            return

        image = output[0].detach().float().cpu().squeeze(0)
        experiment.log({f"{prefix}/y_preview": wandb.Image(image.numpy())})

    @staticmethod
    def _minmax_stretch(image: torch.Tensor) -> np.ndarray:
        arr = temperature_to_plot_unit(image, tensor_is_standardized=True)
        return arr.detach().float().cpu().numpy().astype(np.float32)

    def train_dataloader(self):
        if self.datamodule is None:
            raise RuntimeError("No datamodule was provided to the model.")
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        if self.datamodule is None:
            return None
        return self.datamodule.val_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        return optimizer


class PixelDiffusionConditional(PixelDiffusion):
    def __init__(
        self,
        datamodule: pl.LightningDataModule | None = None,
        generated_channels: int = 1,
        condition_channels: int = 1,
        condition_mask_channels: int = 1,
        num_timesteps: int = 1000,
        unet_dim: int = 64,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        unet_with_time_emb: bool = True,
        unet_output_mean_scale: bool = False,
        unet_residual: bool = False,
        batch_size: int = 1,
        lr: float = 1e-3,
        lr_scheduler_enabled: bool = False,
        lr_scheduler_monitor: str = "val/loss_ckpt",
        lr_scheduler_mode: str = "min",
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        lr_scheduler_threshold: float = 1e-4,
        lr_scheduler_threshold_mode: str = "rel",
        lr_scheduler_cooldown: int = 0,
        lr_scheduler_min_lr: float = 0.0,
        lr_scheduler_eps: float = 1e-8,
        wandb_verbose: bool = True,
        log_stats_every_n_steps: int = 1,
        log_images_every_n_steps: int = 200,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["datamodule"])

        self.datamodule = datamodule
        self.lr = lr
        self.batch_size = batch_size
        self.lr_scheduler_enabled = bool(lr_scheduler_enabled)
        self.lr_scheduler_monitor = str(lr_scheduler_monitor)
        self.lr_scheduler_mode = str(lr_scheduler_mode)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_threshold = float(lr_scheduler_threshold)
        self.lr_scheduler_threshold_mode = str(lr_scheduler_threshold_mode)
        self.lr_scheduler_cooldown = int(lr_scheduler_cooldown)
        self.lr_scheduler_min_lr = float(lr_scheduler_min_lr)
        self.lr_scheduler_eps = float(lr_scheduler_eps)
        self.condition_mask_channels = int(max(0, condition_mask_channels))
        self.wandb_verbose = wandb_verbose
        self.log_stats_every_n_steps = max(1, int(log_stats_every_n_steps))
        self.log_images_every_n_steps = max(1, int(log_images_every_n_steps))

        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=generated_channels,
            condition_channels=condition_channels,
            num_timesteps=num_timesteps,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            unet_with_time_emb=unet_with_time_emb,
            unet_output_mean_scale=unet_output_mean_scale,
            unet_residual=unet_residual,
        )

    @classmethod
    def from_config(
        cls,
        model_config_path: str = "configs/model_config.yaml",
        data_config_path: str = "configs/data_config.yaml",
        datamodule: pl.LightningDataModule | None = None,
    ) -> "PixelDiffusionConditional":
        model_cfg = cls._load_yaml(model_config_path)
        data_cfg = cls._load_yaml(data_config_path)

        m = model_cfg.get("model", {})
        t = model_cfg.get("training", {})
        w = model_cfg.get("wandb", {})
        d = data_cfg.get("dataloader", {})
        scheduler_cfg = data_cfg.get("scheduler", {})
        plateau_cfg = scheduler_cfg.get(
            "reduce_on_plateau",
            scheduler_cfg.get("reduce_lr_on_plateau", {}),
        )
        unet_kwargs = cls._parse_unet_config(m)

        return cls(
            datamodule=datamodule,
            generated_channels=int(m.get("generated_channels", 1)),
            condition_channels=int(m.get("condition_channels", m.get("bands", 1))),
            condition_mask_channels=int(m.get("condition_mask_channels", 1)),
            num_timesteps=int(m.get("num_timesteps", 1000)),
            **unet_kwargs,
            batch_size=int(t.get("batch_size", d.get("batch_size", 1))),
            lr=float(t.get("lr", 1e-3)),
            lr_scheduler_enabled=bool(plateau_cfg.get("enabled", False)),
            lr_scheduler_monitor=str(plateau_cfg.get("monitor", "val/loss_ckpt")),
            lr_scheduler_mode=str(plateau_cfg.get("mode", "min")),
            lr_scheduler_factor=float(plateau_cfg.get("factor", 0.5)),
            lr_scheduler_patience=int(plateau_cfg.get("patience", 10)),
            lr_scheduler_threshold=float(plateau_cfg.get("threshold", 1e-4)),
            lr_scheduler_threshold_mode=str(plateau_cfg.get("threshold_mode", "rel")),
            lr_scheduler_cooldown=int(plateau_cfg.get("cooldown", 0)),
            lr_scheduler_min_lr=float(plateau_cfg.get("min_lr", 0.0)),
            lr_scheduler_eps=float(plateau_cfg.get("eps", 1e-8)),
            wandb_verbose=bool(w.get("verbose", True)),
            log_stats_every_n_steps=int(w.get("log_stats_every_n_steps", 1)),
            log_images_every_n_steps=int(w.get("log_images_every_n_steps", 200)),
        )

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tensor.min(), tensor.mean(), tensor.std(unbiased=False)

    def _split_condition_data_and_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        channels = int(x.size(1))
        if channels <= 1:
            return x, None

        mask_channels = min(self.condition_mask_channels, channels - 1)
        if mask_channels <= 0:
            return x, None

        data = x[:, : channels - mask_channels, ...]
        mask = x[:, channels - mask_channels :, ...]
        return data, mask

    def _prepare_condition_for_model(self, x: torch.Tensor) -> torch.Tensor:
        data, mask = self._split_condition_data_and_mask(x)
        data_t = self.input_T(data)
        if mask is None:
            return data_t
        return torch.cat([data_t, mask], dim=1)

    def _masked_fraction(self, x: torch.Tensor) -> torch.Tensor:
        _, mask = self._split_condition_data_and_mask(x)
        if mask is None:
            return (x == 0).float().mean()
        return (mask < 0.5).float().mean()

    def _log_validation_triplet_stats(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        x_data, _ = self._split_condition_data_and_mask(x)
        x_min, x_mean, x_std = self._tensor_stats(x_data)
        y_min, y_mean, y_std = self._tensor_stats(y)
        y_hat_min, y_hat_mean, y_hat_std = self._tensor_stats(y_hat)
        sync_dist = self._should_sync_dist()

        self.log("val_triplet/min_x", x_min, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/min_y", y_min, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/min_y_hat", y_hat_min, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/mean_x", x_mean, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/mean_y", y_mean, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/mean_y_hat", y_hat_mean, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/std_x", x_std, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/std_y", y_std, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))
        self.log("val_triplet/std_y_hat", y_hat_std, on_step=False, on_epoch=True, logger=True, sync_dist=sync_dist, batch_size=y.size(0))

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        model_condition = self._prepare_condition_for_model(x)
        loss = self.model.p_loss(self.input_T(y), model_condition)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = self._masked_fraction(x)
            self.log("train/masked_fraction", masked_fraction, on_step=True, on_epoch=False, logger=True, batch_size=y.size(0))

        self._log_common_batch_stats(y, prefix="train", batch_size=int(y.size(0)))
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        model_condition = self._prepare_condition_for_model(x)
        # Run a full reverse diffusion pass on x during validation to assess real inference quality.
        y_hat = self(model_condition)
        loss = self.model.p_loss(self.input_T(y), model_condition)
        recon_mse = torch.mean((y_hat - y) ** 2)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )
        loss_ckpt = torch.nan_to_num(loss.detach(), nan=1e9, posinf=1e9, neginf=1e9)
        self.log(
            "val/loss_ckpt",
            loss_ckpt,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )
        self.log(
            "val/recon_mse",
            recon_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=y.size(0),
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = self._masked_fraction(x)
            self.log("val/masked_fraction", masked_fraction, on_step=False, on_epoch=True, logger=True, sync_dist=True, batch_size=y.size(0))

        self._log_validation_triplet_stats(x=x, y=y, y_hat=y_hat)
        self._log_common_batch_stats(y_hat, prefix="val_pred", batch_size=int(y.size(0)))
        self._log_common_batch_stats(y, prefix="val_target", batch_size=int(y.size(0)))

        self._maybe_log_wandb_images_conditional(x=x, y_hat=y_hat, y_target=y, prefix="val")
        return loss

    def _maybe_log_wandb_images_conditional(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y_target: torch.Tensor,
        prefix: str,
    ) -> None:
        if not self.wandb_verbose:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        if self.global_step % self.log_images_every_n_steps != 0:
            return
        if not hasattr(self.logger, "experiment"):
            return

        experiment = self.logger.experiment
        if not hasattr(experiment, "log"):
            return

        try:
            import wandb
        except Exception:
            return

        num_to_plot = min(5, int(x.size(0)))
        if num_to_plot <= 0:
            return

        fig, axes = plt.subplots(num_to_plot, 3, figsize=(12, 3 * num_to_plot), squeeze=False)
        x_data, _ = self._split_condition_data_and_mask(x)

        for i in range(num_to_plot):
            x_img = self._minmax_stretch(x_data[i, 0])
            y_hat_img = self._minmax_stretch(y_hat[i, 0])
            y_target_img = self._minmax_stretch(y_target[i, 0])

            axes[i, 0].imshow(x_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[i, 0].set_axis_off()
            axes[i, 1].imshow(y_hat_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[i, 1].set_axis_off()
            axes[i, 2].imshow(y_target_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[i, 2].set_axis_off()

            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Reconstruction")
            axes[i, 2].set_title("Target")

        fig.tight_layout()
        experiment.log({f"{prefix}/x_y_batch_preview": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        if not self.lr_scheduler_enabled:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.lr_scheduler_mode,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            threshold_mode=self.lr_scheduler_threshold_mode,
            cooldown=self.lr_scheduler_cooldown,
            min_lr=self.lr_scheduler_min_lr,
            eps=self.lr_scheduler_eps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
