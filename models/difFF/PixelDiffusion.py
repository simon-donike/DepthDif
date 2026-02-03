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


class PixelDiffusion(pl.LightningModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule | None = None,
        generated_channels: int = 1,
        num_timesteps: int = 1000,
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

        return cls(
            datamodule=datamodule,
            generated_channels=int(m.get("generated_channels", m.get("bands", 1))),
            num_timesteps=int(m.get("num_timesteps", 1000)),
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
            batch_size=target.size(0),
        )
        self._log_common_batch_stats(target, prefix="train")
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
            batch_size=target.size(0),
        )
        self._log_common_batch_stats(target, prefix="val")
        self._maybe_log_wandb_images(batch=batch, output=target, prefix="val")
        return loss

    def _log_common_batch_stats(self, tensor: torch.Tensor, prefix: str) -> None:
        if not self.wandb_verbose:
            return
        if self.global_step % self.log_stats_every_n_steps != 0:
            return

        self.log(f"{prefix}/batch_mean", tensor.mean(), on_step=True, on_epoch=False, logger=True)
        self.log(f"{prefix}/batch_std", tensor.std(), on_step=True, on_epoch=False, logger=True)
        self.log(f"{prefix}/batch_min", tensor.min(), on_step=True, on_epoch=False, logger=True)
        self.log(f"{prefix}/batch_max", tensor.max(), on_step=True, on_epoch=False, logger=True)

    def _maybe_log_wandb_images(self, batch: Any, output: torch.Tensor, prefix: str) -> None:
        if not self.wandb_verbose:
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
        experiment.log(
            {f"{prefix}/y_preview": wandb.Image(image.numpy())},
            step=self.global_step,
        )

    @staticmethod
    def _minmax_stretch(image: torch.Tensor) -> np.ndarray:
        arr = image.detach().float().cpu().numpy()
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        if arr_max <= arr_min:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)

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
        num_timesteps: int = 1000,
        batch_size: int = 1,
        lr: float = 1e-3,
        wandb_verbose: bool = True,
        log_stats_every_n_steps: int = 1,
        log_images_every_n_steps: int = 200,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["datamodule"])

        self.datamodule = datamodule
        self.lr = lr
        self.batch_size = batch_size
        self.wandb_verbose = wandb_verbose
        self.log_stats_every_n_steps = max(1, int(log_stats_every_n_steps))
        self.log_images_every_n_steps = max(1, int(log_images_every_n_steps))

        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=generated_channels,
            condition_channels=condition_channels,
            num_timesteps=num_timesteps,
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

        return cls(
            datamodule=datamodule,
            generated_channels=int(m.get("generated_channels", 1)),
            condition_channels=int(m.get("condition_channels", m.get("bands", 1))),
            num_timesteps=int(m.get("num_timesteps", 1000)),
            batch_size=int(t.get("batch_size", d.get("batch_size", 1))),
            lr=float(t.get("lr", 1e-3)),
            wandb_verbose=bool(w.get("verbose", True)),
            log_stats_every_n_steps=int(w.get("log_stats_every_n_steps", 1)),
            log_images_every_n_steps=int(w.get("log_images_every_n_steps", 200)),
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        loss = self.model.p_loss(self.input_T(y), self.input_T(x))

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = (x == 0).float().mean()
            self.log("train/masked_fraction", masked_fraction, on_step=True, on_epoch=False, logger=True)

        self._log_common_batch_stats(y, prefix="train")
        self._maybe_log_wandb_images_conditional(x=x, y=y, prefix="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]
        # Run a full reverse diffusion pass on x during validation to assess real inference quality.
        y_hat = self(self.input_T(x))
        loss = self.model.p_loss(self.input_T(y), self.input_T(x))
        recon_mse = torch.mean((y_hat - y) ** 2)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
        )
        self.log(
            "val/recon_mse",
            recon_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.size(0),
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = (x == 0).float().mean()
            self.log("val/masked_fraction", masked_fraction, on_step=False, on_epoch=True, logger=True)

        self._log_common_batch_stats(y_hat, prefix="val_pred")
        self._maybe_log_wandb_images_conditional(x=x, y=y_hat, prefix="val")
        return loss

    def _maybe_log_wandb_images_conditional(self, x: torch.Tensor, y: torch.Tensor, prefix: str) -> None:
        if not self.wandb_verbose:
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

        fig, axes = plt.subplots(num_to_plot, 2, figsize=(8, 3 * num_to_plot), squeeze=False)

        for i in range(num_to_plot):
            x_img = self._minmax_stretch(x[i].squeeze(0))
            y_img = self._minmax_stretch(y[i].squeeze(0))

            axes[i, 0].imshow(x_img, cmap="viridis")
            axes[i, 0].set_axis_off()
            axes[i, 1].imshow(y_img, cmap="viridis")
            axes[i, 1].set_axis_off()

            axes[i, 0].set_title("X")
            axes[i, 1].set_title("Y")

        fig.tight_layout()
        experiment.log(
            {
                f"{prefix}/x_y_batch_preview": wandb.Image(fig),
            },
            step=self.global_step,
        )
        plt.close(fig)
