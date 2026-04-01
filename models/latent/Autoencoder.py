from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from utils.normalizations import temperature_normalize


class DepthBandAutoencoder(nn.Module):
    """Band-first autoencoder used to compress multiband depth tensors."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        encoder_hidden_channels: Sequence[int] = (64, 96, 128),
        decoder_hidden_channels: Sequence[int] = (128, 96, 64),
        spatial_downsample: int = 1,
    ) -> None:
        """Initialize DepthBandAutoencoder with configured parameters.

        Args:
            in_channels (int): Input value.
            latent_channels (int): Input value.
            encoder_hidden_channels (Sequence[int]): Input value.
            decoder_hidden_channels (Sequence[int]): Input value.
            spatial_downsample (int): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.spatial_downsample = max(1, int(spatial_downsample))

        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1.")
        if self.latent_channels < 1:
            raise ValueError("latent_channels must be >= 1.")

        enc_hidden = tuple(int(v) for v in encoder_hidden_channels)
        dec_hidden = tuple(int(v) for v in decoder_hidden_channels)

        enc_layers: list[nn.Module] = []
        prev = self.in_channels
        for width in enc_hidden:
            if width < 1:
                continue
            enc_layers.append(nn.Conv2d(prev, width, kernel_size=3, padding=1))
            enc_layers.append(nn.GELU())
            prev = width
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else nn.Identity()
        self.to_latent = nn.Conv2d(prev, self.latent_channels, kernel_size=1)

        dec_layers: list[nn.Module] = []
        prev = self.latent_channels
        for width in dec_hidden:
            if width < 1:
                continue
            dec_layers.append(nn.Conv2d(prev, width, kernel_size=3, padding=1))
            dec_layers.append(nn.GELU())
            prev = width
        self.decoder = nn.Sequential(*dec_layers) if dec_layers else nn.Identity()
        self.to_output = nn.Conv2d(prev, self.in_channels, kernel_size=1)

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def from_config(cls, config_path: str) -> "DepthBandAutoencoder":
        """Build a depth-band autoencoder from ae config yaml."""
        cfg = cls._load_yaml(config_path)
        ae = cfg.get("ae", {})
        encoder_cfg = ae.get("encoder", {})
        decoder_cfg = ae.get("decoder", {})
        return cls(
            in_channels=int(ae.get("in_channels", 50)),
            latent_channels=int(ae.get("latent_channels", 12)),
            encoder_hidden_channels=encoder_cfg.get("hidden_channels", [64, 96, 128]),
            decoder_hidden_channels=decoder_cfg.get("hidden_channels", [128, 96, 64]),
            spatial_downsample=int(ae.get("spatial_downsample", 1)),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Encode full-band tensor into latent space."""
        if value.ndim != 4:
            raise RuntimeError(
                f"Autoencoder expects 4D tensors (B,C,H,W), got shape {tuple(value.shape)}."
            )
        if int(value.size(1)) != self.in_channels:
            raise RuntimeError(
                "Autoencoder input channel mismatch: "
                f"got {int(value.size(1))}, expected {self.in_channels}."
            )
        x = value
        if self.spatial_downsample > 1:
            x = F.avg_pool2d(
                x,
                kernel_size=self.spatial_downsample,
                stride=self.spatial_downsample,
            )
        return self.to_latent(self.encoder(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor back to full-band space."""
        if latent.ndim != 4:
            raise RuntimeError(
                f"Latent tensor must be 4D (B,C,H,W), got shape {tuple(latent.shape)}."
            )
        if int(latent.size(1)) != self.latent_channels:
            raise RuntimeError(
                "Autoencoder latent channel mismatch: "
                f"got {int(latent.size(1))}, expected {self.latent_channels}."
            )
        y = self.to_output(self.decoder(latent))
        if self.spatial_downsample > 1:
            y = F.interpolate(
                y,
                scale_factor=float(self.spatial_downsample),
                mode="bilinear",
                align_corners=False,
            )
        return y

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the autoencoder forward computation."""
        return self.decode(self.encode(value))


class DepthBandAutoencoderLightning(pl.LightningModule):
    """Lightning wrapper that trains the depth-band autoencoder."""

    def __init__(
        self,
        *,
        autoencoder: DepthBandAutoencoder,
        datamodule: pl.LightningDataModule | None = None,
        lr: float = 1e-4,
        batch_size: int = 1,
        recon_l1_weight: float = 1.0,
        recon_l2_weight: float = 0.5,
        masked_only: bool = False,
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
    ) -> None:
        """Initialize DepthBandAutoencoderLightning with configured parameters.

        Args:
            autoencoder (DepthBandAutoencoder): Input value.
            datamodule (pl.LightningDataModule | None): Input value.
            lr (float): Input value.
            batch_size (int): Size/count parameter.
            recon_l1_weight (float): Input value.
            recon_l2_weight (float): Input value.
            masked_only (bool): Boolean flag controlling behavior.
            lr_scheduler_enabled (bool): Boolean flag controlling behavior.
            lr_scheduler_monitor (str): Input value.
            lr_scheduler_mode (str): Input value.
            lr_scheduler_factor (float): Input value.
            lr_scheduler_patience (int): Input value.
            lr_scheduler_threshold (float): Input value.
            lr_scheduler_threshold_mode (str): Input value.
            lr_scheduler_cooldown (int): Input value.
            lr_scheduler_min_lr (float): Input value.
            lr_scheduler_eps (float): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule", "autoencoder"])
        self.model = autoencoder
        self.datamodule = datamodule
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.recon_l1_weight = float(recon_l1_weight)
        self.recon_l2_weight = float(recon_l2_weight)
        self.masked_only = bool(masked_only)

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

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def from_configs(
        cls,
        *,
        ae_config_path: str,
        training_config_path: str,
        datamodule: pl.LightningDataModule | None = None,
    ) -> "DepthBandAutoencoderLightning":
        """Build Lightning AE module from ae/training config files."""
        ae_cfg = cls._load_yaml(ae_config_path)
        training_cfg = cls._load_yaml(training_config_path)

        ae = ae_cfg.get("ae", {})
        ae_training = ae.get("training", {})
        ae_loss = ae.get("loss", {})

        t = training_cfg.get("training", {})
        d = training_cfg.get("dataloader", {})
        scheduler_cfg = training_cfg.get("scheduler", {})
        plateau_cfg = scheduler_cfg.get(
            "reduce_on_plateau",
            scheduler_cfg.get("reduce_lr_on_plateau", {}),
        )

        return cls(
            autoencoder=DepthBandAutoencoder.from_config(ae_config_path),
            datamodule=datamodule,
            lr=float(ae_training.get("lr", t.get("lr", 1e-4))),
            batch_size=int(ae_training.get("batch_size", d.get("batch_size", 1))),
            recon_l1_weight=float(ae_loss.get("recon_l1_weight", 1.0)),
            recon_l2_weight=float(ae_loss.get("recon_l2_weight", 0.5)),
            masked_only=bool(ae_loss.get("masked_only", False)),
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
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        if self.datamodule is None:
            raise RuntimeError("No datamodule was provided to the model.")
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> torch.utils.data.DataLoader[Any] | None:
        if self.datamodule is None:
            return None
        return self.datamodule.val_dataloader()

    @staticmethod
    def _align_mask_to_reference(
        mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        m = mask
        if m.ndim == 3:
            m = m.unsqueeze(1)
        if m.ndim != 4:
            raise RuntimeError("valid_mask must be shaped as (B,C,H,W) or (B,H,W).")
        if m.shape[0] != reference.shape[0] or m.shape[2:] != reference.shape[2:]:
            raise RuntimeError(
                f"Mask shape {tuple(m.shape)} is incompatible with reference {tuple(reference.shape)}."
            )
        if m.size(1) != reference.size(1):
            # Collapse to one shared spatial mask and broadcast so arbitrary
            # source-band masks can supervise any AE channel count.
            m = m.amax(dim=1, keepdim=True)
            if reference.size(1) > 1:
                m = m.expand(-1, reference.size(1), -1, -1)
        return (m > 0.5).to(device=reference.device, dtype=reference.dtype)

    def _recon_losses(
        self,
        target: torch.Tensor,
        recon: torch.Tensor,
        y_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.masked_only:
            return F.l1_loss(recon, target), F.mse_loss(recon, target)

        aligned_valid = self._align_mask_to_reference(y_valid_mask, target)
        if aligned_valid is None:
            return F.l1_loss(recon, target), F.mse_loss(recon, target)

        denom = aligned_valid.sum()
        if float(denom.item()) <= 0.0:
            return F.l1_loss(recon, target), F.mse_loss(recon, target)

        abs_diff = (recon - target).abs() * aligned_valid
        sq_diff = ((recon - target) ** 2) * aligned_valid
        return abs_diff.sum() / denom, sq_diff.sum() / denom

    def _shared_step(self, batch: dict[str, Any], *, prefix: str) -> torch.Tensor:
        y = batch["y"]
        y_valid_mask = batch.get("y_valid_mask")

        recon = self.model(y)
        loss_l1, loss_l2 = self._recon_losses(y, recon, y_valid_mask)
        loss = (self.recon_l1_weight * loss_l1) + (self.recon_l2_weight * loss_l2)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=int(y.size(0)),
        )
        if prefix == "val":
            loss_ckpt = torch.nan_to_num(loss.detach(), nan=1e9, posinf=1e9, neginf=1e9)
            self.log(
                "val/loss_ckpt",
                loss_ckpt,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=int(y.size(0)),
            )
        self.log(
            f"{prefix}/loss_l1",
            loss_l1,
            on_step=(prefix == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=int(y.size(0)),
        )
        self.log(
            f"{prefix}/loss_l2",
            loss_l2,
            on_step=(prefix == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=int(y.size(0)),
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute training step and return the result."""
        _ = batch_idx
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute validation step and return the result."""
        _ = batch_idx
        return self._shared_step(batch, prefix="val")

    @torch.no_grad()
    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Compute predict step and return reconstructions."""
        _ = (batch_idx, dataloader_idx)
        y = batch["y"]
        recon = self.model(y)
        return {
            "y_hat": recon,
            "y_hat_denorm": temperature_normalize(mode="denorm", tensor=recon),
        }

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Create optimizer and optional scheduler configuration."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
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
