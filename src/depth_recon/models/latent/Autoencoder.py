from __future__ import annotations

from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

from depth_recon.paths import resolve_config_path
from depth_recon.utils.normalizations import (
    temperature_normalize,
    salinity_normalize,
    Y_MEAN,
    Y_STD,
    SALINITY_MEAN,
    SALINITY_STD,
    CELSIUS_TO_KELVIN_OFFSET,
)


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
        output_fields: Sequence[str] = ("temperature",),
        climatology_residual: bool = False,
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
        self.spatial_downsample = int(spatial_downsample)
        if self.spatial_downsample != 1:
            raise ValueError(
                "Mask-aware latent workflow requires spatial_downsample=1."
            )
        self.output_fields = tuple(output_fields)
        if self.output_fields not in (
            ("temperature",),
            ("salinity",),
            ("temperature", "salinity"),
        ):
            raise ValueError("Unsupported autoencoder output_fields/order.")
        if self.in_channels % len(self.output_fields):
            raise ValueError("AE channels must divide evenly into output fields.")
        self.climatology_residual = bool(climatology_residual)
        if self.in_channels < 1 or self.latent_channels < 1:
            raise ValueError("AE input and latent channels must be positive.")
        self.register_buffer("latent_mean", torch.zeros(1, self.latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, self.latent_channels, 1, 1))
        self.register_buffer("latent_calibrated", torch.tensor(False))

        enc_hidden = tuple(int(v) for v in encoder_hidden_channels)
        dec_hidden = tuple(int(v) for v in decoder_hidden_channels)
        self.encoder_hidden_channels = enc_hidden
        self.decoder_hidden_channels = dec_hidden

        enc_layers: list[nn.Module] = []
        # Values, observation validity and physical domain have distinct meanings.
        prev = 3 * self.in_channels
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
        with resolve_config_path(path).open("r", encoding="utf-8") as f:
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
            output_fields=ae.get("output_fields", ["temperature"]),
            climatology_residual=bool(ae.get("climatology_residual", False)),
        )

    def contract(self) -> dict[str, Any]:
        """Describe checkpoint semantics independently of tensor shapes."""
        return {
            "version": 2,
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "spatial_downsample": self.spatial_downsample,
            "output_fields": list(self.output_fields),
            "mask_channels": "per_depth",
            "climatology_residual": self.climatology_residual,
            "encoder_hidden_channels": list(self.encoder_hidden_channels),
            "decoder_hidden_channels": list(self.decoder_hidden_channels),
            "normalization": [
                Y_MEAN,
                Y_STD,
                SALINITY_MEAN,
                SALINITY_STD,
                CELSIUS_TO_KELVIN_OFFSET,
            ],
        }

    @staticmethod
    def align_mask(mask: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        """Broadcast spatial masks without collapsing depth-specific validity."""
        if mask is None:
            return torch.ones_like(value, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if (
            mask.ndim != 4
            or mask.shape[0] != value.shape[0]
            or mask.shape[2:] != value.shape[2:]
            or mask.shape[1] not in (1, value.shape[1])
        ):
            raise ValueError(
                "Mask must match physical values or have one spatial channel."
            )
        return (mask.to(value.device) > 0.5).expand_as(value)

    def encode(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        domain = self.align_mask(wet_mask, value)
        valid = self.align_mask(valid_mask, value) & domain
        if not torch.isfinite(value[valid]).all():
            raise ValueError("Nonfinite values on valid AE input support.")
        # torch.where makes encoded features invariant to all missing-value fills.
        x = torch.cat(
            (
                torch.where(valid, value, 0.0),
                valid.to(value.dtype),
                domain.to(value.dtype),
            ),
            dim=1,
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
        return y

    def forward(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the autoencoder forward computation."""
        return self.decode(self.encode(value, valid_mask, wet_mask))


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
        masked_only: bool = True,
        dense_weight: float = 1.0,
        sparse_dense_weight: float = 1.0,
        observation_weight: float = 1.0,
        profile_keep_probability: float = 0.1,
        holdout_fraction: float = 0.2,
        lr_scheduler_enabled: bool = False,
        lr_scheduler_monitor: str = "val/loss_ckpt",
        lr_scheduler_interval: str = "epoch",
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
            lr_scheduler_interval (str): Scheduler cadence, "step" or "epoch".
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
        self.output_fields = autoencoder.output_fields
        self.case_weights = (
            float(dense_weight),
            float(sparse_dense_weight),
            float(observation_weight),
        )
        if any(weight < 0 for weight in self.case_weights) or not any(
            self.case_weights
        ):
            raise ValueError(
                "AE reconstruction weights must be nonnegative with a positive term."
            )
        self.profile_keep_probability = float(profile_keep_probability)
        self.holdout_fraction = float(holdout_fraction)
        if (
            not 0 < self.profile_keep_probability < 1
            or not 0 < self.holdout_fraction < 1
        ):
            raise ValueError(
                "AE corruption probabilities must lie between zero and one."
            )

        self.lr_scheduler_enabled = bool(lr_scheduler_enabled)
        self.lr_scheduler_monitor = str(lr_scheduler_monitor)
        self.lr_scheduler_interval = str(lr_scheduler_interval).strip().lower()
        if self.lr_scheduler_interval.endswith("s"):
            # Accept the human-readable config spellings "steps" and "epochs".
            self.lr_scheduler_interval = self.lr_scheduler_interval[:-1]
        if self.lr_scheduler_interval not in {"step", "epoch"}:
            raise ValueError('lr_scheduler_interval must be "step" or "epoch".')
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
        with resolve_config_path(path).open("r", encoding="utf-8") as f:
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
        plateau_interval = str(plateau_cfg.get("interval", "epoch"))

        return cls(
            autoencoder=DepthBandAutoencoder.from_config(ae_config_path),
            datamodule=datamodule,
            lr=float(ae_training.get("lr", t.get("lr", 1e-4))),
            batch_size=int(ae_training.get("batch_size", d.get("batch_size", 1))),
            recon_l1_weight=float(ae_loss.get("recon_l1_weight", 1.0)),
            recon_l2_weight=float(ae_loss.get("recon_l2_weight", 0.5)),
            masked_only=True,
            dense_weight=float(ae_loss.get("dense_weight", 1.0)),
            sparse_dense_weight=float(ae_loss.get("sparse_dense_weight", 1.0)),
            observation_weight=float(ae_loss.get("observation_weight", 1.0)),
            profile_keep_probability=float(
                ae_training.get("profile_keep_probability", 0.1)
            ),
            holdout_fraction=float(ae_training.get("holdout_fraction", 0.2)),
            lr_scheduler_enabled=bool(plateau_cfg.get("enabled", False)),
            lr_scheduler_monitor=str(plateau_cfg.get("monitor", "val/loss_ckpt")),
            lr_scheduler_interval=plateau_interval,
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

    def _recon_losses(
        self,
        target: torch.Tensor,
        recon: torch.Tensor,
        y_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self.model.align_mask(y_valid_mask, target)
        if not torch.isfinite(target[valid]).all():
            raise ValueError("Nonfinite reconstruction targets on valid support.")
        # Empty support contributes a differentiable zero, never an unmasked loss.
        difference = torch.where(valid, recon - torch.where(valid, target, 0.0), 0.0)
        denom = valid.sum().clamp_min(1)
        return difference.abs().sum() / denom, difference.square().sum() / denom

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store the field and normalization contract alongside AE weights."""
        checkpoint["ae_contract"] = self.model.contract()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject same-shaped checkpoints with incompatible physical semantics."""
        if checkpoint.get("ae_contract") != self.model.contract():
            raise ValueError(
                "Autoencoder checkpoint contract mismatch; retrain the mask-aware AE."
            )

    def physical_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Stack fields in checkpoint order and keep validity separate from domain."""

        def stack(role: str) -> torch.Tensor:
            """Resolve dataset keys without changing field order."""
            salinity_key = (
                role.replace("_valid_mask", "_salinity_valid_mask")
                if role.endswith("_valid_mask")
                else role + "_salinity"
            )
            keys = {"temperature": role, "salinity": salinity_key}
            return torch.cat(
                [batch[keys[field]] for field in self.output_fields], dim=1
            )

        values = {key: stack(key) for key in ("x", "y", "x_valid_mask", "y_valid_mask")}
        # Static domain is required by the supported preset; legacy toy batches may
        # use target validity, but never infer ocean support from observed profiles.
        values["wet_mask"] = (
            stack("wet_mask")
            if all(
                ("wet_mask" if field == "temperature" else "wet_mask_salinity") in batch
                for field in self.output_fields
            )
            else values["y_valid_mask"]
        )
        values["wet_mask"] = self.model.align_mask(values["wet_mask"], values["y"])
        for role in ("x", "y"):
            mask = self.model.align_mask(values[role + "_valid_mask"], values[role])
            if role == "y" and (mask & ~values["wet_mask"]).any():
                raise ValueError(
                    "Target-valid water lies outside the static wet domain."
                )
            values[role + "_valid_mask"] = mask & values["wet_mask"]
        if self.model.climatology_residual:
            background = stack("climatology")
            if not torch.isfinite(background).all():
                raise ValueError("Climatology must be finite.")
            values["background"] = background
            for role in ("x", "y"):
                values[role] = torch.where(
                    values[role + "_valid_mask"], values[role] - background, 0.0
                )
        return values

    def reconstruction_cases(
        self, batch: dict[str, Any]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Reconstruct dense, profile-corrupted dense, and real sparse fields."""
        data = self.physical_batch(batch)
        x, y = data["x"], data["y"]
        xm, ym, wet = data["x_valid_mask"], data["y_valid_mask"], data["wet_mask"]
        # Reuse real profile/depth coverage where available; add synthetic profiles
        # for empty patches so the dense corruption task remains defined.
        keep = torch.rand_like(y[:, :1]) < self.profile_keep_probability
        depth_count = y.shape[1] // len(self.output_fields)
        cutoff = torch.randint(1, depth_count + 1, keep.shape, device=y.device)
        depth = torch.arange(depth_count, device=y.device).repeat(
            len(self.output_fields)
        )[None, :, None, None]
        artificial = keep & (depth < cutoff) & ym
        sparse = torch.where(
            xm.flatten(1).any(1)[:, None, None, None], xm & ym, artificial
        )
        return {
            "dense": (self.model(y, ym, wet), y, ym),
            "sparse_dense": (self.model(y, sparse, wet), y, ym),
            "observed": (self.model(x, xm, wet), x, xm),
        }

    def _shared_step(self, batch: dict[str, Any], *, prefix: str) -> torch.Tensor:
        cases = self.reconstruction_cases(batch)
        loss_l1 = cases["dense"][0].new_zeros(())
        loss_l2 = loss_l1.clone()
        y = cases["dense"][1]
        for weight, (name, (recon, target, mask)) in zip(
            self.case_weights, cases.items()
        ):
            l1, l2 = self._recon_losses(target, recon, mask)
            loss_l1 = loss_l1 + weight * l1
            loss_l2 = loss_l2 + weight * l2
            self.log(
                f"{prefix}/{name}_l1",
                l1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=y.shape[0],
            )
        loss = self.recon_l1_weight * loss_l1 + self.recon_l2_weight * loss_l2

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
        data = self.physical_batch(batch)
        recon = self.model(data["y"], data["y_valid_mask"], data["wet_mask"])
        if "background" in data:
            recon = recon + data["background"]
        outputs = {}
        for field, value in zip(
            self.output_fields, recon.chunk(len(self.output_fields), dim=1)
        ):
            denorm = (
                temperature_normalize if field == "temperature" else salinity_normalize
            )
            outputs[f"y_hat_{field}"] = value
            outputs[f"y_hat_{field}_denorm"] = denorm(mode="denorm", tensor=value)
        outputs["y_hat"] = outputs[f"y_hat_{self.output_fields[0]}"]
        outputs["y_hat_denorm"] = outputs[f"y_hat_{self.output_fields[0]}_denorm"]
        return outputs

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
        scheduler_strict = not (
            self.lr_scheduler_interval == "step"
            and self.lr_scheduler_monitor.startswith("val/")
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_monitor,
                "interval": self.lr_scheduler_interval,
                "frequency": 1,
                # Step-based validation metrics are unavailable before the first
                # validation run; let Lightning skip those early scheduler checks.
                "strict": scheduler_strict,
            },
        }
