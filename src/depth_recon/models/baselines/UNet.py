from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from depth_recon.models.baselines.IDW import IDWInterpolationBaseline
from depth_recon.utils.normalizations import (
    PLOT_CMAP,
    PLOT_SALINITY_CMAP,
    salinity_normalize,
    temperature_normalize,
)
from depth_recon.utils.validation_denoise import (
    log_wandb_conditional_reconstruction_grid,
)


def _group_count(channels: int, requested_groups: int) -> int:
    """Return a valid GroupNorm group count for one channel width."""
    groups = min(max(1, int(requested_groups)), int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return groups


class _DoubleConvBlock(torch.nn.Module):
    """Two 3D convolution, normalization, and activation layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm_groups: int,
        dropout: float,
    ) -> None:
        """Initialize the 3D convolution block."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_channels = int(in_channels)
        for _ in range(2):
            layers.extend(
                [
                    torch.nn.Conv3d(
                        current_channels,
                        int(out_channels),
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.GroupNorm(
                        _group_count(int(out_channels), int(norm_groups)),
                        int(out_channels),
                    ),
                    torch.nn.ReLU(inplace=True),
                ]
            )
            current_channels = int(out_channels)
        if float(dropout) > 0.0:
            layers.append(torch.nn.Dropout3d(float(dropout)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block forward pass."""
        return self.net(x)


class _PlainUNet3D(torch.nn.Module):
    """Standard 3D U-Net for depth-aware patch-to-patch regression."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_mults: Sequence[int],
        norm_groups: int,
        dropout: float,
    ) -> None:
        """Initialize the 3D U-Net."""
        super().__init__()
        widths = [int(base_channels) * int(mult) for mult in channel_mults]
        if not widths:
            raise ValueError("model.unet_baseline.channel_mults must not be empty.")

        self.down_blocks = torch.nn.ModuleList()
        current_channels = int(in_channels)
        for width in widths:
            self.down_blocks.append(
                _DoubleConvBlock(
                    current_channels,
                    width,
                    norm_groups=int(norm_groups),
                    dropout=float(dropout),
                )
            )
            current_channels = width

        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.up_transpose = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()
        for skip_width in reversed(widths[:-1]):
            self.up_transpose.append(
                torch.nn.ConvTranspose3d(
                    current_channels,
                    skip_width,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.up_blocks.append(
                _DoubleConvBlock(
                    skip_width * 2,
                    skip_width,
                    norm_groups=int(norm_groups),
                    dropout=float(dropout),
                )
            )
            current_channels = skip_width

        self.output = torch.nn.Conv3d(
            current_channels, int(out_channels), kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run U-Net encoder, decoder, and final projection."""
        skips: list[torch.Tensor] = []
        h = x
        for idx, block in enumerate(self.down_blocks):
            h = block(h)
            skips.append(h)
            if idx != len(self.down_blocks) - 1:
                h = self.pool(h)

        for upsample, block, skip in zip(
            self.up_transpose,
            self.up_blocks,
            reversed(skips[:-1]),
        ):
            h = upsample(h)
            if tuple(h.shape[-3:]) != tuple(skip.shape[-3:]):
                # Odd depth or patch sizes can drift by one voxel through pooling;
                # align to the skip feature before concatenating.
                h = F.interpolate(
                    h,
                    size=tuple(skip.shape[-3:]),
                    mode="trilinear",
                    align_corners=False,
                )
            h = block(torch.cat([skip, h], dim=1))

        return self.output(h)


class UNetInfillingBaseline(IDWInterpolationBaseline):
    """Trainable 3D U-Net baseline for sparse patch infilling."""

    def __init__(
        self,
        *,
        generated_channels: int,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        norm_groups: int = 8,
        dropout: float = 0.0,
        condition_include_eo: bool = True,
        condition_use_valid_mask: bool = True,
        condition_use_land_mask: bool = True,
        condition_mask_channels: int | None = None,
        per_channel_valid_mask: bool = True,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        output_fields: Sequence[str] | str | None = None,
        variable_scenario: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
        skip_full_reconstruction_in_sanity_check: bool = True,
        max_full_reconstruction_samples: int = 1,
        model_summary_input_size: int = 128,
    ) -> None:
        """Initialize the U-Net infilling baseline."""
        pl.LightningModule.__init__(self)
        if int(generated_channels) < 1:
            raise ValueError("model.generated_channels must be >= 1.")
        if int(base_channels) < 1:
            raise ValueError("model.unet_baseline.base_channels must be >= 1.")
        if int(norm_groups) < 1:
            raise ValueError("model.unet_baseline.norm_groups must be >= 1.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("model.unet_baseline.dropout must be >= 0.0 and < 1.0.")
        if float(lr) <= 0.0:
            raise ValueError("model.unet_baseline.lr must be > 0.")
        if float(weight_decay) < 0.0:
            raise ValueError("model.unet_baseline.weight_decay must be >= 0.0.")

        self.generated_channels = int(generated_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = self._parse_channel_mults(channel_mults)
        self.norm_groups = int(norm_groups)
        self.dropout = float(dropout)
        self.condition_include_eo = bool(condition_include_eo)
        self.condition_use_valid_mask = bool(condition_use_valid_mask)
        self.condition_use_land_mask = bool(condition_use_land_mask)
        self.per_channel_valid_mask = bool(per_channel_valid_mask)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.output_fields = self._normalize_output_fields(output_fields)
        self.field_channels = len(self.output_fields)
        if self.generated_channels % self.field_channels != 0:
            raise ValueError(
                "model.generated_channels must be divisible by the active output "
                "field count."
            )
        self.depth_channels = self.generated_channels // self.field_channels
        if condition_mask_channels is None:
            default_mask_channels = (
                self.field_channels if self.per_channel_valid_mask else 1
            )
        else:
            default_mask_channels = max(0, int(condition_mask_channels))
        self.condition_mask_channels = default_mask_channels
        self.variable_scenario = variable_scenario
        self.datamodule = datamodule
        self.skip_full_reconstruction_in_sanity_check = bool(
            skip_full_reconstruction_in_sanity_check
        )
        self.max_full_reconstruction_samples = max(
            1, int(max_full_reconstruction_samples)
        )
        self.automatic_optimization = True
        self._cached_val_example: dict[str, Any] | None = None

        input_channels = self.field_channels
        if self.condition_include_eo:
            input_channels += 1
        if self.condition_use_valid_mask:
            input_channels += self.condition_mask_channels
        if self.condition_use_land_mask:
            input_channels += 1
        self.condition_channels = int(input_channels)

        self.net = _PlainUNet3D(
            in_channels=self.condition_channels,
            out_channels=self.field_channels,
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
            norm_groups=self.norm_groups,
            dropout=self.dropout,
        )
        self.example_input_array = torch.zeros(
            (
                1,
                self.condition_channels,
                self.depth_channels,
                max(1, int(model_summary_input_size)),
                max(1, int(model_summary_input_size)),
            ),
            dtype=torch.float32,
        )
        self.save_hyperparameters(ignore=["datamodule"])
        self.register_buffer("_empty", torch.empty(0), persistent=False)

    @staticmethod
    def _training_section(training_cfg: dict[str, Any]) -> dict[str, Any]:
        """Return the nested optimizer/training section from a config."""
        section = training_cfg.get("training", training_cfg)
        if not isinstance(section, dict):
            return {}
        nested = section.get("training")
        return nested if isinstance(nested, dict) else section

    @staticmethod
    def _validation_sampling_section(training_cfg: dict[str, Any]) -> dict[str, Any]:
        """Return validation sampling settings from a config."""
        section = UNetInfillingBaseline._training_section(training_cfg)
        val_sampling = section.get("validation_sampling", {})
        return val_sampling if isinstance(val_sampling, dict) else {}

    @staticmethod
    def _parse_channel_mults(value: Sequence[int]) -> tuple[int, ...]:
        """Return normalized positive U-Net channel multipliers."""
        mults = tuple(int(mult) for mult in value)
        if not mults:
            raise ValueError("model.unet_baseline.channel_mults must not be empty.")
        if any(mult < 1 for mult in mults):
            raise ValueError("model.unet_baseline.channel_mults values must be >= 1.")
        return mults

    @classmethod
    def from_config(
        cls,
        model_config_path: str | None = None,
        data_config_path: str | None = None,
        training_config_path: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
    ) -> "UNetInfillingBaseline":
        """Build the U-Net baseline from config files."""
        if model_config_path is None:
            raise ValueError("model_config_path is required for UNetInfillingBaseline.")
        model_cfg = cls._load_yaml(model_config_path)
        training_cfg = (
            cls._load_yaml(training_config_path) if training_config_path else {}
        )
        m = model_cfg.get("model", {})
        unet_cfg = m.get("unet_baseline", {})
        if unet_cfg is None:
            unet_cfg = {}
        if not isinstance(unet_cfg, dict):
            raise ValueError("model.unet_baseline must be a mapping when provided.")
        _ = data_config_path

        output_fields = cls._normalize_output_fields(m.get("output_fields", None))
        depth_channels = int(m.get("depth_channels", 50))
        generated_channels = int(
            m.get("generated_channels", depth_channels * len(output_fields))
        )
        training_section = cls._training_section(training_cfg)
        val_sampling_cfg = cls._validation_sampling_section(training_cfg)
        lr_cfg = unet_cfg.get("lr", None)
        lr = training_section.get("lr", 1.0e-3) if lr_cfg is None else lr_cfg

        return cls(
            generated_channels=generated_channels,
            base_channels=int(unet_cfg.get("base_channels", 32)),
            channel_mults=unet_cfg.get("channel_mults", [1, 2, 4, 8]),
            norm_groups=int(unet_cfg.get("norm_groups", 8)),
            dropout=float(unet_cfg.get("dropout", 0.0)),
            condition_include_eo=bool(m.get("condition_include_eo", True)),
            condition_use_valid_mask=bool(m.get("condition_use_valid_mask", True)),
            condition_use_land_mask=bool(m.get("condition_use_land_mask", True)),
            condition_mask_channels=(
                int(m["condition_mask_channels"])
                if "condition_mask_channels" in m
                else None
            ),
            per_channel_valid_mask=bool(unet_cfg.get("per_channel_valid_mask", True)),
            lr=float(lr),
            weight_decay=float(unet_cfg.get("weight_decay", 1.0e-4)),
            output_fields=output_fields,
            variable_scenario=m.get("scenario", None),
            datamodule=datamodule,
            skip_full_reconstruction_in_sanity_check=bool(
                val_sampling_cfg.get("skip_full_reconstruction_in_sanity_check", True)
            ),
            max_full_reconstruction_samples=int(
                val_sampling_cfg.get("max_full_reconstruction_samples", 1)
            ),
            model_summary_input_size=int(unet_cfg.get("model_summary_input_size", 128)),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Predict dense normalized output fields from a 3D condition volume."""
        if condition.ndim != 5:
            raise RuntimeError(
                "condition must be shaped (B,C,D,H,W), "
                f"got {tuple(condition.shape)}."
            )
        if int(condition.size(1)) != self.condition_channels:
            raise RuntimeError(
                "U-Net condition channel mismatch: "
                f"got {int(condition.size(1))}, expected {self.condition_channels}."
            )
        if int(condition.size(2)) != self.depth_channels:
            raise RuntimeError(
                "U-Net condition depth mismatch: "
                f"got {int(condition.size(2))}, expected {self.depth_channels}."
            )
        return self._volume_to_flat(self.net(condition))

    def _flat_to_volume(
        self, tensor: torch.Tensor, *, tensor_name: str
    ) -> torch.Tensor:
        """Reshape flattened field/depth channels into a 3D volume."""
        if tensor.ndim != 4:
            raise RuntimeError(
                f"{tensor_name} must be shaped (B,C,H,W), got {tuple(tensor.shape)}."
            )
        if int(tensor.size(1)) != self.generated_channels:
            raise RuntimeError(
                f"{tensor_name} channel mismatch: got {int(tensor.size(1))}, "
                f"expected {self.generated_channels}."
            )
        batch_size, _channels, height, width = tensor.shape
        return tensor.reshape(
            int(batch_size),
            self.field_channels,
            self.depth_channels,
            int(height),
            int(width),
        )

    def _volume_to_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flatten field/depth volume channels back to the output contract."""
        if tensor.ndim != 5:
            raise RuntimeError(
                f"U-Net output must be shaped (B,F,D,H,W), got {tuple(tensor.shape)}."
            )
        if int(tensor.size(1)) != self.field_channels:
            raise RuntimeError(
                "U-Net output field-channel mismatch: "
                f"got {int(tensor.size(1))}, expected {self.field_channels}."
            )
        if int(tensor.size(2)) != self.depth_channels:
            raise RuntimeError(
                "U-Net output depth mismatch: "
                f"got {int(tensor.size(2))}, expected {self.depth_channels}."
            )
        batch_size, _fields, _depth, height, width = tensor.shape
        return tensor.reshape(
            int(batch_size), self.generated_channels, int(height), int(width)
        )

    def _prepare_surface_condition(
        self,
        tensor: torch.Tensor | None,
        reference: torch.Tensor,
        *,
        tensor_name: str,
    ) -> torch.Tensor:
        """Return one surface condition channel repeated over depth."""
        if tensor is None:
            raise RuntimeError(
                f"{tensor_name}=true requires the matching batch tensor."
            )
        surface = tensor.to(device=reference.device, dtype=reference.dtype)
        if surface.ndim == 3:
            surface = surface.unsqueeze(1)
        if surface.ndim != 4:
            raise RuntimeError(f"{tensor_name} must be shaped as (B,1,H,W) or (B,H,W).")
        if int(surface.size(0)) != int(reference.size(0)):
            raise RuntimeError(f"{tensor_name} batch size does not match x.")
        if tuple(surface.shape[2:]) != tuple(reference.shape[2:]):
            raise RuntimeError(f"{tensor_name} spatial shape does not match x.")
        if int(surface.size(1)) != 1:
            surface = surface.amax(dim=1, keepdim=True)
        # EO and land mask are surface rasters, so repeat them along depth while
        # preserving depth as the Conv3d axis.
        return surface.unsqueeze(2).expand(-1, -1, self.depth_channels, -1, -1)

    def _prepare_eo_condition(
        self, eo: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Return EO condition aligned to the 3D sparse input volume."""
        if not self.condition_include_eo:
            return None
        return self._prepare_surface_condition(
            eo,
            reference,
            tensor_name="condition_include_eo",
        )

    def _prepare_valid_mask_condition(
        self, valid_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Return ARGO support mask volumes for the condition tensor."""
        if not self.condition_use_valid_mask or self.condition_mask_channels <= 0:
            return None
        if valid_mask is None:
            raise RuntimeError(
                "condition_use_valid_mask=true requires batch['x_valid_mask']."
            )
        mask = self._align_mask_to_reference(
            valid_mask.to(device=reference.device),
            reference,
            mask_name="x_valid_mask",
        ).to(dtype=reference.dtype)
        mask_volume = self._flat_to_volume(mask, tensor_name="x_valid_mask")
        expected_channels = int(self.condition_mask_channels)
        if int(mask_volume.size(1)) == expected_channels:
            return mask_volume
        if expected_channels == 1:
            return mask_volume.amax(dim=1, keepdim=True)
        if int(mask_volume.size(1)) == 1 and expected_channels > 1:
            return mask_volume.expand(-1, expected_channels, -1, -1, -1)
        raise RuntimeError(
            "Could not match x_valid_mask field channels to condition_mask_channels "
            f"(mask={int(mask_volume.size(1))}, expected={expected_channels})."
        )

    def _prepare_land_condition(
        self, land_mask: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Return GLORYS spatial-support condition volume when enabled."""
        if not self.condition_use_land_mask:
            return None
        return self._prepare_surface_condition(
            land_mask,
            reference,
            tensor_name="condition_use_land_mask",
        )

    def _build_condition(
        self,
        batch: dict[str, Any],
        *,
        include_y: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build the U-Net condition volume and model-facing batch tensors."""
        model_batch = self._prepare_model_batch_tensors(batch, include_y=include_y)
        x = model_batch["x"]
        valid_mask = model_batch["x_valid_mask"]
        mask = self._align_mask_to_reference(
            valid_mask.to(device=x.device), x, mask_name="x_valid_mask"
        )
        # Invalid sparse inputs are semantic missing values; keep them at zero and
        # let the explicit mask channel distinguish missingness from observations.
        sparse_x = torch.where(
            (mask > 0.5) & torch.isfinite(x),
            x,
            torch.zeros_like(x),
        )
        sparse_volume = self._flat_to_volume(sparse_x, tensor_name="x")

        parts: list[torch.Tensor] = []
        eo_condition = self._prepare_eo_condition(batch.get("eo"), sparse_x)
        if eo_condition is not None:
            parts.append(eo_condition)
        parts.append(sparse_volume)
        mask_condition = self._prepare_valid_mask_condition(valid_mask, sparse_x)
        if mask_condition is not None:
            parts.append(mask_condition)
        land_condition = self._prepare_land_condition(batch.get("land_mask"), sparse_x)
        if land_condition is not None:
            parts.append(land_condition)

        condition = torch.cat(parts, dim=1)
        if int(condition.size(1)) != self.condition_channels:
            raise RuntimeError(
                "UNet condition channel mismatch: "
                f"built={int(condition.size(1))}, expected={self.condition_channels}."
            )
        return condition, model_batch

    def _apply_no_argo_to_prediction(
        self,
        prediction: torch.Tensor,
        batch: dict[str, Any],
        model_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Set sample/field predictions without ARGO support to NaN."""
        mask_by_field = self._split_output_tensor(model_batch["x_valid_mask"], batch)
        prediction_by_field = self._split_output_tensor(prediction, batch)
        cleaned: list[torch.Tensor] = []
        for field in self.output_fields:
            field_prediction = prediction_by_field[field]
            has_argo = (mask_by_field[field] > 0.5).flatten(1).any(dim=1)
            keep_shape = [int(has_argo.size(0))] + [1] * (
                int(field_prediction.ndim) - 1
            )
            keep = has_argo.to(device=field_prediction.device).reshape(keep_shape)
            cleaned.append(
                torch.where(
                    keep,
                    field_prediction,
                    torch.full_like(field_prediction, float("nan")),
                )
            )
        return torch.cat(cleaned, dim=1) if len(cleaned) > 1 else cleaned[0]

    def _predict_normalized(self, batch: dict[str, Any]) -> torch.Tensor:
        """Predict normalized output fields from an unmodified dataloader batch."""
        condition, model_batch = self._build_condition(batch, include_y=False)
        prediction = self.forward(condition)
        return self._apply_no_argo_to_prediction(prediction, batch, model_batch)

    def _shared_step(
        self,
        batch: dict[str, Any],
        *,
        prefix: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Run one train/validation step and log masked normalized MSE."""
        condition, model_batch = self._build_condition(batch, include_y=True)
        prediction = self.forward(condition)
        prediction = self._apply_no_argo_to_prediction(prediction, batch, model_batch)
        mask = self._supervision_mask(
            model_batch["y"],
            model_batch["y_valid_mask"],
            batch.get("land_mask"),
        )
        loss = self._masked_mse(prediction, model_batch["y"], mask)
        trainable_zero = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for parameter in self.parameters():
            if parameter.requires_grad:
                trainable_zero = trainable_zero + parameter.sum() * 0.0
        loss = loss + trainable_zero
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=prefix == "val",
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if prefix == "val":
            self.log(
                "val/loss_ckpt",
                torch.nan_to_num(loss.detach(), nan=1.0e9, posinf=1.0e9, neginf=1.0e9),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Log train loss for one U-Net baseline optimization step."""
        _ = batch_idx
        return self._shared_step(
            batch,
            prefix="train",
            batch_size=int(
                self._prepare_model_batch_tensors(batch, include_y=False)["x"].size(0)
            ),
        )

    def _cache_validation_batch(self, batch: dict[str, Any], *, n_cache: int) -> None:
        """Cache a small validation batch for epoch-end reconstruction metrics."""
        cache_keys = (
            "x",
            "y",
            "x_valid_mask",
            "y_valid_mask",
            "x_salinity",
            "y_salinity",
            "x_salinity_valid_mask",
            "y_salinity_valid_mask",
            "eo",
            "land_mask",
            "output_land_mask",
            "coords",
            "date",
        )
        cached: dict[str, Any] = {}
        for key in cache_keys:
            if key not in batch:
                continue
            value = batch[key]
            if torch.is_tensor(value):
                cached[key] = value[:n_cache].detach().clone()
            elif isinstance(value, (list, tuple)):
                cached[key] = value[:n_cache]
            else:
                cached[key] = value
        self._cached_val_example = cached

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Log validation loss and cache one batch for full-reconstruction metrics."""
        loss = self._shared_step(
            batch,
            prefix="val",
            batch_size=int(
                self._prepare_model_batch_tensors(batch, include_y=False)["x"].size(0)
            ),
        )
        if batch_idx == 0 and self._cached_val_example is None:
            model_batch = self._prepare_model_batch_tensors(batch, include_y=True)
            n_cache = min(
                self.max_full_reconstruction_samples, int(model_batch["y"].size(0))
            )
            self._cache_validation_batch(batch, n_cache=n_cache)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset cached validation data at the start of each validation epoch."""
        self._cached_val_example = None

    def _log_full_reconstruction_metrics(
        self,
        *,
        recon_mse: torch.Tensor,
        recon_l1: torch.Tensor,
        recon_psnr: torch.Tensor,
        recon_ssim: torch.Tensor,
        batch_size: int,
        metric_prefix: str,
    ) -> None:
        """Log full-reconstruction validation metrics under one prefix."""
        for name, value, show in (
            ("recon_mse_full_recon", recon_mse, True),
            ("recon_l1_full_recon", recon_l1, True),
            ("recon_psnr_full_recon", recon_psnr, True),
            ("recon_ssim_full_recon", recon_ssim, False),
        ):
            self.log(
                f"{metric_prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=show and metric_prefix == "val",
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _log_full_reconstruction_placeholders(self) -> None:
        """Log zero placeholders so validation metric keys remain stable."""
        placeholder = torch.zeros((), device=self.device, dtype=torch.float32)
        self._log_full_reconstruction_metrics(
            recon_mse=placeholder,
            recon_l1=placeholder,
            recon_psnr=placeholder,
            recon_ssim=placeholder,
            batch_size=1,
            metric_prefix="val",
        )
        if "salinity" in self.output_fields and len(self.output_fields) > 1:
            self._log_full_reconstruction_metrics(
                recon_mse=placeholder,
                recon_l1=placeholder,
                recon_psnr=placeholder,
                recon_ssim=placeholder,
                batch_size=1,
                metric_prefix="val_salinity",
            )

    def _compute_full_reconstruction_metrics(
        self,
        *,
        prediction_denorm: torch.Tensor,
        target_denorm: torch.Tensor,
        eval_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute full-reconstruction MSE, L1, PSNR, and SSIM scalars."""
        metric_zero = torch.zeros(
            (), device=target_denorm.device, dtype=target_denorm.dtype
        )
        if eval_mask is None:
            eval_support = torch.isfinite(prediction_denorm) & torch.isfinite(
                target_denorm
            )
        else:
            eval_support = (
                (eval_mask > 0.5)
                & torch.isfinite(prediction_denorm)
                & torch.isfinite(target_denorm)
            )

        if bool(eval_support.any().item()):
            diff = prediction_denorm - target_denorm
            recon_mse = diff.pow(2)[eval_support].mean()
            recon_l1 = torch.abs(diff)[eval_support].mean()
        else:
            recon_mse = metric_zero
            recon_l1 = metric_zero

        recon_psnr = metric_zero
        recon_ssim = metric_zero
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            target_np = target_denorm.detach().float().cpu().numpy()
            pred_np = prediction_denorm.detach().float().cpu().numpy()
            mask_np = (
                (eval_mask > 0.5).detach().cpu().numpy()
                if eval_mask is not None
                else None
            )
            if target_np.ndim == 2:
                target_np = target_np[None, None, ...]
                pred_np = pred_np[None, None, ...]
                if mask_np is not None:
                    mask_np = mask_np[None, None, ...]
            elif target_np.ndim == 3:
                target_np = target_np[:, None, ...]
                pred_np = pred_np[:, None, ...]
                if mask_np is not None:
                    mask_np = mask_np[:, None, ...]
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            for sample_idx in range(target_np.shape[0]):
                for band_idx in range(target_np.shape[1]):
                    target_band = target_np[sample_idx, band_idx]
                    pred_band = pred_np[sample_idx, band_idx]
                    if mask_np is not None:
                        band_mask = mask_np[sample_idx, band_idx] > 0.5
                    else:
                        band_mask = np.isfinite(target_band) & np.isfinite(pred_band)
                    if not np.any(band_mask):
                        continue
                    target_valid = target_band[band_mask]
                    pred_valid = pred_band[band_mask]
                    data_range = float(target_valid.max() - target_valid.min())
                    if data_range <= 0.0:
                        continue
                    psnr_values.append(
                        float(
                            peak_signal_noise_ratio(
                                target_valid,
                                pred_valid,
                                data_range=data_range,
                            )
                        )
                    )
                    if bool(np.all(band_mask)):
                        ssim_values.append(
                            float(
                                structural_similarity(
                                    target_band,
                                    pred_band,
                                    data_range=data_range,
                                )
                            )
                        )
            if psnr_values:
                recon_psnr = torch.tensor(
                    float(sum(psnr_values) / len(psnr_values)),
                    device=target_denorm.device,
                    dtype=target_denorm.dtype,
                )
            if ssim_values:
                recon_ssim = torch.tensor(
                    float(sum(ssim_values) / len(ssim_values)),
                    device=target_denorm.device,
                    dtype=target_denorm.dtype,
                )
        except Exception:
            pass
        return recon_mse, recon_l1, recon_psnr, recon_ssim

    @staticmethod
    def _denormalize_field(field: str, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize one physical field tensor."""
        if field == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

    def _move_cached_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move cached tensor values to the current model device."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _trainer_or_none(self) -> pl.Trainer | None:
        """Return the attached Trainer, or None when called outside Trainer scope."""
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _apply_invalid_to_nan(
        self, tensor: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return tensor with values outside an optional mask set to NaN."""
        if mask is None:
            return tensor
        aligned_mask = self._align_mask_to_reference(
            (mask > 0.5).to(dtype=tensor.dtype, device=tensor.device),
            tensor,
            mask_name="validation_plot_mask",
        )
        return torch.where(
            aligned_mask > 0.5,
            tensor,
            torch.full_like(tensor, float("nan")),
        )

    def _log_full_reconstruction_image(
        self,
        *,
        field: str,
        cached: dict[str, Any],
        pred: dict[str, Any],
        target_denorm: torch.Tensor,
        eval_mask: torch.Tensor | None,
    ) -> None:
        """Log one denormalized U-Net validation reconstruction image grid."""
        input_key = self._field_batch_key(field, role="x")
        target_key = self._field_batch_key(field, role="y")
        input_valid_key = "x_valid_mask"
        target_valid_key = "y_valid_mask"
        if field == "salinity":
            input_valid_key = "x_salinity_valid_mask"
            target_valid_key = "y_salinity_valid_mask"
        input_tensor = self._require_batch_tensor(cached, input_key)
        target_tensor = self._require_batch_tensor(cached, target_key)
        input_valid_mask = cached.get(input_valid_key)
        target_valid_mask = cached.get(target_valid_key)
        input_denorm = self._denormalize_field(field, input_tensor)
        target_full_denorm = self._denormalize_field(field, target_tensor)
        y_denorm_masked = self._apply_invalid_to_nan(
            target_full_denorm,
            self._supervision_mask(
                target_full_denorm,
                target_valid_mask if torch.is_tensor(target_valid_mask) else None,
                cached.get("land_mask"),
            ),
        )
        target_denorm_masked = self._apply_invalid_to_nan(target_denorm, eval_mask)
        y_hat_denorm = pred[f"y_hat_{field}_denorm"]
        y_hat_denorm_for_plot = pred.get(f"y_hat_{field}_denorm_for_plot", y_hat_denorm)
        is_salinity = field == "salinity"

        eo_denorm = None
        if torch.is_tensor(cached.get("eo")):
            eo_denorm = (
                salinity_normalize(mode="denorm", tensor=cached["eo"])
                if is_salinity
                else temperature_normalize(mode="denorm", tensor=cached["eo"])
            )

        prefix = "val_salinity_imgs" if is_salinity else "val_imgs"
        image_key = (
            "salinity_full_reconstruction" if is_salinity else "x_y_full_reconstruction"
        )
        try:
            log_wandb_conditional_reconstruction_grid(
                logger=self.logger,
                x=input_denorm,
                y=y_denorm_masked,
                eo=eo_denorm,
                y_hat=y_hat_denorm_for_plot,
                y_target=target_denorm_masked,
                valid_mask=(
                    input_valid_mask if torch.is_tensor(input_valid_mask) else None
                ),
                land_mask=(
                    cached.get("land_mask")
                    if torch.is_tensor(cached.get("land_mask"))
                    else None
                ),
                prefix=prefix,
                image_key=image_key,
                cmap=PLOT_SALINITY_CMAP if is_salinity else PLOT_CMAP,
                show_valid_mask_panel=False,
                plot_unit="salinity" if is_salinity else "temperature",
                error_metric_prefix=(
                    "val_salinity_absolute_band_error"
                    if is_salinity
                    else "val_absolute_band_error"
                ),
                error_metric_unit="psu" if is_salinity else "deg",
                error_metric_label="L1 (PSU)" if is_salinity else "L1 (deg)",
                error_metric_title=(
                    "Generated-Pixel Salinity L1 by Band"
                    if is_salinity
                    else "Generated-Pixel L1 by Band"
                ),
            )
        except Exception as exc:
            warnings.warn(
                f"U-Net baseline validation image logging failed for {field}: {exc}",
                stacklevel=2,
            )

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        """Log denormalized full-reconstruction metrics for the cached val batch."""
        trainer = self._trainer_or_none()
        if (
            trainer is not None
            and trainer.sanity_checking
            and self.skip_full_reconstruction_in_sanity_check
        ):
            self._log_full_reconstruction_placeholders()
            self._cached_val_example = None
            return
        if self._cached_val_example is None:
            self._log_full_reconstruction_placeholders()
            return

        try:
            cached = self._move_cached_batch_to_device(self._cached_val_example)
            pred = self.predict_step(cached, batch_idx=0)
            batch_size = int(pred["y_hat"].size(0))
            for field in self.output_fields:
                target_key = self._field_batch_key(field, role="y")
                valid_key = self._field_batch_key(field, role="y_valid_mask")
                target = self._require_batch_tensor(cached, target_key)
                valid_mask = cached.get(valid_key)
                target_denorm = self._denormalize_field(field, target)
                eval_mask = self._supervision_mask(
                    target_denorm,
                    valid_mask if torch.is_tensor(valid_mask) else None,
                    cached.get("land_mask"),
                )
                metrics = self._compute_full_reconstruction_metrics(
                    prediction_denorm=pred[f"y_hat_{field}_denorm"],
                    target_denorm=target_denorm,
                    eval_mask=eval_mask,
                )
                if field == "salinity" and len(self.output_fields) > 1:
                    metric_prefix = "val_salinity"
                elif field == "temperature" or len(self.output_fields) == 1:
                    metric_prefix = "val"
                else:
                    metric_prefix = f"val_{field}"
                self._log_full_reconstruction_metrics(
                    recon_mse=metrics[0],
                    recon_l1=metrics[1],
                    recon_psnr=metrics[2],
                    recon_ssim=metrics[3],
                    batch_size=batch_size,
                    metric_prefix=metric_prefix,
                )
                self._log_full_reconstruction_image(
                    field=field,
                    cached=cached,
                    pred=pred,
                    target_denorm=target_denorm,
                    eval_mask=eval_mask,
                )
        except Exception as exc:
            warnings.warn(
                "U-Net baseline full validation reconstruction failed; logging "
                f"placeholder metrics instead. Error: {exc}",
                stacklevel=2,
            )
            self._log_full_reconstruction_placeholders()
        finally:
            self._cached_val_example = None

    def _apply_no_argo_nodata(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        model_batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Keep all prediction outputs as nodata for samples without ARGO support."""
        mask_by_field = self._split_output_tensor(model_batch["x_valid_mask"], batch)
        for field in self.output_fields:
            has_argo = (mask_by_field[field] > 0.5).flatten(1).any(dim=1)
            for key in (
                f"y_hat_{field}",
                f"y_hat_{field}_denorm",
                f"y_hat_{field}_denorm_for_plot",
            ):
                tensor = outputs.get(key)
                if not torch.is_tensor(tensor):
                    continue
                keep_shape = [int(has_argo.size(0))] + [1] * (int(tensor.ndim) - 1)
                keep = has_argo.to(device=tensor.device).reshape(keep_shape)
                outputs[key] = torch.where(
                    keep, tensor, torch.full_like(tensor, float("nan"))
                )

        alias_field = (
            "temperature"
            if "temperature" in self.output_fields
            else self.output_fields[0]
        )
        outputs["y_hat_denorm"] = outputs[f"y_hat_{alias_field}_denorm"]
        outputs["y_hat_denorm_for_plot"] = outputs[
            f"y_hat_{alias_field}_denorm_for_plot"
        ]
        return outputs

    @torch.no_grad()
    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Predict one batch and return diffusion-compatible prediction keys."""
        _ = batch_idx, dataloader_idx
        model_batch = self._prepare_model_batch_tensors(batch, include_y=False)
        y_hat = self._predict_normalized(batch)
        outputs = self._build_prediction_outputs(
            y_hat,
            batch,
            y_valid_mask=model_batch["y_valid_mask"],
            land_mask=batch.get("land_mask"),
            output_land_mask=batch.get("output_land_mask"),
        )
        outputs = self._apply_no_argo_nodata(outputs, batch, model_batch)
        outputs.update(
            {
                "denoise_samples": [],
                "x0_denoise_samples": [],
                "sampler": None,
                "further_valid_mask": None,
            }
        )
        return outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the AdamW optimizer for trainable U-Net baseline weights."""
        return torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
