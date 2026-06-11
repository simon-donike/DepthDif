from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

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


class PointwiseLSTMBaseline(IDWInterpolationBaseline):
    """Point-wise vertical LSTM baseline for sparse field reconstruction."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
        include_eo: bool = True,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        depth_axis_m: Sequence[float] | torch.Tensor | None = None,
        output_fields: Sequence[str] | str | None = None,
        variable_scenario: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
        skip_full_reconstruction_in_sanity_check: bool = True,
        max_full_reconstruction_samples: int = 1,
    ) -> None:
        """Initialize the point-wise LSTM baseline.

        Args:
            hidden_size (int): Hidden size for each field-specific LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Inter-layer LSTM dropout.
            bidirectional (bool): Whether to use bidirectional LSTMs.
            include_eo (bool): Whether to append the per-pixel EO surface value.
            lr (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay.
            depth_axis_m (Sequence[float] | torch.Tensor | None): Optional depth axis.
            output_fields (Sequence[str] | str | None): Active output fields.
            variable_scenario (str | None): Scenario metadata stored in checkpoints.
            datamodule (pl.LightningDataModule | None): Optional Lightning datamodule.
            skip_full_reconstruction_in_sanity_check (bool): Skip expensive sanity logs.
            max_full_reconstruction_samples (int): Cached val samples for recon metrics.

        Returns:
            None: No value is returned.
        """
        pl.LightningModule.__init__(self)
        if int(hidden_size) < 1:
            raise ValueError("model.lstm.hidden_size must be >= 1.")
        if int(num_layers) < 1:
            raise ValueError("model.lstm.num_layers must be >= 1.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("model.lstm.dropout must be >= 0.0 and < 1.0.")
        if float(lr) <= 0.0:
            raise ValueError("model.lstm.lr must be > 0.")
        if float(weight_decay) < 0.0:
            raise ValueError("model.lstm.weight_decay must be >= 0.0.")

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.include_eo = bool(include_eo)
        self.condition_include_eo = self.include_eo
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.output_fields = self._normalize_output_fields(output_fields)
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

        input_size = 4 if self.include_eo else 3
        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0
        output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.field_lstms = torch.nn.ModuleDict()
        self.field_heads = torch.nn.ModuleDict()
        for field in self.output_fields:
            self.field_lstms[field] = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=self.bidirectional,
            )
            self.field_heads[field] = torch.nn.Linear(output_size, 1)

        resolved_depth_axis = self._resolve_depth_axis_m(
            depth_axis_m=depth_axis_m,
            datamodule=datamodule,
        )
        self.generated_channels = (
            len(self.output_fields) * int(resolved_depth_axis.numel())
            if int(resolved_depth_axis.numel()) > 0
            else len(self.output_fields)
        )
        self.save_hyperparameters(ignore=["datamodule", "depth_axis_m"])
        self.register_buffer("depth_axis_m", resolved_depth_axis, persistent=True)
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
        section = PointwiseLSTMBaseline._training_section(training_cfg)
        val_sampling = section.get("validation_sampling", {})
        return val_sampling if isinstance(val_sampling, dict) else {}

    @staticmethod
    def _depth_axis_from_dataset(dataset: Any) -> torch.Tensor | None:
        """Return depth_axis_m from a dataset or wrapped subset, when available."""
        visited: set[int] = set()
        current = dataset
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            depth_axis = getattr(current, "depth_axis_m", None)
            if depth_axis is not None:
                tensor = torch.as_tensor(depth_axis, dtype=torch.float32).reshape(-1)
                if int(tensor.numel()) > 0:
                    return tensor
            # torch.utils.data.Subset exposes the wrapped dataset as .dataset.
            current = getattr(current, "dataset", None)
        return None

    @classmethod
    def _resolve_depth_axis_m(
        cls,
        *,
        depth_axis_m: Sequence[float] | torch.Tensor | None,
        datamodule: pl.LightningDataModule | None,
    ) -> torch.Tensor:
        """Resolve the physical depth axis from config or datamodule metadata."""
        if depth_axis_m is not None:
            tensor = torch.as_tensor(depth_axis_m, dtype=torch.float32).reshape(-1)
            if int(tensor.numel()) == 0:
                raise ValueError("model.lstm.depth_axis_m must not be empty.")
            return tensor

        for name in ("dataset", "val_dataset", "train_dataset"):
            dataset = (
                getattr(datamodule, name, None) if datamodule is not None else None
            )
            tensor = cls._depth_axis_from_dataset(dataset)
            if tensor is not None:
                return tensor
        return torch.empty(0, dtype=torch.float32)

    @classmethod
    def from_config(
        cls,
        model_config_path: str | None = None,
        data_config_path: str | None = None,
        training_config_path: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
    ) -> "PointwiseLSTMBaseline":
        """Build the point-wise LSTM baseline from config files."""
        if model_config_path is None:
            raise ValueError("model_config_path is required for PointwiseLSTMBaseline.")
        model_cfg = cls._load_yaml(model_config_path)
        training_cfg = (
            cls._load_yaml(training_config_path) if training_config_path else {}
        )
        m = model_cfg.get("model", {})
        lstm_cfg = m.get("lstm", {})
        if lstm_cfg is None:
            lstm_cfg = {}
        if not isinstance(lstm_cfg, dict):
            raise ValueError("model.lstm must be a mapping when provided.")
        _ = data_config_path

        training_section = cls._training_section(training_cfg)
        val_sampling_cfg = cls._validation_sampling_section(training_cfg)
        lr = float(lstm_cfg.get("lr", training_section.get("lr", 1.0e-3)))

        return cls(
            hidden_size=int(lstm_cfg.get("hidden_size", 64)),
            num_layers=int(lstm_cfg.get("num_layers", 2)),
            dropout=float(lstm_cfg.get("dropout", 0.0)),
            bidirectional=bool(lstm_cfg.get("bidirectional", True)),
            include_eo=bool(
                lstm_cfg.get("include_eo", m.get("condition_include_eo", True))
            ),
            lr=lr,
            weight_decay=float(lstm_cfg.get("weight_decay", 0.0)),
            depth_axis_m=lstm_cfg.get("depth_axis_m", None),
            output_fields=m.get("output_fields", None),
            variable_scenario=m.get("scenario", None),
            datamodule=datamodule,
            skip_full_reconstruction_in_sanity_check=bool(
                val_sampling_cfg.get("skip_full_reconstruction_in_sanity_check", True)
            ),
            max_full_reconstruction_samples=int(
                val_sampling_cfg.get("max_full_reconstruction_samples", 1)
            ),
        )

    def _normalized_depth_positions(
        self, depth_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return normalized depth coordinates for the active channel count."""
        if int(depth_size) < 1:
            raise RuntimeError("Depth channel count must be >= 1.")
        depth_axis = self.depth_axis_m.to(device=device, dtype=dtype)
        if int(depth_axis.numel()) == 0:
            return torch.linspace(0.0, 1.0, int(depth_size), device=device, dtype=dtype)
        if int(depth_axis.numel()) != int(depth_size):
            raise RuntimeError(
                "Configured depth_axis_m length does not match input channels: "
                f"{int(depth_axis.numel())} != {int(depth_size)}."
            )
        finite = torch.isfinite(depth_axis)
        if not bool(finite.all().item()):
            raise RuntimeError("depth_axis_m must contain only finite values.")
        depth_min = depth_axis.min()
        depth_range = depth_axis.max() - depth_min
        if bool(depth_range <= 0):
            return torch.zeros(int(depth_size), device=device, dtype=dtype)
        return (depth_axis - depth_min) / depth_range

    def _prepare_eo_sequence(
        self,
        eo: torch.Tensor | None,
        *,
        batch_size: int,
        depth_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return EO as a repeated depth sequence feature."""
        if not self.include_eo:
            return None
        if eo is None:
            raise RuntimeError("PointwiseLSTMBaseline requires batch['eo'].")
        eo_tensor = eo.to(device=device, dtype=dtype)
        if eo_tensor.ndim == 3:
            eo_tensor = eo_tensor.unsqueeze(1)
        if eo_tensor.ndim != 4:
            raise RuntimeError("batch['eo'] must be shaped as (B,1,H,W) or (B,H,W).")
        if int(eo_tensor.size(0)) != int(batch_size):
            raise RuntimeError("batch['eo'] batch size does not match input x.")
        if int(eo_tensor.size(1)) != 1:
            raise RuntimeError("PointwiseLSTMBaseline expects exactly one EO channel.")
        if tuple(eo_tensor.shape[2:]) != (int(height), int(width)):
            raise RuntimeError("batch['eo'] spatial shape does not match input x.")
        return eo_tensor.expand(-1, int(depth_size), -1, -1)

    def _forward_field(
        self,
        field: str,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        eo: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict one physical field from independent per-pixel depth sequences."""
        if x.ndim != 4:
            raise RuntimeError(f"x must be shaped (B,C,H,W), got {tuple(x.shape)}.")
        batch_size, depth_size, height, width = map(int, x.shape)
        mask = self._align_mask_to_reference(
            valid_mask.to(device=x.device), x, mask_name=f"{field}_valid_mask"
        )
        observed = (mask > 0.5) & torch.isfinite(x)
        safe_x = torch.where(observed, x, torch.zeros_like(x))
        depth_positions = self._normalized_depth_positions(
            depth_size, device=x.device, dtype=x.dtype
        ).view(1, depth_size, 1, 1)
        features = [
            safe_x,
            observed.to(dtype=x.dtype),
            depth_positions.expand(batch_size, -1, height, width),
        ]
        eo_sequence = self._prepare_eo_sequence(
            eo,
            batch_size=batch_size,
            depth_size=depth_size,
            height=height,
            width=width,
            device=x.device,
            dtype=x.dtype,
        )
        if eo_sequence is not None:
            features.append(eo_sequence)

        sequence = torch.stack(features, dim=-1)
        sequence = sequence.permute(0, 2, 3, 1, 4).reshape(
            batch_size * height * width, depth_size, len(features)
        )
        lstm_out, _ = self.field_lstms[field](sequence)
        prediction = self.field_heads[field](lstm_out).squeeze(-1)
        prediction = prediction.reshape(batch_size, height, width, depth_size).permute(
            0, 3, 1, 2
        )

        return prediction

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        eo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict dense normalized fields while preserving patch tensor shape."""
        if x.ndim != 4:
            raise RuntimeError(f"x must be shaped (B,C,H,W), got {tuple(x.shape)}.")
        if int(x.size(1)) % len(self.output_fields) != 0:
            raise RuntimeError(
                "Input channels must be divisible by the number of output fields."
            )
        mask = self._align_mask_to_reference(
            valid_mask.to(device=x.device), x, mask_name="x_valid_mask"
        )
        channels_per_field = int(x.size(1)) // len(self.output_fields)
        predictions: list[torch.Tensor] = []
        for idx, field in enumerate(self.output_fields):
            start = idx * channels_per_field
            end = start + channels_per_field
            predictions.append(
                self._forward_field(field, x[:, start:end], mask[:, start:end], eo)
            )
        return torch.cat(predictions, dim=1) if len(predictions) > 1 else predictions[0]

    def _predict_normalized(self, batch: dict[str, Any]) -> torch.Tensor:
        """Predict normalized output fields from an unmodified dataloader batch."""
        model_batch = self._prepare_model_batch_tensors(batch, include_y=False)
        x_by_field = self._split_output_tensor(model_batch["x"], batch)
        mask_by_field = self._split_output_tensor(model_batch["x_valid_mask"], batch)
        predictions = [
            self._forward_field(
                field,
                x_by_field[field],
                mask_by_field[field],
                batch.get("eo"),
            )
            for field in self.output_fields
        ]
        return torch.cat(predictions, dim=1) if len(predictions) > 1 else predictions[0]

    def _shared_step(
        self,
        batch: dict[str, Any],
        *,
        prefix: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Run one train/validation step and log masked normalized MSE."""
        model_batch = self._prepare_model_batch_tensors(batch, include_y=True)
        prediction = self._predict_normalized(batch)
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
        """Log train loss for one LSTM baseline optimization step."""
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
        """Log one denormalized LSTM validation reconstruction image grid."""
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
                f"LSTM baseline validation image logging failed for {field}: {exc}",
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
                "LSTM baseline full validation reconstruction failed; logging "
                f"placeholder metrics instead. Error: {exc}",
                stacklevel=2,
            )
            self._log_full_reconstruction_placeholders()
        finally:
            self._cached_val_example = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the AdamW optimizer for trainable LSTM baseline weights."""
        return torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _apply_no_argo_nodata(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        model_batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Return predictions unchanged so EO-only inference remains available."""
        _ = batch, model_batch
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

    @torch.no_grad()
    def uncertainty_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
        num_samples: int = 20,
        sampler: torch.nn.Module | None = None,
        collapse_channels: bool = True,
    ) -> dict[str, Any]:
        """Return deterministic zero uncertainty for the LSTM baseline."""
        return super().uncertainty_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            num_samples=num_samples,
            sampler=sampler,
            collapse_channels=collapse_channels,
        )
