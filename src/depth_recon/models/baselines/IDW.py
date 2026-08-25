from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from depth_recon.utils.normalizations import (
    PLOT_CMAP,
    PLOT_SALINITY_CMAP,
    salinity_normalize,
    temperature_normalize,
)
from depth_recon.utils.validation_denoise import (
    log_wandb_conditional_reconstruction_grid,
)


class IDWInterpolationBaseline(pl.LightningModule):
    """Inverse-distance weighting baseline for sparse field reconstruction."""

    def __init__(
        self,
        *,
        power: float = 2.0,
        eps: float = 1.0e-6,
        chunk_size: int = 4096,
        output_fields: Sequence[str] | str | None = None,
        variable_scenario: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
        skip_full_reconstruction_in_sanity_check: bool = True,
        max_full_reconstruction_samples: int = 1,
    ) -> None:
        """Initialize the IDW baseline.

        Args:
            power (float): Inverse-distance exponent.
            eps (float): Minimum distance used to avoid division by zero.
            chunk_size (int): Number of target pixels processed per IDW chunk.
            output_fields (Sequence[str] | str | None): Active output fields.
            variable_scenario (str | None): Scenario metadata stored in checkpoints.
            datamodule (pl.LightningDataModule | None): Optional Lightning datamodule.
            skip_full_reconstruction_in_sanity_check (bool): Skip costly sanity metrics.
            max_full_reconstruction_samples (int): Cached validation sample count.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        if float(power) <= 0.0:
            raise ValueError("IDW power must be > 0.")
        if float(eps) <= 0.0:
            raise ValueError("IDW eps must be > 0.")
        if int(chunk_size) < 1:
            raise ValueError("IDW chunk_size must be >= 1.")

        self.power = float(power)
        self.eps = float(eps)
        self.chunk_size = int(chunk_size)
        self.output_fields = self._normalize_output_fields(output_fields)
        self.variable_scenario = variable_scenario
        self.datamodule = datamodule
        self.skip_full_reconstruction_in_sanity_check = bool(
            skip_full_reconstruction_in_sanity_check
        )
        self.max_full_reconstruction_samples = max(
            1, int(max_full_reconstruction_samples)
        )
        self._cached_val_example: dict[str, Any] | None = None
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["datamodule"])
        self.register_buffer("_empty", torch.empty(0), persistent=False)

    @staticmethod
    def _normalize_output_fields(
        output_fields: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        """Return normalized output field names."""
        if output_fields is None:
            fields = ["temperature"]
        elif isinstance(output_fields, str):
            fields = [output_fields]
        else:
            fields = list(output_fields)
        normalized = tuple(str(field).strip().lower() for field in fields if field)
        if not normalized:
            raise ValueError("IDW baseline requires at least one output field.")
        unsupported = sorted(
            field
            for field in set(normalized)
            if field not in {"temperature", "salinity"}
        )
        if unsupported:
            raise ValueError(
                "Unsupported IDW output field(s): " + ", ".join(unsupported)
            )
        return normalized

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        """Load one YAML file."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def from_config(
        cls,
        model_config_path: str | None = None,
        data_config_path: str | None = None,
        training_config_path: str | None = None,
        datamodule: pl.LightningDataModule | None = None,
    ) -> "IDWInterpolationBaseline":
        """Build the IDW baseline from model/data/training config files."""
        if model_config_path is None:
            raise ValueError(
                "model_config_path is required for IDWInterpolationBaseline."
            )
        model_cfg = cls._load_yaml(model_config_path)
        m = model_cfg.get("model", {})
        idw_cfg = m.get("idw", {})
        if idw_cfg is None:
            idw_cfg = {}
        if not isinstance(idw_cfg, dict):
            raise ValueError("model.idw must be a mapping when provided.")
        training_cfg = (
            cls._load_yaml(training_config_path) if training_config_path else {}
        )
        training_section = training_cfg.get("training", training_cfg)
        validation_cfg = training_section.get("validation_sampling", {})
        return cls(
            power=float(idw_cfg.get("power", 2.0)),
            eps=float(idw_cfg.get("eps", 1.0e-6)),
            chunk_size=int(idw_cfg.get("chunk_size", 4096)),
            output_fields=m.get("output_fields", None),
            variable_scenario=m.get("scenario", None),
            datamodule=datamodule,
            skip_full_reconstruction_in_sanity_check=bool(
                validation_cfg.get("skip_full_reconstruction_in_sanity_check", True)
            ),
            max_full_reconstruction_samples=int(
                validation_cfg.get("max_full_reconstruction_samples", 1)
            ),
        )

    def _require_batch_tensor(self, batch: dict[str, Any], key: str) -> torch.Tensor:
        """Return a required tensor from a batch dictionary."""
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"batch['{key}'] must be a tensor.")
        return value

    def _validate_stack_shape(
        self,
        reference: torch.Tensor,
        tensor: torch.Tensor,
        *,
        reference_key: str,
        key: str,
    ) -> None:
        """Validate that two tensors can be concatenated by channel."""
        if int(tensor.ndim) != int(reference.ndim):
            raise RuntimeError(
                f"batch[{key}] ndim ({int(tensor.ndim)}) does not match "
                f"batch[{reference_key}] ndim ({int(reference.ndim)})."
            )
        if int(tensor.size(0)) != int(reference.size(0)):
            raise RuntimeError(
                f"batch[{key}] batch size does not match batch[{reference_key}]."
            )
        if tuple(tensor.shape[2:]) != tuple(reference.shape[2:]):
            raise RuntimeError(
                f"batch[{key}] spatial shape does not match batch[{reference_key}]."
            )

    def _field_batch_key(self, field: str, *, role: str) -> str:
        """Return the dataset batch key for an output field."""
        if field == "temperature":
            return role
        if field == "salinity":
            return f"{role}_salinity"
        raise RuntimeError(f"Unsupported output field: {field}.")

    def _stack_output_tensor(
        self,
        batch: dict[str, Any],
        *,
        temperature_key: str,
        salinity_key: str,
    ) -> torch.Tensor:
        """Stack configured output fields along the channel dimension."""
        key_by_field = {"temperature": temperature_key, "salinity": salinity_key}
        tensors: list[torch.Tensor] = []
        reference: torch.Tensor | None = None
        reference_key: str | None = None
        for field in self.output_fields:
            key = key_by_field[field]
            tensor = self._require_batch_tensor(batch, key)
            if reference is None:
                reference = tensor
                reference_key = key
            else:
                self._validate_stack_shape(
                    reference,
                    tensor,
                    reference_key=str(reference_key),
                    key=key,
                )
            tensors.append(tensor)
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=1)

    def _prepare_model_batch_tensors(
        self, batch: dict[str, Any], *, include_y: bool
    ) -> dict[str, torch.Tensor]:
        """Build model-facing tensors from explicit dataset batch fields."""
        model_batch = {
            "x": self._stack_output_tensor(
                batch, temperature_key="x", salinity_key="x_salinity"
            ),
            "x_valid_mask": self._stack_output_tensor(
                batch,
                temperature_key="x_valid_mask",
                salinity_key="x_salinity_valid_mask",
            ),
            "y_valid_mask": self._stack_output_tensor(
                batch,
                temperature_key="y_valid_mask",
                salinity_key="y_salinity_valid_mask",
            ),
        }
        if include_y:
            model_batch["y"] = self._stack_output_tensor(
                batch, temperature_key="y", salinity_key="y_salinity"
            )
        return model_batch

    def _split_output_tensor(
        self, tensor: torch.Tensor, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Split a stacked output tensor back into configured physical fields."""
        channels_by_field: dict[str, int] = {}
        expected_channels = 0
        for field in self.output_fields:
            key = self._field_batch_key(field, role="x")
            field_tensor = self._require_batch_tensor(batch, key)
            channels_by_field[field] = int(field_tensor.size(1))
            expected_channels += channels_by_field[field]

        if int(tensor.size(1)) != expected_channels:
            raise RuntimeError(
                "Output channel mismatch: "
                f"got {int(tensor.size(1))}, expected {expected_channels}."
            )

        split: dict[str, torch.Tensor] = {}
        start = 0
        for field in self.output_fields:
            end = start + channels_by_field[field]
            split[field] = tensor[:, start:end]
            start = end
        return split

    def _align_mask_to_reference(
        self, mask: torch.Tensor, reference: torch.Tensor, *, mask_name: str
    ) -> torch.Tensor:
        """Align a validity mask to a reference tensor shape."""
        aligned = mask
        if aligned.ndim == 3:
            aligned = aligned.unsqueeze(1)
        if aligned.ndim != 4:
            raise RuntimeError(f"{mask_name} must be shaped as (B,C,H,W) or (B,H,W).")
        if int(aligned.size(0)) != int(reference.size(0)):
            raise RuntimeError(f"{mask_name} batch size does not match reference.")
        if tuple(aligned.shape[2:]) != tuple(reference.shape[2:]):
            raise RuntimeError(f"{mask_name} spatial shape does not match reference.")
        if int(aligned.size(1)) == int(reference.size(1)):
            return aligned
        if int(aligned.size(1)) == 1 and int(reference.size(1)) > 1:
            return aligned.expand(-1, int(reference.size(1)), -1, -1)
        if int(reference.size(1)) == 1 and int(aligned.size(1)) > 1:
            return aligned.amax(dim=1, keepdim=True)
        raise RuntimeError(
            f"{mask_name} channels ({int(aligned.size(1))}) do not match "
            f"reference channels ({int(reference.size(1))})."
        )

    def _interpolate_band(
        self, values: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate one 2D band with inverse-distance weighting."""
        valid = (valid_mask > 0.5) & torch.isfinite(values)
        if not bool(valid.any().item()):
            return torch.full_like(values, float("nan"))

        height, width = int(values.size(0)), int(values.size(1))
        rows, cols = torch.nonzero(valid, as_tuple=True)
        obs_coords = torch.stack([rows, cols], dim=1).to(
            dtype=values.dtype, device=values.device
        )
        obs_values = values[valid].reshape(-1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=values.device),
            torch.arange(width, device=values.device),
            indexing="ij",
        )
        target_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1).to(
            dtype=values.dtype,
            device=values.device,
        )

        output = torch.empty(height * width, dtype=values.dtype, device=values.device)
        eps_squared = self.eps * self.eps
        for start in range(0, height * width, self.chunk_size):
            stop = min(start + self.chunk_size, height * width)
            delta = target_coords[start:stop, None, :] - obs_coords[None, :, :]
            distance_squared = torch.clamp(delta.pow(2).sum(dim=-1), min=eps_squared)
            # IDW uses Euclidean distance; using squared distances avoids an
            # unnecessary sqrt while preserving the configured distance exponent.
            weights = distance_squared.pow(-0.5 * self.power)
            output[start:stop] = (weights * obs_values[None, :]).sum(dim=1) / (
                weights.sum(dim=1)
            )

        output = output.reshape(height, width)
        # Preserve exact observed values instead of relying on finite epsilon weights.
        output = torch.where(valid, values, output)
        return output

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Interpolate sparse normalized inputs into dense normalized outputs."""
        if x.ndim != 4:
            raise RuntimeError(f"x must be shaped (B,C,H,W), got {tuple(x.shape)}.")
        mask = self._align_mask_to_reference(
            valid_mask.to(device=x.device), x, mask_name="x_valid_mask"
        )
        outputs = torch.empty_like(x)
        for batch_idx in range(int(x.size(0))):
            for channel_idx in range(int(x.size(1))):
                outputs[batch_idx, channel_idx] = self._interpolate_band(
                    x[batch_idx, channel_idx],
                    mask[batch_idx, channel_idx],
                )
        return outputs

    def _supervision_mask(
        self,
        reference: torch.Tensor,
        y_valid_mask: torch.Tensor | None,
        land_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return the evaluation mask intersected with land/ocean support."""
        mask: torch.Tensor | None = None
        if y_valid_mask is not None:
            mask = self._align_mask_to_reference(
                (y_valid_mask > 0.5).to(dtype=reference.dtype, device=reference.device),
                reference,
                mask_name="y_valid_mask",
            )
        if land_mask is not None:
            land = self._align_mask_to_reference(
                (land_mask > 0.5).to(dtype=reference.dtype, device=reference.device),
                reference,
                mask_name="land_mask",
            )
            mask = land if mask is None else mask * land
        return mask

    def _masked_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute MSE over finite values and optional mask support."""
        support = torch.isfinite(prediction) & torch.isfinite(target)
        if mask is not None:
            support = support & (mask > 0.5)
        if not bool(support.any().item()):
            return torch.zeros((), dtype=target.dtype, device=target.device)
        return (prediction - target).pow(2)[support].mean()

    def _shared_step(
        self,
        batch: dict[str, Any],
        *,
        prefix: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Run one train/validation step and log masked MSE."""
        model_batch = self._prepare_model_batch_tensors(batch, include_y=True)
        prediction = self.forward(model_batch["x"], model_batch["x_valid_mask"])
        mask = self._supervision_mask(
            model_batch["y"],
            model_batch["y_valid_mask"],
            batch.get("land_mask"),
        )
        loss = self._masked_mse(prediction, model_batch["y"], mask)
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
        """Log IDW train loss without optimizer updates."""
        _ = batch_idx
        return self._shared_step(
            batch,
            prefix="train",
            batch_size=int(
                self._prepare_model_batch_tensors(batch, include_y=False)["x"].size(0)
            ),
        )

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Log IDW validation loss and cache one reconstruction batch."""
        loss = self._shared_step(
            batch,
            prefix="val",
            batch_size=int(
                self._prepare_model_batch_tensors(batch, include_y=False)["x"].size(0)
            ),
        )
        if batch_idx == 0 and self._cached_val_example is None:
            batch_size = int(
                self._prepare_model_batch_tensors(batch, include_y=True)["y"].size(0)
            )
            self._cache_validation_batch(
                batch,
                n_cache=min(self.max_full_reconstruction_samples, batch_size),
            )
        return loss

    def _cache_validation_batch(self, batch: dict[str, Any], *, n_cache: int) -> None:
        """Cache tensors needed for epoch-end reconstruction metrics and figures."""
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

    def on_validation_epoch_start(self) -> None:
        """Reset the cached validation batch before each validation pass."""
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
        """Log the shared full-reconstruction validation metric keys."""
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
        """Log stable zero-valued metric keys when reconstruction is unavailable."""
        placeholder = torch.zeros((), device=self.device, dtype=torch.float32)
        prefixes = (
            (
                "val"
                if len(self.output_fields) == 1
                else ("val_salinity" if field == "salinity" else f"val_{field}")
            )
            for field in self.output_fields
        )
        for metric_prefix in prefixes:
            self._log_full_reconstruction_metrics(
                recon_mse=placeholder,
                recon_l1=placeholder,
                recon_psnr=placeholder,
                recon_ssim=placeholder,
                batch_size=1,
                metric_prefix=metric_prefix,
            )

    @staticmethod
    def _compute_full_reconstruction_metrics(
        *,
        prediction_denorm: torch.Tensor,
        target_denorm: torch.Tensor,
        eval_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute physical-unit MSE, L1, PSNR, and SSIM scalars."""
        zero = torch.zeros((), device=target_denorm.device, dtype=target_denorm.dtype)
        support = torch.isfinite(prediction_denorm) & torch.isfinite(target_denorm)
        if eval_mask is not None:
            support = support & (eval_mask > 0.5)
        if bool(support.any().item()):
            difference = prediction_denorm - target_denorm
            recon_mse = difference.square()[support].mean()
            recon_l1 = difference.abs()[support].mean()
        else:
            recon_mse = zero
            recon_l1 = zero

        recon_psnr = zero
        recon_ssim = zero
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            target_np = target_denorm.detach().float().cpu().numpy()
            prediction_np = prediction_denorm.detach().float().cpu().numpy()
            mask_np = (
                (eval_mask > 0.5).detach().cpu().numpy()
                if eval_mask is not None
                else None
            )
            if target_np.ndim == 2:
                target_np = target_np[None, None, ...]
                prediction_np = prediction_np[None, None, ...]
                mask_np = None if mask_np is None else mask_np[None, None, ...]
            elif target_np.ndim == 3:
                target_np = target_np[:, None, ...]
                prediction_np = prediction_np[:, None, ...]
                mask_np = None if mask_np is None else mask_np[:, None, ...]
            psnr_values: list[float] = []
            ssim_values: list[float] = []
            for sample_index in range(target_np.shape[0]):
                for band_index in range(target_np.shape[1]):
                    target_band = target_np[sample_index, band_index]
                    prediction_band = prediction_np[sample_index, band_index]
                    band_mask = (
                        mask_np[sample_index, band_index] > 0.5
                        if mask_np is not None
                        else np.isfinite(target_band) & np.isfinite(prediction_band)
                    )
                    band_mask &= np.isfinite(target_band) & np.isfinite(prediction_band)
                    if not bool(np.any(band_mask)):
                        continue
                    target_valid = target_band[band_mask]
                    prediction_valid = prediction_band[band_mask]
                    data_range = float(target_valid.max() - target_valid.min())
                    if data_range <= 0.0:
                        continue
                    psnr_values.append(
                        float(
                            peak_signal_noise_ratio(
                                target_valid,
                                prediction_valid,
                                data_range=data_range,
                            )
                        )
                    )
                    if bool(np.all(band_mask)):
                        ssim_values.append(
                            float(
                                structural_similarity(
                                    target_band,
                                    prediction_band,
                                    data_range=data_range,
                                )
                            )
                        )
            if psnr_values:
                recon_psnr = target_denorm.new_tensor(float(np.mean(psnr_values)))
            if ssim_values:
                recon_ssim = target_denorm.new_tensor(float(np.mean(ssim_values)))
        except Exception:
            # MSE/L1 remain available when optional image metrics cannot be computed.
            pass
        return recon_mse, recon_l1, recon_psnr, recon_ssim

    @staticmethod
    def _denormalize_field(field: str, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a model field into its physical units."""
        if field == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

    def _move_cached_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move cached tensors onto the active model device."""
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _trainer_or_none(self) -> pl.Trainer | None:
        """Return the attached trainer when the module is inside a Trainer run."""
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _apply_invalid_to_nan(
        self, tensor: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Mask invalid values with NaN for W&B reconstruction figures."""
        if mask is None:
            return tensor
        aligned = self._align_mask_to_reference(
            (mask > 0.5).to(dtype=tensor.dtype, device=tensor.device),
            tensor,
            mask_name="validation_plot_mask",
        )
        return torch.where(aligned > 0.5, tensor, torch.full_like(tensor, float("nan")))

    def _log_full_reconstruction_image(
        self,
        *,
        field: str,
        cached: dict[str, Any],
        prediction: dict[str, Any],
        target_denorm: torch.Tensor,
        eval_mask: torch.Tensor | None,
    ) -> None:
        """Log one denormalized IDW reconstruction image grid to W&B."""
        is_salinity = field == "salinity"
        input_key = "x_salinity" if is_salinity else "x"
        target_key = "y_salinity" if is_salinity else "y"
        input_valid_key = "x_salinity_valid_mask" if is_salinity else "x_valid_mask"
        target_valid_key = "y_salinity_valid_mask" if is_salinity else "y_valid_mask"
        input_tensor = self._require_batch_tensor(cached, input_key)
        target_tensor = self._require_batch_tensor(cached, target_key)
        input_valid_mask = cached.get(input_valid_key)
        target_valid_mask = cached.get(target_valid_key)
        input_denorm = self._denormalize_field(field, input_tensor)
        target_full_denorm = self._denormalize_field(field, target_tensor)
        target_full_denorm = self._apply_invalid_to_nan(
            target_full_denorm,
            self._supervision_mask(
                target_full_denorm,
                target_valid_mask if torch.is_tensor(target_valid_mask) else None,
                cached.get("land_mask"),
            ),
        )
        eo_denorm = None
        if torch.is_tensor(cached.get("eo")):
            eo_denorm = self._denormalize_field(field, cached["eo"])
        try:
            log_wandb_conditional_reconstruction_grid(
                logger=self.logger,
                x=input_denorm,
                y=target_full_denorm,
                eo=eo_denorm,
                y_hat=prediction.get(
                    f"y_hat_{field}_denorm_for_plot",
                    prediction[f"y_hat_{field}_denorm"],
                ),
                y_target=self._apply_invalid_to_nan(target_denorm, eval_mask),
                valid_mask=(
                    input_valid_mask if torch.is_tensor(input_valid_mask) else None
                ),
                land_mask=(
                    cached.get("land_mask")
                    if torch.is_tensor(cached.get("land_mask"))
                    else None
                ),
                prefix="val_salinity_imgs" if is_salinity else "val_imgs",
                image_key=(
                    "salinity_full_reconstruction"
                    if is_salinity
                    else "x_y_full_reconstruction"
                ),
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
                f"IDW validation image logging failed for {field}: {exc}",
                stacklevel=2,
            )

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        """Log IDW full-reconstruction metrics and images for the cached batch."""
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
            prediction = self.predict_step(cached, batch_idx=0)
            batch_size = int(prediction["y_hat"].size(0))
            for field in self.output_fields:
                is_salinity = field == "salinity"
                target_key = "y_salinity" if is_salinity else "y"
                valid_key = "y_salinity_valid_mask" if is_salinity else "y_valid_mask"
                target = self._require_batch_tensor(cached, target_key)
                valid_mask = cached.get(valid_key)
                target_denorm = self._denormalize_field(field, target)
                eval_mask = self._supervision_mask(
                    target_denorm,
                    valid_mask if torch.is_tensor(valid_mask) else None,
                    cached.get("land_mask"),
                )
                metrics = self._compute_full_reconstruction_metrics(
                    prediction_denorm=prediction[f"y_hat_{field}_denorm"],
                    target_denorm=target_denorm,
                    eval_mask=eval_mask,
                )
                metric_prefix = (
                    "val"
                    if len(self.output_fields) == 1
                    else ("val_salinity" if is_salinity else f"val_{field}")
                )
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
                    prediction=prediction,
                    target_denorm=target_denorm,
                    eval_mask=eval_mask,
                )
        except Exception as exc:
            warnings.warn(
                "IDW full validation reconstruction failed; logging placeholder "
                f"metrics instead. Error: {exc}",
                stacklevel=2,
            )
            self._log_full_reconstruction_placeholders()
        finally:
            self._cached_val_example = None

    def configure_optimizers(self) -> list[Any]:
        """Return no optimizers because IDW has no trainable weights."""
        return []

    def _postprocess_prediction_field(
        self,
        normalized: torch.Tensor,
        valid_mask: torch.Tensor | None,
        land_mask: torch.Tensor | None,
        output_land_mask: torch.Tensor | None,
        *,
        normalize_fn: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalize and apply standard output masking."""
        denorm = normalize_fn(mode="denorm", tensor=normalized)
        denorm_for_plot = denorm
        if valid_mask is not None:
            keep_mask = self._align_mask_to_reference(
                (valid_mask > 0.5).to(dtype=denorm.dtype, device=denorm.device),
                denorm,
                mask_name="y_valid_mask",
            )
            denorm = torch.where(
                keep_mask > 0.5, denorm, torch.full_like(denorm, float("nan"))
            )
            denorm_for_plot = torch.where(
                keep_mask > 0.5,
                denorm_for_plot,
                torch.full_like(denorm_for_plot, float("nan")),
            )
        for mask_name, raw_mask in (
            ("land_mask", land_mask),
            ("output_land_mask", output_land_mask),
        ):
            if raw_mask is None:
                continue
            ocean_mask = self._align_mask_to_reference(
                (raw_mask > 0.5).to(dtype=denorm.dtype, device=denorm.device),
                denorm,
                mask_name=mask_name,
            )
            denorm = torch.where(ocean_mask > 0.5, denorm, torch.zeros_like(denorm))
            denorm_for_plot = torch.where(
                ocean_mask > 0.5,
                denorm_for_plot,
                torch.zeros_like(denorm_for_plot),
            )
        return denorm, denorm_for_plot

    def _build_prediction_outputs(
        self,
        y_hat: torch.Tensor,
        batch: dict[str, Any],
        *,
        y_valid_mask: torch.Tensor | None,
        land_mask: torch.Tensor | None,
        output_land_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Build predict_step outputs in normalized and physical units."""
        normalize_by_field = {
            "temperature": temperature_normalize,
            "salinity": salinity_normalize,
        }
        if len(self.output_fields) == 1:
            field = self.output_fields[0]
            y_hat_denorm, y_hat_denorm_for_plot = self._postprocess_prediction_field(
                y_hat,
                y_valid_mask,
                land_mask,
                output_land_mask,
                normalize_fn=normalize_by_field[field],
            )
            return {
                "y_hat": y_hat,
                f"y_hat_{field}": y_hat,
                f"y_hat_{field}_denorm": y_hat_denorm,
                f"y_hat_{field}_denorm_for_plot": y_hat_denorm_for_plot,
                "y_hat_denorm": y_hat_denorm,
                "y_hat_denorm_for_plot": y_hat_denorm_for_plot,
            }

        y_hat_by_field = self._split_output_tensor(y_hat, batch)
        mask_by_field = (
            self._split_output_tensor(y_valid_mask, batch)
            if y_valid_mask is not None
            else {field: None for field in self.output_fields}
        )
        outputs: dict[str, torch.Tensor] = {"y_hat": y_hat}
        denorm_by_field: dict[str, torch.Tensor] = {}
        denorm_for_plot_by_field: dict[str, torch.Tensor] = {}
        for field in self.output_fields:
            field_denorm, field_denorm_for_plot = self._postprocess_prediction_field(
                y_hat_by_field[field],
                mask_by_field[field],
                land_mask,
                output_land_mask,
                normalize_fn=normalize_by_field[field],
            )
            outputs[f"y_hat_{field}"] = y_hat_by_field[field]
            outputs[f"y_hat_{field}_denorm"] = field_denorm
            outputs[f"y_hat_{field}_denorm_for_plot"] = field_denorm_for_plot
            denorm_by_field[field] = field_denorm
            denorm_for_plot_by_field[field] = field_denorm_for_plot

        alias_field = (
            "temperature"
            if "temperature" in self.output_fields
            else self.output_fields[0]
        )
        outputs["y_hat_denorm"] = denorm_by_field[alias_field]
        outputs["y_hat_denorm_for_plot"] = denorm_for_plot_by_field[alias_field]
        return outputs

    def _apply_no_argo_nodata(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        model_batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Set sample/field predictions without ARGO support to NaN."""
        mask_by_field = self._split_output_tensor(model_batch["x_valid_mask"], batch)

        def mask_field_tensor(tensor: torch.Tensor, field: str) -> torch.Tensor:
            has_argo = (mask_by_field[field] > 0.5).flatten(1).any(dim=1)
            keep_shape = [int(has_argo.size(0))] + [1] * (int(tensor.ndim) - 1)
            keep = has_argo.to(device=tensor.device).reshape(keep_shape)
            return torch.where(keep, tensor, torch.full_like(tensor, float("nan")))

        def mask_stacked_tensor(tensor: torch.Tensor) -> torch.Tensor:
            tensor_by_field = self._split_output_tensor(tensor, batch)
            cleaned = [
                mask_field_tensor(tensor_by_field[field], field)
                for field in self.output_fields
            ]
            return torch.cat(cleaned, dim=1) if len(cleaned) > 1 else cleaned[0]

        masked = dict(outputs)
        if torch.is_tensor(masked.get("y_hat")):
            masked["y_hat"] = mask_stacked_tensor(masked["y_hat"])
        for field in self.output_fields:
            for suffix in ("", "_denorm", "_denorm_for_plot"):
                key = f"y_hat_{field}{suffix}"
                if torch.is_tensor(masked.get(key)):
                    masked[key] = mask_field_tensor(masked[key], field)
        alias_field = (
            "temperature"
            if "temperature" in self.output_fields
            else self.output_fields[0]
        )
        for key in ("y_hat_denorm", "y_hat_denorm_for_plot"):
            if torch.is_tensor(masked.get(key)):
                masked[key] = mask_field_tensor(masked[key], alias_field)
        return masked

    @torch.no_grad()
    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Interpolate one batch and return diffusion-compatible prediction keys."""
        _ = batch_idx, dataloader_idx
        model_batch = self._prepare_model_batch_tensors(batch, include_y=False)
        y_hat = self.forward(model_batch["x"], model_batch["x_valid_mask"])
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

    def _normalize_uncertainty_raster(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty rasters to 0-1 while preserving invalid pixels."""
        normalized = torch.empty_like(tensor)
        for batch_idx in range(int(tensor.size(0))):
            for channel_idx in range(int(tensor.size(1))):
                raster = tensor[batch_idx, channel_idx]
                finite_mask = torch.isfinite(raster)
                normalized_raster = torch.full_like(raster, float("nan"))
                if finite_mask.any():
                    finite_values = raster[finite_mask]
                    value_range = finite_values.max() - finite_values.min()
                    if bool(value_range > 0):
                        normalized_raster[finite_mask] = (
                            finite_values - finite_values.min()
                        ) / value_range
                    else:
                        normalized_raster[finite_mask] = torch.zeros_like(finite_values)
                normalized[batch_idx, channel_idx] = normalized_raster
        return normalized

    def _collapse_uncertainty_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Collapse per-channel uncertainty into one spatial raster."""
        return torch.nanmean(tensor, dim=1, keepdim=True)

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
        """Return deterministic zero uncertainty for the IDW baseline."""
        if int(num_samples) < 2:
            raise ValueError("num_samples must be at least 2 for uncertainty_step.")
        _ = sampler
        pred = self.predict_step(
            batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        outputs: dict[str, Any] = {
            "uncertainty_num_samples": int(num_samples),
            "uncertainty_stat": "deterministic_zero",
            "sampler": None,
            "further_valid_mask": None,
        }
        uncertainty_by_field: dict[str, torch.Tensor] = {}
        for field in self.output_fields:
            field_uncertainty = torch.zeros_like(pred[f"y_hat_{field}_denorm"])
            if collapse_channels:
                field_uncertainty = self._collapse_uncertainty_channels(
                    field_uncertainty
                )
            uncertainty_by_field[field] = field_uncertainty
            outputs[f"uncertainty_{field}"] = field_uncertainty
            outputs[f"uncertainty_{field}_normalized"] = (
                self._normalize_uncertainty_raster(field_uncertainty)
            )

        alias_field = (
            "temperature"
            if "temperature" in self.output_fields
            else self.output_fields[0]
        )
        outputs["uncertainty"] = uncertainty_by_field[alias_field]
        outputs["uncertainty_normalized"] = outputs[
            f"uncertainty_{alias_field}_normalized"
        ]
        return outputs
