from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.difFF.PixelDiffusion import PixelDiffusionConditional

from .Autoencoder import DepthBandAutoencoder


class LatentDiffusionConditional(PixelDiffusionConditional):
    """Conditional diffusion module operating in autoencoder latent space."""

    def __init__(
        self,
        *,
        autoencoder: DepthBandAutoencoder,
        autoencoder_frozen: bool = True,
        eo_in_pixel_space: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize LatentDiffusionConditional with configured parameters.

        Args:
            autoencoder (DepthBandAutoencoder): Input value.
            autoencoder_frozen (bool): Boolean flag controlling behavior.
            eo_in_pixel_space (bool): Boolean flag controlling behavior.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None: No value is returned.
        """
        generated_channels = int(kwargs.get("generated_channels", 1))
        if generated_channels != int(autoencoder.latent_channels):
            raise ValueError(
                "generated_channels must match autoencoder latent size. "
                f"Got generated_channels={generated_channels}, "
                f"autoencoder.latent_channels={int(autoencoder.latent_channels)}."
            )

        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.autoencoder_frozen = bool(autoencoder_frozen)
        self.eo_in_pixel_space = bool(eo_in_pixel_space)

        if self.autoencoder_frozen:
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad = False
            self.autoencoder.eval()

    @staticmethod
    def _extract_autoencoder_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        state_dict: Any
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Autoencoder checkpoint must contain a state_dict mapping."
            )

        # Lightning checkpoints often namespace weights under "model.".
        # Keep best-effort prefix stripping for common wrappers.
        prefixes = ("model.", "autoencoder.", "ae.", "module.")
        for prefix in prefixes:
            if any(str(key).startswith(prefix) for key in state_dict.keys()):
                stripped = {
                    str(key)[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if str(key).startswith(prefix)
                }
                if stripped:
                    return stripped
        return {str(key): value for key, value in state_dict.items()}

    @classmethod
    def from_config(
        cls,
        model_config_path: str = "configs/lat_space/model_config.yaml",
        data_config_path: str = "configs/lat_space/data_config.yaml",
        training_config_path: str = "configs/lat_space/training_config.yaml",
        datamodule: pl.LightningDataModule | None = None,
    ) -> "LatentDiffusionConditional":
        """Build latent diffusion model from config files."""
        model_cfg = cls._load_yaml(model_config_path)
        data_cfg = cls._load_yaml(data_config_path)
        training_cfg = cls._load_yaml(training_config_path)

        m = model_cfg.get("model", {})
        t = training_cfg.get("training", model_cfg.get("training", {}))
        noise_cfg = t.get("noise", {})
        w = training_cfg.get("wandb", model_cfg.get("wandb", {}))
        d = training_cfg.get("dataloader", data_cfg.get("dataloader", {}))
        scheduler_cfg = training_cfg.get("scheduler", data_cfg.get("scheduler", {}))
        plateau_cfg = scheduler_cfg.get(
            "reduce_on_plateau",
            scheduler_cfg.get("reduce_lr_on_plateau", {}),
        )
        warmup_cfg = scheduler_cfg.get("warmup", {})
        val_sampling_cfg = t.get("validation_sampling", {})
        coord_cfg = m.get("coord_conditioning", {})
        ambient_cfg = m.get("ambient_occlusion", {})
        postprocess_cfg = m.get("post_process", m.get("post-process", {}))
        gaussian_blur_cfg = postprocess_cfg.get("gaussian_blur", {})
        latent_cfg = m.get("latent", {})

        ae_config_path = str(latent_cfg.get("ae_config_path", "")).strip()
        if not ae_config_path:
            raise ValueError(
                "latent.ae_config_path is required for model.model_type='latent_cond_dif'."
            )
        if not Path(ae_config_path).is_file():
            raise FileNotFoundError(f"AE config not found: {ae_config_path}")

        autoencoder = DepthBandAutoencoder.from_config(ae_config_path)

        ae_checkpoint = latent_cfg.get("ae_checkpoint", False)
        if ae_checkpoint not in (False, None):
            ae_checkpoint_path = Path(str(ae_checkpoint)).expanduser()
            if not ae_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"AE checkpoint not found: {ae_checkpoint_path}"
                )
            checkpoint = torch.load(ae_checkpoint_path, map_location="cpu")
            state_dict = cls._extract_autoencoder_state_dict(checkpoint)
            missing, unexpected = autoencoder.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                warnings.warn(
                    "Loaded AE checkpoint with non-strict matching: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}.",
                    stacklevel=2,
                )

        latent_channels = int(
            latent_cfg.get("latent_channels", autoencoder.latent_channels)
        )
        if latent_channels != int(autoencoder.latent_channels):
            raise ValueError(
                "latent.latent_channels must match AE config latent_channels. "
                f"Got {latent_channels} vs {int(autoencoder.latent_channels)}."
            )
        spatial_downsample = int(
            latent_cfg.get("spatial_downsample", autoencoder.spatial_downsample)
        )
        if spatial_downsample != int(autoencoder.spatial_downsample):
            raise ValueError(
                "latent.spatial_downsample must match AE config spatial_downsample. "
                f"Got {spatial_downsample} vs {int(autoencoder.spatial_downsample)}."
            )

        generated_channels = int(m.get("generated_channels", latent_channels))
        if generated_channels != latent_channels:
            raise ValueError(
                "model.generated_channels must equal latent latent_channels in latent mode. "
                f"Got {generated_channels} vs {latent_channels}."
            )

        unet_kwargs = cls._parse_unet_config(m)
        coord_embed_dim = coord_cfg.get("embed_dim", None)
        if coord_embed_dim is not None:
            coord_embed_dim = int(coord_embed_dim)

        return cls(
            datamodule=datamodule,
            autoencoder=autoencoder,
            autoencoder_frozen=bool(latent_cfg.get("freeze_autoencoder", True)),
            eo_in_pixel_space=bool(latent_cfg.get("eo_in_pixel_space", True)),
            generated_channels=generated_channels,
            condition_channels=int(m.get("condition_channels", latent_channels)),
            condition_mask_channels=int(m.get("condition_mask_channels", 1)),
            condition_include_eo=bool(m.get("condition_include_eo", True)),
            condition_use_valid_mask=bool(m.get("condition_use_valid_mask", True)),
            clamp_known_pixels=bool(m.get("clamp_known_pixels", False)),
            mask_loss_with_valid_pixels=bool(
                m.get("mask_loss_with_valid_pixels", True)
            ),
            parameterization=str(m.get("parameterization", "x0")),
            num_timesteps=int(
                noise_cfg.get("num_timesteps", m.get("num_timesteps", 1000))
            ),
            noise_schedule=str(
                noise_cfg.get("schedule", m.get("noise_schedule", "cosine"))
            ),
            noise_beta_start=float(
                noise_cfg.get("beta_start", m.get("noise_beta_start", 1e-4))
            ),
            noise_beta_end=float(
                noise_cfg.get("beta_end", m.get("noise_beta_end", 2e-2))
            ),
            **unet_kwargs,
            coord_conditioning_enabled=bool(coord_cfg.get("enabled", False)),
            coord_encoding=str(coord_cfg.get("encoding", "unit_sphere")),
            date_conditioning_enabled=bool(coord_cfg.get("include_date", False)),
            date_encoding=str(coord_cfg.get("date_encoding", "day_of_year_sincos")),
            coord_embed_dim=coord_embed_dim,
            batch_size=int(t.get("batch_size", d.get("batch_size", 1))),
            lr=float(t.get("lr", 1e-4)),
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
            lr_warmup_enabled=bool(warmup_cfg.get("enabled", False)),
            lr_warmup_steps=int(warmup_cfg.get("steps", 1000)),
            lr_warmup_start_ratio=float(warmup_cfg.get("start_ratio", 0.1)),
            val_inference_sampler=str(val_sampling_cfg.get("sampler", "ddpm")),
            val_ddim_num_timesteps=int(val_sampling_cfg.get("ddim_num_timesteps", 200)),
            val_ddim_eta=float(val_sampling_cfg.get("ddim_eta", 0.0)),
            log_intermediates=bool(
                val_sampling_cfg.get(
                    "log_intermediates", m.get("log_intermediates", True)
                )
            ),
            ambient_occlusion_enabled=bool(ambient_cfg.get("enabled", False)),
            ambient_further_drop_prob=float(ambient_cfg.get("further_drop_prob", 0.1)),
            ambient_apply_to_noisy_branch=bool(
                ambient_cfg.get("apply_to_noisy_branch", True)
            ),
            ambient_shared_spatial_mask=bool(
                ambient_cfg.get("shared_spatial_mask", True)
            ),
            ambient_min_kept_observed_pixels=int(
                ambient_cfg.get("min_kept_observed_pixels", 1)
            ),
            ambient_require_x0_parameterization=bool(
                ambient_cfg.get("require_x0_parameterization", True)
            ),
            skip_full_reconstruction_in_sanity_check=bool(
                val_sampling_cfg.get("skip_full_reconstruction_in_sanity_check", True)
            ),
            max_full_reconstruction_samples=int(
                val_sampling_cfg.get("max_full_reconstruction_samples", 4)
            ),
            postprocess_gaussian_blur_enabled=bool(
                gaussian_blur_cfg.get("enabled", False)
            ),
            postprocess_gaussian_blur_sigma=float(gaussian_blur_cfg.get("sigma", 0.5)),
            postprocess_gaussian_blur_kernel_size=int(
                gaussian_blur_cfg.get("kernel_size", 3)
            ),
            wandb_verbose=bool(w.get("verbose", True)),
            log_stats_every_n_steps=int(w.get("log_stats_every_n_steps", 100)),
            log_images_every_n_steps=int(w.get("log_images_every_n_steps", 10)),
        )

    def _maybe_encode_with_autoencoder(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4 and int(value.size(1)) == int(self.autoencoder.in_channels):
            if self.autoencoder_frozen:
                with torch.no_grad():
                    return self.autoencoder.encode(value)
            return self.autoencoder.encode(value)
        return value

    def _maybe_decode_with_autoencoder(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4 and int(value.size(1)) == int(
            self.autoencoder.latent_channels
        ):
            if self.autoencoder_frozen:
                with torch.no_grad():
                    return self.autoencoder.decode(value)
            return self.autoencoder.decode(value)
        return value

    def input_T(self, value: torch.Tensor) -> torch.Tensor:
        """Encode depth tensors into latent space when channel count matches AE input."""
        return self._maybe_encode_with_autoencoder(value)

    def output_T(self, value: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors back to depth-band space when channel count matches."""
        return self._maybe_decode_with_autoencoder(value)

    def _prepare_condition_for_model(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        eo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # EO can stay in pixel-space while x is encoded to latent channels.
        condition_parts: list[torch.Tensor] = []
        if self.condition_include_eo:
            if eo is None:
                raise RuntimeError("condition_include_eo=true requires batch['eo'].")
            eo_t = eo if self.eo_in_pixel_space else self.input_T(eo)
            condition_parts.append(eo_t)

        data_t = self.input_T(x)
        condition_parts.append(data_t)

        mask_t = self._prepare_condition_mask(
            valid_mask,
            batch_size=int(data_t.size(0)),
            height=int(data_t.size(-2)),
            width=int(data_t.size(-1)),
        )
        if mask_t is not None:
            mask_t = mask_t.to(device=data_t.device, dtype=data_t.dtype)
            condition_parts.append(mask_t)

        condition = torch.cat(condition_parts, dim=1)
        expected_channels = int(getattr(self.model, "condition_channels", 0))
        if expected_channels > 0 and int(condition.size(1)) != expected_channels:
            raise RuntimeError(
                "Conditioning channel mismatch: "
                f"built={int(condition.size(1))}, expected={expected_channels}. "
                "Check condition_channels / condition_mask_channels / "
                "condition_include_eo / condition_use_valid_mask."
            )
        return condition

    def _collapse_mask_channels(
        self,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        m = mask
        if m.ndim == 3:
            m = m.unsqueeze(1)
        if m.ndim != 4:
            return m
        if int(m.size(1)) in {1, int(self.model.generated_channels)}:
            return m
        return m.amax(dim=1, keepdim=True)

    def _downsample_mask_to_latent_grid(
        self,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        downsample = int(getattr(self.autoencoder, "spatial_downsample", 1))
        if downsample <= 1:
            return mask

        mask_was_3d = mask.ndim == 3
        m = mask.unsqueeze(1) if mask_was_3d else mask
        if m.ndim != 4:
            return mask

        # Latent-space masks should stay valid if any source pixel in the pooled block was valid.
        pooled = F.max_pool2d(
            m.to(dtype=torch.float32),
            kernel_size=downsample,
            stride=downsample,
        )
        if mask.dtype == torch.bool:
            pooled_out: torch.Tensor = pooled > 0.5
        else:
            pooled_out = (pooled > 0.5).to(dtype=mask.dtype)
        if mask_was_3d:
            pooled_out = pooled_out.squeeze(1)
        return pooled_out

    def _prepare_batch_for_latent_loss(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = dict(batch)
        prepared["x_valid_mask"] = self._downsample_mask_to_latent_grid(
            self._collapse_mask_channels(batch.get("x_valid_mask"))
        )
        prepared["y_valid_mask"] = self._downsample_mask_to_latent_grid(
            self._collapse_mask_channels(batch.get("y_valid_mask"))
        )
        prepared["x_valid_mask_1d"] = self._downsample_mask_to_latent_grid(
            self._collapse_mask_channels(batch.get("x_valid_mask_1d"))
        )
        prepared["land_mask"] = self._downsample_mask_to_latent_grid(
            self._collapse_mask_channels(batch.get("land_mask"))
        )
        return prepared

    def _build_ambient_further_valid_mask(
        self,
        valid_mask: torch.Tensor | None,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        latent_reference = self.input_T(reference)
        return super()._build_ambient_further_valid_mask(
            valid_mask,
            reference=latent_reference,
        )

    def _build_task_supervision_mask(
        self,
        *,
        reference: torch.Tensor,
        x_valid_mask: torch.Tensor | None,
        y_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        latent_reference = self.input_T(reference)
        return super()._build_task_supervision_mask(
            reference=latent_reference,
            x_valid_mask=x_valid_mask,
            y_valid_mask=y_valid_mask,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.autoencoder_frozen:
            self.autoencoder.eval()
        return super().training_step(
            self._prepare_batch_for_latent_loss(batch),
            batch_idx,
        )

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.autoencoder_frozen:
            self.autoencoder.eval()
        return super().validation_step(
            self._prepare_batch_for_latent_loss(batch),
            batch_idx,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Create optimizer and optional scheduler configuration."""
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        params.extend(filter(lambda p: p.requires_grad, self.autoencoder.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr)
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
