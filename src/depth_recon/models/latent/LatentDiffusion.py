from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from depth_recon.models.diffusion.PixelDiffusion import PixelDiffusionConditional
from depth_recon.paths import config_path, resolve_config_path

from .Autoencoder import DepthBandAutoencoder


class LatentDiffusionConditional(PixelDiffusionConditional):
    """Conditional diffusion module operating in autoencoder latent space."""

    def __init__(
        self,
        *,
        autoencoder: DepthBandAutoencoder,
        autoencoder_frozen: bool = True,
        eo_in_pixel_space: bool = True,
        decoded_observation_weight: float = 0.0,
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

        if kwargs.get("ambient_occlusion_enabled", False) or kwargs.get(
            "clamp_known_pixels", False
        ):
            raise ValueError(
                "Latent diffusion does not support ambient corruption or observation clamping."
            )
        if not eo_in_pixel_space:
            raise ValueError("EO must remain in physical grid space.")
        if kwargs.get("condition_per_depth_valid_mask", False):
            # The parent checks masks against generated channels; latent models
            # instead preserve one mask per physical channel.
            if int(kwargs.get("condition_mask_channels", 0)) != autoencoder.in_channels:
                raise ValueError(
                    "Latent conditioning requires one mask per physical depth."
                )
            kwargs["condition_per_depth_valid_mask"] = False
        super().__init__(**kwargs)
        if self.ocean_loss.any_extra_enabled():
            raise ValueError(
                "Pixel auxiliary losses are unsupported in latent space; use decoded_observation_weight."
            )
        if (
            tuple(self.output_fields) != autoencoder.output_fields
            or self.climatology_residual != autoencoder.climatology_residual
        ):
            raise ValueError("AE and diffusion field/residual contracts must match.")
        self.decoded_observation_weight = float(decoded_observation_weight)
        if self.decoded_observation_weight < 0:
            raise ValueError("decoded_observation_weight must be nonnegative.")
        self.full_reconstruction_logging_enabled = False
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
        model_config_path: str = str(config_path("lat_space", "model_config.yaml")),
        data_config_path: str = str(
            config_path("px_space", "training_super_config.yaml")
        ),
        training_config_path: str = str(
            config_path("lat_space", "training_config.yaml")
        ),
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
        plateau_interval = str(plateau_cfg.get("interval", "epoch"))
        warmup_cfg = scheduler_cfg.get("warmup", {})
        val_sampling_cfg = t.get("validation_sampling", {})
        coord_cfg = m.get("coord_conditioning", {})
        ambient_cfg = m.get("ambient_occlusion", {})
        postprocess_cfg = m.get("post_process", m.get("post-process", {}))
        gaussian_blur_cfg = postprocess_cfg.get("gaussian_blur", {})
        latent_cfg = m.get("latent", {})
        if not bool(latent_cfg.get("freeze_autoencoder", True)):
            raise ValueError(
                "The supported two-stage workflow requires freeze_autoencoder=true."
            )

        ae_config_value = str(latent_cfg.get("ae_config_path", "")).strip()
        if not ae_config_value:
            raise ValueError(
                "latent.ae_config_path is required for model.model_type='latent_cond_dif'."
            )
        ae_config_path = resolve_config_path(ae_config_value)
        if not ae_config_path.is_file():
            raise FileNotFoundError(f"AE config not found: {ae_config_path}")

        autoencoder = DepthBandAutoencoder.from_config(str(ae_config_path))

        ae_checkpoint = latent_cfg.get("ae_checkpoint", False)
        if ae_checkpoint not in (False, None):
            ae_checkpoint_path = Path(str(ae_checkpoint)).expanduser()
            if not ae_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"AE checkpoint not found: {ae_checkpoint_path}"
                )
            checkpoint = torch.load(ae_checkpoint_path, map_location="cpu")
            state_dict = cls._extract_autoencoder_state_dict(checkpoint)
            if checkpoint.get("ae_contract") != autoencoder.contract():
                raise ValueError(
                    "Autoencoder checkpoint contract mismatch; use a calibrated mask-aware AE export."
                )
            autoencoder.load_state_dict(state_dict, strict=True)
            if (
                not torch.isfinite(autoencoder.latent_mean).all()
                or not torch.isfinite(autoencoder.latent_std).all()
                or not (autoencoder.latent_std > 0).all()
            ):
                raise ValueError("Invalid AE latent normalization statistics.")
            if not bool(autoencoder.latent_calibrated):
                raise ValueError(
                    "AE checkpoint lacks training-only latent calibration."
                )
        elif bool(latent_cfg.get("freeze_autoencoder", True)):
            raise ValueError("Frozen latent diffusion requires latent.ae_checkpoint.")

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

        if (
            int(m.get("physical_channels", autoencoder.in_channels))
            != autoencoder.in_channels
        ):
            raise ValueError("AE physical channels do not match the resolved scenario.")
        if m.get("coastal_loss", {}).get("enabled", False):
            raise ValueError(
                "Pixel coastal loss is not supported by the latent workflow."
            )
        unet_kwargs = cls._parse_unet_config(m)
        coord_embed_dim = coord_cfg.get("embed_dim", None)
        if coord_embed_dim is not None:
            coord_embed_dim = int(coord_embed_dim)

        return cls(
            datamodule=datamodule,
            output_fields=m.get("output_fields", list(autoencoder.output_fields)),
            variable_scenario=m.get("scenario", None),
            condition_eo_channels=int(m.get("condition_eo_channels", 1)),
            condition_per_depth_valid_mask=bool(
                m.get("condition_per_depth_valid_mask", False)
            ),
            condition_use_wet_mask=bool(m.get("condition_use_wet_mask", False)),
            mask_diffusion_with_wet_mask=bool(
                m.get("mask_diffusion_with_wet_mask", False)
            ),
            climatology_residual=bool(m.get("climatology_residual", False)),
            losses_config=m.get("losses"),
            decoded_observation_weight=float(
                latent_cfg.get("decoded_observation_weight", 0.0)
            ),
            autoencoder=autoencoder,
            autoencoder_frozen=bool(latent_cfg.get("freeze_autoencoder", True)),
            eo_in_pixel_space=bool(latent_cfg.get("eo_in_pixel_space", True)),
            generated_channels=generated_channels,
            condition_channels=int(m.get("condition_channels", latent_channels)),
            condition_mask_channels=int(m.get("condition_mask_channels", 1)),
            condition_include_eo=bool(m.get("condition_include_eo", True)),
            condition_use_valid_mask=bool(m.get("condition_use_valid_mask", True)),
            condition_use_land_mask=bool(m.get("condition_use_land_mask", False)),
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
            lr_scheduler_interval=plateau_interval,
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
                val_sampling_cfg.get("max_full_reconstruction_samples", 5)
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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist AE semantics with the diffusion state for strict resumes."""
        checkpoint["ae_contract"] = self.autoencoder.contract()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject checkpoints from different field or normalization contracts."""
        if checkpoint.get("ae_contract") != self.autoencoder.contract():
            raise ValueError("Latent checkpoint AE contract mismatch.")

    def encode_fields(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode physical fields and apply fixed training-set latent scaling."""
        encoded = self.autoencoder.encode(value, valid_mask, wet_mask)
        return (encoded - self.autoencoder.latent_mean) / self.autoencoder.latent_std

    def input_T(self, value: torch.Tensor) -> torch.Tensor:
        """Encode a fully observed physical tensor; sparse callers pass explicit masks."""
        return self.encode_fields(value)

    def output_T(self, value: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents while retaining gradients to the denoiser."""
        return self.autoencoder.decode(
            value * self.autoencoder.latent_std + self.autoencoder.latent_mean
        )

    def _prepare_condition_for_model(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        eo: torch.Tensor | None = None,
        land_mask: torch.Tensor | None = None,
        wet_mask: torch.Tensor | None = None,
        background: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine encoded observations with uncompressed depth/domain evidence."""
        domain = wet_mask if wet_mask is not None else land_mask
        valid_mask = self.autoencoder.align_mask(
            valid_mask, x
        ) & self.autoencoder.align_mask(domain, x)
        data_t = self.encode_fields(x, valid_mask, domain)
        parts = []
        if self.condition_include_eo:
            if eo is None or eo.shape[1] != self.condition_eo_channels:
                raise ValueError(
                    "EO channels do not match the latent condition contract."
                )
            parts.append(eo)
        parts.append(data_t)
        mask = self._prepare_condition_mask(
            valid_mask, batch_size=x.shape[0], height=x.shape[2], width=x.shape[3]
        )
        if mask is not None:
            parts.append(mask.to(x.dtype))
        land = self._prepare_land_condition_mask(
            land_mask, batch_size=x.shape[0], height=x.shape[2], width=x.shape[3]
        )
        if land is not None:
            parts.append(land.to(x.dtype))
        if self.condition_use_wet_mask:
            if wet_mask is None:
                raise ValueError(
                    "Latent conditioning requires a static depth wet_mask."
                )
            parts.append(wet_mask.to(x.dtype))
        if self.climatology_residual:
            if background is None:
                raise ValueError("Residual conditioning requires climatology.")
            parts.append(background)
        condition = torch.cat(parts, dim=1)
        if condition.shape[1] != self.model.condition_channels:
            raise ValueError(
                f"Latent condition channels: built {condition.shape[1]}, expected {self.model.condition_channels}."
            )
        return condition

    def _extract_known_values_and_mask(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> tuple[None, None]:
        """Physical observations cannot clamp individual encoded features."""
        return None, None

    @torch.no_grad()
    def forward(
        self,
        condition: torch.Tensor,
        *args: Any,
        wet_mask: torch.Tensor | None = None,
        clamp_known_pixels: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Sample on the shared horizontal ocean domain, then decode physical depths."""
        if clamp_known_pixels:
            raise ValueError("Latent observation clamping is unsupported.")
        # A latent channel mixes all depths, so only wholly dry columns are zeroed.
        domain = wet_mask.any(dim=1, keepdim=True) if wet_mask is not None else None
        return super().forward(
            condition, *args, wet_mask=domain, clamp_known_pixels=False, **kwargs
        )

    def _latent_step(self, batch: dict[str, Any], *, prefix: str) -> torch.Tensor:
        """Denoise dense target latents; score optional observations after decoding."""
        if self.autoencoder_frozen:
            self.autoencoder.eval()
        data = self._prepare_model_batch_tensors(batch, include_y=True)
        x, y = data["x"], data["y"]
        xm, ym = data["x_valid_mask"], data["y_valid_mask"]
        wet, background = self._prepare_depth_context(batch, x)
        if wet is None:
            wet = self.autoencoder.align_mask(batch.get("land_mask"), y)
        xm = self.autoencoder.align_mask(xm, x) & wet
        ym = self.autoencoder.align_mask(ym, y) & wet
        if any(
            key in batch
            for key in ("y_supervision_weight", "y_salinity_supervision_weight")
        ):
            raise ValueError(
                "Latent dense targets do not support synthetic-prior confidence weights."
            )
        condition = self._prepare_condition_for_model(
            self._subtract_background(x, xm, background),
            xm,
            eo=batch.get("eo"),
            land_mask=batch.get("land_mask"),
            wet_mask=wet,
            background=background,
        )
        target = self.encode_fields(
            self._subtract_background(y, ym, background), ym, wet
        )
        latent_valid = ym.any(dim=1, keepdim=True)
        result = self.model.p_loss(
            target,
            condition,
            loss_mask=latent_valid,
            mask_loss=True,
            wet_mask=(
                wet.any(dim=1, keepdim=True)
                if self.mask_diffusion_with_wet_mask
                else None
            ),
            coord=batch.get("coords"),
            date=batch.get("date"),
            return_context=self.decoded_observation_weight > 0,
        )
        if self.decoded_observation_weight > 0:
            loss, context = result
            decoded = self.output_T(context["x0_pred"])
            if background is not None:
                decoded = decoded + background
            difference = torch.where(xm, decoded - torch.where(xm, x, 0.0), 0.0)
            observed_loss = difference.abs().sum() / xm.sum().clamp_min(1)
            self.log(
                f"{prefix}/decoded_observation_l1",
                observed_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
            loss = loss + self.decoded_observation_weight * observed_loss
        else:
            loss = result
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        if prefix == "val":
            self.log(
                "val/loss_ckpt",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Train the denoiser in latent space with physical conditioning masks."""
        return self._latent_step(batch, prefix="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Log denoising loss; the fixed reconstruction callback selects checkpoints."""
        return self._latent_step(batch, prefix="val")

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
