from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as tvf
import yaml

from .DenoisingDiffusionProcess import (
    DDIM_Sampler,
    DenoisingDiffusionConditionalProcess,
)
from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch
from utils.validation_denoise import (
    build_evenly_spaced_capture_steps,
    log_wandb_diffusion_schedule_profile,
    log_wandb_conditional_reconstruction_grid,
    log_wandb_denoise_timestep_grid,
)
from utils.validation_step_reconstruction_dump import (
    dump_validation_reconstruction_sequence,
)


class PixelDiffusionConditional(pl.LightningModule):
    # Prefixes that are allowed to differ between checkpoints when only the
    # validation sampler implementation/config changed (e.g., DDPM <-> DDIM).
    """Lightning module that trains and samples conditional pixel diffusion."""
    _SAMPLER_STATE_PREFIXES: tuple[str, ...] = ("val_sampler.",)
    _SUPPORTED_X_HOLDOUT_VARIANTS: tuple[str, ...] = ("eo_4band", "4band_eo", "4bands")

    def __init__(
        self,
        datamodule: pl.LightningDataModule | None = None,
        generated_channels: int = 1,
        condition_channels: int = 1,
        condition_mask_channels: int = 1,
        condition_include_eo: bool = False,
        condition_use_valid_mask: bool = True,
        clamp_known_pixels: bool = True,
        mask_loss_with_valid_pixels: bool = False,
        training_objective_mode: str = "standard",
        training_objective_holdout_fraction: float = 0.15,
        training_objective_deterministic_val_mask: bool = True,
        training_objective_dataset_variant: str = "eo_4band",
        training_objective_dump_val_reconstruction_enabled: bool = False,
        training_objective_dump_val_reconstruction_output_dir: str = "temp/val_step_reconstruction",
        training_objective_dump_val_reconstruction_max_samples: int = 1,
        training_objective_dump_val_reconstruction_only_first_batch: bool = True,
        parameterization: str = "epsilon",
        num_timesteps: int = 1000,
        noise_schedule: str = "linear",
        noise_beta_start: float = 1e-4,
        noise_beta_end: float = 2e-2,
        unet_dim: int = 64,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        unet_with_time_emb: bool = True,
        unet_output_mean_scale: bool = False,
        unet_residual: bool = False,
        coord_conditioning_enabled: bool = False,
        coord_encoding: str = "unit_sphere",
        date_conditioning_enabled: bool = False,
        date_encoding: str = "day_of_year_sincos",
        coord_embed_dim: int | None = None,
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
        lr_warmup_enabled: bool = True,
        lr_warmup_steps: int = 1000,
        lr_warmup_start_ratio: float = 0.1,
        val_inference_sampler: str = "ddpm",
        val_ddim_num_timesteps: int = 200,
        val_ddim_eta: float = 0.0,
        log_intermediates: bool = True,
        skip_full_reconstruction_in_sanity_check: bool = True,
        max_full_reconstruction_samples: int = 4,
        postprocess_gaussian_blur_enabled: bool = False,
        postprocess_gaussian_blur_sigma: float = 0.35,
        postprocess_gaussian_blur_kernel_size: int = 3,
        wandb_verbose: bool = True,
        log_stats_every_n_steps: int = 1,
        log_images_every_n_steps: int = 200,
    ) -> None:
        """Initialize PixelDiffusionConditional with configured parameters.

        Args:
            datamodule (pl.LightningDataModule | None): Input value.
            generated_channels (int): Input value.
            condition_channels (int): Input value.
            condition_mask_channels (int): Mask tensor controlling valid or known pixels.
            condition_include_eo (bool): Boolean flag controlling behavior.
            condition_use_valid_mask (bool): Mask tensor controlling valid or known pixels.
            clamp_known_pixels (bool): Boolean flag controlling behavior.
            mask_loss_with_valid_pixels (bool): Mask tensor controlling valid or known pixels.
            training_objective_mode (str): Input value.
            training_objective_holdout_fraction (float): Input value.
            training_objective_deterministic_val_mask (bool): Boolean flag controlling behavior.
            training_objective_dataset_variant (str): Input value.
            training_objective_dump_val_reconstruction_enabled (bool): Boolean flag controlling behavior.
            training_objective_dump_val_reconstruction_output_dir (str): Path to an input or output file.
            training_objective_dump_val_reconstruction_max_samples (int): Size/count parameter.
            training_objective_dump_val_reconstruction_only_first_batch (bool): Boolean flag controlling behavior.
            parameterization (str): Input value.
            num_timesteps (int): Step or timestep value.
            noise_schedule (str): Input value.
            noise_beta_start (float): Input value.
            noise_beta_end (float): Input value.
            unet_dim (int): Input value.
            unet_dim_mults (tuple[int, ...]): Input value.
            unet_with_time_emb (bool): Boolean flag controlling behavior.
            unet_output_mean_scale (bool): Boolean flag controlling behavior.
            unet_residual (bool): Boolean flag controlling behavior.
            coord_conditioning_enabled (bool): Boolean flag controlling behavior.
            coord_encoding (str): Input value.
            date_conditioning_enabled (bool): Boolean flag controlling behavior.
            date_encoding (str): Input value.
            coord_embed_dim (int | None): Input value.
            batch_size (int): Size/count parameter.
            lr (float): Input value.
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
            lr_warmup_enabled (bool): Boolean flag controlling behavior.
            lr_warmup_steps (int): Step or timestep value.
            lr_warmup_start_ratio (float): Input value.
            val_inference_sampler (str): Input value.
            val_ddim_num_timesteps (int): Input value.
            val_ddim_eta (float): Input value.
            log_intermediates (bool): Boolean flag controlling behavior.
            skip_full_reconstruction_in_sanity_check (bool): Boolean flag controlling behavior.
            max_full_reconstruction_samples (int): Input value.
            postprocess_gaussian_blur_enabled (bool): Boolean flag controlling behavior.
            postprocess_gaussian_blur_sigma (float): Input value.
            postprocess_gaussian_blur_kernel_size (int): Input value.
            wandb_verbose (bool): Boolean flag controlling behavior.
            log_stats_every_n_steps (int): Step or timestep value.
            log_images_every_n_steps (int): Step or timestep value.

        Returns:
            None: No value is returned.
        """
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
        self.lr_warmup_enabled = bool(lr_warmup_enabled)
        self.lr_warmup_steps = int(lr_warmup_steps)
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0.")
        self.lr_warmup_start_ratio = float(lr_warmup_start_ratio)
        if not 0.0 <= self.lr_warmup_start_ratio <= 1.0:
            raise ValueError("lr_warmup_start_ratio must be in [0.0, 1.0].")
        self.val_inference_sampler = str(val_inference_sampler).strip().lower()
        self.val_ddim_num_timesteps = max(1, int(val_ddim_num_timesteps))
        self.val_ddim_eta = float(val_ddim_eta)
        self.log_intermediates = bool(log_intermediates)
        self.skip_full_reconstruction_in_sanity_check = bool(
            skip_full_reconstruction_in_sanity_check
        )
        self.max_full_reconstruction_samples = max(
            1, int(max_full_reconstruction_samples)
        )
        self.postprocess_gaussian_blur_enabled = bool(postprocess_gaussian_blur_enabled)
        self.postprocess_gaussian_blur_sigma = max(
            0.0, float(postprocess_gaussian_blur_sigma)
        )
        kernel_size = int(postprocess_gaussian_blur_kernel_size)
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.postprocess_gaussian_blur_kernel_size = kernel_size
        self.condition_mask_channels = int(max(0, condition_mask_channels))
        self.condition_include_eo = bool(condition_include_eo)
        self.condition_use_valid_mask = bool(condition_use_valid_mask)
        self.clamp_known_pixels = bool(clamp_known_pixels)
        self.mask_loss_with_valid_pixels = bool(mask_loss_with_valid_pixels)
        self.training_objective_mode_requested = self._normalize_training_objective_mode(
            training_objective_mode
        )
        self.training_objective_dataset_variant = str(
            training_objective_dataset_variant
        ).strip().lower()
        holdout_fraction = float(training_objective_holdout_fraction)
        if holdout_fraction < 0.0 or holdout_fraction > 1.0:
            raise ValueError("training_objective_holdout_fraction must be in [0.0, 1.0].")
        self.training_objective_holdout_fraction = holdout_fraction
        self.training_objective_deterministic_val_mask = bool(
            training_objective_deterministic_val_mask
        )
        self.training_objective_dump_val_reconstruction_enabled = bool(
            training_objective_dump_val_reconstruction_enabled
        )
        self.training_objective_dump_val_reconstruction_output_dir = str(
            training_objective_dump_val_reconstruction_output_dir
        ).strip()
        self.training_objective_dump_val_reconstruction_max_samples = max(
            1, int(training_objective_dump_val_reconstruction_max_samples)
        )
        self.training_objective_dump_val_reconstruction_only_first_batch = bool(
            training_objective_dump_val_reconstruction_only_first_batch
        )
        self.training_objective_mode = self.training_objective_mode_requested
        if (
            self.training_objective_mode_requested == "x_holdout_sparse"
            and self.training_objective_dataset_variant
            not in self._SUPPORTED_X_HOLDOUT_VARIANTS
        ):
            warnings.warn(
                "model.training_objective.mode='x_holdout_sparse' is only supported "
                "for non-OSTIA eo_4band datasets. Falling back to 'standard'.",
                stacklevel=2,
            )
            self.training_objective_mode = "standard"
        self.wandb_verbose = wandb_verbose
        self.log_stats_every_n_steps = max(1, int(log_stats_every_n_steps))
        self.log_images_every_n_steps = max(1, int(log_images_every_n_steps))
        self._warned_known_pixel_mismatch = False

        # Conditional diffusion predicts y while conditioned on x (and optional mask channels).
        self.model = DenoisingDiffusionConditionalProcess(
            generated_channels=generated_channels,
            condition_channels=condition_channels,
            parameterization=parameterization,
            num_timesteps=num_timesteps,
            schedule=noise_schedule,
            beta_start=noise_beta_start,
            beta_end=noise_beta_end,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            unet_with_time_emb=unet_with_time_emb,
            unet_output_mean_scale=unet_output_mean_scale,
            unet_residual=unet_residual,
            coord_conditioning_enabled=coord_conditioning_enabled,
            coord_encoding=coord_encoding,
            date_conditioning_enabled=date_conditioning_enabled,
            date_encoding=date_encoding,
            coord_embed_dim=coord_embed_dim,
        )
        train_betas = self.model.forward_process.betas.detach().clone()
        self.val_sampler = self._build_validation_sampler(train_betas)
        # Cached validation mini-batch
        # (x_context, y, eo, valid_mask_context, land_mask, coords, date, loss_mask)
        # used for one epoch-end full reverse-diffusion reconstruction pass.
        self._cached_val_example: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
            | None
        ) = None
        self._logged_schedule_profile_in_sanity = False

    @classmethod
    def from_config(
        cls,
        model_config_path: str = "configs/model_config.yaml",
        data_config_path: str = "configs/data.yaml",
        training_config_path: str = "configs/training_config.yaml",
        datamodule: pl.LightningDataModule | None = None,
    ) -> "PixelDiffusionConditional":
        """Compute from config and return the result.

        Args:
            model_config_path (str): Path to an input or output file.
            data_config_path (str): Path to an input or output file.
            training_config_path (str): Path to an input or output file.
            datamodule (pl.LightningDataModule | None): Input value.

        Returns:
            'PixelDiffusionConditional': Computed output value.
        """
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
        objective_cfg = m.get("training_objective", {})
        objective_dump_cfg = objective_cfg.get("dump_val_reconstruction", {})
        dataset_variant = cls._resolve_dataset_variant_from_data_cfg(
            data_cfg, data_config_path
        )
        postprocess_cfg = m.get("post_process", m.get("post-process", {}))
        gaussian_blur_cfg = postprocess_cfg.get("gaussian_blur", {})
        unet_kwargs = cls._parse_unet_config(m)
        coord_embed_dim = coord_cfg.get("embed_dim", None)
        if coord_embed_dim is not None:
            coord_embed_dim = int(coord_embed_dim)

        return cls(
            datamodule=datamodule,
            generated_channels=int(m.get("generated_channels", 1)),
            condition_channels=int(m.get("condition_channels", m.get("bands", 1))),
            condition_mask_channels=int(m.get("condition_mask_channels", 1)),
            condition_include_eo=bool(m.get("condition_include_eo", False)),
            condition_use_valid_mask=bool(m.get("condition_use_valid_mask", True)),
            clamp_known_pixels=bool(m.get("clamp_known_pixels", True)),
            mask_loss_with_valid_pixels=bool(
                m.get("mask_loss_with_valid_pixels", False)
            ),
            training_objective_mode=str(objective_cfg.get("mode", "standard")),
            training_objective_holdout_fraction=float(
                objective_cfg.get("holdout_fraction", 0.15)
            ),
            training_objective_deterministic_val_mask=bool(
                objective_cfg.get("deterministic_val_mask", True)
            ),
            training_objective_dataset_variant=dataset_variant,
            training_objective_dump_val_reconstruction_enabled=bool(
                objective_dump_cfg.get("enabled", False)
            ),
            training_objective_dump_val_reconstruction_output_dir=str(
                objective_dump_cfg.get("output_dir", "temp/val_step_reconstruction")
            ),
            training_objective_dump_val_reconstruction_max_samples=int(
                objective_dump_cfg.get("max_samples", 1)
            ),
            training_objective_dump_val_reconstruction_only_first_batch=bool(
                objective_dump_cfg.get("only_first_batch", True)
            ),
            parameterization=str(m.get("parameterization", "epsilon")),
            num_timesteps=int(
                noise_cfg.get("num_timesteps", m.get("num_timesteps", 1000))
            ),
            noise_schedule=str(
                noise_cfg.get("schedule", m.get("noise_schedule", "linear"))
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
            lr_warmup_enabled=bool(warmup_cfg.get("enabled", True)),
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
            skip_full_reconstruction_in_sanity_check=bool(
                val_sampling_cfg.get("skip_full_reconstruction_in_sanity_check", True)
            ),
            max_full_reconstruction_samples=int(
                val_sampling_cfg.get("max_full_reconstruction_samples", 4)
            ),
            postprocess_gaussian_blur_enabled=bool(
                gaussian_blur_cfg.get("enabled", False)
            ),
            postprocess_gaussian_blur_sigma=float(gaussian_blur_cfg.get("sigma", 0.35)),
            postprocess_gaussian_blur_kernel_size=int(
                gaussian_blur_cfg.get("kernel_size", 3)
            ),
            wandb_verbose=bool(w.get("verbose", True)),
            log_stats_every_n_steps=int(w.get("log_stats_every_n_steps", 1)),
            log_images_every_n_steps=int(w.get("log_images_every_n_steps", 200)),
        )

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        """Load and return yaml data.

        Args:
            path (str): Path to an input or output file.

        Returns:
            dict[str, Any]: Dictionary containing computed outputs.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _resolve_dataset_variant_from_data_cfg(
        data_cfg: dict[str, Any], data_config_path: str
    ) -> str:
        dataset_cfg = data_cfg.get("dataset", {})
        core_cfg = dataset_cfg.get("core", {})
        value = core_cfg.get("dataset_variant", dataset_cfg.get("dataset_variant"))
        if value is not None:
            return str(value).strip().lower()
        stem = Path(data_config_path).stem.lower()
        if "ostia" in stem:
            return "ostia"
        return "eo_4band"

    @staticmethod
    def _normalize_training_objective_mode(mode: str) -> str:
        value = str(mode).strip().lower().replace("-", "_")
        if value in {"standard", "default"}:
            return "standard"
        if value in {"x_holdout_sparse", "x_holdout", "x_only_holdout"}:
            return "x_holdout_sparse"
        raise ValueError(
            "model.training_objective.mode must be one of "
            "{'standard', 'x_holdout_sparse'}."
        )

    @staticmethod
    def _parse_unet_dim_mults(value: Any) -> tuple[int, ...]:
        """Helper that computes parse unet dim mults.

        Args:
            value (Any): Input value.

        Returns:
            tuple[int, ...]: Tuple containing computed outputs.
        """
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
        """Helper that computes parse unet config.

        Args:
            model_section (dict[str, Any]): Input value.

        Returns:
            dict[str, Any]: Dictionary containing computed outputs.
        """
        unet = model_section.get("unet", {})
        return {
            "unet_dim": int(unet.get("dim", 64)),
            "unet_dim_mults": cls._parse_unet_dim_mults(
                unet.get("dim_mults", [1, 2, 4, 8])
            ),
            "unet_with_time_emb": bool(unet.get("with_time_emb", True)),
            "unet_output_mean_scale": bool(unet.get("output_mean_scale", False)),
            "unet_residual": bool(unet.get("residual", False)),
        }

    def input_T(self, value: torch.Tensor) -> torch.Tensor:
        # Dataset already provides standardized temperatures (z-score), so keep identity here.
        # This avoids a second normalization pass and preserves the training target distribution.
        """Compute input T and return the result.

        Args:
            value (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        return value

    def output_T(self, value: torch.Tensor) -> torch.Tensor:
        # Return samples in the same standardized space used by the dataset.
        """Compute output T and return the result.

        Args:
            value (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        return value

    @staticmethod
    def _should_sync_dist() -> bool:
        """Helper that computes should sync dist.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            bool: Computed scalar output.
        """
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _log_common_batch_stats(
        self,
        tensor: torch.Tensor,
        prefix: str,
        batch_size: int | None = None,
        *,
        on_step: bool = True,
        on_epoch: bool = False,
    ) -> None:
        """Helper that computes log common batch stats.

        Args:
            tensor (torch.Tensor): Tensor input for the computation.
            prefix (str): Input value.
            batch_size (int | None): Size/count parameter.
            on_step (bool): Step or timestep value.
            on_epoch (bool): Boolean flag controlling behavior.

        Returns:
            None: No value is returned.
        """
        if not self.wandb_verbose:
            return
        if self.global_step % self.log_stats_every_n_steps != 0:
            return
        sync_dist = self._should_sync_dist() if on_epoch else False

        self.log(
            f"stats/{prefix}_batch_mean",
            tensor.mean(),
            on_step=on_step,
            on_epoch=on_epoch,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"stats/{prefix}_batch_std",
            tensor.std(),
            on_step=on_step,
            on_epoch=on_epoch,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"stats/{prefix}_batch_min",
            tensor.min(),
            on_step=on_step,
            on_epoch=on_epoch,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"stats/{prefix}_batch_max",
            tensor.max(),
            on_step=on_step,
            on_epoch=on_epoch,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )

    def _log_pre_diffusion_stats(
        self, tensor: torch.Tensor, prefix: str, batch_size: int | None = None
    ) -> None:
        """Helper that computes log pre diffusion stats.

        Args:
            tensor (torch.Tensor): Tensor input for the computation.
            prefix (str): Input value.
            batch_size (int | None): Size/count parameter.

        Returns:
            None: No value is returned.
        """
        if not self.wandb_verbose:
            return
        if self.global_step % self.log_stats_every_n_steps != 0:
            return
        sync_dist = self._should_sync_dist()

        self.log(
            f"pre_diffusion/{prefix}_mean",
            tensor.mean(),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"pre_diffusion/{prefix}_std",
            tensor.std(unbiased=False),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )

    def _maybe_log_wandb_images(
        self,
        batch: Any,
        output: torch.Tensor,
        prefix: str,
        *,
        use_minmax: bool = False,
    ) -> None:
        """Helper that computes maybe log wandb images.

        Args:
            batch (Any): Input value.
            output (torch.Tensor): Tensor input for the computation.
            prefix (str): Input value.
            use_minmax (bool): Boolean flag controlling behavior.

        Returns:
            None: No value is returned.
        """
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

        image = output[0].detach().float().squeeze(0)
        if use_minmax:
            image_np = self._minmax_stretch(image)
        else:
            image_np = image.cpu().numpy()
        # Quick qualitative sanity check (ground-truth sample, not reconstructed output).
        experiment.log({f"{prefix}/y_preview": wandb.Image(image_np)})

    @staticmethod
    def _minmax_stretch(
        image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        nodata_value: float | None = 0.0,
    ) -> np.ndarray:
        """Helper that computes minmax stretch.

        Args:
            image (torch.Tensor): Tensor input for the computation.
            mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            nodata_value (float | None): Input value.

        Returns:
            np.ndarray: Computed output value.
        """
        image = image.detach().float()
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        # image = temperature_normalize(mode="denorm", tensor=image)
        stretched = minmax_stretch(image, mask=mask, nodata_value=nodata_value)
        return stretched.cpu().numpy().astype(np.float32)

    def train_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """Return the training dataloader from the attached datamodule.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            torch.utils.data.DataLoader[Any]: Computed output value.
        """
        if self.datamodule is None:
            raise RuntimeError("No datamodule was provided to the model.")
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> torch.utils.data.DataLoader[Any] | None:
        """Return the validation dataloader from the attached datamodule.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            torch.utils.data.DataLoader[Any] | None: Computed output value.
        """
        if self.datamodule is None:
            return None
        return self.datamodule.val_dataloader()

    @staticmethod
    def _tensor_stats(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper that computes tensor stats.

        Args:
            tensor (torch.Tensor): Tensor input for the computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing computed outputs.
        """
        return tensor.min(), tensor.mean(), tensor.std(unbiased=False)

    def _build_validation_sampler(
        self, train_betas: torch.Tensor
    ) -> DDIM_Sampler | None:
        # Validation can use either:
        # - DDPM (full stochastic chain, faithful to training dynamics)
        # - DDIM (fewer deterministic/stochastic steps, faster previews).
        """Helper that computes build validation sampler.

        Args:
            train_betas (torch.Tensor): Tensor input for the computation.

        Returns:
            DDIM_Sampler | None: Computed output value.
        """
        if self.val_inference_sampler == "ddpm":
            return None
        if self.val_inference_sampler == "ddim":
            return DDIM_Sampler(
                num_timesteps=min(
                    self.val_ddim_num_timesteps, int(train_betas.numel())
                ),
                train_timesteps=int(train_betas.numel()),
                betas=train_betas,
                parameterization=str(self.model.parameterization),
                # Keep sampler in standardized-space domain; do not clamp to [-1, 1].
                clip_sample=False,
                eta=self.val_ddim_eta,
            )
        raise ValueError(
            "training.validation_sampling.sampler must be one of {'ddpm', 'ddim'} "
            f"(got '{self.val_inference_sampler}')."
        )

    @classmethod
    def _is_sampler_only_state_mismatch(
        cls,
        missing_keys: set[str],
        unexpected_keys: set[str],
    ) -> bool:
        # Accept fallback only when every mismatched key belongs to an
        # explicitly whitelisted sampler namespace.
        """Helper that computes is sampler only state mismatch.

        Args:
            missing_keys (set[str]): Input value.
            unexpected_keys (set[str]): Input value.

        Returns:
            bool: Computed scalar output.
        """
        mismatched_keys = missing_keys | unexpected_keys
        if not mismatched_keys:
            return False
        return all(
            any(key.startswith(prefix) for prefix in cls._SAMPLER_STATE_PREFIXES)
            for key in mismatched_keys
        )

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> Any:
        # Keep default PyTorch/Lightning behavior unless we can prove the
        # mismatch is sampler-only. This preserves strictness for all learned
        # weights and model architecture changes.
        """Load checkpoint weights into the current module.

        Args:
            state_dict (dict[str, torch.Tensor]): Tensor input for the computation.
            strict (bool): Boolean flag controlling behavior.

        Returns:
            Any: Computed output value.
        """
        if not strict:
            return super().load_state_dict(state_dict, strict=False)

        try:
            return super().load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model_state_keys = set(self.state_dict().keys())
            checkpoint_state_keys = set(state_dict.keys())
            missing_keys = model_state_keys - checkpoint_state_keys
            unexpected_keys = checkpoint_state_keys - model_state_keys

            # Only tolerate key drift introduced by validation sampler switches.
            # Any non-sampler key mismatch still raises and blocks resume.
            if not self._is_sampler_only_state_mismatch(
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            ):
                raise

            warnings.warn(
                "Sampler-only checkpoint mismatch detected; "
                "retrying load_state_dict with strict=False.",
                stacklevel=2,
            )
            return super().load_state_dict(state_dict, strict=False)

    def _prepare_condition_mask(
        self,
        valid_mask: torch.Tensor | None,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        """Helper that computes prepare condition mask.

        Args:
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            batch_size (int): Size/count parameter.
            height (int): Size/count parameter.
            width (int): Size/count parameter.

        Returns:
            torch.Tensor | None: Tensor output produced by this call.
        """
        if not self.condition_use_valid_mask or self.condition_mask_channels <= 0:
            return None
        if valid_mask is None:
            raise RuntimeError(
                "condition_use_valid_mask=true requires batch['valid_mask']."
            )

        mask = valid_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise RuntimeError("valid_mask must be shaped as (B,C,H,W) or (B,H,W).")
        if int(mask.size(0)) != int(batch_size):
            raise RuntimeError("valid_mask batch size does not match x batch size.")
        if int(mask.size(-2)) != int(height) or int(mask.size(-1)) != int(width):
            raise RuntimeError("valid_mask spatial shape does not match x.")

        if int(mask.size(1)) == int(self.condition_mask_channels):
            return mask
        if int(mask.size(1)) == 1 and int(self.condition_mask_channels) > 1:
            return mask.expand(-1, int(self.condition_mask_channels), -1, -1)
        if int(self.condition_mask_channels) == 1 and int(mask.size(1)) > 1:
            return mask.amax(dim=1, keepdim=True)
        raise RuntimeError(
            "Could not match valid_mask channels to condition_mask_channels "
            f"(mask={int(mask.size(1))}, expected={int(self.condition_mask_channels)})."
        )

    def _prepare_condition_for_model(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        eo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Keep conditioning data in the same normalized range as diffusion targets.
        """Helper that computes prepare condition for model.

        Args:
            x (torch.Tensor): Tensor input for the computation.
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            eo (torch.Tensor | None): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        condition_parts: list[torch.Tensor] = []
        if self.condition_include_eo:
            if eo is None:
                raise RuntimeError(
                    "condition_include_eo=true requires batch['eo']."
                )
            condition_parts.append(self.input_T(eo))

        data_t = self.input_T(x)
        condition_parts.append(data_t)

        mask_t = self._prepare_condition_mask(
            valid_mask,
            batch_size=int(data_t.size(0)),
            height=int(data_t.size(-2)),
            width=int(data_t.size(-1)),
        )
        if mask_t is not None:
            # Keep mask numeric and device-aligned for concatenation with data channels.
            mask_t = mask_t.to(device=data_t.device, dtype=data_t.dtype)
            # Mask channel remains as-is; it is a semantic conditioning signal.
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

    def _extract_known_values_and_mask(
        self, data: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper that computes extract known values and mask.

        Args:
            data (torch.Tensor): Tensor input for the computation.
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]: Tuple containing computed outputs.
        """
        if valid_mask is None:
            return None, None

        known_mask = (valid_mask > 0.5).float()
        if known_mask.ndim == 3:
            known_mask = known_mask.unsqueeze(1)

        data_t = self.input_T(data)
        data_channels = int(data_t.size(1))
        generated_channels = int(self.model.generated_channels)
        if known_mask.ndim == 4:
            # Preserve per-channel known masks when they already match generated channels.
            if int(known_mask.size(1)) == generated_channels:
                pass
            elif int(known_mask.size(1)) == 1 and generated_channels > 1:
                known_mask = known_mask.expand(-1, generated_channels, -1, -1)
            elif generated_channels == 1 and int(known_mask.size(1)) > 1:
                known_mask = known_mask.amax(dim=1, keepdim=True)
            else:
                known_mask = known_mask.amax(dim=1, keepdim=True)
                if generated_channels > 1:
                    known_mask = known_mask.expand(-1, generated_channels, -1, -1)

        if data_channels == generated_channels:
            known_values = data_t
        elif data_channels == 1 and generated_channels > 1:
            known_values = data_t.repeat(1, generated_channels, 1, 1)
        else:
            if not self._warned_known_pixel_mismatch:
                warnings.warn(
                    "Known-pixel clamping skipped: condition data channels "
                    f"({data_channels}) do not match generated channels "
                    f"({generated_channels}).",
                    stacklevel=2,
                )
                self._warned_known_pixel_mismatch = True
            return None, None

        return known_values, known_mask

    @torch.no_grad()
    def forward(
        self,
        condition: torch.Tensor,
        sampler: torch.nn.Module | None = None,
        verbose: bool = False,
        clamp_known_pixels: bool | None = None,
        *,
        known_mask: torch.Tensor | None = None,
        known_values: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        date: torch.Tensor | None = None,
        return_intermediates: bool = False,
        intermediate_step_indices: list[int] | None = None,
        return_x0_intermediates: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]
        | tuple[
            torch.Tensor,
            list[tuple[int, torch.Tensor]],
            list[tuple[int, torch.Tensor]],
        ]
    ):
        """Run the module forward computation.

        Args:
            condition (torch.Tensor): Tensor input for the computation.
            sampler (torch.nn.Module | None): Sampler instance used for reverse diffusion.
            verbose (bool): Boolean flag controlling behavior.
            clamp_known_pixels (bool | None): Boolean flag controlling behavior.
            known_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
            known_values (torch.Tensor | None): Tensor input for the computation.
            coords (torch.Tensor | None): Coordinate conditioning values.
            date (torch.Tensor | None): Date conditioning values.
            return_intermediates (bool): Boolean flag controlling behavior.
            intermediate_step_indices (list[int] | None): Input value.
            return_x0_intermediates (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]] | tuple[torch.Tensor, list[tuple[int, torch.Tensor]], list[tuple[int, torch.Tensor]]]: Tensor output produced by this call.
        """
        if not self.log_intermediates:
            return_intermediates = False
            intermediate_step_indices = None
        if not return_intermediates:
            return_x0_intermediates = False
        if clamp_known_pixels is None:
            clamp_known_pixels = self.clamp_known_pixels
        if not clamp_known_pixels:
            known_values = None
            known_mask = None
        elif known_values is None or known_mask is None:
            # Caller did not provide known pixels; skip clamping.
            known_values = None
            known_mask = None
        model_output = self.model(
            condition,
            sampler=sampler,
            verbose=verbose,
            known_mask=known_mask,
            known_values=known_values,
            coord=coords,
            date=date,
            return_intermediates=return_intermediates,
            intermediate_step_indices=intermediate_step_indices,
            return_x0_intermediates=return_x0_intermediates,
        )
        if not return_intermediates:
            return self.output_T(model_output)

        if return_x0_intermediates:
            final_sample, intermediates, x0_intermediates = model_output
            out_intermediates = [
                (step_idx, self.output_T(sample_t)) for step_idx, sample_t in intermediates
            ]
            out_x0_intermediates = [
                (step_idx, self.output_T(sample_t))
                for step_idx, sample_t in x0_intermediates
            ]
            return self.output_T(final_sample), out_intermediates, out_x0_intermediates

        final_sample, intermediates = model_output
        out_intermediates = [
            (step_idx, self.output_T(sample_t)) for step_idx, sample_t in intermediates
        ]
        return self.output_T(final_sample), out_intermediates

    def _masked_fraction(self, valid_mask: torch.Tensor | None) -> torch.Tensor:
        """Helper that computes masked fraction.

        Args:
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if valid_mask is None:
            return torch.as_tensor(0.0, device=self.device)
        return (valid_mask < 0.5).float().mean()

    def _prepare_objective_batch(
        self, batch: dict[str, Any], *, is_validation: bool, batch_idx: int
    ) -> dict[str, Any]:
        _ = is_validation
        _ = batch_idx
        x = batch["x"]
        eo = batch.get("eo")
        valid_mask = batch.get("valid_mask")
        land_mask = batch.get("land_mask")
        coords = batch.get("coords")
        date = batch.get("date")

        if self.training_objective_mode == "x_holdout_sparse":
            supervision_mask = batch.get("loss_mask")
            if supervision_mask is None:
                raise RuntimeError(
                    "x_holdout_sparse requires batch['loss_mask'] from dataset."
                )
            target = x
            x_context = x
            valid_mask_context = valid_mask
            mask_loss = True
        else:
            target = batch.get("y")
            if target is None:
                raise RuntimeError("Standard objective requires batch['y'].")
            x_context = x
            valid_mask_context = valid_mask
            supervision_mask = None
            mask_loss = self.mask_loss_with_valid_pixels

        model_condition = self._prepare_condition_for_model(
            x_context, valid_mask_context, eo=eo
        )
        return {
            "x_raw": x,
            "x_context": x_context,
            "target": target,
            "target_t": self.input_T(target),
            "condition": model_condition,
            "valid_mask_context": valid_mask_context,
            "supervision_mask": supervision_mask,
            "mask_loss": mask_loss,
            "eo": eo,
            "land_mask": land_mask,
            "coords": coords,
            "date": date,
        }

    @staticmethod
    def _loss_mask_for_logging(
        *,
        supervision_mask: torch.Tensor | None,
        valid_mask_context: torch.Tensor | None,
        mask_loss: bool,
    ) -> torch.Tensor | None:
        # Keep logging mask aligned with the exact region used for masked-loss reduction.
        if supervision_mask is not None:
            return supervision_mask
        if mask_loss and valid_mask_context is not None:
            return (valid_mask_context <= 0.5).to(dtype=valid_mask_context.dtype)
        return None

    def _maybe_dump_validation_reconstruction(
        self,
        *,
        batch_idx: int,
        objective_batch: dict[str, Any],
        y_target: torch.Tensor | None,
    ) -> None:
        if not self.training_objective_dump_val_reconstruction_enabled:
            return
        if (
            self.training_objective_dump_val_reconstruction_only_first_batch
            and int(batch_idx) != 0
        ):
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.sanity_checking:
            return

        x_context = objective_batch["x_context"]
        eo = objective_batch["eo"]
        valid_mask_context = objective_batch["valid_mask_context"]
        land_mask = objective_batch["land_mask"]
        coords = objective_batch["coords"]
        date = objective_batch["date"]
        supervision_mask = objective_batch["supervision_mask"]
        mask_loss = bool(objective_batch["mask_loss"])

        valid_mask_for_loss = self._loss_mask_for_logging(
            supervision_mask=supervision_mask,
            valid_mask_context=valid_mask_context,
            mask_loss=mask_loss,
        )

        pred = self.predict_step(
            {
                "x": x_context,
                "eo": eo,
                "valid_mask": valid_mask_context,
                "land_mask": land_mask,
                "coords": coords,
                "date": date,
                "sampler": self.val_sampler,
                "return_intermediates": True,
                # None captures all reverse-diffusion steps.
                "intermediate_step_indices": None,
            },
            batch_idx=batch_idx,
        )
        denoise_samples = pred.get("denoise_samples", [])
        if not denoise_samples:
            return

        epoch_i = int(self.current_epoch) if trainer is not None else 0
        dump_validation_reconstruction_sequence(
            output_dir=self.training_objective_dump_val_reconstruction_output_dir,
            epoch=epoch_i,
            global_step=int(self.global_step),
            batch_idx=int(batch_idx),
            x_context=x_context.detach(),
            denoise_samples=denoise_samples,
            y_target=None if y_target is None else y_target.detach(),
            eo=None if eo is None else eo.detach(),
            valid_mask_for_loss=(
                None if valid_mask_for_loss is None else valid_mask_for_loss.detach()
            ),
            valid_mask_context=(
                None if valid_mask_context is None else valid_mask_context.detach()
            ),
            land_mask=None if land_mask is None else land_mask.detach(),
            max_samples=self.training_objective_dump_val_reconstruction_max_samples,
        )

    def _apply_postprocess_gaussian_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        """Helper that computes apply postprocess gaussian blur.

        Args:
            tensor (torch.Tensor): Tensor input for the computation.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if not self.postprocess_gaussian_blur_enabled:
            return tensor
        if self.postprocess_gaussian_blur_sigma <= 0.0:
            return tensor
        if tensor.ndim not in (3, 4):
            return tensor

        kernel_size = int(self.postprocess_gaussian_blur_kernel_size)
        if kernel_size <= 1:
            return tensor

        sigma = float(self.postprocess_gaussian_blur_sigma)
        # torchvision applies the same spatial kernel per channel/band (depthwise),
        # so all bands are blurred equally with no channel mixing.
        return tvf.gaussian_blur(
            tensor,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma],
        )

    def _apply_postprocess_zero_land_pixels(
        self, tensor: torch.Tensor, land_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Helper that computes apply postprocess zero land pixels.

        Args:
            tensor (torch.Tensor): Tensor input for the computation.
            land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if land_mask is None:
            return tensor
        if tensor.ndim not in (3, 4):
            return tensor
        ocean_mask = (land_mask > 0.5).to(dtype=tensor.dtype, device=tensor.device)
        return tensor * ocean_mask

    def _apply_postprocess_merge_observed_pixels(
        self,
        generated: torch.Tensor,
        observed: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Helper that computes apply postprocess merge observed pixels.

        Args:
            generated (torch.Tensor): Tensor input for the computation.
            observed (torch.Tensor): Tensor input for the computation.
            valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        if valid_mask is None:
            return generated
        if generated.ndim != 4 or observed.ndim != 4:
            return generated
        keep_mask = (valid_mask > 0.5).to(
            dtype=generated.dtype, device=generated.device
        )
        if keep_mask.ndim == 3:
            keep_mask = keep_mask.unsqueeze(1)
        if keep_mask.ndim != 4:
            raise RuntimeError(
                "valid_mask must be shaped as (B,C,H,W) or (B,H,W) in predict_step."
            )
        if keep_mask.shape[0] != generated.shape[0] or keep_mask.shape[2:] != generated.shape[2:]:
            raise RuntimeError(
                f"valid_mask shape {tuple(keep_mask.shape)} does not match generated "
                f"shape {tuple(generated.shape)}."
            )
        if keep_mask.size(1) == 1 and generated.size(1) > 1:
            keep_mask = keep_mask.expand(-1, generated.size(1), -1, -1)
        elif keep_mask.size(1) != generated.size(1):
            raise RuntimeError(
                "valid_mask channels must match generated channels or be 1 "
                f"(got mask={int(keep_mask.size(1))}, generated={int(generated.size(1))})."
            )
        if observed.shape != generated.shape:
            if (
                observed.shape[0] == generated.shape[0]
                and observed.shape[2:] == generated.shape[2:]
                and observed.size(1) == 1
                and generated.size(1) > 1
            ):
                observed = observed.expand(-1, generated.size(1), -1, -1)
            else:
                raise RuntimeError(
                    f"Observed tensor shape {tuple(observed.shape)} does not match "
                    f"generated shape {tuple(generated.shape)}."
                )
        # Keep known observations from x and only use model predictions on missing pixels.
        return (generated * (1.0 - keep_mask)) + (observed * keep_mask)

    @torch.no_grad()
    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """Compute predict step and return the result.

        Args:
            batch (dict[str, Any]): Input value.
            batch_idx (int): Zero-based index for selecting a sample or batch.
            dataloader_idx (int): Input value.

        Returns:
            dict[str, Any]: Dictionary containing computed outputs.
        """
        x = batch["x"]
        eo = batch.get("eo")
        valid_mask = batch.get("valid_mask")
        land_mask = batch.get("land_mask")
        coords = batch.get("coords")
        date = batch.get("date")
        sampler = batch.get("sampler", self.val_sampler)
        clamp_known_pixels = batch.get("clamp_known_pixels", None)
        return_intermediates = bool(batch.get("return_intermediates", False))
        intermediate_step_indices = batch.get("intermediate_step_indices")

        model_condition = self._prepare_condition_for_model(x, valid_mask, eo=eo)
        known_values, known_mask = self._extract_known_values_and_mask(x, valid_mask)

        denoise_samples: list[tuple[int, torch.Tensor]] = []
        x0_denoise_samples: list[tuple[int, torch.Tensor]] = []
        if return_intermediates:
            y_hat, denoise_samples, x0_denoise_samples = self(
                model_condition,
                sampler=sampler,
                clamp_known_pixels=clamp_known_pixels,
                known_mask=known_mask,
                known_values=known_values,
                coords=coords,
                date=date,
                return_intermediates=True,
                intermediate_step_indices=intermediate_step_indices,
                return_x0_intermediates=True,
            )
        else:
            y_hat = self(
                model_condition,
                sampler=sampler,
                clamp_known_pixels=clamp_known_pixels,
                known_mask=known_mask,
                known_values=known_values,
                coords=coords,
                date=date,
            )

        # Keep all post-processing centralized in Lightning inference.
        y_hat_denorm = temperature_normalize(mode="denorm", tensor=y_hat)
        y_hat_denorm = self._apply_postprocess_gaussian_blur(y_hat_denorm)
        # Keep an unmerged version for visualization so reconstruction panels
        # show model output (plus land masking) rather than observed-pixel copy-in.
        y_hat_denorm_for_plot = self._apply_postprocess_zero_land_pixels(
            y_hat_denorm, land_mask
        )
        x_denorm = temperature_normalize(mode="denorm", tensor=x)
        y_hat_denorm = self._apply_postprocess_merge_observed_pixels(
            generated=y_hat_denorm,
            observed=x_denorm,
            valid_mask=valid_mask,
        )
        y_hat_denorm = self._apply_postprocess_zero_land_pixels(
            y_hat_denorm, land_mask
        )

        return {
            "y_hat": y_hat,
            "y_hat_denorm": y_hat_denorm,
            "y_hat_denorm_for_plot": y_hat_denorm_for_plot,
            "denoise_samples": denoise_samples,
            "x0_denoise_samples": x0_denoise_samples,
            "sampler": sampler,
        }

    def _log_validation_triplet_stats(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> None:
        """Helper that computes log validation triplet stats.

        Args:
            x (torch.Tensor): Tensor input for the computation.
            y (torch.Tensor): Tensor input for the computation.
            y_hat (torch.Tensor): Tensor input for the computation.

        Returns:
            None: No value is returned.
        """
        x_min, x_mean, x_std = self._tensor_stats(x)
        y_min, y_mean, y_std = self._tensor_stats(y)
        y_hat_min, y_hat_mean, y_hat_std = self._tensor_stats(y_hat)
        sync_dist = self._should_sync_dist()

        self.log(
            "val_triplet/min_x",
            x_min,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/min_y",
            y_min,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/min_y_hat",
            y_hat_min,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/mean_x",
            x_mean,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/mean_y",
            y_mean,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/mean_y_hat",
            y_hat_mean,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/std_x",
            x_std,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/std_y",
            y_std,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )
        self.log(
            "val_triplet/std_y_hat",
            y_hat_std,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=y.size(0),
        )

    def on_validation_epoch_start(self) -> None:
        # Reset cache every validation epoch to avoid carrying stale tensors across epochs.
        """Compute on validation epoch start and return the result.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            None: No value is returned.
        """
        self._cached_val_example = None
        if (
            self.trainer is not None
            and self.trainer.sanity_checking
            and not self._logged_schedule_profile_in_sanity
        ):
            sampler_for_profile = (
                self.val_sampler if self.val_sampler is not None else self.model.sampler
            )
            if sampler_for_profile is not None and int(sampler_for_profile.num_timesteps) > 0:
                log_wandb_diffusion_schedule_profile(
                    logger=self.logger,
                    sampler=sampler_for_profile,
                    total_steps=int(sampler_for_profile.num_timesteps),
                    prefix="val_imgs",
                )
                self._logged_schedule_profile_in_sanity = True

    @torch.no_grad()
    def _run_single_image_full_reconstruction_and_log(self) -> None:
        # Expensive diagnostic path: run full reverse diffusion only once per epoch
        # on a small cached validation mini-batch.
        """Helper that computes run single image full reconstruction and log.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            None: No value is returned.
        """
        if self._cached_val_example is None:
            return
        # Keep Lightning sanity check cheap: do not run the reverse diffusion chain here.
        if (
            self.skip_full_reconstruction_in_sanity_check
            and self.trainer is not None
            and self.trainer.sanity_checking
        ):
            return

        x, y, eo, valid_mask, land_mask, coords, date, loss_mask = self._cached_val_example
        denoise_samples: list[tuple[int, torch.Tensor]] = []
        sampler_for_val = None
        total_steps = 0
        pred_batch: dict[str, Any] = {
            "x": x,
            "eo": eo,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
            "coords": coords,
            "date": date,
            "sampler": self.val_sampler,
        }
        if self.log_intermediates:
            sampler_for_val = (
                self.val_sampler if self.val_sampler is not None else self.model.sampler
            )
            total_steps = int(sampler_for_val.num_timesteps)
            denoise_capture_steps = build_evenly_spaced_capture_steps(
                total_steps=total_steps,
                num_frames=16,
            )
            pred_batch["return_intermediates"] = True
            pred_batch["intermediate_step_indices"] = denoise_capture_steps

        # Use Lightning's inference path for full-validation reconstruction.
        pred = self.predict_step(pred_batch, batch_idx=0)
        y_hat = pred["y_hat"]
        y_hat_denorm = pred["y_hat_denorm"]
        y_hat_denorm_for_plot = pred.get("y_hat_denorm_for_plot", y_hat_denorm)
        denoise_samples = pred["denoise_samples"]
        x0_denoise_samples = pred.get("x0_denoise_samples", [])

        # Denormalize data channels only (masks stay in 0/1 space).
        x_denorm = temperature_normalize(mode="denorm", tensor=x)
        y_denorm = temperature_normalize(mode="denorm", tensor=y)
        eo_denorm = (
            temperature_normalize(mode="denorm", tensor=eo) if eo is not None else None
        )

        recon_batch_size = int(y_denorm.size(0))
        recon_mse = torch.mean((y_hat_denorm - y_denorm) ** 2)
        recon_psnr = None
        recon_ssim = None
        # Calculate PSNR and SSIM over samples and bands and average them,
        # if skimage is available and each slice has valid data range.
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            y_np = y_denorm.detach().float().cpu().numpy()
            y_hat_np = y_hat_denorm.detach().float().cpu().numpy()
            if y_np.ndim == 2:
                y_np = y_np[None, None, ...]
                y_hat_np = y_hat_np[None, None, ...]
            elif y_np.ndim == 3:
                y_np = y_np[:, None, ...]
                y_hat_np = y_hat_np[:, None, ...]
            psnr_vals: list[float] = []
            ssim_vals: list[float] = []
            for sample_idx in range(y_np.shape[0]):
                for band_idx in range(y_np.shape[1]):
                    y_band = y_np[sample_idx, band_idx]
                    y_hat_band = y_hat_np[sample_idx, band_idx]
                    data_range = float(y_band.max() - y_band.min())
                    if data_range <= 0.0:
                        continue
                    psnr_vals.append(
                        float(
                            peak_signal_noise_ratio(
                                y_band, y_hat_band, data_range=data_range
                            )
                        )
                    )
                    ssim_vals.append(
                        float(
                            structural_similarity(
                                y_band, y_hat_band, data_range=data_range
                            )
                        )
                    )
            if psnr_vals:
                recon_psnr = float(sum(psnr_vals) / len(psnr_vals))
            if ssim_vals:
                recon_ssim = float(sum(ssim_vals) / len(ssim_vals))
        except Exception:
            recon_psnr = None
            recon_ssim = None

        self.log(
            "val/recon_mse_full_recon",
            recon_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self._should_sync_dist(),
            batch_size=recon_batch_size,
        )
        if recon_psnr is not None:
            self.log(
                "val/recon_psnr_full_recon",
                float(recon_psnr),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=self._should_sync_dist(),
                batch_size=recon_batch_size,
            )
        if recon_ssim is not None:
            self.log(
                "val/recon_ssim_full_recon",
                float(recon_ssim),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=self._should_sync_dist(),
                batch_size=recon_batch_size,
            )
        self._log_validation_triplet_stats(x=x_denorm, y=y_denorm, y_hat=y_hat_denorm)
        self._log_common_batch_stats(
            y_hat_denorm,
            prefix="val_pred",
            batch_size=recon_batch_size,
            on_step=False,
            on_epoch=True,
        )
        self._log_common_batch_stats(
            y_denorm,
            prefix="val_target",
            batch_size=recon_batch_size,
            on_step=False,
            on_epoch=True,
        )
        # This is the one expensive full reconstruction for the epoch; always log it.
        log_wandb_conditional_reconstruction_grid(
            logger=self.logger,
            x=x_denorm,
            eo=eo_denorm,
            y_hat=y_hat_denorm_for_plot,
            y_target=y_denorm,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            land_mask=land_mask,
            prefix="val_imgs",
            image_key="x_y_full_reconstruction",
            cmap=PLOT_CMAP,
            show_valid_mask_panel=True,
            show_loss_mask_panel=True,
        )
        if self.log_intermediates and sampler_for_val is not None:
            log_wandb_denoise_timestep_grid(
                logger=self.logger,
                denoise_samples=denoise_samples,
                mae_samples=x0_denoise_samples,
                total_steps=total_steps,
                sampler=sampler_for_val,
                conditioning_image=x,
                ground_truth=y,
                valid_mask=valid_mask,
                land_mask=land_mask,
                prefix="val_imgs",
                cmap=PLOT_CMAP,
            )
        # Drop local tensor refs from this heavy validation path promptly.
        del recon_mse, y_hat, pred, pred_batch, y, x
        del y_denorm, y_hat_denorm, y_hat_denorm_for_plot, x_denorm, eo_denorm
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        # Run one full-reconstruction pass after cheap validation metrics are accumulated.
        """Compute on validation epoch end and return the result.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            None: No value is returned.
        """
        self._run_single_image_full_reconstruction_and_log()
        self._cached_val_example = None

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute training step and return the result.

        Args:
            batch (dict[str, Any]): Input value.
            batch_idx (int): Zero-based index for selecting a sample or batch.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        objective_batch = self._prepare_objective_batch(
            batch, is_validation=False, batch_idx=batch_idx
        )
        target = objective_batch["target"]
        target_t = objective_batch["target_t"]
        model_condition = objective_batch["condition"]
        valid_mask_context = objective_batch["valid_mask_context"]
        supervision_mask = objective_batch["supervision_mask"]
        mask_loss = bool(objective_batch["mask_loss"])
        land_mask = objective_batch["land_mask"]
        coords = objective_batch["coords"]
        date = objective_batch["date"]
        batch_size = int(target.size(0))
        # Log target and condition stats in the exact space seen by diffusion.
        self._log_pre_diffusion_stats(
            target_t, prefix="train_target", batch_size=batch_size
        )
        self._log_pre_diffusion_stats(
            model_condition, prefix="train_condition", batch_size=batch_size
        )
        # Conditional p_loss uses x as context while learning selected denoising target.
        loss = self.model.p_loss(
            target_t,
            model_condition,
            valid_mask=valid_mask_context,
            supervision_mask=supervision_mask,
            land_mask=land_mask,
            mask_loss=mask_loss,
            coord=coords,
            date=date,
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = self._masked_fraction(valid_mask_context)
            self.log(
                "train/masked_fraction",
                masked_fraction,
                on_step=True,
                on_epoch=False,
                logger=True,
                batch_size=batch_size,
            )
            if supervision_mask is not None:
                self.log(
                    "train/objective_supervision_fraction",
                    supervision_mask.float().mean(),
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    batch_size=batch_size,
                )

        self._log_common_batch_stats(target, prefix="train", batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute validation step and return the result.

        Args:
            batch (dict[str, Any]): Input value.
            batch_idx (int): Zero-based index for selecting a sample or batch.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        objective_batch = self._prepare_objective_batch(
            batch, is_validation=True, batch_idx=batch_idx
        )
        x_context = objective_batch["x_context"]
        target = objective_batch["target"]
        target_t = objective_batch["target_t"]
        model_condition = objective_batch["condition"]
        valid_mask_context = objective_batch["valid_mask_context"]
        supervision_mask = objective_batch["supervision_mask"]
        mask_loss = bool(objective_batch["mask_loss"])
        eo = objective_batch["eo"]
        land_mask = objective_batch["land_mask"]
        coords = objective_batch["coords"]
        date = objective_batch["date"]
        batch_size = int(target.size(0))
        # Reconstruction diagnostics still use the full y field when available.
        y = batch.get("y")
        # Log target and condition stats in the exact space seen by diffusion.
        self._log_pre_diffusion_stats(
            target_t, prefix="val_target", batch_size=batch_size
        )
        self._log_pre_diffusion_stats(
            model_condition, prefix="val_condition", batch_size=batch_size
        )
        # Same training objective for validation; full reverse-chain recon is logged at epoch end.
        loss = self.model.p_loss(
            target_t,
            model_condition,
            valid_mask=valid_mask_context,
            supervision_mask=supervision_mask,
            land_mask=land_mask,
            mask_loss=mask_loss,
            coord=coords,
            date=date,
        )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
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
            batch_size=batch_size,
        )

        if self.wandb_verbose and self.global_step % self.log_stats_every_n_steps == 0:
            masked_fraction = self._masked_fraction(valid_mask_context)
            self.log(
                "val/masked_fraction",
                masked_fraction,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            if supervision_mask is not None:
                self.log(
                    "val/objective_supervision_fraction",
                    supervision_mask.float().mean(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

        self._maybe_dump_validation_reconstruction(
            batch_idx=batch_idx,
            objective_batch=objective_batch,
            y_target=y,
        )

        if batch_idx == 0 and self._cached_val_example is None and y is not None:
            # Cache up to N validation samples from the first val batch for one
            # epoch-end full reverse-diffusion reconstruction pass.
            n_cache = min(self.max_full_reconstruction_samples, int(x_context.size(0)))
            cached_valid_mask = (
                valid_mask_context[:n_cache].detach()
                if valid_mask_context is not None
                else None
            )
            cached_land_mask = (
                land_mask[:n_cache].detach() if land_mask is not None else None
            )
            cached_loss_mask = self._loss_mask_for_logging(
                supervision_mask=supervision_mask,
                valid_mask_context=valid_mask_context,
                mask_loss=mask_loss,
            )
            if cached_loss_mask is not None:
                cached_loss_mask = cached_loss_mask[:n_cache].detach()
            cached_eo = eo[:n_cache].detach() if eo is not None else None
            cached_coords = coords[:n_cache].detach() if coords is not None else None
            cached_date = date[:n_cache].detach() if date is not None else None
            self._cached_val_example = (
                x_context[:n_cache].detach(),
                y[:n_cache].detach(),
                cached_eo,
                cached_valid_mask,
                cached_land_mask,
                cached_coords,
                cached_date,
                cached_loss_mask,
            )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        """Create optimizer and optional scheduler configuration.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            torch.optim.Optimizer | dict[str, Any]: Computed output value.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        # Keep warmup in optimizer-step space so schedule is stable across epoch lengths.
        """Perform one optimizer step with optional learning-rate warmup.

        Args:
            epoch (int): Step or timestep value.
            batch_idx (int): Zero-based index for selecting a sample or batch.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            optimizer_closure (Any | None): Optimizer used for parameter updates.

        Returns:
            None: No value is returned.
        """
        if (
            self.lr_warmup_enabled
            and self.lr_warmup_steps > 0
            and self.global_step < self.lr_warmup_steps
        ):
            warmup_progress = float(self.global_step + 1) / float(self.lr_warmup_steps)
            warmup_factor = self.lr_warmup_start_ratio + (
                1.0 - self.lr_warmup_start_ratio
            ) * warmup_progress
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(self.lr) * warmup_factor

        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_closure=optimizer_closure,
        )
