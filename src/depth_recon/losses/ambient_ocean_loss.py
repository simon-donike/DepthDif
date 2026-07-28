"""Aggregator for ambient ocean diffusion loss components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .increments import sparse_increment_loss
from .sparse_observation import sparse_observation_loss
from .spectral import PairedSpectralEnergyFloorLoss, SpectralEnergyFloorLoss
from .structure_function import PairedStructureFunctionLoss, StructureFunctionPriorLoss
from .timestep_weighting import aux_timestep_weight


class AmbientOceanLoss(nn.Module):
    """Combine base diffusion loss with optional sparse and GLORYS-prior terms."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize loss components from ``model.losses`` config."""
        super().__init__()
        self.config = dict(config or {})
        self.ambient_cfg = dict(self.config.get("ambient", {}))
        self.obs_cfg = dict(self.config.get("sparse_observation", {}))
        self.increment_cfg = dict(self.config.get("increment", {}))
        self.s2_cfg = dict(self.config.get("structure_function_prior", {}))
        self.spectral_cfg = dict(self.config.get("spectral_energy_floor", {}))
        self.feature_cfg = dict(self.config.get("feature_gram_prior", {}))
        self.aux_timestep_weighting_cfg = dict(
            self.config.get("aux_timestep_weighting", {})
        )
        if self._enabled(self.feature_cfg):
            raise NotImplementedError(
                "model.losses.feature_gram_prior is reserved but not implemented."
            )
        self.structure_function = self._build_structure_loss()
        self.spectral_floor = self._build_spectral_loss()

    @staticmethod
    def _enabled(cfg: dict[str, Any]) -> bool:
        """Return whether a loss component is enabled."""
        return bool(cfg.get("enabled", False))

    @staticmethod
    def _weight(cfg: dict[str, Any], default: float) -> float:
        """Return a scalar component weight."""
        return float(cfg.get("weight", default))

    @staticmethod
    def _target_mode(cfg: dict[str, Any]) -> str:
        """Return the auxiliary target mode."""
        mode = str(cfg.get("target", "reference")).strip().lower()
        if mode not in {"reference", "paired_glorys"}:
            raise ValueError(
                "model.losses.*.target must be one of {'reference', 'paired_glorys'} "
                f"(got {mode!r})."
            )
        return mode

    def _build_structure_loss(
        self,
    ) -> StructureFunctionPriorLoss | PairedStructureFunctionLoss | None:
        """Build the optional structure-function loss."""
        if not self._enabled(self.s2_cfg):
            return None
        if self._target_mode(self.s2_cfg) == "paired_glorys":
            return PairedStructureFunctionLoss(
                num_pairs=int(self.s2_cfg.get("num_pairs", 8192)),
                num_bins=int(self.s2_cfg.get("num_bins", 16)),
                eps=float(self.s2_cfg.get("eps", 1e-6)),
                per_depth=bool(self.s2_cfg.get("per_depth", True)),
            )
        reference_path = self.s2_cfg.get("reference_path")
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "model.losses.structure_function_prior.reference_path is required "
                "when the prior is enabled in reference mode."
            )
        return StructureFunctionPriorLoss(
            reference_path=Path(str(reference_path)),
            num_pairs=int(self.s2_cfg.get("num_pairs", 8192)),
            eps=float(self.s2_cfg.get("eps", 1e-6)),
            per_depth=bool(self.s2_cfg.get("per_depth", True)),
        )

    def _build_spectral_loss(
        self,
    ) -> SpectralEnergyFloorLoss | PairedSpectralEnergyFloorLoss | None:
        """Build the optional spectral floor loss."""
        if not self._enabled(self.spectral_cfg):
            return None
        if self._target_mode(self.spectral_cfg) == "paired_glorys":
            return PairedSpectralEnergyFloorLoss(
                eps=float(self.spectral_cfg.get("eps", 1e-8)),
                margin=float(self.spectral_cfg.get("margin", 0.0)),
                min_band=int(self.spectral_cfg.get("min_band", 1)),
                max_band=self.spectral_cfg.get("max_band", None),
                per_depth=bool(self.spectral_cfg.get("per_depth", True)),
            )
        reference_path = self.spectral_cfg.get("reference_path")
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "model.losses.spectral_energy_floor.reference_path is required "
                "when the prior is enabled in reference mode."
            )
        return SpectralEnergyFloorLoss(
            reference_path=Path(str(reference_path)),
            eps=float(self.spectral_cfg.get("eps", 1e-8)),
            margin=float(self.spectral_cfg.get("margin", 0.0)),
            min_band=int(self.spectral_cfg.get("min_band", 1)),
            max_band=self.spectral_cfg.get("max_band", None),
            per_depth=bool(self.spectral_cfg.get("per_depth", True)),
        )

    def any_extra_enabled(self) -> bool:
        """Return whether any non-base loss component is enabled."""
        return any(
            [
                self._enabled(self.obs_cfg),
                self._enabled(self.increment_cfg),
                self._enabled(self.s2_cfg),
                self._enabled(self.spectral_cfg),
                self._enabled(self.feature_cfg),
            ]
        )

    def forward(
        self,
        *,
        loss_ambient: torch.Tensor,
        x0_pred: torch.Tensor,
        obs_grid: torch.Tensor | None = None,
        obs_mask_grid: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        land_mask: torch.Tensor | None = None,
        target_grid: torch.Tensor | None = None,
        target_mask_grid: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        alphas_cumprod: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss and detached component dictionary."""
        ambient_weighted = loss_ambient * self._weight(self.ambient_cfg, 1.0)
        total = ambient_weighted
        aux_total = torch.zeros_like(loss_ambient)
        aux_weight = (
            aux_timestep_weight(
                self.aux_timestep_weighting_cfg,
                t=t,
                alphas_cumprod=alphas_cumprod,
                reference=loss_ambient,
            )
            if self.any_extra_enabled()
            else torch.ones_like(loss_ambient)
        )
        components = {
            "loss_ambient": loss_ambient,
            "loss_ambient_weighted": ambient_weighted,
            "loss_obs": torch.zeros_like(loss_ambient),
            "loss_obs_weighted": torch.zeros_like(loss_ambient),
            "loss_increment": torch.zeros_like(loss_ambient),
            "loss_increment_weighted": torch.zeros_like(loss_ambient),
            "loss_s2_glorys": torch.zeros_like(loss_ambient),
            "loss_s2_glorys_weighted": torch.zeros_like(loss_ambient),
            "loss_spectral_glorys": torch.zeros_like(loss_ambient),
            "loss_spectral_glorys_weighted": torch.zeros_like(loss_ambient),
            "loss_aux_timestep_weight": aux_weight,
            "loss_aux_static_weighted": torch.zeros_like(loss_ambient),
            "loss_aux_timestep_weighted": torch.zeros_like(loss_ambient),
        }

        if self._enabled(self.obs_cfg):
            obs_loss = sparse_observation_loss(
                x0_pred,
                obs_grid=obs_grid,
                obs_mask_grid=obs_mask_grid,
                eps=float(self.obs_cfg.get("eps", 1e-3)),
            )
            components["loss_obs"] = obs_loss
            obs_weighted = self._weight(self.obs_cfg, 1.0) * obs_loss
            components["loss_obs_weighted"] = obs_weighted
            aux_total = aux_total + obs_weighted

        if self._enabled(self.increment_cfg):
            inc_loss = sparse_increment_loss(
                x0_pred,
                obs_grid=obs_grid,
                obs_mask_grid=obs_mask_grid,
                eps=float(self.increment_cfg.get("eps", 1e-3)),
                max_pairs_per_sample=int(
                    self.increment_cfg.get("max_pairs_per_sample", 4096)
                ),
                vertical_pairs=bool(self.increment_cfg.get("vertical_pairs", True)),
                horizontal_pairs=bool(
                    self.increment_cfg.get("horizontal_pairs", False)
                ),
                horizontal_max_distance=self.increment_cfg.get(
                    "horizontal_max_distance", None
                ),
            )
            components["loss_increment"] = inc_loss
            inc_weighted = self._weight(self.increment_cfg, 0.5) * inc_loss
            components["loss_increment_weighted"] = inc_weighted
            aux_total = aux_total + inc_weighted

        support_mask = target_mask_grid if target_mask_grid is not None else valid_mask
        if support_mask is not None:
            support_mask = support_mask.to(device=x0_pred.device) > 0
            if support_mask.ndim == 3:
                support_mask = support_mask.unsqueeze(1)
        if land_mask is not None:
            land = land_mask.to(device=x0_pred.device) > 0
            if land.ndim == 3:
                land = land.unsqueeze(1)
            support_mask = land if support_mask is None else (support_mask & land)

        if self.structure_function is not None:
            if self._target_mode(self.s2_cfg) == "paired_glorys":
                if target_grid is None:
                    raise RuntimeError(
                        "paired_glorys structure loss requires target_grid."
                    )
                s2_loss = self.structure_function(
                    x0_pred, target_grid=target_grid, valid_mask=support_mask
                )
            else:
                s2_loss = self.structure_function(x0_pred, valid_mask=support_mask)
            components["loss_s2_glorys"] = s2_loss
            s2_weighted = self._weight(self.s2_cfg, 0.1) * s2_loss
            components["loss_s2_glorys_weighted"] = s2_weighted
            aux_total = aux_total + s2_weighted

        if self.spectral_floor is not None:
            if self._target_mode(self.spectral_cfg) == "paired_glorys":
                if target_grid is None:
                    raise RuntimeError(
                        "paired_glorys spectral loss requires target_grid."
                    )
                spectral_loss = self.spectral_floor(
                    x0_pred, target_grid=target_grid, valid_mask=support_mask
                )
            else:
                spectral_loss = self.spectral_floor(x0_pred)
            components["loss_spectral_glorys"] = spectral_loss
            spectral_weighted = self._weight(self.spectral_cfg, 0.05) * spectral_loss
            components["loss_spectral_glorys_weighted"] = spectral_weighted
            aux_total = aux_total + spectral_weighted

        aux_timestep_weighted = aux_weight * aux_total
        total = total + aux_timestep_weighted
        components["loss_aux_static_weighted"] = aux_total
        components["loss_aux_timestep_weighted"] = aux_timestep_weighted
        components["loss_total"] = total
        return total, components
