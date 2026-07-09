"""Aggregator for ambient ocean diffusion loss components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .increments import sparse_increment_loss
from .sparse_observation import sparse_observation_loss
from .spectral import SpectralEnergyFloorLoss
from .structure_function import StructureFunctionPriorLoss


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

    def _build_structure_loss(self) -> StructureFunctionPriorLoss | None:
        """Build the optional structure-function prior."""
        if not self._enabled(self.s2_cfg):
            return None
        reference_path = self.s2_cfg.get("reference_path")
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "model.losses.structure_function_prior.reference_path is required "
                "when the prior is enabled."
            )
        return StructureFunctionPriorLoss(
            reference_path=Path(str(reference_path)),
            num_pairs=int(self.s2_cfg.get("num_pairs", 8192)),
            eps=float(self.s2_cfg.get("eps", 1e-6)),
            per_depth=bool(self.s2_cfg.get("per_depth", True)),
        )

    def _build_spectral_loss(self) -> SpectralEnergyFloorLoss | None:
        """Build the optional spectral floor prior."""
        if not self._enabled(self.spectral_cfg):
            return None
        reference_path = self.spectral_cfg.get("reference_path")
        if reference_path is None or str(reference_path).strip() == "":
            raise ValueError(
                "model.losses.spectral_energy_floor.reference_path is required "
                "when the prior is enabled."
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss and detached component dictionary."""
        total = loss_ambient * self._weight(self.ambient_cfg, 1.0)
        components = {
            "loss_ambient": loss_ambient,
            "loss_obs": torch.zeros_like(loss_ambient),
            "loss_increment": torch.zeros_like(loss_ambient),
            "loss_s2_glorys": torch.zeros_like(loss_ambient),
            "loss_spectral_glorys": torch.zeros_like(loss_ambient),
        }

        if self._enabled(self.obs_cfg):
            obs_loss = sparse_observation_loss(
                x0_pred,
                obs_grid=obs_grid,
                obs_mask_grid=obs_mask_grid,
                eps=float(self.obs_cfg.get("eps", 1e-3)),
            )
            components["loss_obs"] = obs_loss
            total = total + self._weight(self.obs_cfg, 1.0) * obs_loss

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
            total = total + self._weight(self.increment_cfg, 0.5) * inc_loss

        support_mask = valid_mask
        if land_mask is not None:
            land = land_mask.to(device=x0_pred.device) > 0
            support_mask = land if support_mask is None else ((support_mask > 0) & land)

        if self.structure_function is not None:
            s2_loss = self.structure_function(x0_pred, valid_mask=support_mask)
            components["loss_s2_glorys"] = s2_loss
            total = total + self._weight(self.s2_cfg, 0.1) * s2_loss

        if self.spectral_floor is not None:
            spectral_loss = self.spectral_floor(x0_pred)
            components["loss_spectral_glorys"] = spectral_loss
            total = total + self._weight(self.spectral_cfg, 0.05) * spectral_loss

        components["loss_total"] = total
        return total, components
