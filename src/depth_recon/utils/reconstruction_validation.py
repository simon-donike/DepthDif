"""Fixed, distributed full reconstructions for training checkpoint selection."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, default_collate

from depth_recon.configs.config_resolver_pixel import FULL_RECONSTRUCTION_MONITOR
from depth_recon.utils.normalizations import (
    SALINITY_STD,
    Y_STD,
    salinity_normalize,
    temperature_normalize,
)


@contextmanager
def _evaluation_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed evaluation without advancing training's Python, NumPy or torch RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # Seed only the active device; other ranks' CUDA generators are untouched.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[device.index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class FullReconstructionValidation(pl.Callback):
    """Score a fixed global patch set before ModelCheckpoint's validation-end hook."""

    def __init__(
        self,
        *,
        dataset: Dataset | None = None,
        sample_count: int = 128,
        batch_size: int = 4,
        seed: int = 7,
        output_dir: str | Path | None = None,
    ) -> None:
        """Configure bounded reconstruction batches, independent of preview logging."""
        super().__init__()
        if sample_count < 1 or batch_size < 1 or seed < 0:
            raise ValueError(
                "Reconstruction sample_count/batch_size must be positive and seed nonnegative."
            )
        self.dataset = dataset
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.patch_indices: list[int] | None = None

    def _select_patches(self, trainer: pl.Trainer) -> None:
        """Select unique validation indices without touching the shuffled loader."""
        if self.patch_indices is not None:
            return
        if self.dataset is None:
            self.dataset = trainer.datamodule.val_dataset
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError(
                "Full reconstruction checkpoint selection requires a nonempty validation dataset."
            )
        self.patch_indices = (
            np.random.default_rng(self.seed)
            .choice(
                len(self.dataset),
                size=min(self.sample_count, len(self.dataset)),
                replace=False,
            )
            .tolist()
        )
        if self.output_dir is not None and trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "full_reconstruction_selection.json").write_text(
                json.dumps(
                    {
                        "dataset_size": len(self.dataset),
                        "requested_sample_count": self.sample_count,
                        "sample_count": len(self.patch_indices),
                        "batch_size": self.batch_size,
                        "seed": self.seed,
                        "patch_indices": self.patch_indices,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def local_batches(
        self, rank: int, world_size: int
    ) -> Iterator[tuple[int, list[int]]]:
        """Shard whole fixed batches without padding or repeating any patches."""
        if self.patch_indices is None:
            raise RuntimeError("Reconstruction patches have not been selected.")
        for batch_index, offset in enumerate(
            range(0, len(self.patch_indices), self.batch_size)
        ):
            if batch_index % world_size == rank:
                yield batch_index, self.patch_indices[offset : offset + self.batch_size]

    @staticmethod
    def _target_keys(field: str) -> tuple[str, str]:
        """Return the existing validation target and its dated support mask."""
        if field == "temperature":
            return "y", "y_valid_mask"
        if field == "salinity":
            return "y_salinity", "y_salinity_valid_mask"
        raise ValueError(f"Unsupported reconstruction field: {field!r}.")

    @staticmethod
    def _mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Broadcast spatial support while rejecting mismatched depth masks."""
        mask = mask.to(device=reference.device) > 0.5
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if (
            mask.ndim != 4
            or mask.shape[0] != reference.shape[0]
            or mask.shape[2:] != reference.shape[2:]
            or mask.shape[1] not in (1, reference.shape[1])
        ):
            raise ValueError("Reconstruction mask does not match the target shape.")
        return mask.expand_as(reference)

    @classmethod
    def _batch_statistics(
        cls,
        batch: dict[str, Any],
        prediction: dict[str, Any],
        field: str,
        *,
        single_field: bool,
    ) -> torch.Tensor:
        """Return per-depth squared error, absolute error and valid counts in physical units."""
        target_key, mask_key = cls._target_keys(field)
        target_norm = batch[target_key].float()
        denormalize = (
            temperature_normalize if field == "temperature" else salinity_normalize
        )
        target = denormalize(mode="denorm", tensor=target_norm)
        predicted = prediction.get(f"y_hat_{field}_denorm")
        if predicted is None and single_field:
            predicted = prediction.get("y_hat_denorm")
        if not torch.is_tensor(predicted) or predicted.shape != target.shape:
            raise ValueError(
                f"Full reconstruction requires a matching {field} prediction."
            )
        support = cls._mask(batch[mask_key], target)
        if torch.is_tensor(batch.get("land_mask")):
            support = support & cls._mask(batch["land_mask"], target)
        # Invalid predictions on valid water must worsen the score, never disappear
        # through a finite-value filter. Invalid/dry reference cells contribute zero.
        finite = torch.isfinite(target) & torch.isfinite(predicted)
        error = torch.where(finite, predicted.double() - target.double(), torch.inf)
        error = torch.where(support, error, 0.0)
        axes = (0, 2, 3)
        return torch.stack(
            (
                error.square().sum(axes),
                error.abs().sum(axes),
                support.double().sum(axes),
            )
        )

    @staticmethod
    def _metrics(statistics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pool pixels before averaging depth RMSEs and normalized field scores."""
        metrics: dict[str, torch.Tensor] = {}
        scores = []
        for field, stats in statistics.items():
            squared, absolute, counts = stats
            supported = counts > 0
            if not bool(supported.any()):
                raise ValueError(
                    f"No valid {field} targets in the reconstruction subset; increase sample_count."
                )
            depth_rmse = torch.sqrt(squared[supported] / counts[supported])
            prefix = f"val/full_reconstruction/{field}"
            metrics[f"{prefix}_rmse"] = torch.sqrt(squared.sum() / counts.sum())
            metrics[f"{prefix}_mae"] = absolute.sum() / counts.sum()
            metrics[f"{prefix}_equal_depth_rmse"] = depth_rmse.mean()
            metrics[f"{prefix}_equal_depth_mae"] = (
                absolute[supported] / counts[supported]
            ).mean()
            metrics[f"{prefix}_valid_values"] = counts.sum()
            metrics[f"{prefix}_supported_depths"] = supported.sum().double()
            # Equal field weighting avoids adding degrees Celsius directly to PSU.
            scores.append(
                depth_rmse.mean() / (Y_STD if field == "temperature" else SALINITY_STD)
            )
        metrics[FULL_RECONSTRUCTION_MONITOR] = torch.stack(scores).mean()
        return metrics

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reconstruct every selected patch and globally reduce errors before saving."""
        if trainer.sanity_checking:
            return
        self._select_patches(trainer)
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        statistics = {}
        # Every rank creates the same accumulator layout, including ranks with no
        # assigned batches. Only sufficient statistics cross the distributed boundary.
        with _evaluation_rng(pl_module.device, self.seed):
            example = self.dataset[self.patch_indices[0]]
        for field in fields:
            target_key, _ = self._target_keys(field)
            statistics[field] = torch.zeros(
                (3, example[target_key].shape[0]),
                dtype=torch.float64,
                device=pl_module.device,
            )
        for batch_index, indices in self.local_batches(
            trainer.global_rank, trainer.world_size
        ):
            # Global batch seeds keep stochastic sampling repeatable across checks
            # and GPU counts, without changing the main training RNG sequence.
            with _evaluation_rng(pl_module.device, self.seed + batch_index):
                batch = default_collate([self.dataset[index] for index in indices])
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                batch["return_intermediates"] = False
                with trainer.precision_plugin.forward_context():
                    prediction = pl_module.predict_step(batch, batch_idx=batch_index)
                for field in fields:
                    statistics[field] += self._batch_statistics(
                        batch, prediction, field, single_field=len(fields) == 1
                    )
                del prediction, batch
        for field in fields:
            statistics[field] = trainer.strategy.reduce(
                statistics[field], reduce_op="sum"
            )
        metrics = self._metrics(statistics)
        metrics["val/full_reconstruction/sample_count"] = torch.tensor(
            float(len(self.patch_indices)), device=pl_module.device
        )
        # All ranks already hold the same globally pooled metric. A second mean
        # reduction would hide weighting bugs and is unnecessary here.
        pl_module.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            logger=True,
            batch_size=1,
        )
