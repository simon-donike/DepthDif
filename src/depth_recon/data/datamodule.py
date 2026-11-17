from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Subset,
    default_collate,
    random_split,
)


def _collate_candidate_validation_batch(samples: list[object]) -> dict[str, Any]:
    """Mark batches from the optional EN4 validation dataloader."""
    batch = default_collate(samples)
    batch["_en4_candidate_eval"] = True
    return batch


class DepthTileDataModule(pl.LightningDataModule):
    """Lightning DataModule that builds train and validation dataloaders."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        val_dataset: Dataset | None = None,
        dataloader_cfg: dict[str, Any] | None = None,
        val_fraction: float = 0.2,
        seed: int = 7,
    ) -> None:
        """Initialize DepthTileDataModule with configured parameters.

        Args:
            dataset (Dataset): Input value.
            val_dataset (Dataset | None): Input value.
            dataloader_cfg (dict[str, Any] | None): Configuration dictionary or section.
            val_fraction (float): Input value.
            seed (int): Input value.

        Returns:
            None: No value is returned.
        """
        super().__init__()
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.dataloader_cfg = dataloader_cfg or {}
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)

        # When callers pass explicit train/val datasets, keep the training dataset
        # attached immediately so train_dataloader() does not depend on setup().
        self.train_dataset: Subset | Dataset | None = (
            dataset if val_dataset is not None else None
        )
        self._train_val_split_done = val_dataset is not None
        self._val_generator = torch.Generator().manual_seed(self.seed)
        self.candidate_val_dataset: Dataset | None = None
        self.candidate_val_batch_size: int | None = None

    def _world_size(self) -> int:
        """Return the active distributed world size, defaulting to one."""
        trainer = getattr(self, "trainer", None)
        value = getattr(trainer, "world_size", None)
        if value is None:
            value = self.dataloader_cfg.get("world_size", 1)
        world_size = int(value)
        if world_size < 1:
            raise ValueError("Distributed world size must be positive.")
        return world_size

    def _resolved_train_batch_size(self) -> int:
        """Resolve the per-rank batch size while preserving legacy batch_size."""
        global_batch = self.dataloader_cfg.get("global_batch_size")
        if global_batch is None:
            return int(self.dataloader_cfg.get("batch_size", 16))
        global_batch = int(global_batch)
        world_size = self._world_size()
        if global_batch < 1 or global_batch % world_size:
            raise ValueError(
                "dataloader.global_batch_size must be positive and divisible by "
                f"world_size ({world_size}); got {global_batch}."
            )
        batch_size = global_batch // world_size
        print(
            "Resolved training batch: "
            f"global={global_batch}, local={batch_size}, world_size={world_size}"
        )
        return batch_size

    def resolved_train_batch_metadata(self) -> dict[str, int] | None:
        """Return global-batch metadata using Lightning's active world size.

        This is deliberately resolved at fit startup instead of module
        construction, because the final DDP world size is not known until the
        trainer has initialized its strategy.
        """
        global_batch = self.dataloader_cfg.get("global_batch_size")
        if global_batch is None:
            return None
        local_batch = self._resolved_train_batch_size()
        world_size = self._world_size()
        return {
            "global_batch_size": int(global_batch),
            "local_batch_size": local_batch,
            "world_size": world_size,
            "actual_global_batch_size": local_batch * world_size,
        }

    def set_candidate_validation_dataset(
        self, dataset: Dataset, *, batch_size: int
    ) -> None:
        """Attach the optional EN4 candidate dataset as a second val loader."""
        self.candidate_val_dataset = dataset
        self.candidate_val_batch_size = max(1, int(batch_size))

    def setup(self, stage: str | None = None) -> None:
        # Reuse existing split when a dedicated val_dataset was provided.
        """Compute setup and return the result.

        Args:
            stage (str | None): Input value.

        Returns:
            None: No value is returned.
        """
        if self._train_val_split_done:
            return

        # Build a deterministic train/val partition from one base dataset.
        total_len = len(self.dataset)
        if total_len == 0:
            raise RuntimeError("Dataset is empty; cannot create train/val split.")

        val_len = int(round(total_len * self.val_fraction))
        if total_len > 1:
            val_len = min(
                max(val_len, 1 if self.val_fraction > 0.0 else 0), total_len - 1
            )
        else:
            val_len = 0
        train_len = total_len - val_len

        # Seeded split keeps train/val assignment stable across runs.
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_len, val_len],
            generator=generator,
        )
        self._train_val_split_done = True

    def _build_loader(
        self,
        dataset: Dataset,
        is_val: bool = False,
        *,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        collate_fn: Any | None = None,
    ) -> DataLoader:
        """Helper that computes build loader.

        Args:
            dataset (Dataset): Input value.
            is_val (bool): Boolean flag controlling behavior.

        Returns:
            DataLoader: Computed output value.
        """
        cfg = self.dataloader_cfg
        # Resolve train/val-specific overrides with sensible defaults.
        batch_size = int(
            batch_size
            if batch_size is not None
            else (
                self._resolved_train_batch_size()
                if not is_val
                else cfg.get("val_batch_size", 16)
            )
        )
        num_workers_key = "val_num_workers" if is_val else "num_workers"
        num_workers = int(cfg.get(num_workers_key, 0 if is_val else 4))
        persistent_workers = (
            bool(
                cfg.get(
                    "val_persistent_workers" if is_val else "persistent_workers",
                    False,
                )
            )
            and num_workers > 0
        )
        pin_memory = bool(cfg.get("pin_memory", True))
        # Default to shuffling both train and validation unless explicitly disabled.
        shuffle = bool(
            cfg.get("val_shuffle" if is_val else "shuffle", True)
            if shuffle is None
            else shuffle
        )
        prefetch_factor = cfg.get("prefetch_factor", 2)
        kwargs: dict[str, Any] = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if (
            not is_val
            and cfg.get("global_batch_size") is not None
            and self._world_size() > 1
        ):
            # drop_last prevents both partial local batches and DistributedSampler
            # padding, which would otherwise duplicate samples across ranks.
            kwargs["sampler"] = DistributedSampler(
                dataset,
                num_replicas=self._world_size(),
                rank=int(getattr(getattr(self, "trainer", None), "global_rank", 0)),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=True,
            )
            kwargs["shuffle"] = False
        if not is_val and self.dataloader_cfg.get("global_batch_size") is not None:
            kwargs["drop_last"] = True
        if is_val and shuffle:
            # Keep validation shuffled while advancing the order across validation runs.
            kwargs["generator"] = self._val_generator
        # prefetch_factor is only valid when worker processes are enabled.
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        multiprocessing_context = cfg.get("multiprocessing_context")
        if num_workers > 0 and multiprocessing_context:
            kwargs["multiprocessing_context"] = str(multiprocessing_context)
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader from the attached datamodule."""
        if not self._train_val_split_done:
            self.setup("fit")
        return self._build_loader(self.train_dataset, is_val=False)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader from the attached datamodule."""
        if not self._train_val_split_done:
            self.setup("fit")
        # Lightning sanity checking: force single-worker if trainer requests it (handled in trainer configs)
        main_loader = self._build_loader(self.val_dataset, is_val=True)
        if self.candidate_val_dataset is None:
            return main_loader
        candidate_loader = self._build_loader(
            self.candidate_val_dataset,
            is_val=True,
            batch_size=self.candidate_val_batch_size,
            shuffle=False,
            collate_fn=_collate_candidate_validation_batch,
        )
        return [main_loader, candidate_loader]
