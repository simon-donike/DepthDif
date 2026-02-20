from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


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

        self.train_dataset: Subset | Dataset | None = None
        self._train_val_split_done = val_dataset is not None

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
            val_len = min(max(val_len, 1 if self.val_fraction > 0.0 else 0), total_len - 1)
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

    def _build_loader(self, dataset: Dataset, is_val: bool = False) -> DataLoader:
        """Helper that computes build loader.

        Args:
            dataset (Dataset): Input value.
            is_val (bool): Boolean flag controlling behavior.

        Returns:
            DataLoader: Computed output value.
        """
        cfg = self.dataloader_cfg
        # Resolve train/val-specific overrides with sensible defaults.
        batch_size = int(cfg.get("val_batch_size" if is_val else "batch_size", 16))
        num_workers_key = "val_num_workers" if is_val else "num_workers"
        num_workers = int(cfg.get(num_workers_key, 0 if is_val else 4))
        persistent_workers = bool(
            cfg.get(
                "val_persistent_workers" if is_val else "persistent_workers",
                False,
            )
        ) and num_workers > 0
        pin_memory = bool(cfg.get("pin_memory", True))
        # Default to shuffling both train and validation unless explicitly disabled.
        shuffle = bool(cfg.get("val_shuffle" if is_val else "shuffle", True))
        prefetch_factor = cfg.get("prefetch_factor", 2)
        kwargs: dict[str, Any] = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # prefetch_factor is only valid when worker processes are enabled.
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader from the attached datamodule.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            DataLoader: Computed output value.
        """
        if not self._train_val_split_done:
            self.setup("fit")
        return self._build_loader(self.train_dataset, is_val=False)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader from the attached datamodule.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            DataLoader: Computed output value.
        """
        if not self._train_val_split_done:
            self.setup("fit")
        # Lightning sanity checking: force single-worker if trainer requests it (handled in trainer configs)
        return self._build_loader(self.val_dataset, is_val=True)
