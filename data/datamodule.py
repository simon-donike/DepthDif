from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data.dataset import DepthTileDataset


class DepthTileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config_path: str = "configs/data_config.yaml",
        val_fraction: float = 0.2,
        rebuild_index: bool = False,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.rebuild_index = rebuild_index
        self._cfg = self._load_config(config_path)

        ds_cfg = self._cfg["dataset"]
        dl_cfg = self._cfg["dataloader"]

        self.batch_size = int(dl_cfg.get("batch_size", 16))
        self.num_workers = int(dl_cfg.get("num_workers", 4))
        self.pin_memory = bool(dl_cfg.get("pin_memory", True))
        self.shuffle = bool(dl_cfg.get("shuffle", True))

        # Reuse dataset seed so split is reproducible across runs.
        self.seed = int(ds_cfg.get("random_seed", 7))
        self.val_fraction = float(val_fraction)

        self.dataset: Dataset | None = None
        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = DepthTileDataset(
                config_path=self.config_path,
                rebuild_index=self.rebuild_index,
            )

        if self.train_dataset is not None and self.val_dataset is not None:
            return

        total_len = len(self.dataset)
        if total_len == 0:
            raise RuntimeError("DepthTileDataset is empty; cannot create train/val split.")

        # 20% validation split, clamped so both splits are non-empty when possible.
        val_len = int(round(total_len * self.val_fraction))
        if total_len > 1:
            val_len = min(max(val_len, 1), total_len - 1)
        else:
            val_len = 0
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_len, val_len],
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":
    # Example usage
    data_module = DepthTileDataModule(config_path="configs/data_config.yaml")
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")