from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data.dataset import SurfaceTempPatchDataset
from data.dataset_light import SurfaceTempPatchLightDataset


class DepthTileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config_path: str = "configs/data_config.yaml",
        training_config_path: str = "configs/training_config.yaml",
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.training_config_path = training_config_path
        self._cfg_data = self._load_config(config_path)
        self._cfg_training = self._load_config(training_config_path)

        dl_cfg = self._cfg_training.get("dataloader", {})
        split_cfg = self._cfg_data.get("split", {})
        ds_cfg = self._cfg_data["dataset"]
        self.dataloader_type = str(ds_cfg.get("dataloader_type", "raw")).strip().lower()
        valid_loader_types = {"raw", "light"}
        if self.dataloader_type not in valid_loader_types:
            raise ValueError(
                f"dataset.dataloader_type must be one of {sorted(valid_loader_types)} "
                f"(got '{self.dataloader_type}')."
            )

        self.batch_size = int(dl_cfg.get("batch_size", 16))
        self.val_batch_size = int(dl_cfg.get("val_batch_size", self.batch_size))
        self.num_workers = int(dl_cfg.get("num_workers", 4))
        self.val_num_workers = int(dl_cfg.get("val_num_workers", self.num_workers))
        self.persistent_workers = bool(dl_cfg.get("persistent_workers", False))
        self.val_persistent_workers = bool(
            dl_cfg.get("val_persistent_workers", self.persistent_workers)
        )
        prefetch_factor_cfg = dl_cfg.get("prefetch_factor", 2)
        self.prefetch_factor = (
            None if prefetch_factor_cfg is None else int(prefetch_factor_cfg)
        )
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError("dataloader.prefetch_factor must be >= 1 or null.")
        self.pin_memory = bool(dl_cfg.get("pin_memory", True))
        self.shuffle = bool(dl_cfg.get("shuffle", True))
        self.val_shuffle = bool(dl_cfg.get("val_shuffle", True))

        self.val_fraction = float(split_cfg.get("val_fraction", 0.2))
        self.seed = int(ds_cfg.get("random_seed", 7))

        self.dataset: Dataset | None = None
        self.train_dataset: Subset | None = None
        self.val_dataset: Subset | None = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None or self.val_dataset is None:
            if self.dataloader_type == "light":
                self.train_dataset = SurfaceTempPatchLightDataset.from_config(
                    self.config_path, split="train"
                )
                self.val_dataset = SurfaceTempPatchLightDataset.from_config(
                    self.config_path, split="val"
                )
                self.dataset = None
            else:
                if self.dataset is None:
                    self.dataset = SurfaceTempPatchDataset.from_config(self.config_path)

                total_len = len(self.dataset)
                if total_len == 0:
                    raise RuntimeError("Dataset is empty; cannot create train/val split.")

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
        print(
            f"Created datamodule of type '{self.dataloader_type}' with "
            f"{len(self.train_dataset)} train and {len(self.val_dataset)} val images."
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        num_workers = self.val_num_workers
        # HDF5-backed NetCDF reads can deadlock in multi-worker validation on some setups.
        # Sanity checking only needs a couple of batches, so force single-process loading there.
        if self.trainer is not None and self.trainer.sanity_checking:
            num_workers = 0
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0 and self.val_persistent_workers,
        )
        if num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)


if __name__ == "__main__":
    dm = DepthTileDataModule(config_path="configs/data_config.yaml")
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for batch in train_loader:
        print(batch)
        break

    for batch in val_loader:
        print(batch)
        break
