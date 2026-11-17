from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import Dataset, DistributedSampler

from depth_recon.data.datamodule import DepthTileDataModule


class _IndexedDataset(Dataset):
    """Small dataset whose values make sampler duplication visible."""

    def __len__(self) -> int:
        return 101

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)


class TestGlobalBatch(unittest.TestCase):
    def _module(self, world_size: int, rank: int, **cfg: object) -> DepthTileDataModule:
        config = {"global_batch_size": 48, "num_workers": 0, **cfg}
        module = DepthTileDataModule(
            dataset=_IndexedDataset(),
            val_dataset=_IndexedDataset(),
            dataloader_cfg=config,
        )
        module.trainer = SimpleNamespace(world_size=world_size, global_rank=rank)
        return module

    def test_global_batch_is_divided_and_drops_tail_without_padding(self) -> None:
        for world_size in (1, 2, 4, 8):
            with self.subTest(world_size=world_size):
                seen: list[int] = []
                for rank in range(world_size):
                    module = self._module(world_size, rank)
                    loader = module.train_dataloader()
                    self.assertEqual(loader.batch_size, 48 // world_size)
                    self.assertTrue(loader.drop_last)
                    self.assertEqual(
                        module.resolved_train_batch_metadata(),
                        {
                            "global_batch_size": 48,
                            "local_batch_size": 48 // world_size,
                            "world_size": world_size,
                            "actual_global_batch_size": 48,
                        },
                    )
                    if world_size > 1:
                        self.assertIsInstance(loader.sampler, DistributedSampler)
                    seen.extend(int(value) for batch in loader for value in batch)
                self.assertEqual(len(seen), (101 // 48) * 48)
                self.assertEqual(len(set(seen)), len(seen))

    def test_nondivisible_global_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._module(4, 0, global_batch_size=50).train_dataloader()

    def test_legacy_batch_size_keeps_shuffle_and_tail_semantics(self) -> None:
        for world_size in (1, 2):
            module = self._module(world_size, 0, global_batch_size=None, batch_size=7)
            loader = module.train_dataloader()
            self.assertEqual(loader.batch_size, 7)
            self.assertFalse(loader.drop_last)
            # Legacy DDP sampler construction stays with Lightning unless the feature is enabled.
            self.assertNotIsInstance(loader.sampler, DistributedSampler)
            self.assertTrue(loader.sampler.__class__.__name__ == "RandomSampler")


if __name__ == "__main__":
    unittest.main()
