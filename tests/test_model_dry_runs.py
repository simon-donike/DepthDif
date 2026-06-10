from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from typing import Any

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import yaml

from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.inference.core import build_model, model_requires_checkpoint
from depth_recon.models.baselines import (
    IDWInterpolationBaseline,
    PointwiseLSTMBaseline,
    UNetInfillingBaseline,
)
from depth_recon.utils.normalizations import salinity_normalize

matplotlib.use("Agg")
os.environ.setdefault("WANDB_MODE", "disabled")


class _StaticBatchDataset(Dataset):
    """Small deterministic dataset for baseline dry runs."""

    def __init__(
        self,
        *,
        length: int = 2,
        channels: int = 2,
        size: int = 8,
        include_eo: bool = False,
        include_salinity: bool = False,
    ) -> None:
        """Initialize the static dataset."""
        self.length = int(length)
        self.channels = int(channels)
        self.size = int(size)
        self.include_eo = bool(include_eo)
        self.include_salinity = bool(include_salinity)
        self.depth_axis_m = np.linspace(0.0, 100.0, num=self.channels).astype(
            np.float32
        )

    def __len__(self) -> int:
        """Return the dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one deterministic sparse/dense training sample."""
        offset = float(idx) * 0.05
        y = torch.linspace(
            -0.4 + offset,
            0.6 + offset,
            steps=self.channels * self.size * self.size,
            dtype=torch.float32,
        ).reshape(self.channels, self.size, self.size)
        x = y.clone()
        x[:, ::2, ::2] = 0.0
        x_valid_mask = torch.ones_like(y, dtype=torch.bool)
        x_valid_mask[:, ::2, ::2] = False
        y_valid_mask = torch.ones_like(y, dtype=torch.bool)
        y_valid_mask[:, -1, -1] = False
        sample: dict[str, Any] = {
            "x": x,
            "y": y,
            "x_valid_mask": x_valid_mask,
            "y_valid_mask": y_valid_mask,
            "x_valid_mask_1d": x_valid_mask.any(dim=0, keepdim=True),
            "land_mask": y_valid_mask.any(dim=0, keepdim=True).float(),
            "coords": torch.tensor([10.0, 20.0], dtype=torch.float32),
            "date": 20240115,
        }
        if self.include_eo:
            sample["eo"] = torch.full(
                (1, self.size, self.size), 0.25 + offset, dtype=torch.float32
            )
        if self.include_salinity:
            salinity_psu = torch.linspace(
                33.5 + offset,
                35.5 + offset,
                steps=self.channels * self.size * self.size,
                dtype=torch.float32,
            ).reshape(self.channels, self.size, self.size)
            y_salinity = salinity_normalize(mode="norm", tensor=salinity_psu)
            x_salinity = y_salinity.clone()
            x_salinity[:, 1::2, 1::2] = 0.0
            x_salinity_valid_mask = torch.ones_like(y_salinity, dtype=torch.bool)
            x_salinity_valid_mask[:, 1::2, 1::2] = False
            y_salinity_valid_mask = torch.ones_like(y_salinity, dtype=torch.bool)
            y_salinity_valid_mask[:, -1, 0] = False
            sample.update(
                {
                    "x_salinity": x_salinity,
                    "y_salinity": y_salinity,
                    "x_salinity_valid_mask": x_salinity_valid_mask,
                    "y_salinity_valid_mask": y_salinity_valid_mask,
                    "x_salinity_valid_mask_1d": x_salinity_valid_mask.any(
                        dim=0, keepdim=True
                    ),
                }
            )
        return sample


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write one YAML fixture file."""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _make_datamodule(
    *, channels: int = 2, include_eo: bool = False, include_salinity: bool = False
) -> DepthTileDataModule:
    """Build a deterministic one-batch datamodule."""
    train_dataset = _StaticBatchDataset(
        length=2,
        channels=channels,
        include_eo=include_eo,
        include_salinity=include_salinity,
    )
    val_dataset = _StaticBatchDataset(
        length=1,
        channels=channels,
        include_eo=include_eo,
        include_salinity=include_salinity,
    )
    return DepthTileDataModule(
        dataset=train_dataset,
        val_dataset=val_dataset,
        dataloader_cfg={
            "batch_size": 1,
            "val_batch_size": 1,
            "num_workers": 0,
            "val_num_workers": 0,
            "shuffle": False,
            "val_shuffle": False,
            "pin_memory": False,
        },
    )


def _trainer_kwargs(tmp_path: Path) -> dict[str, Any]:
    """Return Lightning trainer settings for a one-batch CPU fit."""
    return {
        "default_root_dir": str(tmp_path),
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
    }


def _make_pixel_batch(*, include_salinity: bool = False) -> dict[str, Any]:
    """Return one collated batch from the static dataset."""
    sample = _StaticBatchDataset(
        length=1, channels=2, include_eo=False, include_salinity=include_salinity
    )[0]
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0) if value.ndim >= 1 else value.view(1)
        else:
            batch[key] = torch.tensor([value])
    return batch


class TestBaselineDryRuns(unittest.TestCase):
    """Dry-run tests for the extracted baseline models."""

    def setUp(self) -> None:
        """Seed deterministic torch/numpy operations."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_idw_predict_step_returns_contract(self) -> None:
        model = IDWInterpolationBaseline(output_fields=("temperature",))
        batch = _make_pixel_batch()

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), tuple(batch["x"].shape))
        self.assertEqual(tuple(pred["y_hat_denorm"].shape), tuple(batch["x"].shape))
        self.assertIn("y_hat_temperature_denorm", pred)
        self.assertEqual(pred["denoise_samples"], [])
        self.assertIsNone(pred["sampler"])

    def test_idw_empty_argo_patch_returns_nan(self) -> None:
        model = IDWInterpolationBaseline(output_fields=("temperature",))
        batch = _make_pixel_batch()
        batch["x_valid_mask"] = torch.zeros_like(batch["x_valid_mask"])

        pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.isnan(pred["y_hat"]).all())

    def test_lstm_predict_step_returns_contract(self) -> None:
        model = PointwiseLSTMBaseline(
            hidden_size=4,
            num_layers=1,
            include_eo=True,
            depth_axis_m=(0.0, 10.0),
            output_fields=("temperature",),
        )
        batch = _make_pixel_batch()
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), tuple(batch["x"].shape))
        self.assertIn("y_hat_temperature_denorm", pred)
        self.assertEqual(pred["denoise_samples"], [])

    def test_lstm_trainer_fit_completes_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PointwiseLSTMBaseline(
                hidden_size=4,
                num_layers=1,
                include_eo=True,
                output_fields=("temperature",),
            )
            trainer = pl.Trainer(**_trainer_kwargs(Path(tmpdir)))

            trainer.fit(model, datamodule=_make_datamodule(include_eo=True))

    def test_unet_baseline_predict_step_returns_contract(self) -> None:
        model = UNetInfillingBaseline(
            generated_channels=2,
            base_channels=4,
            channel_mults=(1,),
            norm_groups=2,
            condition_include_eo=True,
            condition_use_valid_mask=True,
            condition_use_land_mask=True,
            output_fields=("temperature",),
        )
        batch = _make_pixel_batch()
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), tuple(batch["x"].shape))
        self.assertEqual(tuple(pred["y_hat_denorm"].shape), tuple(batch["x"].shape))
        self.assertIn("y_hat_temperature_denorm", pred)
        self.assertEqual(pred["denoise_samples"], [])
        self.assertIsNone(pred["sampler"])

    def test_unet_baseline_empty_argo_patch_returns_nan(self) -> None:
        model = UNetInfillingBaseline(
            generated_channels=2,
            base_channels=4,
            channel_mults=(1,),
            norm_groups=2,
            condition_include_eo=True,
            condition_use_valid_mask=True,
            condition_use_land_mask=True,
            output_fields=("temperature",),
        )
        batch = _make_pixel_batch()
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)
        batch["x_valid_mask"] = torch.zeros_like(batch["x_valid_mask"])

        pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.isnan(pred["y_hat"]).all())

    def test_unet_baseline_joint_outputs_split_fields(self) -> None:
        model = UNetInfillingBaseline(
            generated_channels=4,
            base_channels=4,
            channel_mults=(1,),
            norm_groups=2,
            condition_include_eo=True,
            condition_use_valid_mask=True,
            condition_use_land_mask=True,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), (1, 4, 8, 8))
        self.assertEqual(tuple(pred["y_hat_temperature"].shape), (1, 2, 8, 8))
        self.assertEqual(tuple(pred["y_hat_salinity"].shape), (1, 2, 8, 8))
        self.assertIn("y_hat_temperature_denorm", pred)
        self.assertIn("y_hat_salinity_denorm", pred)

    def test_build_model_factory_selects_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            _write_yaml(data_config_path, {"dataset": {}})
            _write_yaml(training_config_path, {"training": {"lr": 2.0e-3}})

            for model_type, expected_cls in (
                ("idw_baseline", IDWInterpolationBaseline),
                ("lstm_baseline", PointwiseLSTMBaseline),
                ("unet_baseline", UNetInfillingBaseline),
            ):
                with self.subTest(model_type=model_type):
                    model_cfg = {
                        "model": {
                            "model_type": model_type,
                            "output_fields": ["temperature"],
                            "scenario": "temperature",
                            "depth_channels": 2,
                            "generated_channels": 2,
                            "condition_include_eo": True,
                            "condition_use_valid_mask": True,
                            "condition_use_land_mask": True,
                            "condition_mask_channels": 1,
                            "condition_channels": 4,
                            "lstm": {"hidden_size": 4, "num_layers": 1, "lr": None},
                            "unet_baseline": {
                                "base_channels": 4,
                                "channel_mults": [1],
                                "norm_groups": 2,
                                "lr": None,
                            },
                        }
                    }
                    _write_yaml(model_config_path, model_cfg)

                    model = build_model(
                        model_config_path=str(model_config_path),
                        data_config_path=str(data_config_path),
                        training_config_path=str(training_config_path),
                        model_cfg=model_cfg,
                        datamodule=_make_datamodule(include_eo=True),
                    )

                    self.assertIsInstance(model, expected_cls)

    def test_model_checkpoint_requirement_flags(self) -> None:
        self.assertFalse(
            model_requires_checkpoint({"model": {"model_type": "idw_baseline"}})
        )
        self.assertTrue(
            model_requires_checkpoint({"model": {"model_type": "lstm_baseline"}})
        )
        self.assertTrue(
            model_requires_checkpoint({"model": {"model_type": "unet_baseline"}})
        )


if __name__ == "__main__":
    unittest.main()
