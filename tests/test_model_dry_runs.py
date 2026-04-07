from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import yaml

from data.datamodule import DepthTileDataModule
from models.difFF.PixelDiffusion import PixelDiffusionConditional
from models.latent.Autoencoder import (
    DepthBandAutoencoder,
    DepthBandAutoencoderLightning,
)
from models.latent.LatentDiffusion import LatentDiffusionConditional


matplotlib.use("Agg")
os.environ.setdefault("WANDB_MODE", "disabled")


class _StaticBatchDataset(Dataset):
    def __init__(
        self,
        *,
        length: int = 2,
        channels: int = 2,
        size: int = 8,
        include_eo: bool = False,
    ) -> None:
        self.length = int(length)
        self.channels = int(channels)
        self.size = int(size)
        self.include_eo = bool(include_eo)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
            sample["eo"] = torch.full((1, self.size, self.size), 0.25 + offset, dtype=torch.float32)
        return sample


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _make_datamodule(*, channels: int = 2, include_eo: bool = False) -> DepthTileDataModule:
    train_dataset = _StaticBatchDataset(length=2, channels=channels, include_eo=include_eo)
    val_dataset = _StaticBatchDataset(length=1, channels=channels, include_eo=include_eo)
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


def _make_pixel_model(**overrides: object) -> PixelDiffusionConditional:
    kwargs = dict(
        generated_channels=2,
        condition_channels=3,
        condition_mask_channels=1,
        condition_include_eo=False,
        condition_use_valid_mask=True,
        mask_loss_with_valid_pixels=True,
        parameterization="x0",
        num_timesteps=2,
        noise_schedule="linear",
        unet_dim=8,
        unet_dim_mults=(1,),
        lr=1.0e-3,
        wandb_verbose=False,
        log_intermediates=False,
        val_inference_sampler="ddim",
        val_ddim_num_timesteps=2,
        max_full_reconstruction_samples=1,
        skip_full_reconstruction_in_sanity_check=True,
    )
    kwargs.update(overrides)
    return PixelDiffusionConditional(**kwargs)


def _make_pixel_batch() -> dict[str, torch.Tensor]:
    sample = _StaticBatchDataset(length=1, channels=2, include_eo=False)[0]
    batch: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0) if value.ndim >= 1 else value.view(1)
        else:
            batch[key] = torch.tensor([value])
    return batch


class TestModelDryRuns(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

    def test_pixel_diffusion_from_config_wires_nested_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            _write_yaml(
                model_config_path,
                {
                    "model": {
                        "generated_channels": 2,
                        "condition_channels": 3,
                        "condition_mask_channels": 1,
                        "condition_include_eo": False,
                        "condition_use_valid_mask": True,
                        "mask_loss_with_valid_pixels": True,
                        "parameterization": "x0",
                        "ambient_occlusion": {
                            "enabled": True,
                            "further_drop_prob": 0.25,
                            "apply_to_noisy_branch": False,
                            "shared_spatial_mask": False,
                            "min_kept_observed_pixels": 3,
                            "require_x0_parameterization": True,
                        },
                        "post_process": {
                            "gaussian_blur": {"enabled": True, "sigma": 0.7, "kernel_size": 4}
                        },
                        "coord_conditioning": {"enabled": False},
                        "unet": {"dim": 8, "dim_mults": [1], "with_time_emb": True},
                    }
                },
            )
            _write_yaml(data_config_path, {"dataset": {}, "dataloader": {"batch_size": 9}})
            _write_yaml(
                training_config_path,
                {
                    "training": {
                        "lr": 2.0e-4,
                        "batch_size": 5,
                        "noise": {
                            "num_timesteps": 6,
                            "schedule": "sigmoid",
                            "beta_start": 0.001,
                            "beta_end": 0.05,
                        },
                        "validation_sampling": {
                            "sampler": "ddim",
                            "ddim_num_timesteps": 3,
                            "ddim_eta": 0.2,
                            "log_intermediates": False,
                            "skip_full_reconstruction_in_sanity_check": True,
                            "max_full_reconstruction_samples": 1,
                        },
                    },
                    "wandb": {
                        "verbose": False,
                        "log_stats_every_n_steps": 7,
                        "log_images_every_n_steps": 11,
                    },
                    "dataloader": {"batch_size": 4},
                    "scheduler": {
                        "warmup": {"enabled": True, "steps": 12, "start_ratio": 0.3},
                        "reduce_on_plateau": {
                            "enabled": True,
                            "monitor": "val/custom",
                            "mode": "max",
                            "factor": 0.6,
                            "patience": 4,
                            "threshold": 0.2,
                            "threshold_mode": "abs",
                            "cooldown": 2,
                            "min_lr": 1.0e-6,
                            "eps": 1.0e-9,
                        },
                    },
                },
            )

            model = PixelDiffusionConditional.from_config(
                str(model_config_path),
                str(data_config_path),
                str(training_config_path),
            )
            optim_config = model.configure_optimizers()

            self.assertTrue(model.ambient_occlusion_enabled)
            self.assertEqual(model.ambient_further_drop_prob, 0.25)
            self.assertFalse(model.ambient_apply_to_noisy_branch)
            self.assertFalse(model.ambient_shared_spatial_mask)
            self.assertEqual(model.ambient_min_kept_observed_pixels, 3)
            self.assertEqual(model.lr, 2.0e-4)
            self.assertEqual(model.batch_size, 5)
            self.assertTrue(model.lr_scheduler_enabled)
            self.assertTrue(model.lr_warmup_enabled)
            self.assertEqual(model.lr_warmup_steps, 12)
            self.assertEqual(model.lr_warmup_start_ratio, 0.3)
            self.assertTrue(model.postprocess_gaussian_blur_enabled)
            # Even kernel sizes are promoted to the next odd size in __init__.
            self.assertEqual(model.postprocess_gaussian_blur_kernel_size, 5)
            self.assertEqual(model.model.forward_process.num_timesteps, 6)
            self.assertEqual(model.model.parameterization, "x0")
            self.assertEqual(model.model.condition_channels, 3)
            self.assertEqual(model.val_sampler.num_timesteps, 3)
            self.assertIsInstance(optim_config, dict)
            self.assertEqual(optim_config["lr_scheduler"]["monitor"], "val/custom")

    def test_pixel_training_step_uses_standard_target_and_passes_land_mask(self) -> None:
        model = _make_pixel_model()
        batch = _make_pixel_batch()
        captured: dict[str, Any] = {}

        def fake_p_loss(output: torch.Tensor, condition: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(1.25, requires_grad=True)

        with patch.object(model.model, "p_loss", fake_p_loss), patch.object(
            model, "log", lambda *args, **kwargs: None
        ):
            loss = model.training_step(batch, batch_idx=0)

        expected_condition = model._prepare_condition_for_model(
            batch["x"],
            batch["x_valid_mask"],
        )
        expected_loss_mask = model._build_standard_loss_mask(
            reference=batch["y"],
            y_valid_mask=batch["y_valid_mask"],
        )
        self.assertTrue(torch.equal(captured["output"], batch["y"]))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask))
        self.assertTrue(torch.equal(captured["kwargs"]["land_mask"], batch["land_mask"]))
        self.assertTrue(torch.isclose(loss, torch.tensor(1.25)))

    def test_pixel_validation_step_uses_ambient_target_and_intersection_mask(self) -> None:
        model = _make_pixel_model(
            ambient_occlusion_enabled=True,
            ambient_further_drop_prob=0.0,
            ambient_apply_to_noisy_branch=True,
        )
        batch = _make_pixel_batch()
        further_mask = batch["x_valid_mask"].clone()
        further_mask[:, :, 0, 1] = False
        captured: dict[str, Any] = {}

        def fake_p_loss(output: torch.Tensor, condition: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(0.75)

        with patch.object(model.model, "p_loss", fake_p_loss), patch.object(
            model,
            "_build_ambient_further_valid_mask",
            lambda valid_mask, reference: further_mask,
        ), patch.object(model, "log", lambda *args, **kwargs: None):
            loss = model.validation_step(batch, batch_idx=0)

        expected_condition = model._prepare_condition_for_model(
            batch["x"] * further_mask,
            further_mask,
        )
        expected_loss_mask = model._build_ambient_loss_mask(
            reference=batch["x"],
            x_valid_mask=batch["x_valid_mask"],
            y_valid_mask=batch["y_valid_mask"],
        )
        self.assertTrue(torch.equal(captured["output"], batch["x"]))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask))
        self.assertTrue(torch.equal(captured["kwargs"]["further_valid_mask"], further_mask))
        self.assertTrue(captured["kwargs"]["apply_further_corruption_to_noisy_branch"])
        self.assertTrue(torch.equal(captured["kwargs"]["land_mask"], batch["land_mask"]))
        self.assertTrue(torch.isclose(loss, torch.tensor(0.75)))

    def test_pixel_diffusion_completes_one_training_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            datamodule = _make_datamodule(channels=2, include_eo=False)
            model = _make_pixel_model(datamodule=datamodule)
            trainer = pl.Trainer(**_trainer_kwargs(tmp_path))

            trainer.fit(model, datamodule=datamodule)

            self.assertEqual(trainer.global_step, 1)

    def test_latent_diffusion_completes_one_training_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            datamodule = _make_datamodule(channels=2, include_eo=False)
            autoencoder = DepthBandAutoencoder(
                in_channels=2,
                latent_channels=1,
                encoder_hidden_channels=(4,),
                decoder_hidden_channels=(4,),
                spatial_downsample=1,
            )
            model = LatentDiffusionConditional(
                autoencoder=autoencoder,
                autoencoder_frozen=False,
                eo_in_pixel_space=True,
                datamodule=datamodule,
                generated_channels=1,
                condition_channels=2,
                condition_mask_channels=1,
                condition_include_eo=False,
                condition_use_valid_mask=True,
                mask_loss_with_valid_pixels=True,
                parameterization="x0",
                num_timesteps=2,
                noise_schedule="linear",
                unet_dim=8,
                unet_dim_mults=(1,),
                lr=1.0e-3,
                wandb_verbose=False,
                log_intermediates=False,
                val_inference_sampler="ddim",
                val_ddim_num_timesteps=2,
                max_full_reconstruction_samples=1,
                skip_full_reconstruction_in_sanity_check=True,
            )
            trainer = pl.Trainer(**_trainer_kwargs(tmp_path))

            trainer.fit(model, datamodule=datamodule)

            self.assertEqual(trainer.global_step, 1)

    def test_autoencoder_lightning_completes_one_training_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            datamodule = _make_datamodule(channels=2, include_eo=False)
            autoencoder = DepthBandAutoencoder(
                in_channels=2,
                latent_channels=1,
                encoder_hidden_channels=(4,),
                decoder_hidden_channels=(4,),
                spatial_downsample=2,
            )
            model = DepthBandAutoencoderLightning(
                autoencoder=autoencoder,
                datamodule=datamodule,
                lr=1.0e-3,
                batch_size=1,
                recon_l1_weight=1.0,
                recon_l2_weight=0.5,
                masked_only=True,
            )
            trainer = pl.Trainer(**_trainer_kwargs(tmp_path))

            trainer.fit(model, datamodule=datamodule)

            self.assertEqual(trainer.global_step, 1)

    def test_autoencoder_from_configs_wires_loss_and_scheduler_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ae_config_path = tmp_path / "ae.yaml"
            training_config_path = tmp_path / "training.yaml"
            _write_yaml(
                ae_config_path,
                {
                    "ae": {
                        "in_channels": 2,
                        "latent_channels": 1,
                        "spatial_downsample": 2,
                        "encoder": {"hidden_channels": [4]},
                        "decoder": {"hidden_channels": [4]},
                        "training": {"lr": 3.0e-4, "batch_size": 6},
                        "loss": {
                            "recon_l1_weight": 1.2,
                            "recon_l2_weight": 0.3,
                            "masked_only": True,
                        },
                    }
                },
            )
            _write_yaml(
                training_config_path,
                {
                    "training": {"lr": 1.0e-4},
                    "dataloader": {"batch_size": 2},
                    "scheduler": {
                        "reduce_on_plateau": {
                            "enabled": True,
                            "monitor": "val/loss_ckpt",
                            "mode": "min",
                            "factor": 0.5,
                            "patience": 3,
                            "threshold": 1.0e-4,
                            "threshold_mode": "rel",
                            "cooldown": 1,
                            "min_lr": 1.0e-6,
                            "eps": 1.0e-8,
                        }
                    },
                },
            )

            model = DepthBandAutoencoderLightning.from_configs(
                ae_config_path=str(ae_config_path),
                training_config_path=str(training_config_path),
            )
            optim_config = model.configure_optimizers()

            self.assertEqual(model.lr, 3.0e-4)
            self.assertEqual(model.batch_size, 6)
            self.assertEqual(model.recon_l1_weight, 1.2)
            self.assertEqual(model.recon_l2_weight, 0.3)
            self.assertTrue(model.masked_only)
            self.assertTrue(model.lr_scheduler_enabled)
            self.assertIsInstance(optim_config, dict)
            self.assertEqual(optim_config["lr_scheduler"]["monitor"], "val/loss_ckpt")


if __name__ == "__main__":
    unittest.main()
