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

from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.inference.core import build_model, model_requires_checkpoint
from depth_recon.models.baselines import IDWInterpolationBaseline, PointwiseLSTMBaseline
from depth_recon.models.diffusion.PixelDiffusion import PixelDiffusionConditional
from depth_recon.models.latent.Autoencoder import (
    DepthBandAutoencoder,
    DepthBandAutoencoderLightning,
)
from depth_recon.models.latent.LatentDiffusion import LatentDiffusionConditional
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.validation_denoise import average_observed_argo_pixels_per_image

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
        include_salinity: bool = False,
    ) -> None:
        self.length = int(length)
        self.channels = int(channels)
        self.size = int(size)
        self.include_eo = bool(include_eo)
        self.include_salinity = bool(include_salinity)

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
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _make_datamodule(
    *, channels: int = 2, include_eo: bool = False, include_salinity: bool = False
) -> DepthTileDataModule:
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


def _make_pixel_batch(*, include_salinity: bool = False) -> dict[str, torch.Tensor]:
    sample = _StaticBatchDataset(
        length=1, channels=2, include_eo=False, include_salinity=include_salinity
    )[0]
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

    def test_observed_argo_pixels_count_spatial_locations_once(self) -> None:
        batch = _make_pixel_batch()
        observed_pixels = average_observed_argo_pixels_per_image(batch["x_valid_mask"])

        self.assertTrue(torch.isclose(observed_pixels, torch.tensor(48.0)))

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
                        "condition_channels": 4,
                        "condition_mask_channels": 1,
                        "condition_include_eo": False,
                        "condition_use_valid_mask": True,
                        "condition_use_land_mask": True,
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
                            "gaussian_blur": {
                                "enabled": True,
                                "sigma": 0.7,
                                "kernel_size": 4,
                            }
                        },
                        "coord_conditioning": {"enabled": False},
                        "unet": {"dim": 8, "dim_mults": [1], "with_time_emb": True},
                    }
                },
            )
            _write_yaml(
                data_config_path, {"dataset": {}, "dataloader": {"batch_size": 9}}
            )
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
                            "ddim_temperature": 0.5,
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
                            "interval": "steps",
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
            self.assertEqual(model.lr_scheduler_interval, "step")
            self.assertTrue(model.postprocess_gaussian_blur_enabled)
            # Even kernel sizes are promoted to the next odd size in __init__.
            self.assertEqual(model.postprocess_gaussian_blur_kernel_size, 5)
            self.assertEqual(model.model.forward_process.num_timesteps, 6)
            self.assertEqual(model.model.parameterization, "x0")
            self.assertTrue(model.condition_use_land_mask)
            self.assertEqual(model.model.condition_channels, 4)
            self.assertEqual(model.val_sampler.num_timesteps, 3)
            self.assertEqual(model.val_sampler.temperature, 0.5)
            self.assertIsInstance(optim_config, dict)
            self.assertEqual(optim_config["lr_scheduler"]["monitor"], "val/loss_ckpt")
            self.assertEqual(optim_config["lr_scheduler"]["interval"], "step")
            self.assertFalse(optim_config["lr_scheduler"]["strict"])

    def test_pixel_diffusion_from_config_wires_output_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            _write_yaml(
                model_config_path,
                {
                    "model": {
                        "generated_channels": 4,
                        "condition_channels": 5,
                        "output_fields": ["temperature", "salinity"],
                        "condition_mask_channels": 1,
                        "condition_include_eo": False,
                        "condition_use_valid_mask": True,
                        "mask_loss_with_valid_pixels": True,
                        "parameterization": "x0",
                        "unet": {"dim": 8, "dim_mults": [1]},
                    }
                },
            )
            _write_yaml(data_config_path, {"dataset": {}, "dataloader": {}})
            _write_yaml(training_config_path, {"training": {}, "wandb": {}})

            model = PixelDiffusionConditional.from_config(
                str(model_config_path),
                str(data_config_path),
                str(training_config_path),
            )

            self.assertEqual(model.output_fields, ("temperature", "salinity"))
            self.assertTrue(model.predicts_salinity)

    def test_idw_baseline_predict_step_interpolates_temperature(self) -> None:
        model = IDWInterpolationBaseline(power=2.0, output_fields=("temperature",))
        x = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        x[:, :, 0, 0] = 1.0
        x[:, :, 0, 2] = 3.0
        x_valid_mask = torch.zeros_like(x, dtype=torch.bool)
        x_valid_mask[:, :, 0, 0] = True
        x_valid_mask[:, :, 0, 2] = True
        batch = {
            "x": x,
            "x_valid_mask": x_valid_mask,
            "y_valid_mask": torch.ones_like(x, dtype=torch.bool),
            "land_mask": torch.ones((1, 1, 3, 3), dtype=torch.float32),
        }

        pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.equal(pred["y_hat"][:, :, 0, 0], x[:, :, 0, 0]))
        self.assertTrue(torch.equal(pred["y_hat"][:, :, 0, 2], x[:, :, 0, 2]))
        self.assertTrue(
            torch.allclose(
                pred["y_hat"][:, :, 0, 1],
                torch.tensor([[2.0]], dtype=torch.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                pred["y_hat"][:, :, 2, 2],
                torch.tensor([[2.3333333]], dtype=torch.float32),
                atol=1e-5,
            )
        )

    def test_idw_baseline_empty_argo_patch_returns_nan(self) -> None:
        model = IDWInterpolationBaseline(power=2.0, output_fields=("temperature",))
        x = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        batch = {
            "x": x,
            "x_valid_mask": torch.zeros_like(x, dtype=torch.bool),
            "y_valid_mask": torch.ones_like(x, dtype=torch.bool),
            "land_mask": torch.ones((1, 1, 3, 3), dtype=torch.float32),
        }

        pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.isnan(pred["y_hat"]).all())
        self.assertTrue(torch.isnan(pred["y_hat_denorm"]).all())

    def test_idw_baseline_from_factory_needs_no_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            model_cfg = {
                "model": {
                    "model_type": "idw_baseline",
                    "output_fields": ["temperature"],
                    "scenario": "temperature",
                    "idw": {"power": 1.5, "chunk_size": 2},
                }
            }
            _write_yaml(model_config_path, model_cfg)
            _write_yaml(data_config_path, {"dataset": {}})
            _write_yaml(training_config_path, {"training": {}})

            model = build_model(
                model_config_path=str(model_config_path),
                data_config_path=str(data_config_path),
                training_config_path=str(training_config_path),
                model_cfg=model_cfg,
                datamodule=_make_datamodule(),
            )

        self.assertIsInstance(model, IDWInterpolationBaseline)
        self.assertEqual(model.power, 1.5)
        self.assertFalse(model_requires_checkpoint(model_cfg))

    def test_lstm_baseline_predict_step_returns_contract(self) -> None:
        model = PointwiseLSTMBaseline(
            hidden_size=4,
            num_layers=1,
            bidirectional=False,
            output_fields=("temperature",),
            depth_axis_m=[0.0, 10.0],
        )
        batch = _make_pixel_batch()
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), tuple(batch["x"].shape))
        self.assertEqual(tuple(pred["y_hat_denorm"].shape), tuple(batch["x"].shape))
        self.assertEqual(
            tuple(pred["y_hat_temperature_denorm"].shape), tuple(batch["x"].shape)
        )
        self.assertIn("y_hat_denorm_for_plot", pred)
        self.assertEqual(pred["denoise_samples"], [])
        self.assertEqual(pred["x0_denoise_samples"], [])
        self.assertIsNone(pred["sampler"])
        self.assertIsNone(pred["further_valid_mask"])

    def test_lstm_baseline_pointwise_independence(self) -> None:
        model = PointwiseLSTMBaseline(
            hidden_size=4,
            num_layers=1,
            bidirectional=False,
            output_fields=("temperature",),
            depth_axis_m=[0.0, 10.0],
        )
        x = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        x[:, :, 0, 0] = torch.tensor([0.2, -0.1])
        x[:, :, 1, 1] = torch.tensor([0.2, -0.1])
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :, 0, 0] = True
        mask[:, :, 1, 1] = True
        eo = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

        with torch.no_grad():
            pred = model(x, mask, eo)

        self.assertTrue(torch.allclose(pred[:, :, 0, 0], pred[:, :, 1, 1]))

    def test_lstm_baseline_empty_argo_patch_returns_nan(self) -> None:
        model = PointwiseLSTMBaseline(
            hidden_size=4,
            num_layers=1,
            bidirectional=False,
            output_fields=("temperature",),
            depth_axis_m=[0.0, 10.0],
        )
        batch = _make_pixel_batch()
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)
        batch["x_valid_mask"] = torch.zeros_like(batch["x_valid_mask"])

        pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.isnan(pred["y_hat"]).all())
        self.assertTrue(torch.isnan(pred["y_hat_denorm"]).all())

    def test_lstm_baseline_from_factory_requires_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            model_cfg = {
                "model": {
                    "model_type": "lstm_baseline",
                    "output_fields": ["temperature"],
                    "scenario": "temperature",
                    "condition_include_eo": True,
                    "lstm": {"hidden_size": 4, "num_layers": 1},
                }
            }
            _write_yaml(model_config_path, model_cfg)
            _write_yaml(data_config_path, {"dataset": {}})
            _write_yaml(training_config_path, {"training": {"lr": 2.0e-3}})

            model = build_model(
                model_config_path=str(model_config_path),
                data_config_path=str(data_config_path),
                training_config_path=str(training_config_path),
                model_cfg=model_cfg,
                datamodule=_make_datamodule(include_eo=True),
            )

        self.assertIsInstance(model, PointwiseLSTMBaseline)
        self.assertEqual(model.lr, 2.0e-3)
        self.assertTrue(model_requires_checkpoint(model_cfg))

    def test_lstm_baseline_trainer_fit_completes_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PointwiseLSTMBaseline(
                hidden_size=4,
                num_layers=1,
                bidirectional=False,
                output_fields=("temperature",),
                depth_axis_m=[0.0, 10.0],
            )
            trainer = pl.Trainer(**_trainer_kwargs(Path(tmpdir)))

            trainer.fit(model, datamodule=_make_datamodule(include_eo=True))

    def test_lstm_baseline_joint_outputs_split_fields(self) -> None:
        model = PointwiseLSTMBaseline(
            hidden_size=4,
            num_layers=1,
            bidirectional=False,
            output_fields=("temperature", "salinity"),
            depth_axis_m=[0.0, 10.0],
        )
        batch = _make_pixel_batch(include_salinity=True)
        batch["eo"] = torch.full((1, 1, 8, 8), 0.25, dtype=torch.float32)

        pred = model.predict_step(batch, batch_idx=0)

        self.assertEqual(tuple(pred["y_hat"].shape), (1, 4, 8, 8))
        self.assertEqual(
            tuple(pred["y_hat_temperature"].shape), tuple(batch["x"].shape)
        )
        self.assertEqual(
            tuple(pred["y_hat_salinity"].shape), tuple(batch["x_salinity"].shape)
        )
        self.assertIn("y_hat_temperature_denorm", pred)
        self.assertIn("y_hat_salinity_denorm", pred)

    def test_latent_diffusion_from_config_wires_land_mask_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ae_config_path = tmp_path / "ae.yaml"
            model_config_path = tmp_path / "model.yaml"
            data_config_path = tmp_path / "data.yaml"
            training_config_path = tmp_path / "training.yaml"
            _write_yaml(
                ae_config_path,
                {
                    "ae": {
                        "in_channels": 2,
                        "latent_channels": 1,
                        "spatial_downsample": 1,
                        "encoder": {"hidden_channels": [4]},
                        "decoder": {"hidden_channels": [4]},
                    }
                },
            )
            _write_yaml(
                model_config_path,
                {
                    "model": {
                        "generated_channels": 1,
                        "condition_channels": 3,
                        "condition_mask_channels": 1,
                        "condition_include_eo": False,
                        "condition_use_valid_mask": True,
                        "condition_use_land_mask": True,
                        "mask_loss_with_valid_pixels": True,
                        "parameterization": "x0",
                        "latent": {
                            "ae_config_path": str(ae_config_path),
                            "ae_checkpoint": False,
                            "freeze_autoencoder": True,
                            "latent_channels": 1,
                            "spatial_downsample": 1,
                            "eo_in_pixel_space": True,
                        },
                        "unet": {"dim": 8, "dim_mults": [1]},
                    }
                },
            )
            _write_yaml(data_config_path, {"dataset": {}, "dataloader": {}})
            _write_yaml(training_config_path, {"training": {}, "wandb": {}})

            model = LatentDiffusionConditional.from_config(
                str(model_config_path),
                str(data_config_path),
                str(training_config_path),
            )

            self.assertTrue(model.condition_use_land_mask)
            self.assertEqual(model.model.condition_channels, 3)

    def test_pixel_condition_includes_glorys_land_mask_when_enabled(self) -> None:
        model = _make_pixel_model(
            condition_channels=4,
            condition_use_land_mask=True,
        )
        batch = _make_pixel_batch()

        condition = model._prepare_condition_for_model(
            batch["x"],
            batch["x_valid_mask"],
            land_mask=batch["land_mask"],
        )

        self.assertEqual(condition.shape[1], 4)
        self.assertTrue(
            torch.equal(condition[:, -1:], batch["land_mask"].to(condition.dtype))
        )

    def test_pixel_training_step_uses_standard_target_and_passes_land_mask(
        self,
    ) -> None:
        model = _make_pixel_model(wandb_verbose=True)
        batch = _make_pixel_batch()
        batch["output_land_mask"] = torch.zeros_like(batch["land_mask"])
        captured: dict[str, Any] = {}
        logged_names: list[str] = []

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(1.25, requires_grad=True)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(
                model, "log", lambda *args, **kwargs: logged_names.append(args[0])
            ),
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
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertTrue(
            torch.equal(captured["kwargs"]["land_mask"], batch["land_mask"])
        )
        self.assertFalse(
            torch.equal(captured["kwargs"]["land_mask"], batch["output_land_mask"])
        )
        self.assertIn("train/argo_observed_pixels_per_image", logged_names)
        self.assertTrue(torch.isclose(loss, torch.tensor(1.25)))

    def test_pixel_training_step_stacks_joint_temperature_salinity(self) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        captured: dict[str, Any] = {}

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(1.5, requires_grad=True)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(model, "log", lambda *args, **kwargs: None),
        ):
            loss = model.training_step(batch, batch_idx=0)

        expected_x = torch.cat([batch["x"], batch["x_salinity"]], dim=1)
        expected_y = torch.cat([batch["y"], batch["y_salinity"]], dim=1)
        expected_x_mask = torch.cat(
            [batch["x_valid_mask"], batch["x_salinity_valid_mask"]], dim=1
        )
        expected_y_mask = torch.cat(
            [batch["y_valid_mask"], batch["y_salinity_valid_mask"]], dim=1
        )
        expected_condition = model._prepare_condition_for_model(
            expected_x, expected_x_mask
        )
        expected_loss_mask = model._build_standard_loss_mask(
            reference=expected_y, y_valid_mask=expected_y_mask
        )

        self.assertTrue(torch.equal(captured["output"], expected_y))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertTrue(torch.isclose(loss, torch.tensor(1.5)))

    def test_pixel_training_step_uses_salinity_only_fields(self) -> None:
        model = _make_pixel_model(output_fields=("salinity",))
        batch = _make_pixel_batch(include_salinity=True)
        captured: dict[str, Any] = {}

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(1.75, requires_grad=True)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(model, "log", lambda *args, **kwargs: None),
        ):
            loss = model.training_step(batch, batch_idx=0)

        expected_condition = model._prepare_condition_for_model(
            batch["x_salinity"], batch["x_salinity_valid_mask"]
        )
        expected_loss_mask = model._build_standard_loss_mask(
            reference=batch["y_salinity"],
            y_valid_mask=batch["y_salinity_valid_mask"],
        )

        self.assertTrue(torch.equal(captured["output"], batch["y_salinity"]))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertTrue(torch.isclose(loss, torch.tensor(1.75)))

    def test_pixel_validation_step_uses_salinity_only_batch_without_temperature_keys(
        self,
    ) -> None:
        model = _make_pixel_model(output_fields=("salinity",))
        batch = _make_pixel_batch(include_salinity=True)
        for key in ("x", "y", "x_valid_mask", "y_valid_mask", "x_valid_mask_1d"):
            batch.pop(key)
        captured: dict[str, Any] = {}

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(1.25)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(model, "log", lambda *args, **kwargs: None),
        ):
            loss = model.validation_step(batch, batch_idx=0)

        expected_condition = model._prepare_condition_for_model(
            batch["x_salinity"], batch["x_salinity_valid_mask"]
        )
        expected_loss_mask = model._build_standard_loss_mask(
            reference=batch["y_salinity"],
            y_valid_mask=batch["y_salinity_valid_mask"],
        )

        self.assertTrue(torch.equal(captured["output"], batch["y_salinity"]))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertIsNotNone(model._cached_val_example)
        self.assertIsNone(model._cached_val_example["x"])
        self.assertIsNone(model._cached_val_example["y"])
        self.assertIn("x_salinity", model._cached_val_example)
        self.assertTrue(torch.isclose(loss, torch.tensor(1.25)))

    def test_pixel_predict_step_returns_salinity_only_outputs(self) -> None:
        model = _make_pixel_model(output_fields=("salinity",))
        batch = _make_pixel_batch(include_salinity=True)
        batch["y_salinity_valid_mask"] = torch.ones_like(batch["y_salinity_valid_mask"])
        batch["land_mask"] = torch.ones_like(batch["land_mask"])
        generated_salinity_psu = torch.full_like(batch["x_salinity"], 34.75)
        generated_norm = salinity_normalize(mode="norm", tensor=generated_salinity_psu)

        with patch.object(model, "forward", lambda *args, **kwargs: generated_norm):
            pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.equal(pred["y_hat"], generated_norm))
        self.assertTrue(torch.equal(pred["y_hat_salinity"], generated_norm))
        self.assertTrue(
            torch.allclose(
                pred["y_hat_salinity_denorm"], generated_salinity_psu, atol=1e-5
            )
        )
        self.assertTrue(
            torch.equal(pred["y_hat_denorm"], pred["y_hat_salinity_denorm"])
        )
        self.assertNotIn("y_hat_temperature", pred)

    def test_salinity_only_requires_salinity_batch_keys(self) -> None:
        model = _make_pixel_model(output_fields=("salinity",))
        batch = _make_pixel_batch(include_salinity=False)

        with self.assertRaisesRegex(RuntimeError, r"requires batch\[x_salinity\]"):
            model.training_step(batch, batch_idx=0)

    def test_pixel_validation_step_stacks_joint_temperature_salinity_masks(
        self,
    ) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        captured: dict[str, Any] = {}

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(0.5)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(model, "log", lambda *args, **kwargs: None),
        ):
            loss = model.validation_step(batch, batch_idx=0)

        expected_y = torch.cat([batch["y"], batch["y_salinity"]], dim=1)
        expected_x_mask = torch.cat(
            [batch["x_valid_mask"], batch["x_salinity_valid_mask"]], dim=1
        )
        expected_y_mask = torch.cat(
            [batch["y_valid_mask"], batch["y_salinity_valid_mask"]], dim=1
        )
        expected_condition = model._prepare_condition_for_model(
            torch.cat([batch["x"], batch["x_salinity"]], dim=1),
            expected_x_mask,
        )
        expected_loss_mask = model._build_standard_loss_mask(
            reference=expected_y, y_valid_mask=expected_y_mask
        )

        self.assertTrue(torch.equal(captured["output"], expected_y))
        self.assertTrue(torch.equal(captured["condition"], expected_condition))
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertIsNotNone(model._cached_val_example)
        self.assertIn("x_salinity", model._cached_val_example)
        self.assertNotIn("output_land_mask", model._cached_val_example)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.5)))

    def test_pixel_validation_step_uses_ambient_target_and_intersection_mask(
        self,
    ) -> None:
        model = _make_pixel_model(
            ambient_occlusion_enabled=True,
            ambient_further_drop_prob=0.0,
            ambient_apply_to_noisy_branch=True,
            wandb_verbose=True,
        )
        batch = _make_pixel_batch()
        further_mask = batch["x_valid_mask"].clone()
        further_mask[:, :, 0, 1] = False
        captured: dict[str, Any] = {}
        logged_names: list[str] = []

        def fake_p_loss(
            output: torch.Tensor, condition: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor:
            captured["output"] = output.detach().clone()
            captured["condition"] = condition.detach().clone()
            captured["kwargs"] = kwargs
            return torch.tensor(0.75)

        with (
            patch.object(model.model, "p_loss", fake_p_loss),
            patch.object(
                model,
                "_build_ambient_further_valid_mask",
                lambda valid_mask, reference: further_mask,
            ),
            patch.object(
                model, "log", lambda *args, **kwargs: logged_names.append(args[0])
            ),
        ):
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
        self.assertTrue(
            torch.equal(captured["kwargs"]["loss_mask"], expected_loss_mask)
        )
        self.assertTrue(
            torch.equal(captured["kwargs"]["further_valid_mask"], further_mask)
        )
        self.assertTrue(captured["kwargs"]["apply_further_corruption_to_noisy_branch"])
        self.assertTrue(
            torch.equal(captured["kwargs"]["land_mask"], batch["land_mask"])
        )
        self.assertIn("val/argo_observed_pixels_per_image", logged_names)
        self.assertTrue(torch.isclose(loss, torch.tensor(0.75)))

    def test_pixel_predict_step_zeroes_land_only_when_land_mask_present(
        self,
    ) -> None:
        model = _make_pixel_model()
        batch = _make_pixel_batch()
        batch["y_valid_mask"] = torch.ones_like(batch["y_valid_mask"])
        land_mask = torch.ones_like(batch["land_mask"])
        land_mask[..., 1, 0] = 0.0
        output_land_mask = torch.ones_like(batch["land_mask"])
        output_land_mask[..., 0, 1] = 0.0
        batch["land_mask"] = land_mask
        batch["output_land_mask"] = output_land_mask
        generated_celsius = torch.full_like(batch["x"], 5.0)
        generated_norm = temperature_normalize(
            mode="norm",
            tensor=generated_celsius,
        )

        with patch.object(model, "forward", lambda *args, **kwargs: generated_norm):
            masked = model.predict_step(batch, batch_idx=0)
            without_land_mask_batch = dict(batch)
            without_land_mask_batch.pop("land_mask")
            without_land_mask_batch.pop("output_land_mask")
            unmasked = model.predict_step(without_land_mask_batch, batch_idx=0)

        for row, col in ((1, 0), (0, 1)):
            self.assertTrue(
                torch.equal(
                    masked["y_hat_denorm"][:, :, row, col],
                    torch.zeros_like(masked["y_hat_denorm"][:, :, row, col]),
                )
            )
            self.assertTrue(
                torch.equal(
                    masked["y_hat_denorm_for_plot"][:, :, row, col],
                    torch.zeros_like(masked["y_hat_denorm_for_plot"][:, :, row, col]),
                )
            )
        self.assertTrue(
            torch.allclose(
                masked["y_hat_denorm"][:, :, 0, 0],
                torch.full_like(masked["y_hat_denorm"][:, :, 0, 0], 5.0),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                unmasked["y_hat_denorm"][:, :, 0, 1],
                torch.full_like(unmasked["y_hat_denorm"][:, :, 0, 1], 5.0),
                atol=1e-5,
            )
        )

    def test_pixel_predict_step_splits_joint_temperature_salinity_outputs(
        self,
    ) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        batch["y_valid_mask"] = torch.ones_like(batch["y_valid_mask"])
        batch["y_salinity_valid_mask"] = torch.ones_like(batch["y_salinity_valid_mask"])
        batch["land_mask"] = torch.ones_like(batch["land_mask"])
        batch["output_land_mask"] = torch.ones_like(batch["land_mask"])
        generated_temperature_c = torch.full_like(batch["x"], 5.0)
        generated_salinity_psu = torch.full_like(batch["x_salinity"], 34.75)
        temperature_norm = temperature_normalize(
            mode="norm", tensor=generated_temperature_c
        )
        salinity_norm = salinity_normalize(mode="norm", tensor=generated_salinity_psu)
        generated_norm = torch.cat([temperature_norm, salinity_norm], dim=1)

        with patch.object(model, "forward", lambda *args, **kwargs: generated_norm):
            pred = model.predict_step(batch, batch_idx=0)

        self.assertTrue(torch.equal(pred["y_hat"], generated_norm))
        self.assertTrue(torch.equal(pred["y_hat_temperature"], temperature_norm))
        self.assertTrue(torch.equal(pred["y_hat_salinity"], salinity_norm))
        self.assertTrue(
            torch.allclose(
                pred["y_hat_temperature_denorm"], generated_temperature_c, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                pred["y_hat_salinity_denorm"], generated_salinity_psu, atol=1e-5
            )
        )
        self.assertTrue(
            torch.equal(pred["y_hat_denorm"], pred["y_hat_temperature_denorm"])
        )
        self.assertTrue(
            torch.equal(
                pred["y_hat_denorm_for_plot"],
                pred["y_hat_temperature_denorm_for_plot"],
            )
        )

    def test_pixel_uncertainty_step_returns_temperature_uncertainty_only(
        self,
    ) -> None:
        model = _make_pixel_model()
        batch = _make_pixel_batch()
        batch["y_valid_mask"] = torch.ones_like(batch["y_valid_mask"])
        batch["land_mask"] = torch.ones_like(batch["land_mask"])
        batch["output_land_mask"] = torch.ones_like(batch["land_mask"])
        first_celsius = torch.zeros_like(batch["x"])
        second_celsius = torch.full_like(batch["x"], 2.0)
        second_celsius[:, 1] = 4.0
        second_celsius[:, 0, 0, 0] = 6.0
        second_celsius[:, 1, 0, 0] = 8.0
        first_norm = temperature_normalize(mode="norm", tensor=first_celsius)
        second_norm = temperature_normalize(mode="norm", tensor=second_celsius)

        with patch.object(model, "forward", side_effect=[first_norm, second_norm]):
            pred = model.uncertainty_step(batch, batch_idx=0, num_samples=2)

        expected = torch.full(
            (1, 1, batch["x"].size(-2), batch["x"].size(-1)),
            1.5,
            dtype=batch["x"].dtype,
            device=batch["x"].device,
        )
        expected[:, :, 0, 0] = 3.5
        self.assertNotIn("y_hat", pred)
        self.assertNotIn("y_hat_temperature", pred)
        self.assertEqual(pred["uncertainty_num_samples"], 2)
        self.assertEqual(pred["uncertainty_stat"], "std")
        self.assertEqual(tuple(pred["uncertainty"].shape), tuple(expected.shape))
        self.assertTrue(torch.allclose(pred["uncertainty"], expected, atol=1e-5))
        self.assertTrue(
            torch.equal(pred["uncertainty"], pred["uncertainty_temperature"])
        )
        self.assertTrue(
            torch.equal(
                pred["uncertainty_normalized"],
                pred["uncertainty_temperature_normalized"],
            )
        )
        self.assertTrue(
            torch.allclose(pred["uncertainty_normalized"][:, :, 0, 0], torch.ones(1, 1))
        )
        self.assertTrue(
            torch.allclose(
                pred["uncertainty_normalized"][:, :, 0, 1], torch.zeros(1, 1)
            )
        )

    def test_pixel_uncertainty_step_requires_multiple_samples(self) -> None:
        model = _make_pixel_model()
        batch = _make_pixel_batch()

        with self.assertRaisesRegex(ValueError, "num_samples must be at least 2"):
            model.uncertainty_step(batch, batch_idx=0, num_samples=1)

    def test_pixel_uncertainty_step_splits_joint_temperature_salinity_maps(
        self,
    ) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        batch["y_valid_mask"] = torch.ones_like(batch["y_valid_mask"])
        batch["y_salinity_valid_mask"] = torch.ones_like(batch["y_salinity_valid_mask"])
        batch["land_mask"] = torch.ones_like(batch["land_mask"])
        batch["output_land_mask"] = torch.ones_like(batch["land_mask"])
        first_temperature_c = torch.zeros_like(batch["x"])
        second_temperature_c = torch.full_like(batch["x"], 4.0)
        first_salinity_psu = torch.full_like(batch["x_salinity"], 34.0)
        second_salinity_psu = torch.full_like(batch["x_salinity"], 36.0)
        first_norm = torch.cat(
            [
                temperature_normalize(mode="norm", tensor=first_temperature_c),
                salinity_normalize(mode="norm", tensor=first_salinity_psu),
            ],
            dim=1,
        )
        second_norm = torch.cat(
            [
                temperature_normalize(mode="norm", tensor=second_temperature_c),
                salinity_normalize(mode="norm", tensor=second_salinity_psu),
            ],
            dim=1,
        )

        with patch.object(model, "forward", side_effect=[first_norm, second_norm]):
            pred = model.uncertainty_step(batch, batch_idx=0, num_samples=2)

        expected_temperature = torch.full(
            (1, 1, batch["x"].size(-2), batch["x"].size(-1)),
            2.0,
            dtype=batch["x"].dtype,
            device=batch["x"].device,
        )
        expected_salinity = torch.full_like(expected_temperature, 1.0)
        self.assertTrue(
            torch.allclose(
                pred["uncertainty_temperature"], expected_temperature, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(pred["uncertainty_salinity"], expected_salinity, atol=1e-5)
        )
        self.assertTrue(
            torch.equal(pred["uncertainty"], pred["uncertainty_temperature"])
        )
        self.assertTrue(
            torch.equal(
                pred["uncertainty_normalized"],
                pred["uncertainty_temperature_normalized"],
            )
        )
        self.assertNotIn("y_hat", pred)

    def test_full_reconstruction_metrics_use_glorys_land_mask_support(self) -> None:
        model = _make_pixel_model()
        batch = _make_pixel_batch()
        batch["land_mask"] = torch.ones_like(batch["land_mask"])
        batch["land_mask"][..., 0, 1] = 0.0
        batch["y_valid_mask"] = torch.ones_like(batch["y_valid_mask"])
        batch["y_valid_mask"][:, :, 1, 0] = False
        model._cache_validation_batch(batch, n_cache=1)
        temperature_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
        pred = {
            "y_hat": batch["y"],
            "y_hat_denorm": temperature_denorm,
            "y_hat_denorm_for_plot": temperature_denorm,
            "further_valid_mask": None,
            "denoise_samples": [],
            "x0_denoise_samples": [],
        }
        metric_masks: list[torch.Tensor] = []

        def capture_metrics(**kwargs: Any) -> tuple[torch.Tensor, ...]:
            metric_masks.append(kwargs["eval_mask"].detach().clone())
            return tuple(torch.zeros((), dtype=torch.float32) for _ in range(4))

        with (
            patch.object(model, "predict_step", lambda *args, **kwargs: pred),
            patch.object(model, "log", lambda *args, **kwargs: None),
            patch.object(
                model, "_compute_full_reconstruction_metrics", capture_metrics
            ),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_conditional_reconstruction_grid",
                lambda **kwargs: None,
            ),
        ):
            model._run_single_image_full_reconstruction_for_current_weights(
                log_profile=False, log_denoise=False
            )

        self.assertEqual(len(metric_masks), 1)
        self.assertTrue(torch.equal(metric_masks[0][:, :, 0, 1], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(metric_masks[0][:, :, 1, 0], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(metric_masks[0][:, :, 0, 0], torch.ones(1, 2)))

    def test_full_reconstruction_logs_separate_salinity_grid(self) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        model._cache_validation_batch(batch, n_cache=1)
        temperature_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
        salinity_denorm = salinity_normalize(mode="denorm", tensor=batch["y_salinity"])
        pred = {
            "y_hat": torch.cat([batch["y"], batch["y_salinity"]], dim=1),
            "y_hat_temperature": batch["y"],
            "y_hat_salinity": batch["y_salinity"],
            "y_hat_denorm": temperature_denorm,
            "y_hat_denorm_for_plot": temperature_denorm,
            "y_hat_salinity_denorm": salinity_denorm,
            "y_hat_salinity_denorm_for_plot": salinity_denorm,
            "further_valid_mask": None,
            "denoise_samples": [],
            "x0_denoise_samples": [],
        }
        reconstruction_calls: list[dict[str, Any]] = []

        def capture_reconstruction_grid(**kwargs: Any) -> None:
            reconstruction_calls.append(kwargs)

        with (
            patch.object(model, "predict_step", lambda *args, **kwargs: pred),
            patch.object(model, "log", lambda *args, **kwargs: None),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_conditional_reconstruction_grid",
                capture_reconstruction_grid,
            ),
        ):
            model._run_single_image_full_reconstruction_for_current_weights(
                log_profile=False, log_denoise=False
            )

        self.assertEqual(len(reconstruction_calls), 2)
        salinity_call = reconstruction_calls[1]
        self.assertEqual(salinity_call["prefix"], "val_salinity_imgs")
        self.assertEqual(salinity_call["image_key"], "salinity_full_reconstruction")
        self.assertEqual(salinity_call["cmap"], "winter")
        self.assertEqual(salinity_call["plot_unit"], "salinity")
        self.assertEqual(
            salinity_call["error_metric_prefix"],
            "val_salinity_absolute_band_error",
        )
        self.assertEqual(salinity_call["error_metric_unit"], "psu")
        self.assertTrue(
            torch.equal(
                salinity_call["x"],
                salinity_normalize(mode="denorm", tensor=batch["x_salinity"]),
            )
        )
        self.assertTrue(torch.equal(salinity_call["land_mask"], batch["land_mask"]))
        self.assertTrue(torch.equal(salinity_call["y_hat"], salinity_denorm))

    def test_full_reconstruction_logs_salinity_only_grid_metadata(self) -> None:
        model = _make_pixel_model(output_fields=("salinity",), log_intermediates=True)
        batch = _make_pixel_batch(include_salinity=True)
        model._cache_validation_batch(batch, n_cache=1)
        salinity_denorm = salinity_normalize(mode="denorm", tensor=batch["y_salinity"])
        pred = {
            "y_hat": batch["y_salinity"],
            "y_hat_salinity": batch["y_salinity"],
            "y_hat_denorm": salinity_denorm,
            "y_hat_denorm_for_plot": salinity_denorm,
            "y_hat_salinity_denorm": salinity_denorm,
            "y_hat_salinity_denorm_for_plot": salinity_denorm,
            "further_valid_mask": None,
            "denoise_samples": [(0, batch["y_salinity"]), (2, batch["y_salinity"])],
            "x0_denoise_samples": [(0, batch["y_salinity"])],
        }
        reconstruction_calls: list[dict[str, Any]] = []
        profile_calls: list[dict[str, Any]] = []
        denoise_calls: list[dict[str, Any]] = []

        with (
            patch.object(model, "predict_step", lambda *args, **kwargs: pred),
            patch.object(model, "log", lambda *args, **kwargs: None),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_conditional_reconstruction_grid",
                lambda **kwargs: reconstruction_calls.append(kwargs),
            ),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_glorys_profile_comparison",
                lambda **kwargs: profile_calls.append(kwargs),
            ),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_denoise_timestep_grid",
                lambda **kwargs: denoise_calls.append(kwargs),
            ),
        ):
            model._run_single_image_full_reconstruction_for_current_weights()

        self.assertEqual(len(reconstruction_calls), 1)
        salinity_call = reconstruction_calls[0]
        self.assertEqual(salinity_call["cmap"], "winter")
        self.assertEqual(salinity_call["plot_unit"], "salinity")
        self.assertEqual(
            salinity_call["error_metric_prefix"],
            "val_salinity_absolute_band_error",
        )
        self.assertEqual(salinity_call["error_metric_unit"], "psu")
        self.assertEqual(salinity_call["error_metric_label"], "L1 (PSU)")
        self.assertEqual(
            salinity_call["error_metric_title"],
            "Generated-Pixel Salinity L1 by Band",
        )
        self.assertTrue(torch.equal(salinity_call["y_hat"], salinity_denorm))
        self.assertEqual(len(profile_calls), 1)
        self.assertEqual(profile_calls[0]["profile_x_label"], "Salinity (PSU)")
        self.assertEqual(len(denoise_calls), 1)
        self.assertEqual(denoise_calls[0]["cmap"], "winter")
        self.assertEqual(denoise_calls[0]["plot_unit"], "salinity")

    def test_full_reconstruction_logs_salinity_scalar_metrics(self) -> None:
        model = _make_pixel_model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
        )
        batch = _make_pixel_batch(include_salinity=True)
        model._cache_validation_batch(batch, n_cache=1)
        temperature_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
        salinity_denorm = salinity_normalize(mode="denorm", tensor=batch["y_salinity"])
        pred = {
            "y_hat": torch.cat([batch["y"], batch["y_salinity"]], dim=1),
            "y_hat_temperature": batch["y"],
            "y_hat_salinity": batch["y_salinity"],
            "y_hat_denorm": temperature_denorm,
            "y_hat_denorm_for_plot": temperature_denorm,
            "y_hat_salinity_denorm": salinity_denorm,
            "y_hat_salinity_denorm_for_plot": salinity_denorm,
            "further_valid_mask": None,
            "denoise_samples": [],
            "x0_denoise_samples": [],
        }
        logged_names: list[str] = []

        with (
            patch.object(model, "predict_step", lambda *args, **kwargs: pred),
            patch.object(
                model, "log", lambda name, *args, **kwargs: logged_names.append(name)
            ),
            patch(
                "depth_recon.models.diffusion.PixelDiffusion.log_wandb_conditional_reconstruction_grid",
                lambda **kwargs: None,
            ),
        ):
            model._run_single_image_full_reconstruction_for_current_weights(
                log_profile=False, log_denoise=False
            )

        self.assertIn("val/recon_l1_full_recon", logged_names)
        self.assertIn("val_salinity/recon_mse_full_recon", logged_names)
        self.assertIn("val_salinity/recon_l1_full_recon", logged_names)
        self.assertIn("val_salinity/recon_psnr_full_recon", logged_names)
        self.assertIn("val_salinity/recon_ssim_full_recon", logged_names)

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
                            "interval": "epochs",
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
            self.assertEqual(model.lr_scheduler_interval, "epoch")
            self.assertIsInstance(optim_config, dict)
            self.assertEqual(optim_config["lr_scheduler"]["monitor"], "val/loss_ckpt")
            self.assertEqual(optim_config["lr_scheduler"]["interval"], "epoch")
            self.assertTrue(optim_config["lr_scheduler"]["strict"])


if __name__ == "__main__":
    unittest.main()
