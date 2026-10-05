"""Regression coverage for mask-aware AE exports and conditional latent diffusion."""

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from depth_recon.configs.config_resolver_pixel import load_pixel_training_config
from depth_recon.models.latent import (
    DepthBandAutoencoder,
    DepthBandAutoencoderLightning,
    LatentDiffusionConditional,
)
from depth_recon.models.latent.workflow import export_calibrated_autoencoder


def make_batch(fields=("temperature",)):
    """Create physical targets with partial profiles and a dry depth cell."""
    batch = {"land_mask": torch.ones(2, 1, 8, 8), "eo": torch.randn(2, 3, 8, 8)}
    for field in fields:
        suffix = "" if field == "temperature" else "_salinity"
        target = torch.randn(2, 2, 8, 8)
        wet = torch.ones_like(target, dtype=torch.bool)
        wet[:, 1, 0, 0] = False
        mask = wet.clone()
        mask[:, :, 2:] = False
        batch["y" + suffix] = target
        batch["x" + suffix] = torch.where(mask, target + 0.1, 0.0)
        batch["y" + suffix + "_valid_mask"] = wet
        batch["x" + suffix + "_valid_mask"] = mask
        batch["wet_mask" + suffix] = wet
        batch["climatology" + suffix] = torch.full_like(target, 0.2)
    return batch


def make_ae(fields=("temperature",), residual=False):
    """Build a small instance of the production mask-aware architecture."""
    return DepthBandAutoencoder(
        in_channels=2 * len(fields),
        latent_channels=1,
        encoder_hidden_channels=(4,),
        decoder_hidden_channels=(4,),
        output_fields=fields,
        climatology_residual=residual,
    )


class TestLatentWorkflow(unittest.TestCase):
    def test_missing_fills_do_not_affect_encoded_features_or_gradients(self):
        ae = make_ae()
        batch = make_batch()
        values = batch["x"].clone().requires_grad_()
        mask = batch["x_valid_mask"]
        reference = ae.encode(values, mask, batch["wet_mask"])
        changed = torch.where(
            mask, values.detach(), torch.full_like(values, float("nan"))
        )
        self.assertTrue(
            torch.equal(reference, ae.encode(changed, mask, batch["wet_mask"]))
        )
        reference.sum().backward()
        self.assertEqual(float(values.grad[~mask].abs().sum()), 0.0)
        # Valid zeros and missing zeros are distinct encoder inputs.
        self.assertFalse(
            torch.equal(
                ae.encode(torch.zeros_like(values), mask),
                ae.encode(torch.zeros_like(values), ~mask),
            )
        )

    def test_empty_reconstruction_support_is_zero_not_unmasked(self):
        model = DepthBandAutoencoderLightning(autoencoder=make_ae())
        prediction = torch.randn(1, 2, 4, 4, requires_grad=True)
        losses = model._recon_losses(
            torch.full_like(prediction, float("nan")),
            prediction,
            torch.zeros_like(prediction),
        )
        sum(losses).backward()
        self.assertEqual(float(sum(losses)), 0.0)
        self.assertEqual(float(prediction.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            model.model.align_mask(torch.ones(1, 3, 4, 4), prediction)

    def test_reconstruction_cases_preserve_real_observation_support(self):
        model = DepthBandAutoencoderLightning(autoencoder=make_ae())
        batch = make_batch()
        cases = model.reconstruction_cases(batch)
        self.assertEqual(set(cases), {"dense", "sparse_dense", "observed"})
        self.assertTrue(torch.equal(cases["observed"][2], batch["x_valid_mask"]))
        self.assertTrue(torch.equal(cases["sparse_dense"][2], batch["y_valid_mask"]))

    def test_export_requires_supported_training_values(self):
        with tempfile.TemporaryDirectory() as directory:
            model = DepthBandAutoencoderLightning(autoencoder=make_ae())
            batch = make_batch()
            export_calibrated_autoencoder(model, [batch], Path(directory) / "ae.ckpt")
            self.assertTrue(model.model.latent_calibrated)
            z = model.model.encode(batch["y"], batch["y_valid_mask"], batch["wet_mask"])
            support = batch["y_valid_mask"].any(1, keepdim=True)
            expected = z[support].mean()
            self.assertTrue(
                torch.allclose(
                    model.model.latent_mean.flatten()[0], expected, atol=1e-6
                )
            )
            batch["y_valid_mask"].zero_()
            with self.assertRaisesRegex(ValueError, "No valid training"):
                export_calibrated_autoencoder(
                    model, [batch], Path(directory) / "empty.ckpt"
                )

    def test_calibrated_checkpoint_to_diffusion_and_sampling(self):
        for fields in (("temperature",), ("salinity",), ("temperature", "salinity")):
            with (
                self.subTest(fields=fields),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                batch = make_batch(fields)
                ae = make_ae(fields, residual=True)
                module = DepthBandAutoencoderLightning(autoencoder=ae)
                optimizer = module.configure_optimizers()
                loss = sum(
                    sum(module._recon_losses(target, prediction, mask))
                    for prediction, target, mask in module.reconstruction_cases(
                        batch
                    ).values()
                )
                loss.backward()
                optimizer.step()
                export_calibrated_autoencoder(module, [batch], root / "ae.ckpt")
                channels = 2 * len(fields)
                configs = {
                    "ae": {
                        "ae": {
                            "in_channels": channels,
                            "latent_channels": 1,
                            "output_fields": list(fields),
                            "climatology_residual": True,
                            "encoder": {"hidden_channels": [4]},
                            "decoder": {"hidden_channels": [4]},
                        }
                    },
                    "model": {
                        "model": {
                            "generated_channels": 1,
                            "output_fields": list(fields),
                            "condition_channels": 1 + 3 + channels * 3 + 1,
                            "condition_mask_channels": channels,
                            "condition_eo_channels": 3,
                            "condition_include_eo": True,
                            "condition_use_valid_mask": True,
                            "condition_use_land_mask": True,
                            "condition_use_wet_mask": True,
                            "mask_diffusion_with_wet_mask": True,
                            "climatology_residual": True,
                            "parameterization": "x0",
                            "unet": {"dim": 8, "dim_mults": [1]},
                            "latent": {
                                "ae_config_path": str(root / "ae.yaml"),
                                "ae_checkpoint": str(root / "ae.ckpt"),
                                "decoded_observation_weight": 1.0,
                            },
                        }
                    },
                    "training": {
                        "training": {
                            "noise": {"num_timesteps": 2},
                            "validation_sampling": {
                                "sampler": "ddim",
                                "ddim_num_timesteps": 2,
                            },
                        },
                        "wandb": {"verbose": False},
                    },
                    "data": {"dataset": {}},
                }
                for name, config in configs.items():
                    (root / f"{name}.yaml").write_text(yaml.safe_dump(config))
                model = LatentDiffusionConditional.from_config(
                    str(root / "model.yaml"),
                    str(root / "data.yaml"),
                    str(root / "training.yaml"),
                )
                model.log = lambda *args, **kwargs: None
                loss = model.training_step(batch, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.model.parameters()
                    )
                )
                self.assertTrue(
                    all(p.grad is None for p in model.autoencoder.parameters())
                )
                latent = torch.randn(2, 1, 8, 8, requires_grad=True)
                model.output_T(latent).sum().backward()
                self.assertGreater(float(latent.grad.abs().sum()), 0)
                prediction = model.predict_step(batch, 0)
                for field in fields:
                    self.assertEqual(
                        prediction[f"y_hat_{field}_denorm"].shape, (2, 2, 8, 8)
                    )
                checkpoint = {"state_dict": model.state_dict()}
                model.on_save_checkpoint(checkpoint)
                model.on_load_checkpoint(checkpoint)
                checkpoint["ae_contract"]["output_fields"] = ["invalid"]
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    model.on_load_checkpoint(checkpoint)

    def test_frozen_random_ae_and_ambient_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ae.yaml").write_text("ae: {in_channels: 2, latent_channels: 1}")
            (root / "model.yaml").write_text(
                yaml.safe_dump(
                    {"model": {"latent": {"ae_config_path": str(root / "ae.yaml")}}}
                )
            )
            (root / "empty.yaml").write_text("{}")
            with self.assertRaisesRegex(ValueError, "requires latent.ae_checkpoint"):
                LatentDiffusionConditional.from_config(
                    str(root / "model.yaml"),
                    str(root / "empty.yaml"),
                    str(root / "empty.yaml"),
                )
        with self.assertRaisesRegex(ValueError, "ambient"):
            LatentDiffusionConditional(
                autoencoder=make_ae(),
                generated_channels=1,
                ambient_occlusion_enabled=True,
            )

    def test_preset_resolves_physical_and_latent_channels_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario, physical in (
                ("temperature", 50),
                ("salinity", 50),
                ("joint", 100),
            ):
                bundle = load_pixel_training_config(
                    config_path_value="src/depth_recon/configs/lat_space/training_super_config.yaml",
                    scenario_override=scenario,
                    runtime_config_dir=directory,
                )
                model = bundle.model_cfg["model"]
                self.assertEqual(model["physical_channels"], physical)
                self.assertEqual(model["generated_channels"], 12)
                self.assertEqual(model["condition_channels"], 12 + 3 + 2 * physical + 1)
                self.assertTrue(bundle.training_cfg["dataloader"]["val_shuffle"])

    def test_fixed_ae_validation_selects_dense_checkpoint_and_scores_holdouts(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from depth_recon.data.datamodule import DepthTileDataModule
        from depth_recon.models.latent.workflow import (
            AutoencoderReconstructionValidation,
        )

        with tempfile.TemporaryDirectory() as directory:
            batch = make_batch()
            samples = [
                {key: value[index] for key, value in batch.items()}
                for index in range(2)
            ]
            datamodule = DepthTileDataModule(
                dataset=samples,
                val_dataset=samples,
                dataloader_cfg={
                    "batch_size": 2,
                    "val_batch_size": 2,
                    "num_workers": 0,
                    "val_num_workers": 0,
                    "val_shuffle": True,
                },
            )
            model = DepthBandAutoencoderLightning(autoencoder=make_ae())
            checkpoint = ModelCheckpoint(
                dirpath=directory, monitor="val/full_reconstruction_score", mode="min"
            )
            callback = AutoencoderReconstructionValidation(
                sample_count=2, batch_size=2, output_dir=directory
            )
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[checkpoint, callback],
            )
            trainer.fit(model, datamodule=datamodule)
            self.assertTrue(Path(checkpoint.best_model_path).is_file())
            self.assertTrue(
                torch.isfinite(
                    trainer.callback_metrics["val/full_reconstruction_score"]
                )
            )
            self.assertGreater(
                float(
                    trainer.callback_metrics[
                        "val/ae/held_profiles/temperature_valid_values"
                    ]
                ),
                0,
            )
            saved = torch.load(checkpoint.best_model_path, weights_only=False)
            self.assertEqual(saved["ae_contract"], model.model.contract())
            self.assertEqual(callback.patch_indices, [0, 1])

    def test_ae_runner_uses_explicit_splits_and_exports_selected_checkpoint(self):
        from unittest.mock import patch
        import os
        import train_autoencoder
        from pytorch_lightning.loggers import CSVLogger

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = make_batch()
            samples = [
                {key: value[index] for key, value in batch.items()}
                for index in range(2)
            ]
            data = {
                "scenario": "temperature",
                "data": {
                    "dataset": {"wet_domain": {"enabled": True}},
                    "split": {"val_year": 2016},
                },
                "model": {
                    "model_type": "latent_cond_dif",
                    "depth_channels": 2,
                    "condition_include_eo": True,
                    "condition_use_valid_mask": True,
                    "condition_use_land_mask": True,
                    "latent": {"latent_channels": 1},
                },
                "training": {},
            }
            training = {
                "training": {
                    "reconstruction_eval": {
                        "sample_count": 2,
                        "batch_size": 2,
                        "seed": 7,
                    }
                },
                "trainer": {
                    "num_gpus": 0,
                    "precision": "32-true",
                    "num_sanity_val_steps": 0,
                    "enable_model_summary": False,
                    "limit_val_batches": 1,
                },
                "dataloader": {
                    "batch_size": 2,
                    "val_batch_size": 2,
                    "num_workers": 0,
                    "val_num_workers": 0,
                },
                "scheduler": {"reduce_on_plateau": {"enabled": False}},
            }
            ae = {
                "ae": {
                    "in_channels": 2,
                    "latent_channels": 1,
                    "encoder": {"hidden_channels": [4]},
                    "decoder": {"hidden_channels": [4]},
                    "training": {"max_epochs": 1, "calibration_batches": 1},
                }
            }
            for name, config in (("data", data), ("training", training), ("ae", ae)):
                (root / f"{name}.yaml").write_text(yaml.safe_dump(config))
            with (
                patch.dict(os.environ),
                patch.object(
                    train_autoencoder, "build_dataset", return_value=samples
                ) as build,
                patch.object(
                    train_autoencoder,
                    "build_wandb_logger",
                    return_value=CSVLogger(root / "csv"),
                ),
                patch.object(train_autoencoder, "upload_configs_to_wandb"),
            ):
                train_autoencoder.main(
                    ae_config_path=str(root / "ae.yaml"),
                    data_config_path=str(root / "data.yaml"),
                    training_config_path=str(root / "training.yaml"),
                    resume_checkpoint=None,
                    load_checkpoint=None,
                    run_dir_value=str(root / "run"),
                )
            self.assertEqual(build.call_count, 2)
            self.assertEqual(
                build.call_args_list[0].kwargs.get("split", "train"), "train"
            )
            self.assertEqual(build.call_args_list[1].kwargs["split"], "val")
            exported = torch.load(
                root / "run" / "autoencoder_calibrated.ckpt", weights_only=False
            )
            self.assertTrue(exported["state_dict"]["latent_calibrated"])
            self.assertEqual(exported["calibration"]["split"], "train")
