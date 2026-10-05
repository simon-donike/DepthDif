from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from depth_recon.configs.config_resolver_pixel import (
    FULL_RECONSTRUCTION_MONITOR,
    apply_reconstruction_checkpoint_contract,
)
from depth_recon.models.diffusion.EMA import EMA
from depth_recon.models.baselines import UNet2DInfillingBaseline
from depth_recon.models.latent.Autoencoder import (
    DepthBandAutoencoder,
    DepthBandAutoencoderLightning,
)
from depth_recon.models.latent.LatentDiffusion import LatentDiffusionConditional
from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.utils.normalizations import (
    SALINITY_STD,
    Y_STD,
    salinity_normalize,
    temperature_normalize,
)
from depth_recon.utils.reconstruction_validation import FullReconstructionValidation
from tests.test_model_dry_runs import _StaticBatchDataset, _make_pixel_model


class _EvaluationDataset(Dataset):
    """Unequal depth support with an entirely unsupported final depth."""

    def __len__(self):
        return 5

    def __getitem__(self, index):
        target = torch.zeros(3, 2, 2)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[1] = False
        mask[1, 0, 0] = True
        mask[2] = False
        return {
            "y": target,
            "y_valid_mask": mask,
            "y_salinity": target,
            "y_salinity_valid_mask": mask,
            "sample_id": index,
        }


class _EvaluationModel(pl.LightningModule):
    """Lower denoising losses deliberately accompany worse full reconstructions."""

    def __init__(self, fields=("temperature",), varying_errors=False):
        super().__init__()
        self.output_fields = fields
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.varying_errors = varying_errors
        self.seen = []

    def training_step(self, batch, batch_idx):
        return self.weight * 0

    def validation_step(self, batch, batch_idx):
        self.log(
            "val/loss_ckpt", 10.0 - self.current_epoch, sync_dist=True, batch_size=1
        )

    def predict_step(self, batch, batch_idx):
        self.seen.extend(batch["sample_id"].tolist())
        # Consume all global generators to exercise evaluation RNG restoration.
        random.random()
        np.random.random()
        torch.rand(1, device=self.device)
        output = {}
        for field in self.output_fields:
            key = "y" if field == "temperature" else "y_salinity"
            normalizer = (
                temperature_normalize if field == "temperature" else salinity_normalize
            )
            scale = Y_STD if field == "temperature" else SALINITY_STD
            reference = normalizer(mode="denorm", tensor=batch[key])
            error = (
                torch.arange(1, 4, device=self.device).view(1, 3, 1, 1)
                * scale
                * (self.current_epoch + 1 + self.weight)
            )
            if self.varying_errors:
                error = error * (batch["sample_id"].view(-1, 1, 1, 1) + 1)
            output[f"y_hat_{field}_denorm"] = reference + error
        return output

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _SaveDistributedMetrics(pl.Callback):
    """Write each rank's reduced results for the parent test process."""

    def __init__(self, root):
        self.root = Path(root)

    def on_validation_end(self, trainer, pl_module):
        payload = {
            name: float(value) for name, value in trainer.callback_metrics.items()
        }
        payload["seen"] = pl_module.seen
        (self.root / f"rank{trainer.global_rank}.json").write_text(json.dumps(payload))


class TestReconstructionValidation(unittest.TestCase):
    def _trainer(self, **kwargs):
        """Build a quiet CPU trainer that still executes real Lightning hooks."""
        settings = dict(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            limit_val_batches=1,
        )
        settings.update(kwargs)
        return pl.Trainer(**settings)

    def test_contract_migrates_old_runs_and_validates_budget(self):
        config = {"trainer": {"ckpt_monitor": "val/loss_ckpt"}}
        apply_reconstruction_checkpoint_contract(config)
        self.assertEqual(config["trainer"]["ckpt_monitor"], FULL_RECONSTRUCTION_MONITOR)
        self.assertEqual(config["training"]["reconstruction_eval"]["sample_count"], 128)
        config["training"]["reconstruction_eval"]["sample_count"] = 0
        with self.assertRaisesRegex(ValueError, "sample_count"):
            apply_reconstruction_checkpoint_contract(config)

    def test_all_scenarios_pool_depths_and_ignore_unsupported_bands(self):
        dataset = _EvaluationDataset()
        for fields in (("temperature",), ("salinity",), ("temperature", "salinity")):
            with self.subTest(fields=fields):
                callback = FullReconstructionValidation(
                    dataset=dataset, sample_count=5, batch_size=2
                )
                model = _EvaluationModel(fields)
                trainer = self._trainer(
                    callbacks=[callback], enable_checkpointing=False
                )
                trainer.validate(
                    model, dataloaders=DataLoader(dataset, batch_size=1), verbose=False
                )
                self.assertAlmostEqual(
                    float(trainer.callback_metrics[FULL_RECONSTRUCTION_MONITOR]),
                    1.5,
                    places=5,
                )
                self.assertEqual(sorted(model.seen), list(range(5)))
                self.assertEqual(
                    float(
                        trainer.callback_metrics["val/full_reconstruction/sample_count"]
                    ),
                    5,
                )
                for field in fields:
                    self.assertEqual(
                        float(
                            trainer.callback_metrics[
                                f"val/full_reconstruction/{field}_supported_depths"
                            ]
                        ),
                        2,
                    )

    def test_actual_diffusion_sampling_runs_for_all_scenarios_without_previews(self):
        dataset = _StaticBatchDataset(length=3, include_salinity=True)
        for fields in (("temperature",), ("salinity",), ("temperature", "salinity")):
            with self.subTest(fields=fields):
                channels = 2 * len(fields)
                model = _make_pixel_model(
                    output_fields=fields,
                    generated_channels=channels,
                    condition_channels=channels + 1,
                    full_reconstruction_logging_enabled=False,
                    val_inference_sampler="ddpm",
                )
                callback = FullReconstructionValidation(
                    dataset=dataset, sample_count=3, batch_size=2
                )
                trainer = self._trainer(
                    callbacks=[callback], enable_checkpointing=False
                )
                trainer.validate(
                    model, dataloaders=DataLoader(dataset, batch_size=1), verbose=False
                )
                self.assertTrue(
                    torch.isfinite(
                        trainer.callback_metrics[FULL_RECONSTRUCTION_MONITOR]
                    )
                )

    def test_best_checkpoint_follows_reconstruction_even_when_loss_improves(self):
        dataset = _EvaluationDataset()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = ModelCheckpoint(
                dirpath=tmp,
                filename="best-epoch{epoch:03d}-step{step:09d}",
                monitor=FULL_RECONSTRUCTION_MONITOR,
                mode="min",
                save_top_k=3,
                save_on_train_epoch_end=False,
            )
            callback = FullReconstructionValidation(
                dataset=dataset, sample_count=5, batch_size=2, output_dir=tmp
            )
            trainer = self._trainer(
                callbacks=[checkpoint, callback], max_epochs=4, limit_train_batches=1
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            trainer.fit(
                _EvaluationModel(), train_dataloaders=loader, val_dataloaders=loader
            )
            saved = torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(saved["epoch"], 0)
            self.assertAlmostEqual(float(checkpoint.best_model_score), 1.5, places=5)
            self.assertAlmostEqual(
                float(trainer.callback_metrics[FULL_RECONSTRUCTION_MONITOR]),
                6.0,
                places=5,
            )
            self.assertEqual(float(trainer.callback_metrics["val/loss_ckpt"]), 7.0)
            self.assertEqual(len(checkpoint.best_k_models), 3)
            self.assertTrue(all("step" in path for path in checkpoint.best_k_models))
            retained_epochs = sorted(
                torch.load(path, map_location="cpu", weights_only=False)["epoch"]
                for path in checkpoint.best_k_models
            )
            self.assertEqual(retained_epochs, [0, 1, 2])
            self.assertTrue(
                (Path(tmp) / "full_reconstruction_selection.json").is_file()
            )

    def test_baseline_latent_and_autoencoder_use_full_prediction_path(self):
        dataset = _StaticBatchDataset(length=3, include_eo=True)
        autoencoder = DepthBandAutoencoder(
            in_channels=2,
            latent_channels=1,
            encoder_hidden_channels=(4,),
            decoder_hidden_channels=(4,),
            spatial_downsample=1,
        )
        models = [
            UNet2DInfillingBaseline(
                generated_channels=2, base_channels=4, channel_mults=(1,)
            ),
            DepthBandAutoencoderLightning(autoencoder=autoencoder),
            LatentDiffusionConditional(
                autoencoder=autoencoder,
                generated_channels=1,
                condition_channels=2,
                condition_mask_channels=1,
                condition_include_eo=False,
                condition_use_valid_mask=True,
                parameterization="x0",
                num_timesteps=2,
                noise_schedule="linear",
                unet_dim=8,
                unet_dim_mults=(1,),
                full_reconstruction_logging_enabled=False,
                wandb_verbose=False,
                log_intermediates=False,
            ),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                # Resolve the validation dataset through the DataModule, as the
                # standalone autoencoder runner does after Lightning setup.
                datamodule = DepthTileDataModule(
                    dataset=dataset,
                    val_dataset=dataset,
                    dataloader_cfg={"val_batch_size": 1, "val_num_workers": 0},
                )
                callback = FullReconstructionValidation(sample_count=3, batch_size=2)
                trainer = self._trainer(
                    callbacks=[callback], enable_checkpointing=False
                )
                trainer.validate(model, datamodule=datamodule, verbose=False)
                self.assertTrue(
                    torch.isfinite(
                        trainer.callback_metrics[FULL_RECONSTRUCTION_MONITOR]
                    )
                )

    def test_joint_score_normalizes_units_and_weights_fields_equally(self):
        stats = {
            "temperature": torch.tensor(
                [[Y_STD**2 * 100], [Y_STD * 100], [100.0]], dtype=torch.float64
            ),
            "salinity": torch.tensor(
                [[(3 * SALINITY_STD) ** 2], [3 * SALINITY_STD], [1.0]],
                dtype=torch.float64,
            ),
        }
        score = FullReconstructionValidation._metrics(stats)[
            FULL_RECONSTRUCTION_MONITOR
        ]
        self.assertAlmostEqual(float(score), 2.0)

    def test_ema_score_and_saved_raw_and_ema_weights_stay_consistent(self):
        dataset = _EvaluationDataset()
        model = _EvaluationModel()
        model.weight.data.fill_(3.0)
        ema = EMA(
            decay=1.0,
            evaluate_ema_weights_instead=True,
            save_ema_weights_in_callback_state=True,
        )
        ema._ema_model_weights = {"weight": torch.tensor(0.0)}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = ModelCheckpoint(
                dirpath=tmp,
                monitor=FULL_RECONSTRUCTION_MONITOR,
                save_on_train_epoch_end=False,
            )
            callback = FullReconstructionValidation(dataset=dataset, sample_count=5)
            trainer = self._trainer(
                callbacks=[ema, checkpoint, callback],
                max_epochs=1,
                limit_train_batches=1,
            )
            loader = DataLoader(dataset, batch_size=1)
            trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
            saved = torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=False
            )
            self.assertAlmostEqual(float(checkpoint.best_model_score), 1.5, places=5)
            self.assertEqual(float(saved["state_dict"]["weight"]), 3.0)
            self.assertEqual(
                float(saved["callbacks"][ema.state_key]["ema_weights"]["weight"]), 0.0
            )
            self.assertFalse(ema.weights_are_applied)

    def test_invalid_predictions_cannot_improve_the_score(self):
        batch = default_collate([_EvaluationDataset()[0]])
        prediction = temperature_normalize(mode="denorm", tensor=batch["y"])
        prediction[:, 2] = float("nan")
        stats = FullReconstructionValidation._batch_statistics(
            batch, {"y_hat_denorm": prediction}, "temperature", single_field=True
        )
        self.assertEqual(
            float(
                FullReconstructionValidation._metrics({"temperature": stats})[
                    FULL_RECONSTRUCTION_MONITOR
                ]
            ),
            0.0,
        )
        prediction[:, 0, 0, 0] = float("nan")
        stats = FullReconstructionValidation._batch_statistics(
            batch, {"y_hat_denorm": prediction}, "temperature", single_field=True
        )
        self.assertTrue(
            torch.isinf(
                FullReconstructionValidation._metrics({"temperature": stats})[
                    FULL_RECONSTRUCTION_MONITOR
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "No valid"):
            FullReconstructionValidation._metrics({"temperature": torch.zeros(3, 3)})

    def test_rng_is_restored_and_sanity_checks_skip_reconstructions(self):
        callback = FullReconstructionValidation(
            dataset=_EvaluationDataset(), sample_count=5
        )
        model = _EvaluationModel().eval()
        trainer = SimpleNamespace(
            sanity_checking=True,
            global_rank=0,
            world_size=1,
            is_global_zero=True,
            strategy=SimpleNamespace(reduce=lambda value, **kwargs: value),
            precision_plugin=SimpleNamespace(forward_context=nullcontext),
        )
        callback.on_validation_epoch_end(trainer, model)
        self.assertEqual(model.seen, [])
        trainer.sanity_checking = False
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        with patch.object(model, "log_dict"):
            callback.on_validation_epoch_end(trainer, model)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_two_ranks_pool_uneven_batches_without_duplicate_patches(self):
        dataset = _EvaluationDataset()
        with tempfile.TemporaryDirectory() as tmp:
            callback = FullReconstructionValidation(
                dataset=dataset, sample_count=5, batch_size=2
            )
            trainer = self._trainer(
                devices=2,
                strategy="ddp_fork",
                enable_checkpointing=False,
                callbacks=[callback, _SaveDistributedMetrics(tmp)],
            )
            trainer.validate(
                _EvaluationModel(varying_errors=True),
                dataloaders=DataLoader(dataset, batch_size=1),
                verbose=False,
            )
            results = [
                json.loads((Path(tmp) / f"rank{rank}.json").read_text())
                for rank in range(2)
            ]
            self.assertEqual(
                sorted(results[0]["seen"] + results[1]["seen"]), list(range(5))
            )
            expected = 1.5 * np.sqrt(np.mean(np.arange(1, 6) ** 2))
            for result in results:
                self.assertAlmostEqual(
                    result[FULL_RECONSTRUCTION_MONITOR], expected, places=5
                )
                self.assertEqual(result["val/full_reconstruction/sample_count"], 5)


if __name__ == "__main__":
    unittest.main()
