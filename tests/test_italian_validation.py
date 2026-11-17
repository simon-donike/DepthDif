"""Tests for the training-time Italian validation callback."""

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import depth_recon.utils.italian_validation as italian_validation
from depth_recon.utils.italian_validation import (
    ItalianValidationCallback,
    ItalianValidationConfig,
    build_italian_validation_callback,
    score_italian_predictions,
)
from depth_recon.models.diffusion.EMA import EMA


def _rows() -> pd.DataFrame:
    """Return common-support score rows with an invalid model value."""
    return pd.DataFrame(
        {
            "profile_id": ["a", "a", "b"],
            "depth_m": [5.0, 150.0, 5.0],
            "observation": [10.0, 8.0, 12.0],
            "model": [11.0, np.nan, 12.0],
            "raw_glorys": [10.0, 10.0, 10.0],
            "corrected_glorys": [10.5, 8.5, 11.5],
            "climatology": [9.0, 9.0, 9.0],
        }
    )


class _Logger:
    """Minimal logger used by callback tests."""

    def __init__(self) -> None:
        self.metrics = {}

    def log_metrics(self, values, step=None) -> None:
        self.metrics.update(values)


class ItalianValidationTest(unittest.TestCase):
    """Test scoring, state, deterministic sharding, and tensor predictions."""

    def _profiles(self, root: Path) -> Path:
        path = root / "profiles.csv"
        pd.DataFrame(
            {
                "profile_id": ["a"],
                "observation_time": ["2016-01-01T00:00:00Z"],
                "depth_m": [5.0],
                "observation": [10.0],
            }
        ).to_csv(path, index=False)
        return path

    def _callback(self, root: Path, evaluator) -> ItalianValidationCallback:
        callback = build_italian_validation_callback(
            {
                "training": {
                    "italian_validation": {
                        "enabled": True,
                        "profiles": str(self._profiles(root)),
                    }
                }
            },
            evaluator,
        )
        self.assertIsNotNone(callback)
        return callback

    @staticmethod
    def _trainer(root: Path, *, epoch: int = 0, sanity: bool = False, callbacks=()):
        return SimpleNamespace(
            sanity_checking=sanity,
            current_epoch=epoch,
            global_step=1,
            logger=_Logger(),
            callback_metrics={},
            log_dir=root,
            default_root_dir=root,
            callbacks=list(callbacks),
        )

    def test_scores_keep_invalid_prediction_in_denominator(self) -> None:
        scores = score_italian_predictions(_rows(), invalid_penalty=10.0)
        self.assertEqual(scores["model"]["overall"]["count"], 3)
        self.assertGreater(scores["model"]["overall"]["rmse"], 5.0)
        self.assertNotEqual(
            scores["model"]["overall"]["count"],
            scores["model"]["equal_profile"]["count"],
        )

    def test_band_profile_mae_and_skill_use_matching_reference(self) -> None:
        rows = pd.DataFrame(
            {
                "profile_id": ["a", "a", "b"],
                "depth_m": [5.0, 6.0, 150.0],
                "observation": [0.0, 0.0, 0.0],
                "model": [2.0, -2.0, 0.0],
                "raw_glorys": [1.0, -1.0, 10.0],
                "corrected_glorys": [0.0, 0.0, 0.0],
                "climatology": [0.0, 0.0, 0.0],
            }
        )
        scores = score_italian_predictions(rows, bands_m=(0.0, 10.0, 200.0))
        self.assertEqual(scores["model"]["0_10_equal_profile"]["mae"], 2.0)
        self.assertEqual(scores["model"]["0_10"]["skill_vs_raw_glorys"], -1.0)

    def test_config_verifies_registry_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "screening.parquet"
            pd.DataFrame({"profile_id": ["a"]}).to_parquet(registry)
            loaded, metadata = ItalianValidationConfig(
                self._profiles(root), screening_registry=registry
            ).load_profiles()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(metadata["profile_count"], 1)
            self.assertEqual(len(metadata["screening_registry_sha256"]), 64)

    def test_sanity_once_and_recovery_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []

            def evaluator(model, rows, steps, seed):
                calls.append((steps, seed))
                return _rows()

            callback = self._callback(root, evaluator)
            trainer = self._trainer(root, sanity=True)
            callback.on_validation_epoch_end(trainer, torch.nn.Identity())
            self.assertEqual(calls, [])
            trainer.sanity_checking = False
            callback.on_validation_epoch_end(trainer, torch.nn.Identity())
            callback.on_validation_epoch_end(trainer, torch.nn.Identity())
            self.assertEqual(calls, [(1000, 20261103)])
            resumed = self._callback(root, evaluator)
            resumed.load_state_dict(callback.state_dict())
            resumed.on_validation_epoch_end(trainer, torch.nn.Identity())
            trainer.current_epoch = 1
            resumed.on_validation_epoch_end(trainer, torch.nn.Identity())
            self.assertEqual(calls, [(1000, 20261103), (1000, 20261103)])

    def test_rng_is_restored_after_success_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run(raises: bool) -> None:
                def evaluator(model, rows, steps, seed):
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    if raises:
                        raise RuntimeError("expected evaluator failure")
                    return _rows()

                callback, trainer = self._callback(root, evaluator), self._trainer(root)
                random.seed(17)
                np.random.seed(17)
                torch.random.default_generator.manual_seed(17)
                before_python, before_numpy, before_torch = (
                    random.getstate(),
                    np.random.get_state(),
                    torch.random.get_rng_state(),
                )
                if raises:
                    with self.assertRaisesRegex(
                        RuntimeError, "expected evaluator failure"
                    ):
                        callback.on_validation_epoch_end(trainer, torch.nn.Identity())
                else:
                    callback.on_validation_epoch_end(trainer, torch.nn.Identity())
                self.assertEqual(random.getstate(), before_python)
                self.assertEqual(np.random.get_state()[0], before_numpy[0])
                self.assertTrue(
                    np.array_equal(np.random.get_state()[1], before_numpy[1])
                )
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before_torch))

            run(False)
            run(True)

    def test_artifact_fallback_does_not_access_collective_log_dir(self) -> None:
        """Rank-zero artifact writing must not enter Trainer.log_dir's broadcast."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            callback = self._callback(root, lambda *args: _rows())

            class Trainer:
                default_root_dir = root

                @property
                def log_dir(self):
                    raise AssertionError("log_dir requires all distributed ranks")

            callback._write_artifact(
                Trainer(), epoch=0, rows=_rows(), scores={"model": {}}
            )
            self.assertTrue((root / "italian_validation" / "epoch_0000.json").is_file())

    def test_metric_logging_does_not_read_lightning_callback_metrics(self) -> None:
        """Direct diagnostic logging must not trigger deferred metric collectives."""
        with tempfile.TemporaryDirectory() as directory:
            callback = self._callback(Path(directory), lambda *args: _rows())
            logger = _Logger()

            class Trainer:
                global_step = 17

                @property
                def callback_metrics(self):
                    raise AssertionError(
                        "callback_metrics may synchronize distributed state"
                    )

            trainer = Trainer()
            trainer.logger = logger
            callback._log(trainer, {"model/overall/rmse": 2.5})
            self.assertEqual(
                logger.metrics["italian_validation/model/overall/rmse"], 2.5
            )

    def test_explicit_shards_have_no_padding(self) -> None:
        length = 14
        for world_size in (1, 2, 4):
            shards = []
            for rank in range(world_size):
                with (
                    patch("torch.distributed.is_available", return_value=True),
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch("torch.distributed.get_world_size", return_value=world_size),
                    patch("torch.distributed.get_rank", return_value=rank),
                ):
                    shards.append(ItalianValidationCallback._rank_indices(length))
            self.assertEqual(
                sorted(index for shard in shards for index in shard),
                list(range(length)),
            )
            self.assertEqual(sum(map(len, shards)), length)

    def test_gloo_group_precedes_uneven_sampling_and_carries_objects(self) -> None:
        """Create the CPU group on every rank before sharding model batches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = SimpleNamespace(
                assignments=_rows(), patch_indices=[0], exclusions=pd.DataFrame()
            )
            with patch.object(
                italian_validation, "_prepare_cohort", return_value=(cohort, {})
            ):
                callback = ItalianValidationCallback(
                    ItalianValidationConfig(
                        self._profiles(root), data_config_path=root / "data.yaml"
                    )
                )

            events = []
            cpu_group = object()

            def batches():
                events.append(("batches", callback._object_collective_group))
                return iter(())

            def gather(output, local, *, group=None):
                events.append(("gather", group))
                output[0], output[1] = _rows(), local

            callback._batches = batches
            model = torch.nn.Module()
            model.model = SimpleNamespace(sampler=SimpleNamespace(num_timesteps=1000))
            model.val_sampler = SimpleNamespace(num_timesteps=100)
            trainer = self._trainer(root)
            with (
                patch("torch.distributed.is_available", return_value=True),
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=0),
                patch("torch.distributed.get_world_size", return_value=2),
                patch(
                    "torch.distributed.new_group", return_value=cpu_group
                ) as new_group,
                patch("torch.distributed.all_gather_object", side_effect=gather),
            ):
                callback.on_validation_epoch_end(trainer, model)
                trainer.current_epoch = 1
                callback.on_validation_epoch_end(trainer, model)

            new_group.assert_called_once_with(backend="gloo")
            self.assertEqual(
                events,
                [
                    ("batches", cpu_group),
                    ("gather", cpu_group),
                    ("batches", cpu_group),
                    ("gather", cpu_group),
                ],
            )

    def test_tensor_prediction_uses_ema_ddpm_and_restores_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Dataset:
                _rows = pd.DataFrame([{"date": 20160101, "grid_y0": 0, "grid_x0": 0}])

                def __getitem__(self, index):
                    field = torch.zeros((1, 2, 2), dtype=torch.float32)
                    return {
                        "x": field.clone(),
                        "x_valid_mask": torch.ones_like(field, dtype=torch.bool),
                        "x_valid_mask_1d": torch.ones((1, 2, 2), dtype=torch.bool),
                        "y": field.clone(),
                        "y_glorys": field.clone(),
                        "climatology": field.clone(),
                        "y_valid_mask": torch.ones_like(field, dtype=torch.bool),
                        "land_mask": torch.ones((1, 2, 2), dtype=torch.bool),
                    }

            assignments = pd.DataFrame(
                {
                    "profile_id": ["a"],
                    "observation": [10.0],
                    "depth_m": [0.0],
                    "patch_index": [0],
                    "row": [0],
                    "col": [0],
                    "depth_index_lower": [0],
                    "depth_index_upper": [0],
                    "depth_weight": [0.0],
                }
            )
            cohort = italian_validation._ItalianCohort(
                Dataset(), assignments, [0], {(20160101, 0, 0)}, pd.DataFrame()
            )
            with patch.object(
                italian_validation, "_prepare_cohort", return_value=(cohort, {})
            ):
                callback = ItalianValidationCallback(
                    ItalianValidationConfig(
                        self._profiles(root), data_config_path=root / "data.yaml"
                    )
                )
            ddpm, ddim = SimpleNamespace(num_timesteps=1000), SimpleNamespace(
                num_timesteps=100
            )

            precision_plugin = SimpleNamespace(active=False)
            precision_plugin.forward_context = lambda: patch.object(
                precision_plugin, "active", True
            )

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(()))
                    self.model, self.val_sampler, self.used_sampler = (
                        SimpleNamespace(sampler=ddpm),
                        ddim,
                        None,
                    )

                @property
                def device(self):
                    """Expose the Lightning device contract used by the real EMA callback."""
                    return self.weight.device

                def predict_step(self, batch, batch_idx):
                    self.assert_precision_context()
                    self.used_sampler = batch["sampler"]
                    return {
                        "y_hat_temperature_denorm": torch.full_like(
                            batch["y"], 10.0 + self.weight.item()
                        )
                    }

                @staticmethod
                def assert_precision_context():
                    if not precision_plugin.active:
                        raise AssertionError(
                            "predict_step ran outside precision context"
                        )

            model = Model()
            ema = EMA(decay=0.999, evaluate_ema_weights_instead=True)
            ema._setup_ema_weights(model)
            for value in ema._ema_model_weights.values():
                if torch.is_floating_point(value):
                    value.fill_(2.0)
            ema.replace_model_weights(model)
            trainer = self._trainer(root, callbacks=[ema])
            trainer.precision_plugin = precision_plugin
            callback.on_validation_epoch_end(trainer, model)
            self.assertIs(model.used_sampler, ddpm)
            self.assertIs(model.val_sampler, ddim)
            self.assertTrue(callback.ema_evaluation_active)
            self.assertEqual(
                trainer.logger.metrics["italian_validation/model/overall/rmse"], 2.0
            )
            artifact = json.loads(
                (root / "italian_validation" / "epoch_0000.json").read_text()
            )
            self.assertTrue(artifact["ema_evaluation_active"])
            ema.on_validation_end(trainer, model)
            self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
