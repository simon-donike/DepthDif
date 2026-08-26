from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from train import (
    build_early_stopping_callback,
    build_en4_candidate_validation_callback,
    build_hard_region_validation_callback,
    build_wandb_logger,
    load_weights_only_checkpoint,
    resolve_load_checkpoint_only,
    resolve_resume_ckpt_path,
)


class _TinyCheckpointModule(torch.nn.Module):
    """Tiny module with one parameter and one buffer for checkpoint loading tests."""

    def __init__(self) -> None:
        """Initialize deterministic raw module state."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("counter", torch.tensor([1], dtype=torch.long))


class TestTrainCheckpointConfig(unittest.TestCase):
    def test_early_stopping_callback_uses_configured_validation_metric(self) -> None:
        callback = build_early_stopping_callback(
            {
                "trainer": {
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "val/loss_ckpt",
                        "mode": "min",
                        "patience": 5,
                        "min_delta": 0.01,
                    }
                }
            }
        )

        self.assertIsNotNone(callback)
        self.assertEqual(callback.monitor, "val/loss_ckpt")
        self.assertEqual(callback.patience, 5)
        self.assertAlmostEqual(callback.min_delta, -0.01)
        self.assertIsNone(
            build_early_stopping_callback(
                {"trainer": {"early_stopping": {"enabled": False}}}
            )
        )

    def test_wandb_logger_forwards_suite_identity_and_resume_metadata(self) -> None:
        model = MagicMock()
        training_cfg = {
            "wandb": {
                "offline": False,
                "project": "DepthDif_Simon",
                "entity": "esa-phi-lab",
                "run_name": "baseline-temperature",
                "run_id": "stable-id",
                "resume": "allow",
                "group": "baseline-2016",
                "job_type": "training",
                "tags": ["baseline", "temperature"],
                "watch_gradients": False,
                "watch_parameters": False,
                "log_model": False,
            }
        }
        with patch("train.WandbLogger") as logger_class:
            logger = build_wandb_logger(training_cfg, model, run_config={"seed": 7})

        self.assertIs(logger, logger_class.return_value)
        logger_class.assert_called_once_with(
            offline=False,
            project="DepthDif_Simon",
            entity="esa-phi-lab",
            name="baseline-temperature",
            id="stable-id",
            resume="allow",
            group="baseline-2016",
            job_type="training",
            tags=["baseline", "temperature"],
            log_model=False,
            config={"seed": 7},
        )

    def test_hard_region_callback_builder_requires_validation_year_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            regions_path = Path(tmpdir) / "regions.yaml"
            regions_path.touch()
            val_dataset = MagicMock()
            expected_callback = object()
            training_cfg = {
                "training": {
                    "hard_region_eval": {
                        "enabled": True,
                        "regions_path": str(regions_path),
                        "evaluation_year": 2016,
                        "max_patches_per_region": 2,
                        "seed": 11,
                    }
                }
            }

            with patch(
                "train.HardRegionValidationCallback", return_value=expected_callback
            ) as callback_class:
                callback = build_hard_region_validation_callback(
                    val_dataset=val_dataset,
                    data_cfg={"split": {"val_year": 2016}},
                    training_cfg=training_cfg,
                )

            self.assertIs(callback, expected_callback)
            callback_class.assert_called_once_with(
                dataset=val_dataset,
                regions_path=regions_path,
                evaluation_year=2016,
                max_patches_per_region=2,
                random_seed=11,
                image_depths_m=(0.0, 100.0, 500.0),
            )
            with self.assertRaisesRegex(ValueError, "must match"):
                build_hard_region_validation_callback(
                    val_dataset=val_dataset,
                    data_cfg={"split": {"val_year": 2018}},
                    training_cfg=training_cfg,
                )

    def test_en4_candidate_callback_builder_selects_configured_week(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / "candidates.parquet"
            candidate_path.touch()
            val_dataset = MagicMock()
            val_dataset._rows = pd.DataFrame({"date": [20160624]})
            val_dataset.root_dir = Path(tmpdir)
            val_dataset.argo_store = object()
            candidates = pd.DataFrame(
                {"date": [20160624], "grid_row": [1], "grid_col": [2]}
            )
            expected_callback = object()
            training_cfg = {
                "training": {
                    "en4_candidate_eval": {
                        "enabled": True,
                        "candidate_profiles_path": str(candidate_path),
                        "iso_week": 25,
                        "min_input_profiles": 8,
                        "image_depths_m": [0.0, 100.0, 500.0],
                    }
                }
            }

            with (
                patch("train.load_dataset_context", return_value=object()),
                patch("train.load_en4_candidate_profiles", return_value=candidates),
                patch(
                    "train.EN4CandidateValidationCallback",
                    return_value=expected_callback,
                ) as callback_cls,
            ):
                callback = build_en4_candidate_validation_callback(
                    val_dataset=val_dataset,
                    data_cfg={"split": {"val_year": 2016}},
                    training_cfg=training_cfg,
                )

            self.assertIs(callback, expected_callback)
            callback_cls.assert_called_once()
            callback_kwargs = callback_cls.call_args.kwargs
            self.assertIs(callback_kwargs["candidate_df"], candidates)
            self.assertEqual(callback_kwargs["min_input_profiles"], 8)
            self.assertEqual(callback_kwargs["image_depths_m"], (0.0, 100.0, 500.0))

    def test_resume_checkpoint_false_starts_from_scratch(self) -> None:
        model_cfg = {"model": {"resume_checkpoint": False}}

        self.assertIsNone(resolve_resume_ckpt_path(model_cfg))

    def test_resume_checkpoint_path_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "last.ckpt"
            ckpt_path.touch()
            model_cfg = {"model": {"resume_checkpoint": str(ckpt_path)}}

            self.assertEqual(resolve_resume_ckpt_path(model_cfg), str(ckpt_path))

    def test_load_checkpoint_only_is_boolean_mode(self) -> None:
        self.assertTrue(
            resolve_load_checkpoint_only({"model": {"load_checkpoint_only": True}})
        )
        self.assertFalse(
            resolve_load_checkpoint_only({"model": {"load_checkpoint_only": False}})
        )

    def test_load_checkpoint_only_rejects_checkpoint_paths(self) -> None:
        model_cfg = {"model": {"load_checkpoint_only": "weights.ckpt"}}

        with self.assertRaisesRegex(
            ValueError, "model.load_checkpoint_only must be true or false"
        ):
            resolve_load_checkpoint_only(model_cfg)

    def test_weights_only_checkpoint_prefers_ema_state_when_present(self) -> None:
        module = _TinyCheckpointModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ema.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "weight": torch.tensor([5.0]),
                        "counter": torch.tensor([2], dtype=torch.long),
                    },
                    "callbacks": {
                        "depth_recon.models.diffusion.EMA": {
                            "ema_weights": {
                                "weight": torch.tensor([3.0]),
                                "counter": torch.tensor([4], dtype=torch.long),
                            }
                        }
                    },
                },
                checkpoint_path,
            )

            weight_source = load_weights_only_checkpoint(module, str(checkpoint_path))

        self.assertEqual(weight_source, "ema")
        self.assertTrue(torch.allclose(module.weight.detach(), torch.tensor([3.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([4])))


if __name__ == "__main__":
    unittest.main()
