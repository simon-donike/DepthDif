"""Focused tests for the validation inference output contract."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from val.inference import _prediction_points, run_model


class _FakeModel:
    """Deterministic physical-output model used without a dataset or GPU."""

    def to(self, device):
        """Return the fake model on the requested device."""
        return self

    def eval(self):
        """Return the fake model in evaluation mode."""
        return self

    def predict_step(self, batch, batch_idx):
        """Return a physical temperature tensor with an obvious ramp."""
        del batch_idx
        return {"y_hat_temperature_denorm": batch["prediction"]}


class _FakeJointModel(_FakeModel):
    """Fake model exposing distinct physical temperature and salinity fields."""

    output_fields = ("temperature", "salinity")

    def predict_step(self, batch, batch_idx):
        """Return distinct physical tensors for each requested variable."""
        del batch_idx
        return {
            "y_hat_temperature_denorm": batch["prediction"],
            "y_hat_salinity_denorm": batch["prediction"] + 30.0,
        }


def _bundle(root=None):
    """Return the minimal resolved-config object consumed by the adapter."""
    if root is not None:
        for name in ("model.yaml", "data.yaml", "training.yaml"):
            (root / name).write_text("{}\n")
        paths = tuple(
            str(root / name) for name in ("model.yaml", "data.yaml", "training.yaml")
        )
    else:
        paths = ("model.yaml", "data.yaml", "training.yaml")
    return SimpleNamespace(
        scenario="temperature",
        data_cfg={},
        training_cfg={},
        model_cfg={"model": {"output_fields": ["temperature"]}},
        effective_model_config_path=paths[0],
        effective_data_config_path=paths[1],
        effective_training_config_path=paths[2],
    )


class InferenceTests(unittest.TestCase):
    """Validate point extraction and persisted ensemble artifacts."""

    def test_prediction_points_extracts_physical_locations(self):
        """Each assignment maps to exactly one physical tensor element."""
        assignments = pd.DataFrame(
            [{"point_id": "p0", "batch_index": 0, "row": 1, "col": 0, "depth_index": 1}]
        )
        result = _prediction_points(
            torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),
            assignments,
            batch_offset=0,
        )
        self.assertEqual(result.loc[0, "prediction"], 4.0)

    def test_prediction_points_interpolates_to_raw_observation_depth(self):
        """Raw observation depths use linear interpolation on adjacent model levels."""
        assignments = pd.DataFrame(
            [
                {
                    "point_id": "p0",
                    "batch_index": 0,
                    "row": 0,
                    "col": 0,
                    "depth_index_lower": 0,
                    "depth_index_upper": 1,
                    "depth_weight": 0.25,
                }
            ]
        )
        prediction = torch.tensor([[[[2.0]], [[6.0]]]])
        result = _prediction_points(prediction, assignments, batch_offset=0)
        self.assertAlmostEqual(result.loc[0, "prediction"], 3.0)

    def test_run_model_writes_members_and_metadata(self):
        """Repeated predictions produce the required residual array contract."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            checkpoint = root / "model.ckpt"
            config_path.write_text("scenario: temperature\n")
            checkpoint.write_bytes(b"checkpoint")
            assignments = pd.DataFrame(
                [
                    {
                        "point_id": "p0",
                        "batch_index": 0,
                        "row": 0,
                        "col": 0,
                        "depth_index": 0,
                    }
                ]
            )
            batch = {"prediction": torch.tensor([[[[2.5]]]])}
            config = {
                "model": {
                    "config": str(config_path),
                    "checkpoint": str(checkpoint),
                    "weights": "ema",
                },
                "sampling": {"members": 1, "seed": 7},
                "device": "cpu",
            }
            with (
                patch(
                    "val.inference.load_pixel_training_config",
                    return_value=_bundle(root),
                ),
                patch("val.inference.build_datamodule", return_value=object()),
                patch("val.inference.build_model", return_value=_FakeModel()),
                patch("val.inference.load_checkpoint_weights", return_value="ema"),
            ):
                metadata = run_model(config, root / "out", [(batch, assignments)])
            self.assertEqual(metadata["point_count"], 1)
            self.assertFalse(metadata["uncertainty_available"])
            with __import__("numpy").load(root / "out" / "members.npz") as saved:
                self.assertEqual(saved["residual"].shape, (1, 1))
            self.assertTrue((root / "out" / "points.csv").is_file())
            self.assertTrue((root / "out" / "selection.csv").is_file())
            self.assertTrue((root / "out" / "exclusions.csv").is_file())

    def test_joint_run_uses_field_specific_physical_outputs(self):
        """Joint assignments select distinct temperature and salinity outputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            checkpoint = root / "model.ckpt"
            config_path.write_text("scenario: joint\n")
            checkpoint.write_bytes(b"checkpoint")
            bundle = _bundle(root)
            bundle.scenario = "joint"
            bundle.model_cfg = {"model": {"output_fields": ["temperature", "salinity"]}}
            assignments = pd.DataFrame(
                [
                    {
                        "point_id": "t",
                        "variable": "temperature",
                        "batch_index": 0,
                        "row": 0,
                        "col": 0,
                        "depth_index": 0,
                    },
                    {
                        "point_id": "s",
                        "variable": "salinity",
                        "batch_index": 0,
                        "row": 0,
                        "col": 0,
                        "depth_index": 0,
                    },
                ]
            )
            config = {
                "model": {
                    "config": str(config_path),
                    "checkpoint": str(checkpoint),
                    "weights": "ema",
                },
                "sampling": {"members": 2, "variables": "joint"},
                "baselines": [
                    {
                        "name": "joint",
                        "config": str(config_path),
                        "checkpoint": str(checkpoint),
                        "weights": "ema",
                    }
                ],
                "device": "cpu",
            }
            with (
                patch("val.inference.load_pixel_training_config", return_value=bundle),
                patch("val.inference.build_datamodule", return_value=object()),
                patch("val.inference.build_model", return_value=_FakeJointModel()),
                patch("val.inference.load_checkpoint_weights", return_value="ema"),
            ):
                run_model(
                    config,
                    root / "out",
                    [({"prediction": torch.tensor([[[[2.0]]]])}, assignments)],
                )
            points = pd.read_csv(root / "out" / "points.csv").set_index("point_id")
            self.assertEqual(points.loc["t", "prediction_mean"], 2.0)
            self.assertEqual(points.loc["s", "prediction_mean"], 32.0)
            self.assertEqual(points.loc["t", "baseline_joint"], 2.0)
            self.assertEqual(points.loc["s", "baseline_joint"], 32.0)

    def test_learned_baseline_requires_checkpoint(self):
        """A learned baseline cannot silently run with random weights."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            checkpoint = root / "model.ckpt"
            config_path.write_text("scenario: temperature\n")
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "model": {"config": str(config_path), "checkpoint": str(checkpoint)},
                "sampling": {"members": 2},
                "baselines": [{"name": "lstm", "config": str(config_path)}],
                "device": "cpu",
            }
            with (
                patch(
                    "val.inference.load_pixel_training_config",
                    return_value=_bundle(root),
                ),
                patch("val.inference.build_datamodule", return_value=object()),
                patch("val.inference.build_model", return_value=_FakeModel()),
                patch("val.inference.load_checkpoint_weights", return_value="ema"),
                patch("val.inference.model_requires_checkpoint", return_value=True),
            ):
                with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
                    run_model(config, root / "out", [])

    def test_run_model_rejects_missing_checkpoint_before_model_build(self):
        """Missing data artifacts fail before any model construction or GPU use."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text("scenario: temperature\n")
            config = {
                "model": {
                    "config": str(config_path),
                    "checkpoint": str(root / "missing.ckpt"),
                },
                "device": "cuda:0",
            }
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                run_model(config, root / "out", [])


if __name__ == "__main__":
    unittest.main()
