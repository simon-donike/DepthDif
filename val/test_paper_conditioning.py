"""Tests for nested conditioning masks and paper-stage score outputs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from val.conditioning import nested_selections
from val.conditioning import local_oi_predictions
from val.inference import run_model
from val.techniques.conditioning import _scores, run


class ConditioningTests(unittest.TestCase):
    """Keep profile-budget selection reproducible and leakage-safe."""

    def test_nested_masks_remove_whole_temperature_and_salinity_locations(self):
        """Retained budgets mask whole locations in both fields."""
        batch = {
            "x": torch.ones(1, 3, 2, 3),
            "x_salinity": torch.ones(1, 3, 2, 3),
            "x_valid_mask": torch.ones(1, 3, 2, 3, dtype=torch.bool),
            "x_salinity_valid_mask": torch.ones(1, 3, 2, 3, dtype=torch.bool),
            "x_valid_mask_1d": torch.ones(1, 1, 2, 3, dtype=torch.bool),
            "x_salinity_valid_mask_1d": torch.ones(1, 1, 2, 3, dtype=torch.bool),
            "surface": torch.arange(6).reshape(1, 1, 2, 3),
        }
        assignments = pd.DataFrame(
            [{"batch_index": 0, "row": 0, "col": 0, "point_id": "target"}]
        )
        selections = nested_selections(batch, assignments, [0, 1, 2], seed=11)
        self.assertEqual(selections[0].selected, 0)
        self.assertEqual(selections[1].selected, 1)
        self.assertEqual(selections[2].selected, 2)
        for selection in selections:
            self.assertTrue(torch.equal(selection.batch["surface"], batch["surface"]))
        self.assertFalse(selections[1].batch["x_valid_mask"].logical_not().sum() == 0)
        self.assertTrue(
            torch.equal(
                selections[1].batch["x_valid_mask"],
                selections[1].batch["x_salinity_valid_mask"],
            )
        )
        self.assertFalse(selections[1].batch["x_valid_mask"][0, :, 0, 0].any())

    def test_selection_is_reproducible_and_target_is_protected(self):
        """The same batch seed yields identical nested selections."""
        batch = {
            "x": torch.ones(1, 2, 1, 3),
            "x_valid_mask": torch.ones(1, 2, 1, 3, dtype=torch.bool),
        }
        assignments = pd.DataFrame(
            [{"batch_index": 0, "row": 0, "col": 1, "point_id": "target"}]
        )
        first = nested_selections(batch, assignments, [1, 2], seed=4)
        second = nested_selections(batch, assignments, [1, 2], seed=4)
        for left, right in zip(first, second):
            self.assertTrue(
                torch.equal(left.batch["x_valid_mask"], right.batch["x_valid_mask"])
            )
            self.assertFalse(left.batch["x_valid_mask"][0, :, 0, 1].any())

    def test_stage_scores_captured_budget_arrays(self):
        """The paper stage emits a table and status for captured arrays."""
        points = pd.DataFrame(
            {
                "point_id": ["a", "b"],
                "profile_id": ["p1", "p2"],
                "variable": ["temperature", "temperature"],
                "region": ["r", "r"],
                "observation": [1.0, 2.0],
                "glorys": [1.5, 1.5],
            }
        )
        context = SimpleNamespace(
            evaluation=SimpleNamespace(
                points=points,
                members={
                    "residual": np.array([[1.1, 0.9], [1.9, 2.1]]),
                    "conditioning_0": np.array([[1.4, 1.6], [1.4, 1.6]]),
                },
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run(context, Path(directory))
            self.assertEqual(result["status"], "complete")
            output = pd.read_csv(Path(directory) / "conditioning_scores.csv")
            self.assertEqual(set(output.budget), {"full", "0"})

    def test_stage_blocks_without_budget_capture(self):
        """A full ensemble alone cannot be reported as a conditioning curve."""
        points = pd.DataFrame(
            {
                "point_id": ["a"],
                "profile_id": ["p1"],
                "variable": ["temperature"],
                "region": ["r"],
                "observation": [1.0],
                "glorys": [1.5],
                "climatology": [1.2],
            }
        )
        context = SimpleNamespace(
            evaluation=SimpleNamespace(
                points=points, members={"residual": np.ones((1, 2))}
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run(context, Path(directory))["status"], "blocked")

    def test_joint_profile_id_is_separated_by_variable(self):
        """Temperature and salinity rows from one cast keep separate scores."""
        points = pd.DataFrame(
            {
                "point_id": ["t", "s"],
                "profile_id": ["cast", "cast"],
                "variable": ["temperature", "salinity"],
                "region": ["r", "r"],
                "observation": [1.0, 30.0],
                "glorys": [1.5, 30.5],
            }
        )
        scored = _scores(points, np.array([[1.0, 1.0], [30.0, 30.0]]), "0")
        self.assertEqual(set(scored.variable), {"temperature", "salinity"})
        self.assertEqual(len(scored), 2)

    def test_local_oi_budget_zero_is_background_and_known_innovation_updates(self):
        """OI returns the background with no context and responds to one profile."""
        from depth_recon.utils.normalizations import temperature_normalize

        background = torch.full((1, 2, 1, 2), 10.0)
        observed = background.clone()
        observed[0, 0, 0, 1] = 12.0
        batch = {
            "x": temperature_normalize("norm", observed),
            "x_valid_mask": torch.tensor([[[[False, True]], [[False, True]]]]),
            "y_glorys": temperature_normalize("norm", background),
        }
        assignments = pd.DataFrame(
            [
                {
                    "point_id": "target",
                    "variable": "temperature",
                    "batch_index": 0,
                    "row": 0,
                    "col": 0,
                    "depth_index_lower": 0,
                    "depth_index_upper": 1,
                    "depth_weight": 0.5,
                    "depth_m": 50.0,
                    "latitude": 0.0,
                }
            ]
        )
        # Real common-support filtering leaves sparse labels, not 0..N-1.
        assignments.index = [55]
        empty = torch.zeros(1, 1, 1, 2, dtype=torch.bool)
        retained = torch.tensor([[[False, True]]])
        zero = local_oi_predictions(
            batch,
            assignments,
            empty,
            background="glorys",
            variables=["temperature"],
            depth_axis_m=[0.0, 100.0],
        )
        updated = local_oi_predictions(
            batch,
            assignments,
            retained,
            background="glorys",
            variables=["temperature"],
            depth_axis_m=[0.0, 100.0],
        )
        self.assertAlmostEqual(float(zero[0]), 10.0, places=5)
        self.assertGreater(float(updated[0]), 11.0)
        self.assertLess(float(updated[0]), 11.3)

    def test_inference_persists_budget_ensembles_with_common_seeds(self):
        """The inference adapter captures aligned arrays for each requested budget."""
        from val.test_inference import _FakeModel, _bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, checkpoint = root / "config.yaml", root / "model.ckpt"
            config_path.write_text("scenario: temperature\n")
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "model": {
                    "config": str(config_path),
                    "checkpoint": str(checkpoint),
                    "weights": "ema",
                },
                "sampling": {"members": 2, "seed": 7},
                "paper": {"conditioning": {"enabled": True, "budgets": [0, 1]}},
                "device": "cpu",
            }
            assignments = pd.DataFrame(
                [
                    {
                        "point_id": "p0",
                        "variable": "temperature",
                        "batch_index": 0,
                        "row": 0,
                        "col": 0,
                        "depth_index": 0,
                    }
                ]
            )
            batch = {
                "prediction": torch.tensor([[[[2.5]]]]),
                "x": torch.ones(1, 2, 1, 1),
                "x_valid_mask": torch.ones(1, 2, 1, 1, dtype=torch.bool),
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
                run_model(config, root / "out", [(batch, assignments)])
            with np.load(root / "out" / "members.npz") as saved:
                self.assertEqual(saved["residual"].shape, (1, 2))
                self.assertEqual(saved["conditioning_0"].shape, (1, 2))
                self.assertEqual(saved["conditioning_1"].shape, (1, 2))
            self.assertTrue((root / "out" / "conditioning_selection.csv").is_file())


if __name__ == "__main__":
    unittest.main()
