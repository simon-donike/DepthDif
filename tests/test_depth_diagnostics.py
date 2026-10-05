"""Tests for mask-aware depth diagnostics."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from depth_recon.utils.validation_denoise import (
    compute_depth_diagnostics,
)
from depth_recon.scripts.evaluate_depth_baselines import evaluate_configured_methods


class DepthDiagnosticsTests(unittest.TestCase):
    """Validate per-depth statistics and empty-level handling."""

    def test_masked_metrics_do_not_include_invalid_fill_values(self) -> None:
        """Invalid values are excluded and valid counts are reported."""
        prediction = np.array([[[[2.0, 100.0]], [[5.0, 100.0]]]])
        target = np.array([[[[1.0, -900.0]], [[1.0, -900.0]]]])
        valid_mask = np.array([[[True, False]]])

        result = compute_depth_diagnostics(prediction, target, valid_mask)

        np.testing.assert_allclose(result["mae"], [1.0, 4.0])
        np.testing.assert_allclose(result["rmse"], [1.0, 4.0])
        np.testing.assert_allclose(result["bias"], [1.0, 4.0])
        np.testing.assert_array_equal(result["valid_count"], [1, 1])

    def test_empty_depth_is_nan(self) -> None:
        """An unsupported depth has NaN metrics and zero support."""
        prediction = np.zeros((1, 2, 1, 1), dtype=np.float32)
        target = np.ones_like(prediction)
        result = compute_depth_diagnostics(
            prediction,
            target,
            np.array([[[[True]], [[False]]]]),
            depth_dimension=1,
        )
        self.assertEqual(int(result["valid_count"][1]), 0)
        self.assertTrue(np.isnan(result["mae"][1]))
        self.assertAlmostEqual(float(result["equal_depth_mae"]), 1.0)

    def test_configured_evaluation_reuses_fixed_indices(self) -> None:
        """Configured mode evaluates mocked models on the same selected rows."""

        class FakeDataset:
            depth_axis_m = np.asarray([0.5, 1000.0])
            rows = [
                {"date": 1, "grid_y0": 0, "grid_x0": 0, "patch_id": 0},
                {"date": 2, "grid_y0": 0, "grid_x0": 1, "patch_id": 1},
                {"date": 3, "grid_y0": 1, "grid_x0": 0, "patch_id": 2},
            ]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return {
                    "y": np.full((2, 1, 1), float(index + 1), dtype=np.float32),
                    "y_valid_mask": np.ones((2, 1, 1), dtype=bool),
                    "climatology": np.zeros((2, 1, 1), dtype=np.float32),
                }

        bundle = SimpleNamespace(
            effective_data_config_path="data.yaml",
            effective_model_config_path="model.yaml",
            effective_training_config_path="training.yaml",
            data_cfg={"dataset": {}},
            model_cfg={"model": {}},
            training_cfg={},
        )

        class FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def predict_step(self, batch, batch_idx):
                return {"y_hat_temperature_denorm": batch["y"].float()}

        with (
            patch(
                "depth_recon.scripts.evaluate_depth_baselines.load_pixel_training_config",
                return_value=bundle,
            ),
            patch(
                "depth_recon.scripts.evaluate_depth_baselines.build_dataset",
                return_value=FakeDataset(),
            ),
            patch(
                "depth_recon.scripts.evaluate_depth_baselines.build_datamodule",
                return_value=object(),
            ),
            patch(
                "depth_recon.scripts.evaluate_depth_baselines.build_model",
                return_value=FakeModel(),
            ),
            patch(
                "depth_recon.scripts.evaluate_depth_baselines.load_checkpoint_weights",
                return_value="standard",
            ) as load_weights,
        ):
            report = evaluate_configured_methods(
                [("mock", Path("config.yaml"), Path("model.ckpt"))],
                sample_count=2,
                seed=7,
                batch_size=1,
                device="cpu",
            )

        self.assertIn("mock", report)
        self.assertEqual(report["mock"]["valid_count"], [2, 2])
        self.assertEqual(report["mock"]["depth_axis_m"], [0.5, 1000.0])
        self.assertEqual(report["mock"]["seed"], 7)
        self.assertTrue(load_weights.call_args.kwargs["strict"])


if __name__ == "__main__":
    unittest.main()
