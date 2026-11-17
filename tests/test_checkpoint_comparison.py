"""Checks for matched inputs and honest checkpoint-comparison metrics."""

import unittest

import numpy as np
import torch

from depth_recon.scripts.compare_checkpoints import (
    error_statistics,
    holdout_profiles,
    summarize_statistics,
)


class CheckpointComparisonTests(unittest.TestCase):
    """Protect against profile leakage and silently excluding bad predictions."""

    def test_holdout_removes_all_depths_and_mask_representations(self):
        """Hidden grid columns are absent from values and both support masks."""
        batch = {
            "x": torch.ones(2, 3, 2, 2),
            "x_valid_mask": torch.ones(2, 3, 2, 2, dtype=torch.bool),
            "x_valid_mask_1d": torch.ones(2, 1, 2, 2, dtype=torch.bool),
        }
        # A singleton remains usable conditioning, but contributes no held-out score.
        batch["x_valid_mask"][1] = False
        batch["x_valid_mask"][1, :, 0, 0] = True
        original, hidden, locations = holdout_profiles(batch, [10, 11], 7, 0.5)
        self.assertEqual(int(hidden[0].sum()), 6)
        self.assertEqual(int(hidden[1].sum()), 0)
        self.assertTrue(torch.all(original[hidden] == 1))
        self.assertTrue(torch.all(batch["x"][hidden] == 0))
        self.assertFalse(batch["x_valid_mask"][hidden].any())
        self.assertTrue(
            torch.equal(
                batch["x_valid_mask_1d"], batch["x_valid_mask"].any(1, keepdim=True)
            )
        )
        repeated = {
            "x": original.clone(),
            "x_valid_mask": batch["x_valid_mask"] | hidden,
        }
        _, repeated_hidden, repeated_locations = holdout_profiles(
            repeated, [10, 11], 7, 0.5
        )
        self.assertTrue(torch.equal(hidden, repeated_hidden))
        self.assertEqual(locations, repeated_locations)

    def test_metric_pools_support_and_rejects_nonfinite_predictions(self):
        """Depth and pixel weighting differ, and bad scored values fail loudly."""
        target = np.zeros((1, 2, 1, 2), dtype=np.float32)
        prediction = np.array([[[[1, 3]], [[4, 999]]]], dtype=np.float32)
        mask = np.array([[[[True, True]], [[True, False]]]])
        stats = error_statistics(prediction, target, mask)
        result = summarize_statistics(stats, [0, 100])
        self.assertEqual(result["equal_depth_mae_c"], 3)
        self.assertAlmostEqual(result["pixel_mae_c"], 8 / 3)
        prediction[0, 1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite prediction"):
            error_statistics(prediction, target, mask)


if __name__ == "__main__":
    unittest.main()
