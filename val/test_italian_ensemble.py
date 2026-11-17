"""Numerical checks for the Italian probabilistic and spatial diagnostics."""

import unittest

import numpy as np
import pandas as pd
import torch

from val.metrics import _crps
from val.score_italian_ensemble import (
    lpips_layers,
    point_scores,
    score_profiles,
    spectral_error,
)


class ItalianEnsembleTests(unittest.TestCase):
    """Check score conventions and prevent sample/depth weighting regressions."""

    def test_zero_spectral_power_is_penalized_not_dropped(self):
        """A constant prediction cannot escape comparison by having zero power."""
        self.assertEqual(spectral_error(np.array([0.0]), np.array([1.0])), 12)
        self.assertEqual(spectral_error(np.array([0.0]), np.array([0.0])), 0)
        self.assertEqual(
            spectral_error(np.array([1.0, 100.0]), np.array([10.0, 10.0])), 1
        )

    def test_crps_matches_pairwise_definition(self):
        """Empirical and fair scores agree with independent pairwise calculations."""
        observations = np.array([1.0, -2.0, 3.0])
        members = np.array([[0, 1, 2], [-4, -1, 0], [3, 3, 3]], dtype=float)
        result = point_scores(observations, members)
        for index, (observation, sample) in enumerate(zip(observations, members)):
            pair = np.abs(sample[:, None] - sample[None, :]).sum()
            first = np.abs(sample - observation).mean()
            self.assertAlmostEqual(result["crps"][index], first - pair / 18)
            self.assertAlmostEqual(result["fair_crps"][index], first - pair / 12)
            np.testing.assert_allclose(
                _crps(observation, sample),
                [result["crps"][index], result["fair_crps"][index]],
            )

    def test_deterministic_score_and_equal_profile_weighting(self):
        """A densely sampled cast must not dominate the overall score."""
        points = pd.DataFrame(
            {
                "profile_id": ["a", "a", "a", "b"],
                "cluster_id": ["one"] * 4,
                "depth_m": [1, 2, 3, 4],
                "observation": [0.0] * 4,
            }
        )
        overall, depth, _ = score_profiles(
            points, {"baseline": np.array([[0], [0], [0], [4]])}
        )
        self.assertEqual(overall[0]["crps"], 2)
        self.assertEqual(depth[0]["crps"], 2)
        self.assertTrue(np.isnan(overall[0]["fair_crps"]))

    def test_lpips_fixed_scaling_rgb_and_ocean_mask(self):
        """The metric sees repeated physical layers, finite fills and fixed scaling."""

        class Distance(torch.nn.Module):
            """Inspect LPIPS inputs without downloading a network in unit tests."""

            def forward(self, first, second):
                """Return pixelwise distance while checking the RGB contract."""
                assert first.shape[1] == 3
                assert torch.equal(first[:, 0], first[:, 1])
                assert torch.equal(first[:, 1], first[:, 2])
                assert torch.isfinite(first).all()
                assert first.min() >= -1 and first.max() <= 1
                return (first - second).abs().mean(dim=1, keepdim=True)

        reference = np.zeros((2, 64, 64), dtype=np.float32)
        prediction = reference + 3.7
        valid = np.ones_like(reference, dtype=bool)
        valid[1] = False
        prediction[1] = np.nan
        result = lpips_layers(Distance(), prediction, reference, valid, "cpu")
        self.assertAlmostEqual(result[0], 0.2, places=6)
        self.assertTrue(np.isnan(result[1]))


if __name__ == "__main__":
    unittest.main()
