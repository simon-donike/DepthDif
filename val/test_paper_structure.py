"""Numerical tests for physical and dependence validation techniques."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from val.techniques.dependence import (
    energy_score,
    _weighted_profile_mean,
    variogram_score,
    run as dependence_run,
)
from val.techniques.physical import _profile_diagnostics, run as physical_run


class PaperStructureTests(unittest.TestCase):
    """Protect diagnostics from extrapolation and accidental marginal changes."""

    def test_layer_does_not_extrapolate(self):
        """A profile ending above 200 m cannot produce a 10-200 m layer mean."""
        result = _profile_diagnostics(
            np.array([10.0, 100.0]), np.array([20.0, 15.0]), {}
        )
        self.assertTrue(np.isnan(result["layer_mean_10_200m"]))

    def test_energy_and_variogram_are_zero_for_exact_ensemble(self):
        """Both multivariate scores vanish for a degenerate exact forecast."""
        truth = np.array([1.0, 2.0, 3.0])
        ensemble = np.repeat(truth[None, :], 3, axis=0)
        self.assertAlmostEqual(energy_score(truth, ensemble), 0.0)
        self.assertAlmostEqual(variogram_score(truth, ensemble), 0.0)

    def test_layer_mean_uses_trapezoidal_depth_weights(self):
        """Nonuniform vertical spacing must affect the functional mean."""
        values = np.array([[0.0, 1.0, 10.0]])
        weighted = _weighted_profile_mean(values, np.array([0.0, 1.0, 10.0]))[0]
        self.assertAlmostEqual(weighted, 5.0)

    def test_dependence_permutation_preserves_scalar_marginals(self):
        """Independent depth permutations preserve each depth's member marginal."""
        points = pd.DataFrame(
            [
                {
                    "profile_id": "p",
                    "variable": "temperature",
                    "depth_m": float(d),
                    "observation": float(d),
                }
                for d in (10, 50, 100, 200)
            ]
        )
        members = {"residual": np.arange(16, dtype=float).reshape(4, 4)}
        context = {"points": points, "members": members}
        with tempfile.TemporaryDirectory() as temporary:
            result = dependence_run(context, Path(temporary), {"permutations": 2})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                len(pd.read_csv(Path(temporary) / "dependence_scores.csv")), 3
            )

    def test_physical_run_writes_member_and_summary_tables(self):
        """Physical output contains model and deterministic reference rows."""
        points = pd.DataFrame(
            [
                {
                    "profile_id": "p",
                    "variable": "temperature",
                    "depth_m": float(d),
                    "observation": 20 - d / 100,
                    "glorys": 20 - d / 110,
                    "climatology": 19.0,
                }
                for d in (10, 50, 100, 200, 300)
            ]
        )
        context = {
            "points": points,
            "members": {
                "residual": np.tile(points.observation.to_numpy()[:, None], (1, 3))
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = physical_run(context, Path(temporary), {})
            self.assertIn(result["status"], {"complete", "partial"})
            self.assertTrue((Path(temporary) / "physical_summary.csv").is_file())
            self.assertIn(
                "residual",
                set(
                    pd.read_csv(
                        Path(temporary) / "physical_member_diagnostics.csv"
                    ).method
                ),
            )

    def test_spatial_case_has_intact_and_permuted_scores(self):
        """Whole-profile spatial permutations produce a separate matched case."""
        rows = []
        for profile, offset in (("a", 0.0), ("b", 10.0)):
            for depth in (10.0, 50.0, 100.0):
                rows.append(
                    {
                        "profile_id": profile,
                        "variable": "temperature",
                        "depth_m": depth,
                        "observation": offset + depth,
                        "patch_index": 1,
                        "target_time": "t",
                    }
                )
        points = pd.DataFrame(rows)
        members = np.asarray(
            [[0, 1], [1, 2], [2, 3], [10, 11], [11, 12], [12, 13]], float
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = dependence_run(
                {"points": points, "members": {"residual": members}},
                Path(temporary),
                {"permutations": 2},
            )
            self.assertEqual(result["status"], "complete")
            scores = pd.read_csv(Path(temporary) / "dependence_scores.csv")
            spatial = scores[scores.profile_id.astype(str).str.startswith("spatial:")]
            self.assertEqual(
                set(spatial.control),
                {"spatial_intact", "spatial_case_0", "spatial_case_1"},
            )
            self.assertGreater(spatial.energy_score.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
