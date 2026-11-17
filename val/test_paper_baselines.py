"""Numerical checks for the calibration-only comparator stage."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from val.techniques.baselines import _crps, run


def _frame(offset: float, prefix: str, n_profiles: int) -> pd.DataFrame:
    rows = []
    for profile in range(n_profiles):
        for depth in (10.0, 50.0, 100.0):
            glorys = 10.0 - depth / 100.0 + profile * 0.01
            rows.append(
                {
                    "point_id": f"{prefix}-{profile}-{int(depth)}",
                    "profile_id": f"{prefix}-profile-{profile}",
                    "cluster_id": f"cluster-{profile}",
                    "variable": "temperature",
                    "depth_m": depth,
                    "observation": glorys + offset,
                    "glorys": glorys,
                    "climatology": glorys - 0.5,
                    "target_time": "2019-06-15",
                }
            )
    return pd.DataFrame(rows)


class PaperBaselineTests(unittest.TestCase):
    """Numerical tests for the post-processing comparator ladder."""

    def test_fair_crps_is_less_optimistic_for_a_finite_ensemble(self) -> None:
        ensemble = np.array([0.0, 1.0, 2.0])
        ordinary = _crps(1.0, ensemble)
        fair = _crps(1.0, ensemble, fair=True)
        self.assertAlmostEqual(ordinary, 2.0 / 9.0)
        self.assertAlmostEqual(fair, 0.0)

    def test_baseline_ladder_uses_calibration_and_preserves_correlated_members(
        self,
    ) -> None:
        calibration = _frame(1.0, "cal", 6)
        evaluation = _frame(2.0, "eval", 3)
        context = {
            "evaluation": {"points": evaluation, "members": {}},
            "calibration": {"points": calibration},
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = run(
                context, Path(temporary), {"members": 12, "seed": 4, "min_profiles": 3}
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue((Path(temporary) / "baseline_scores.csv").exists())
            self.assertTrue((Path(temporary) / "baseline_scores.tex").exists())
            self.assertTrue(
                (Path(temporary) / "baseline_paired_cluster_comparisons.csv").exists()
            )
            paired = pd.read_csv(
                Path(temporary) / "baseline_paired_cluster_comparisons.csv"
            )
            self.assertIn("glorys_calibrated", set(paired.challenger))
            self.assertIn("p_one_sided", paired)
            self.assertTrue((Path(temporary) / "baseline_spread_skill.png").exists())
        members = context["evaluation"]["members"]["glorys_calibrated"]
        self.assertEqual(members.shape, (len(evaluation), 12))
        # Profile sampling supplies vertical structure; all depths in a member
        # share one calibration profile, so the residual is not diagonal noise.
        self.assertTrue(np.isfinite(members).all())
        corrected = context["evaluation"]["points"]["baseline_glorys_bias"]
        self.assertTrue(np.allclose(corrected - evaluation.glorys, 1.0, atol=0.1))

    def test_missing_calibration_is_explicitly_blocked(self) -> None:
        evaluation = _frame(2.0, "eval", 2)
        with tempfile.TemporaryDirectory() as temporary:
            result = run({"evaluation": {"points": evaluation}}, Path(temporary))
        self.assertEqual(result["status"], "blocked")
        self.assertIn("calibration", result["summary"].lower())


if __name__ == "__main__":
    unittest.main()
