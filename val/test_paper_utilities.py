"""Numerical tests for extreme, selective, and efficiency validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from val.techniques.efficiency import run as efficiency_run
from val.techniques.extremes import _thresholds, run as extremes_run
from val.techniques.selective import run as selective_run


def _context():
    """Build a small point aligned ensemble with a distinct tail event."""
    rows = []
    for profile in range(6):
        for depth in (10.0, 100.0):
            rows.append(
                {
                    "point_id": f"p{profile}-{depth}",
                    "profile_id": f"p{profile}",
                    "cluster_id": f"c{profile}",
                    "source": "synthetic",
                    "region": "r",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "observation_time": "2020-01-01",
                    "target_time": "2020-01-01",
                    "depth_m": depth,
                    "variable": "temperature",
                    "observation": 1.0 if profile == 0 else 0.0,
                    "glorys": 0.2,
                    "climatology": 0.0,
                    "independence": "known",
                    "split_role": "confirmation",
                    "nearest_context_distance_km": float(profile + 1),
                    "context_density": float(6 - profile),
                }
            )
    points = pd.DataFrame(rows)
    members = {
        "residual": np.tile(np.array([[-0.1, 0.0, 0.1, 0.2, 0.3]]), (len(points), 1))
    }
    calibration = type(
        "Bundle",
        (),
        {
            "points": points.assign(
                observation=0.1,
                profile_id=lambda frame: "cal-" + frame.profile_id.astype(str),
            ),
            "members": members,
        },
    )()
    evaluation = type(
        "Bundle", (), {"points": points, "members": members, "metadata": {}}
    )()
    return type("Context", (), {"evaluation": evaluation, "calibration": calibration})()


class PaperUtilityTests(unittest.TestCase):
    """Check scientific invariants of the three utility stages."""

    def test_extremes_scores_all_cases_and_writes_reliability(self):
        """The event case remains in the denominator of Brier scores."""
        with tempfile.TemporaryDirectory() as temporary:
            result = extremes_run(_context(), Path(temporary), {"quantile": 0.9})
            self.assertEqual(result["status"], "complete")
            table = pd.read_csv(Path(temporary) / "extreme_point_scores.csv")
            self.assertEqual(len(table), 12)
            self.assertTrue((table.event == 1).any())

    def test_extremes_runs_both_frozen_quantiles_and_baseline_scores(self):
        """Ninetieth and ninety-fifth thresholds share every verification case."""
        with tempfile.TemporaryDirectory() as temporary:
            result = extremes_run(
                _context(), Path(temporary), {"quantiles": [0.9, 0.95]}
            )
            self.assertEqual(result["status"], "complete")
            table = pd.read_csv(Path(temporary) / "extreme_point_scores.csv")
            self.assertEqual(set(table["quantile"]), {0.9, 0.95})
            self.assertIn("brier_raw_glorys", table.columns)
            self.assertIn("tail_crps_upper_empirical", table.columns)

    def test_extremes_uses_calibrated_baseline_member_array(self):
        """Calibrated GLORYS members produce finite same-case Brier scores."""
        context = _context()
        context.evaluation.members["glorys_calibrated"] = np.full(
            (len(context.evaluation.points), 4), 0.5
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = extremes_run(context, Path(temporary), {"quantiles": [0.9, 0.95]})
            self.assertEqual(result["status"], "complete")
            table = pd.read_csv(Path(temporary) / "extreme_summary.csv")
            self.assertTrue(table["brier_glorys_calibrated"].notna().all())

    def test_extremes_blocks_when_a_variable_has_no_calibration_support(self):
        """A missing salinity calibration cohort cannot borrow temperature tails."""
        context = _context()
        context.evaluation.points.loc[:, "variable"] = "salinity"
        context.evaluation.points.loc[:, "profile_id"] = (
            "eval-" + context.evaluation.points.profile_id.astype(str)
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = extremes_run(context, Path(temporary))
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["details"]["unsupported_variables"], ["salinity"])

    def test_extreme_thresholds_weight_profiles_without_averaging_depths(self):
        """A one-point profile retains half the calibration mass against 100 depths."""
        rows = []
        for depth in range(100):
            rows.append(
                {
                    "profile_id": "deep",
                    "variable": "temperature",
                    "depth_m": depth,
                    "observation": 0.0,
                    "climatology": 0.0,
                }
            )
        rows.append(
            {
                "profile_id": "sparse",
                "variable": "temperature",
                "depth_m": 1,
                "observation": 10.0,
                "climatology": 0.0,
            }
        )
        calibration = type("Bundle", (), {"points": pd.DataFrame(rows)})()
        thresholds = _thresholds(calibration, 0.9)
        key = next(
            key for key in thresholds if key[0] == "temperature" and key[2] == "upper"
        )
        self.assertEqual(thresholds[key], 10.0)

    def test_selective_risk_is_oracle_free(self):
        """Selective output records deployment covariates and no truth ranking."""
        with tempfile.TemporaryDirectory() as temporary:
            result = selective_run(_context(), Path(temporary))
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["details"]["oracle_free"])
            self.assertIn(
                "spread",
                set(
                    pd.read_csv(
                        Path(temporary) / "selective_risk_coverage.csv"
                    ).covariate
                ),
            )

    def test_efficiency_prefixes_are_monotone_in_available_members(self):
        """Unavailable prefixes are skipped while requested prefixes are retained."""
        with tempfile.TemporaryDirectory() as temporary:
            result = efficiency_run(
                _context(), Path(temporary), {"member_counts": [2, 5, 10]}
            )
            self.assertEqual(result["status"], "partial")
            counts = pd.read_csv(
                Path(temporary) / "efficiency_member_counts.csv"
            ).n_members.unique()
            np.testing.assert_array_equal(counts, [2, 5])

    def test_residual_matrix_aligns_mixed_temperature_salinity_points(self):
        """The persisted all-point residual matrix is split by variable safely."""
        context = _context()
        salinity = context.evaluation.points.copy()
        salinity["point_id"] = salinity.point_id + "-s"
        salinity["profile_id"] = salinity.profile_id + "-s"
        salinity["variable"] = "salinity"
        salinity["observation"] = 34.5
        salinity["glorys"] = 34.4
        salinity["climatology"] = 34.0
        context.evaluation.points = pd.concat(
            [context.evaluation.points, salinity], ignore_index=True
        )
        context.evaluation.members = {
            "residual": np.tile(
                np.array([[-0.1, 0.0, 0.1, 0.2, 0.3]]),
                (len(context.evaluation.points), 1),
            )
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = efficiency_run(context, Path(temporary), {"member_counts": [2]})
            self.assertEqual(result["status"], "partial")
            self.assertEqual(
                set(
                    pd.read_csv(
                        Path(temporary) / "efficiency_member_counts.csv"
                    ).variable
                ),
                {"temperature", "salinity"},
            )

    def test_member_timing_trace_is_consumed_without_glorys_cost_claim(self):
        """Recorded member timings produce measured prefix costs only."""
        context = _context()
        context.evaluation.metadata = {
            "member_timings": [
                {"member": 0, "batch": 0, "elapsed_seconds": 1.0},
                {"member": 1, "batch": 0, "elapsed_seconds": 2.0},
                {"member": 2, "batch": 0, "elapsed_seconds": 3.0},
                {"member": 3, "batch": 0, "elapsed_seconds": 4.0},
                {"member": 4, "batch": 0, "elapsed_seconds": 5.0},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = efficiency_run(context, Path(temporary), {"member_counts": [2, 5]})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                result["details"]["glorys_cost_comparison"], "not_estimated"
            )
            timing = pd.read_csv(Path(temporary) / "efficiency_timing.csv")
            self.assertEqual(timing.iloc[0].seconds, 3.0)

    def test_efficiency_ignores_conditioning_budget_timings_for_full_curve(self):
        """Full residual timing excludes slower conditioning-ablation runs."""
        context = _context()
        context.evaluation.metadata = {
            "member_timings": [
                {"member": 0, "budget": "full", "elapsed_seconds": 1.0},
                {"member": 0, "budget": "1", "elapsed_seconds": 100.0},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            efficiency_run(context, Path(temporary), {"member_counts": [2]})
            timing = pd.read_csv(Path(temporary) / "efficiency_timing.csv")
            self.assertEqual(timing.iloc[0].seconds, 1.0)
            self.assertTrue(timing.iloc[0].elapsed_measured)


if __name__ == "__main__":
    unittest.main()
