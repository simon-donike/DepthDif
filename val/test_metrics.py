import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from val.metrics import (
    _cluster_bootstrap,
    _crps,
    _rank_mass,
    score_bundle,
    validate_bundle,
)


class MetricsTests(unittest.TestCase):
    def points(self, variables=("temperature",), profiles_per_variable=12):
        """Build a bundle with unequal cluster sizes and two depths per profile."""
        rows = []
        for variable_index, variable in enumerate(variables):
            for profile in range(profiles_per_variable):
                cluster = 0 if profile < 5 else 1 + (profile - 5) // 2
                for depth_index, depth in enumerate((5.0, 75.0)):
                    observation = 10 * variable_index + profile + depth_index
                    rows.append(
                        {
                            "point_id": len(rows),
                            "profile_id": f"{variable}-{profile}",
                            "cluster_id": f"{variable}-cluster-{cluster}",
                            "source": "xbt",
                            "region": "southern_ocean",
                            "latitude": -55.0,
                            "longitude": 10.0,
                            "observation_time": "2020-01-01",
                            "target_time": "2020-01-01",
                            "depth_m": depth,
                            "variable": variable,
                            "observation": observation,
                            "glorys": observation + 1.0,
                            "climatology": observation + 2.0,
                            "baseline_ridge": observation + 0.5,
                            "independence": "verified_not_assimilated",
                            "split_role": "confirmation",
                        }
                    )
        return pd.DataFrame(rows)

    def test_crps_and_tied_rank_mass(self):
        self.assertAlmostEqual(_crps(0, np.array([-1.0, 1.0]))[0], 0.5)
        self.assertAlmostEqual(_crps(0, np.array([-1.0, 1.0]))[1], 0.0)
        self.assertEqual(_rank_mass(3, np.array([1.0, 2.0])).tolist(), [0, 0, 1])
        tied = _rank_mass(1, np.array([1.0, 1.0, 2.0]))
        self.assertAlmostEqual(tied.sum(), 1.0)
        self.assertEqual(np.count_nonzero(tied), 3)

    def test_unequal_clusters_preserve_profile_weighted_estimand(self):
        profile = pd.DataFrame(
            {
                "cluster_id": ["large"] * 9 + ["small"],
                "profile_id": list(range(10)),
                "model_loss": [0.0] * 9 + [1.0],
                "baseline_loss": [1.0] * 9 + [0.0],
            }
        )
        result = _cluster_bootstrap(profile, 50, 4, 2, 2, lambda values: values)
        self.assertAlmostEqual(result["difference"], -0.8)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["profiles"], 10)

    def test_validation_checks_baselines_and_threshold_provenance(self):
        points = self.points()
        members = np.ones((len(points), 2))
        points.loc[0, "baseline_ridge"] = np.nan
        with self.assertRaisesRegex(ValueError, "baseline_ridge"):
            validate_bundle(points, members)
        points = self.points()
        points["threshold_low"], points["threshold_high"] = 4.0, 20.0
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_bundle(points, members)
        validate_bundle(
            points,
            members,
            {"thresholds_external": True, "thresholds_provenance": "training-only"},
        )

    def test_variable_depth_groups_are_never_pooled(self):
        points = self.points(("temperature", "salinity"))
        members = np.c_[points.observation - 0.2, points.observation + 0.2]
        with tempfile.TemporaryDirectory() as temporary:
            score_bundle(
                points,
                members,
                Path(temporary),
                bootstrap_replicates=20,
                min_clusters=2,
                min_profiles=2,
            )
            scores = pd.read_csv(Path(temporary) / "scores.csv")
            bands = scores[scores.group_level == "depth_band"]
            self.assertEqual(set(bands.variable), {"temperature", "salinity"})
            self.assertEqual(
                bands.groupby(["variable", "depth_band"]).n_points.max().max(), 12
            )
            counts = pd.read_csv(Path(temporary) / "counts.csv")
            self.assertFalse(counts.duplicated().any())

    def test_baselines_widths_brier_safe_skill_and_demo_report(self):
        points = self.points()
        points.loc[points.profile_id == "temperature-0", "glorys"] += 10
        points["threshold_low"] = points.observation - 0.1
        points["threshold_high"] = points.observation + 0.1
        members = np.c_[points.observation - 0.2, points.observation + 0.2]
        metadata = {
            "thresholds_external": True,
            "thresholds_provenance": "training-only quantiles",
            "demo": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            score_bundle(
                points,
                members,
                output,
                bootstrap_replicates=20,
                min_clusters=2,
                min_profiles=2,
                metadata=metadata,
            )
            scores = pd.read_csv(output / "scores.csv")
            overall = scores[scores.group_level == "all"]
            self.assertIn("baseline_ridge", set(overall.method))
            self.assertTrue(
                np.isfinite(overall[overall.method == "baseline_ridge"].mae).all()
            )
            self.assertTrue(overall.brier_low.notna().all())
            self.assertFalse(np.isinf(scores.skill_vs_glorys).any())
            reliability = pd.read_csv(output / "reliability.csv")
            self.assertIn("width_90", set(reliability.metric))
            ranks = reliability[reliability.metric.str.startswith("rank_histogram_")]
            self.assertAlmostEqual(ranks[ranks.group_level == "all"].score.sum(), 1.0)
            self.assertIn("DEMO DATA", (output / "report.html").read_text())
            self.assertIn("<svg", (output / "report.html").read_text())

    def test_perfect_glorys_does_not_create_infinite_skill(self):
        points = self.points()
        points["glorys"] = points.observation
        members = np.c_[points.observation, points.observation]
        with tempfile.TemporaryDirectory() as temporary:
            score_bundle(
                points,
                members,
                Path(temporary),
                bootstrap_replicates=20,
                min_clusters=2,
                min_profiles=2,
            )
            scores = pd.read_csv(Path(temporary) / "scores.csv")
            self.assertFalse(np.isinf(scores.skill_vs_glorys).any())
            self.assertTrue(scores.skill_vs_glorys.isna().all())

    def test_support_gates_holm_and_crps_inference(self):
        points = self.points(profiles_per_variable=14)
        members = np.c_[points.observation - 0.1, points.observation + 0.1]
        gates = {
            "checkpoint_verified": True,
            "training_split_verified": True,
            "conditioning_exclusion_verified": True,
            "confirmation_untouched": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = score_bundle(
                points,
                members,
                output,
                bootstrap_replicates=100,
                min_clusters=2,
                min_profiles=2,
                metadata=gates,
            )
            comparisons = pd.read_csv(output / "comparisons.csv")
            self.assertIn("crps_ensemble", set(comparisons.endpoint))
            self.assertEqual(set(comparisons.evidence_class), {"exploratory"})
            self.assertTrue(comparisons.p_holm.notna().all())
            declared = comparisons.iloc[0].comparison_id
            second = score_bundle(
                points,
                members,
                output,
                bootstrap_replicates=100,
                min_clusters=2,
                min_profiles=2,
                metadata=gates | {"confirmatory_endpoints": [declared]},
            )
            comparisons = pd.read_csv(output / "comparisons.csv")
            selected = comparisons[comparisons.comparison_id == declared].iloc[0]
            self.assertEqual(selected.evidence_class, "confirmatory")
            self.assertTrue(np.isfinite(selected.p_holm))
            self.assertTrue(first["confirmation_eligible"])
            self.assertEqual(second["confirmatory_comparisons"], 1)

    def test_aggregated_depth_and_underpowered_outputs(self):
        points = self.points(profiles_per_variable=4)
        members = np.c_[points.observation, points.observation]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            score_bundle(
                points,
                members,
                output,
                bootstrap_replicates=10,
                min_clusters=20,
                min_profiles=30,
            )
            by_depth = pd.read_csv(output / "by_depth.csv")
            self.assertLess(len(by_depth), len(points))
            self.assertNotIn("point_id", by_depth.columns)
            comparisons = pd.read_csv(output / "comparisons.csv")
            self.assertEqual(set(comparisons.decision), {"underpowered"})
            self.assertTrue((output / "regional_decisions.csv").exists())


if __name__ == "__main__":
    unittest.main()
