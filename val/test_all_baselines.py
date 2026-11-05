"""Ensure configured baselines reach the shared scientific comparisons."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from val.paper_common import Bundle, Context, method_arrays
from val.baselines import registered_baselines
from val.run import _config
from val.techniques import (
    baselines,
    conditioning,
    dependence,
    efficiency,
    extremes,
    physical,
    regional,
    selective,
)

NAMES = {"idw", "lstm", "cnn", "unet3d", "unet2d"}


def _context():
    """Give five distinct methods the same observed profiles and depth support."""
    records = []
    for profile in range(6):
        for depth in (10, 50, 100, 150, 200, 300):
            records.append(
                dict(
                    point_id=f"p{profile}:{depth}",
                    profile_id=f"p{profile}",
                    cluster_id=f"c{profile}",
                    variable="temperature",
                    source="synthetic",
                    region="r",
                    latitude=0.0,
                    longitude=0.0,
                    observation_time="2016-01-01",
                    target_time="2016-01-01",
                    depth_m=depth,
                    observation=20 - depth / 100 + profile,
                    glorys=20 - depth / 100 + profile + 1,
                    climatology=20 - depth / 100,
                    independence="unknown",
                    split_role="discovery",
                )
            )
    points = pd.DataFrame(records)
    members = {
        "residual": points.observation.to_numpy()[:, None] + np.array([-0.1, 0, 0.1])
    }
    for index, name in enumerate(sorted(NAMES)):
        points[f"baseline_{name}"] = points.observation + (index + 1) * 0.2
        members[name] = points[f"baseline_{name}"].to_numpy()[:, None]
        members[f"baseline_model_{name}_budget_1"] = members[name] + 1
    members["conditioning_1"] = members["residual"] + 0.5
    metadata = {
        "baselines": {
            name: {"status": "available", "elapsed_seconds": 1.0} for name in NAMES
        }
    }
    calibration = points.copy()
    for key in ("point_id", "profile_id", "cluster_id"):
        calibration[key] = "cal-" + calibration[key]
    config = {
        "statistics": {"bootstrap_replicates": 10, "min_profiles": 2, "min_clusters": 2}
    }
    return Context(
        Bundle(points, members, metadata), Bundle(calibration, {}, {}), config, 7
    )


class AllBaselineTests(unittest.TestCase):
    """Catch silent method omissions across score tables and conditioning budgets."""

    def test_every_default_cohort_uses_the_complete_roster(self):
        """Standalone EN4, XBT and salinity evaluations share the five methods."""
        for path in (
            "val/config.yaml",
            "val/config_xbt.yaml",
            "val/config_salinity.yaml",
        ):
            with self.subTest(path=path):
                config = _config(Path(path))
                self.assertEqual({spec["name"] for spec in config["baselines"]}, NAMES)
                self.assertEqual(registered_baselines(config), config["baselines"])

    def test_full_methods_exclude_conditioning_budget_runs(self):
        """A reduced-input run cannot be mistaken for an additional baseline."""
        context = _context()
        methods = method_arrays(context.evaluation.points, context.evaluation.members)
        self.assertTrue(NAMES.issubset(methods))
        self.assertFalse(
            any(
                "budget_" in name or name.startswith("conditioning_")
                for name in methods
            )
        )

    def test_all_scientific_stages_retain_each_baseline(self):
        """Methods appear in paired, physical, tail, selective and cost scorecards."""
        cases = (
            (
                baselines,
                "baseline_scores.csv",
                "method",
                {
                    "members": 4,
                    "min_profiles": 2,
                    "min_clusters": 2,
                    "bootstrap_replicates": 10,
                },
            ),
            (regional, "paired_comparisons.csv", "baseline", {}),
            (physical, "physical_member_diagnostics.csv", "method", {}),
            (dependence, "dependence_scores.csv", "method", {"permutations": 1}),
            (
                efficiency,
                "efficiency_all_methods.csv",
                "method",
                {"member_counts": [2, 3]},
            ),
            (selective, "selective_risk_coverage.csv", "method", {}),
            (conditioning, "conditioning_scores.csv", "method", {}),
        )
        with tempfile.TemporaryDirectory() as directory:
            for module, filename, column, options in cases:
                with self.subTest(stage=module.__name__):
                    output = Path(directory) / module.__name__.rsplit(".", 1)[-1]
                    module.run(_context(), output, options)
                    table = pd.read_csv(output / filename)
                    self.assertTrue(
                        NAMES.issubset(set(table[column])), set(table[column])
                    )
            output = Path(directory) / "extremes"
            extremes.run(_context(), output, {"quantiles": [0.9]})
            table = pd.read_csv(output / "extreme_summary.csv")
            self.assertTrue(all(f"brier_{name}" in table for name in NAMES))


if __name__ == "__main__":
    unittest.main()
