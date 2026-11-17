"""Guard the published dashboard against calibration leakage and mixed support."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from val.export_dashboard import (
    evaluation_points,
    export_cohort,
    overall_scores,
    published_rows,
)


class DashboardExportTests(unittest.TestCase):
    """Exercise scientific publication boundaries without model execution."""

    def test_only_original_glorys_is_published(self):
        """Remove adjusted references at every budget without altering other scores."""
        methods = [
            "glorys",
            "glorys_packaged",
            "glorys_bias",
            "glorys_bias_corrected",
            "baseline_oi_glorys_full",
            "baseline_oi_glorys_budget_0",
            "baseline_oi_glorys_budget_8",
            "residual",
            "baseline_oi_climatology_full",
            "baseline_oi_climatology_budget_8",
            "climatology",
        ]
        frame = pd.DataFrame({"method": methods, "mae": range(len(methods))})
        expected = frame.iloc[[0, 1, 7, 10]]
        pd.testing.assert_frame_equal(published_rows(frame), expected)
        pd.testing.assert_frame_equal(
            published_rows(frame.rename(columns={"method": "baseline"})),
            expected.rename(columns={"method": "baseline"}),
        )

    def test_examples_exclude_calibration_and_preserve_bundle_indices(self):
        """Member arrays retain their alignment after evaluation-only filtering."""
        points = pd.DataFrame(
            {
                "point_id": ["a", "b", "c"],
                "profile_id": ["p1", "p2", "p2"],
                "cluster_id": ["c1", "c2", "c2"],
            }
        )
        partition = pd.DataFrame(
            {
                "point_id": ["c", "a", "b"],
                "partition": ["evaluation", "calibration", "evaluation"],
            }
        )
        selected = evaluation_points(points, partition)
        self.assertEqual(selected.index.tolist(), [1, 2])
        self.assertEqual(selected.point_id.tolist(), ["b", "c"])
        points.loc[0, "cluster_id"] = "c2"
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation_points(points, partition)

    def test_incomplete_partition_is_rejected(self):
        """Missing assignments cannot silently drop observations from publication."""
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluation_points(
                pd.DataFrame({"point_id": [1, 2]}),
                pd.DataFrame({"point_id": [1], "partition": ["evaluation"]}),
            )

    def test_overall_uses_paper_rmse_and_rejects_different_support(self):
        """Do not replace sqrt(mean profile MSE) with mean profile RMSE."""
        paired = pd.DataFrame(
            {
                "stratum": ["all", "all"],
                "group": ["all", "all"],
                "baseline": ["a", "b"],
                "metric": ["mae", "mae"],
                "baseline_score": [0.4, 0.5],
                "model_score": [0.3, 0.3],
            }
        )
        scores = overall_scores(paired).set_index("method")
        self.assertAlmostEqual(scores.loc["residual", "mae"], 0.3)
        paired.loc[1, "model_score"] = 0.2
        with self.assertRaisesRegex(ValueError, "different model support"):
            overall_scores(paired)

    def test_unfinished_cohort_has_no_placeholder_scores(self):
        """A future EN4 cohort is pending, never an apparent zero-error result."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = export_cohort(root / "en4_temperature", root / "export", "running")
            self.assertFalse(result["available"])
            self.assertNotIn("overall", result)
            self.assertTrue(
                all(stage["status"] == "pending" for stage in result["stages"])
            )
