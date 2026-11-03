"""End-to-end contracts for leakage control and sequential publication reports."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from val.paper import _assert_disjoint, load_bundle, make_context
from val.paper_common import Bundle, Context, point_crps, save_table
from val.techniques.comparators import run as import_comparators
from val.run import validate


class PaperWorkflowTests(unittest.TestCase):
    """Protect scientific separation rather than mirroring export code."""

    def _bundle(self, role="discovery"):
        """Construct several independent casts with linked T/S rows."""
        points = pd.DataFrame(
            [
                dict(
                    point_id=f"{index}:{variable}",
                    profile_id=str(index),
                    cluster_id=str(index // 2),
                    variable=variable,
                    split_role=role,
                    observation_time="2020-01-01",
                    latitude=float(index),
                    longitude=0.0,
                    depth_m=10.0,
                )
                for index in range(8)
                for variable in ("temperature", "salinity")
            ]
        )
        return Bundle(
            points, {"residual": np.arange(len(points) * 3).reshape(-1, 3)}, {}
        )

    def test_cluster_partition_preserves_joint_member_alignment(self):
        """Whole clusters and linked variables cannot straddle fitted/test units."""
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as directory:
            context = make_context(bundle, {}, Path(directory))
        self.assertIsNotNone(context.calibration)
        _assert_disjoint(context.evaluation, context.calibration)
        for subset in (context.evaluation, context.calibration):
            for row, point in enumerate(subset.points.point_id):
                original = bundle.points.index[bundle.points.point_id.eq(point)][0]
                np.testing.assert_array_equal(
                    subset.members["residual"][row],
                    bundle.members["residual"][original],
                )

    def test_confirmation_never_automatically_fits_labels(self):
        """An untouched confirmation set requires separate calibration inputs."""
        with tempfile.TemporaryDirectory() as directory:
            context = make_context(self._bundle("confirmation"), {}, Path(directory))
        self.assertIsNone(context.calibration)
        self.assertEqual(len(context.evaluation.points), 16)

    def test_aliasing_a_physical_cast_is_detected(self):
        """Renaming identifiers cannot hide duplicate calibration observations."""
        evaluation, calibration = self._bundle(), self._bundle()
        for key in ("point_id", "profile_id", "cluster_id"):
            calibration.points[key] = "renamed:" + calibration.points[key]
        with self.assertRaisesRegex(ValueError, "physical observation"):
            _assert_disjoint(evaluation, calibration)

    def test_fair_crps_and_portable_table(self):
        """Fair CRPS uses distinct-member pairs and LaTeX escapes real labels."""
        members = np.array([[0.0, 2.0]])
        self.assertAlmostEqual(point_crps(members, np.array([1.0]))[0], 0.5)
        self.assertAlmostEqual(point_crps(members, np.array([1.0]), fair=True)[0], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            save_table(
                pd.DataFrame({"model_name": ["T&S_90%"]}), Path(directory), "table"
            )
            tex = (Path(directory) / "table.tex").read_text()
            self.assertIn(r"T\&S\_90\%", tex)

    def test_real_bundle_requires_hashes(self):
        """Unrecorded scientific inputs cannot acquire frozen provenance by reuse."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text(json.dumps({"demo": False}))
            with self.assertRaisesRegex(ValueError, "recorded SHA256"):
                load_bundle(path)

    def test_external_comparator_aligns_ids_and_rejects_missing_cases(self):
        """Published model outputs are compared on exact cases, irrespective of file order."""
        bundle = self._bundle()
        bundle.points["observation"] = np.arange(len(bundle.points), dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            predictions = bundle.points[["point_id", "observation"]].iloc[::-1].copy()
            predictions["prediction"] = predictions.observation + 1
            source = path / "predictions.csv"
            predictions.to_csv(source, index=False)
            spec = {
                "name": "published_model",
                "points": str(source),
                "sha256": {"points": hashlib.sha256(source.read_bytes()).hexdigest()},
                "provenance": dict(
                    method="example",
                    reference="doi:example",
                    inputs="SST and sparse T/S",
                    training_data="separate",
                    checkpoint="recorded-checkpoint",
                ),
            }
            result = import_comparators(
                Context(bundle, None, {}, 1), path / "report", {"predictions": [spec]}
            )
            self.assertEqual(result["status"], "complete")
            np.testing.assert_allclose(
                bundle.points.baseline_published_model, bundle.points.observation + 1
            )
            spec["name"] = "missing_model"
            predictions.iloc[1:].to_csv(source, index=False)
            spec["sha256"]["points"] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "lacks 1"):
                import_comparators(
                    Context(bundle, None, {}, 1),
                    path / "other",
                    {"predictions": [spec]},
                )

    def test_campaign_continues_after_failure_and_publishes_status(self):
        """One cohort failure cannot hide the status of subsequent cohorts."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path / "campaign.yaml"
            manifest.write_text(
                "cohorts:\n  - {name: first, config: val/config_salinity.yaml}\n  - {name: second, config: val/config_salinity.yaml}\n"
            )
            with (
                patch("val.run.preflight", return_value={"ready": True, "issues": []}),
                patch(
                    "val.run.evaluate",
                    side_effect=[
                        RuntimeError("injected failure"),
                        {"paper_suite": {"status": "partial"}},
                    ],
                ) as evaluate,
            ):
                with self.assertRaisesRegex(RuntimeError, "cohort failed"):
                    validate(manifest, path)
                self.assertEqual(evaluate.call_count, 2)
            statuses = pd.read_csv(path / "cohorts.csv")
            self.assertEqual(list(statuses.status), ["failed", "partial"])
            self.assertTrue((path / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
