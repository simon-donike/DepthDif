"""Scientific matching and inference isolation for the released LSTM diagnostic."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from val.cavaiola import LABEL, _file, predict, run
from val.paper_common import Bundle, Context
from val.techniques.comparators import run as comparators_run


class CavaiolaTests(unittest.TestCase):
    """Keep training-era results labelled and compare only identical support."""

    def _fixture(self, root):
        """Create a frozen cast with unsupported depths and a known temperature."""
        points = pd.DataFrame(
            {
                "point_id": [f"p:{depth}" for depth in (0, 5, 7.5, 200, 250)],
                "profile_id": "p",
                "cluster_id": "cruise",
                "variable": "temperature",
                "observation_time": "2016-01-01T12:00:00Z",
                "latitude": 45.0,
                "longitude": -30.0,
                "depth_m": [0, 5, 7.5, 200, 250],
                "observation": 10.0,
                "glorys": 11.0,
                "reference_salinity": 35.0,
            }
        )
        bundle = Bundle(
            points,
            {"residual": np.full((5, 3), 10.5)},
            {"temperature_kind_compared": "potential"},
        )
        context = Context(bundle, None, {"statistics": {"bootstrap_replicates": 10}}, 7)
        upstream = root / "upstream"
        for name in (
            "prepare_input_v2.py",
            "infer.py",
            "utils/model.py",
            "utils/saved_model/scalers.pkl",
            "utils/saved_model/model.weights.h5",
        ):
            path = upstream / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture")
        source = root / "points.csv"
        points.to_csv(source, index=False)

        def prepare(*args, **kwargs):
            """Record an ERA5 fixture without any external data requests."""
            Path(args[3]).write_text("prepared ERA5 fixture")

        prepare_module = SimpleNamespace(prepare_input=prepare)
        inference_module = SimpleNamespace(
            infer=lambda **kwargs: {
                "depth_grid": np.arange(5, 201, 5),
                "ensemble": np.tile(
                    (np.arange(5, 201, 5) / 100 + 10)[:, None], (1, 100)
                ),
            }
        )
        with (
            patch(
                "val.cavaiola.import_module",
                side_effect=[prepare_module, inference_module],
            ),
            patch("val.cavaiola.sys.path", list(__import__("sys").path)),
        ):
            manifest = predict(source, upstream, root / "predictions", "cds")
        return context, {
            "manifest": manifest["path"],
            "manifest_sha256": manifest["sha256"],
        }

    def _gsw(self):
        """Use a known pressure correction to test conversion of every head."""
        return patch.dict(
            "sys.modules",
            {
                "gsw": SimpleNamespace(
                    p_from_z=lambda z, lat: -z,
                    SA_from_SP=lambda sp, p, lon, lat: sp,
                    pt0_from_t=lambda sa, t, p: t - p / 100,
                )
            },
        )

    def test_diagnostic_uses_common_support_and_converts_every_head(self):
        """Out-of-range depths stay in the main benchmark; converted heads score zero."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            original = context.evaluation.points.copy()
            result = comparators_run(
                context, root / "report", {"cavaiola2026": options}
            )
            scores = pd.read_csv(root / "report/cavaiola_scores.csv").set_index(
                "method"
            )
            self.assertAlmostEqual(scores.loc["cavaiola2026", "rmse"], 0.0)
            self.assertAlmostEqual(scores.loc["cavaiola2026", "empirical_crps"], 0.0)
            self.assertEqual(set(scores.points), {3})
            self.assertNotIn("fair_crps", scores)
            self.assertIn(LABEL, set(scores.interpretation))
            paired = pd.read_csv(root / "report/cavaiola_paired_comparisons.csv")
            self.assertFalse(paired.confirmation_eligible.any())
            self.assertEqual(result["status"], "complete")
            pd.testing.assert_frame_equal(original, context.evaluation.points)
            self.assertEqual(set(context.evaluation.members), {"residual"})

    def test_modified_prediction_fails_before_scoring(self):
        """A frozen manifest cannot silently accept replacement predictions."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            manifest = json.loads(Path(options["manifest"]).read_text())
            Path(manifest["profiles"][0]["prediction"]["path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                run(context, root / "report", options)

    def test_cast_time_and_salinity_are_required(self):
        """Matching an ID cannot hide a different timestamp or missing conversion input."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            context.evaluation.points["observation_time"] = "2016-01-02T12:00:00Z"
            with self.assertRaisesRegex(ValueError, "cast time/coordinates"):
                run(context, root / "report", options)
            context.evaluation.points["observation_time"] = "2016-01-01T12:00:00Z"
            context.evaluation.points.drop(columns="reference_salinity", inplace=True)
            with self.assertRaisesRegex(ValueError, "reference_salinity"):
                run(context, root / "report", options)

    def test_external_salinity_aligns_by_point_id(self):
        """Salinity row order must not change the member conversion."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            salinity = context.evaluation.points[
                ["point_id", "reference_salinity"]
            ].iloc[::-1]
            path = root / "salinity.csv"
            salinity.to_csv(path, index=False)
            options["reference_salinity"] = _file(path)
            context.evaluation.points.drop(columns="reference_salinity", inplace=True)
            result = run(context, root / "report", options)
            self.assertFalse(result["details"]["fair_crps_applied"])

    def test_missing_profile_id_cannot_leave_unscored_rows(self):
        """Missing group keys cannot turn an uninitialized forecast into a score."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            context.evaluation.points.loc[1, "profile_id"] = None
            with self.assertRaisesRegex(ValueError, "populated profile identifiers"):
                run(context, root / "report", options)

    def test_diagnostic_includes_every_available_baseline(self):
        """The external-model scorecard uses all full-input baselines on common support."""
        with tempfile.TemporaryDirectory() as directory, self._gsw():
            root = Path(directory)
            context, options = self._fixture(root)
            for name in ("idw", "lstm", "cnn", "unet3d", "unet2d", "glorys_calibrated"):
                context.evaluation.points[f"baseline_{name}"] = 11.0
                context.evaluation.members[name] = np.full((5, 1), 11.0)
            run(context, root / "report", options)
            methods = set(pd.read_csv(root / "report/cavaiola_scores.csv").method)
            self.assertTrue(
                {
                    "idw",
                    "lstm",
                    "cnn",
                    "unet3d",
                    "unet2d",
                    "glorys_calibrated",
                }.issubset(methods)
            )


if __name__ == "__main__":
    unittest.main()
