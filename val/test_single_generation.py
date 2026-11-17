"""Keep one-generation validation usable without inventing ensemble uncertainty."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from val.metrics import _crps, score_bundle
from val.paper import run_suite
from val.run import _config
from val.test_all_baselines import _context


class SingleGenerationTests(unittest.TestCase):
    """Exercise point scoring and the complete scientific suite with one draw."""

    def test_every_validation_entrypoint_defaults_to_one_generation(self):
        """Campaign overrides cannot silently restore a large ensemble."""
        for name in ("config", "config_xbt", "config_salinity"):
            self.assertEqual(
                _config(Path(f"val/{name}.yaml"))["sampling"]["members"], 1
            )
        for name in ("validation", "validation_temperature"):
            config = yaml.safe_load(Path(f"val/{name}.yaml").read_text())
            self.assertEqual(config["sampling"]["members"], 1)

    def test_point_scores_remain_finite_and_uncertainty_is_unavailable(self):
        """CRPS reduces to absolute error while fair CRPS and calibration are absent."""
        context = _context()
        points = context.evaluation.points
        members = points.observation.to_numpy()[:, None] + 0.25
        ordinary, fair = _crps(0.0, np.array([0.25]))
        self.assertAlmostEqual(ordinary, 0.25)
        self.assertTrue(np.isnan(fair))
        with (
            tempfile.TemporaryDirectory() as directory,
            np.errstate(divide="raise", invalid="raise"),
        ):
            output = Path(directory)
            result = score_bundle(
                points,
                members,
                output,
                bootstrap_replicates=10,
                min_profiles=2,
                min_clusters=2,
            )
            self.assertFalse(result["uncertainty_available"])
            scores = pd.read_csv(output / "scores.csv")
            forecast = scores[scores.method.eq("single_generation")]
            np.testing.assert_allclose(forecast.mae, 0.25)
            np.testing.assert_allclose(forecast.crps, 0.25)
            self.assertTrue(forecast.fair_crps.isna().all())
            reliability = pd.read_csv(output / "reliability.csv")
            self.assertEqual(set(reliability.metric), {"unavailable_single_generation"})

    def test_entire_suite_handles_one_generation_and_retains_baselines(self):
        """Every scientific stage finishes and no single-sample coverage is reported."""
        context = _context()
        # Casts need distinct physical identities before the real calibration
        # split; renaming colocated copies must not bypass its overlap checks.
        context.evaluation.points["latitude"] = pd.factorize(
            context.evaluation.points.profile_id
        )[0].astype(float)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            context.evaluation.points.to_csv(bundle / "points.csv", index=False)
            np.savez_compressed(
                bundle / "members.npz",
                **{
                    name: values[:, :1]
                    for name, values in context.evaluation.members.items()
                },
            )
            (bundle / "metadata.json").write_text(
                json.dumps({**context.evaluation.metadata, "demo": True})
            )
            config = {
                **context.config,
                "sampling": {"members": 1},
                "paper": {
                    "baselines": {"members": 4, "bootstrap_replicates": 10},
                    "efficiency": {"member_counts": [1]},
                    "dependence": {"permutations": 1},
                },
            }
            run_suite(bundle, root / "suite", config)
            stages = pd.read_csv(root / "suite/stages.csv")
            self.assertNotIn("failed", set(stages.status))
            scores = pd.read_csv(root / "suite/03_baselines/baseline_scores.csv")
            self.assertTrue(
                {"idw", "lstm", "cnn", "unet3d", "unet2d", "residual"}.issubset(
                    set(scores.method)
                )
            )
            self.assertTrue(
                scores.loc[scores.method.eq("residual"), "coverage80"].isna().all()
            )
            efficiency = pd.read_csv(
                root / "suite/10_efficiency/efficiency_member_counts.csv"
            )
            self.assertEqual(set(efficiency.n_members), {1})
            self.assertTrue(efficiency.crps_fair.isna().all())
            self.assertFalse(efficiency.uncertainty_available.any())


if __name__ == "__main__":
    unittest.main()
