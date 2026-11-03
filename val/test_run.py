"""Integration checks for the isolated CLI and shared reference observation operator."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from val.run import _batches, _bundle, _config, _output, demo, preflight


class WorkflowTests(unittest.TestCase):
    """Check scientific guardrails without depending on ocean data or a GPU."""

    def test_frozen_training_config_resolves_evaluation_overrides(self):
        """Use the actual resolver contract and preserve the residual architecture."""
        config = _config(Path("val/config.yaml"))
        config["sampling"].update(sampler="ddim", steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = _bundle(config, Path(temporary))
        self.assertTrue(bundle.model_cfg["model"]["climatology_residual"])
        self.assertEqual(bundle.model_cfg["model"]["condition_channels"], 202)
        self.assertEqual(
            bundle.model_cfg["model"]["generated_channels"]
            + bundle.model_cfg["model"]["condition_channels"],
            252,
        )
        self.assertEqual(
            bundle.training_cfg["training"]["validation_sampling"]["sampler"], "ddim"
        )
        self.assertEqual(
            bundle.training_cfg["training"]["validation_sampling"][
                "ddim_num_timesteps"
            ],
            2,
        )
        self.assertTrue(bundle.data_cfg["dataloader"]["val_shuffle"])

    def test_demo_exercises_both_variables_without_confirmation(self):
        """The entire report pipeline stays clearly labelled as synthetic evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            summary = demo(path)
            self.assertTrue(summary["metadata"]["demo"])
            self.assertFalse(summary["confirmation_eligible"])
            scores = pd.read_csv(path / "report" / "scores.csv")
            self.assertEqual(set(scores.variable), {"temperature", "salinity"})
            self.assertIn("DEMO", (path / "report" / "report.html").read_text())

    def test_outputs_cannot_escape_val(self):
        """Artifacts stay in the directory requested by the user."""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "below"):
                _output(Path(temporary) / "escaped")

    def test_missing_salinity_weights_are_explicit(self):
        """A salinity template cannot accidentally evaluate temperature weights."""
        config = _config(Path("val/config_salinity.yaml"))
        result = preflight(config)
        self.assertFalse(result["ready"])
        self.assertTrue(any("Missing model" in item for item in result["issues"]))

    def test_both_references_use_same_raw_depth_interpolation(self):
        """Depth interpolation and masks are common to GLORYS and climatology."""
        sample = {
            "x": torch.ones(2, 1, 1),
            "x_salinity": torch.ones(2, 1, 1),
            "x_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "x_salinity_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "y": temperature_normalize("norm", torch.tensor([[[4.0]], [[8.0]]])),
            "climatology": temperature_normalize(
                "norm", torch.tensor([[[2.0]], [[6.0]]])
            ),
            "y_salinity": salinity_normalize(
                "norm", torch.tensor([[[30.0]], [[34.0]]])
            ),
            "climatology_salinity": salinity_normalize(
                "norm", torch.tensor([[[31.0]], [[35.0]]])
            ),
            "y_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "y_salinity_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "land_mask": torch.ones(1, 1, 1),
        }
        assignments = pd.DataFrame(
            [
                dict(
                    point_id=variable,
                    patch_index=0,
                    row=0,
                    col=0,
                    depth_index_lower=0,
                    depth_index_upper=1,
                    depth_weight=0.25,
                    variable=variable,
                    observation=1.0,
                )
                for variable in ("temperature", "salinity")
            ]
        )
        assignments["observation_time"] = [
            "2016-01-02T00:00:00+00:00",
            "2016-01-02T01:00:00.12345+00:00",
        ]
        assignments["target_time"] = "2016-01-01T00:00:00Z"
        cohort = SimpleNamespace(
            patch_indices=[0],
            assignments=assignments,
            masked_sample=lambda index: sample,
        )
        config = {"cohort": {"batch_size": 1, "min_input_profiles": 1}}
        batches = list(_batches(None, cohort, config, []))
        points = batches[0][1].set_index("variable")
        np.testing.assert_allclose(points.glorys, [5.0, 31.0], atol=1e-4)
        np.testing.assert_allclose(points.climatology, [3.0, 32.0], atol=1e-4)
        # One colocated T/S column is one input profile, so a threshold of two excludes it.
        config["cohort"]["min_input_profiles"] = 2
        self.assertEqual(list(_batches(None, cohort, config, [])), [])


if __name__ == "__main__":
    unittest.main()
