"""Tests for the external observation cohort adapter."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch

from val.observations import prepare_external


class _Store:
    def __init__(self, path: Path):
        self.paths_by_date = {20160108: path}
        self.ds = None


class _Dataset:
    tile_size = 2
    depth_axis_m = np.array([0.0, 100.0, 200.0])

    def __init__(self, path: Path):
        self.glorys_store = _Store(path)
        self.argo_store = SimpleNamespace(ds=None)
        self._rows = pd.DataFrame([{"date": 20160108, "grid_y0": 0, "grid_x0": 0}])

    def __getitem__(self, index):
        return {
            "x": torch.ones(3, 2, 2),
            "x_valid_mask": torch.ones(3, 2, 2, dtype=torch.bool),
            "x_valid_mask_1d": torch.ones(1, 2, 2, dtype=torch.bool),
        }


class _Variable:
    def __init__(self, values):
        self.values = np.asarray(values)


class _ClusterDataset:
    variables = {"platform"}

    def __getitem__(self, key):
        return _Variable(["cruise", "other"])


def _point(depth: float, observation: float, **updates):
    """Return one potential-temperature observation fixture."""
    point = {
        "profile_id": "x1",
        "cluster_id": "cruise",
        "source": "xbt",
        "observation_time": "2016-01-08T12:00:00Z",
        "longitude": 0.5,
        "latitude": 1.5,
        "depth_m": depth,
        "variable": "temperature",
        "observation": observation,
        "independence": "unknown",
        "temperature_kind": "potential",
        "qc_flag": 1,
    }
    point.update(updates)
    return point


class TestExternalObservations(unittest.TestCase):
    def _dataset(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "20160108.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint8",
            transform=from_origin(0.0, 2.0, 1.0, 1.0),
        ) as source:
            source.write(np.ones((1, 2, 2), dtype=np.uint8))
        return _Dataset(path)

    def test_preserves_raw_depth_and_interpolates_every_method_to_it(self):
        points = pd.DataFrame(
            [
                _point(50.0, 8.25, region="southern_ocean"),
                _point(100.0, 7.0, region="southern_ocean"),
                _point(250.0, 5.0, region="southern_ocean"),
                _point(75.0, 99.0, qc_flag=2, region="southern_ocean"),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            cohort = prepare_external(
                self._dataset(), points, {"max_vertical_gap_m": 100}
            )
        self.assertTrue(
            any("no cluster metadata" in str(item.message) for item in caught)
        )
        self.assertEqual(len(cohort.assignments), 2)
        raw = cohort.assignments.loc[cohort.assignments.depth_m.eq(50.0)].iloc[0]
        self.assertEqual(float(raw.observation), 8.25)
        self.assertEqual(float(raw.observation_raw), 8.25)
        self.assertEqual(int(raw.depth_index_lower), 0)
        self.assertEqual(int(raw.depth_index_upper), 1)
        self.assertAlmostEqual(float(raw.depth_weight), 0.5)
        self.assertEqual(raw.region, "southern_ocean")
        self.assertEqual(raw.profile_id, "xbt:x1")
        self.assertAlmostEqual(float(raw.time_delta_days), 0.5)
        exact = cohort.assignments.loc[cohort.assignments.depth_m.eq(100.0)].iloc[0]
        self.assertEqual(int(exact.depth_index_lower), 1)
        self.assertEqual(int(exact.depth_index_upper), 1)
        self.assertEqual(float(exact.depth_weight), 0.0)
        masked = cohort.masked_sample(0)
        self.assertFalse(masked["x_valid_mask"][:, 0, 0].any())
        self.assertTrue(masked["x_valid_mask"][:, 1, 1].all())

    def test_requires_explicit_temperature_kind(self):
        point = _point(50.0, 8.0)
        point.pop("temperature_kind")
        with self.assertRaisesRegex(ValueError, "temperature_kind must be explicitly"):
            prepare_external(self._dataset(), pd.DataFrame([point]))

    def test_rejects_in_situ_temperature_without_declared_operator(self):
        points = pd.DataFrame([_point(50.0, 8.0, temperature_kind="in_situ")])
        with self.assertRaisesRegex(ValueError, "in_situ_to_potential_teos10"):
            prepare_external(self._dataset(), points)

    def test_confirmation_requires_cluster_metadata(self):
        points = pd.DataFrame([_point(50.0, 8.0)])
        with self.assertRaisesRegex(ValueError, "requires cluster metadata"):
            prepare_external(self._dataset(), points, {"split_role": "confirmation"})

    def test_cluster_cells_are_buffered_once_and_verification_is_recorded(self):
        dataset = self._dataset()
        dataset.argo_store = SimpleNamespace(
            ds=_ClusterDataset(),
            target_date=np.array([20160108, 20160108]),
            grid_row=np.array([1, 1]),
            grid_col=np.array([1, 0]),
        )
        cohort = prepare_external(
            dataset,
            pd.DataFrame([_point(50.0, 8.0)]),
            {"split_role": "confirmation", "buffer_cells": 1},
        )
        self.assertEqual(cohort.metadata["cluster_match_count"], 1)
        self.assertTrue(cohort.metadata["conditioning_exclusion_verified"])
        # One-cell buffers around (0, 0) and (1, 1) have a 14-cell union.
        self.assertEqual(len(cohort.holdouts), 14)


if __name__ == "__main__":
    unittest.main()
