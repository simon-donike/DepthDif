from __future__ import annotations

import unittest

import numpy as np

from inference.export_global import (
    _argo_point_features_for_patch,
    build_global_mosaic,
    select_export_indices,
)


class TestGlobalInferenceExport(unittest.TestCase):
    def test_select_export_indices_picks_earliest_date_in_iso_week(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260105, "lat0": 0.0, "lat1": 1.0, "lon0": 1.0, "lon1": 2.0},
            {"date": 20260108, "lat0": 1.0, "lat1": 2.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260112, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
        ]

        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        self.assertEqual(selection.selected_date, 20260105)
        self.assertEqual(selection.iso_year, 2026)
        self.assertEqual(selection.iso_week, 2)
        self.assertEqual(selection.indices, [0, 1])

    def test_build_global_mosaic_places_adjacent_tiles_in_expected_grid(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 2.0, "lon0": 0.0, "lon1": 2.0},
            {"date": 20260105, "lat0": 0.0, "lat1": 2.0, "lon0": 2.0, "lon1": 4.0},
        ]
        patches = [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        ]

        mosaic, transform = build_global_mosaic(
            rows=rows,
            top_band_predictions=patches,
            nodata=-9999.0,
        )

        np.testing.assert_allclose(
            mosaic,
            np.asarray([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(transform.a, 1.0, places=6)
        self.assertAlmostEqual(transform.e, -1.0, places=6)
        self.assertAlmostEqual(transform.c, 0.0, places=6)
        self.assertAlmostEqual(transform.f, 2.0, places=6)

    def test_argo_point_features_use_pixel_centers(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-7",
            "export_index": 11,
            "lat0": 0.0,
            "lat1": 2.0,
            "lon0": 10.0,
            "lon1": 12.0,
        }
        observed_mask = np.asarray([[True, False], [False, True]], dtype=bool)

        features = _argo_point_features_for_patch(row=row, observed_mask_2d=observed_mask)

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["geometry"]["coordinates"], [10.5, 1.5])
        self.assertEqual(features[1]["geometry"]["coordinates"], [11.5, 0.5])
        self.assertEqual(features[0]["properties"]["date"], 20260105)
        self.assertEqual(features[0]["properties"]["patch_id"], "patch-7")
        self.assertEqual(features[0]["properties"]["export_index"], 11)


if __name__ == "__main__":
    unittest.main()
