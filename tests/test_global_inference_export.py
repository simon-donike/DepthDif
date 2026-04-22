from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from inference.export_global import (
    _argo_point_features_for_patch,
    _patch_split_feature_for_row,
    _default_run_stem,
    _prepare_run_directory,
    _promote_production_run,
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

        features = _argo_point_features_for_patch(
            row=row,
            observed_mask_2d=observed_mask,
        )

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["geometry"]["coordinates"], [10.5, 1.5])
        self.assertEqual(features[1]["geometry"]["coordinates"], [11.5, 0.5])
        self.assertEqual(features[0]["properties"]["date"], 20260105)
        self.assertEqual(features[0]["properties"]["patch_id"], "patch-7")
        self.assertEqual(features[0]["properties"]["export_index"], 11)
        self.assertNotIn("observed_depth_index", features[0]["properties"])
        self.assertNotIn("observed_temp_c", features[0]["properties"])

    def test_patch_split_feature_builds_closed_polygon_and_keeps_split(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-9",
            "export_index": 13,
            "lat0": -2.0,
            "lat1": 1.0,
            "lon0": 10.0,
            "lon1": 14.0,
            "split": "train",
        }

        feature = _patch_split_feature_for_row(row)

        self.assertIsNotNone(feature)
        assert feature is not None
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(
            feature["geometry"]["coordinates"][0],
            [[10.0, 1.0], [14.0, 1.0], [14.0, -2.0], [10.0, -2.0], [10.0, 1.0]],
        )
        self.assertEqual(feature["properties"]["split"], "train")
        self.assertEqual(feature["properties"]["patch_id"], "patch-9")

    def test_patch_split_feature_skips_rows_without_train_val_label(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-10",
            "export_index": 14,
            "lat0": -2.0,
            "lat1": 1.0,
            "lon0": 10.0,
            "lon1": 14.0,
            "split": "invalid",
        }

        self.assertIsNone(_patch_split_feature_for_row(row))

    def test_default_run_stem_keeps_selected_date_in_artifact_names(self) -> None:
        self.assertEqual(_default_run_stem(20260105), "global_top_band_20260105")

    def test_prepare_run_directory_defaults_to_date_stamped_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"

            run_dir, production_dir = _prepare_run_directory(
                output_root,
                run_stem="global_top_band_20260105",
                output_name=None,
            )

            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir, output_root / "global_top_band_20260105")
            self.assertIsNone(production_dir)

    def test_promote_production_run_strips_date_from_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = Path(tmp_dir) / "staging"
            production_dir = Path(tmp_dir) / "production"
            staging_dir.mkdir(parents=True, exist_ok=True)

            (staging_dir / "global_top_band_20260105_prediction.tif").write_text(
                "prediction", encoding="utf-8"
            )
            (staging_dir / "global_top_band_20260105_argo_points.geojson").write_text(
                "points", encoding="utf-8"
            )
            (staging_dir / "global_top_band_20260105_patch_splits.geojson").write_text(
                "splits", encoding="utf-8"
            )
            with (staging_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "selected_date": 20260105,
                        "run_dir": str(staging_dir),
                        "prediction_tif_path": "global_top_band_20260105_prediction.tif",
                        "ground_truth_tif_path": None,
                        "argo_points_geojson_path": "global_top_band_20260105_argo_points.geojson",
                        "patch_splits_geojson_path": "global_top_band_20260105_patch_splits.geojson",
                    },
                    f,
                    sort_keys=False,
                )

            _promote_production_run(staging_dir, production_dir)

            self.assertFalse(staging_dir.exists())
            self.assertTrue((production_dir / "global_top_band_prediction.tif").exists())
            self.assertTrue(
                (production_dir / "global_top_band_argo_points.geojson").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_patch_splits.geojson").exists()
            )

            with (production_dir / "run_summary.yaml").open("r", encoding="utf-8") as f:
                summary = yaml.safe_load(f)

            self.assertEqual(summary["run_dir"], str(production_dir))
            self.assertEqual(summary["prediction_tif_path"], "global_top_band_prediction.tif")
            self.assertEqual(
                summary["argo_points_geojson_path"],
                "global_top_band_argo_points.geojson",
            )
            self.assertEqual(
                summary["patch_splits_geojson_path"],
                "global_top_band_patch_splits.geojson",
            )


if __name__ == "__main__":
    unittest.main()
