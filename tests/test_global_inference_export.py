from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from inference.export_global import (
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_FULL_SAMPLE_COUNT,
    FullProfileSample,
    MosaicLayout,
    _cleanup_accumulator,
    _argo_point_features_for_patch,
    _patch_split_feature_for_row,
    _full_profile_feature_for_sample,
    _default_run_stem,
    _normalize_cli_args,
    _profile_graph_figure_title,
    _prepare_run_directory,
    _promote_production_run,
    _repair_small_nodata_gaps_2d,
    create_raster_accumulator,
    build_global_mosaic,
    resolve_depth_export_levels,
    select_export_indices,
    write_global_top_band_geotiff,
)


class TestGlobalInferenceExport(unittest.TestCase):
    def test_default_full_sample_count_is_two_hundred_fifty(self) -> None:
        self.assertEqual(DEFAULT_FULL_SAMPLE_COUNT, 250)

    def test_default_export_gaussian_blur_sigma_is_one(self) -> None:
        self.assertEqual(DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA, 1.0)

    def test_normalize_cli_args_accepts_sigma_colon_zero(self) -> None:
        self.assertEqual(
            _normalize_cli_args(["--device", "cpu", "sigma:0"]),
            ["--device", "cpu", "--sigma", "0"],
        )

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

    def test_repair_small_nodata_gaps_fills_only_tiny_internal_seams(self) -> None:
        raster = np.asarray(
            [
                [7.0, 7.0, -9999.0, 7.0, 7.0],
                [7.0, 7.0, -9999.0, 7.0, 7.0],
            ],
            dtype=np.float32,
        )

        repaired, repaired_mask = _repair_small_nodata_gaps_2d(
            raster,
            nodata=-9999.0,
        )

        self.assertTrue(repaired_mask[:, 2].all())
        np.testing.assert_allclose(repaired, np.full((2, 5), 7.0, dtype=np.float32))

    def test_write_global_top_band_geotiff_repairs_small_internal_seam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scratch_dir = tmp_path / "scratch"
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=32.0,
                top=32.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=32,
                height=32,
                patch_width=32,
                patch_height=32,
                transform=from_origin(0.0, 32.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=scratch_dir,
                stem="prediction_top_band",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 0
                accumulator.sum_array[:, :16] = 7.0
                accumulator.count_array[:, :16] = 1
                accumulator.sum_array[:, 17:] = 7.0
                accumulator.count_array[:, 17:] = 1

                tif_path = tmp_path / "prediction.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_top_band_celsius",
                    tags={"kind": "prediction"},
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)
                    self.assertEqual(ds.nodata, -9999.0)

                np.testing.assert_allclose(band, np.full((32, 32), 7.0, dtype=np.float32))
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_blurs_completed_raster_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_blur",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[8, 8] = 9.0

                tif_path = tmp_path / "prediction_blurred.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={"kind": "prediction"},
                    extra_gaussian_blur_sigma=1.0,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertLess(band[8, 8], 9.0)
                self.assertGreater(band[8, 7], 0.0)
                self.assertEqual(band[0, 0], 0.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_skips_extra_blur_when_sigma_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_no_blur",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[8, 8] = 9.0

                tif_path = tmp_path / "prediction_unblurred.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={"kind": "prediction"},
                    extra_gaussian_blur_sigma=0.0,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertEqual(band[8, 8], 9.0)
                self.assertEqual(band[8, 7], 0.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_resolve_depth_export_levels_uses_nearest_glorys_depths(self) -> None:
        levels = resolve_depth_export_levels(
            np.asarray(
                [0.5, 97.0, 247.0, 505.0, 980.0, 2502.0, 4997.0],
                dtype=np.float64,
            )
        )

        self.assertEqual(
            [level.suffix for level in levels],
            ["surface", "100m", "250m", "500m", "1000m", "2500m", "5000m"],
        )
        self.assertEqual([level.channel_index for level in levels], [0, 1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(levels[1].requested_depth_m, 100.0)
        self.assertAlmostEqual(levels[1].actual_depth_m, 97.0)

    def test_depth_geotiff_metadata_records_requested_and_actual_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_100m",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 12.5
                accumulator.count_array[:] = 1
                tif_path = tmp_path / "global_top_band_20260105_prediction_100m.tif"

                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_100m_celsius",
                    tags={
                        "kind": "prediction",
                        "depth_label": "100m",
                        "requested_depth_m": "100.000",
                        "actual_depth_m": "97.000",
                        "channel_index": "1",
                    },
                )

                with rasterio.open(tif_path) as ds:
                    tags = ds.tags()
                    self.assertEqual(ds.descriptions[0], "predicted_100m_celsius")
                    self.assertEqual(tags["depth_label"], "100m")
                    self.assertEqual(tags["requested_depth_m"], "100.000")
                    self.assertEqual(tags["actual_depth_m"], "97.000")
                    self.assertEqual(tags["channel_index"], "1")
            finally:
                _cleanup_accumulator(accumulator)

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

    def test_full_profile_feature_keeps_depth_stacks_and_graph_path(self) -> None:
        sample = FullProfileSample(
            dataset_index=4,
            row={
                "date": 20260105,
                "patch_id": "patch-12",
                "export_index": 18,
                "lat0": 0.0,
                "lat1": 2.0,
                "lon0": 10.0,
                "lon1": 12.0,
            },
            point_row=1,
            point_col=0,
            patch_height=2,
            patch_width=2,
            lon=10.5,
            lat=0.5,
            x_profile_c=np.asarray([12.0, 13.0, 14.0], dtype=np.float32),
            y_hat_profile_c=np.asarray([11.5, 12.5, 13.5], dtype=np.float32),
            y_target_profile_c=np.asarray([10.5, 11.5, 12.5], dtype=np.float32),
            ostia_sst_c=19.25,
            observed_profile=np.asarray([True, False, True], dtype=bool),
            target_valid_profile=np.asarray([True, True, False], dtype=bool),
        )

        feature = _full_profile_feature_for_sample(
            sample=sample,
            location_id="full_sample_001",
            graph_png_path="graphs/full_sample_001.png",
            depth_axis_m=np.asarray([0.5, 5.0, 25.0], dtype=np.float64),
        )

        self.assertEqual(feature["geometry"]["coordinates"], [10.5, 0.5])
        self.assertEqual(feature["properties"]["date"], 20260105)
        self.assertEqual(feature["properties"]["location_id"], "full_sample_001")
        self.assertEqual(
            feature["properties"]["graph_png_path"], "graphs/full_sample_001.png"
        )
        self.assertEqual(feature["properties"]["depth_m"], [0.5, 5.0, 25.0])
        self.assertEqual(feature["properties"]["ostia_sst_c"], 19.25)
        self.assertEqual(feature["properties"]["argo_profile_c"], [12.0, None, 14.0])
        self.assertEqual(
            feature["properties"]["prediction_profile_c"], [11.5, 12.5, None]
        )
        self.assertEqual(feature["properties"]["glorys_profile_c"], [10.5, 11.5, None])

    def test_profile_graph_title_uses_iso_week_and_geographic_coordinates_only(self) -> None:
        title = _profile_graph_figure_title(
            sample_date=20260630,
            lat=-12.345678,
            lon=45.125,
        )

        self.assertEqual(
            title,
            "Week: ISO week 2026-W27 (Jul)\nLocation: 12.3457 deg S, 45.1250 deg E",
        )
        self.assertNotIn("Pixel", title)

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
            (staging_dir / "global_top_band_20260105_prediction_100m.tif").write_text(
                "prediction depth", encoding="utf-8"
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
                        "depth_exports": [
                            {
                                "suffix": "100m",
                                "prediction_tif_path": "global_top_band_20260105_prediction_100m.tif",
                                "ground_truth_tif_path": None,
                            }
                        ],
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
                (production_dir / "global_top_band_prediction_100m.tif").exists()
            )
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
                summary["depth_exports"][0]["prediction_tif_path"],
                "global_top_band_prediction_100m.tif",
            )
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
