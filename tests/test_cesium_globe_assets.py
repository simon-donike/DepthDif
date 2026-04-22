from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from inference.export_cesium_globe_assets import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_LAT,
    DEFAULT_CAMERA_LON,
    DEFAULT_RCLONE_SYNC_SCOPE,
    _build_parser,
    _build_gdal2tiles_command,
    _read_raster_metadata,
    _resolve_rclone_sync_source,
    _sync_with_rclone,
    build_globe_config,
)


class TestCesiumGlobeAssets(unittest.TestCase):
    def test_read_raster_metadata_uses_exported_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tif_path = Path(tmp_dir) / "prediction.tif"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(10.0, 20.0, 0.5, 1.0),
            ) as ds:
                ds.write(np.ones((1, 2, 3), dtype=np.float32))
                ds.update_tags(source="DepthDif global weekly inference export")

            metadata = _read_raster_metadata(tif_path)

        self.assertAlmostEqual(metadata["west"], 10.0, places=6)
        self.assertAlmostEqual(metadata["east"], 11.5, places=6)
        self.assertAlmostEqual(metadata["north"], 20.0, places=6)
        self.assertAlmostEqual(metadata["south"], 18.0, places=6)
        self.assertEqual(metadata["credit"], "DepthDif global weekly inference export")
        self.assertAlmostEqual(
            metadata["default_camera_destination"]["lon"],
            DEFAULT_CAMERA_LON,
            places=6,
        )
        self.assertAlmostEqual(
            metadata["default_camera_destination"]["lat"],
            DEFAULT_CAMERA_LAT,
            places=6,
        )
        self.assertEqual(
            metadata["default_camera_destination"]["height"], DEFAULT_CAMERA_HEIGHT
        )

    def test_build_globe_config_preserves_expected_urls_and_bounds(self) -> None:
        template = {
            "selected_date": None,
            "prediction_tiles_url": None,
            "ground_truth_tiles_url": None,
            "argo_points_url": None,
            "patch_splits_url": None,
            "west": -180.0,
            "south": -90.0,
            "east": 180.0,
            "north": 90.0,
            "default_camera_destination": {"lon": 0.0, "lat": 0.0, "height": 1.0},
            "credits": {},
        }
        bounds = {
            "west": 1.0,
            "south": 2.0,
            "east": 3.0,
            "north": 4.0,
            "default_camera_destination": {"lon": 2.0, "lat": 3.0, "height": 5000.0},
        }

        config = build_globe_config(
            selected_date=20260105,
            prediction_tiles_url="./prediction_tiles",
            ground_truth_tiles_url="./ground_truth_tiles",
            argo_points_url="./argo_points.geojson",
            patch_splits_url="./patch_splits.geojson",
            bounds=bounds,
            prediction_credit="Prediction source",
            ground_truth_credit="Ground truth source",
            points_credit="Observed Argo points",
            patch_splits_credit="Train/val patch split grid",
            color_scale_min_c=0.0,
            color_scale_max_c=30.0,
            color_palette="temperature_blue_red",
            template=template,
        )

        self.assertEqual(config["selected_date"], 20260105)
        self.assertEqual(config["prediction_tiles_url"], "./prediction_tiles")
        self.assertEqual(config["ground_truth_tiles_url"], "./ground_truth_tiles")
        self.assertEqual(config["argo_points_url"], "./argo_points.geojson")
        self.assertEqual(config["patch_splits_url"], "./patch_splits.geojson")
        self.assertEqual(config["west"], 1.0)
        self.assertEqual(config["south"], 2.0)
        self.assertEqual(config["east"], 3.0)
        self.assertEqual(config["north"], 4.0)
        self.assertEqual(config["color_scale_min_c"], 0.0)
        self.assertEqual(config["color_scale_max_c"], 30.0)
        self.assertEqual(config["color_palette"], "temperature_blue_red")
        self.assertEqual(config["credits"]["prediction"], "Prediction source")
        self.assertEqual(config["credits"]["ground_truth"], "Ground truth source")
        self.assertEqual(config["credits"]["points"], "Observed Argo points")
        self.assertEqual(
            config["credits"]["patch_splits"], "Train/val patch split grid"
        )

    def test_template_is_valid_json(self) -> None:
        template_path = Path("inference/transforms/globe-config.template.json")
        with template_path.open("r", encoding="utf-8") as f:
            template = json.load(f)

        self.assertIn("prediction_tiles_url", template)
        self.assertIn("patch_splits_url", template)
        self.assertIn("default_camera_destination", template)
        self.assertIn("color_scale_min_c", template)

    def test_sync_with_rclone_warns_when_missing(self) -> None:
        with mock.patch(
            "inference.export_cesium_globe_assets.shutil.which", return_value=None
        ):
            ok, message = _sync_with_rclone(
                Path("inference/outputs/example/globe"), "r2:bucket/path"
            )

        self.assertFalse(ok)
        self.assertIn("rclone was not found", message)

    def test_sync_with_rclone_streams_progress(self) -> None:
        with mock.patch(
            "inference.export_cesium_globe_assets.shutil.which",
            return_value="/usr/bin/rclone",
        ), mock.patch(
            "inference.export_cesium_globe_assets.subprocess.run"
        ) as run_mock:
            ok, message = _sync_with_rclone(
                Path("inference/outputs/example/globe"), "r2:bucket/path"
            )

        self.assertTrue(ok)
        self.assertIn("completed successfully", message)
        run_mock.assert_called_once_with(
            [
                "/usr/bin/rclone",
                "sync",
                "--progress",
                "inference/outputs/example/globe",
                "r2:bucket/path",
            ],
            check=True,
            text=True,
        )

    def test_resolve_rclone_sync_source_uses_globe_dir_for_globe_scope(self) -> None:
        run_dir = Path("inference/outputs/inference_production")
        globe_dir = run_dir / "globe"

        source, label = _resolve_rclone_sync_source(
            run_dir=run_dir,
            globe_dir=globe_dir,
            sync_scope="globe",
        )

        self.assertEqual(source, globe_dir)
        self.assertEqual(label, "globe assets")

    def test_resolve_rclone_sync_source_uses_run_dir_for_run_scope(self) -> None:
        run_dir = Path("inference/outputs/inference_production")
        globe_dir = run_dir / "globe"

        source, label = _resolve_rclone_sync_source(
            run_dir=run_dir,
            globe_dir=globe_dir,
            sync_scope="run",
        )

        self.assertEqual(source, run_dir)
        self.assertEqual(label, "run directory")

    def test_build_gdal2tiles_command_uses_nearest_neighbor_and_respects_extra_zoom_levels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tif_path = Path(tmp_dir) / "prediction.tif"
            output_dir = Path(tmp_dir) / "tiles"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=128,
                width=256,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(-180.0, 90.0, 360.0 / 256.0, 180.0 / 128.0),
            ) as ds:
                ds.write(np.ones((1, 128, 256), dtype=np.float32))

            with mock.patch(
                "inference.export_cesium_globe_assets.shutil.which",
                return_value="/usr/bin/gdal2tiles.py",
            ):
                command = _build_gdal2tiles_command(
                    tif_path,
                    output_dir,
                    extra_zoom_levels=1,
                )

        self.assertEqual(command[0], "/usr/bin/gdal2tiles.py")
        self.assertIn("-r", command)
        self.assertIn("near", command)
        self.assertIn("-z", command)
        self.assertIn("0-1", command)

    def test_build_parser_defaults_to_zero_extra_zoom_levels(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--run-dir", "inference/outputs/example"])

        self.assertEqual(args.extra_zoom_levels, 0)
        self.assertEqual(args.rclone_sync_scope, DEFAULT_RCLONE_SYNC_SCOPE)


if __name__ == "__main__":
    unittest.main()
