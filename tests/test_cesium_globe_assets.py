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
    _read_raster_metadata,
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
        self.assertGreater(metadata["default_camera_destination"]["height"], 0.0)

    def test_build_globe_config_preserves_expected_urls_and_bounds(self) -> None:
        template = {
            "selected_date": None,
            "prediction_tiles_url": None,
            "ground_truth_tiles_url": None,
            "argo_points_url": None,
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
            bounds=bounds,
            prediction_credit="Prediction source",
            ground_truth_credit="Ground truth source",
            points_credit="Observed Argo points",
            template=template,
        )

        self.assertEqual(config["selected_date"], 20260105)
        self.assertEqual(config["prediction_tiles_url"], "./prediction_tiles")
        self.assertEqual(config["ground_truth_tiles_url"], "./ground_truth_tiles")
        self.assertEqual(config["argo_points_url"], "./argo_points.geojson")
        self.assertEqual(config["west"], 1.0)
        self.assertEqual(config["south"], 2.0)
        self.assertEqual(config["east"], 3.0)
        self.assertEqual(config["north"], 4.0)
        self.assertEqual(config["credits"]["prediction"], "Prediction source")
        self.assertEqual(config["credits"]["ground_truth"], "Ground truth source")
        self.assertEqual(config["credits"]["points"], "Observed Argo points")

    def test_template_is_valid_json(self) -> None:
        template_path = Path("inference/transforms/globe-config.template.json")
        with template_path.open("r", encoding="utf-8") as f:
            template = json.load(f)

        self.assertIn("prediction_tiles_url", template)
        self.assertIn("default_camera_destination", template)

    def test_sync_with_rclone_warns_when_missing(self) -> None:
        with mock.patch("inference.export_cesium_globe_assets.shutil.which", return_value=None):
            ok, message = _sync_with_rclone(Path("inference/outputs/example/globe"), "r2:bucket/path")

        self.assertFalse(ok)
        self.assertIn("rclone was not found", message)


if __name__ == "__main__":
    unittest.main()
