from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depth_recon.inference.export_cesium_globe_assets import (
    ARGO_SAMPLE_LOCATION_PROPERTY_KEYS,
    ARGO_POINT_PROPERTY_KEYS,
    DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_LAT,
    DEFAULT_CAMERA_LON,
    DEFAULT_ERROR_ANALYSIS_JSON_NAME,
    DEFAULT_RASTER_EDGE_EROSION_PIXELS,
    DEFAULT_RASTER_EDGE_FEATHER_PIXELS,
    DEFAULT_RCLONE_SYNC_SCOPE,
    DEFAULT_SALINITY_COLOR_RAMP_PATH,
    DEFAULT_TEMPLATE_PATH,
    DEFAULT_WEBP_QUALITY,
    FULL_SAMPLE_PROPERTY_KEYS,
    _absolute_error_color_scale,
    _apply_alpha_mask_to_colorized_raster,
    _build_parser,
    _build_gdal2tiles_command,
    _copy_precomputed_analysis_grid_geojson,
    _copy_precomputed_error_analysis_json,
    _estimate_native_zoom_level,
    _export_base_map_tiles,
    _prefix_geojson_graph_paths,
    _prefix_variable_config_asset_urls,
    _read_raster_metadata,
    _remove_gdal_auxiliary_files,
    _resolve_depth_export_artifacts,
    _resolve_rclone_sync_source,
    _rewrite_argo_sample_locations_geojson,
    _run_variable_metadata,
    _rewrite_geojson,
    _sync_with_rclone,
    _validate_raster_transparency_contract,
    _write_absolute_error_color_ramp,
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
            "target_date": None,
            "iso_year": None,
            "iso_week": None,
            "prediction_tiles_url": None,
            "ground_truth_tiles_url": None,
            "absolute_error_tiles_url": None,
            "uncertainty_tiles_url": None,
            "depth_levels": [],
            "argo_sample_locations_url": None,
            "argo_points_url": None,
            "patch_splits_url": None,
            "full_sample_points_url": None,
            "base_map_tiles_url": None,
            "base_map_credit": None,
            "west": -180.0,
            "south": -90.0,
            "east": 180.0,
            "north": 90.0,
            "default_camera_destination": {"lon": 0.0, "lat": 0.0, "height": 1.0},
            "raster_transparency": {},
            "credits": {"base_map": "Natural Earth II"},
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
            target_date=20260105,
            iso_year=2026,
            iso_week=2,
            prediction_tiles_url="./prediction_tiles",
            ground_truth_tiles_url="./ground_truth_tiles",
            absolute_error_tiles_url="./absolute_error_tiles",
            uncertainty_tiles_url="./uncertainty_tiles",
            uncertainty_credit="Uncertainty source",
            uncertainty_color_scale_min=0.0,
            uncertainty_color_scale_max=2.5,
            uncertainty_legend_max=3,
            depth_levels=[
                {
                    "label": "Surface",
                    "requested_depth_m": 0.0,
                    "actual_depth_m": 0.5,
                    "channel_index": 0,
                    "prediction_tiles_url": "./prediction_tiles_surface",
                    "ground_truth_tiles_url": "./ground_truth_tiles_surface",
                    "absolute_error_tiles_url": "./absolute_error_tiles_surface",
                    "absolute_error_legend_max_c": 5,
                }
            ],
            argo_sample_locations_url="./argo_sample_locations.geojson",
            argo_points_url="./argo_points.geojson",
            patch_splits_url="./patch_splits.geojson",
            full_sample_points_url="./full_sample_locations.geojson",
            bounds=bounds,
            prediction_credit="Prediction source",
            ground_truth_credit="Ground truth source",
            absolute_error_credit="Absolute error source",
            points_credit="Observed Argo points",
            patch_splits_credit="Inference patch grid",
            full_sample_points_credit="Random full-depth profile locations",
            color_scale_min_c=0.0,
            color_scale_max_c=30.0,
            color_palette="temperature_blue_red",
            raster_transparency={
                "nodata_value": -9999.0,
                "land_mask_applied_value": -9999.0,
                "land_mask_mode": "nodata",
                "land_mask_alpha": 0,
                "valid_alpha": 255,
            },
            template=template,
            error_analysis_data_url="./error-analysis.json",
            analysis_grid_geojson_url="./analysis-grid.geojson",
        )

        self.assertEqual(config["selected_date"], 20260105)
        self.assertEqual(config["target_date"], 20260105)
        self.assertEqual(config["iso_year"], 2026)
        self.assertEqual(config["iso_week"], 2)
        self.assertEqual(config["prediction_tiles_url"], "./prediction_tiles")
        self.assertEqual(config["ground_truth_tiles_url"], "./ground_truth_tiles")
        self.assertEqual(config["absolute_error_tiles_url"], "./absolute_error_tiles")
        self.assertEqual(config["uncertainty_tiles_url"], "./uncertainty_tiles")
        self.assertEqual(config["uncertainty_color_scale_min"], 0.0)
        self.assertEqual(config["uncertainty_color_scale_max"], 2.5)
        self.assertEqual(config["uncertainty_legend_max"], 3)
        self.assertEqual(config["depth_levels"][0]["label"], "Surface")
        self.assertEqual(
            config["depth_levels"][0]["prediction_tiles_url"],
            "./prediction_tiles_surface",
        )
        self.assertEqual(
            config["depth_levels"][0]["absolute_error_tiles_url"],
            "./absolute_error_tiles_surface",
        )
        self.assertEqual(
            config["argo_sample_locations_url"], "./argo_sample_locations.geojson"
        )
        self.assertEqual(config["argo_points_url"], "./argo_points.geojson")
        self.assertEqual(config["patch_splits_url"], "./patch_splits.geojson")
        self.assertEqual(
            config["full_sample_points_url"], "./full_sample_locations.geojson"
        )
        self.assertEqual(config["west"], 1.0)
        self.assertEqual(config["south"], 2.0)
        self.assertEqual(config["east"], 3.0)
        self.assertEqual(config["north"], 4.0)
        self.assertEqual(config["color_scale_min_c"], 0.0)
        self.assertEqual(config["color_scale_max_c"], 30.0)
        self.assertEqual(config["color_palette"], "temperature_blue_red")
        self.assertEqual(config["raster_transparency"]["land_mask_alpha"], 0)
        self.assertEqual(config["raster_transparency"]["valid_alpha"], 255)
        self.assertEqual(config["credits"]["prediction"], "Prediction source")
        self.assertEqual(config["credits"]["ground_truth"], "Ground truth source")
        self.assertEqual(config["credits"]["absolute_error"], "Absolute error source")
        self.assertEqual(config["credits"]["uncertainty"], "Uncertainty source")
        self.assertEqual(config["credits"]["points"], "Observed Argo points")
        self.assertEqual(config["credits"]["patch_splits"], "Inference patch grid")
        self.assertEqual(
            config["credits"]["full_sample_points"],
            "Random full-depth profile locations",
        )
        self.assertEqual(config["variable"], "temperature")
        self.assertEqual(config["value_units"], "degree_Celsius")
        self.assertEqual(config["color_scale_min"], 0.0)
        self.assertEqual(config["color_scale_max"], 30.0)
        self.assertNotIn("base_map_tiles_url", config)
        self.assertNotIn("base_map_credit", config)
        self.assertNotIn("base_map", config["credits"])
        self.assertNotIn("error_analysis_url", config)
        self.assertEqual(config["error_analysis_data_url"], "./error-analysis.json")
        self.assertEqual(config["analysis_grid_geojson_url"], "./analysis-grid.geojson")

    def test_build_globe_config_keeps_multivariable_config_and_legacy_fields(
        self,
    ) -> None:
        template = {"credits": {}, "raster_transparency": {}}
        bounds = {
            "west": -10.0,
            "south": -20.0,
            "east": 10.0,
            "north": 20.0,
            "default_camera_destination": {"lon": 0.0, "lat": 0.0, "height": 1.0},
        }
        variables = {
            "temperature": {
                "variable": "temperature",
                "prediction_tiles_url": "temperature/prediction_tiles_surface",
                "depth_levels": [
                    {"prediction_tiles_url": "temperature/prediction_tiles_surface"}
                ],
            },
            "salinity": {
                "variable": "salinity",
                "prediction_tiles_url": "salinity/prediction_tiles_surface",
                "value_unit_label": "PSU",
                "depth_levels": [
                    {"prediction_tiles_url": "salinity/prediction_tiles_surface"}
                ],
            },
        }

        config = build_globe_config(
            selected_date=20260105,
            target_date=20260105,
            iso_year=2026,
            iso_week=2,
            prediction_tiles_url="temperature/prediction_tiles_surface",
            ground_truth_tiles_url=None,
            absolute_error_tiles_url=None,
            depth_levels=variables["temperature"]["depth_levels"],
            argo_sample_locations_url="temperature/argo_sample_locations.geojson",
            argo_points_url=None,
            patch_splits_url=None,
            full_sample_points_url=None,
            bounds=bounds,
            prediction_credit="Prediction source",
            ground_truth_credit=None,
            absolute_error_credit=None,
            points_credit=None,
            patch_splits_credit=None,
            full_sample_points_credit=None,
            color_scale_min_c=0.0,
            color_scale_max_c=30.0,
            color_palette="temperature_blue_red",
            raster_transparency={},
            template=template,
            variables=variables,
            default_variable="temperature",
        )

        self.assertEqual(config["default_variable"], "temperature")
        self.assertEqual(
            config["variables"]["salinity"]["prediction_tiles_url"],
            "salinity/prediction_tiles_surface",
        )
        self.assertEqual(config["color_scale_min"], 0.0)
        self.assertEqual(config["color_scale_min_c"], 0.0)
        self.assertEqual(
            config["prediction_tiles_url"], "temperature/prediction_tiles_surface"
        )

    def test_prefix_variable_config_asset_urls_rewrites_relative_tile_paths(
        self,
    ) -> None:
        single_config = {
            "prediction_tiles_url": "./prediction_tiles_surface",
            "ground_truth_tiles_url": "./ground_truth_tiles_surface",
            "absolute_error_tiles_url": None,
            "uncertainty_tiles_url": "./uncertainty_tiles",
            "argo_sample_locations_url": "./argo_sample_locations.geojson",
            "analysis_grid_geojson_url": "./analysis-grid.geojson",
            "depth_levels": [
                {
                    "prediction_tiles_url": "./prediction_tiles_surface",
                    "ground_truth_tiles_url": "./ground_truth_tiles_surface",
                    "absolute_error_tiles_url": "./absolute_error_tiles_surface",
                }
            ],
        }

        prefixed = _prefix_variable_config_asset_urls(
            single_config,
            variable="salinity",
            public_base_url=None,
        )
        hosted = _prefix_variable_config_asset_urls(
            single_config,
            variable="salinity",
            public_base_url="https://example.test/globe",
        )

        self.assertEqual(
            prefixed["prediction_tiles_url"], "salinity/prediction_tiles_surface"
        )
        self.assertEqual(
            prefixed["argo_sample_locations_url"],
            "salinity/argo_sample_locations.geojson",
        )
        self.assertEqual(
            prefixed["uncertainty_tiles_url"], "salinity/uncertainty_tiles"
        )
        self.assertEqual(
            prefixed["analysis_grid_geojson_url"], "salinity/analysis-grid.geojson"
        )
        self.assertEqual(
            prefixed["depth_levels"][0]["absolute_error_tiles_url"],
            "salinity/absolute_error_tiles_surface",
        )
        self.assertEqual(
            hosted["prediction_tiles_url"],
            "https://example.test/globe/salinity/prediction_tiles_surface",
        )
        self.assertEqual(
            hosted["uncertainty_tiles_url"],
            "https://example.test/globe/salinity/uncertainty_tiles",
        )
        self.assertEqual(
            hosted["analysis_grid_geojson_url"],
            "https://example.test/globe/salinity/analysis-grid.geojson",
        )

    def test_prefix_geojson_graph_paths_rewrites_combined_profile_graphs(
        self,
    ) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"graph_png_path": "graphs/full_sample_001.png"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                    "properties": {"graph_png_path": "graphs/full_sample_002.webp"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            geojson_path = Path(tmp_dir) / "samples.geojson"
            geojson_path.write_text(json.dumps(payload), encoding="utf-8")

            _prefix_geojson_graph_paths(
                geojson_path,
                variable="salinity",
                public_base_url=None,
            )

            rewritten = json.loads(geojson_path.read_text(encoding="utf-8"))

        self.assertEqual(
            rewritten["features"][0]["properties"]["graph_png_path"],
            "salinity/graphs/full_sample_001.png",
        )
        self.assertEqual(
            rewritten["features"][1]["properties"]["graph_png_path"],
            "salinity/graphs/full_sample_002.webp",
        )

    def test_run_variable_metadata_uses_salinity_defaults(self) -> None:
        metadata = _run_variable_metadata({"variable": "salinity"})

        self.assertEqual(metadata["label"], "Salinity")
        self.assertEqual(metadata["value_units"], "PSU")
        self.assertEqual(metadata["value_unit_label"], "PSU")
        self.assertEqual(metadata["color_scale_min"], 30.0)
        self.assertEqual(metadata["color_scale_max"], 40.0)
        self.assertEqual(metadata["color_ramp_path"], DEFAULT_SALINITY_COLOR_RAMP_PATH)

    def test_apply_alpha_mask_to_colorized_raster_hides_nodata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.tif"
            colorized_path = tmp_path / "colorized.tif"
            transform = from_origin(0.0, 2.0, 1.0, 1.0)

            with rasterio.open(
                input_path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=1,
                dtype="float32",
                nodata=-9999.0,
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(
                    np.array(
                        [[[10.0, -9999.0, 0.0], [8.0, 0.0, 12.0]]],
                        dtype=np.float32,
                    )
                )

            with rasterio.open(
                colorized_path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(np.full((4, 2, 3), 200, dtype=np.uint8))

            _apply_alpha_mask_to_colorized_raster(
                input_path,
                colorized_path,
                raster_edge_erosion_pixels=0,
                raster_edge_feather_pixels=0,
            )

            with rasterio.open(colorized_path) as ds:
                rgba = ds.read()

        self.assertEqual(int(rgba[3, 0, 1]), 0)
        self.assertEqual(int(rgba[3, 1, 1]), 255)
        self.assertEqual(int(rgba[3, 0, 2]), 255)

    def test_apply_alpha_mask_to_colorized_raster_erodes_and_feathers_edges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.tif"
            colorized_path = tmp_path / "colorized.tif"
            transform = from_origin(0.0, 13.0, 1.0, 1.0)
            data = np.full((1, 13, 13), -9999.0, dtype=np.float32)
            data[:, 1:12, 1:12] = 10.0

            with rasterio.open(
                input_path,
                "w",
                driver="GTiff",
                height=13,
                width=13,
                count=1,
                dtype="float32",
                nodata=-9999.0,
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(data)

            with rasterio.open(
                colorized_path,
                "w",
                driver="GTiff",
                height=13,
                width=13,
                count=4,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(np.full((4, 13, 13), 200, dtype=np.uint8))

            _apply_alpha_mask_to_colorized_raster(
                input_path,
                colorized_path,
                raster_edge_erosion_pixels=2,
                raster_edge_feather_pixels=4,
            )

            with rasterio.open(colorized_path) as ds:
                rgba = ds.read()

        self.assertEqual(int(rgba[3, 0, 6]), 0)
        self.assertEqual(int(rgba[3, 1, 6]), 0)
        self.assertEqual(int(rgba[3, 2, 6]), 0)
        self.assertEqual(int(rgba[3, 3, 6]), 64)
        self.assertEqual(int(rgba[3, 4, 6]), 128)
        self.assertEqual(int(rgba[3, 6, 6]), 255)
        self.assertEqual(int(rgba[0, 0, 6]), 0)
        self.assertEqual(int(rgba[0, 2, 6]), 0)
        self.assertEqual(int(rgba[0, 3, 6]), 200)
        self.assertEqual(int(rgba[0, 6, 6]), 200)

    def test_validate_raster_transparency_contract_rejects_zeroed_land(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.tif"
            transform = from_origin(0.0, 1.0, 1.0, 1.0)

            with rasterio.open(
                input_path,
                "w",
                driver="GTiff",
                height=1,
                width=3,
                count=1,
                dtype="float32",
                nodata=-9999.0,
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(np.array([[[0.0, 0.0, 2.0]]], dtype=np.float32))
                ds.update_tags(land_zeroed="true")

            with self.assertRaisesRegex(RuntimeError, "old land-mask convention"):
                _validate_raster_transparency_contract(input_path)

    def test_validate_raster_transparency_contract_accepts_land_nodata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tif_path = Path(tmp_dir) / "input.tif"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=1,
                width=1,
                count=1,
                dtype="float32",
                nodata=-9999.0,
                crs="EPSG:4326",
                transform=from_origin(0.0, 1.0, 1.0, 1.0),
            ) as ds:
                ds.write(np.array([[[-9999.0]]], dtype=np.float32))
                ds.update_tags(land_zeroed="false", land_masked_to_nodata="true")

            _validate_raster_transparency_contract(tif_path)

    def test_template_is_valid_json(self) -> None:
        template_path = DEFAULT_TEMPLATE_PATH
        with template_path.open("r", encoding="utf-8") as f:
            template = json.load(f)

        self.assertIn("prediction_tiles_url", template)
        self.assertIn("depth_levels", template)
        self.assertIn("absolute_error_tiles_url", template)
        self.assertIn("uncertainty_tiles_url", template)
        self.assertIn("uncertainty_color_scale_max", template)
        self.assertIn("argo_sample_locations_url", template)
        self.assertIn("patch_splits_url", template)
        self.assertIn("full_sample_points_url", template)
        self.assertIn("base_map_tiles_url", template)
        self.assertIn("base_map_credit", template)
        self.assertIn("default_camera_destination", template)
        self.assertIn("color_scale_min", template)
        self.assertIn("color_scale_min_c", template)
        self.assertIn("default_variable", template)
        self.assertIn("variables", template)
        self.assertIn("raster_transparency", template)
        self.assertEqual(
            template["raster_transparency"]["land_mask_applied_value"],
            None,
        )
        self.assertEqual(template["raster_transparency"]["land_mask_mode"], "none")
        self.assertEqual(
            template["raster_transparency"]["edge_erosion_pixels"],
            DEFAULT_RASTER_EDGE_EROSION_PIXELS,
        )
        self.assertEqual(
            template["raster_transparency"]["edge_feather_pixels"],
            DEFAULT_RASTER_EDGE_FEATHER_PIXELS,
        )

    def test_standalone_globe_page_uses_full_window_root_shell(self) -> None:
        html = Path("docs/globe/index.html").read_text(encoding="utf-8")
        css = Path("docs/stylesheets/globe.css").read_text(encoding="utf-8")
        loader = Path("docs/javascripts/load-cesium-globe.js").read_text(
            encoding="utf-8"
        )
        globe_script = Path("docs/javascripts/cesium-globe.js").read_text(
            encoding="utf-8"
        )
        default_config = json.loads(
            Path("docs/globe/globe-config.json").read_text(encoding="utf-8")
        )

        self.assertIn('class="standalone-globe-root"', html)
        self.assertIn('id="globe-page-eyebrow"', html)
        self.assertIn('id="globe-depth-level-ticks"', html)
        self.assertIn("Rasters", html)
        self.assertIn("Prediction", html)
        self.assertIn("GLORYS", html)
        self.assertIn("Error", html)
        self.assertIn("Uncertainty", html)
        self.assertIn('id="globe-toggle-uncertainty"', html)
        self.assertIn("Vector", html)
        self.assertIn("ARGO Locations", html)
        self.assertIn("Inference Patches", html)
        self.assertIn("Depth Levels", html)
        self.assertIn("Modality", html)
        self.assertIn('id="globe-variable-control"', html)
        self.assertIn('name="globe-variable"', html)
        self.assertIn('name="globe-raster-layer"', html)
        self.assertIn("globe-segmented-toggle--raster", html)
        self.assertIn('name="globe-points-layer"', html)
        self.assertIn('name="globe-patch-splits-layer"', html)
        self.assertIn("Salinity", html)
        self.assertIn("Ocean Variable Reconstruction", html)
        self.assertIn('href="../visualizations/">Back to Analysis</a>', html)
        self.assertNotIn('href="../analysis/">Analysis Dashboard</a>', html)
        self.assertNotIn('href="../temporal-globe/">Temporal Globe</a>', html)
        self.assertIn('id="globe-error-legend"', html)
        self.assertIn('id="globe-error-legend-title"', html)
        self.assertIn('loading="lazy"', html)
        self.assertIn(".standalone-globe-root,", css)
        self.assertIn("box-sizing: border-box;", css)
        self.assertIn(".globe-toolbar__sections", css)
        self.assertIn("width: min(18rem, calc(100% - 2rem));", css)
        self.assertIn(".globe-segmented-toggle--raster", css)
        self.assertIn("globe-legend__bar--error", css)
        self.assertIn("globe-legend__bar--salinity", css)
        self.assertIn('document.querySelectorAll("script[src]")', loader)
        self.assertIn('new URL("/javascripts/", document.baseURI)', loader)
        self.assertIn("function resolveConfigUrl()", globe_script)
        self.assertIn("base_map_tiles_url", globe_script)
        self.assertIn("function addBundledNaturalEarthFallback()", globe_script)
        self.assertIn('const PATCH_FILL_COLOR = "#f97316";', globe_script)
        self.assertIn(
            'const titleText = "Ocean Variable Reconstruction";', globe_script
        )
        self.assertIn("function addAbsoluteErrorLayer", globe_script)
        self.assertIn("function addUncertaintyLayer", globe_script)
        self.assertIn("function hasUncertaintyLayer", globe_script)
        self.assertIn("function activeVariableConfig", globe_script)
        self.assertIn("function reloadVariableLayers", globe_script)
        self.assertIn("selectedVariable", globe_script)
        self.assertIn("function hasActiveRasterSelection", globe_script)
        self.assertIn('return "off";', globe_script)
        self.assertIn('+ "k m"', globe_script)
        self.assertIn(
            "new URL(configParam, window.location.href).toString()", globe_script
        )
        self.assertIn("Full Sample #", globe_script)
        self.assertIn("depth_levels", default_config)
        self.assertIn("default_variable", default_config)
        self.assertIn("variables", default_config)
        self.assertIn("uncertainty_tiles_url", default_config)

    def test_sync_with_rclone_warns_when_missing(self) -> None:
        with mock.patch(
            "depth_recon.inference.export_cesium_globe_assets.shutil.which",
            return_value=None,
        ):
            ok, message = _sync_with_rclone(
                Path("inference/outputs/example/globe"), "r2:bucket/path"
            )

        self.assertFalse(ok)
        self.assertIn("rclone was not found", message)

    def test_sync_with_rclone_streams_progress(self) -> None:
        with (
            mock.patch(
                "depth_recon.inference.export_cesium_globe_assets.shutil.which",
                return_value="/usr/bin/rclone",
            ),
            mock.patch(
                "depth_recon.inference.export_cesium_globe_assets.subprocess.run"
            ) as run_mock,
        ):
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
                "depth_recon.inference.export_cesium_globe_assets.shutil.which",
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
        self.assertIn("--tiledriver=WEBP", command)
        self.assertIn(f"--webp-quality={DEFAULT_WEBP_QUALITY}", command)

    def test_build_gdal2tiles_command_supports_bilinear_basemap_webp_tiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tif_path = Path(tmp_dir) / "base_map.tif"
            output_dir = Path(tmp_dir) / "tiles"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=8100,
                width=16200,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(-180.0, 90.0, 360.0 / 16200.0, 180.0 / 8100.0),
            ):
                pass

            with mock.patch(
                "depth_recon.inference.export_cesium_globe_assets.shutil.which",
                return_value="/usr/bin/gdal2tiles.py",
            ):
                command = _build_gdal2tiles_command(
                    tif_path,
                    output_dir,
                    extra_zoom_levels=0,
                    resampling="bilinear",
                )

        self.assertIn("bilinear", command)
        self.assertIn("0-6", command)
        self.assertIn("--tiledriver=WEBP", command)
        self.assertIn(f"--webp-quality={DEFAULT_WEBP_QUALITY}", command)

    def test_remove_gdal_auxiliary_files_deletes_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sidecar = root / "0" / "0" / "0.webp.aux.xml"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text("aux", encoding="utf-8")
            tile = root / "0" / "0" / "0.webp"
            tile.write_text("tile", encoding="utf-8")

            removed = _remove_gdal_auxiliary_files(root)

            self.assertEqual(removed, 1)
            self.assertFalse(sidecar.exists())
            self.assertTrue(tile.exists())

    def test_export_base_map_tiles_skips_missing_source_tif(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            globe_dir = root / "globe"

            tiles_dir, tiles_url, credit = _export_base_map_tiles(
                globe_dir,
                public_base_url="https://example.test/globe",
                base_map_raster_path=root / "missing.tif",
            )

        self.assertIsNone(tiles_dir)
        self.assertIsNone(tiles_url)
        self.assertIsNone(credit)

    def test_export_base_map_tiles_uses_configured_hosted_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tif_path = root / "base_map.tif"
            tif_path.write_bytes(b"placeholder")
            globe_dir = root / "globe"

            with mock.patch(
                "depth_recon.inference.export_cesium_globe_assets._run_gdal2tiles"
            ) as run_gdal2tiles:
                tiles_dir, tiles_url, credit = _export_base_map_tiles(
                    globe_dir,
                    public_base_url="https://example.test/globe",
                    base_map_raster_path=tif_path,
                )

        self.assertEqual(
            tiles_dir, globe_dir / "basemaps" / "natural_earth_ii_webp_q95"
        )
        self.assertEqual(
            tiles_url,
            "https://example.test/globe/basemaps/natural_earth_ii_webp_q95",
        )
        self.assertEqual(credit, "Natural Earth II")
        run_gdal2tiles.assert_called_once_with(
            tif_path,
            globe_dir / "basemaps" / "natural_earth_ii_webp_q95",
            extra_zoom_levels=0,
            resampling="bilinear",
        )

    def test_build_parser_defaults_to_zero_extra_zoom_levels(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--run-dir", "inference/outputs/example"])

        self.assertEqual(args.extra_zoom_levels, 0)
        self.assertEqual(
            args.raster_edge_erosion_pixels,
            DEFAULT_RASTER_EDGE_EROSION_PIXELS,
        )
        self.assertEqual(
            args.raster_edge_feather_pixels,
            DEFAULT_RASTER_EDGE_FEATHER_PIXELS,
        )
        self.assertEqual(args.rclone_sync_scope, DEFAULT_RCLONE_SYNC_SCOPE)
        self.assertTrue(args.include_error_analysis)

        args = parser.parse_args(
            ["--run-dir", "inference/outputs/example", "--no-error-analysis"]
        )
        self.assertFalse(args.include_error_analysis)

    def test_copy_precomputed_error_analysis_json_prefers_run_summary_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            globe_dir = run_dir / "globe"
            run_dir.mkdir(parents=True)
            source_path = run_dir / DEFAULT_ERROR_ANALYSIS_JSON_NAME
            source_path.write_text(
                json.dumps({"depth_levels": ["precomputed"]}) + "\n",
                encoding="utf-8",
            )

            copied_path = _copy_precomputed_error_analysis_json(
                run_dir=run_dir,
                globe_dir=globe_dir,
                run_summary={"error_analysis_json_path": source_path.name},
            )

            self.assertEqual(copied_path, globe_dir / DEFAULT_ERROR_ANALYSIS_JSON_NAME)
            self.assertEqual(
                json.loads(copied_path.read_text(encoding="utf-8")),
                {"depth_levels": ["precomputed"]},
            )

    def test_copy_precomputed_analysis_grid_geojson_prefers_run_summary_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            globe_dir = run_dir / "globe"
            run_dir.mkdir(parents=True)
            source_path = run_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
            source_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": []}) + "\n",
                encoding="utf-8",
            )

            copied_path = _copy_precomputed_analysis_grid_geojson(
                run_dir=run_dir,
                globe_dir=globe_dir,
                run_summary={"error_analysis_grid_geojson_path": source_path.name},
            )

            self.assertEqual(
                copied_path, globe_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
            )
            self.assertEqual(
                json.loads(copied_path.read_text(encoding="utf-8")),
                {"type": "FeatureCollection", "features": []},
            )

    def test_resolve_depth_export_artifacts_uses_run_summary_depth_exports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            prediction_surface = (
                run_dir / "global_top_band_20260105_prediction_surface.tif"
            )
            prediction_100m = run_dir / "global_top_band_20260105_prediction_100m.tif"
            ground_truth_surface = (
                run_dir / "global_top_band_20260105_glorys_surface.tif"
            )
            absolute_error_surface = (
                run_dir / "global_top_band_20260105_absolute_error_surface.tif"
            )
            for path in (
                prediction_surface,
                prediction_100m,
                ground_truth_surface,
                absolute_error_surface,
            ):
                path.write_text("placeholder", encoding="utf-8")

            depth_exports = _resolve_depth_export_artifacts(
                run_dir=run_dir,
                run_summary={
                    "depth_exports": [
                        {
                            "suffix": "surface",
                            "label": "Surface",
                            "requested_depth_m": 0.0,
                            "actual_depth_m": 0.5,
                            "channel_index": 0,
                            "prediction_tif_path": prediction_surface.name,
                            "ground_truth_tif_path": ground_truth_surface.name,
                            "absolute_error_tif_path": absolute_error_surface.name,
                        },
                        {
                            "suffix": "100m",
                            "label": "100m",
                            "requested_depth_m": 100.0,
                            "actual_depth_m": 97.0,
                            "channel_index": 1,
                            "prediction_tif_path": prediction_100m.name,
                            "ground_truth_tif_path": None,
                        },
                    ]
                },
                prediction_path=prediction_surface,
                ground_truth_path=ground_truth_surface,
            )

        self.assertEqual(
            [item["suffix"] for item in depth_exports], ["surface", "100m"]
        )
        self.assertEqual(depth_exports[0]["prediction_path"], prediction_surface)
        self.assertEqual(depth_exports[0]["ground_truth_path"], ground_truth_surface)
        self.assertEqual(
            depth_exports[0]["absolute_error_path"], absolute_error_surface
        )
        self.assertIsNone(depth_exports[1]["ground_truth_path"])
        self.assertEqual(depth_exports[1]["channel_index"], 1)

    def test_absolute_error_color_scale_uses_robust_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tif_path = tmp_path / "absolute_error.tif"
            values = np.arange(100, dtype=np.float32).reshape(10, 10)
            values[0, 0] = -9999.0
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="float32",
                nodata=-9999.0,
                crs="EPSG:4326",
                transform=from_origin(0.0, 10.0, 1.0, 1.0),
            ) as ds:
                ds.write(values, 1)

            scale = _absolute_error_color_scale(tif_path)
            ramp_path = tmp_path / "error_ramp.txt"
            _write_absolute_error_color_ramp(
                ramp_path,
                color_scale_min_c=float(scale["color_scale_min_c"]),
                color_scale_max_c=float(scale["color_scale_max_c"]),
                valid_max_c=float(scale["valid_max_c"]),
            )
            ramp_text = ramp_path.read_text(encoding="utf-8")

        self.assertGreater(float(scale["color_scale_min_c"]), 0.0)
        self.assertLess(float(scale["color_scale_max_c"]), 99.0)
        self.assertGreater(int(scale["legend_max_c"]), 0)
        self.assertIn("34 197 94", ramp_text)
        self.assertIn("220 38 38", ramp_text)
        self.assertIn("nv   0 0 0 0", ramp_text)

    def test_estimate_native_zoom_level_matches_global_point_one_degree_raster(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tif_path = Path(tmp_dir) / "global_0p1deg.tif"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=1800,
                width=3600,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(-180.0, 90.0, 0.1, 0.1),
            ) as ds:
                ds.write(np.ones((1, 1800, 3600), dtype=np.float32))

            zoom_level = _estimate_native_zoom_level(tif_path)

        self.assertEqual(zoom_level, 4)

    def test_rewrite_geojson_rounds_coordinates_and_drops_unused_point_properties(
        self,
    ) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "argo-1",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [12.1234567, -45.7654321],
                    },
                    "properties": {"profile_id": 7, "temperature": 18.4},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.geojson"
            destination_path = Path(tmp_dir) / "rewritten.geojson"
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            _rewrite_geojson(
                source_path,
                destination_path,
                allowed_property_keys=(),
            )

            rewritten = json.loads(destination_path.read_text(encoding="utf-8"))

        feature = rewritten["features"][0]
        self.assertEqual(feature["id"], "argo-1")
        self.assertEqual(feature["geometry"]["coordinates"], [12.1235, -45.7654])
        self.assertNotIn("properties", feature)

    def test_rewrite_geojson_keeps_argo_point_popup_properties(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [12.1234567, -45.7654321],
                    },
                    "properties": {
                        "date": 20260105,
                        "patch_id": "patch-7",
                        "export_index": 11,
                        "pixel_row": 2,
                        "pixel_col": 3,
                        "temperature": 18.4,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.geojson"
            destination_path = Path(tmp_dir) / "rewritten.geojson"
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            _rewrite_geojson(
                source_path,
                destination_path,
                allowed_property_keys=ARGO_POINT_PROPERTY_KEYS,
            )

            rewritten = json.loads(destination_path.read_text(encoding="utf-8"))

        self.assertEqual(
            rewritten["features"][0]["properties"],
            {
                "date": 20260105,
                "patch_id": "patch-7",
                "export_index": 11,
                "pixel_row": 2,
                "pixel_col": 3,
            },
        )
        self.assertNotIn("temperature", rewritten["features"][0]["properties"])

    def test_rewrite_geojson_keeps_only_required_popup_and_split_properties(
        self,
    ) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [1.123456, 2.654321],
                                [3.123456, 4.654321],
                                [5.123456, 6.654321],
                                [1.123456, 2.654321],
                            ]
                        ],
                    },
                    "properties": {
                        "split": "train",
                        "unused": "value",
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-10.123456, 44.987654],
                    },
                    "properties": {
                        "date": 20260105,
                        "graph_png_path": "graphs/example.png",
                        "location_id": "sample-17",
                        "patch_id": "patch-3",
                        "pixel_row": 9,
                        "pixel_col": 11,
                        "extra": "drop-me",
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "source.geojson"
            patch_destination_path = Path(tmp_dir) / "patch.geojson"
            full_sample_destination_path = Path(tmp_dir) / "full_sample.geojson"
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            _rewrite_geojson(
                source_path,
                patch_destination_path,
                allowed_property_keys=("split",),
            )
            _rewrite_geojson(
                source_path,
                full_sample_destination_path,
                allowed_property_keys=FULL_SAMPLE_PROPERTY_KEYS,
            )

            rewritten_patch = json.loads(
                patch_destination_path.read_text(encoding="utf-8")
            )
            rewritten_full_sample = json.loads(
                full_sample_destination_path.read_text(encoding="utf-8")
            )

        self.assertEqual(
            rewritten_patch["features"][0]["properties"],
            {"split": "train"},
        )
        self.assertEqual(
            rewritten_patch["features"][0]["geometry"]["coordinates"][0][0],
            [1.1235, 2.6543],
        )
        self.assertEqual(
            rewritten_full_sample["features"][1]["properties"],
            {
                "date": 20260105,
                "graph_png_path": "graphs/example.png",
                "location_id": "sample-17",
                "patch_id": "patch-3",
                "pixel_row": 9,
                "pixel_col": 11,
            },
        )
        self.assertEqual(
            rewritten_full_sample["features"][1]["geometry"]["coordinates"],
            [-10.1235, 44.9877],
        )

    def test_rewrite_argo_sample_locations_geojson_merges_marker_types_without_duplicates(
        self,
    ) -> None:
        argo_payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [10.123456, -20.654321],
                    },
                    "properties": {
                        "date": 20260105,
                        "patch_id": "patch-1",
                        "export_index": 4,
                        "pixel_row": 5,
                        "pixel_col": 6,
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.987654, -30.123456],
                    },
                    "properties": {
                        "date": 20260105,
                        "patch_id": "patch-2",
                        "export_index": 8,
                        "pixel_row": 7,
                        "pixel_col": 9,
                    },
                },
            ],
        }
        full_sample_payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.9876544, -30.1234564],
                    },
                    "properties": {
                        "date": 20260105,
                        "graph_png_path": "graphs/full_sample_001.png",
                        "location_id": "full_sample_001",
                        "patch_id": "patch-2",
                        "pixel_row": 7,
                        "pixel_col": 9,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            argo_source_path = Path(tmp_dir) / "argo.geojson"
            full_sample_source_path = Path(tmp_dir) / "full_sample.geojson"
            destination_path = Path(tmp_dir) / "combined.geojson"
            argo_source_path.write_text(json.dumps(argo_payload), encoding="utf-8")
            full_sample_source_path.write_text(
                json.dumps(full_sample_payload), encoding="utf-8"
            )

            _rewrite_argo_sample_locations_geojson(
                destination_path,
                points_path=argo_source_path,
                full_sample_points_path=full_sample_source_path,
            )

            combined = json.loads(destination_path.read_text(encoding="utf-8"))

        self.assertEqual(len(combined["features"]), 2)
        argo_feature = combined["features"][0]
        full_sample_feature = combined["features"][1]
        self.assertEqual(
            argo_feature["properties"],
            {
                "date": 20260105,
                "patch_id": "patch-1",
                "export_index": 4,
                "pixel_row": 5,
                "pixel_col": 6,
                "marker_kind": "argo",
                "has_full_depth_graph": False,
            },
        )
        self.assertEqual(
            full_sample_feature["properties"],
            {
                "date": 20260105,
                "graph_png_path": "graphs/full_sample_001.png",
                "location_id": "full_sample_001",
                "patch_id": "patch-2",
                "pixel_row": 7,
                "pixel_col": 9,
                "marker_kind": "full_depth_profile",
                "has_full_depth_graph": True,
            },
        )
        self.assertEqual(
            full_sample_feature["geometry"]["coordinates"],
            [15.9877, -30.1235],
        )
        self.assertIn("marker_kind", ARGO_SAMPLE_LOCATION_PROPERTY_KEYS)
        self.assertIn("has_full_depth_graph", ARGO_SAMPLE_LOCATION_PROPERTY_KEYS)


if __name__ == "__main__":
    unittest.main()
