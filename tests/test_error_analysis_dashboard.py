from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

from depth_recon.inference.export_error_analysis_dashboard import (
    DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
    DEFAULT_GRID_SIZE_DEGREES,
    aggregate_by_grid,
    assign_ocean_basin,
    build_analysis_grid_geojson_payload,
    build_error_analysis_payload,
    export_error_analysis_dashboard,
    summarize_values,
    write_analysis_grid_geojson,
)


class TestErrorAnalysisDashboard(unittest.TestCase):
    def test_default_dashboard_grid_uses_five_degree_cells(self) -> None:
        self.assertEqual(DEFAULT_GRID_SIZE_DEGREES, 5.0)

    def test_assign_ocean_basin_uses_expected_diagnostic_buckets(self) -> None:
        self.assertEqual(assign_ocean_basin(-150.0, 20.0), "Pacific")
        self.assertEqual(assign_ocean_basin(-40.0, 20.0), "Atlantic")
        self.assertEqual(assign_ocean_basin(80.0, -20.0), "Indian")
        self.assertEqual(assign_ocean_basin(10.0, -70.0), "Southern")
        self.assertEqual(assign_ocean_basin(30.0, 75.0), "Arctic")
        self.assertEqual(assign_ocean_basin(30.0, 42.0), "Atlantic")
        self.assertEqual(assign_ocean_basin(80.0, 45.0), "Other")
        self.assertEqual(assign_ocean_basin(135.0, 35.0), "Pacific")

    def test_summarize_values_ignores_nan_and_reports_percentiles(self) -> None:
        stats = summarize_values(np.asarray([1.0, 2.0, 3.0, np.nan], dtype=np.float64))

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(float(stats["p90"]), 2.8)
        self.assertAlmostEqual(float(stats["p95"]), 2.9)

    def test_aggregate_by_grid_groups_fixed_lon_lat_cells(self) -> None:
        cells = aggregate_by_grid(
            np.asarray([1.0, 3.0, 9.0], dtype=np.float64),
            np.asarray([-175.0, -171.0, 35.0], dtype=np.float64),
            np.asarray([-85.0, -82.0, 5.0], dtype=np.float64),
            grid_size_degrees=10.0,
        )

        self.assertEqual(len(cells), 2)
        first = cells[0]
        self.assertEqual(first["count"], 2)
        self.assertEqual(first["west"], -180.0)
        self.assertEqual(first["south"], -90.0)
        self.assertEqual(first["median"], 2.0)
        self.assertEqual(first["basin"], "Southern")

    def test_analysis_grid_geojson_clips_cells_to_ocean_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = Path(tmp_dir) / "land_mask.tif"
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=4,
                width=4,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=from_origin(-180.0, 90.0, 90.0, 45.0),
            ) as ds:
                ds.write(
                    np.asarray(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 1, 1, 0],
                        ],
                        dtype=np.uint8,
                    ).reshape(1, 4, 4)
                )

            payload = build_analysis_grid_geojson_payload(
                land_mask_path=mask_path,
                grid_size_degrees=90.0,
            )
            output_path = write_analysis_grid_geojson(
                output_path=Path(tmp_dir) / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
                land_mask_path=mask_path,
                grid_size_degrees=90.0,
            )
            features_by_id = {
                feature["properties"]["id"]: feature for feature in payload["features"]
            }

            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.name, DEFAULT_ANALYSIS_GRID_GEOJSON_NAME)
            self.assertIn("cell_1_0", features_by_id)
            self.assertNotIn("cell_0_1", features_by_id)
            clipped = features_by_id["cell_1_0"]
            self.assertEqual(clipped["geometry"]["type"], "MultiPolygon")
            self.assertEqual(clipped["properties"]["ocean_pixel_count"], 1)
            self.assertEqual(clipped["properties"]["total_pixel_count"], 2)
            self.assertAlmostEqual(clipped["properties"]["ocean_fraction"], 0.5)
            exported = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exported["features"], payload["features"])

    def test_build_payload_filters_land_mask_and_labels_grid_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            transform = from_origin(-180.0, 90.0, 180.0, 90.0)
            error_path = run_dir / "absolute_error.tif"
            prediction_path = run_dir / "prediction.tif"
            ground_truth_path = run_dir / "ground_truth.tif"
            land_mask_path = run_dir / "land_mask.tif"

            data = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            for path in (error_path, prediction_path, ground_truth_path):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=2,
                    width=2,
                    count=1,
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=-9999.0,
                ) as ds:
                    ds.write(data.reshape(1, 2, 2))

            with rasterio.open(
                land_mask_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(np.asarray([[1, 0], [0, 0]], dtype=np.uint8).reshape(1, 2, 2))

            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variable": "temperature",
                        "prediction_tif_path": prediction_path.name,
                        "ground_truth_tif_path": ground_truth_path.name,
                        "land_mask_path": land_mask_path.name,
                        "depth_exports": [
                            {
                                "suffix": "surface",
                                "label": "Surface",
                                "requested_depth_m": 0.0,
                                "actual_depth_m": 0.0,
                                "channel_index": 0,
                                "prediction_tif_path": prediction_path.name,
                                "ground_truth_tif_path": ground_truth_path.name,
                                "absolute_error_tif_path": error_path.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_error_analysis_payload(
                run_dir=run_dir,
                grid_size_degrees=90.0,
            )

            depth = payload["depth_levels"][0]
            self.assertTrue(depth["is_aggregate"])
            self.assertEqual(depth["label"], "All Depths")
            self.assertEqual(depth["depth_count"], 1)
            self.assertEqual(depth["global"]["count"], 3)
            basin_by_label = {
                cell["label"]: cell["basin"] for cell in depth["grid_cells"]
            }
            self.assertEqual(basin_by_label["0 to 90 lat, 90 to 180 lon"], "Other")
            self.assertEqual(basin_by_label["-90 to 0 lat, -90 to 0 lon"], "Pacific")
            self.assertEqual(basin_by_label["-90 to 0 lat, 90 to 180 lon"], "Indian")

    def test_build_payload_and_export_dashboard_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            error_surface = run_dir / "surface_error.tif"
            error_10m = run_dir / "10m_error.tif"
            prediction = run_dir / "surface_prediction.tif"
            ground_truth = run_dir / "surface_glorys.tif"
            transform = from_origin(-180.0, 90.0, 90.0, 45.0)
            data = np.asarray(
                [
                    [1.0, 2.0, -9999.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                dtype=np.float32,
            )
            for path, offset in (
                (error_surface, 0.0),
                (error_10m, 10.0),
                (prediction, 0.0),
                (ground_truth, 0.0),
            ):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=4,
                    width=4,
                    count=1,
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=-9999.0,
                ) as ds:
                    raster = data + offset
                    raster[data == -9999.0] = -9999.0
                    ds.write(raster.reshape(1, 4, 4))

            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "selected_date": 20150615,
                        "iso_year": 2015,
                        "iso_week": 25,
                        "variable": "temperature",
                        "prediction_tif_path": prediction.name,
                        "ground_truth_tif_path": ground_truth.name,
                        "depth_exports": [
                            {
                                "suffix": "surface",
                                "label": "Surface",
                                "requested_depth_m": 0.0,
                                "actual_depth_m": 0.0,
                                "channel_index": 0,
                                "prediction_tif_path": prediction.name,
                                "ground_truth_tif_path": ground_truth.name,
                                "absolute_error_tif_path": error_surface.name,
                            },
                            {
                                "suffix": "10m",
                                "label": "10m",
                                "requested_depth_m": 10.0,
                                "actual_depth_m": 9.6,
                                "channel_index": 1,
                                "prediction_tif_path": prediction.name,
                                "ground_truth_tif_path": ground_truth.name,
                                "absolute_error_tif_path": error_10m.name,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_error_analysis_payload(
                run_dir=run_dir,
                grid_size_degrees=90.0,
                top_cell_count=3,
            )
            parallel_payload = build_error_analysis_payload(
                run_dir=run_dir,
                grid_size_degrees=90.0,
                top_cell_count=3,
                analysis_workers=2,
            )
            result = export_error_analysis_dashboard(
                run_dir=run_dir,
                grid_size_degrees=90.0,
                top_cell_count=3,
                analysis_workers=2,
            )

            self.assertEqual(payload["depth_levels"], parallel_payload["depth_levels"])
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["run"]["selected_date"], 20150615)
            self.assertEqual(payload["variable"]["name"], "temperature")
            self.assertEqual(len(payload["depth_levels"]), 3)
            self.assertTrue(payload["depth_levels"][0]["is_aggregate"])
            self.assertEqual(payload["depth_levels"][0]["label"], "All Depths")
            self.assertEqual(payload["depth_levels"][0]["depth_count"], 2)
            self.assertEqual(payload["depth_levels"][0]["global"]["count"], 30)
            self.assertEqual(payload["depth_levels"][1]["global"]["count"], 15)
            self.assertIn("basins", payload["depth_levels"][0])
            self.assertIn("grid_cells", payload["depth_levels"][0])
            self.assertEqual(len(payload["depth_levels"][0]["top_cells"]["p95"]), 3)

            json_path = Path(result["json_path"])
            self.assertTrue(json_path.is_file())
            self.assertNotIn("html_path", result)
            self.assertFalse((json_path.parent / "error-analysis.html").exists())
            exported = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(exported["schema_version"], 1)

    def test_standalone_analysis_dashboard_page_is_nav_linked(self) -> None:
        html = Path("docs/analysis/index.html").read_text(encoding="utf-8")
        css = Path("docs/stylesheets/analysis-dashboard.css").read_text(
            encoding="utf-8"
        )
        script = Path("docs/javascripts/analysis-dashboard.js").read_text(
            encoding="utf-8"
        )
        mkdocs_config = Path("mkdocs.yml").read_text(encoding="utf-8")

        self.assertIn('class="standalone-analysis-root"', html)
        self.assertIn("Analysis Dashboard", html)
        self.assertIn('id="analysis-map"', html)
        self.assertIn('id="analysis-depth-profile"', html)
        self.assertIn('id="analysis-depth-scale-toggle"', html)
        self.assertIn('id="analysis-basin-fan-toggle"', html)
        self.assertIn('id="analysis-basin-chart"', html)
        self.assertIn("analysis-panel--depth-profile", html)
        self.assertIn("Basin Selector", html)
        self.assertNotIn("Rankings", html)
        self.assertNotIn("analysis-kpis", html)
        self.assertLess(
            html.index("analysis-panel--depth-profile"),
            html.index('class="analysis-charts"'),
        )
        self.assertLess(
            html.index("analysis-panel--rankings"),
            html.index('id="analysis-basin-chart"'),
        )
        self.assertIn("analysis-dashboard.js", html)
        self.assertIn("leaflet@1.9.4", html)
        self.assertIn("plotly-2.35.2", html)
        self.assertIn("analysis-map-legend", html)
        self.assertIn('id="analysis-reset-focus"', html)
        self.assertIn('class="analysis-chart"', html)
        self.assertNotIn("analysis-focus-select", html)
        self.assertNotIn("Hotspot Cells", html)
        self.assertIn("standalone-analysis-page", css)
        self.assertIn("#061726", css)
        self.assertIn("#7cc8ff", css)
        self.assertIn("analysis-error-state", css)
        self.assertIn("analysis-leaflet-tooltip", css)
        self.assertIn("analysis-ranking-reset", css)
        self.assertIn("analysis-scale-toggle", css)
        self.assertIn("analysis-depth-actions", css)
        self.assertIn("analysis-panel--depth-profile", css)
        self.assertIn("repeat(2, minmax(0, 1fr))", css)
        self.assertIn("analysis-chart", css)
        self.assertIn("DEFAULT_GLOBE_CONFIG_URL", script)
        self.assertIn("function loadAllAnalysisData", script)
        self.assertIn("analysisSourcesFromConfig", script)
        self.assertIn("analysis-modality-select", script)
        self.assertIn("function renderLoadFailure", script)
        self.assertIn("validateAnalysisPayload", script)
        self.assertIn("analysis-load-failed", script)
        self.assertIn("function requireDashboardLibraries", script)
        self.assertIn("function mapColorDomain", script)
        self.assertIn("analysis_grid_geojson_url", script)
        self.assertIn("function indexAnalysisGridGeoJson", script)
        self.assertIn("function mapCellLayer", script)
        self.assertIn("window.L.geoJSON", script)
        self.assertIn("coast-clipped", script)
        self.assertIn("quantile(values, 0.95)", script)
        self.assertIn("cell.basin", script)
        self.assertIn("lonValue >= 100", script)
        self.assertIn("analysis-reset-focus", script)
        self.assertIn("depthProfileLogX", script)
        self.assertIn("showBasinFan: true", script)
        self.assertIn("function logDepthXValues", script)
        self.assertIn("COMMON_SELECTOR_DEPTHS_M", script)
        self.assertIn("BASIN_ORDER", script)
        self.assertIn("BASIN_FAN_COLORS", script)
        self.assertIn("function chartDepthLevels", script)
        self.assertIn("function selectableDepthIndices", script)
        self.assertIn("function updateBasinFanToggle", script)
        self.assertIn("function selectedDepthPointIndex", script)
        self.assertIn("function commonRasterDepthPointIndices", script)
        self.assertIn("function primaryMarkerSymbols", script)
        self.assertIn("Raster export depth", script)
        self.assertIn("function basinFanTraces", script)
        self.assertIn("function selectedHoverData", script)
        self.assertIn("function selectDepthFromProfileClick", script)
        self.assertIn("!depth.is_aggregate", script)
        self.assertIn("primaryMarkerColors(depthLevels", script)
        self.assertIn('chart.on("plotly_click"', script)
        self.assertIn("state.depthIndex === clickedDepthIndex", script)
        self.assertIn("metricLabel(state.metric)", script)
        self.assertIn("absolute error across depth", script)
        self.assertIn(
            'state.focus = { type: "global", id: "global", label: "Global" }', script
        )
        self.assertIn("window.L.map", script)
        self.assertIn("window.Plotly.react", script)
        self.assertIn("displayBasinName", script)
        self.assertIn(
            'layout.xaxis.type = state.depthProfileLogX ? "log" : "linear"',
            script,
        )
        self.assertNotIn("function renderKpis", script)
        self.assertIn("state.focus = active", script)
        self.assertNotIn('params.get("data")', script)
        self.assertIn(
            "Analysis Dashboard: https://depthdif.donike.net/analysis/", mkdocs_config
        )
        self.assertIn("3D Globe: https://depthdif.donike.net/globe/", mkdocs_config)


if __name__ == "__main__":
    unittest.main()
