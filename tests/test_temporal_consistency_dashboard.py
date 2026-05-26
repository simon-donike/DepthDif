from __future__ import annotations

from types import SimpleNamespace
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

from depth_recon.inference.export_error_analysis_dashboard import BASIN_NAMES
from depth_recon.inference.export_temporal_consistency_dashboard import (
    DEFAULT_TEMPORAL_BASIN_MAP_GEOJSON_NAME,
    DEFAULT_TEMPORAL_CONFIG_NAME,
    build_temporal_analysis_payload,
    export_temporal_dashboard_assets,
)
from depth_recon.inference.export_temporal_cesium_globe_assets import (
    DEFAULT_TEMPORAL_GLOBE_CONFIG_NAME,
    export_temporal_cesium_globe_assets,
)
from depth_recon.inference.export_global_variables import (
    _apply_sampling_defaults_from_config,
    _build_parser as _build_standard_variable_parser,
    _temporal_single_export_args,
    run_global_variable_inference,
)
from depth_recon.inference.export_temporal_global_variables import (
    _build_parser as _build_temporal_runner_parser,
    run_temporal_global_variable_inference,
)


class TestTemporalConsistencyDashboard(unittest.TestCase):
    def _write_raster(self, path: Path, data: np.ndarray) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-180.0, 90.0, 180.0, 90.0),
            nodata=-9999.0,
        ) as dataset:
            dataset.write(np.asarray(data, dtype=np.float32).reshape(1, *data.shape))

    def _basin_rows(
        self, basin: str, *, count: int, total: float
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for name in BASIN_NAMES:
            basin_count = int(count) if name == basin else 0
            basin_total = float(total) if name == basin else 0.0
            rows.append(
                {
                    "name": name,
                    "count": basin_count,
                    "sum_absolute_error": basin_total,
                    "mean_absolute_error": (
                        None if basin_count <= 0 else basin_total / float(basin_count)
                    ),
                }
            )
        return rows

    def _compact_depth_levels(
        self, basin: str, step_index: int
    ) -> list[dict[str, object]]:
        return [
            {
                "index": 0,
                "suffix": "depth_000",
                "label": "Surface",
                "requested_depth_m": 0.0,
                "actual_depth_m": 0.5,
                "channel_index": 0,
                "basins": self._basin_rows(
                    basin,
                    count=2,
                    total=4.0 + float(step_index) * 2.0,
                ),
            },
            {
                "index": 1,
                "suffix": "depth_001",
                "label": "10m",
                "requested_depth_m": 10.0,
                "actual_depth_m": 9.6,
                "channel_index": 1,
                "basins": self._basin_rows(
                    basin,
                    count=4,
                    total=8.0 + float(step_index) * 4.0,
                ),
            },
        ]

    def _write_run(
        self,
        root: Path,
        *,
        variable: str,
        date_value: int,
        iso_year: int,
        iso_week: int,
        basin: str = "North Pacific Ocean",
        step_index: int = 0,
    ) -> Path:
        run_dir = root / variable / f"{iso_year}_W{iso_week:02d}"
        run_dir.mkdir(parents=True)
        prediction_path = run_dir / "prediction_10m.tif"
        absolute_error_path = run_dir / "absolute_error_10m.tif"
        self._write_raster(
            prediction_path, np.full((2, 2), step_index, dtype=np.float32)
        )
        self._write_raster(
            absolute_error_path,
            np.full((2, 2), step_index + 1, dtype=np.float32),
        )
        compact_path = run_dir / "temporal-basin-depth-errors.json"
        compact_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "basin_depth_error_summary",
                    "run": {
                        "selected_date": date_value,
                        "target_date": date_value,
                        "iso_year": iso_year,
                        "iso_week": iso_week,
                    },
                    "variable": {
                        "name": variable,
                        "label": (
                            "Temperature" if variable == "temperature" else "Salinity"
                        ),
                        "value_units": (
                            "celsius" if variable == "temperature" else "psu"
                        ),
                        "value_unit_label": (
                            "deg C" if variable == "temperature" else "PSU"
                        ),
                    },
                    "depth_levels": self._compact_depth_levels(basin, step_index),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_summary.yaml").write_text(
            yaml.safe_dump(
                {
                    "selected_date": date_value,
                    "target_date": date_value,
                    "iso_year": iso_year,
                    "iso_week": iso_week,
                    "variable": variable,
                    "variable_label": (
                        "Temperature" if variable == "temperature" else "Salinity"
                    ),
                    "value_units": "celsius" if variable == "temperature" else "psu",
                    "value_unit_label": "deg C" if variable == "temperature" else "PSU",
                    "prediction_tif_path": prediction_path.name,
                    "temporal_basin_depth_error_json_path": compact_path.name,
                    "depth_exports": [
                        {
                            "suffix": "10m",
                            "label": "10m",
                            "requested_depth_m": 10.0,
                            "actual_depth_m": 9.6,
                            "channel_index": 1,
                            "prediction_tif_path": prediction_path.name,
                            "ground_truth_tif_path": None,
                            "absolute_error_tif_path": absolute_error_path.name,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def _write_two_week_series(self, root: Path, variable: str) -> list[Path]:
        return [
            self._write_run(
                root,
                variable=variable,
                date_value=20180103,
                iso_year=2018,
                iso_week=1,
                step_index=0,
            ),
            self._write_run(
                root,
                variable=variable,
                date_value=20180110,
                iso_year=2018,
                iso_week=2,
                step_index=1,
            ),
        ]

    def test_build_temporal_payload_aggregates_yearly_basin_depth_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dirs = self._write_two_week_series(root, "temperature")

            payload = build_temporal_analysis_payload(
                run_dirs=run_dirs, validation_year=2018
            )

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["validation_year"], 2018)
        self.assertEqual(payload["variable"]["name"], "temperature")
        self.assertEqual(payload["run"]["run_count"], 2)
        north_pacific = payload["basin_depth_errors"]["North Pacific Ocean"]
        self.assertEqual(
            [depth["label"] for depth in payload["depth_levels"]], ["Surface", "10m"]
        )
        self.assertEqual(north_pacific[0]["count"], 4)
        self.assertEqual(north_pacific[0]["sum_absolute_error"], 10.0)
        self.assertEqual(north_pacific[0]["mean_absolute_error"], 2.5)
        self.assertEqual(north_pacific[1]["count"], 8)
        self.assertEqual(north_pacific[1]["mean_absolute_error"], 2.5)
        self.assertEqual(
            payload["basin_depth_errors"]["North Atlantic Ocean"][0]["count"], 0
        )
        self.assertIsNone(
            payload["basin_depth_errors"]["North Atlantic Ocean"][0][
                "mean_absolute_error"
            ]
        )

    def test_export_temporal_assets_writes_schema_v2_config_and_basin_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            temperature_runs = self._write_two_week_series(root, "temperature")
            salinity_runs = self._write_two_week_series(root, "salinity")
            output_dir = root / "temporal"

            result = export_temporal_dashboard_assets(
                variable_run_dirs={
                    "temperature": temperature_runs,
                    "salinity": salinity_runs,
                },
                output_dir=output_dir,
                public_base_url="https://example.com/temporal",
                copy_dashboard=False,
                validation_year=2018,
            )

            config = json.loads(
                (output_dir / DEFAULT_TEMPORAL_CONFIG_NAME).read_text(encoding="utf-8")
            )
            pacific_payload = json.loads(
                (output_dir / "basins" / "north_pacific_ocean.json").read_text(
                    encoding="utf-8"
                )
            )
            basin_map_exists = (
                output_dir / DEFAULT_TEMPORAL_BASIN_MAP_GEOJSON_NAME
            ).is_file()
            temperature_prediction_exists = (
                output_dir
                / "weekly"
                / "temperature"
                / "2018_W01"
                / "prediction_10m.tif"
            ).is_file()
            temperature_error_exists = (
                output_dir
                / "weekly"
                / "temperature"
                / "2018_W01"
                / "absolute_error_10m.tif"
            ).is_file()
            old_temporal_json_exists = (
                output_dir / "temperature" / "temporal-analysis.json"
            ).exists()

        self.assertEqual(result["default_variable"], "temperature")
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["validation_year"], 2018)
        self.assertEqual(config["default_basin"], "North Pacific Ocean")
        self.assertEqual(
            config["basin_map_geojson_url"],
            "https://example.com/temporal/basin-map.geojson",
        )
        self.assertEqual(
            config["basin_data_urls"]["North Pacific Ocean"],
            "https://example.com/temporal/basins/north_pacific_ocean.json",
        )
        self.assertTrue(basin_map_exists)
        self.assertTrue(temperature_prediction_exists)
        self.assertTrue(temperature_error_exists)
        self.assertFalse(old_temporal_json_exists)
        self.assertEqual(pacific_payload["schema_version"], 2)
        self.assertIn("temperature", pacific_payload["variables"])
        self.assertIn("salinity", pacific_payload["variables"])
        self.assertEqual(
            pacific_payload["variables"]["temperature"]["depth_errors"][0][
                "mean_absolute_error"
            ],
            2.5,
        )

    def test_export_temporal_globe_assets_tiles_weekly_10m_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            temperature_runs = self._write_two_week_series(root / "runs", "temperature")
            salinity_runs = self._write_two_week_series(root / "runs", "salinity")
            output_dir = root / "temporal-globe"

            with (
                mock.patch(
                    "depth_recon.inference.export_temporal_cesium_globe_assets._colorize_raster"
                ) as colorize_mock,
                mock.patch(
                    "depth_recon.inference.export_temporal_cesium_globe_assets._run_gdal2tiles"
                ) as tiles_mock,
                mock.patch(
                    "depth_recon.inference.export_temporal_cesium_globe_assets._export_base_map_tiles",
                    return_value=(None, None, None),
                ),
            ):
                result = export_temporal_cesium_globe_assets(
                    variable_run_dirs={
                        "temperature": temperature_runs,
                        "salinity": salinity_runs,
                    },
                    output_dir=output_dir,
                    public_base_url="https://example.com/temporal-globe",
                    validation_year=2018,
                    copy_viewer=True,
                )

            config = json.loads(
                (output_dir / DEFAULT_TEMPORAL_GLOBE_CONFIG_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                result["config_path"],
                str(output_dir / DEFAULT_TEMPORAL_GLOBE_CONFIG_NAME),
            )
            self.assertEqual(config["schema_version"], 1)
            self.assertEqual(config["validation_year"], 2018)
            self.assertEqual(config["depth_suffix"], "10m")
            self.assertEqual(config["default_layer"], "prediction")
            self.assertEqual(config["frame_interval_ms"], 1000)
            self.assertEqual(config["webp_quality"], 80)
            self.assertEqual(config["max_zoom_level"], 4)
            self.assertEqual(config["available_variables"], ["temperature", "salinity"])
            first_frame = config["variables"]["temperature"]["frames"][0]
            self.assertEqual(first_frame["label"], "2018-W01")
            self.assertEqual(
                first_frame["prediction_tiles_url"],
                "https://example.com/temporal-globe/frames/temperature/2018_W01/prediction_tiles_10m",
            )
            self.assertEqual(
                first_frame["absolute_error_tiles_url"],
                "https://example.com/temporal-globe/frames/temperature/2018_W01/absolute_error_tiles_10m",
            )
            self.assertEqual(colorize_mock.call_count, 8)
            self.assertEqual(tiles_mock.call_count, 8)
            first_tiles_kwargs = tiles_mock.call_args_list[0].kwargs
            self.assertEqual(first_tiles_kwargs["extra_zoom_levels"], 0)
            self.assertEqual(first_tiles_kwargs["max_zoom_level"], 4)
            self.assertEqual(first_tiles_kwargs["webp_quality"], 80)
            self.assertTrue((output_dir / "index.html").exists())
            self.assertTrue((output_dir / "javascripts/temporal-globe.js").exists())

    def test_export_temporal_assets_rejects_mismatched_variable_weeks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            temperature_runs = self._write_two_week_series(root, "temperature")
            salinity_runs = [
                self._write_run(
                    root,
                    variable="salinity",
                    date_value=20180117,
                    iso_year=2018,
                    iso_week=3,
                )
            ]

            with self.assertRaisesRegex(ValueError, "matching weekly runs"):
                export_temporal_dashboard_assets(
                    variable_run_dirs={
                        "temperature": temperature_runs,
                        "salinity": salinity_runs,
                    },
                    output_dir=root / "temporal",
                    copy_dashboard=False,
                )

    def test_temporal_runner_defaults_to_complete_2018_validation_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_temporal_runner_parser().parse_args(
                [
                    "--temperature-checkpoint",
                    "temperature.ckpt",
                    "--salinity-checkpoint",
                    "salinity.ckpt",
                    "--device",
                    "cpu",
                    "--output-root",
                    str(output_root),
                    "--output-name",
                    "temporal_test",
                    "--no-multi-gpu",
                ]
            )

            def fake_run_global_inference(parsed_args):
                run_dir = Path(parsed_args.output_root) / parsed_args.output_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_summary.yaml").write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(run_dir=run_dir)

            with (
                mock.patch(
                    "depth_recon.inference.export_temporal_global_variables.run_global_inference",
                    side_effect=fake_run_global_inference,
                ) as run_mock,
                mock.patch(
                    "depth_recon.inference.export_temporal_global_variables.export_temporal_dashboard_assets",
                    return_value={"config_path": "temporal-config.json"},
                ) as export_mock,
            ):
                summary = run_temporal_global_variable_inference(args)

        self.assertEqual(run_mock.call_count, 104)
        first_call_args = run_mock.call_args_list[0].args[0]
        last_call_args = run_mock.call_args_list[-1].args[0]
        self.assertEqual(first_call_args.scenario, "temperature")
        self.assertEqual(first_call_args.year, 2018)
        self.assertEqual(first_call_args.iso_week, 1)
        self.assertEqual(first_call_args.depth_export_suffix, ["10m"])
        self.assertTrue(first_call_args.compact_basin_depth_error)
        self.assertFalse(first_call_args.persist_ground_truth_rasters)
        self.assertEqual(last_call_args.scenario, "salinity")
        self.assertEqual(last_call_args.year, 2018)
        self.assertEqual(last_call_args.iso_week, 52)
        export_mock.assert_called_once()
        self.assertEqual(summary["week_count"], 52)
        self.assertEqual(summary["weeks"][-1], {"iso_year": 2018, "iso_week": 52})

    def test_temporal_runner_can_package_temporal_globe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_temporal_runner_parser().parse_args(
                [
                    "--temperature-checkpoint",
                    "temperature.ckpt",
                    "--salinity-checkpoint",
                    "salinity.ckpt",
                    "--device",
                    "cpu",
                    "--output-root",
                    str(output_root),
                    "--output-name",
                    "temporal_globe_test",
                    "--public-base-url",
                    "https://example.com/temporal",
                    "--rclone-remote",
                    "r2:bucket/temporal",
                    "--export-temporal-globe",
                    "--no-multi-gpu",
                ]
            )

            def fake_run_global_inference(parsed_args):
                run_dir = Path(parsed_args.output_root) / parsed_args.output_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_summary.yaml").write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(run_dir=run_dir)

            with (
                mock.patch(
                    "depth_recon.inference.export_temporal_global_variables.run_global_inference",
                    side_effect=fake_run_global_inference,
                ),
                mock.patch(
                    "depth_recon.inference.export_temporal_global_variables.export_temporal_dashboard_assets",
                    return_value={"config_path": "temporal-config.json"},
                ),
                mock.patch(
                    "depth_recon.inference.export_temporal_global_variables.export_temporal_cesium_globe_assets",
                    return_value={"config_path": "temporal-globe-config.json"},
                ) as globe_mock,
            ):
                summary = run_temporal_global_variable_inference(args)

        globe_kwargs = globe_mock.call_args.kwargs
        self.assertEqual(
            globe_kwargs["output_dir"],
            output_root / "temporal_globe_test" / "temporal-globe",
        )
        self.assertEqual(
            globe_kwargs["public_base_url"], "https://example.com/temporal-globe"
        )
        self.assertEqual(globe_kwargs["rclone_remote"], "r2:bucket/temporal-globe")
        self.assertEqual(globe_kwargs["max_zoom_level"], 4)
        self.assertEqual(globe_kwargs["webp_quality"], 80)
        self.assertEqual(
            summary["temporal_globe"], {"config_path": "temporal-globe-config.json"}
        )

    def test_standard_variable_inference_can_export_temporal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_standard_variable_parser().parse_args(
                [
                    "--year",
                    "2018",
                    "--iso-week",
                    "25",
                    "--temperature-checkpoint",
                    "temperature.ckpt",
                    "--salinity-checkpoint",
                    "salinity.ckpt",
                    "--device",
                    "cpu",
                    "--output-root",
                    str(output_root),
                    "--output-name",
                    "standard_temporal",
                    "--export-temporal-consistency",
                    "--export-temporal-globe",
                    "--no-multi-gpu",
                ]
            )

            def fake_run_global_inference(parsed_args):
                run_dir = Path(parsed_args.output_root) / parsed_args.output_name
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "run_summary.yaml"
                summary_path.write_text("{}\n", encoding="utf-8")
                return SimpleNamespace(
                    run_dir=run_dir,
                    summary_path=summary_path,
                    selected_date=20180620,
                    iso_year=parsed_args.year,
                    iso_week=parsed_args.iso_week,
                    uncertainty_tif_path=None,
                )

            with (
                mock.patch(
                    "depth_recon.inference.export_global_variables.run_global_inference",
                    side_effect=fake_run_global_inference,
                ) as run_mock,
                mock.patch(
                    "depth_recon.inference.export_global_variables.export_cesium_globe_variable_assets",
                    return_value={"globe_dir": "globe"},
                ),
                mock.patch(
                    "depth_recon.inference.export_global_variables.export_temporal_dashboard_assets",
                    return_value={"output_dir": "temporal", "upload_ok": True},
                ) as temporal_mock,
                mock.patch(
                    "depth_recon.inference.export_global_variables.export_temporal_cesium_globe_assets",
                    return_value={"output_dir": "temporal-globe", "upload_ok": True},
                ) as temporal_globe_mock,
            ):
                summary = run_global_variable_inference(args)

        self.assertEqual(run_mock.call_count, 106)
        first_call_args = run_mock.call_args_list[0].args[0]
        extra_call_args = run_mock.call_args_list[2].args[0]
        last_call_args = run_mock.call_args_list[-1].args[0]
        self.assertEqual(first_call_args.scenario, "temperature")
        self.assertEqual(first_call_args.output_name, "temperature")
        self.assertEqual(extra_call_args.scenario, "temperature")
        self.assertEqual(extra_call_args.year, 2018)
        self.assertEqual(extra_call_args.iso_week, 1)
        self.assertEqual(extra_call_args.output_name, "2018_W01")
        self.assertEqual(extra_call_args.depth_export_suffix, ["10m"])
        self.assertTrue(extra_call_args.compact_basin_depth_error)
        self.assertFalse(extra_call_args.persist_ground_truth_rasters)
        self.assertEqual(last_call_args.scenario, "salinity")
        self.assertEqual(last_call_args.year, 2018)
        self.assertEqual(last_call_args.iso_week, 52)

        temporal_kwargs = temporal_mock.call_args.kwargs
        self.assertEqual(
            temporal_kwargs["output_dir"],
            output_root / "standard_temporal" / "temporal",
        )
        self.assertEqual(temporal_kwargs["validation_year"], 2018)
        self.assertEqual(len(temporal_kwargs["variable_run_dirs"]["temperature"]), 52)
        self.assertEqual(len(temporal_kwargs["variable_run_dirs"]["salinity"]), 52)
        globe_kwargs = temporal_globe_mock.call_args.kwargs
        self.assertEqual(
            globe_kwargs["output_dir"],
            output_root / "standard_temporal" / "temporal-globe",
        )
        self.assertEqual(len(globe_kwargs["variable_run_dirs"]["temperature"]), 52)
        self.assertEqual(globe_kwargs["max_zoom_level"], 4)
        self.assertEqual(globe_kwargs["webp_quality"], 80)
        self.assertTrue(summary["temporal_consistency"]["enabled"])
        self.assertEqual(summary["temporal_consistency"]["week_count"], 52)
        self.assertEqual(
            summary["temporal_consistency"]["weeks"][-1],
            {"iso_year": 2018, "iso_week": 52},
        )
        self.assertEqual(
            summary["temporal_consistency"]["globe"],
            {"output_dir": "temporal-globe", "upload_ok": True},
        )

    def test_standard_variable_temporal_sampling_defaults_use_yaml_uncertainty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_standard_variable_parser().parse_args(
                [
                    "--year",
                    "2018",
                    "--iso-week",
                    "22",
                    "--temperature-checkpoint",
                    "temperature.ckpt",
                    "--salinity-checkpoint",
                    "salinity.ckpt",
                    "--device",
                    "cpu",
                    "--output-root",
                    str(output_root),
                ]
            )

            config_bundle = SimpleNamespace(
                data_cfg={"split": {"val_year": 2018}},
                inference_cfg={
                    "inference": {
                        "sampling": {
                            "sampler": "ddim",
                            "ddim_num_timesteps": 100,
                        },
                        "uncertainty_sampling": {
                            "sampler": "ddim",
                            "ddim_num_timesteps": 50,
                        },
                    }
                },
            )
            with mock.patch(
                "depth_recon.inference.export_global_variables.load_pixel_inference_config",
                return_value=config_bundle,
            ):
                resolved_args = _apply_sampling_defaults_from_config(args)

            temporal_args = _temporal_single_export_args(
                resolved_args,
                scenario="temperature",
                checkpoint_path="temperature.ckpt",
                year=2018,
                iso_week=23,
                output_root=output_root,
            )

        self.assertEqual(resolved_args.sampler, "ddim")
        self.assertEqual(resolved_args.ddim_num_timesteps, 100)
        self.assertEqual(resolved_args.uncertainty_sampler, "ddim")
        self.assertEqual(resolved_args.uncertainty_ddim_num_timesteps, 50)
        self.assertEqual(resolved_args.configured_val_year, 2018)
        self.assertEqual(temporal_args.sampler, "ddim")
        self.assertEqual(temporal_args.ddim_num_timesteps, 50)
        self.assertEqual(temporal_args.depth_export_suffix, ["10m"])
        self.assertTrue(temporal_args.compact_basin_depth_error)
        self.assertFalse(temporal_args.persist_ground_truth_rasters)

    def test_temporal_globe_page_is_standalone_and_lightweight(self) -> None:
        html = Path("docs/temporal-globe/index.html").read_text(encoding="utf-8")
        loader = Path("docs/javascripts/load-temporal-globe.js").read_text(
            encoding="utf-8"
        )
        script = Path("docs/javascripts/temporal-globe.js").read_text(encoding="utf-8")
        css = Path("docs/stylesheets/globe.css").read_text(encoding="utf-8")
        mkdocs_config = Path("mkdocs.yml").read_text(encoding="utf-8")

        self.assertIn("Temporal Globe", html)
        self.assertIn('href="../visualizations/">Back to Analysis</a>', html)
        self.assertIn('id="temporal-globe-week-slider"', html)
        self.assertIn('id="temporal-globe-play-toggle"', html)
        self.assertIn('name="temporal-globe-variable"', html)
        self.assertIn('name="temporal-globe-layer"', html)
        self.assertIn("load-temporal-globe.js", html)
        self.assertIn("temporal-globe.js", loader)
        self.assertIn("DEFAULT_TEMPORAL_GLOBE_CONFIG_URL", script)
        self.assertIn('params.get("config")', script)
        self.assertIn("function preloadNeighborFrames", script)
        self.assertIn("function pruneLayerCache", script)
        self.assertIn("frame_interval_ms", script)
        self.assertNotIn("ground_truth", script)
        self.assertNotIn("uncertainty", script)
        self.assertIn(".globe-toolbar--temporal", css)
        self.assertIn("Analysis: /visualizations/", mkdocs_config)
        self.assertNotIn(
            "Temporal Globe: https://depthdif.donike.net/temporal-globe/",
            mkdocs_config,
        )

    def test_temporal_dashboard_page_is_standalone_and_nav_linked(self) -> None:
        html = Path("docs/temporal/index.html").read_text(encoding="utf-8")
        css = Path("docs/stylesheets/temporal-dashboard.css").read_text(
            encoding="utf-8"
        )
        script = Path("docs/javascripts/temporal-dashboard.js").read_text(
            encoding="utf-8"
        )
        mkdocs_config = Path("mkdocs.yml").read_text(encoding="utf-8")

        self.assertIn('class="standalone-temporal-root"', html)
        self.assertIn("Temporal Dashboard", html)
        self.assertIn('<a href="../visualizations/">Back to Analysis</a>', html)
        self.assertNotIn('href="../analysis/">Spatial Dashboard</a>', html)
        self.assertNotIn('href="../globe/">Globe</a>', html)
        self.assertIn('id="temporal-map"', html)
        self.assertIn('id="temporal-dashboard-select"', html)
        self.assertIn('id="temporal-basin-select"', html)
        self.assertIn('id="temporal-variable-select"', html)
        self.assertIn('id="temporal-depth-error"', html)
        self.assertNotIn('id="temporal-field-select"', html)
        self.assertNotIn('id="temporal-period-select"', html)
        self.assertNotIn('id="temporal-metric-toggle"', html)
        self.assertNotIn("Time Series", html)
        self.assertIn("temporal-dashboard.js", html)
        self.assertIn("standalone-temporal-page", css)
        self.assertIn("--temporal-teal", css)
        self.assertIn("temporal-workspace", css)
        self.assertIn("DEFAULT_TEMPORAL_CONFIG_URL", script)
        self.assertIn('params.get("config")', script)
        self.assertIn("schema_version", script)
        self.assertIn("basin_data_urls", script)
        self.assertIn("function loadActiveBasinData", script)
        self.assertIn("function renderDepthErrorGraph", script)
        self.assertIn('window.location.href = "../analysis/"', script)
        self.assertNotIn("prediction_flicker", script)
        self.assertNotIn("change_error", script)
        self.assertNotIn("temporal-analysis.json", script)
        self.assertIn("Analysis: /visualizations/", mkdocs_config)
        self.assertNotIn(
            "Temporal Dashboard: https://depthdif.donike.net/temporal/",
            mkdocs_config,
        )


if __name__ == "__main__":
    unittest.main()
