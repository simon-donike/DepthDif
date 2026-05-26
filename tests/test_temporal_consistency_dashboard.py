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

from depth_recon.inference.export_temporal_consistency_dashboard import (
    DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME,
    DEFAULT_TEMPORAL_CONFIG_NAME,
    build_temporal_analysis_payload,
    export_temporal_dashboard_assets,
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

    def _write_run(
        self,
        root: Path,
        *,
        variable: str,
        date_value: int,
        iso_year: int,
        iso_week: int,
        surface_scale: float,
        depth_scale: float,
        step_index: int,
    ) -> Path:
        run_dir = root / variable / f"{iso_year}_W{iso_week:02d}"
        run_dir.mkdir(parents=True)
        base = np.zeros((2, 2), dtype=np.float32)
        truth = base + float(step_index)
        surface_prediction = truth.copy()
        depth_prediction = truth.copy()
        if step_index == 2:
            surface_prediction = surface_prediction + float(surface_scale)
            depth_prediction = depth_prediction + float(depth_scale)

        surface_prediction_path = run_dir / "prediction_surface.tif"
        surface_truth_path = run_dir / "glorys_surface.tif"
        depth_prediction_path = run_dir / "prediction_10m.tif"
        depth_truth_path = run_dir / "glorys_10m.tif"
        for path, data in (
            (surface_prediction_path, surface_prediction),
            (surface_truth_path, truth),
            (depth_prediction_path, depth_prediction),
            (depth_truth_path, truth),
        ):
            self._write_raster(path, data)

        land_mask_path = run_dir / "land_mask.tif"
        with rasterio.open(
            land_mask_path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(-180.0, 90.0, 180.0, 90.0),
        ) as dataset:
            dataset.write(np.zeros((1, 2, 2), dtype=np.uint8))

        (run_dir / "run_summary.yaml").write_text(
            yaml.safe_dump(
                {
                    "selected_date": date_value,
                    "target_date": date_value,
                    "iso_year": iso_year,
                    "iso_week": iso_week,
                    "variable": variable,
                    "land_mask_path": land_mask_path.name,
                    "prediction_tif_path": surface_prediction_path.name,
                    "ground_truth_tif_path": surface_truth_path.name,
                    "depth_exports": [
                        {
                            "suffix": "surface",
                            "label": "Surface",
                            "requested_depth_m": 0.0,
                            "actual_depth_m": 0.0,
                            "channel_index": 0,
                            "prediction_tif_path": surface_prediction_path.name,
                            "ground_truth_tif_path": surface_truth_path.name,
                        },
                        {
                            "suffix": "10m",
                            "label": "10m",
                            "requested_depth_m": 10.0,
                            "actual_depth_m": 9.6,
                            "channel_index": 1,
                            "prediction_tif_path": depth_prediction_path.name,
                            "ground_truth_tif_path": depth_truth_path.name,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def _write_three_week_series(self, root: Path, variable: str) -> list[Path]:
        return [
            self._write_run(
                root,
                variable=variable,
                date_value=20180530,
                iso_year=2018,
                iso_week=22,
                surface_scale=1.0,
                depth_scale=3.0,
                step_index=0,
            ),
            self._write_run(
                root,
                variable=variable,
                date_value=20180606,
                iso_year=2018,
                iso_week=23,
                surface_scale=1.0,
                depth_scale=3.0,
                step_index=1,
            ),
            self._write_run(
                root,
                variable=variable,
                date_value=20180613,
                iso_year=2018,
                iso_week=24,
                surface_scale=1.0,
                depth_scale=3.0,
                step_index=2,
            ),
        ]

    def test_build_temporal_payload_reports_change_and_flicker_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dirs = self._write_three_week_series(root, "temperature")

            payload = build_temporal_analysis_payload(
                run_dirs=run_dirs,
                grid_size_degrees=90.0,
                top_cell_count=2,
            )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["variable"]["name"], "temperature")
        self.assertEqual(payload["run"]["run_count"], 3)
        self.assertEqual(len(payload["depth_levels"]), 3)
        all_depths = payload["depth_levels"][0]
        surface = payload["depth_levels"][1]
        depth_10m = payload["depth_levels"][2]
        self.assertTrue(all_depths["is_aggregate"])
        self.assertEqual(all_depths["depth_count"], 2)

        surface_change_periods = surface["fields"]["change_error"]["periods"]
        depth_change_periods = depth_10m["fields"]["change_error"]["periods"]
        all_change_periods = all_depths["fields"]["change_error"]["periods"]
        all_flicker_periods = all_depths["fields"]["prediction_flicker"]["periods"]

        self.assertEqual(len(surface_change_periods), 2)
        self.assertEqual(surface_change_periods[0]["label"], "2018-05-30 to 2018-06-06")
        self.assertEqual(
            all_flicker_periods[0]["label"],
            "2018-05-30 to 2018-06-06 to 2018-06-13",
        )
        self.assertEqual(surface_change_periods[0]["global"]["median"], 0.0)
        self.assertEqual(surface_change_periods[1]["global"]["median"], 1.0)
        self.assertEqual(depth_change_periods[1]["global"]["median"], 3.0)
        self.assertEqual(all_change_periods[1]["global"]["median"], 2.0)
        self.assertEqual(all_change_periods[1]["global"]["count"], 8)
        self.assertEqual(all_flicker_periods[0]["global"]["median"], 2.0)
        self.assertEqual(len(all_change_periods[1]["grid_cells"]), 4)
        self.assertEqual(len(all_change_periods[1]["top_cells"]["p95"]), 2)
        self.assertIn("basins", all_change_periods[1])

    def test_export_temporal_assets_writes_multivariable_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            temperature_runs = self._write_three_week_series(root, "temperature")
            salinity_runs = self._write_three_week_series(root, "salinity")
            output_dir = root / "temporal"

            result = export_temporal_dashboard_assets(
                variable_run_dirs={
                    "temperature": temperature_runs,
                    "salinity": salinity_runs,
                },
                output_dir=output_dir,
                public_base_url="https://example.com/temporal",
                grid_size_degrees=90.0,
                top_cell_count=2,
                copy_dashboard=False,
            )

            config = json.loads(
                (output_dir / DEFAULT_TEMPORAL_CONFIG_NAME).read_text(encoding="utf-8")
            )
            temperature_payload = json.loads(
                (
                    output_dir / "temperature" / DEFAULT_TEMPORAL_ANALYSIS_JSON_NAME
                ).read_text(encoding="utf-8")
            )
            temperature_grid_exists = (
                output_dir / "temperature" / "analysis-grid.geojson"
            ).is_file()

        self.assertEqual(result["default_variable"], "temperature")
        self.assertEqual(config["available_variables"], ["temperature", "salinity"])
        self.assertEqual(config["default_variable"], "temperature")
        self.assertEqual(
            config["variables"]["temperature"]["temporal_analysis_data_url"],
            "https://example.com/temporal/temperature/temporal-analysis.json",
        )
        self.assertEqual(
            config["variables"]["salinity"]["analysis_grid_geojson_url"],
            "https://example.com/temporal/salinity/analysis-grid.geojson",
        )
        self.assertTrue(temperature_grid_exists)
        self.assertEqual(temperature_payload["run"]["start_date"], 20180530)

    def test_temporal_runner_sequences_weeks_and_passes_export_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_temporal_runner_parser().parse_args(
                [
                    "--start-year",
                    "2018",
                    "--start-iso-week",
                    "52",
                    "--week-count",
                    "3",
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
                    "--temporal-sampler",
                    "ddim",
                    "--temporal-ddim-steps",
                    "50",
                    "--batch-size",
                    "2",
                    "--patch-stride",
                    "64",
                    "--rectangle",
                    "-10",
                    "30",
                    "10",
                    "45",
                    "--no-multi-gpu",
                    "--public-base-url",
                    "https://example.com/temporal",
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

        self.assertEqual(run_mock.call_count, 6)
        first_call_args = run_mock.call_args_list[0].args[0]
        last_call_args = run_mock.call_args_list[-1].args[0]
        self.assertEqual(first_call_args.scenario, "temperature")
        self.assertEqual(first_call_args.year, 2018)
        self.assertEqual(first_call_args.iso_week, 52)
        self.assertEqual(first_call_args.full_sample_count, 0)
        self.assertEqual(first_call_args.sampler, "ddim")
        self.assertEqual(first_call_args.ddim_num_timesteps, 50)
        self.assertEqual(first_call_args.batch_size, 2)
        self.assertEqual(first_call_args.patch_stride, 64)
        self.assertEqual(first_call_args.rectangle, [-10.0, 30.0, 10.0, 45.0])
        self.assertFalse(first_call_args.multi_gpu)
        self.assertIsNone(first_call_args.public_base_url)
        self.assertIsNone(first_call_args.rclone_remote)
        self.assertEqual(last_call_args.scenario, "salinity")
        self.assertEqual(last_call_args.year, 2019)
        self.assertEqual(last_call_args.iso_week, 2)
        export_mock.assert_called_once()
        self.assertEqual(summary["weeks"][-1], {"iso_year": 2019, "iso_week": 2})

    def test_standard_variable_inference_can_export_temporal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            args = _build_standard_variable_parser().parse_args(
                [
                    "--year",
                    "2018",
                    "--iso-week",
                    "52",
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
                    "--public-base-url",
                    "https://example.com/inference_production/globe",
                    "--rclone-remote",
                    "r2:depth-data/inference_production/globe",
                    "--export-temporal-consistency",
                    "--temporal-week-count",
                    "3",
                    "--sampler",
                    "ddim",
                    "--ddim-steps",
                    "200",
                    "--uncertainty-sampler",
                    "ddim",
                    "--uncertainty-ddim-steps",
                    "50",
                    "--temporal-ddim-steps",
                    "25",
                    "--batch-size",
                    "2",
                    "--no-multi-gpu",
                ]
            )

            def fake_run_global_inference(parsed_args):
                run_dir = Path(parsed_args.output_root) / parsed_args.output_name
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "run_summary.yaml"
                summary_path.write_text("{}\n", encoding="utf-8")
                selected_date = (
                    20181226
                    if parsed_args.iso_week == 52
                    else 20190000 + parsed_args.iso_week
                )
                return SimpleNamespace(
                    run_dir=run_dir,
                    summary_path=summary_path,
                    selected_date=selected_date,
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
                ) as globe_mock,
                mock.patch(
                    "depth_recon.inference.export_global_variables.export_temporal_dashboard_assets",
                    return_value={"output_dir": "temporal", "upload_ok": True},
                ) as temporal_mock,
            ):
                summary = run_global_variable_inference(args)

        self.assertEqual(run_mock.call_count, 8)
        self.assertEqual(globe_mock.call_count, 1)
        first_call_args = run_mock.call_args_list[0].args[0]
        extra_call_args = run_mock.call_args_list[2].args[0]
        last_call_args = run_mock.call_args_list[-1].args[0]
        self.assertEqual(first_call_args.scenario, "temperature")
        self.assertEqual(first_call_args.output_name, "temperature")
        self.assertEqual(first_call_args.full_sample_count, 1000)
        self.assertEqual(first_call_args.sampler, "ddim")
        self.assertEqual(first_call_args.ddim_num_timesteps, 200)
        self.assertEqual(first_call_args.uncertainty_sampler, "ddim")
        self.assertEqual(first_call_args.uncertainty_ddim_num_timesteps, 50)
        self.assertEqual(extra_call_args.scenario, "temperature")
        self.assertEqual(extra_call_args.year, 2018)
        self.assertEqual(extra_call_args.iso_week, 52)
        self.assertEqual(extra_call_args.output_name, "2018_W52")
        self.assertEqual(extra_call_args.full_sample_count, 0)
        self.assertEqual(extra_call_args.sampler, "ddim")
        self.assertEqual(extra_call_args.ddim_num_timesteps, 25)
        self.assertIsNone(extra_call_args.public_base_url)
        self.assertIsNone(extra_call_args.rclone_remote)
        self.assertFalse(extra_call_args.multi_gpu)
        self.assertEqual(last_call_args.scenario, "salinity")
        self.assertEqual(last_call_args.year, 2019)
        self.assertEqual(last_call_args.iso_week, 2)

        temporal_kwargs = temporal_mock.call_args.kwargs
        self.assertEqual(
            temporal_kwargs["output_dir"],
            output_root / "standard_temporal" / "temporal",
        )
        self.assertEqual(
            temporal_kwargs["public_base_url"],
            "https://example.com/inference_production/temporal",
        )
        self.assertEqual(
            temporal_kwargs["rclone_remote"],
            "r2:depth-data/inference_production/temporal",
        )
        self.assertEqual(len(temporal_kwargs["variable_run_dirs"]["temperature"]), 3)
        self.assertEqual(len(temporal_kwargs["variable_run_dirs"]["salinity"]), 3)
        self.assertEqual(
            temporal_kwargs["variable_run_dirs"]["temperature"][0],
            output_root
            / "standard_temporal"
            / "temporal_runs"
            / "temperature"
            / "2018_W52",
        )
        self.assertTrue(summary["temporal_consistency"]["enabled"])
        self.assertEqual(
            summary["temporal_consistency"]["weeks"][-1],
            {"iso_year": 2019, "iso_week": 2},
        )
        self.assertEqual(
            summary["temporal_consistency"]["dashboard"],
            {"output_dir": "temporal", "upload_ok": True},
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
                }
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
        self.assertEqual(temporal_args.sampler, "ddim")
        self.assertEqual(temporal_args.ddim_num_timesteps, 50)

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
        self.assertIn('id="temporal-depth-select"', html)
        self.assertIn('id="temporal-depth-error"', html)
        self.assertIn("Mean absolute error across the validation year", html)
        self.assertIn("temporal-dashboard.js", html)
        self.assertIn("standalone-temporal-page", css)
        self.assertIn("#061726", css)
        self.assertIn("#7cc8ff", css)
        self.assertIn("rgba(5, 20, 32, 0.78)", css)
        self.assertIn("background: #071d2d;", css)
        self.assertIn("DEFAULT_TEMPORAL_CONFIG_URL", script)
        self.assertIn("dark_nolabels", script)
        self.assertIn('params.get("config")', script)
        self.assertIn("schema_version", script)
        self.assertIn("basin_data_urls", script)
        self.assertIn("function loadActiveBasinData", script)
        self.assertIn("function populateDepthSelect", script)
        self.assertIn("function selectedDepthErrors", script)
        self.assertIn("function renderDepthErrorGraph", script)
        self.assertIn('orientation: "h"', script)
        self.assertIn("Mean absolute error: %{x:.3f}", script)
        self.assertIn('window.location.href = "../analysis/"', script)
        self.assertNotIn("temporal-analysis.json", script)
        self.assertIn("Analysis: /visualizations/", mkdocs_config)
        self.assertNotIn(
            "Temporal Dashboard: https://depthdif.donike.net/temporal/",
            mkdocs_config,
        )

    def test_temporal_globe_page_links_back_to_analysis_only(self) -> None:
        html = Path("docs/temporal-globe/index.html").read_text(encoding="utf-8")

        self.assertIn("Temporal Globe", html)
        self.assertIn(
            'href="../visualizations/">Back to Analysis</a>',
            html,
        )
        self.assertNotIn('href="../globe/">Globe</a>', html)
        self.assertNotIn('href="../temporal/">Temporal Dashboard</a>', html)


if __name__ == "__main__":
    unittest.main()
