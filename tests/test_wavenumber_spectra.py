from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import yaml

from depth_recon.inference.export_wavenumber_spectra import (
    ALL_OCEANS_BASIN,
    assign_patch_basin_by_overlap,
    decode_stretched_uint8,
    detrend_plane_2d,
    discover_variable_runs,
    export_wavenumber_spectra,
    radial_wavenumber_spectrum,
    read_patch_window,
)


class TestWavenumberSpectra(unittest.TestCase):
    def _write_float_raster(
        self,
        path: Path,
        data: np.ndarray,
        *,
        transform: rasterio.Affine,
        nodata: float = -9999.0,
    ) -> None:
        """Write a single-band float GeoTIFF fixture."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
        ) as ds:
            ds.write(data.astype(np.float32).reshape(1, data.shape[0], data.shape[1]))

    def _write_uint8_source_raster(
        self,
        path: Path,
        data: np.ndarray,
        *,
        transform: rasterio.Affine,
    ) -> None:
        """Write a stretched uint8 GeoTIFF fixture with decoding tags."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
            nodata=255,
        ) as ds:
            ds.write(data.astype(np.uint8).reshape(1, data.shape[0], data.shape[1]))
            ds.update_tags(
                stretch_min="270.15",
                stretch_max="310.15",
                valid_code_max="254",
                nodata="255",
                stretch_units="K",
                source_product="ostia",
            )

    def test_detrend_plane_2d_removes_linear_plane(self) -> None:
        rows, cols = np.indices((9, 11), dtype=np.float64)
        values = 3.0 * cols - 1.5 * rows + 7.0

        detrended = detrend_plane_2d(values)

        self.assertLess(float(np.nanmax(np.abs(detrended))), 1.0e-10)

    def test_radial_spectrum_recovers_hann_windowed_fft_peak(self) -> None:
        x = np.arange(64, dtype=np.float64)
        values = np.sin(2.0 * np.pi * x[None, :] / 8.0)
        values = np.repeat(values, 64, axis=0)

        result = radial_wavenumber_spectrum(
            values,
            pixel_size_x_km=1.0,
            pixel_size_y_km=1.0,
            wavelength_edges_km=np.asarray([4.0, 6.0, 10.0, 16.0]),
        )

        self.assertIsNotNone(result)
        spectrum, counts = result
        self.assertEqual(int(np.nanargmax(spectrum)), 1)
        self.assertGreater(int(counts[1]), 0)

    def test_radial_spectrum_skips_incomplete_patch_when_required(self) -> None:
        values = np.ones((8, 8), dtype=np.float64)
        values[2, 3] = np.nan

        result = radial_wavenumber_spectrum(
            values,
            pixel_size_x_km=1.0,
            pixel_size_y_km=1.0,
            wavelength_edges_km=np.asarray([2.0, 4.0, 8.0]),
            require_complete=True,
        )

        self.assertIsNone(result)

    def test_radial_spectrum_allows_incomplete_patch_with_zero_filled_residuals(
        self,
    ) -> None:
        x = np.arange(8, dtype=np.float64)
        values = np.sin(2.0 * np.pi * x[None, :] / 4.0)
        values = np.repeat(values, 8, axis=0)
        values[2, 3] = np.nan

        result = radial_wavenumber_spectrum(
            values,
            pixel_size_x_km=1.0,
            pixel_size_y_km=1.0,
            wavelength_edges_km=np.asarray([2.0, 4.0, 8.0]),
            require_complete=False,
        )

        self.assertIsNotNone(result)
        spectrum, counts = result
        self.assertTrue(np.any(np.isfinite(spectrum)))
        self.assertGreater(int(np.sum(counts)), 0)

    def test_read_patch_window_marks_raster_nodata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "nodata.tif"
            transform = from_origin(-160.0, 30.0, 1.0, 1.0)
            data = np.asarray([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
            self._write_float_raster(raster_path, data, transform=transform)
            row = {"lon0": -160.0, "lon1": -158.0, "lat0": 28.0, "lat1": 30.0}

            patch = read_patch_window(raster_path, row, variable="temperature")

            self.assertIsNotNone(patch)
            self.assertTrue(np.isnan(patch[0, 1]))

    def test_decode_stretched_uint8_converts_kelvin_to_celsius(self) -> None:
        values = np.asarray([[0, 254, 255]], dtype=np.uint8)
        tags = {
            "stretch_min": "270.15",
            "stretch_max": "308.15",
            "valid_code_max": "254",
            "nodata": "255",
            "stretch_units": "K",
        }

        decoded = decode_stretched_uint8(
            values,
            tags,
            variable="temperature",
            nodata=255,
        )

        self.assertAlmostEqual(float(decoded[0, 0]), -3.0)
        self.assertAlmostEqual(float(decoded[0, 1]), 35.0)
        self.assertTrue(np.isnan(decoded[0, 2]))

    def test_assign_patch_basin_by_overlap_uses_threshold(self) -> None:
        row = {"lon0": -160.0, "lon1": -150.0, "lat0": 20.0, "lat1": 30.0}

        self.assertEqual(assign_patch_basin_by_overlap(row), "North Pacific Ocean")
        self.assertIsNone(assign_patch_basin_by_overlap(row, threshold=1.01))

    def test_discover_variable_runs_expands_paired_and_temporal_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "paired"
            temperature_run = root / "temperature"
            salinity_run = root / "salinity"
            temporal_run = root / "temporal_runs" / "temperature" / "2018_W02"
            for run_dir, variable in (
                (temperature_run, "temperature"),
                (salinity_run, "salinity"),
                (temporal_run, "temperature"),
            ):
                run_dir.mkdir(parents=True)
                (run_dir / "run_summary.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "variable": variable,
                            "selected_date": 20180620,
                            "iso_year": 2018,
                            "iso_week": 25,
                            "depth_exports": [],
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variables": {
                            "temperature": {"run_dir": "temperature"},
                            "salinity": {"summary_path": "salinity/run_summary.yaml"},
                        },
                        "temporal_consistency": {
                            "variable_run_dirs": {
                                "temperature": ["temporal_runs/temperature/2018_W02"]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            paired_runs = discover_variable_runs(
                [root],
                variables=["temperature"],
                include_temporal_runs=True,
            )
            direct_runs = discover_variable_runs(
                [temperature_run],
                variables=["temperature"],
                include_temporal_runs=True,
            )

            self.assertEqual(
                [run.variable for run in paired_runs], ["temperature", "temperature"]
            )
            self.assertEqual(
                {run.run_dir for run in paired_runs}, {temperature_run, temporal_run}
            )
            self.assertEqual(
                {run.run_dir for run in direct_runs}, {temperature_run, temporal_run}
            )

    def test_export_wavenumber_spectra_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "temperature"
            dataset_root = root / "dataset"
            run_dir.mkdir()
            (dataset_root / "masks").mkdir(parents=True)
            source_dir = dataset_root / "rasters" / "ostia" / "analysed_sst"
            source_dir.mkdir(parents=True)

            transform = from_origin(-160.0, 30.0, 0.1, 0.1)
            y, x = np.indices((4, 8), dtype=np.float64)
            data = np.sin(2.0 * np.pi * x / 4.0) + 0.5 * np.cos(2.0 * np.pi * y / 4.0)
            prediction_path = run_dir / "temperature_prediction_surface.tif"
            glorys_path = run_dir / "temperature_glorys_surface.tif"
            source_path = source_dir / "analysed_sst_20180620.tif"
            self._write_float_raster(prediction_path, data, transform=transform)
            self._write_float_raster(glorys_path, data + 0.25, transform=transform)
            self._write_uint8_source_raster(
                source_path,
                np.full((4, 8), 120, dtype=np.uint8),
                transform=transform,
            )

            pd.DataFrame.from_records(
                [
                    {
                        "patch_id": "a",
                        "grid_y0": 0,
                        "grid_x0": 0,
                        "lon0": -160.0,
                        "lon1": -159.6,
                        "lat0": 29.6,
                        "lat1": 30.0,
                    },
                    {
                        "patch_id": "b",
                        "grid_y0": 0,
                        "grid_x0": 4,
                        "lon0": -159.6,
                        "lon1": -159.2,
                        "lat0": 29.6,
                        "lat1": 30.0,
                    },
                ]
            ).to_csv(run_dir / "selected_patches.csv", index=False)
            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variable": "temperature",
                        "selected_date": 20180620,
                        "iso_year": 2018,
                        "iso_week": 25,
                        "land_mask_path": str(
                            dataset_root / "masks" / "world_land_mask_glorys_0p1.tif"
                        ),
                        "depth_exports": [
                            {
                                "suffix": "surface",
                                "label": "Surface",
                                "requested_depth_m": 0.0,
                                "actual_depth_m": 0.0,
                                "channel_index": 0,
                                "prediction_tif_path": prediction_path.name,
                                "ground_truth_tif_path": glorys_path.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_dir = root / "spectra"
            summary = export_wavenumber_spectra(
                run_dirs=[run_dir],
                output_dir=output_dir,
                variables=["temperature"],
                min_wavelength_km=20.0,
                max_wavelength_km=200.0,
                wavelength_bin_count=4,
            )

            records = pd.read_csv(output_dir / "patch_spectra_records.csv")
            aggregated = pd.read_csv(output_dir / "aggregated_spectra.csv")
            saved_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["spectrum_count"], 6)
            self.assertEqual(saved_summary["spectrum_count"], 6)
            self.assertTrue((output_dir / "patch_spectra.npz").is_file())
            self.assertTrue((output_dir / "spectral-config.json").is_file())
            self.assertTrue((output_dir / "basin-map.geojson").is_file())
            self.assertTrue((output_dir / "basins" / "all_oceans.json").is_file())
            self.assertTrue(
                (output_dir / "basins" / "north_pacific_ocean.json").is_file()
            )
            self.assertTrue((output_dir / "index.html").is_file())
            self.assertTrue(
                (output_dir / "javascripts" / "spectral-dashboard.js").is_file()
            )
            self.assertTrue(
                (output_dir / "stylesheets" / "spectral-dashboard.css").is_file()
            )
            self.assertEqual(
                set(records["layer"].tolist()),
                {"prediction", "glorys", "surface_observation"},
            )
            self.assertIn(ALL_OCEANS_BASIN, set(aggregated["basin"].tolist()))
            self.assertIn("North Pacific Ocean", set(aggregated["basin"].tolist()))
            self.assertEqual(int(aggregated["spectrum_count"].max()), 2)
            self.assertIn("psd_mean", aggregated.columns)
            finite_psd = aggregated[aggregated["psd_mean"].notna()].iloc[0]
            self.assertGreater(float(finite_psd["wavenumber_bin_width_cpkm"]), 0.0)
            self.assertAlmostEqual(
                float(finite_psd["psd_mean"]),
                float(finite_psd["power_mean"])
                / float(finite_psd["wavenumber_bin_width_cpkm"]),
            )
            self.assertTrue(any((output_dir / "plots").glob("*.png")))
            config = json.loads(
                (output_dir / "spectral-config.json").read_text(encoding="utf-8")
            )
            all_oceans_payload = json.loads(
                (output_dir / "basins" / "all_oceans.json").read_text(encoding="utf-8")
            )
            copied_html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertEqual(config["kind"], "wavenumber_spectral_dashboard")
            self.assertEqual(config["available_variables"], ["temperature"])
            self.assertEqual(config["layers"]["surface_observation"], "OSTIA")
            self.assertIn("horizontal_wavenumber_bin_centers_cpkm", config)
            self.assertIn(ALL_OCEANS_BASIN, config["basin_data_urls"])
            self.assertIn("Mediterranean Sea", config["basin_data_urls"])
            mediterranean_payload = json.loads(
                (output_dir / "basins" / "mediterranean_sea.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(mediterranean_payload["rows"], [])
            self.assertGreater(len(all_oceans_payload["rows"]), 0)
            self.assertIn("stylesheets/spectral-dashboard.css", copied_html)
            self.assertIn("javascripts/spectral-dashboard.js", copied_html)

    def test_export_wavenumber_spectra_can_skip_dashboard_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "temperature"
            run_dir.mkdir()
            pd.DataFrame(columns=["lon0", "lon1", "lat0", "lat1"]).to_csv(
                run_dir / "selected_patches.csv",
                index=False,
            )
            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variable": "temperature",
                        "selected_date": 20180620,
                        "iso_year": 2018,
                        "iso_week": 25,
                        "depth_exports": [],
                    }
                ),
                encoding="utf-8",
            )

            output_dir = Path(tmp_dir) / "spectra"
            summary = export_wavenumber_spectra(
                run_dirs=[run_dir],
                output_dir=output_dir,
                variables=["temperature"],
                write_dashboard=False,
            )

            self.assertFalse(summary["dashboard_enabled"])
            self.assertFalse((output_dir / "spectral-config.json").exists())
            self.assertFalse((output_dir / "basin-map.geojson").exists())
            self.assertFalse((output_dir / "index.html").exists())

    def test_export_wavenumber_spectra_uploads_bundle_when_remote_configured(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "temperature"
            run_dir.mkdir()
            pd.DataFrame(columns=["lon0", "lon1", "lat0", "lat1"]).to_csv(
                run_dir / "selected_patches.csv",
                index=False,
            )
            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variable": "temperature",
                        "selected_date": 20180620,
                        "iso_year": 2018,
                        "iso_week": 25,
                        "depth_exports": [],
                    }
                ),
                encoding="utf-8",
            )

            output_dir = Path(tmp_dir) / "spectra"
            with mock.patch(
                "depth_recon.inference.export_wavenumber_spectra._sync_with_rclone",
                return_value=(True, "uploaded"),
            ) as sync_mock:
                summary = export_wavenumber_spectra(
                    run_dirs=[run_dir],
                    output_dir=output_dir,
                    variables=["temperature"],
                    write_dashboard=False,
                    public_base_url="https://example.test/globe/wavenumber_spectra",
                    rclone_remote="r2:bucket/globe/wavenumber_spectra",
                )

            sync_mock.assert_called_once_with(
                output_dir, "r2:bucket/globe/wavenumber_spectra"
            )
            self.assertTrue(summary["upload_requested"])
            self.assertTrue(summary["upload_ok"])
            self.assertEqual(summary["upload_message"], "uploaded")
            self.assertEqual(
                summary["public_base_url"],
                "https://example.test/globe/wavenumber_spectra",
            )

    def test_spectral_dashboard_static_page_exposes_expected_controls(self) -> None:
        html = Path("docs/spectral-dashboard/index.html").read_text(encoding="utf-8")

        for expected_id in (
            "spectral-dashboard-select",
            "spectral-variable-select",
            "spectral-basin-select",
            "spectral-period-type-select",
            "spectral-period-label-select",
            "spectral-depth-select",
            "spectral-metric-select",
            "spectral-x-axis-select",
            "spectral-map",
            "spectral-spectrum-chart",
            "spectral-bias-chart",
            "spectral-summary-cards",
        ):
            self.assertIn(f'id="{expected_id}"', html)

    def test_spectral_axes_are_log_increasing_and_switchable(self) -> None:
        script = Path("docs/javascripts/spectral-dashboard.js").read_text(
            encoding="utf-8"
        )
        exporter = Path(
            "src/depth_recon/inference/export_wavenumber_spectra.py"
        ).read_text(encoding="utf-8")

        self.assertIn('activeXAxisUnit: "cpkm"', script)
        self.assertIn("xAxisValue", script)
        self.assertIn("Wavelength [km]", script)
        self.assertIn("OSTIA", script)
        self.assertIn("surface_observation", script)
        self.assertNotIn('autorange: "reversed"', script)
        self.assertNotIn("ax.invert_xaxis()", exporter)

    def test_analysis_landing_and_dashboard_switchers_link_spectral_dashboard(
        self,
    ) -> None:
        landing = Path("docs/analysis/index.html").read_text(encoding="utf-8")
        landing_css = Path("docs/stylesheets/analysis-landing.css").read_text(
            encoding="utf-8"
        )
        analysis_html = Path("docs/spatial-dashboard/index.html").read_text(
            encoding="utf-8"
        )
        temporal_html = Path("docs/temporal-dashboard/index.html").read_text(
            encoding="utf-8"
        )
        analysis_js = Path("docs/javascripts/analysis-dashboard.js").read_text(
            encoding="utf-8"
        )
        temporal_js = Path("docs/javascripts/temporal-dashboard.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('href="../spectral-dashboard/"', landing)
        self.assertIn("spectral_dashboard_tile.webp", landing)
        self.assertIn("analysis-landing-tile--spectral", landing_css)
        self.assertIn('value="spectral"', analysis_html)
        self.assertIn('value="spectral"', temporal_html)
        self.assertIn("../spectral-dashboard/", analysis_js)
        self.assertIn("../spectral-dashboard/", temporal_js)
        self.assertTrue(
            Path("docs/assets/tiles/spectral_dashboard_tile.webp").is_file()
        )


if __name__ == "__main__":
    unittest.main()
