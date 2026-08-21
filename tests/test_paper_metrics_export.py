from __future__ import annotations

import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.inference.export_paper_metrics import (
    _metric_stats,
    export_paper_metrics,
    idw_fill_2d,
    load_dataset_context,
    load_method_runs,
    prediction_specs_by_method,
    select_en4_holdout_locations,
    summarize_equal_depth_metrics,
)
from depth_recon.utils.en4_candidate_validation import (
    CandidateProfileResult,
    EN4CandidateValidationCallback,
    _metric_summary,
    _profile_figure,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from tests.test_argo_geotiff_gridded_dataset import _make_geotiff_dataset


class TestPaperMetricsExport(unittest.TestCase):
    def _write_prediction_raster(self, path: Path, data: np.ndarray) -> None:
        """Write a tiny one-band prediction raster."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(10.0, 2.0, 1.0, 1.0),
            nodata=-9999.0,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def _write_multiband_raster(self, path: Path, bands: list[np.ndarray]) -> None:
        """Write a tiny multiband raster matching the fake dataset grid."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=int(bands[0].shape[0]),
            width=int(bands[0].shape[1]),
            count=len(bands),
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(10.0, 2.0, 1.0, 1.0),
            nodata=-9999.0,
        ) as dst:
            for band_index, band in enumerate(bands, start=1):
                dst.write(np.asarray(band, dtype=np.float32), band_index)

    def _write_climatology_artifacts(self, root: Path) -> Path:
        """Write minimal climatology artifacts resolvable by the metrics exporter."""
        references = root / "references"
        temp_path = references / "climatology_temperature.tif"
        sal_path = references / "climatology_salinity.tif"
        self._write_multiband_raster(
            temp_path,
            [
                np.full((2, 2), 11.0, dtype=np.float32),
                np.full((2, 2), 21.0, dtype=np.float32),
            ],
        )
        self._write_multiband_raster(
            sal_path,
            [
                np.full((2, 2), 35.1, dtype=np.float32),
                np.full((2, 2), 36.1, dtype=np.float32),
            ],
        )
        summary_path = references / "climatology_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "en4_climatology_idw",
                    "artifacts": {
                        "temperature": temp_path.name,
                        "salinity": sal_path.name,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return summary_path

    def _write_paper_bundle_manifest(
        self,
        root: Path,
        *,
        geotiff_root: Path,
        methods: list[str],
    ) -> Path:
        """Write a fake paper-week manifest with dynamic methods and disk refs."""
        references = root / "references"
        references.mkdir(parents=True, exist_ok=True)
        climatology_summary = self._write_climatology_artifacts(root)
        holdout_locations = references / "en4_holdout_locations.csv"
        pd.DataFrame.from_records(
            [
                {
                    "date": 20240108,
                    "grid_row": 0,
                    "grid_col": 0,
                    "lon": 10.5,
                    "lat": 1.5,
                    "profile_index": 0,
                    "temperature_valid_depth_count": 2,
                    "salinity_valid_depth_count": 2,
                    "holdout_fraction": 0.2,
                    "split_seed": 7,
                }
            ]
        ).to_csv(holdout_locations, index=False)
        profile_rows = []
        for variable, values in {
            "temperature": [10.0, 20.0],
            "salinity": [35.0, 36.0],
        }.items():
            for channel_index, value in enumerate(values):
                profile_rows.append(
                    {
                        "date": 20240108,
                        "grid_row": 0,
                        "grid_col": 0,
                        "profile_index": 0,
                        "variable": variable,
                        "channel_index": channel_index,
                        "depth_m": 0.0 if channel_index == 0 else 10.0,
                        "value": value,
                    }
                )
        holdout_profiles = references / "en4_holdout_profiles.csv"
        pd.DataFrame.from_records(profile_rows).to_csv(holdout_profiles, index=False)

        glorys_refs: dict[str, dict[str, list[dict[str, object]]]] = {}
        for variable, base_values in {
            "temperature": [10.0, 20.0],
            "salinity": [35.0, 36.0],
        }.items():
            exports = []
            for channel_index, base in enumerate(base_values):
                suffix = f"depth_{channel_index:03d}"
                tif_path = references / "glorys" / f"{variable}_{suffix}.tif"
                self._write_prediction_raster(
                    tif_path, np.full((2, 2), base, dtype=np.float32)
                )
                exports.append(
                    {
                        "suffix": suffix,
                        "label": suffix,
                        "requested_depth_m": 0.0 if channel_index == 0 else 10.0,
                        "actual_depth_m": 0.0 if channel_index == 0 else 10.0,
                        "channel_index": channel_index,
                        "path": str(tif_path.relative_to(root)),
                        "band_index": 1,
                    }
                )
            glorys_refs[variable] = {"depth_exports": exports}

        manifest = {
            "schema_version": 1,
            "kind": "paper_week_inference_bundle",
            "year": 2024,
            "iso_week": 2,
            "selected_date": 20240108,
            "validation_year": 2018,
            "en4_holdout_fraction": 0.2,
            "seed": 7,
            "dataset_root": str(geotiff_root),
            "variables": ["temperature", "salinity"],
            "depth_export_mode": "native",
            "method_order": methods,
            "methods": {
                method: {
                    "kind": "model",
                    "label": {"idw": "IDW", "cnn": "CNN", "depthdif": "DepthDif"}[
                        method
                    ],
                    "model_type": {
                        "idw": "idw_baseline",
                        "cnn": "cnn_baseline",
                        "depthdif": "cond_px_dif",
                    }[method],
                    "run_dir": str((Path("methods") / method)),
                }
                for method in methods
            },
            "references": {
                "en4_holdout_locations_csv": str(holdout_locations.relative_to(root)),
                "en4_holdout_profiles_csv": str(holdout_profiles.relative_to(root)),
                "climatology_summary_json": str(climatology_summary.relative_to(root)),
                "glorys": glorys_refs,
            },
        }
        manifest_path = root / "paper_week_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest_path

    def _write_method_run(
        self,
        root: Path,
        *,
        method: str,
        geotiff_root: Path,
        offset: float,
    ) -> Path:
        """Write a paired fake run dir with all native depth predictions."""
        method_root = root / method
        data_config = method_root / "data.yaml"
        data_config.parent.mkdir(parents=True, exist_ok=True)
        data_config.write_text(
            yaml.safe_dump(
                {"dataset": {"core": {"geotiff_root_dir": str(geotiff_root)}}}
            ),
            encoding="utf-8",
        )
        for variable, base_values in {
            "temperature": [12.0, 22.0],
            "salinity": [35.2, 36.2],
        }.items():
            run_dir = method_root / variable
            run_dir.mkdir(parents=True, exist_ok=True)
            depth_exports = []
            for channel_index, base in enumerate(base_values):
                suffix = f"depth_{channel_index:03d}"
                tif_path = run_dir / f"{variable}_{suffix}.tif"
                self._write_prediction_raster(
                    tif_path,
                    np.full((2, 2), float(base) + float(offset), dtype=np.float32),
                )
                depth_exports.append(
                    {
                        "suffix": suffix,
                        "label": "Surface" if channel_index == 0 else "10m",
                        "requested_depth_m": 0.0 if channel_index == 0 else 10.0,
                        "actual_depth_m": 0.0 if channel_index == 0 else 10.0,
                        "channel_index": channel_index,
                        "prediction_tif_path": tif_path.name,
                        "ground_truth_tif_path": None,
                    }
                )
            (run_dir / "selected_patches.csv").write_text(
                "patch_id,grid_y0,grid_x0,lon0,lon1,lat0,lat1\n0,0,0,10,12,0,2\n",
                encoding="utf-8",
            )
            (run_dir / "run_summary.yaml").write_text(
                yaml.safe_dump(
                    {
                        "variable": variable,
                        "selected_date": 20240108,
                        "iso_year": 2024,
                        "iso_week": 2,
                        "data_config": str(data_config),
                        "depth_exports": depth_exports,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        (method_root / "run_summary.yaml").write_text(
            yaml.safe_dump(
                {
                    "variables": {
                        "temperature": {"run_dir": "temperature"},
                        "salinity": {"run_dir": "salinity"},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return method_root

    def test_dataset_holdout_locations_remove_sparse_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                include_salinity=True,
            )

            unmasked = dataset[0]
            self.assertTrue(bool(unmasked["x_valid_mask"][:, 0, 0].all().item()))

            dataset.set_heldout_argo_locations([(20240108, 0, 0)])
            masked = dataset[0]

            self.assertFalse(bool(masked["x_valid_mask"][:, 0, 0].any().item()))
            self.assertFalse(
                bool(masked["x_salinity_valid_mask"][:, 0, 0].any().item())
            )
            self.assertTrue(torch.equal(masked["x"], torch.zeros_like(masked["x"])))

    def test_candidate_holdout_matches_exact_provenance_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir, _cache_dir, _land_mask_path = _make_geotiff_dataset(root)
            candidate_path = root / "candidate_profiles.parquet"
            pd.DataFrame.from_records(
                [
                    {
                        "profile_source_file": "EN.4.2.2.f.profiles.g10.202401.nc",
                        "source_profile_idx": 7,
                    },
                    {
                        "profile_source_file": "EN.4.2.2.f.profiles.g10.202401.nc",
                        "source_profile_idx": 7,
                    },
                    {
                        "profile_source_file": "EN.4.2.2.f.profiles.g10.202401.nc",
                        "source_profile_idx": 999,
                    },
                ]
            ).to_parquet(candidate_path, index=False)

            holdout = select_en4_holdout_locations(
                context=load_dataset_context(output_dir),
                date_value=20240108,
                fraction=0.2,
                seed=7,
                candidate_profiles_path=candidate_path,
            )

            self.assertEqual(len(holdout), 1)
            self.assertEqual(int(holdout.iloc[0]["source_profile_idx"]), 7)
            self.assertEqual(
                holdout.iloc[0]["profile_source_file"],
                "EN.4.2.2.f.profiles.g10.202401.nc",
            )
            self.assertEqual(
                holdout.iloc[0]["candidate_evidence"],
                "no_nearby_glorys_source_candidate",
            )
            self.assertEqual(holdout.attrs["eligible_profile_count"], 1)
            self.assertEqual(holdout.attrs["eligible_location_count"], 1)
            self.assertEqual(holdout.attrs["selected_location_count"], 1)

    def test_candidate_holdout_rejects_malformed_or_unmatched_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir, _cache_dir, _land_mask_path = _make_geotiff_dataset(root)
            context = load_dataset_context(output_dir)
            malformed_path = root / "malformed.parquet"
            pd.DataFrame(
                {"profile_source_file": ["EN.4.2.2.f.profiles.g10.202401.nc"]}
            ).to_parquet(malformed_path, index=False)
            with self.assertRaisesRegex(ValueError, "source_profile_idx"):
                select_en4_holdout_locations(
                    context=context,
                    date_value=20240108,
                    fraction=0.2,
                    seed=7,
                    candidate_profiles_path=malformed_path,
                )

            unmatched_path = root / "unmatched.parquet"
            pd.DataFrame(
                {
                    "profile_source_file": ["EN.4.2.2.f.profiles.g10.202401.nc"],
                    "source_profile_idx": [999],
                }
            ).to_parquet(unmatched_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "matched"):
                select_en4_holdout_locations(
                    context=context,
                    date_value=20240108,
                    fraction=0.2,
                    seed=7,
                    candidate_profiles_path=unmatched_path,
                )

    def test_validation_callback_compares_prediction_and_glorys_to_en4(self) -> None:
        """Epoch monitoring must score both curves against the exact EN4 profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(root)
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="all",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2016,
                require_argo_for_all=False,
                include_salinity=True,
            )
            candidate_path = root / "candidate_profiles.parquet"
            pd.DataFrame(
                {
                    "profile_source_file": ["EN.4.2.2.f.profiles.g10.202401.nc"],
                    "source_profile_idx": [7],
                }
            ).to_parquet(candidate_path, index=False)
            holdout = select_en4_holdout_locations(
                context=load_dataset_context(output_dir),
                date_value=20240108,
                fraction=0.2,
                seed=7,
                candidate_profiles_path=candidate_path,
                profile_store=dataset.argo_store,
            )
            dataset.set_heldout_argo_locations(
                [
                    (
                        20240108,
                        int(holdout.iloc[0].grid_row),
                        int(holdout.iloc[0].grid_col),
                    )
                ]
            )
            callback = EN4CandidateValidationCallback(
                dataset=dataset,
                holdout_df=holdout,
                max_patches=1,
                max_profiles_to_plot=1,
                random_seed=7,
            )

            class ExactGlorysModel(torch.nn.Module):
                output_fields = ("temperature", "salinity")
                device = torch.device("cpu")

                def transfer_batch_to_device(self, batch, device, dataloader_idx):
                    return batch

                def predict_step(self, batch, batch_idx):
                    return {
                        "y_hat_temperature_denorm": temperature_normalize(
                            mode="denorm", tensor=batch["y"]
                        ),
                        "y_hat_salinity_denorm": salinity_normalize(
                            mode="denorm", tensor=batch["y_salinity"]
                        ),
                    }

            results = callback.evaluate(ExactGlorysModel())
            temperature_summary = _metric_summary(results["temperature"])

            self.assertEqual(len(results["temperature"]), 1)
            self.assertEqual(len(results["salinity"]), 1)
            self.assertAlmostEqual(
                temperature_summary["prediction_rmse"],
                temperature_summary["glorys_rmse"],
            )
            self.assertAlmostEqual(temperature_summary["skill_vs_glorys"], 0.0)
            masked_batch = callback._build_batch()
            self.assertFalse(bool(masked_batch["x_valid_mask"].any()))

            synthetic_batch = dict(masked_batch)
            synthetic_batch["y_glorys"] = masked_batch["y"] + 0.25
            synthetic_batch["y_salinity_glorys"] = masked_batch["y_salinity"] + 0.25
            with patch.object(callback, "_build_batch", return_value=synthetic_batch):
                synthetic_results = callback.evaluate(ExactGlorysModel())
            assignment = callback.profile_assignments.iloc[0]
            batch_idx = int(assignment["eval_batch_index"])
            row_idx = int(assignment["local_grid_row"])
            col_idx = int(assignment["local_grid_col"])
            expected_glorys = temperature_normalize(
                mode="denorm", tensor=synthetic_batch["y_glorys"]
            )[batch_idx, :, row_idx, col_idx]
            expected_salinity_glorys = salinity_normalize(
                mode="denorm", tensor=synthetic_batch["y_salinity_glorys"]
            )[batch_idx, :, row_idx, col_idx]
            np.testing.assert_allclose(
                synthetic_results["temperature"][0].glorys,
                expected_glorys.numpy(),
            )
            np.testing.assert_allclose(
                synthetic_results["salinity"][0].glorys,
                expected_salinity_glorys.numpy(),
            )

            logged_payloads: list[dict[str, object]] = []
            trainer = SimpleNamespace(
                sanity_checking=False,
                is_global_zero=True,
                global_step=1,
                logger=SimpleNamespace(
                    experiment=SimpleNamespace(log=logged_payloads.append)
                ),
            )
            fake_wandb = SimpleNamespace(Image=lambda figure: figure)
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                callback.on_validation_epoch_end(trainer, ExactGlorysModel())

            self.assertEqual(len(logged_payloads), 1)
            logged = logged_payloads[0]
            self.assertIn("en4_candidate_eval/temperature_prediction_rmse", logged)
            self.assertIn("en4_candidate_eval/temperature_glorys_rmse", logged)
            self.assertIn("en4_candidate_eval/temperature_profiles", logged)
            self.assertIn("en4_candidate_eval/salinity_profiles", logged)

    def test_candidate_metric_summary_reports_positive_skill(self) -> None:
        """Candidate skill is positive only when prediction improves on GLORYS."""
        result = CandidateProfileResult(
            variable="temperature",
            date=20160624,
            latitude=0.0,
            longitude=0.0,
            profile_source_file="profiles.nc",
            source_profile_idx=1,
            prediction=np.asarray([1.0, 2.0]),
            glorys=np.asarray([3.0, 4.0]),
            en4=np.asarray([1.0, 1.0]),
        )
        summary = _metric_summary([result])

        self.assertGreater(summary["skill_vs_glorys"], 0.0)
        self.assertLess(summary["prediction_rmse"], summary["glorys_rmse"])

    def test_candidate_profile_figure_stops_at_deepest_en4_value(self) -> None:
        result = CandidateProfileResult(
            variable="temperature",
            date=20160624,
            latitude=0.0,
            longitude=0.0,
            profile_source_file="profiles.nc",
            source_profile_idx=1,
            prediction=np.asarray([1.0, 2.0, 3.0]),
            glorys=np.asarray([1.5, 2.5, 3.5]),
            en4=np.asarray([1.0, np.nan, 3.0]),
        )

        figure = _profile_figure(
            [result], depth_axis_m=np.asarray([0.0, 10.0, 20.0]), max_profiles=1
        )
        try:
            for axis in figure.axes:
                self.assertEqual(axis.get_ylim(), (20.0, 0.0))
        finally:
            plt.close(figure)

    def test_metrics_math_and_equal_depth_average(self) -> None:
        stats = _metric_stats(
            np.asarray([1.0, 3.0], dtype=np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.mae, 1.0)
        self.assertAlmostEqual(stats.rmse, np.sqrt(2.0))
        self.assertIsNone(stats.r2)

        by_depth = pd.DataFrame.from_records(
            [
                {
                    "method": "idw",
                    "method_label": "IDW",
                    "target": "en4",
                    "target_label": "EN4 Validation Set",
                    "variable": "temperature",
                    "channel_index": 0,
                    "count": 3,
                    "rmse": 1.0,
                    "mae": 0.5,
                    "r2": 0.1,
                },
                {
                    "method": "idw",
                    "method_label": "IDW",
                    "target": "en4",
                    "target_label": "EN4 Validation Set",
                    "variable": "temperature",
                    "channel_index": 1,
                    "count": 5,
                    "rmse": 3.0,
                    "mae": 1.5,
                    "r2": 0.3,
                },
            ]
        )

        summary = summarize_equal_depth_metrics(by_depth)

        self.assertEqual(int(summary.iloc[0]["count"]), 8)
        self.assertAlmostEqual(float(summary.iloc[0]["rmse"]), 2.0)
        self.assertAlmostEqual(float(summary.iloc[0]["mae"]), 1.0)
        self.assertAlmostEqual(float(summary.iloc[0]["r2"]), 0.2)

    def test_idw_fill_2d_preserves_observations_and_fills_mask(self) -> None:
        values = np.asarray([[1.0, np.nan], [np.nan, 3.0]], dtype=np.float32)
        filled = idw_fill_2d(
            values,
            target_mask=np.ones((2, 2), dtype=bool),
            neighbors=2,
            chunk_size=2,
        )

        self.assertAlmostEqual(float(filled[0, 0]), 1.0)
        self.assertAlmostEqual(float(filled[1, 1]), 3.0)
        self.assertTrue(np.all(np.isfinite(filled)))

    def test_export_paper_metrics_writes_table_and_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geotiff_root, _cache_dir, _land_mask_path = _make_geotiff_dataset(root)
            idw_run = self._write_method_run(
                root, method="idw", geotiff_root=geotiff_root, offset=0.0
            )
            lstm_run = self._write_method_run(
                root, method="lstm", geotiff_root=geotiff_root, offset=0.5
            )
            unet_run = self._write_method_run(
                root, method="unet", geotiff_root=geotiff_root, offset=1.0
            )
            output_dir = root / "metrics"

            manifest = export_paper_metrics(
                year=2024,
                iso_week=2,
                output_dir=output_dir,
                idw_run_dir=idw_run,
                lstm_run_dir=lstm_run,
                unet_run_dir=unet_run,
                en4_holdout_fraction=0.2,
                seed=7,
                validation_year=2018,
                climatology_idw_neighbors=1,
                climatology_idw_chunk_size=8,
                profile_chunk_size=2,
            )

            summary = pd.read_csv(output_dir / "paper_metrics_summary.csv")
            by_depth = pd.read_csv(output_dir / "paper_metrics_by_depth.csv")
            holdout = pd.read_csv(output_dir / "en4_holdout_locations.csv")
            table = (output_dir / "recon_results_table.tex").read_text(encoding="utf-8")

            self.assertEqual(manifest["selected_date"], 20240108)
            self.assertEqual(len(summary), 18)
            self.assertEqual(len(by_depth), 36)
            self.assertIn("Climatology", set(summary["method_label"]))
            self.assertIn("GLORYS12", set(summary["method_label"]))
            self.assertIn("U-Net", table)
            self.assertIn("\\textbf", table)
            self.assertIn("evaluation year 2018", table)
            self.assertEqual(len(holdout), 1)
            self.assertTrue((output_dir / "climatology_temperature.tif").is_file())
            self.assertTrue((output_dir / "climatology_salinity.tif").is_file())
            self.assertTrue((output_dir / "paper_metrics_manifest.json").is_file())
            with (output_dir / "paper_metrics_manifest.json").open(
                "r", encoding="utf-8"
            ) as f:
                manifest_json = json.load(f)
            self.assertEqual(manifest_json["depth_averaging"], "equal_depth_mean")
            self.assertEqual(manifest_json["max_depth_m"], 2000.0)
            self.assertEqual(manifest_json["evaluated_depth_count"], 2)
            self.assertIn("no deeper than 2000 m", table)

    def test_bundle_metrics_reads_dynamic_methods_and_disk_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geotiff_root, _cache_dir, _land_mask_path = _make_geotiff_dataset(root)
            methods = ["idw", "cnn", "depthdif"]
            for offset, method in enumerate(methods):
                self._write_method_run(
                    root / "methods",
                    method=method,
                    geotiff_root=geotiff_root,
                    offset=float(offset) * 0.25,
                )
            self._write_paper_bundle_manifest(
                root, geotiff_root=geotiff_root, methods=methods
            )
            output_dir = root / "metrics"

            with (
                patch(
                    "depth_recon.inference.export_paper_metrics._load_holdout_profiles",
                    side_effect=AssertionError("profile store should not be opened"),
                ),
                patch(
                    "depth_recon.inference.export_paper_metrics._manifest_source_raster_path",
                    side_effect=AssertionError(
                        "GLORYS source rasters should not be opened"
                    ),
                ),
            ):
                manifest = export_paper_metrics(
                    paper_run_dir=root,
                    output_dir=output_dir,
                    climatology_idw_neighbors=1,
                    climatology_idw_chunk_size=8,
                    profile_chunk_size=2,
                )

            summary = pd.read_csv(output_dir / "paper_metrics_summary.csv")
            by_depth = pd.read_csv(output_dir / "paper_metrics_by_depth.csv")
            labels = set(summary["method_label"].tolist())

            self.assertEqual(manifest["method_order"], [*methods, "glorys"])
            self.assertEqual(manifest["max_depth_m"], 2000.0)
            self.assertEqual(manifest["evaluated_depth_count"], 2)
            self.assertEqual(len(summary), 14)
            self.assertEqual(len(by_depth), 28)
            self.assertIn("CNN", labels)
            self.assertIn("DepthDif", labels)
            self.assertIn("GLORYS12", labels)
            en4_metrics = pd.read_csv(output_dir / "en4_holdout_metrics.csv")
            glorys_en4 = en4_metrics[en4_metrics["method"] == "glorys"]
            self.assertEqual(len(glorys_en4), 4)
            self.assertTrue(np.allclose(glorys_en4["rmse"], 0.0))
            dense_metrics = pd.read_csv(output_dir / "glorys_field_metrics.csv")
            self.assertNotIn("glorys", set(dense_metrics["method"]))
            self.assertTrue((output_dir / "en4_holdout_metrics.csv").is_file())
            self.assertTrue((output_dir / "glorys_field_metrics.csv").is_file())

    def test_metrics_missing_native_depth_raster_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geotiff_root, _cache_dir, _land_mask_path = _make_geotiff_dataset(root)
            cnn_run = self._write_method_run(
                root, method="cnn", geotiff_root=geotiff_root, offset=0.0
            )
            (cnn_run / "temperature" / "temperature_depth_001.tif").unlink()
            runs = load_method_runs({"cnn": cnn_run}, method_labels={"cnn": "CNN"})

            with self.assertRaisesRegex(
                RuntimeError,
                "CNN temperature run must contain all 2 native depth prediction rasters",
            ):
                prediction_specs_by_method(
                    runs, depth_count=2, method_labels={"cnn": "CNN"}
                )


if __name__ == "__main__":
    unittest.main()
