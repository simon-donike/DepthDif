from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.inference.export_paper_metrics import (
    _metric_stats,
    export_paper_metrics,
    idw_fill_2d,
    summarize_equal_depth_metrics,
)
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
            self.assertEqual(len(summary), 16)
            self.assertEqual(len(by_depth), 32)
            self.assertIn("Climatology", set(summary["method_label"]))
            self.assertIn("U-Net", table)
            self.assertIn("\\textbf", table)
            self.assertEqual(len(holdout), 1)
            self.assertTrue((output_dir / "climatology_temperature.tif").is_file())
            self.assertTrue((output_dir / "climatology_salinity.tif").is_file())
            self.assertTrue((output_dir / "paper_metrics_manifest.json").is_file())
            with (output_dir / "paper_metrics_manifest.json").open(
                "r", encoding="utf-8"
            ) as f:
                manifest_json = json.load(f)
            self.assertEqual(manifest_json["depth_averaging"], "equal_depth_mean")


if __name__ == "__main__":
    unittest.main()
