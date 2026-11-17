"""Small artifact-level checks for training-only climatology fitting."""

from pathlib import Path
import json
import pickle
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import rasterio
import yaml

from depth_recon.data.climatology import MonthlyClimatology, fit_monthly_climatology
from depth_recon.data.preflight_climatology import (
    ensure_climatology,
    file_sha256,
    prepare_training_climatology,
)
from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from tests.test_argo_geotiff_gridded_dataset import _make_geotiff_dataset


class ClimatologyTests(unittest.TestCase):
    """Validate fitting, provenance, and sampling of climatology artifacts."""

    @staticmethod
    def _write_raster(path: Path, values: np.ndarray) -> None:
        """Write one tiny uint8 raster with production stretch metadata."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[-2],
            width=values.shape[-1],
            count=values.shape[0],
            dtype="uint8",
        ) as dst:
            dst.write(values.astype("uint8"))

    def _fixture_root(self, root: Path) -> Path:
        """Create two dates, including a validation-year outlier."""
        raster_dir = root / "rasters"
        raster_dir.mkdir(parents=True)
        entries = []
        for date, value in ((20200115, 10), (20210115, 250)):
            name = f"{date}.tif"
            self._write_raster(
                raster_dir / name, np.full((2, 2, 2), value, dtype=np.uint8)
            )
            entries.append({"date": date, "path": f"rasters/{name}"})
        manifest = {
            "depth_axis_m": [0.0, 10.0],
            "grid": {"width": 2, "height": 2},
            "stretch": {"temperature_kelvin": {"minimum": 273.15, "maximum": 573.15}},
            "rasters": {"glorys": {"thetao": entries}},
        }
        with (root / "manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle)
        return root

    def test_fit_excludes_validation_year_and_records_provenance(self) -> None:
        """The validation outlier must not affect training climatology."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            output = fit_monthly_climatology(
                root, Path(temp_dir) / "clim.npz", val_year=2021, spatial_stride=1
            )
            artifact = MonthlyClimatology(output)
            self.assertEqual(artifact.metadata["val_year_excluded"], 2021)
            self.assertTrue(np.allclose(artifact.temperature[0], 10.0 * 300.0 / 254.0))
            # An absent month falls back to training observations, not zero Celsius.
            np.testing.assert_array_equal(
                artifact.temperature[1], artifact.temperature[0]
            )

    def test_joint_dataset_preserves_field_masks_and_worker_backgrounds(self) -> None:
        """Exercise production export, background loading, field support, and worker restore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root, cache, land = _make_geotiff_dataset(Path(temp_dir))
            artifact = fit_monthly_climatology(
                root,
                root / "climatology.npz",
                val_year=2016,
                spatial_stride=1,
                field="both",
            )
            settings = dict(
                geotiff_root_dir=root,
                metadata_cache_dir=cache,
                split="all",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2016,
                include_salinity=True,
                surface_conditioning={"sources": ["sst"]},
                wet_domain={"enabled": True, "reference_date": None},
                climatology={"enabled": True, "path": str(artifact)},
            )
            dataset = ArgoGeoTIFFGriddedPatchDataset(**settings)
            sample = dataset[0]
            for suffix in ("", "_salinity"):
                np.testing.assert_array_equal(
                    sample[f"wet_mask{suffix}"], sample[f"y{suffix}_valid_mask"]
                )
                self.assertTrue(
                    np.isfinite(sample[f"climatology{suffix}"].numpy()).all()
                )
            self.assertFalse(
                np.array_equal(sample["wet_mask"], sample["wet_mask_salinity"])
            )
            restored = pickle.loads(pickle.dumps(dataset))
            np.testing.assert_array_equal(
                restored[0]["climatology"], sample["climatology"]
            )
            scalar = ArgoGeoTIFFGriddedPatchDataset(
                **dict(settings, output_fields=["salinity"])
            )
            self.assertNotIn("climatology", scalar[0])
            self.assertIn("climatology_salinity", scalar[0])
            # Same-sized but geographically different priors must not pass validation.
            with np.load(artifact, allow_pickle=False) as loaded:
                payload = {key: loaded[key] for key in loaded.files}
            metadata = json.loads(str(payload["metadata"]))
            metadata["unsupported_depth_indices"] = {"temperature": [0]}
            payload["metadata"] = json.dumps(metadata)
            np.savez_compressed(artifact, **payload)
            unsupported = ArgoGeoTIFFGriddedPatchDataset(**settings)
            with self.assertRaisesRegex(ValueError, "unsupported climatology depths"):
                unsupported[0]
            metadata["unsupported_depth_indices"] = {}
            metadata["grid"]["width"] += 1
            payload["metadata"] = json.dumps(metadata)
            np.savez_compressed(artifact, **payload)
            with self.assertRaisesRegex(ValueError, "spatial grid"):
                ArgoGeoTIFFGriddedPatchDataset(**settings)

    def test_empty_depth_preserves_zero_support_and_records_placeholder(self) -> None:
        """A wholly absent source band must not become a fabricated fitted mean."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            with rasterio.open(root / "rasters/20200115.tif", "r+") as raster:
                raster.write(np.full((2, 2), 255, dtype=np.uint8), 2)
            output = fit_monthly_climatology(
                root, Path(temp_dir) / "clim.npz", val_year=2021, spatial_stride=1
            )
            artifact = MonthlyClimatology(output)
            self.assertEqual(
                artifact.metadata["unsupported_depth_indices"], {"temperature": [1]}
            )
            self.assertTrue(np.all(artifact.count[:, 1] == 0))
            self.assertTrue(np.all(artifact.temperature[:, 1] == 0))
            self.assertTrue(np.all(artifact.count[0, 0] == 1))

    def test_preflight_generates_relative_background_once(self) -> None:
        """Concurrent ranks share one fit; repeated preflight preserves frozen bytes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            relative = "hpc_residual/climatology/temperature_monthly_stride4.npz"
            with (
                patch(
                    "depth_recon.data.preflight_climatology.fit_monthly_climatology",
                    wraps=fit_monthly_climatology,
                ) as fit,
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                tasks = [
                    pool.submit(ensure_climatology, root, relative, val_year=2021)
                    for _ in range(2)
                ]
                paths = [task.result() for task in tasks]
                self.assertEqual(fit.call_count, 1)
            self.assertEqual(paths, [root / relative] * 2)
            original = paths[0].read_bytes()
            ensure_climatology(root, relative, val_year=2021)
            self.assertEqual(paths[0].read_bytes(), original)
            receipt = json.loads(
                paths[0].with_suffix(".npz.preflight.json").read_text()
            )
            self.assertEqual(receipt["artifact_sha256"], file_sha256(paths[0]))
            self.assertEqual(
                set(receipt["source_hashes"]), {"manifest.yaml", "rasters/20200115.tif"}
            )
            self.assertEqual(
                receipt["recipe"],
                {"val_year": 2021, "spatial_stride": 4, "field": "temperature"},
            )
            artifact = MonthlyClimatology(paths[0])
            self.assertTrue(np.allclose(artifact.temperature, 10.0 * 300.0 / 254.0))

    def test_preflight_does_not_replace_incompatible_existing_background(self) -> None:
        """An existing wrong-year, wrong-field or corrupt artifact fails intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            path = ensure_climatology(root, "clim.npz", val_year=2021)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "validation year"):
                ensure_climatology(root, "clim.npz", val_year=2020)
            with self.assertRaisesRegex(ValueError, "requested field"):
                ensure_climatology(root, "clim.npz", val_year=2021, field="salinity")
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(b"broken artifact")
            with self.assertRaises(ValueError):
                ensure_climatology(root, "clim.npz", val_year=2021)
            self.assertEqual(path.read_bytes(), b"broken artifact")

    def test_preflight_failed_fit_does_not_publish_partial_artifact(self) -> None:
        """A failed fit can be retried without mistaking a partial file for a baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            path = root / "clim.npz"
            with patch(
                "depth_recon.data.preflight_climatology.fit_monthly_climatology",
                side_effect=RuntimeError("interrupted fit"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted fit"):
                    ensure_climatology(root, path, val_year=2021)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".npz.preflight.json").exists())
            self.assertFalse(list(root.glob(".climatology-*")))
            self.assertTrue(ensure_climatology(root, path, val_year=2021).is_file())

    def test_training_preflight_uses_data_root_and_model_fields(self) -> None:
        """Training prepares joint backgrounds at the configured dataset-relative path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root, _, _ = _make_geotiff_dataset(Path(temp_dir))
            config = {
                "dataset": {
                    "core": {"geotiff_root_dir": str(root)},
                    "climatology": {
                        "enabled": True,
                        "path": "climatology/joint.npz",
                        "spatial_stride": 1,
                    },
                },
                "split": {"val_year": 2016},
            }
            path = prepare_training_climatology(
                config, {"model": {"output_fields": ["temperature", "salinity"]}}
            )
            self.assertEqual(path, root / "climatology/joint.npz")
            artifact = MonthlyClimatology(path)
            self.assertIsNotNone(artifact.temperature)
            self.assertIsNotNone(artifact.salinity)
            self.assertIsNone(prepare_training_climatology({}, {}))
            del config["split"]["val_year"]
            with self.assertRaisesRegex(ValueError, "val_year"):
                prepare_training_climatology(config, {})

    def test_train_entry_point_prepares_missing_background_without_cli_preflight(
        self,
    ) -> None:
        """A direct training launch fits the baseline before its first dataset opens."""
        import os
        import train

        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            relative = "hpc_residual/climatology/temperature_monthly_stride4.npz"
            config = yaml.safe_load(
                (
                    Path(train.__file__).parent
                    / "src/depth_recon/configs/px_space/training_super_config.yaml"
                ).read_text()
            )
            config["data"]["dataset"]["core"]["geotiff_root_dir"] = str(root)
            config["data"]["dataset"]["climatology"] = {
                "enabled": True,
                "path": relative,
            }
            config["data"]["split"]["val_year"] = 2021
            config["model"]["climatology_residual"] = True
            config_path = Path(temp_dir) / "training.yaml"
            config_path.write_text(yaml.safe_dump(config))
            self.assertFalse((root / relative).exists())

            def check_background_then_stop(**kwargs):
                """Check real preparation at the dataset boundary without allocating GPUs."""
                artifact = MonthlyClimatology(root / relative)
                self.assertEqual(artifact.metadata["val_year_excluded"], 2021)
                self.assertEqual(artifact.metadata["dates"], [20200115])
                self.assertTrue(np.allclose(artifact.temperature, 10.0 * 300.0 / 254.0))
                raise RuntimeError("dataset boundary reached")

            # Only the expensive dataset/model phase is stubbed: configuration
            # resolution, the startup hook and climatology fitting run normally.
            with (
                patch.dict(os.environ),
                patch.object(
                    train, "build_dataset", side_effect=check_background_then_stop
                ) as build,
                patch.object(train.pl, "seed_everything"),
                self.assertRaisesRegex(RuntimeError, "dataset boundary reached"),
            ):
                train.main(
                    config_path_value=str(config_path),
                    run_dir_value=str(Path(temp_dir) / "run"),
                )
            build.assert_called_once()
            self.assertTrue(
                (root / relative).with_suffix(".npz.preflight.json").is_file()
            )

    def test_non_aligned_patch_uses_source_grid_coordinates(self) -> None:
        """Coarse-grid sampling must not move non-aligned patches toward the origin."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "climatology.npz"
            values = np.broadcast_to(
                np.arange(9).reshape(1, 1, 3, 3), (12, 1, 3, 3)
            ).copy()
            np.savez_compressed(
                path,
                temperature=values,
                count=np.ones_like(values),
                depth_axis_m=[0.5],
                metadata=json.dumps(
                    {
                        "field": "temperature",
                        "spatial_stride": 2,
                        "source_grid_shape": [5, 5],
                        "grid_shape": [3, 3],
                    }
                ),
            )
            result = MonthlyClimatology(path).sample_patch(
                date=20200101, grid_y0=1, grid_x0=1, tile_size=3
            )
            np.testing.assert_array_equal(result[0], np.arange(9).reshape(3, 3))

    def test_large_members_are_cached_as_mmaps_by_artifact_identity(self) -> None:
        """A temperature run must share its extracted field and ignore inactive salinity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "climatology.npz"
            cache = root / "cache"
            values = np.ones((12, 1, 2, 2), dtype=np.float32)
            metadata = json.dumps(
                {
                    "field": "temperature",
                    "spatial_stride": 1,
                    "source_grid_shape": [2, 2],
                    "grid_shape": [2, 2],
                }
            )
            np.savez_compressed(
                path,
                temperature=values,
                salinity=values + 3.0,
                count=np.ones_like(values, dtype=np.int32),
                depth_axis_m=[0.5],
                metadata=metadata,
            )
            with patch.dict(
                "os.environ", {"DEPTH_RECON_CLIMATOLOGY_CACHE_DIR": str(cache)}
            ):
                first = MonthlyClimatology(path)
                self.assertIsInstance(first.temperature, np.memmap)
                self.assertIsInstance(first.count, np.memmap)
                self.assertIsNone(first.salinity)
                self.assertFalse((first._cache_dir / "salinity.npy").exists())
                first_cache_dir = first._cache_dir
                second = MonthlyClimatology(path)
                self.assertIsInstance(second.temperature, np.memmap)
                self.assertEqual(second._cache_dir, first_cache_dir)

                # Replacing bytes at the same path must select a new cache namespace.
                np.savez_compressed(
                    path,
                    temperature=values * 9.0,
                    salinity=values + 3.0,
                    count=np.ones_like(values, dtype=np.int32),
                    depth_axis_m=[0.5],
                    metadata=metadata,
                )
                replaced = MonthlyClimatology(path)
                self.assertEqual(float(replaced.temperature[0, 0, 0, 0]), 9.0)
                self.assertNotEqual(replaced._cache_dir, first_cache_dir)

    def test_sample_rejects_outside_grid(self) -> None:
        """A patch extending past artifact coverage must fail loudly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            output = fit_monthly_climatology(
                root, Path(temp_dir) / "clim.npz", val_year=2022, spatial_stride=1
            )
            artifact = MonthlyClimatology(output)
            with self.assertRaisesRegex(ValueError, "outside"):
                artifact.sample_patch(date=20200115, grid_y0=1, grid_x0=1, tile_size=2)

    def test_fit_requires_validation_year(self) -> None:
        """Training-only artifacts must declare the held-out year."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._fixture_root(Path(temp_dir) / "dataset")
            with self.assertRaisesRegex(ValueError, "val_year"):
                fit_monthly_climatology(
                    root, Path(temp_dir) / "clim.npz", val_year=None
                )


if __name__ == "__main__":
    unittest.main()
