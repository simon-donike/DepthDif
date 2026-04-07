import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from data.dataset_ostia_argo import OstiaArgoTileDataset


def _make_dataset(temp_dir: Path) -> OstiaArgoTileDataset:
    dataset = OstiaArgoTileDataset.__new__(OstiaArgoTileDataset)
    dataset.return_argo_profiles = True
    dataset.tile_size = 4
    dataset.days = 1
    dataset.output_units = "celsius"
    dataset.glorys_var_name = "thetao"
    dataset.csv_dir = temp_dir
    dataset._depth_v2_root = None
    dataset._ostia_path_col = "ostia_file_path"
    dataset._argo_path_col = "argo_file_path"
    dataset._glorys_path_col = "matched_glorys_file_path"
    dataset._glorys_target_depths_cache = np.arange(1.0, 51.0, dtype=np.float64)
    dataset._rows = [
        {
            "date": "20260115",
            "patch_id": "7",
            "lat0": 10.0,
            "lat1": 11.0,
            "lon0": 20.0,
            "lon1": 21.0,
            "ostia_file_path": "ostia/source.nc",
            "argo_file_path": "argo/source.nc",
            "matched_glorys_file_path": "glorys/source.nc",
        }
    ]
    return dataset


def _make_sample(valid_mask_1d_pixels: int = 16) -> dict[str, object]:
    x = torch.ones((50, 4, 4), dtype=torch.float32)
    y = torch.full((50, 4, 4), 2.0, dtype=torch.float32)
    eo = torch.ones((1, 4, 4), dtype=torch.float32)
    x_valid_mask = torch.ones((50, 4, 4), dtype=torch.bool)
    x_valid_mask_1d = torch.zeros((1, 4, 4), dtype=torch.bool)
    x_valid_mask_1d.view(-1)[:valid_mask_1d_pixels] = True
    return {
        "x": x,
        "y": y,
        "eo": eo,
        "x_valid_mask": x_valid_mask,
        "x_valid_mask_1d": x_valid_mask_1d,
        "info": {},
    }


class SaveToDiskTests(unittest.TestCase):
    def test_align_argo_profile_to_glorys_depths_interpolates_in_range(self) -> None:
        aligned = OstiaArgoTileDataset._align_argo_profile_to_glorys_depths(
            temperature=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            depth=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
            glorys_depths=np.asarray([15.0, 20.0, 35.0], dtype=np.float32),
        )

        self.assertAlmostEqual(float(aligned[0]), 1.5, places=6)
        self.assertAlmostEqual(float(aligned[1]), 2.0, places=6)
        self.assertTrue(np.isnan(aligned[2]))

    def test_align_argo_profile_to_glorys_depths_uses_10m_floor_cutoff(self) -> None:
        aligned = OstiaArgoTileDataset._align_argo_profile_to_glorys_depths(
            temperature=np.asarray([5.0, 25.0], dtype=np.float32),
            depth=np.asarray([0.5, 25.0], dtype=np.float32),
            glorys_depths=np.asarray([10.0], dtype=np.float32),
        )

        self.assertTrue(np.isfinite(aligned[0]))
        self.assertAlmostEqual(float(aligned[0]), 12.7551022, places=5)

    def test_align_argo_profile_to_glorys_depths_sorts_and_collapses_duplicates(self) -> None:
        aligned = OstiaArgoTileDataset._align_argo_profile_to_glorys_depths(
            temperature=np.asarray([3.0, 1.0, 2.0, 4.0], dtype=np.float32),
            depth=np.asarray([30.0, 10.0, 20.0, 20.0], dtype=np.float32),
            glorys_depths=np.asarray([20.0, 25.0], dtype=np.float32),
        )

        self.assertAlmostEqual(float(aligned[0]), 3.0, places=6)
        self.assertAlmostEqual(float(aligned[1]), 3.0, places=6)

    def test_getitem_returns_glorys_aligned_channel_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = OstiaArgoTileDataset.__new__(OstiaArgoTileDataset)
            dataset.return_argo_profiles = True
            dataset.tile_size = 4
            dataset.days = 1
            dataset.output_units = "kelvin"
            dataset.csv_dir = temp_dir
            dataset._depth_v2_root = None
            dataset._ostia_path_col = "ostia_file_path"
            dataset._argo_path_col = "argo_file_path"
            dataset._glorys_path_col = "matched_glorys_file_path"
            dataset._glorys_target_depths_cache = np.arange(1.0, 51.0, dtype=np.float64)
            dataset._rows = [
                {
                    "date": "20260115",
                    "patch_id": "7",
                    "lat0": 10.0,
                    "lat1": 11.0,
                    "lon0": 20.0,
                    "lon1": 21.0,
                    "ostia_file_path": "ostia/source.nc",
                    "argo_file_path": "argo/source.nc",
                    "matched_glorys_file_path": "glorys/source.nc",
                }
            ]
            dataset._select_temporal_rows = lambda row: [row]
            dataset._resolve_index_path = lambda value: temp_dir / str(value)
            dataset._load_ostia_patch = lambda **_: np.ones((4, 4), dtype=np.float32)
            dataset._load_glorys_patch = lambda **_: np.broadcast_to(
                np.arange(50, dtype=np.float32)[:, None, None],
                (50, 4, 4),
            )
            dataset._load_argo_profiles_for_date = lambda **_: {
                "temperature": torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], dtype=torch.float32),
                "depth": torch.tensor([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]], dtype=torch.float32),
                "latitude": torch.tensor([10.25, 10.75], dtype=torch.float32),
                "longitude": torch.tensor([20.25, 20.75], dtype=torch.float32),
                "profile_idx": torch.tensor([0, 1], dtype=torch.long),
                "profile_dates": torch.tensor([20260115, 20260115], dtype=torch.long),
            }

            for rel_path in ("ostia/source.nc", "argo/source.nc", "glorys/source.nc"):
                path = temp_dir / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            sample = dataset[0]

        self.assertEqual(tuple(sample["x"].shape), (50, 4, 4))
        self.assertEqual(tuple(sample["y"].shape), (50, 4, 4))
        self.assertEqual(tuple(sample["x_valid_mask"].shape), (50, 4, 4))
        self.assertEqual(tuple(sample["y_valid_mask"].shape), (50, 4, 4))
        self.assertEqual(sample["info"]["glorys_depth_band_count"], 50)
        self.assertEqual(sample["info"]["argo_depth_semantics"], "glorys_depth_index")

    def test_skips_existing_pair_without_loading_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = _make_dataset(temp_dir)
            output_root = temp_dir / "export"
            basename = dataset._export_basename_from_row(dataset._rows[0])
            argo_path = output_root / "argo" / f"{basename}.tif"
            ostia_path = output_root / "ostia" / f"{basename}.tif"
            glorys_path = output_root / "glorys" / f"{basename}.tif"
            argo_path.parent.mkdir(parents=True, exist_ok=True)
            ostia_path.parent.mkdir(parents=True, exist_ok=True)
            glorys_path.parent.mkdir(parents=True, exist_ok=True)
            argo_path.write_bytes(b"argo")
            ostia_path.write_bytes(b"ostia")
            glorys_path.write_bytes(b"glorys")

            with mock.patch.object(
                dataset,
                "__getitem__",
                side_effect=AssertionError("should not load sample"),
            ), mock.patch.object(
                dataset,
                "_count_valid_spatial_observations_from_argo_tif",
                return_value=16,
            ):
                record = dataset.save_to_disk(0, output_root=output_root, write_manifest=False)

            self.assertEqual(record["files_written"], 0)
            self.assertEqual(record["argo_tif_path"], argo_path.relative_to(output_root).as_posix())
            self.assertEqual(record["ostia_tif_path"], ostia_path.relative_to(output_root).as_posix())
            self.assertEqual(
                record["glorys_tif_path"], glorys_path.relative_to(output_root).as_posix()
            )

    def test_writes_missing_pair_after_loading_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = _make_dataset(temp_dir)
            output_root = temp_dir / "export"

            with mock.patch.object(dataset, "__getitem__", return_value=_make_sample()) as getitem_mock:
                with mock.patch.object(dataset, "_write_sample_tiffs") as write_mock:
                    record = dataset.save_to_disk(0, output_root=output_root, write_manifest=False)

            self.assertEqual(getitem_mock.call_count, 1)
            self.assertEqual(write_mock.call_count, 1)
            self.assertEqual(record["files_written"], 1)

    def test_write_sample_tiffs_preserves_georeferencing_and_depth_metadata(self) -> None:
        try:
            import rasterio
        except ImportError:
            self.skipTest("rasterio is required for GeoTIFF export tests")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = _make_dataset(temp_dir)
            dataset._glorys_path_col = "matched_glorys_file_path"
            dataset._glorys_target_depths_cache = np.arange(1.0, 51.0, dtype=np.float64)
            dataset._rows[0]["matched_glorys_file_path"] = "glorys/source.nc"
            sample = _make_sample()
            argo_path = temp_dir / "argo.tif"
            ostia_path = temp_dir / "ostia.tif"
            glorys_path = temp_dir / "glorys.tif"

            dataset._write_sample_tiffs(
                row=dataset._rows[0],
                x=sample["x"],
                y=sample["y"],
                eo=sample["eo"],
                valid_mask=sample["x_valid_mask"],
                glorys_depths_for_export=np.arange(1.0, 51.0, dtype=np.float64),
                argo_tif_path=argo_path,
                ostia_tif_path=ostia_path,
                glorys_tif_path=glorys_path,
            )

            with rasterio.open(argo_path) as ds:
                self.assertEqual(ds.crs.to_string(), "EPSG:4326")
                self.assertAlmostEqual(ds.bounds.left, 20.0, places=6)
                self.assertAlmostEqual(ds.bounds.right, 21.0, places=6)
                self.assertAlmostEqual(ds.bounds.bottom, 10.0, places=6)
                self.assertAlmostEqual(ds.bounds.top, 11.0, places=6)
                self.assertEqual(ds.descriptions[0], "argo_glorys_level_0_1.000m")
                self.assertEqual(ds.descriptions[1], "argo_glorys_level_1_2.000m")
                self.assertEqual(ds.descriptions[2], "argo_glorys_level_2_3.000m")
                self.assertEqual(ds.tags()["argo_depth_semantics"], "glorys_depth_index")
                self.assertEqual(ds.tags()["argo_glorys_depth_m"], "1.000|2.000|3.000|4.000|5.000|6.000|7.000|8.000|9.000|10.000|11.000|12.000|13.000|14.000|15.000|16.000|17.000|18.000|19.000|20.000|21.000|22.000|23.000|24.000|25.000|26.000|27.000|28.000|29.000|30.000|31.000|32.000|33.000|34.000|35.000|36.000|37.000|38.000|39.000|40.000|41.000|42.000|43.000|44.000|45.000|46.000|47.000|48.000|49.000|50.000")

            with rasterio.open(ostia_path) as ds:
                self.assertEqual(ds.crs.to_string(), "EPSG:4326")
                self.assertAlmostEqual(ds.bounds.left, 20.0, places=6)
                self.assertAlmostEqual(ds.bounds.right, 21.0, places=6)

            with rasterio.open(glorys_path) as ds:
                self.assertEqual(ds.crs.to_string(), "EPSG:4326")
                self.assertEqual(ds.tags()["value_encoding"], "packed_int16_celsius_x100")

    def test_raises_on_partial_existing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = _make_dataset(temp_dir)
            output_root = temp_dir / "export"
            basename = dataset._export_basename_from_row(dataset._rows[0])
            argo_path = output_root / "argo" / f"{basename}.tif"
            argo_path.parent.mkdir(parents=True, exist_ok=True)
            argo_path.write_bytes(b"argo")

            with mock.patch.object(
                dataset,
                "__getitem__",
                side_effect=AssertionError("should not load sample"),
            ):
                with self.assertRaises(FileExistsError):
                    dataset.save_to_disk(0, output_root=output_root, write_manifest=False)

    def test_overwrite_existing_pair_loads_and_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            dataset = _make_dataset(temp_dir)
            output_root = temp_dir / "export"
            basename = dataset._export_basename_from_row(dataset._rows[0])
            argo_path = output_root / "argo" / f"{basename}.tif"
            ostia_path = output_root / "ostia" / f"{basename}.tif"
            glorys_path = output_root / "glorys" / f"{basename}.tif"
            argo_path.parent.mkdir(parents=True, exist_ok=True)
            ostia_path.parent.mkdir(parents=True, exist_ok=True)
            glorys_path.parent.mkdir(parents=True, exist_ok=True)
            argo_path.write_bytes(b"argo")
            ostia_path.write_bytes(b"ostia")
            glorys_path.write_bytes(b"glorys")

            with mock.patch.object(dataset, "__getitem__", return_value=_make_sample()) as getitem_mock:
                with mock.patch.object(dataset, "_write_sample_tiffs") as write_mock:
                    record = dataset.save_to_disk(
                        0,
                        output_root=output_root,
                        overwrite=True,
                        write_manifest=False,
                    )

            self.assertEqual(getitem_mock.call_count, 1)
            self.assertEqual(write_mock.call_count, 1)
            self.assertEqual(record["files_written"], 1)


if __name__ == "__main__":
    unittest.main()
