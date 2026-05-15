from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import xarray as xr
import yaml

from depth_recon.data.dataset_argo_netcdf_gridded import ArgoNetCDFGriddedPatchDataset
from depth_recon.inference.export_global import _load_glorys_depth_axis_m
from depth_recon.paths import config_path
from train import build_dataset
from depth_recon.utils.normalizations import temperature_normalize


def _days_since_1950(date_value: int) -> float:
    text = str(int(date_value))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float((day - np.datetime64("1950-01-01", "D")).astype(int))


def _write_argo_netcdf(
    root_dir: Path,
    *,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
    temperatures: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    dates: np.ndarray | None = None,
) -> None:
    lat = (
        np.asarray([1.5, 1.6], dtype=np.float64)
        if latitudes is None
        else np.asarray(latitudes, dtype=np.float64)
    )
    lon = (
        np.asarray([10.5, 10.6], dtype=np.float64)
        if longitudes is None
        else np.asarray(longitudes, dtype=np.float64)
    )
    temp = (
        np.asarray([[10.0, 20.0], [14.0, 24.0]], dtype=np.float32)
        if temperatures is None
        else np.asarray(temperatures, dtype=np.float32)
    )
    depth_values = (
        np.asarray([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
        if depths is None
        else np.asarray(depths, dtype=np.float32)
    )
    date_values = (
        np.asarray([20240102, 20240102], dtype=np.int64)
        if dates is None
        else np.asarray(dates, dtype=np.int64)
    )
    n_prof = int(lat.size)
    ds = xr.Dataset(
        data_vars={
            "JULD": (
                ("N_PROF",),
                np.asarray([_days_since_1950(int(v)) for v in date_values[:n_prof]]),
            ),
            "LATITUDE": (("N_PROF",), lat[:n_prof]),
            "LONGITUDE": (("N_PROF",), lon[:n_prof]),
            "TEMP": (("N_PROF", "N_LEVELS"), temp[:n_prof]),
            "DEPH_CORRECTED": (("N_PROF", "N_LEVELS"), depth_values[:n_prof]),
        },
        coords={
            "N_PROF": np.arange(n_prof, dtype=np.int64),
            "N_LEVELS": np.arange(int(temp.shape[1]), dtype=np.int64),
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / "EN.4.2.2.f.profiles.g10.202401.nc", engine="h5netcdf")


def _write_glorys(root_dir: Path, *, date_value: int, base: float) -> None:
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    depth = np.asarray([0.0, 10.0], dtype=np.float32)
    thetao = np.asarray(
        [
            [
                [[base + 1.0, base + 2.0], [base + 3.0, base + 4.0]],
                [[base + 11.0, base + 12.0], [base + 13.0, base + 14.0]],
            ]
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {"thetao": (("time", "depth", "latitude", "longitude"), thetao)},
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1950-01-01 00:00:00"
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"glorys_{int(date_value)}.nc", engine="h5netcdf")


def _write_ostia(root_dir: Path, *, date_value: int, base_kelvin: float) -> None:
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    analysed_sst = np.asarray(
        [
            [
                [base_kelvin + 1.0, base_kelvin + 2.0],
                [base_kelvin + 3.0, base_kelvin + 4.0],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.zeros((1, 2, 2), dtype=np.int16)
    ds = xr.Dataset(
        {
            "analysed_sst": (("time", "lat", "lon"), analysed_sst),
            "mask": (("time", "lat", "lon"), mask),
        },
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "lat": lat,
            "lon": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1950-01-01 00:00:00"
    ds["mask"].attrs["flag_meanings"] = "sea land"
    ds["mask"].attrs["flag_masks"] = np.asarray([1, 2], dtype=np.int16)
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"{int(date_value)}120000-ostia.nc", engine="h5netcdf")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_land_mask_geotiff(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(values, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(mask.shape[0]),
        width=int(mask.shape[1]),
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10.0, 2.0, 1.0, 1.0),
    ) as dst:
        dst.write(mask, 1)
    return path


def _make_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    argo_dir = tmp_path / "en4_profiles"
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    cache_dir = tmp_path / "cache"
    _write_argo_netcdf(argo_dir)
    _write_glorys(glorys_dir, date_value=20240102, base=30.0)
    _write_glorys(glorys_dir, date_value=20240103, base=40.0)
    _write_ostia(ostia_dir, date_value=20240102, base_kelvin=280.0)
    _write_ostia(ostia_dir, date_value=20240103, base_kelvin=290.0)
    return argo_dir, glorys_dir, ostia_dir, cache_dir


def _dataset_kwargs(tmp_path: Path) -> dict[str, object]:
    argo_dir, glorys_dir, ostia_dir, cache_dir = _make_sources(tmp_path)
    return {
        "argo_dir": argo_dir,
        "glorys_dir": glorys_dir,
        "ostia_dir": ostia_dir,
        "sealevel_dir": None,
        "metadata_cache_dir": cache_dir,
        "tile_size": 2,
        "resolution_deg": 1.0,
        "patch_grid_source": "ostia_mask",
        "temporal_window_days": 1,
        "invalid_threshold": 0.5,
        "val_fraction": 0.0,
        "random_seed": 7,
        "return_info": True,
        "return_coords": True,
    }


class TestArgoNetCDFGriddedPatchDataset(unittest.TestCase):
    def test_contract_and_duplicate_profile_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArgoNetCDFGriddedPatchDataset(
                **_dataset_kwargs(Path(tmpdir)),
                split="train",
                require_argo_for_train=True,
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]

            self.assertEqual(sample["eo"].shape, (1, 2, 2))
            self.assertEqual(sample["x"].shape, (2, 2, 2))
            self.assertEqual(sample["y"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["y_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask_1d"].shape, (1, 2, 2))
            self.assertEqual(sample["land_mask"].shape, (1, 2, 2))
            self.assertEqual(sample["date"], 20240102)
            self.assertTrue(
                torch.allclose(sample["coords"], torch.tensor([1.0, 11.0]), atol=1e-5)
            )

            x_c = temperature_normalize(mode="denorm", tensor=sample["x"])
            x_mask = sample["x_valid_mask"]
            self.assertTrue(bool(x_mask[:, 0, 0].all().item()))
            self.assertFalse(bool(x_mask[:, 0, 1].any().item()))
            self.assertTrue(
                torch.allclose(
                    x_c[:, 0, 0],
                    torch.tensor([12.0, 22.0], dtype=torch.float32),
                    atol=1e-5,
                )
            )

            eo_c = temperature_normalize(mode="denorm", tensor=sample["eo"])
            self.assertAlmostEqual(float(eo_c[0, 0, 0]), 9.85, places=4)
            self.assertTrue(bool(sample["y_valid_mask"].all().item()))
            self.assertTrue(bool((sample["land_mask"] > 0.5).all().item()))

    def test_all_split_keeps_no_argo_rows_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArgoNetCDFGriddedPatchDataset(
                **_dataset_kwargs(Path(tmpdir)),
                split="all",
                require_argo_for_all=False,
            )

            self.assertEqual(
                [int(row["date"]) for row in dataset.rows], [20240102, 20240103]
            )
            no_argo_sample = dataset[1]
            self.assertFalse(bool(no_argo_sample["x_valid_mask"].any().item()))
            self.assertTrue(bool(no_argo_sample["y_valid_mask"].any().item()))
            self.assertTrue(
                torch.equal(no_argo_sample["x"], torch.zeros_like(no_argo_sample["x"]))
            )

    def test_val_year_assigns_only_that_year_to_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            kwargs = _dataset_kwargs(tmp_path)
            glorys_dir = Path(kwargs["glorys_dir"])
            ostia_dir = Path(kwargs["ostia_dir"])
            _write_glorys(glorys_dir, date_value=20180102, base=10.0)
            _write_glorys(glorys_dir, date_value=20190102, base=20.0)
            _write_ostia(ostia_dir, date_value=20180102, base_kelvin=275.0)
            _write_ostia(ostia_dir, date_value=20190102, base_kelvin=276.0)

            train_dataset = ArgoNetCDFGriddedPatchDataset(
                **kwargs,
                split="train",
                val_year=2018,
                require_argo_for_train=False,
            )
            val_dataset = ArgoNetCDFGriddedPatchDataset(
                **kwargs,
                split="val",
                val_year=2018,
                require_argo_for_val=False,
            )

            self.assertTrue(
                all((int(row["date"]) // 10000) != 2018 for row in train_dataset.rows)
            )
            self.assertEqual(
                {int(row["date"]) // 10000 for row in val_dataset.rows},
                {2018},
            )
            self.assertTrue(all(row["split"] == "train" for row in train_dataset.rows))
            self.assertTrue(all(row["split"] == "val" for row in val_dataset.rows))

    def test_land_mask_grid_stride_filter_and_overlapping_argo_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir, glorys_dir, ostia_dir, cache_dir = _make_sources(tmp_path)
            _write_argo_netcdf(
                argo_dir,
                latitudes=np.asarray([1.5], dtype=np.float64),
                longitudes=np.asarray([11.5], dtype=np.float64),
                temperatures=np.asarray([[10.0, 20.0]], dtype=np.float32),
                depths=np.asarray([[0.0, 10.0]], dtype=np.float32),
                dates=np.asarray([20240102], dtype=np.int64),
            )
            land_mask_path = _write_land_mask_geotiff(
                tmp_path / "land_mask.tif",
                np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.uint8),
            )

            dataset = ArgoNetCDFGriddedPatchDataset(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=None,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                patch_grid_source="land_mask",
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=1.0,
                temporal_window_days=1,
                val_year=2018,
                require_argo_for_train=True,
                return_info=True,
            )

            self.assertEqual([int(row["grid_x0"]) for row in dataset.rows], [0, 1])
            self.assertEqual(
                [int(row["argo_profile_count"]) for row in dataset.rows],
                [1, 1],
            )
            self.assertEqual(
                [float(row["land_fraction"]) for row in dataset.rows],
                [0.0, 0.5],
            )

            filtered = ArgoNetCDFGriddedPatchDataset(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=None,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                patch_grid_source="land_mask",
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=0.30,
                temporal_window_days=1,
                val_year=2018,
                require_argo_for_train=False,
                return_info=True,
            )

            self.assertEqual({int(row["grid_x0"]) for row in filtered.rows}, {0})
            self.assertTrue(
                all(float(row["land_fraction"]) <= 0.30 for row in filtered.rows)
            )

    def test_overlapping_land_mask_grid_requires_val_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir, glorys_dir, ostia_dir, cache_dir = _make_sources(tmp_path)
            land_mask_path = _write_land_mask_geotiff(
                tmp_path / "land_mask.tif",
                np.zeros((2, 3), dtype=np.uint8),
            )

            with self.assertRaisesRegex(ValueError, "Overlapping patch grids"):
                ArgoNetCDFGriddedPatchDataset(
                    argo_dir=argo_dir,
                    glorys_dir=glorys_dir,
                    ostia_dir=ostia_dir,
                    sealevel_dir=None,
                    metadata_cache_dir=cache_dir,
                    split="train",
                    tile_size=2,
                    resolution_deg=1.0,
                    patch_grid_source="land_mask",
                    land_mask_path=land_mask_path,
                    patch_stride=1,
                    max_land_fraction=1.0,
                    temporal_window_days=1,
                    val_year=None,
                    require_argo_for_train=False,
                )

    def test_force_include_region_keeps_relaxed_land_fraction_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir, glorys_dir, ostia_dir, cache_dir = _make_sources(tmp_path)
            land_mask_path = _write_land_mask_geotiff(
                tmp_path / "land_mask.tif",
                np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.uint8),
            )

            dataset = ArgoNetCDFGriddedPatchDataset(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=None,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                patch_grid_source="land_mask",
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=0.30,
                force_include_regions=[
                    {
                        "name": "test_region",
                        "lon_min": 11.5,
                        "lon_max": 12.5,
                        "lat_min": 0.5,
                        "lat_max": 1.5,
                        "max_land_fraction": 0.60,
                    }
                ],
                temporal_window_days=1,
                val_year=2018,
                require_argo_for_train=False,
                return_info=True,
            )

            forced_rows = [
                row for row in dataset.rows if bool(row.get("force_included", False))
            ]
            self.assertEqual({int(row["grid_x0"]) for row in forced_rows}, {1})
            self.assertEqual(
                {row["force_include_region"] for row in forced_rows},
                {"test_region"},
            )
            self.assertEqual(
                {float(row["land_fraction"]) for row in forced_rows},
                {0.5},
            )

    def test_active_netcdf_config_uses_land_mask_grid_defaults(self) -> None:
        with config_path("px_space", "data_ostia_argo_netcdf.yaml").open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = yaml.safe_load(f)

        grid = payload["dataset"]["grid"]
        self.assertEqual(grid["patch_grid_source"], "land_mask")
        self.assertEqual(grid["patch_stride"], 64)
        self.assertEqual(float(grid["max_land_fraction"]), 0.30)
        self.assertTrue(Path(grid["land_mask_path"]).exists())
        self.assertEqual(
            [region["name"] for region in grid["force_include_regions"]],
            ["mediterranean", "baltic", "red_sea", "great_lakes"],
        )

    def test_synthetic_mode_samples_sparse_x_from_glorys_y(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = ArgoNetCDFGriddedPatchDataset(
                **_dataset_kwargs(Path(tmpdir)),
                split="train",
                require_argo_for_train=True,
                synthetic_mode=True,
                synthetic_pixel_count=1,
            )

            self.assertEqual(
                [int(row["date"]) for row in dataset.rows], [20240102, 20240103]
            )
            sample = dataset[1]
            self.assertEqual(int(sample["x_valid_mask_1d"].sum().item()), 1)
            self.assertEqual(int(sample["x_valid_mask"].sum().item()), 2)
            self.assertEqual(sample["info"]["x_source"], "glorys_synthetic")
            self.assertEqual(sample["info"]["synthetic_pixel_count"], 1)

            x_c = temperature_normalize(mode="denorm", tensor=sample["x"])
            y_c = temperature_normalize(mode="denorm", tensor=sample["y"])
            valid = sample["x_valid_mask"]
            self.assertTrue(torch.allclose(x_c[valid], y_c[valid], atol=1e-5))

    def test_train_builder_wires_argo_netcdf_gridded_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir, glorys_dir, ostia_dir, cache_dir = _make_sources(tmp_path)
            config_path = tmp_path / "data.yaml"
            payload = {
                "dataset": {
                    "core": {
                        "dataset_variant": "argo_netcdf_gridded",
                        "argo_dir": str(argo_dir),
                        "glorys_dir": str(glorys_dir),
                        "ostia_dir": str(ostia_dir),
                        "sealevel_dir": None,
                        "metadata_cache_dir": str(cache_dir),
                    },
                    "grid": {
                        "tile_size": 2,
                        "resolution_deg": 1.0,
                        "patch_grid_source": "ostia_mask",
                        "invalid_threshold": 0.5,
                        "invalid_mask_flags": ["land"],
                    },
                    "sampling": {
                        "temporal_window_days": 1,
                        "argo_temp_var_name": "TEMP",
                        "argo_depth_var_name": "DEPH_CORRECTED",
                    },
                    "selection": {
                        "require_argo_for_train": True,
                        "require_argo_for_val": False,
                        "require_argo_for_all": False,
                    },
                    "synthetic": {"enabled": True, "pixel_count": 1},
                    "output": {"return_info": True, "return_coords": False},
                    "runtime": {"random_seed": 7, "cache_size": 2},
                },
                "split": {"val_fraction": 0.0, "val_year": None},
            }
            _write_yaml(config_path, payload)

            dataset = build_dataset(str(config_path), payload["dataset"], split="train")

            self.assertIsInstance(dataset, ArgoNetCDFGriddedPatchDataset)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)
            self.assertTrue(dataset.synthetic_mode)
            self.assertEqual(dataset.synthetic_pixel_count, 1)

    def test_global_inference_depth_axis_uses_dataset_property(self) -> None:
        class _DatasetWithDepthAxis:
            depth_axis_m = np.asarray([0.0, 10.0], dtype=np.float32)

        depth_axis = _load_glorys_depth_axis_m(
            _DatasetWithDepthAxis(),
            {},
            expected_size=2,
        )

        np.testing.assert_allclose(depth_axis, np.asarray([0.0, 10.0]))


if __name__ == "__main__":
    unittest.main()
