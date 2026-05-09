from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import xarray as xr
import yaml

from data.dataset_argo_netcdf_gridded import ArgoNetCDFGriddedPatchDataset
from inference.export_global import _load_glorys_depth_axis_m
from train import build_dataset
from utils.normalizations import temperature_normalize


def _days_since_1950(date_value: int) -> float:
    text = str(int(date_value))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float((day - np.datetime64("1950-01-01", "D")).astype(int))


def _write_argo_netcdf(root_dir: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (("N_PROF",), np.asarray([_days_since_1950(20240102), _days_since_1950(20240102)], dtype=np.float64)),
            "LATITUDE": (("N_PROF",), np.asarray([1.5, 1.6], dtype=np.float64)),
            "LONGITUDE": (("N_PROF",), np.asarray([10.5, 10.6], dtype=np.float64)),
            "TEMP": (
                ("N_PROF", "N_LEVELS"),
                np.asarray([[10.0, 20.0], [14.0, 24.0]], dtype=np.float32),
            ),
            "DEPH_CORRECTED": (
                ("N_PROF", "N_LEVELS"),
                np.asarray([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32),
            ),
        },
        coords={
            "N_PROF": np.asarray([0, 1], dtype=np.int64),
            "N_LEVELS": np.asarray([0, 1], dtype=np.int64),
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
        [[[base_kelvin + 1.0, base_kelvin + 2.0], [base_kelvin + 3.0, base_kelvin + 4.0]]],
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
                synthetic_mode=False,
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
                synthetic_mode=False,
            )

            self.assertEqual([int(row["date"]) for row in dataset.rows], [20240102, 20240103])
            no_argo_sample = dataset[1]
            self.assertFalse(bool(no_argo_sample["x_valid_mask"].any().item()))
            self.assertTrue(bool(no_argo_sample["y_valid_mask"].any().item()))
            self.assertTrue(torch.equal(no_argo_sample["x"], torch.zeros_like(no_argo_sample["x"])))

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
                    "output": {"return_info": True, "return_coords": False},
                    "synthetic": {"enabled": False, "pixel_count": 1},
                    "runtime": {"random_seed": 7, "cache_size": 2},
                },
                "split": {"val_fraction": 0.0},
            }
            _write_yaml(config_path, payload)

            dataset = build_dataset(str(config_path), payload["dataset"], split="train")

            self.assertIsInstance(dataset, ArgoNetCDFGriddedPatchDataset)
            self.assertFalse(dataset.synthetic_mode)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)

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
