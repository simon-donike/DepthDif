from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import xarray as xr
import yaml

from data.dataset_argo_zarr_gridded import ArgoZarrGriddedPatchDataset
from data.dataset_creation.export_dataset_zarr import export_training_zarr_dataset
from data.dataset_creation.export_dataset_zarr.export_dataset_zarr import (
    DEFAULT_ARGO_DEPTH_VAR,
    DEFAULT_ARGO_VARS,
    DEFAULT_GLORYS_VARS,
    DEFAULT_OSTIA_VARS,
    DEFAULT_SEALEVEL_VARS,
    DEFAULT_TARGET_RESOLUTION_DEG,
    SOURCE_VARIABLE_CONFIG_PATH,
    load_zarr_source_variables,
)
from train import build_dataset
from utils.normalizations import temperature_normalize


def _days_since_1950(date_value: int) -> float:
    text = str(int(date_value))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float((day - np.datetime64("1950-01-01", "D")).astype(int))


def _write_argo_netcdf(root_dir: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (
                ("N_PROF",),
                np.asarray(
                    [_days_since_1950(20240102), _days_since_1950(20240102)],
                    dtype=np.float64,
                ),
            ),
            "LATITUDE": (("N_PROF",), np.asarray([1.5, 1.6], dtype=np.float64)),
            "LONGITUDE": (("N_PROF",), np.asarray([10.5, 10.6], dtype=np.float64)),
            "TEMP": (
                ("N_PROF", "N_LEVELS"),
                np.asarray([[10.0, 20.0], [14.0, 24.0]], dtype=np.float32),
            ),
            "PSAL_CORRECTED": (
                ("N_PROF", "N_LEVELS"),
                np.asarray([[35.0, 35.5], [36.0, 36.5]], dtype=np.float32),
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
    salinity = np.asarray(
        [
            [
                [[35.0, 35.1], [35.2, 35.3]],
                [[36.0, 36.1], [36.2, 36.3]],
            ]
        ],
        dtype=np.float32,
    )
    ssh = np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    ds = xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"), thetao),
            "so": (("time", "depth", "latitude", "longitude"), salinity),
            "zos": (("time", "latitude", "longitude"), ssh),
        },
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


def _write_ostia(
    root_dir: Path,
    *,
    date_value: int,
    base_kelvin: float,
    lat_values: np.ndarray | None = None,
    lon_values: np.ndarray | None = None,
) -> None:
    lat = (
        np.asarray([0.5, 1.5], dtype=np.float32)
        if lat_values is None
        else np.asarray(lat_values, dtype=np.float32)
    )
    lon = (
        np.asarray([10.5, 11.5], dtype=np.float32)
        if lon_values is None
        else np.asarray(lon_values, dtype=np.float32)
    )
    offsets = np.arange(int(lat.size) * int(lon.size), dtype=np.float32).reshape(
        int(lat.size),
        int(lon.size),
    )
    analysed_sst = (base_kelvin + 1.0 + offsets)[None, ...].astype(np.float32)
    mask = np.zeros((1, int(lat.size), int(lon.size)), dtype=np.int16)
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


def _write_sealevel(
    root_dir: Path,
    *,
    date_value: int,
    base: float = 1.0,
    lat_values: np.ndarray | None = None,
    lon_values: np.ndarray | None = None,
) -> None:
    lat = (
        np.asarray([0.5, 1.5], dtype=np.float32)
        if lat_values is None
        else np.asarray(lat_values, dtype=np.float32)
    )
    lon = (
        np.asarray([10.5, 11.5], dtype=np.float32)
        if lon_values is None
        else np.asarray(lon_values, dtype=np.float32)
    )
    offsets = np.arange(int(lat.size) * int(lon.size), dtype=np.float32).reshape(
        int(lat.size),
        int(lon.size),
    )
    adt = (base + 0.1 + (0.1 * offsets))[None, ...].astype(np.float32)
    ds = xr.Dataset(
        {"adt": (("time", "latitude", "longitude"), adt)},
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1950-01-01 00:00:00"
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"sealevel_{int(date_value)}.nc", engine="h5netcdf")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _make_raw_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    argo_dir = tmp_path / "en4_profiles"
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    sealevel_dir = tmp_path / "sealevel"
    _write_argo_netcdf(argo_dir)
    _write_glorys(glorys_dir, date_value=20240102, base=30.0)
    _write_glorys(glorys_dir, date_value=20240103, base=40.0)
    _write_ostia(ostia_dir, date_value=20240102, base_kelvin=280.0)
    _write_ostia(ostia_dir, date_value=20240103, base_kelvin=290.0)
    _write_sealevel(sealevel_dir, date_value=20240102)
    _write_sealevel(sealevel_dir, date_value=20240103)
    return argo_dir, glorys_dir, ostia_dir, sealevel_dir


def _export_zarr(tmp_path: Path) -> Path:
    argo_dir, glorys_dir, ostia_dir, sealevel_dir = _make_raw_sources(tmp_path)
    zarr_dir = tmp_path / "zarr_training"
    return export_training_zarr_dataset(
        argo_dir=argo_dir,
        glorys_dir=glorys_dir,
        ostia_dir=ostia_dir,
        sealevel_dir=sealevel_dir,
        output_dir=zarr_dir,
        start_date=20240102,
        end_date=20240103,
        chunk_time=1,
        chunk_profile=2,
        chunk_lat=2,
        chunk_lon=2,
        target_resolution_deg=1.0,
        overwrite=True,
    )


class TestExportDatasetZarr(unittest.TestCase):
    def test_source_variable_config_defines_export_defaults(self) -> None:
        source_variables = load_zarr_source_variables(SOURCE_VARIABLE_CONFIG_PATH)

        self.assertEqual(source_variables.ostia_vars, DEFAULT_OSTIA_VARS)
        self.assertEqual(source_variables.argo_vars, DEFAULT_ARGO_VARS)
        self.assertEqual(source_variables.argo_depth_var, DEFAULT_ARGO_DEPTH_VAR)
        self.assertEqual(source_variables.glorys_vars, DEFAULT_GLORYS_VARS)
        self.assertEqual(source_variables.sealevel_vars, DEFAULT_SEALEVEL_VARS)

    def test_export_resamples_raster_sources_to_target_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir = tmp_path / "en4_profiles"
            glorys_dir = tmp_path / "glorys"
            ostia_dir = tmp_path / "ostia"
            sealevel_dir = tmp_path / "sealevel"
            _write_argo_netcdf(argo_dir)
            _write_glorys(glorys_dir, date_value=20240102, base=30.0)
            _write_glorys(glorys_dir, date_value=20240103, base=40.0)
            source_lat = np.asarray([0.25, 0.75, 1.25, 1.75], dtype=np.float32)
            source_lon = np.asarray([10.25, 10.75, 11.25, 11.75], dtype=np.float32)
            _write_ostia(
                ostia_dir,
                date_value=20240102,
                base_kelvin=280.0,
                lat_values=source_lat,
                lon_values=source_lon,
            )
            _write_ostia(
                ostia_dir,
                date_value=20240103,
                base_kelvin=290.0,
                lat_values=source_lat,
                lon_values=source_lon,
            )
            _write_sealevel(
                sealevel_dir,
                date_value=20240102,
                lat_values=source_lat,
                lon_values=source_lon,
            )
            _write_sealevel(
                sealevel_dir,
                date_value=20240103,
                lat_values=source_lat,
                lon_values=source_lon,
            )
            zarr_dir = export_training_zarr_dataset(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_dir=tmp_path / "zarr_resampled",
                start_date=20240102,
                end_date=20240103,
                chunk_time=1,
                chunk_profile=2,
                chunk_lat=2,
                chunk_lon=2,
                target_resolution_deg=0.5,
                overwrite=True,
            )

            ostia = xr.open_zarr(zarr_dir / "ostia.zarr", consolidated=None)
            glorys = xr.open_zarr(zarr_dir / "glorys.zarr", consolidated=None)
            sealevel = xr.open_zarr(zarr_dir / "sealevel.zarr", consolidated=None)
            manifest = yaml.safe_load((zarr_dir / "manifest.yaml").read_text())

            self.assertEqual(manifest["raster_target_resolution_deg"], 0.5)
            self.assertEqual(manifest["raster_target_grid"], "glorys")
            np.testing.assert_allclose(ostia["lat"].values, [0.5, 1.0, 1.5])
            np.testing.assert_allclose(ostia["lon"].values, [10.5, 11.0, 11.5])
            np.testing.assert_allclose(glorys["latitude"].values, [0.5, 1.0, 1.5])
            np.testing.assert_allclose(
                sealevel["longitude"].values,
                [10.5, 11.0, 11.5],
            )
            np.testing.assert_allclose(ostia["lat"].values, glorys["latitude"].values)
            np.testing.assert_allclose(ostia["lon"].values, glorys["longitude"].values)
            np.testing.assert_allclose(
                sealevel["latitude"].values,
                glorys["latitude"].values,
            )
            np.testing.assert_allclose(
                sealevel["longitude"].values,
                glorys["longitude"].values,
            )
            self.assertEqual(int(ostia["analysed_sst"].sizes["lat"]), 3)
            self.assertEqual(int(glorys["thetao"].sizes["latitude"]), 3)
            self.assertEqual(int(sealevel["adt"].sizes["longitude"]), 3)
            self.assertAlmostEqual(
                float(ostia["analysed_sst"].isel(time=0, lat=1, lon=1).values),
                288.5,
                places=5,
            )
            ostia.close()
            glorys.close()
            sealevel.close()

    def test_export_aggregates_surface_products_to_glorys_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir = tmp_path / "en4_profiles"
            glorys_dir = tmp_path / "glorys"
            ostia_dir = tmp_path / "ostia"
            sealevel_dir = tmp_path / "sealevel"
            _write_argo_netcdf(argo_dir)
            _write_glorys(glorys_dir, date_value=20240108, base=30.0)
            for date_value, ostia_base, sealevel_base in (
                (20240105, 280.0, 1.0),
                (20240108, 290.0, 2.0),
                (20240111, 300.0, 3.0),
            ):
                _write_ostia(ostia_dir, date_value=date_value, base_kelvin=ostia_base)
                _write_sealevel(
                    sealevel_dir,
                    date_value=date_value,
                    base=sealevel_base,
                )

            zarr_dir = export_training_zarr_dataset(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_dir=tmp_path / "zarr_weekly",
                start_date=20240105,
                end_date=20240111,
                chunk_time=1,
                chunk_profile=2,
                chunk_lat=2,
                chunk_lon=2,
                target_resolution_deg=1.0,
                surface_aggregate_days=7,
                overwrite=True,
            )

            ostia = xr.open_zarr(zarr_dir / "ostia.zarr", consolidated=None)
            sealevel = xr.open_zarr(zarr_dir / "sealevel.zarr", consolidated=None)
            manifest = yaml.safe_load((zarr_dir / "manifest.yaml").read_text())

            np.testing.assert_array_equal(ostia["time"].values, [20240108])
            np.testing.assert_array_equal(sealevel["time"].values, [20240108])
            self.assertEqual(
                manifest["surface_temporal_aggregation"],
                {"target": "glorys", "window_days": 7},
            )
            self.assertAlmostEqual(
                float(ostia["analysed_sst"].isel(time=0, lat=0, lon=0).values),
                291.0,
                places=5,
            )
            self.assertAlmostEqual(
                float(sealevel["adt"].isel(time=0, latitude=0, longitude=0).values),
                2.1,
                places=5,
            )
            ostia.close()
            sealevel.close()

    def test_export_uses_packed_zarr_arrays_and_projected_argo_depths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zarr_dir = _export_zarr(tmp_path)
            argo = xr.open_zarr(zarr_dir / "argo.zarr", consolidated=None)

            self.assertEqual(argo["TEMP"].dims, ("N_PROF", "depth"))
            self.assertEqual(argo["PSAL_CORRECTED"].dims, ("N_PROF", "depth"))
            self.assertNotIn("DEPH_CORRECTED", argo)
            np.testing.assert_allclose(argo["depth"].values, [0.0, 10.0])
            np.testing.assert_allclose(
                argo["TEMP"].isel(N_PROF=0).values,
                [10.0, 20.0],
            )

            encoded = {
                "ostia_sst": json.loads(
                    (zarr_dir / "ostia.zarr" / "analysed_sst" / ".zarray").read_text()
                )["dtype"],
                "ostia_mask": json.loads(
                    (zarr_dir / "ostia.zarr" / "mask" / ".zarray").read_text()
                )["dtype"],
                "glorys_thetao": json.loads(
                    (zarr_dir / "glorys.zarr" / "thetao" / ".zarray").read_text()
                )["dtype"],
                "sealevel_adt": json.loads(
                    (zarr_dir / "sealevel.zarr" / "adt" / ".zarray").read_text()
                )["dtype"],
                "argo_temp": json.loads(
                    (zarr_dir / "argo.zarr" / "TEMP" / ".zarray").read_text()
                )["dtype"],
            }
            self.assertEqual(encoded["ostia_sst"], "<i2")
            self.assertEqual(encoded["ostia_mask"], "|i1")
            self.assertEqual(encoded["glorys_thetao"], "<i2")
            self.assertEqual(encoded["sealevel_adt"], "<i2")
            self.assertEqual(encoded["argo_temp"], "<i2")
            argo.close()

    def test_default_raster_target_resolution_is_training_resolution(self) -> None:
        self.assertEqual(DEFAULT_TARGET_RESOLUTION_DEG, 0.1)

    def test_exported_zarr_dataset_reads_training_contract_and_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zarr_dir = _export_zarr(tmp_path)
            dataset = ArgoZarrGriddedPatchDataset(
                zarr_root_dir=zarr_dir,
                metadata_cache_dir=tmp_path / "cache",
                tile_size=2,
                resolution_deg=1.0,
                temporal_window_days=1,
                invalid_threshold=0.5,
                val_fraction=0.0,
                random_seed=7,
                return_info=True,
                return_coords=True,
                return_modalities=True,
                require_argo_for_train=True,
                split="train",
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]

            self.assertEqual(sample["eo"].shape, (1, 2, 2))
            self.assertEqual(sample["x"].shape, (2, 2, 2))
            self.assertEqual(sample["y"].shape, (2, 2, 2))
            self.assertEqual(sample["date"], 20240102)
            self.assertIn("PSAL_CORRECTED", sample["modalities"]["argo"])
            self.assertIn("so", sample["modalities"]["glorys"])
            self.assertIn("zos", sample["modalities"]["glorys"])
            self.assertIn("adt", sample["modalities"]["sealevel"])

            x_c = temperature_normalize(mode="denorm", tensor=sample["x"])
            self.assertTrue(
                torch.allclose(
                    x_c[:, 0, 0],
                    torch.tensor([12.0, 22.0], dtype=torch.float32),
                    atol=1e-5,
                )
            )
            salinity = sample["modalities"]["argo"]["PSAL_CORRECTED"]
            self.assertTrue(
                torch.allclose(
                    salinity[:, 0, 0],
                    torch.tensor([35.5, 36.0], dtype=torch.float32),
                    atol=1e-5,
                )
            )
            self.assertEqual(sample["modalities"]["glorys"]["zos"].shape, (1, 2, 2))
            self.assertEqual(sample["modalities"]["sealevel"]["adt"].shape, (1, 2, 2))

    def test_train_builder_wires_zarr_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            zarr_dir = _export_zarr(tmp_path)
            config_path = tmp_path / "data.yaml"
            payload = {
                "dataset": {
                    "core": {
                        "dataset_variant": "argo_zarr_gridded",
                        "zarr_root_dir": str(zarr_dir),
                        "metadata_cache_dir": str(tmp_path / "cache"),
                    },
                    "grid": {
                        "tile_size": 2,
                        "resolution_deg": 1.0,
                        "invalid_threshold": 0.5,
                        "invalid_mask_flags": ["land"],
                    },
                    "sampling": {
                        "temporal_window_days": 1,
                        "argo_depth_var_name": "DEPH_CORRECTED",
                    },
                    "selection": {
                        "require_argo_for_train": True,
                        "require_argo_for_val": False,
                        "require_argo_for_all": False,
                    },
                    "output": {
                        "return_info": True,
                        "return_coords": False,
                        "return_modalities": True,
                    },
                    "runtime": {"random_seed": 7},
                },
                "split": {"val_fraction": 0.0, "val_year": None},
            }
            _write_yaml(config_path, payload)

            dataset = build_dataset(str(config_path), payload["dataset"], split="train")

            self.assertIsInstance(dataset, ArgoZarrGriddedPatchDataset)
            self.assertTrue(dataset.return_modalities)
            self.assertFalse(dataset.return_coords)


if __name__ == "__main__":
    unittest.main()
