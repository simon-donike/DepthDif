from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles import (
    NEAREST_STATUS,
    NEAREST_EDGE_STATUS,
    DatasetCache,
    export_enriched_argo_profiles,
    nearest_timed_file,
    sample_spatial_value,
    sample_spatial_values_for_points,
    sample_temporal_values,
    sample_temporal_values_for_points,
    sample_temporal_value,
)
from data.dataset_creation.export_aligned_argo.source_files import (
    GLORYS_2D_VARS,
    GLORYS_3D_VARS,
    OSTIA_VARS,
    SEALEVEL_VARS,
    TimedFile,
    date_to_days_since_1950,
)


def _write_point_source(path: Path, value: float) -> None:
    ds = xr.Dataset(
        {
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.full((1, 1, 2, 2), value, dtype=np.float32),
            ),
        },
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "depth": np.asarray([0.0], dtype=np.float32),
            "latitude": np.asarray([1.0, 2.0], dtype=np.float32),
            "longitude": np.asarray([2.0, 3.0], dtype=np.float32),
        },
    )
    ds.to_netcdf(path, engine="h5netcdf")


def _linear_point_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.asarray([[[[0.0, 10.0], [20.0, 30.0]]]], dtype=np.float32),
            ),
        },
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "depth": np.asarray([0.0], dtype=np.float32),
            "latitude": np.asarray([0.0, 1.0], dtype=np.float32),
            "longitude": np.asarray([10.0, 11.0], dtype=np.float32),
        },
    )


def _write_enriched_argo_source(root_dir: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "JULD": (
                ("N_PROF",),
                np.asarray(
                    [
                        date_to_days_since_1950(20240102),
                        date_to_days_since_1950(20240103),
                        date_to_days_since_1950(20240102),
                        date_to_days_since_1950(20240103),
                    ],
                    dtype=np.float64,
                ),
            ),
            "LATITUDE": (("N_PROF",), np.asarray([1.25, 1.75, 1.25, 1.75], dtype=np.float64)),
            "LONGITUDE": (("N_PROF",), np.asarray([10.25, 10.75, 10.75, 10.25], dtype=np.float64)),
            "TEMP": (
                ("N_PROF", "N_LEVELS"),
                np.asarray(
                    [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
                    dtype=np.float32,
                ),
            ),
            "POTM_CORRECTED": (
                ("N_PROF", "N_LEVELS"),
                np.asarray(
                    [[9.0, 19.0], [10.0, 20.0], [11.0, 21.0], [12.0, 22.0]],
                    dtype=np.float32,
                ),
            ),
            "PSAL_CORRECTED": (
                ("N_PROF", "N_LEVELS"),
                np.asarray(
                    [[35.0, 35.5], [36.0, 36.5], [37.0, 37.5], [38.0, 38.5]],
                    dtype=np.float32,
                ),
            ),
            "DEPH_CORRECTED": (
                ("N_PROF", "N_LEVELS"),
                np.asarray([[0.0, 10.0]] * 4, dtype=np.float32),
            ),
        },
        coords={
            "N_PROF": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "N_LEVELS": np.asarray([0, 1], dtype=np.int64),
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / "EN.4.2.2.f.profiles.g10.202401.nc", engine="h5netcdf")


def _write_enriched_glorys_source(root_dir: Path, *, date_value: int, base: float) -> None:
    lat = np.asarray([1.0, 2.0], dtype=np.float32)
    lon = np.asarray([10.0, 11.0], dtype=np.float32)
    depth = np.asarray([0.0, 10.0], dtype=np.float32)
    data_vars = {}
    for offset, name in enumerate(GLORYS_3D_VARS):
        data_vars[name] = (
            ("time", "depth", "latitude", "longitude"),
            np.full((1, 2, 2, 2), base + offset, dtype=np.float32),
        )
    for offset, name in enumerate(GLORYS_2D_VARS):
        data_vars[name] = (
            ("time", "latitude", "longitude"),
            np.full((1, 2, 2), base + 10.0 + offset, dtype=np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"glorys_{int(date_value)}.nc", engine="h5netcdf")


def _write_enriched_surface_source(
    root_dir: Path,
    *,
    filename: str,
    variables: tuple[str, ...],
    base: float,
    lat_name: str,
    lon_name: str,
) -> None:
    lat = np.asarray([1.0, 2.0], dtype=np.float32)
    lon = np.asarray([10.0, 11.0], dtype=np.float32)
    data_vars = {}
    for offset, name in enumerate(variables):
        data_vars[name] = (
            ("time", lat_name, lon_name),
            np.full((1, 2, 2), base + offset, dtype=np.float32),
        )
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            lat_name: lat,
            lon_name: lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / filename, engine="h5netcdf")


def _make_enriched_export_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    argo_dir = tmp_path / "en4_profiles"
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    sealevel_dir = tmp_path / "sealevel"
    _write_enriched_argo_source(argo_dir)
    for date_value, base in ((20240102, 100.0), (20240103, 200.0)):
        _write_enriched_glorys_source(glorys_dir, date_value=date_value, base=base)
        _write_enriched_surface_source(
            ostia_dir,
            filename=f"{date_value}120000-ostia.nc",
            variables=OSTIA_VARS,
            base=base + 1000.0,
            lat_name="lat",
            lon_name="lon",
        )
        _write_enriched_surface_source(
            sealevel_dir,
            filename=f"sealevel_{date_value}.nc",
            variables=SEALEVEL_VARS,
            base=base + 2000.0,
            lat_name="latitude",
            lon_name="longitude",
        )
    return argo_dir, glorys_dir, ostia_dir, sealevel_dir


class TestEnrichedArgoExport(unittest.TestCase):
    def test_sample_spatial_value_uses_bilinear_point_sample(self) -> None:
        value = sample_spatial_value(
            _linear_point_dataset(),
            "thetao",
            lat=0.25,
            lon=10.5,
        )

        self.assertTrue(np.allclose(value, np.asarray([10.0], dtype=np.float32)))

    def test_sample_spatial_values_for_points_uses_bilinear_point_samples(self) -> None:
        values = sample_spatial_values_for_points(
            _linear_point_dataset(),
            ("thetao",),
            lat=np.asarray([0.25, 0.75], dtype=np.float64),
            lon=np.asarray([10.5, 10.5], dtype=np.float64),
        )

        expected = np.asarray([[10.0], [20.0]], dtype=np.float32)
        self.assertTrue(np.allclose(values["thetao"], expected))

    def test_sample_temporal_value_uses_nearest_file_without_blending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            before = tmp_path / "source_20240101.nc"
            after = tmp_path / "source_20240108.nc"
            _write_point_source(before, 10.0)
            _write_point_source(after, 80.0)
            index = [
                TimedFile(before, date_to_days_since_1950(20240101)),
                TimedFile(after, date_to_days_since_1950(20240108)),
            ]
            cache = DatasetCache(max_open=2)
            try:
                value, status = sample_temporal_value(
                    index,
                    cache,
                    "thetao",
                    target_day=date_to_days_since_1950(20240104),
                    lat=1.5,
                    lon=2.5,
                )
            finally:
                cache.close()

            # January 4 is closer to January 1 than January 8, so this must keep
            # the first file value instead of temporal interpolation.
            self.assertEqual(status, NEAREST_STATUS)
            self.assertTrue(np.allclose(value, np.asarray([10.0], dtype=np.float32)))

    def test_sample_temporal_values_samples_group_from_one_nearest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            before = tmp_path / "source_20240101.nc"
            after = tmp_path / "source_20240108.nc"
            _write_point_source(before, 10.0)
            _write_point_source(after, 80.0)
            index = [
                TimedFile(before, date_to_days_since_1950(20240101)),
                TimedFile(after, date_to_days_since_1950(20240108)),
            ]
            cache = DatasetCache(max_open=2)
            try:
                values, status = sample_temporal_values(
                    index,
                    cache,
                    ("thetao",),
                    target_day=date_to_days_since_1950(20240104),
                    lat=1.5,
                    lon=2.5,
                )
            finally:
                cache.close()

            # Grouped sampling uses the same nearest-time rule as scalar sampling.
            self.assertEqual(status, NEAREST_STATUS)
            self.assertTrue(
                np.allclose(values["thetao"], np.asarray([10.0], dtype=np.float32))
            )

    def test_sample_temporal_values_for_points_samples_one_nearest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            before = tmp_path / "source_20240101.nc"
            after = tmp_path / "source_20240108.nc"
            _write_point_source(before, 10.0)
            _write_point_source(after, 80.0)
            index = [
                TimedFile(before, date_to_days_since_1950(20240101)),
                TimedFile(after, date_to_days_since_1950(20240108)),
            ]
            cache = DatasetCache(max_open=2)
            try:
                values, status = sample_temporal_values_for_points(
                    index,
                    cache,
                    ("thetao",),
                    target_day=date_to_days_since_1950(20240104),
                    lat=np.asarray([1.25, 1.75], dtype=np.float64),
                    lon=np.asarray([2.25, 2.75], dtype=np.float64),
                )
            finally:
                cache.close()

            expected = np.asarray([[10.0], [10.0]], dtype=np.float32)
            self.assertEqual(status, NEAREST_STATUS)
            self.assertTrue(np.allclose(values["thetao"], expected))

    def test_nearest_timed_file_reports_edge_status_outside_source_range(self) -> None:
        index = [
            TimedFile(Path("source_20240101.nc"), date_to_days_since_1950(20240101)),
            TimedFile(Path("source_20240108.nc"), date_to_days_since_1950(20240108)),
        ]

        item, status = nearest_timed_file(index, date_to_days_since_1950(20231231))

        self.assertEqual(item, index[0])
        self.assertEqual(status, NEAREST_EDGE_STATUS)

    def test_export_rejects_invalid_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "workers"):
                export_enriched_argo_profiles(
                    argo_dir=tmp_path / "argo",
                    glorys_dir=tmp_path / "glorys",
                    ostia_dir=tmp_path / "ostia",
                    sealevel_dir=tmp_path / "sealevel",
                    output_zarr=tmp_path / "out.zarr",
                    workers=0,
                )

    def test_parallel_export_matches_serial_output_order_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            argo_dir, glorys_dir, ostia_dir, sealevel_dir = _make_enriched_export_sources(
                tmp_path
            )
            serial_path = export_enriched_argo_profiles(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_zarr=tmp_path / "serial.zarr",
                start_date=20240102,
                end_date=20240103,
                batch_size=2,
                cache_size=2,
                workers=1,
                overwrite=True,
            )
            parallel_path = export_enriched_argo_profiles(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_zarr=tmp_path / "parallel.zarr",
                start_date=20240102,
                end_date=20240103,
                batch_size=2,
                cache_size=2,
                workers=2,
                overwrite=True,
            )

            serial = xr.open_zarr(serial_path, consolidated=None)
            parallel = xr.open_zarr(parallel_path, consolidated=None)
            try:
                self.assertEqual(parallel.attrs["workers"], 2)
                self.assertEqual(parallel.attrs["cache_size_per_worker"], 2)
                for name in (
                    "profile_source_file",
                    "profile_idx",
                    "profile_date",
                    "glorys_temporal_status",
                    "ostia_temporal_status",
                    "sealevel_temporal_status",
                ):
                    np.testing.assert_array_equal(serial[name].values, parallel[name].values)
                for name in (
                    "latitude",
                    "longitude",
                    "argo_temp_on_glorys_depth",
                    "argo_potm_on_glorys_depth",
                    "argo_psal_on_glorys_depth",
                    "glorys_thetao",
                    "ostia_analysed_sst",
                    "sealevel_adt",
                ):
                    np.testing.assert_allclose(
                        serial[name].values,
                        parallel[name].values,
                        equal_nan=True,
                    )
            finally:
                serial.close()
                parallel.close()


if __name__ == "__main__":
    unittest.main()
