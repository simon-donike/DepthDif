import tempfile
from pathlib import Path
import unittest

import numpy as np
import xarray as xr

from data.dataset_creation.b_export_enriched_argo_profiles import (
    GLORYS_2D_VARS,
    GLORYS_3D_VARS,
    INTERPOLATED_STATUS,
    NEAREST_EDGE_STATUS,
    OSTIA_VARS,
    SEALEVEL_VARS,
    TimedFile,
    _sanitize_metadata_value,
    bracket_timed_files,
    export_enriched_argo_profiles,
    project_argo_profile_to_glorys_depths,
    scan_timed_files,
)
sanitize_metadata_value = _sanitize_metadata_value


def _write_argo(
    path: Path,
    *,
    juld: float = 21916.0,
    engine: str = "h5netcdf",
    include_qc: bool = True,
) -> None:
    data_vars = {
        "JULD": (("N_PROF",), np.asarray([juld], dtype=np.float64)),
        "LATITUDE": (("N_PROF",), np.asarray([0.5], dtype=np.float64)),
        "LONGITUDE": (("N_PROF",), np.asarray([10.5], dtype=np.float64)),
        "DEPH_CORRECTED": (
            ("N_PROF", "N_LEVELS"),
            np.asarray([[0.0, 10.0, 20.0]], dtype=np.float32),
        ),
        "TEMP": (
            ("N_PROF", "N_LEVELS"),
            np.asarray([[1.0, 3.0, 5.0]], dtype=np.float32),
        ),
        "POTM_CORRECTED": (
            ("N_PROF", "N_LEVELS"),
            np.asarray([[2.0, 4.0, 6.0]], dtype=np.float32),
        ),
        "PSAL_CORRECTED": (
            ("N_PROF", "N_LEVELS"),
            np.asarray([[34.0, 35.0, 36.0]], dtype=np.float32),
        ),
    }
    if include_qc:
        qc_dims = ("N_PROF", "N_LEVELS")
        data_vars.update(
            {
                "JULD_QC": (("N_PROF",), np.asarray([b"1"], dtype="S1")),
                "POSITION_QC": (("N_PROF",), np.asarray([b"2"], dtype="S1")),
                "PROFILE_DEPH_QC": (("N_PROF",), np.asarray([b"1"], dtype="S1")),
                "PROFILE_POTM_QC": (("N_PROF",), np.asarray([b"3"], dtype="S1")),
                "PROFILE_PSAL_QC": (("N_PROF",), np.asarray([b"4"], dtype="S1")),
                "DEPH_CORRECTED_QC": (
                    qc_dims,
                    np.asarray([[b"1", b"2", b"3"]], dtype="S1"),
                ),
                "POTM_CORRECTED_QC": (
                    qc_dims,
                    np.asarray([[b"1", b"1", b"3"]], dtype="S1"),
                ),
                "PSAL_CORRECTED_QC": (
                    qc_dims,
                    np.asarray([[b"2", b"2", b"4"]], dtype="S1"),
                ),
            }
        )
    ds = xr.Dataset(data_vars)
    ds["JULD"].attrs["units"] = "days since 1950-01-01 00:00:00 utc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine=engine)


def _write_glorys(path: Path, *, day: float, base: float) -> None:
    depth = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    lat = np.asarray([0.0, 1.0], dtype=np.float32)
    lon = np.asarray([10.0, 11.0], dtype=np.float32)
    data_vars = {}
    for offset, name in enumerate(GLORYS_3D_VARS):
        values = np.full((1, 3, 2, 2), base + offset, dtype=np.float32)
        data_vars[name] = (("time", "depth", "latitude", "longitude"), values)
    for offset, name in enumerate(GLORYS_2D_VARS):
        values = np.full((1, 2, 2), base + 10.0 + offset, dtype=np.float32)
        data_vars[name] = (("time", "latitude", "longitude"), values)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (("time",), np.asarray([day * 24.0], dtype=np.float64)),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds["time"].attrs["units"] = "hours since 1950-01-01 00:00:00"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")


def _write_ostia(path: Path, *, day: float, base: float) -> None:
    lat = np.asarray([0.0, 1.0], dtype=np.float32)
    lon = np.asarray([10.0, 11.0], dtype=np.float32)
    data_vars = {}
    for offset, name in enumerate(OSTIA_VARS):
        values = np.full((1, 2, 2), base + offset, dtype=np.float32)
        data_vars[name] = (("time", "lat", "lon"), values)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (("time",), np.asarray([day], dtype=np.float64)),
            "lat": lat,
            "lon": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1950-01-01 00:00:00"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")


def _write_sealevel(path: Path, *, day: float, base: float) -> None:
    lat = np.asarray([0.0, 1.0], dtype=np.float32)
    lon = np.asarray([10.0, 11.0], dtype=np.float32)
    data_vars = {}
    for offset, name in enumerate(SEALEVEL_VARS):
        values = np.full((1, 2, 2), base + offset, dtype=np.float32)
        if name == "tpa_correction":
            data_vars[name] = (("time",), np.asarray([base + offset], dtype=np.float32))
        else:
            data_vars[name] = (("time", "latitude", "longitude"), values)
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": (("time",), np.asarray([day], dtype=np.float64)),
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds["time"].attrs["units"] = "days since 1950-01-01 00:00:00"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")


class EnrichedArgoExportTests(unittest.TestCase):
    def test_project_argo_profile_reuses_glorys_depth_semantics(self) -> None:
        projected = project_argo_profile_to_glorys_depths(
            np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
            np.asarray([0.0, 10.0, 20.0], dtype=np.float32),
            np.asarray([5.0, 10.0, 25.0], dtype=np.float32),
        )

        self.assertAlmostEqual(float(projected[0]), 2.0, places=6)
        self.assertAlmostEqual(float(projected[1]), 3.0, places=6)
        self.assertTrue(np.isnan(projected[2]))

    def test_bracket_timed_files_interpolates_and_uses_edges(self) -> None:
        index = [
            TimedFile(Path("a.nc"), 10.0),
            TimedFile(Path("b.nc"), 20.0),
        ]

        before, after, weight, status = bracket_timed_files(index, 15.0)
        self.assertEqual(before.path.name, "a.nc")
        self.assertEqual(after.path.name, "b.nc")
        self.assertAlmostEqual(weight, 0.5)
        self.assertEqual(status, INTERPOLATED_STATUS)

        before, after, weight, status = bracket_timed_files(index, 5.0)
        self.assertEqual(before.path.name, "a.nc")
        self.assertEqual(after.path.name, "a.nc")
        self.assertAlmostEqual(weight, 0.0)
        self.assertEqual(status, NEAREST_EDGE_STATUS)

    def test_scan_timed_files_reads_raw_netcdf_without_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_ostia(root / "20100101120000-test.nc", day=21915.0, base=1.0)

            index = scan_timed_files(root)

            self.assertEqual(len(index), 1)
            self.assertAlmostEqual(index[0].day, 21915.0)

    def test_metadata_sanitizer_removes_absolute_paths_but_keeps_urls(self) -> None:
        sanitized = sanitize_metadata_value(
            "source=/data1/datasets/depth_v2/file.nc "
            "also /tmp/other.nc https://example.test/path/file.nc"
        )

        self.assertEqual(
            sanitized,
            "source=file.nc also other.nc https://example.test/path/file.nc",
        )

    def test_export_enriched_profiles_writes_expected_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            argo_dir = root / "en4_profiles"
            glorys_dir = root / "glorys_weekly"
            ostia_dir = root / "ostia"
            sealevel_dir = root / "sealevel_daily"
            output_zarr = root / "enriched.zarr"

            _write_argo(argo_dir / "EN.4.2.2.f.profiles.g10.201001.nc")
            _write_glorys(glorys_dir / "glorys_20100101.nc", day=21915.0, base=10.0)
            _write_glorys(glorys_dir / "glorys_20100103.nc", day=21917.0, base=14.0)
            _write_ostia(ostia_dir / "20100101120000-ostia.nc", day=21915.0, base=20.0)
            _write_ostia(ostia_dir / "20100103120000-ostia.nc", day=21917.0, base=24.0)
            _write_sealevel(
                sealevel_dir / "dt_global_allsat_phy_l4_20100101_20241016.nc",
                day=21915.0,
                base=30.0,
            )
            _write_sealevel(
                sealevel_dir / "dt_global_allsat_phy_l4_20100103_20241016.nc",
                day=21917.0,
                base=34.0,
            )

            export_enriched_argo_profiles(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_zarr=output_zarr,
                batch_size=1,
                overwrite=True,
            )

            ds = xr.open_zarr(output_zarr)
            self.assertEqual(ds.sizes["profile"], 1)
            self.assertEqual(ds.sizes["glorys_depth"], 3)
            self.assertIn("sealevel_adt", ds)
            self.assertIn("glorys_zos", ds)
            self.assertEqual(
                ds["profile_source_file"].values[0],
                "EN.4.2.2.f.profiles.g10.201001.nc",
            )
            self.assertFalse(str(ds["profile_source_file"].values[0]).startswith("/"))
            self.assertIn("source_metadata", ds.attrs)
            self.assertEqual(
                ds.attrs["source_metadata"]["sealevel"]["representative_file"],
                "dt_global_allsat_phy_l4_20100101_20241016.nc",
            )
            self.assertIn("source_attrs", ds["sealevel_adt"].attrs)
            self.assertEqual(ds["glorys_depth"].attrs["units"], "m")
            self.assertEqual(ds["profile_juld"].attrs["value_units"], "days")
            self.assertEqual(
                ds["profile_juld"].attrs["reference_epoch"],
                "1950-01-01T00:00:00Z",
            )
            self.assertEqual(
                ds["profile_juld"].attrs["source_units"],
                "days since 1950-01-01 00:00:00 utc",
            )
            self.assertIn("argo_potm_qc_on_glorys_depth", ds)
            self.assertIn("argo_juld_qc", ds)
            np.testing.assert_array_equal(
                ds["argo_depth_qc_on_glorys_depth"].values[0],
                np.asarray([1, 2, 3], dtype=np.int8),
            )
            np.testing.assert_array_equal(
                ds["argo_potm_qc_on_glorys_depth"].values[0],
                np.asarray([1, 1, 3], dtype=np.int8),
            )
            np.testing.assert_array_equal(
                ds["argo_temp_qc_on_glorys_depth"].values[0],
                np.asarray([-1, -1, -1], dtype=np.int8),
            )
            self.assertEqual(int(ds["argo_juld_qc"].values[0]), 1)
            self.assertEqual(int(ds["argo_position_qc"].values[0]), 2)
            self.assertEqual(int(ds["argo_profile_psal_qc"].values[0]), 4)
            np.testing.assert_allclose(
                ds["argo_temp_on_glorys_depth"].values[0],
                np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
            )
            np.testing.assert_allclose(
                ds["glorys_thetao"].values[0],
                np.asarray([12.0, 12.0, 12.0], dtype=np.float32),
            )
            self.assertAlmostEqual(float(ds["ostia_analysed_sst"].values[0]), 22.0)
            self.assertAlmostEqual(float(ds["sealevel_adt"].values[0]), 38.0)
            self.assertEqual(int(ds["glorys_temporal_status"].values[0]), 0)
            self.assertEqual(int(ds["ostia_temporal_status"].values[0]), 0)
            self.assertEqual(int(ds["sealevel_temporal_status"].values[0]), 0)
            ds.close()

    def test_export_reads_netcdf3_argo_and_filters_unneeded_months(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            argo_dir = root / "en4_profiles"
            glorys_dir = root / "glorys_weekly"
            ostia_dir = root / "ostia"
            sealevel_dir = root / "sealevel_daily"
            output_zarr = root / "enriched.zarr"

            _write_argo(
                argo_dir / "EN.4.2.2.f.profiles.g10.201001.nc",
                engine="scipy",
                include_qc=False,
            )
            # The date filter should exclude this later month before any NetCDF open.
            (argo_dir / "EN.4.2.2.f.profiles.g10.201002.nc").write_bytes(b"not-netcdf")
            _write_glorys(glorys_dir / "glorys_20100101.nc", day=21915.0, base=10.0)
            _write_glorys(glorys_dir / "glorys_20100103.nc", day=21917.0, base=14.0)
            _write_ostia(ostia_dir / "20100101120000-ostia.nc", day=21915.0, base=20.0)
            _write_ostia(ostia_dir / "20100103120000-ostia.nc", day=21917.0, base=24.0)
            _write_sealevel(
                sealevel_dir / "dt_global_allsat_phy_l4_20100101_20241016.nc",
                day=21915.0,
                base=30.0,
            )
            _write_sealevel(
                sealevel_dir / "dt_global_allsat_phy_l4_20100103_20241016.nc",
                day=21917.0,
                base=34.0,
            )

            export_enriched_argo_profiles(
                argo_dir=argo_dir,
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                output_zarr=output_zarr,
                start_date=20100101,
                end_date=20100131,
                batch_size=1,
                overwrite=True,
            )

            ds = xr.open_zarr(output_zarr)
            self.assertEqual(ds.sizes["profile"], 1)
            self.assertEqual(int(ds["profile_date"].values[0]), 20100102)
            ds.close()


if __name__ == "__main__":
    unittest.main()
