from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles import (
    NEAREST_STATUS,
    NEAREST_EDGE_STATUS,
    DatasetCache,
    nearest_timed_file,
    sample_spatial_value,
    sample_spatial_values_for_points,
    sample_temporal_values,
    sample_temporal_values_for_points,
    sample_temporal_value,
)
from data.dataset_creation.export_aligned_argo.source_files import (
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


if __name__ == "__main__":
    unittest.main()
