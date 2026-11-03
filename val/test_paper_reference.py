"""Numerical tests for native reference and CUGN validation stages."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from val.cugn import convert_file, import_cugn
from val.techniques.reference import run as reference_run


class PaperReferenceTests(unittest.TestCase):
    """Protect reference provenance and external section conversion."""

    def _points(self):
        return pd.DataFrame(
            [
                {
                    "point_id": "p:temperature:10",
                    "profile_id": "p",
                    "variable": "temperature",
                    "observation": 12.0,
                    "glorys": 11.0,
                    "climatology": 10.0,
                    "depth_m": 10.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "observation_time": "2016-01-01T06:00:00Z",
                    "target_time": "2016-01-01T00:00:00Z",
                }
            ]
        )

    def test_reference_without_native_is_explicit_partial(self):
        """A missing native source never masquerades as a native comparison."""
        with tempfile.TemporaryDirectory() as temporary:
            result = reference_run({"points": self._points()}, Path(temporary), {})
            self.assertEqual(result["status"], "partial")
            status = pd.read_csv(Path(temporary) / "native_reference_status.csv")
            self.assertIn("not configured", status.reason.iloc[0])

    def test_reference_samples_native_and_preserves_precision_audit(self):
        """A synthetic floating point source is sampled at the requested depth."""
        with tempfile.TemporaryDirectory() as temporary:
            native_path = Path(temporary) / "native.nc"
            dataset = xr.Dataset(
                {
                    "thetao": (
                        ("time", "depth", "latitude", "longitude"),
                        np.array([[[[12.0]], [[13.0]]]]),
                    )
                },
                coords={
                    "time": pd.to_datetime(["2016-01-01"]),
                    "depth": [0.0, 20.0],
                    "latitude": [0.0],
                    "longitude": [0.0],
                },
            )
            dataset["time_bnds"] = (
                ("time", "bounds"),
                np.array(
                    [
                        [
                            np.datetime64("2015-12-31T12:00"),
                            np.datetime64("2016-01-01T12:00"),
                        ]
                    ]
                ),
            )
            dataset["time"].attrs["bounds"] = "time_bnds"
            dataset.to_netcdf(native_path)
            result = reference_run(
                {"points": self._points()},
                Path(temporary) / "out",
                {"native_glorys": str(native_path)},
            )
            self.assertEqual(result["status"], "complete")
            values = pd.read_csv(
                Path(temporary) / "out" / "native_reference_values.csv"
            )
            self.assertAlmostEqual(values.glorys_native.iloc[0], 12.5)
            self.assertIn("2015-12-31", values.native_time_bound_start.iloc[0])

    def test_valid_bilinear_rejects_dry_corner_and_records_time_bounds(self):
        """Bilinear sampling never averages across a dry corner."""
        with tempfile.TemporaryDirectory() as temporary:
            native_path = Path(temporary) / "native_bounds.nc"
            values = np.array([[[[12.0, 12.0], [12.0, np.nan]]]])
            dataset = xr.Dataset(
                {"thetao": (("time", "depth", "latitude", "longitude"), values)},
                coords={
                    "time": pd.to_datetime(["2016-01-01"]),
                    "depth": [10.0],
                    "latitude": [0.0, 1.0],
                    "longitude": [0.0, 1.0],
                },
            )
            dataset["time_bnds"] = (
                ("time", "bounds"),
                np.array(
                    [
                        [
                            np.datetime64("2015-12-31T12:00"),
                            np.datetime64("2016-01-01T12:00"),
                        ]
                    ]
                ),
            )
            dataset["time"].attrs["bounds"] = "time_bnds"
            dataset.to_netcdf(native_path)
            points = self._points().assign(latitude=0.5, longitude=0.5, depth_m=10.0)
            result = reference_run(
                {"points": points},
                Path(temporary) / "out",
                {
                    "native_glorys": str(native_path),
                    "native_spatial_operator": "valid_bilinear",
                    "native_max_spatial_distance_deg": 1.0,
                },
            )
            self.assertEqual(result["status"], "partial")
            matches = pd.read_csv(
                Path(temporary) / "out" / "native_reference_matches.csv"
            )
            self.assertIn("dry_or_invalid_bilinear_corner", set(matches.reason))
            self.assertTrue(
                (Path(temporary) / "out" / "reference_precision_audit.csv").is_file()
            )

    def test_cugn_import_marks_processed_section_unknown_independence(self):
        """Processed sections retain their provenance instead of being called raw casts."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "CUGN_line_66.nc"
            xr.Dataset(
                {
                    "temperature": (("station", "depth"), [[20.0, 19.0]]),
                    "salinity": (("station", "depth"), [[35.0, 35.1]]),
                },
                coords={
                    "station": [0],
                    "depth": [10.0, 100.0],
                    "time": ("station", ["2016-01-01"]),
                    "latitude": ("station", [45.0]),
                    "longitude": ("station", [12.0]),
                },
            ).to_netcdf(source)
            frame, metadata = convert_file(source, Path(temporary))
            self.assertEqual(len(frame), 4)
            self.assertTrue(frame.processed_section.all())
            self.assertTrue((frame.independence == "unknown").all())
            result = import_cugn([source], Path("val") / "results" / "test_cugn_import")
            self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
