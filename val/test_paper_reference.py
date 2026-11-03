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
            xr.Dataset(
                {"thetao": (("time", "depth", "latitude", "longitude"), np.array([[[[12.0]], [[13.0]]]]))},
                coords={"time": pd.to_datetime(["2016-01-01"]), "depth": [0.0, 20.0], "latitude": [0.0], "longitude": [0.0]},
            ).to_netcdf(native_path)
            result = reference_run({"points": self._points()}, Path(temporary) / "out", {"native_glorys": str(native_path)})
            self.assertEqual(result["status"], "complete")
            values = pd.read_csv(Path(temporary) / "out" / "native_reference_values.csv")
            self.assertAlmostEqual(values.glorys_native.iloc[0], 12.5)

    def test_cugn_import_marks_processed_section_unknown_independence(self):
        """Processed sections retain their provenance instead of being called raw casts."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "CUGN_line_66.nc"
            xr.Dataset(
                {"temperature": (("station", "depth"), [[20.0, 19.0]]), "salinity": (("station", "depth"), [[35.0, 35.1]])},
                coords={"station": [0], "depth": [10.0, 100.0], "time": ("station", ["2016-01-01"]), "latitude": ("station", [45.0]), "longitude": ("station", [12.0])},
            ).to_netcdf(source)
            frame, metadata = convert_file(source, Path(temporary))
            self.assertEqual(len(frame), 4)
            self.assertTrue(frame.processed_section.all())
            self.assertTrue((frame.independence == "unknown").all())
            result = import_cugn([source], Path("val") / "results" / "test_cugn_import")
            self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
