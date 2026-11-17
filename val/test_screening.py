"""Prevent unscreened or misidentified observations from entering validation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from val.screening import verify_screened_profiles


class ProfileScreeningTests(unittest.TestCase):
    """Check registry membership independently of model predictions."""

    def test_en4_requires_the_no_candidate_status_and_selected_year(self):
        """A matching identifier alone cannot bypass archive status or year."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.parquet"
            pd.DataFrame(
                {
                    "profile_source_file": ["cast.nc"] * 3,
                    "source_profile_idx": [1, 2, 3],
                    "datetime_utc": pd.to_datetime(
                        ["2016-01-01", "2016-01-01", "2017-01-01"], utc=True
                    ),
                    "audit_status": [
                        "no_spatiotemporal_candidate",
                        "matched",
                        "no_spatiotemporal_candidate",
                    ],
                }
            ).to_parquet(path)
            cohort = {"kind": "en4_archive", "year": 2016, "audit_path": str(path)}
            for index in (1, 2, 3):
                points = pd.DataFrame(
                    {
                        "profile_id": [f"en4:cast.nc:{index}"],
                        "observation_time": ["2016-01-01"],
                    }
                )
                output = root / str(index)
                if index == 1:
                    result = verify_screened_profiles(points, cohort, output)
                    self.assertEqual(result["verified_profiles"], 1)
                    self.assertFalse(result["verified_not_assimilated"])
                else:
                    with self.assertRaisesRegex(ValueError, "membership failed"):
                        verify_screened_profiles(points, cohort, output)
                    result = json.loads(
                        (output / "profile_membership.json").read_text()
                    )
                    self.assertEqual(result["unlisted_profiles"], 1)

    def test_xbt_checks_native_identity_and_quality(self):
        """A listed XBT ID cannot conceal another cast, position, or bad QC."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.parquet"
            pd.DataFrame(
                {
                    "profile_id": ["italian-xbt:C:1"],
                    "datetime_utc": pd.to_datetime(["2016-01-01"], utc=True),
                    "source_file": ["cast.txt"],
                    "cruise": ["C"],
                    "station": ["1"],
                    "latitude": [-50.0],
                    "longitude_raw": [170.0],
                }
            ).to_parquet(path)
            cohort = {"kind": "external", "year": 2016, "screened_profiles": str(path)}
            points = pd.DataFrame(
                {
                    "profile_id": ["italian-xbt:C:1"],
                    "observation_time": ["2016-01-01"],
                    "source_file": ["cast.txt"],
                    "cruise": ["C"],
                    "station": [1],
                    "latitude": [-50.0],
                    "longitude": [170.0],
                    "qc_flag": [1],
                }
            )
            result = verify_screened_profiles(
                pd.concat([points, points]), cohort, root / "good"
            )
            self.assertEqual(result["verified_profiles"], 1)
            for key, value in (
                ("longitude", 171),
                ("qc_flag", 2),
                ("source_file", "other.txt"),
                ("observation_time", "2016-01-02"),
            ):
                with self.subTest(key=key):
                    changed = points.copy()
                    changed[key] = value
                    with self.assertRaisesRegex(ValueError, "membership failed"):
                        verify_screened_profiles(changed, cohort, root / key)

    def test_changed_screening_registry_is_rejected(self):
        """Pinned list contents cannot change silently between audits and inference."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "registry.parquet"
            path.write_bytes(b"changed")
            cohort = {
                "kind": "external",
                "year": 2016,
                "screened_profiles": str(path),
                "screened_profiles_sha256": hashlib.sha256(b"original").hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_screened_profiles(pd.DataFrame(), cohort, root)


if __name__ == "__main__":
    unittest.main()
