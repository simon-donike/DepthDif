"""Tests for the Italian XBT archive importer."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import pandas as pd

from val.import_xbt import import_xbt_archive

HEADER = "\t".join(
    [
        "Cruise",
        "Station",
        "Type",
        "mon/day/yr",
        "hh:mm",
        "Longitude [degrees_east]",
        "Latitude [degrees_north]",
        "Bot. Depth [m]",
        "Elapsed Time [s]",
        "Depth 1 [m]",
        "Depth 2 [m]",
        "Depth 3 [m]",
        "Temperature 1 [°C]",
        "Temperature 2 [°C] ",
        "QF",
    ]
)


def _row(station: int, depth: float, temperature: float, qc: int) -> str:
    """Create one row in the published 15-column PNRA layout."""
    return "\t".join(
        map(
            str,
            [
                "PNRAXXXI",
                station,
                "XBT",
                "01/16/2016",
                "22:00",
                "173.2726",
                "-47.4915",
                "760",
                "0.1",
                "0.6472",
                "0.6691",
                depth,
                "10.54",
                temperature,
                qc,
            ],
        )
    )


class TestImportXBT(unittest.TestCase):
    def _archive(self, root: Path) -> Path:
        archive = root / "focus.tar.gz"
        content = "// Attribute Name\tValue\n" + "\n".join(
            [
                HEADER,
                _row(1, 0.7677, 10.58, 2),
                _row(1, 3.9163, 10.58, 1),
                _row(2, 3.9163, 9.0, 1),
            ]
        )
        with tarfile.open(archive, "w:gz") as stream:
            payload = content.encode()
            info = tarfile.TarInfo("inputs/italian_xbt/xbt_PNRA_XXXI.txt")
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
        return archive

    def test_uses_qf1_raw_columns_and_metadata_as_profile_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            metadata = root / "metadata.parquet"
            pd.DataFrame(
                [
                    {
                        "profile_id": "italian-xbt:PNRAXXXI:1",
                        "cruise": "PNRAXXXI",
                        "station": "1",
                        "datetime_utc": "2016-01-16T22:00:00Z",
                        "source_file": "xbt_PNRA_XXXI.txt",
                        "latitude": -47.4915,
                        "longitude_raw": 173.2726,
                    }
                ]
            ).to_parquet(metadata)
            output = root / "observations.csv"
            manifest = import_xbt_archive(
                archive, output, metadata_path=metadata, year=2016
            )
            frame = pd.read_csv(output)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].profile_id, "italian-xbt:PNRAXXXI:1")
            self.assertAlmostEqual(float(frame.iloc[0].depth_m), 3.9163)
            self.assertAlmostEqual(float(frame.iloc[0].observation), 10.58)
            self.assertEqual(int(frame.iloc[0].qc_flag), 1)
            self.assertEqual(frame.iloc[0].temperature_kind, "in_situ")
            self.assertEqual(manifest["accepted_qc_flags"], [1])
            self.assertEqual(manifest["metadata_profiles_missing_from_archive"], 0)
            saved = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(saved["rows_parsed"], 1)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                import_xbt_archive(archive, output, metadata_path=metadata)

    def test_ids_without_metadata_include_member_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "observations.csv"
            import_xbt_archive(self._archive(root), output)
            frame = pd.read_csv(output)
            self.assertEqual(len(frame), 2)
            self.assertTrue(
                frame.profile_id.str.startswith(
                    "italian-xbt:xbt_PNRA_XXXI.txt:PNRAXXXI:"
                ).all()
            )


if __name__ == "__main__":
    unittest.main()
