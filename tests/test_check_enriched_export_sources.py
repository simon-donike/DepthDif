import tempfile
from pathlib import Path
import unittest

from data.dataset_creation.a_check_export_sourcefiles import (
    SourceFile,
    _select_timed_files_for_range,
    check_sources,
)
from data.dataset_creation.b_export_enriched_argo_profiles import TimedFile


class CheckEnrichedExportSourcesTests(unittest.TestCase):
    def test_select_timed_files_keeps_range_and_bracketing_edges(self) -> None:
        index = [
            TimedFile(Path("before.nc"), 9.0),
            TimedFile(Path("start.nc"), 10.0),
            TimedFile(Path("middle.nc"), 11.0),
            TimedFile(Path("after.nc"), 13.0),
        ]

        selected = _select_timed_files_for_range(
            index,
            start_date=19500111,
            end_date=19500112,
        )

        self.assertEqual(
            [item.path.name for item in selected],
            ["before.nc", "start.nc", "middle.nc", "after.nc"],
        )

    def test_check_sources_reports_corrupt_argo_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "EN.4.2.2.f.profiles.g10.201001.nc"
            path.write_bytes(b"not-netcdf")

            broken = check_sources([SourceFile("argo", path, 20100101)])

            self.assertEqual(len(broken), 1)
            self.assertEqual(broken[0].source.path, path)


if __name__ == "__main__":
    unittest.main()
