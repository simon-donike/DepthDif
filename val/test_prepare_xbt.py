"""Tests for GLORYS salinity preparation of imported XBT profiles."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import yaml

from val.prepare_xbt import prepare_xbt_reference


class TestPrepareXBT(unittest.TestCase):
    def _dataset(self, root: Path, encoded: tuple[int, int] = (0, 254)) -> Path:
        data_root = root / "dataset"
        raster = data_root / "rasters/glorys/so/so_20160108.tif"
        raster.parent.mkdir(parents=True)
        with rasterio.open(
            raster,
            "w",
            driver="GTiff",
            width=1,
            height=1,
            count=2,
            dtype="uint8",
            transform=from_origin(10.0, -20.0, 1.0, 1.0),
        ) as target:
            target.write(np.asarray(encoded, dtype=np.uint8).reshape(2, 1, 1))
        manifest = {
            "depth_axis_m": [0.0, 100.0],
            "stretch": {
                "salinity": {
                    "minimum": 30.0,
                    "maximum": 40.0,
                    "valid_code_max": 254,
                    "nodata": 255,
                    "units": "PSU",
                }
            },
            "rasters": {
                "glorys": {
                    "so": [
                        {
                            "date": 20160108,
                            "path": "rasters/glorys/so/so_20160108.tif",
                        }
                    ]
                }
            },
        }
        (data_root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        return data_root

    def _profiles(self, root: Path) -> Path:
        path = root / "imported.csv"
        common = {
            "profile_id": "italian-xbt:PNRAXXXI:1",
            "cluster_id": "PNRAXXXI",
            "source": "italian_xbt",
            "observation_time": "2016-01-08T12:00:00Z",
            "longitude": 10.5,
            "latitude": -20.5,
            "variable": "temperature",
            "independence": "unknown",
            "temperature_kind": "in_situ",
        }
        pd.DataFrame(
            [
                {**common, "depth_m": 50.0, "observation": 8.0},
                {**common, "depth_m": 150.0, "observation": 7.0},
            ]
        ).to_csv(path, index=False)
        return path

    def test_decodes_and_interpolates_salinity_to_raw_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out/profiles.csv"
            manifest_path = root / "out/manifest.json"
            profiles = self._profiles(root)
            data_root = self._dataset(root)
            manifest = prepare_xbt_reference(
                profiles,
                data_root,
                output,
                manifest_path,
            )
            result = pd.read_csv(output)
            self.assertEqual(len(result), 1)
            self.assertEqual(float(result.iloc[0].depth_m), 50.0)
            self.assertAlmostEqual(float(result.iloc[0].reference_salinity), 35.0)
            self.assertEqual(result.iloc[0].reference_salinity_kind, "practical")
            self.assertAlmostEqual(
                float(result.iloc[0].reference_salinity_time_delta_days), 0.5
            )
            self.assertEqual(float(result.iloc[0].reference_salinity_longitude), 10.5)
            self.assertEqual(float(result.iloc[0].reference_salinity_latitude), -20.5)
            exclusions = pd.read_csv(output.with_suffix(".exclusions.csv"))
            self.assertEqual(
                exclusions.iloc[0].reason, "outside_supported_salinity_depths_or_gap"
            )
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(
                manifest["config_snippet"]["reference_salinity_provenance"]["sha256"],
                digest,
            )
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                prepare_xbt_reference(
                    profiles,
                    data_root,
                    output,
                    manifest_path,
                )

    def test_rejects_profile_when_salinity_bracket_is_nodata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "no XBT rows"):
                prepare_xbt_reference(
                    self._profiles(root),
                    self._dataset(root, encoded=(255, 254)),
                    root / "out/profiles.csv",
                    root / "out/manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
