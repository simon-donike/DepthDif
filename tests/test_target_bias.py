"""Tests for frozen depth-wise target bias artifacts and the streaming fitter."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml
import zarr

from depth_recon.data.fit_target_bias import fit_target_bias
from depth_recon.data.target_bias import TargetBias


def _write_bias(path: Path, *, supported: np.ndarray | None = None) -> None:
    """Write a minimal valid target-bias artifact."""
    metadata = {
        "kind": "en4_minus_glorys_target_bias",
        "units": "degree_C",
        "sign_convention": "EN4_minus_GLORYS; target=GLORYS+bias",
        "min_profiles": 30,
        "grid": {"height": 2, "width": 2},
        "excluded_years": [2016],
        "source_temperature": "EN4 potential temperature raw compact profile",
        "source_temperature_variable": "POTM_CORRECTED",
        "source_temperature_standard_name": "sea_water_potential_temperature",
        "input_hashes": {"manifest.yaml": "abc"},
        "depth_axis_m": [0.0, 10.0],
    }
    np.savez(
        path,
        bias=np.asarray([1.5, 0.0], dtype=np.float32),
        count=np.asarray([30, 0], dtype=np.int64),
        variance=np.asarray([0.25, 0.0], dtype=np.float32),
        supported=np.asarray([True, False] if supported is None else supported),
        depth_axis_m=np.asarray([0.0, 10.0], dtype=np.float32),
        metadata=json.dumps(metadata),
    )


def _write_fixture(root: Path) -> None:
    """Create a compact POTM_CORRECTED export with controlled duplicate rows."""
    (root / "argo").mkdir()
    (root / "rasters/glorys/thetao").mkdir(parents=True)
    zarr_path = root / "argo/profiles.zarr"
    group = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    group.create_array("glorys_depth", data=np.asarray([0.0, 10.0], dtype=np.float32))
    # Code zero is 273.15 K / 0 C, code 254 is 274.15 K / 1 C.
    encoded = np.asarray(
        [
            [254, 254],  # Included source A: differences [1, 1].
            [0, 0],  # Duplicate source A: must not affect the moments.
            [254, 254],  # Target year 2016: excluded.
            [254, 254],  # Profile year 2016: excluded across a boundary.
            [254, 254],  # Bad QC: excluded.
            [254, 254],  # Included source B: differences [1, 1].
            [254, 254],  # Included source C: depth 1 invalid.
        ],
        dtype=np.uint8,
    )
    group.create_array("argo_temp_kelvin_uint8", data=encoded, chunks=(3, 2))
    group["argo_temp_kelvin_uint8"].attrs.update(
        {
            "minimum": 273.15,
            "maximum": 274.15,
            "nodata": 255,
            "valid_code_max": 254,
            "units": "K",
            "source_variable": "POTM_CORRECTED",
            "standard_name": "sea_water_potential_temperature",
        }
    )
    group.create_array(
        "argo_temp_valid",
        data=np.asarray(
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 0]], dtype=bool
        ),
        chunks=(3, 2),
    )
    group.create_array(
        "argo_temp_profile_qc",
        data=np.asarray([-1, -1, -1, -1, 4, -1, -1], dtype=np.int8),
        chunks=(3,),
    )
    group.create_array(
        "profile_date",
        data=np.asarray(
            [20150101, 20150101, 20150101, 20160101, 20150101, 20150101, 20150101],
            dtype=np.int32,
        ),
        chunks=(3,),
    )
    group.create_array(
        "target_date",
        data=np.asarray(
            [20150101, 20150101, 20160101, 20150101, 20150101, 20150101, 20150101],
            dtype=np.int32,
        ),
        chunks=(3,),
    )
    group.create_array("grid_row", data=np.zeros(7, dtype=np.int32), chunks=(3,))
    group.create_array(
        "grid_col", data=np.asarray([0, 1, 0, 0, 0, 0, 0], dtype=np.int32), chunks=(3,)
    )
    group.create_array(
        "profile_source_file",
        data=np.asarray(["A", "A", "C", "D", "E", "B", "F"], dtype="U1"),
        chunks=(3,),
    )
    group.create_array(
        "source_profile_idx",
        data=np.asarray([1, 1, 2, 3, 4, 5, 6], dtype=np.int32),
        chunks=(3,),
    )
    zarr.consolidate_metadata(zarr_path)
    raster_path = root / "rasters/glorys/thetao/thetao_20150101.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=1,
        width=2,
        count=2,
        dtype="uint8",
        transform=from_origin(0, 1, 1, 1),
    ) as raster:
        raster.write(np.zeros((2, 1, 2), dtype=np.uint8))
    manifest = {
        "depth_axis_m": [0.0, 10.0],
        "grid": {"height": 1, "width": 2},
        "argo": {"path": "argo/profiles.zarr"},
        "stretch": {
            "temperature_kelvin": {
                "minimum": 273.15,
                "maximum": 274.15,
                "nodata": 255,
                "valid_code_max": 254,
            }
        },
        "rasters": {
            "glorys": {
                "thetao": [
                    {
                        "date": 20150101,
                        "path": "rasters/glorys/thetao/thetao_20150101.tif",
                    }
                ]
            }
        },
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


class TargetBiasTests(unittest.TestCase):
    """Validate strict artifact loading and a complete compact-data fit."""

    def test_target_bias_applies_only_supported_finite_cells(self) -> None:
        """Offsets preserve NaNs and unsupported depth bands."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bias.npz"
            _write_bias(path)
            artifact = TargetBias(path, expected_depth_axis_m=np.asarray([0.0, 10.0]))
            values = np.asarray([[[1.0, np.nan]], [[2.0, 3.0]]], dtype=np.float32)
            result = artifact.apply(values)
            self.assertAlmostEqual(float(result[0, 0, 0]), 2.5)
            self.assertTrue(np.isnan(result[0, 0, 1]))
            np.testing.assert_allclose(result[1], values[1])

    def test_target_bias_rejects_wrong_depth_axis(self) -> None:
        """Dataset depth alignment is part of artifact validation."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bias.npz"
            _write_bias(path)
            with self.assertRaisesRegex(ValueError, "depth axis"):
                TargetBias(path, expected_depth_axis_m=np.asarray([0.0, 20.0]))

    def test_fit_target_bias_deduplicates_qc_and_both_2016_boundaries(self) -> None:
        """The fit uses only three clean source profiles and supports depth zero."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            output = root / "bias.npz"
            fit_target_bias(root, output, val_year=2016, min_profiles=3, chunk_size=2)
            artifact = TargetBias(output, expected_depth_axis_m=np.asarray([0.0, 10.0]))
            # A, B, and F each differ from zero by exactly +1 C. The duplicate
            # A row, both 2016 cases, and QC=4 profile cannot change the fit.
            np.testing.assert_array_equal(artifact.count, [3, 2])
            np.testing.assert_allclose(artifact.bias, [1.0, 0.0], atol=1e-6)
            np.testing.assert_array_equal(artifact.supported, [True, False])
            self.assertEqual(artifact.metadata["profile_count_used"], 3)
            self.assertEqual(artifact.metadata["excluded_years"], [2016])
            # Residual construction restores the corrected GLORYS target once.
            corrected = artifact.apply(np.zeros((2, 1, 1), dtype=np.float32))
            np.testing.assert_allclose(corrected, np.asarray([[[1.0]], [[0.0]]]))


if __name__ == "__main__":
    unittest.main()
