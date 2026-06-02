from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo import (
    DEFAULT_ZARR_NAME,
    build_huggingface_aligned_argo_package,
)


def _write_enriched_argo_zarr(path: Path) -> None:
    """Write a tiny enriched ARGO profile zarr matching the package input schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "profile_source_file": (
                ("profile",),
                np.asarray(["EN.4.2.2.f.profiles.g10.202401.nc"] * 2),
            ),
            "profile_idx": (("profile",), np.asarray([3, 4], dtype=np.int32)),
            "profile_date": (
                ("profile",),
                np.asarray([20240102, 20240103], dtype=np.int32),
            ),
            "profile_juld": (
                ("profile",),
                np.asarray([27030.0, 27031.0], dtype=np.float64),
            ),
            "latitude": (("profile",), np.asarray([1.25, 1.75], dtype=np.float32)),
            "longitude": (("profile",), np.asarray([10.25, 10.75], dtype=np.float32)),
            "valid_observed_depth_count": (
                ("profile",),
                np.asarray([2, 1], dtype=np.int16),
            ),
            "argo_temp_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[10.0, 20.0], [11.0, np.nan]], dtype=np.float32),
            ),
            "argo_psal_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[35.0, 36.0], [35.5, np.nan]], dtype=np.float32),
            ),
            "argo_temp_valid_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[True, True], [True, False]], dtype=bool),
            ),
            "argo_psal_valid_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[True, True], [True, False]], dtype=bool),
            ),
            "sss_sos": (("profile",), np.asarray([34.5, 35.5], dtype=np.float32)),
        },
        coords={
            "profile": np.asarray([0, 1], dtype=np.int64),
            "glorys_depth": np.asarray([0.0, 10.0], dtype=np.float32),
        },
        attrs={"created_by": "test", "source_products": {"argo": "EN4"}},
    )
    ds["argo_temp_on_glorys_depth"].attrs["units"] = "degree_C"
    ds["argo_psal_on_glorys_depth"].attrs["units"] = "1e-3"
    ds.to_zarr(path, mode="w", zarr_format=2)


def _write_compact_argo_zarr(path: Path) -> None:
    """Write a tiny compact grid-indexed ARGO zarr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "profile_date": (("profile",), np.asarray([20240102], dtype=np.int32)),
            "target_date": (("profile",), np.asarray([20240102], dtype=np.int32)),
            "latitude": (("profile",), np.asarray([1.25], dtype=np.float32)),
            "longitude": (("profile",), np.asarray([10.25], dtype=np.float32)),
            "grid_row": (("profile",), np.asarray([0], dtype=np.int32)),
            "grid_col": (("profile",), np.asarray([0], dtype=np.int32)),
            "argo_temp_kelvin_uint8": (
                ("profile", "glorys_depth"),
                np.asarray([[100, 101]], dtype=np.uint8),
            ),
            "argo_psal_uint8": (
                ("profile", "glorys_depth"),
                np.asarray([[120, 121]], dtype=np.uint8),
            ),
            "argo_temp_valid": (
                ("profile", "glorys_depth"),
                np.asarray([[True, True]], dtype=bool),
            ),
            "argo_psal_valid": (
                ("profile", "glorys_depth"),
                np.asarray([[True, True]], dtype=bool),
            ),
        },
        coords={
            "profile": np.asarray([0], dtype=np.int64),
            "glorys_depth": np.asarray([0.0, 10.0], dtype=np.float32),
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


class TestHuggingFaceAlignedArgoPackage(unittest.TestCase):
    def test_package_keeps_enriched_zarr_and_writes_indices(self) -> None:
        """The packaged Zarr remains directly readable by xarray."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_zarr = tmp_path / "enriched_argo_profiles.zarr"
            _write_enriched_argo_zarr(input_zarr)

            package_dir = build_huggingface_aligned_argo_package(
                input_zarr=input_zarr,
                output_dir=tmp_path / "hf_argo_package",
                file_mode="copy",
                overwrite=True,
            )

            zarr_path = package_dir / "data" / DEFAULT_ZARR_NAME
            packaged = xr.open_zarr(zarr_path, consolidated=None)
            try:
                self.assertEqual(int(packaged.sizes["profile"]), 2)
                self.assertIn("argo_temp_on_glorys_depth", packaged)
                self.assertIn("sss_sos", packaged)
            finally:
                packaged.close()

            profiles = pd.read_parquet(package_dir / "indices/profiles.parquet")
            variables = pd.read_parquet(package_dir / "indices/variables.parquet")
            self.assertEqual(len(profiles), 2)
            self.assertEqual(int(profiles["argo_temp_valid_depth_count"].iloc[1]), 1)
            self.assertEqual(int(profiles["argo_psal_valid_depth_count"].iloc[0]), 2)
            self.assertIn("argo_temp_on_glorys_depth", set(variables["name"]))
            self.assertTrue((package_dir / "README.md").exists())
            self.assertTrue((package_dir / "examples/open_with_xarray.py").exists())
            self.assertTrue((package_dir / "metadata/stac-item.json").exists())
            self.assertTrue((package_dir / "LICENSE").exists())

    def test_package_can_assemble_full_depthdif_upload_layout(self) -> None:
        """The package builder can stage the complete upload root without links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_zarr = tmp_path / "enriched_argo_profiles.zarr"
            compact_zarr = tmp_path / "argo_profiles_on_grid.zarr"
            raster_root = tmp_path / "rasters"
            masks_dir = tmp_path / "masks"
            manifest_path = tmp_path / "manifest.yaml"
            _write_enriched_argo_zarr(input_zarr)
            _write_compact_argo_zarr(compact_zarr)
            (raster_root / "sss/sos").mkdir(parents=True)
            raster_file = raster_root / "sss/sos/sos_20240102.tif"
            raster_file.write_bytes(b"tif")
            masks_dir.mkdir()
            (masks_dir / "land_mask.tif").write_bytes(b"mask")
            manifest_path.write_text(
                "output_dir: /work/data/depthdif\n", encoding="utf-8"
            )

            package_dir = build_huggingface_aligned_argo_package(
                input_zarr=input_zarr,
                output_dir=tmp_path / "OceanVariableReconstruction",
                file_mode="copy",
                overwrite=True,
                raster_root=raster_root,
                compact_argo_zarr=compact_zarr,
                manifest_path=manifest_path,
                masks_dir=masks_dir,
            )

            self.assertTrue((package_dir / "rasters/sss/sos/sos_20240102.tif").exists())
            self.assertTrue((package_dir / "argo/argo_profiles_on_grid.zarr").exists())
            self.assertTrue((package_dir / "data" / DEFAULT_ZARR_NAME).exists())
            self.assertTrue((package_dir / "manifest.yaml").exists())
            packaged_manifest = yaml.safe_load(
                (package_dir / "manifest.yaml").read_text()
            )
            self.assertEqual(packaged_manifest["output_dir"], package_dir.as_posix())
            self.assertTrue((package_dir / "masks/land_mask.tif").exists())
            readme_text = (package_dir / "README.md").read_text()
            self.assertTrue(
                (package_dir / "assets/figures/depthdif_schema.webp").exists()
            )
            self.assertTrue(
                (
                    package_dir / "assets/data/geotiff_dataset_random100_surface.webp"
                ).exists()
            )
            self.assertTrue(
                (
                    package_dir / "assets/data/profile_comparison_good_alignment.webp"
                ).exists()
            )
            self.assertIn("rasters/", readme_text)
            self.assertIn(
                "assets/data/geotiff_dataset_random100_surface.webp", readme_text
            )
            self.assertIn("ARGO Alignment Examples", readme_text)
            self.assertEqual(
                (package_dir / "rasters/sss/sos/sos_20240102.tif").stat().st_nlink,
                1,
            )


if __name__ == "__main__":
    unittest.main()
