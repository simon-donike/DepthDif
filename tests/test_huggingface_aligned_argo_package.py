from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

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


if __name__ == "__main__":
    unittest.main()
