"""Training conditioning must respect the observation year across target assignments."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFProfileStore


class TrainingProfileYearTests(unittest.TestCase):
    """Exercise year exclusions in the production compact-profile lookup."""

    @staticmethod
    def _store(path: Path, *, with_profile_date: bool = True) -> None:
        """Write two observations assigned to the same training-year target."""
        ds = xr.Dataset(
            {
                "target_date": ("profile", [20151231, 20151231]),
                "grid_row": ("profile", [0, 0]),
                "grid_col": ("profile", [0, 0]),
                "argo_temp_kelvin_uint8": (
                    ("profile", "glorys_depth"),
                    np.asarray([[30, 40], [50, 60]], dtype=np.uint8),
                ),
                "argo_temp_valid": (
                    ("profile", "glorys_depth"),
                    np.ones((2, 2), dtype=bool),
                ),
            },
            coords={"profile": [0, 1], "glorys_depth": [0.5, 10.0]},
        )
        if with_profile_date:
            ds["profile_date"] = ("profile", [20160101, 20151230])
        ds["argo_temp_kelvin_uint8"].attrs.update(minimum=273.15, maximum=313.15)
        ds.to_zarr(path, mode="w", zarr_format=2)

    def test_boundary_observation_is_excluded_without_changing_validation(self) -> None:
        """A 2016 observation assigned to 2015 must not enter training conditions."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.zarr"
            self._store(path)
            training = ArgoGeoTIFFProfileStore(path, exclude_profile_years=[2016])
            validation = ArgoGeoTIFFProfileStore(path)
            query = dict(target_date=20151231, grid_y0=0, grid_x0=0, tile_size=1)
            try:
                np.testing.assert_array_equal(training.query_indices(**query), [1])
                np.testing.assert_array_equal(validation.query_indices(**query), [0, 1])
                self.assertNotEqual(
                    training.quality_cache_signature(),
                    validation.quality_cache_signature(),
                )
            finally:
                training.close()
                validation.close()

    def test_exclusion_requires_actual_observation_date(self) -> None:
        """Missing observation provenance must fail rather than substitute target dates."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.zarr"
            self._store(path, with_profile_date=False)
            with self.assertRaisesRegex(RuntimeError, "profile_date"):
                ArgoGeoTIFFProfileStore(path, exclude_profile_years=[2016])


if __name__ == "__main__":
    unittest.main()
