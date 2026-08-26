from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import xarray as xr
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.data.dataset_creation.export_dataset_geotiff import (
    export_training_geotiff_dataset,
)
from depth_recon.paths import config_path as packaged_config_path
from tests.test_vertical_offset_prior import write_prior_artifact
from train import build_dataset
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize


def _write_land_mask(path: Path, values: np.ndarray | None = None) -> Path:
    """Write a tiny authoritative raster grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (
        np.zeros((2, 2), dtype=np.uint8)
        if values is None
        else np.asarray(values, dtype=np.uint8)
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10.0, 2.0, 1.0, 1.0),
    ) as dst:
        dst.write(mask, 1)
    return path


def _write_glorys(root_dir: Path, *, date_value: int) -> None:
    """Write a tiny GLORYS-like weekly source file."""
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    depth = np.asarray([0.0, 10.0], dtype=np.float32)
    thetao = np.asarray(
        [
            [
                [[10.0, np.nan], [12.0, 13.0]],
                [[20.0, 21.0], [22.0, 23.0]],
            ]
        ],
        dtype=np.float32,
    )
    salinity = np.asarray(
        [
            [
                [[35.0, 35.1], [35.2, 35.3]],
                [[36.0, 36.1], [36.2, 36.3]],
            ]
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"), thetao),
            "so": (("time", "depth", "latitude", "longitude"), salinity),
        },
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"glorys_{int(date_value)}.nc", engine="h5netcdf")


def _write_ostia(root_dir: Path, *, date_value: int, base: float) -> None:
    """Write a tiny OSTIA-like surface source file."""
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    offsets = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    values = (np.float32(base) + offsets)[None, ...]
    ds = xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values.astype(np.float32))},
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "lat": lat,
            "lon": lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"{int(date_value)}120000-ostia.nc", engine="h5netcdf")


def _write_sss(root_dir: Path, *, date_value: int, base: float) -> None:
    """Write a tiny SSS-like surface salinity source file."""
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    offsets = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    values = (np.float32(base) + offsets)[None, ...]
    ds = xr.Dataset(
        {"sos": (("time", "lat", "lon"), values.astype(np.float32))},
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "lat": lat,
            "lon": lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"sss_{int(date_value)}.nc", engine="h5netcdf")


def _write_sealevel(root_dir: Path, *, date_value: int, base: float) -> None:
    """Write a tiny sea-level source file with ADT."""
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    offsets = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    ds = xr.Dataset(
        {"adt": (("time", "latitude", "longitude"), (base + offsets)[None, ...])},
        coords={
            "time": np.asarray([0.0], dtype=np.float64),
            "latitude": lat,
            "longitude": lon,
        },
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root_dir / f"sealevel_{int(date_value)}.nc", engine="h5netcdf")


def _write_enriched_argo_zarr(path: Path) -> None:
    """Write a tiny enriched ARGO profile zarr matching the aligned export schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "profile_date": (
                ("profile",),
                np.asarray([20240108, 20240108], dtype=np.int32),
            ),
            "profile_idx": (("profile",), np.asarray([7, 8], dtype=np.int32)),
            "profile_source_file": (
                ("profile",),
                np.asarray(["EN.4.2.2.f.profiles.g10.202401.nc"] * 2),
            ),
            "latitude": (
                ("profile",),
                np.asarray([1.5, 1.5], dtype=np.float32),
            ),
            "longitude": (
                ("profile",),
                np.asarray([10.5, 10.5], dtype=np.float32),
            ),
            "argo_temp_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            ),
            "argo_potm_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            ),
            "argo_psal_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[35.0, 36.0], [37.0, 38.0]], dtype=np.float32),
            ),
            "argo_depth_qc_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[1, 1], [1, 1]], dtype=np.int8),
            ),
            "argo_temp_qc_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[1, 2], [4, 4]], dtype=np.int8),
            ),
            "argo_potm_qc_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[1, 2], [4, 4]], dtype=np.int8),
            ),
            "argo_psal_qc_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[1, 2], [4, 4]], dtype=np.int8),
            ),
            "argo_juld_qc": (("profile",), np.asarray([1, 1], dtype=np.int8)),
            "argo_position_qc": (
                ("profile",),
                np.asarray([1, 1], dtype=np.int8),
            ),
            "argo_profile_depth_qc": (
                ("profile",),
                np.asarray([1, 1], dtype=np.int8),
            ),
            "argo_profile_potm_qc": (
                ("profile",),
                np.asarray([1, 1], dtype=np.int8),
            ),
            "argo_profile_psal_qc": (
                ("profile",),
                np.asarray([1, 1], dtype=np.int8),
            ),
        },
        coords={
            "profile": np.asarray([0, 1], dtype=np.int64),
            "glorys_depth": np.asarray([0.0, 10.0], dtype=np.float32),
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def _make_geotiff_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Export a tiny GeoTIFF training dataset and return root/cache/land paths."""
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    sealevel_dir = tmp_path / "sealevel"
    sss_dir = tmp_path / "sss"
    land_mask_path = _write_land_mask(
        tmp_path / "land_mask.tif",
        values=np.asarray([[0, 0], [0, 1]], dtype=np.uint8),
    )
    enriched_argo = tmp_path / "aligned_argo" / "enriched_argo_profiles.zarr"
    _write_glorys(glorys_dir, date_value=20240108)
    for date_value, ostia_base, sea_base, sss_base in (
        (20240105, 280.0, 0.0, 34.0),
        (20240108, 290.0, 1.0, 35.0),
        (20240111, 300.0, 2.0, 36.0),
    ):
        _write_ostia(ostia_dir, date_value=date_value, base=ostia_base)
        _write_sealevel(sealevel_dir, date_value=date_value, base=sea_base)
        _write_sss(sss_dir, date_value=date_value, base=sss_base)
    _write_enriched_argo_zarr(enriched_argo)
    output_dir = export_training_geotiff_dataset(
        glorys_dir=glorys_dir,
        ostia_dir=ostia_dir,
        sealevel_dir=sealevel_dir,
        sss_dir=sss_dir,
        enriched_argo_zarr=enriched_argo,
        land_mask_path=land_mask_path,
        output_dir=tmp_path / "geotiff_training",
        start_date=20240105,
        end_date=20240111,
        surface_aggregate_days=7,
        workers=1,
        overwrite=True,
        show_progress=False,
    )
    return output_dir, tmp_path / "cache", land_mask_path


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write a YAML payload."""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class TestArgoGeoTIFFGriddedPatchDataset(unittest.TestCase):
    def test_synthetic_prior_must_exclude_validation_year(self) -> None:
        """Synthetic pretraining must reject prior statistics containing validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            prior_path = write_prior_artifact(
                output_dir / "vertical_offset_prior.npz",
                depth_axis_m=np.asarray([0.0, 10.0], dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "validation year 2016"):
                ArgoGeoTIFFGriddedPatchDataset(
                    geotiff_root_dir=output_dir,
                    metadata_cache_dir=cache_dir,
                    split="train",
                    tile_size=2,
                    resolution_deg=1.0,
                    land_mask_path=land_mask_path,
                    patch_stride=2,
                    max_land_fraction=1.0,
                    val_year=2016,
                    require_argo_for_train=True,
                    include_salinity=True,
                    surface_conditioning={"sources": ["sst", "sss", "adt"]},
                    synthetic_target={
                        "enabled": True,
                        "statistics_path": prior_path.name,
                        "bathymetry_reference_date": 20240108,
                    },
                )

    def test_online_prior_integrates_surface_channels_masks_and_argo_anchors(
        self,
    ) -> None:
        """Prior mode should keep training independent of paired GLORYS values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            supervision_weight = np.ones((2, 2), dtype=np.float32)
            supervision_weight[1] = (0.65, 0.35)
            prior_path = write_prior_artifact(
                output_dir / "vertical_offset_prior.npz",
                depth_axis_m=np.asarray([0.0, 10.0], dtype=np.float32),
                supervision_weight=supervision_weight,
            )
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                include_salinity=True,
                surface_conditioning={"sources": ["sst"]},
                synthetic_target={
                    "enabled": True,
                    "statistics_path": prior_path.name,
                    "bathymetry_reference_date": 20240108,
                },
                return_info=True,
            )

            sample = dataset[0]
            repeated_train_sample = dataset[0]
            y_c = temperature_normalize(mode="denorm", tensor=sample["y"])
            y_salinity = salinity_normalize(mode="denorm", tensor=sample["y_salinity"])
            eo_sst_c = temperature_normalize(mode="denorm", tensor=sample["eo"][0])
            prior_surface_fields = dataset._load_surface_fields(dataset._rows.iloc[0])
            eo_sss = prior_surface_fields["sss"]

            self.assertTrue(dataset.synthetic_target_enabled)
            self.assertEqual(sample["eo"].shape, (1, 2, 2))
            self.assertEqual(set(dataset.surface_stores), {"sst", "sss", "adt"})
            self.assertEqual(sample["info"]["target_kind"], "vertical_offset_prior")
            self.assertNotIn("y_supervision_weight", sample)
            self.assertNotIn("y_salinity_supervision_weight", sample)
            self.assertNotIn("y_glorys", sample)
            self.assertNotIn("y_glorys_valid_mask", sample)
            self.assertNotIn("y_salinity_glorys", sample)
            self.assertNotIn("y_salinity_glorys_valid_mask", sample)
            np.testing.assert_array_equal(
                sample["y_valid_mask"].numpy(),
                sample["y_salinity_valid_mask"].numpy(),
            )
            # Compact ARGO profiles are exact equality constraints.
            self.assertTrue(
                torch.allclose(
                    y_c[:, 0, 0],
                    torch.tensor([10.0, 20.0], dtype=torch.float32),
                    atol=0.25,
                )
            )
            self.assertTrue(
                torch.allclose(
                    y_salinity[:, 0, 0],
                    torch.tensor([35.0, 36.0], dtype=torch.float32),
                    atol=0.05,
                )
            )
            # Away from ARGO, first depth is hard-anchored to dense EO.
            self.assertAlmostEqual(
                float(y_c[0, 0, 1]), float(eo_sst_c[0, 1]), delta=1.0e-5
            )
            self.assertAlmostEqual(
                float(y_salinity[0, 0, 1]), float(eo_sss[0, 1]), delta=1.0e-5
            )
            self.assertTrue(torch.equal(sample["y"], repeated_train_sample["y"]))

            # Deterministic targets are identical for training and validation access.
            dataset.split = "val"
            first_validation_sample = dataset[0]
            repeated_validation_sample = dataset[0]
            glorys_c = temperature_normalize(
                mode="denorm", tensor=first_validation_sample["y_glorys"]
            )
            glorys_salinity = salinity_normalize(
                mode="denorm", tensor=first_validation_sample["y_salinity_glorys"]
            )
            np.testing.assert_array_equal(
                first_validation_sample["y"].numpy(),
                repeated_validation_sample["y"].numpy(),
            )
            np.testing.assert_array_equal(
                first_validation_sample["y_salinity"].numpy(),
                repeated_validation_sample["y_salinity"].numpy(),
            )
            self.assertFalse(
                torch.equal(
                    first_validation_sample["y"],
                    first_validation_sample["y_glorys"],
                )
            )
            self.assertFalse(
                torch.equal(
                    first_validation_sample["y_salinity"],
                    first_validation_sample["y_salinity_glorys"],
                )
            )
            self.assertFalse(
                bool(first_validation_sample["y_glorys_valid_mask"][0, 1, 1])
            )
            self.assertTrue(
                bool(
                    first_validation_sample["y_salinity_glorys_valid_mask"][
                        :, 1, 1
                    ].all()
                )
            )
            self.assertTrue(
                torch.allclose(
                    glorys_c[:, 0, 0],
                    torch.tensor([12.0, 22.0], dtype=torch.float32),
                    atol=0.25,
                )
            )
            self.assertTrue(
                torch.allclose(
                    glorys_salinity[:, 0, 0],
                    torch.tensor([35.2, 36.2], dtype=torch.float32),
                    atol=0.05,
                )
            )

    def test_contract_decodes_stretched_rasters_and_argo_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                return_info=True,
                return_coords=True,
                include_salinity=True,
                surface_conditioning={"sources": ["sst", "sss", "adt"]},
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(int(dataset.rows[0]["argo_profile_count"]), 1)
            self.assertIn("argo_temp_qc_on_glorys_depth", dataset.argo_store.ds)
            self.assertIn("argo_temp_profile_qc", dataset.argo_store.ds)
            np.testing.assert_array_equal(
                dataset.argo_store.ds["argo_temp_profile_qc"].values,
                np.asarray([2, 4], dtype=np.int8),
            )
            self.assertNotIn("argo_juld_qc", dataset.argo_store.ds)
            sample = dataset[0]

            self.assertEqual(sample["eo"].shape, (3, 2, 2))
            self.assertEqual(sample["x"].shape, (2, 2, 2))
            self.assertEqual(sample["y"].shape, (2, 2, 2))
            self.assertEqual(sample["x_salinity"].shape, (2, 2, 2))
            self.assertEqual(sample["y_salinity"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["y_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["x_salinity_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["y_salinity_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask_1d"].shape, (1, 2, 2))
            self.assertEqual(sample["x_salinity_valid_mask_1d"].shape, (1, 2, 2))
            self.assertEqual(sample["land_mask"].shape, (1, 2, 2))
            self.assertNotIn("output_land_mask", sample)
            self.assertTrue(
                torch.equal(
                    sample["land_mask"],
                    torch.ones((1, 2, 2), dtype=torch.float32),
                )
            )
            self.assertEqual(sample["date"], 20240108)
            self.assertTrue(
                torch.allclose(sample["coords"], torch.tensor([1.0, 11.0]), atol=1e-5)
            )

            x_c = temperature_normalize(mode="denorm", tensor=sample["x"])
            y_c = temperature_normalize(mode="denorm", tensor=sample["y"])
            eo_c = temperature_normalize(mode="denorm", tensor=sample["eo"][0])
            eo_salinity_psu = salinity_normalize(mode="denorm", tensor=sample["eo"][1])
            eo_adt_m = sample["eo"][2] * 2.0
            x_salinity_psu = salinity_normalize(
                mode="denorm", tensor=sample["x_salinity"]
            )
            y_salinity_psu = salinity_normalize(
                mode="denorm", tensor=sample["y_salinity"]
            )
            self.assertTrue(bool(sample["x_valid_mask"][:, 0, 0].all().item()))
            self.assertFalse(bool(sample["x_valid_mask"][:, 0, 1].any().item()))
            self.assertTrue(
                torch.allclose(
                    x_c[:, 0, 0],
                    torch.tensor([10.0, 20.0], dtype=torch.float32),
                    atol=0.25,
                )
            )
            self.assertTrue(
                torch.allclose(
                    x_salinity_psu[:, 0, 0],
                    torch.tensor([35.0, 36.0], dtype=torch.float32),
                    atol=0.05,
                )
            )
            self.assertTrue(bool(sample["x_salinity_valid_mask"][:, 0, 0].all().item()))
            self.assertFalse(
                bool(sample["x_salinity_valid_mask"][:, 0, 1].any().item())
            )
            self.assertEqual(float(sample["x_salinity"][0, 0, 1]), 0.0)
            self.assertAlmostEqual(float(y_c[0, 0, 0]), 12.0, delta=0.25)
            self.assertAlmostEqual(float(y_c[1, 0, 0]), 22.0, delta=0.25)
            self.assertAlmostEqual(float(y_salinity_psu[0, 0, 0]), 35.2, delta=0.05)
            self.assertAlmostEqual(float(y_salinity_psu[1, 0, 0]), 36.2, delta=0.05)
            self.assertAlmostEqual(float(eo_c[0, 0]), 18.85, delta=0.25)
            self.assertAlmostEqual(float(eo_salinity_psu[0, 0]), 35.2, delta=0.05)
            self.assertAlmostEqual(float(eo_adt_m[0, 0]), 1.2, delta=0.02)
            self.assertFalse(bool(sample["y_valid_mask"][0, 1, 1].item()))
            self.assertTrue(bool(sample["y_salinity_valid_mask"][0, 1, 1].item()))
            self.assertEqual(sample["info"]["x_source"], "argo")

    def test_accepted_argo_qc_flags_filter_compact_profile_worst_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            strict = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                include_salinity=True,
                accepted_argo_qc_flags=(1, 2),
            )
            permissive = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                include_salinity=True,
                accepted_argo_qc_flags=(1, 2, 4),
            )

            self.assertEqual(int(strict.rows[0]["argo_profile_count"]), 1)
            self.assertEqual(int(permissive.rows[0]["argo_profile_count"]), 2)
            strict.argo_store.close()
            permissive.argo_store.close()

    def test_salinity_only_output_fields_skip_temperature_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
                return_info=True,
                return_coords=True,
                include_salinity=True,
                output_fields=("salinity",),
                eo_source="sss",
                eo_var_name="sos",
                surface_conditioning={"sources": ["sst", "sss", "adt"]},
            )

            sample = dataset[0]

            self.assertEqual(dataset.output_fields, ("salinity",))
            self.assertEqual(dataset.eo_source, "sss")
            self.assertEqual(dataset.eo_var_name, "sos")
            self.assertNotIn("x", sample)
            self.assertNotIn("y", sample)
            self.assertNotIn("x_valid_mask", sample)
            self.assertNotIn("y_valid_mask", sample)
            self.assertEqual(sample["x_salinity"].shape, (2, 2, 2))
            self.assertEqual(sample["y_salinity"].shape, (2, 2, 2))
            self.assertEqual(sample["x_salinity_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["y_salinity_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["land_mask"].shape, (1, 2, 2))

            x_salinity_psu = salinity_normalize(
                mode="denorm", tensor=sample["x_salinity"]
            )
            y_salinity_psu = salinity_normalize(
                mode="denorm", tensor=sample["y_salinity"]
            )
            eo_salinity_psu = salinity_normalize(mode="denorm", tensor=sample["eo"][1])
            self.assertTrue(
                torch.allclose(
                    x_salinity_psu[:, 0, 0],
                    torch.tensor([35.0, 36.0], dtype=torch.float32),
                    atol=0.05,
                )
            )
            self.assertAlmostEqual(float(y_salinity_psu[0, 0, 0]), 35.2, delta=0.05)
            self.assertAlmostEqual(float(y_salinity_psu[1, 0, 0]), 36.2, delta=0.05)
            self.assertAlmostEqual(float(eo_salinity_psu[0, 0]), 35.2, delta=0.05)

    def test_land_mask_fallback_uses_ostia_then_on_disk_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
            )
            row = dataset.rows[0]
            eo_np = np.asarray(
                [[[1.0, np.nan], [2.0, 3.0]]],
                dtype=np.float32,
            )

            ostia_mask = dataset._build_land_mask_patch(
                row,
                y_valid_mask_np=None,
                eo_np=eo_np,
            )
            disk_mask = dataset._build_land_mask_patch(
                row,
                y_valid_mask_np=None,
                eo_np=None,
            )
            dataset.land_mask_path = Path(tmpdir) / "missing_land_mask.tif"

            self.assertTrue(
                torch.equal(
                    torch.from_numpy(ostia_mask),
                    torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float32),
                )
            )
            self.assertTrue(
                torch.equal(
                    torch.from_numpy(disk_mask),
                    torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Could not build land_mask"):
                dataset._build_land_mask_patch(
                    row,
                    y_valid_mask_np=None,
                    eo_np=None,
                )

    def test_salinity_side_channels_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=True,
            )

            sample = dataset[0]

            self.assertFalse(dataset.include_salinity)
            self.assertEqual(sample["x"].shape, (2, 2, 2))
            self.assertEqual(sample["y"].shape, (2, 2, 2))
            for key in (
                "x_salinity",
                "y_salinity",
                "x_salinity_valid_mask",
                "y_salinity_valid_mask",
                "x_salinity_valid_mask_1d",
            ):
                self.assertNotIn(key, sample)

    def test_train_builder_wires_argo_geotiff_gridded_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir, cache_dir, _land_mask_path = _make_geotiff_dataset(tmp_path)
            config_land_mask_path = _write_land_mask(
                tmp_path / "config_land_mask.tif",
                values=np.zeros((2, 2), dtype=np.uint8),
            )
            config_path = tmp_path / "data.yaml"
            payload = {
                "dataset": {
                    "core": {
                        "dataset_variant": "argo_geotiff_gridded",
                        "geotiff_root_dir": str(output_dir),
                        "metadata_cache_dir": str(cache_dir),
                    },
                    "grid": {
                        "tile_size": 2,
                        "resolution_deg": 1.0,
                        "patch_grid_source": "land_mask",
                        "land_mask_path": str(config_land_mask_path),
                        "patch_stride": 2,
                        "max_land_fraction": 1.0,
                    },
                    "selection": {
                        "require_argo_for_train": True,
                        "require_argo_for_val": False,
                        "require_argo_for_all": False,
                    },
                    "synthetic_target": {"enabled": False},
                    "output": {
                        "return_info": True,
                        "return_coords": False,
                        "include_salinity": True,
                    },
                    "runtime": {"random_seed": 7, "cache_size": 2},
                },
                "split": {"val_fraction": 0.0, "val_year": 2018},
            }
            _write_yaml(config_path, payload)

            dataset = build_dataset(str(config_path), payload["dataset"], split="train")

            self.assertIsInstance(dataset, ArgoGeoTIFFGriddedPatchDataset)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)
            self.assertFalse(dataset.synthetic_target_enabled)
            self.assertTrue(dataset.include_salinity)
            self.assertIn("x_salinity", dataset[0])
            sample = dataset[0]
            self.assertTrue(
                torch.equal(
                    sample["land_mask"],
                    torch.ones((1, 2, 2), dtype=torch.float32),
                )
            )
            self.assertNotIn("output_land_mask", sample)

    def test_finetune_sampling_disabled_leaves_rows_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            base = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=False,
            )
            disabled = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=False,
                finetune_sampling={
                    "enabled": False,
                    "hard_fraction": 0.75,
                    "hard_regions": [
                        {
                            "name": "top_row",
                            "lon_min": 10.0,
                            "lon_max": 12.0,
                            "lat_min": 1.0,
                            "lat_max": 2.0,
                        }
                    ],
                },
            )

            self.assertEqual(len(base), 4)
            self.assertEqual(
                [(row["patch_id"], row["date"]) for row in disabled.rows],
                [(row["patch_id"], row["date"]) for row in base.rows],
            )
            self.assertFalse(disabled.finetune_sampling_summary["applied"])
            self.assertEqual(disabled.finetune_sampling_summary["total_rows"], 4)

    def test_finetune_sampling_keeps_target_hard_easy_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=1.0,
                val_year=2018,
                require_argo_for_train=False,
                random_seed=11,
                finetune_sampling={
                    "enabled": True,
                    "hard_fraction": 0.75,
                    "relax_land_filter": False,
                    "hard_regions": [
                        {
                            "name": "top_row",
                            "lon_min": 10.0,
                            "lon_max": 12.0,
                            "lat_min": 1.0,
                            "lat_max": 2.0,
                        },
                        {
                            "name": "bottom_left",
                            "lon_min": 10.0,
                            "lon_max": 11.0,
                            "lat_min": 0.0,
                            "lat_max": 1.0,
                        },
                    ],
                },
            )

            summary = dataset.finetune_sampling_summary
            self.assertTrue(summary["applied"])
            self.assertEqual(summary["hard_rows"], 3)
            self.assertEqual(summary["easy_rows"], 1)
            self.assertEqual(summary["total_rows"], 4)
            self.assertAlmostEqual(summary["actual_hard_fraction"], 0.75)
            self.assertEqual(len(dataset), 4)

    def test_finetune_sampling_leaves_validation_split_unchanged_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="val",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=0.0,
                val_year=2024,
                require_argo_for_val=False,
                finetune_sampling={
                    "enabled": True,
                    "hard_fraction": 0.75,
                    "apply_to_splits": ["train"],
                    "relax_land_filter": True,
                    "hard_regions": [
                        {
                            "name": "land_heavy_corner",
                            "lon_min": 11.0,
                            "lon_max": 12.0,
                            "lat_min": 0.0,
                            "lat_max": 1.0,
                            "max_land_fraction": 1.0,
                        }
                    ],
                },
            )

            self.assertEqual(len(dataset), 3)
            self.assertTrue(dataset.finetune_sampling_summary["enabled"])
            self.assertFalse(dataset.finetune_sampling_summary["applied"])
            self.assertEqual(dataset.finetune_sampling_summary["split"], "val")

    def test_finetune_regions_relax_land_fraction_filter_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            base = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=0.0,
                val_year=2018,
                require_argo_for_train=False,
            )
            finetune = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="train",
                tile_size=1,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=1,
                max_land_fraction=0.0,
                val_year=2018,
                require_argo_for_train=False,
                finetune_sampling={
                    "enabled": True,
                    "hard_fraction": 1.0,
                    "relax_land_filter": True,
                    "hard_regions": [
                        {
                            "name": "land_heavy_corner",
                            "lon_min": 11.0,
                            "lon_max": 12.0,
                            "lat_min": 0.0,
                            "lat_max": 1.0,
                            "max_land_fraction": 1.0,
                        }
                    ],
                },
            )

            self.assertEqual(len(base), 3)
            self.assertEqual(len(finetune), 1)
            self.assertAlmostEqual(float(finetune.rows[0]["land_fraction"]), 1.0)
            self.assertTrue(bool(finetune.rows[0]["force_included"]))
            self.assertEqual(
                finetune.rows[0]["force_include_region"],
                "land_heavy_corner",
            )

    def test_finetune_sampling_raises_when_no_hard_rows_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(Path(tmpdir))
            with self.assertRaisesRegex(RuntimeError, "matched no rows"):
                ArgoGeoTIFFGriddedPatchDataset(
                    geotiff_root_dir=output_dir,
                    metadata_cache_dir=cache_dir,
                    split="train",
                    tile_size=1,
                    resolution_deg=1.0,
                    land_mask_path=land_mask_path,
                    patch_stride=1,
                    max_land_fraction=1.0,
                    val_year=2018,
                    require_argo_for_train=False,
                    finetune_sampling={
                        "enabled": True,
                        "hard_fraction": 0.75,
                        "relax_land_filter": False,
                        "hard_regions": [
                            {
                                "name": "missing",
                                "lon_min": 120.0,
                                "lon_max": 121.0,
                                "lat_min": 40.0,
                                "lat_max": 41.0,
                            }
                        ],
                    },
                )

    def test_active_geotiff_config_uses_land_mask_grid_defaults(self) -> None:
        with packaged_config_path("px_space", "training_super_config.yaml").open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = yaml.safe_load(f)

        self.assertEqual(
            payload["data"]["dataset"]["core"]["dataset_variant"],
            "argo_geotiff_gridded",
        )
        core = payload["data"]["dataset"]["core"]
        self.assertEqual(
            core["geotiff_root_dir"], "/work/data/OceanVariableReconstruction"
        )
        self.assertEqual(
            core["metadata_cache_dir"],
            "/work/data/OceanVariableReconstruction/depthdif_cache",
        )
        grid = payload["data"]["dataset"]["grid"]
        self.assertEqual(grid["patch_grid_source"], "land_mask")
        self.assertEqual(grid["patch_stride"], 32)
        self.assertEqual(float(grid["max_land_fraction"]), 0.30)
        self.assertEqual(grid["land_mask_path"], "masks/world_land_mask_glorys_0p1.tif")
        self.assertEqual(
            [region["name"] for region in grid["force_include_regions"]],
            ["mediterranean", "baltic", "red_sea", "hudson_bay"],
        )
        selection = payload["data"]["dataset"]["selection"]
        self.assertFalse(selection["require_argo_for_train"])
        self.assertFalse(selection["filter_bad_argo_quality"])
        self.assertEqual(selection["accepted_argo_qc_flags"], [1, 2])
        finetune = payload["data"]["dataset"]["finetune_sampling"]
        self.assertFalse(finetune["enabled"])
        self.assertAlmostEqual(float(finetune["hard_fraction"]), 0.5)
        self.assertEqual(finetune["apply_to_splits"], ["train", "val"])
        self.assertTrue(finetune["relax_land_filter"])
        self.assertAlmostEqual(float(finetune["default_max_land_fraction"]), 0.85)
        self.assertEqual(
            [region["name"] for region in finetune["hard_regions"]],
            [
                "mediterranean",
                "black_sea",
                "russian_arctic_west",
                "russian_arctic_east",
                "hudson_bay",
                "baja_california",
                "greenland",
                "svalbard_barents",
                "european_atlantic",
                "persian_gulf",
                "red_sea",
                "yellow_sea",
                "sea_of_japan",
                "sea_of_okhotsk",
                "northern_alaska",
                "bay_of_benin_west",
                "bay_of_benin_east",
                "new_york_nova_scotia",
            ],
        )
        self.assertNotIn("target_source", payload["data"]["dataset"]["sampling"])
        self.assertEqual(
            payload["data"]["dataset"]["surface_conditioning"]["sources"],
            ["sst"],
        )
        self.assertEqual(payload["data"]["split"]["val_year"], 2016)
