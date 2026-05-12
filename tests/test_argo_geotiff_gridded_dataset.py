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
from train import build_dataset
from depth_recon.utils.normalizations import temperature_normalize


def _write_land_mask(path: Path) -> Path:
    """Write a tiny authoritative raster grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        dst.write(np.zeros((2, 2), dtype=np.uint8), 1)
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
            "profile_date": (("profile",), np.asarray([20240108], dtype=np.int32)),
            "profile_idx": (("profile",), np.asarray([7], dtype=np.int32)),
            "latitude": (("profile",), np.asarray([1.5], dtype=np.float32)),
            "longitude": (("profile",), np.asarray([10.5], dtype=np.float32)),
            "argo_temp_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[10.0, 20.0]], dtype=np.float32),
            ),
            "argo_psal_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[35.0, 36.0]], dtype=np.float32),
            ),
        },
        coords={
            "profile": np.asarray([0], dtype=np.int64),
            "glorys_depth": np.asarray([0.0, 10.0], dtype=np.float32),
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def _make_geotiff_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Export a tiny GeoTIFF training dataset and return root/cache/land paths."""
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    sealevel_dir = tmp_path / "sealevel"
    land_mask_path = _write_land_mask(tmp_path / "land_mask.tif")
    enriched_argo = tmp_path / "aligned_argo" / "enriched_argo_profiles.zarr"
    _write_glorys(glorys_dir, date_value=20240108)
    for date_value, ostia_base, sea_base in (
        (20240105, 280.0, 0.0),
        (20240108, 290.0, 1.0),
        (20240111, 300.0, 2.0),
    ):
        _write_ostia(ostia_dir, date_value=date_value, base=ostia_base)
        _write_sealevel(sealevel_dir, date_value=date_value, base=sea_base)
    _write_enriched_argo_zarr(enriched_argo)
    output_dir = export_training_geotiff_dataset(
        glorys_dir=glorys_dir,
        ostia_dir=ostia_dir,
        sealevel_dir=sealevel_dir,
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
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]

            self.assertEqual(sample["eo"].shape, (1, 2, 2))
            self.assertEqual(sample["x"].shape, (2, 2, 2))
            self.assertEqual(sample["y"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["y_valid_mask"].shape, (2, 2, 2))
            self.assertEqual(sample["x_valid_mask_1d"].shape, (1, 2, 2))
            self.assertEqual(sample["land_mask"].shape, (1, 2, 2))
            self.assertEqual(sample["date"], 20240108)
            self.assertTrue(
                torch.allclose(sample["coords"], torch.tensor([1.0, 11.0]), atol=1e-5)
            )

            x_c = temperature_normalize(mode="denorm", tensor=sample["x"])
            y_c = temperature_normalize(mode="denorm", tensor=sample["y"])
            eo_c = temperature_normalize(mode="denorm", tensor=sample["eo"])
            self.assertTrue(bool(sample["x_valid_mask"][:, 0, 0].all().item()))
            self.assertFalse(bool(sample["x_valid_mask"][:, 0, 1].any().item()))
            self.assertTrue(
                torch.allclose(
                    x_c[:, 0, 0],
                    torch.tensor([10.0, 20.0], dtype=torch.float32),
                    atol=0.25,
                )
            )
            self.assertAlmostEqual(float(y_c[0, 0, 0]), 12.0, delta=0.25)
            self.assertAlmostEqual(float(y_c[1, 0, 0]), 22.0, delta=0.25)
            self.assertAlmostEqual(float(eo_c[0, 0, 0]), 18.85, delta=0.25)
            self.assertFalse(bool(sample["y_valid_mask"][0, 1, 1].item()))
            self.assertEqual(sample["info"]["x_source"], "argo")

    def test_train_builder_wires_argo_geotiff_gridded_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(tmp_path)
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
                        "land_mask_path": str(land_mask_path),
                        "patch_stride": 2,
                        "max_land_fraction": 1.0,
                    },
                    "selection": {
                        "require_argo_for_train": True,
                        "require_argo_for_val": False,
                        "require_argo_for_all": False,
                    },
                    "synthetic": {"enabled": False, "pixel_count": 1},
                    "output": {"return_info": True, "return_coords": False},
                    "runtime": {"random_seed": 7, "cache_size": 2},
                },
                "split": {"val_fraction": 0.0, "val_year": 2018},
            }
            _write_yaml(config_path, payload)

            dataset = build_dataset(str(config_path), payload["dataset"], split="train")

            self.assertIsInstance(dataset, ArgoGeoTIFFGriddedPatchDataset)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)
            self.assertFalse(dataset.synthetic_mode)

    def test_active_geotiff_config_uses_land_mask_grid_defaults(self) -> None:
        with packaged_config_path("px_space", "data_ostia_argo_geotiff.yaml").open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = yaml.safe_load(f)

        self.assertEqual(
            payload["dataset"]["core"]["dataset_variant"],
            "argo_geotiff_gridded",
        )
        grid = payload["dataset"]["grid"]
        self.assertEqual(grid["patch_grid_source"], "land_mask")
        self.assertEqual(grid["patch_stride"], 64)
        self.assertEqual(float(grid["max_land_fraction"]), 0.30)
        self.assertTrue(Path(grid["land_mask_path"]).exists())
        self.assertEqual(grid["force_include_regions"][0]["name"], "mediterranean")
        self.assertEqual(payload["split"]["val_year"], 2018)
