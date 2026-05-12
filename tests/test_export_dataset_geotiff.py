from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import yaml

from data.dataset_creation.export_dataset_geotiff import export_training_geotiff_dataset
from data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff import (
    SEA_HEIGHT_STRETCH,
    SALINITY_STRETCH,
    STRETCH_SPECS,
    TEMPERATURE_KELVIN_STRETCH,
    decode_stretched_uint8,
)


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


def _write_glorys(path: Path, *, date_value: int) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path.parent / f"glorys_{int(date_value)}.nc", engine="h5netcdf")


def _write_ostia(
    root_dir: Path,
    *,
    date_value: int,
    base: float,
    values_are_kelvin: bool = True,
) -> None:
    """Write a tiny OSTIA-like surface source file."""
    lat = np.asarray([0.5, 1.5], dtype=np.float32)
    lon = np.asarray([10.5, 11.5], dtype=np.float32)
    offsets = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    values = (np.float32(base) + offsets)[None, ...]
    _ = values_are_kelvin
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
            "profile_date": (("profile",), np.asarray([20240108, 20240101])),
            "profile_idx": (("profile",), np.asarray([7, 8], dtype=np.int32)),
            "latitude": (("profile",), np.asarray([1.5, 1.5], dtype=np.float32)),
            "longitude": (("profile",), np.asarray([10.5, 10.5], dtype=np.float32)),
            "argo_temp_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32),
            ),
            "argo_psal_on_glorys_depth": (
                ("profile", "glorys_depth"),
                np.asarray([[35.0, 36.0], [35.5, 36.5]], dtype=np.float32),
            ),
        },
        coords={
            "profile": np.asarray([0, 1], dtype=np.int64),
            "glorys_depth": np.asarray([0.0, 10.0], dtype=np.float32),
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=2)


def _make_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Create all synthetic source roots for a GeoTIFF export."""
    glorys_dir = tmp_path / "glorys"
    ostia_dir = tmp_path / "ostia"
    sealevel_dir = tmp_path / "sealevel"
    land_mask_path = _write_land_mask(tmp_path / "land_mask.tif")
    enriched_argo = tmp_path / "aligned_argo" / "enriched_argo_profiles.zarr"
    _write_glorys(glorys_dir / "ignored.nc", date_value=20240108)
    for date_value, ostia_base, sea_base in (
        (20240105, 280.0, 0.0),
        (20240108, 290.0, 1.0),
        (20240111, 300.0, 2.0),
    ):
        _write_ostia(ostia_dir, date_value=date_value, base=ostia_base)
        _write_sealevel(sealevel_dir, date_value=date_value, base=sea_base)
    _write_enriched_argo_zarr(enriched_argo)
    return glorys_dir, ostia_dir, sealevel_dir, land_mask_path, enriched_argo


class TestExportDatasetGeoTiff(unittest.TestCase):
    def test_export_writes_aligned_uint8_rasters_and_preprocessed_argo(self) -> None:
        """Dense rasters share a grid and ARGO profiles are grid indexed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            glorys_dir, ostia_dir, sealevel_dir, land_mask_path, enriched_argo = (
                _make_sources(tmp_path)
            )
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
                workers=2,
                overwrite=True,
            )

            thetao_path = output_dir / "rasters/glorys/thetao/thetao_20240108.tif"
            so_path = output_dir / "rasters/glorys/so/so_20240108.tif"
            ostia_path = (
                output_dir / "rasters/ostia/analysed_sst/analysed_sst_20240108.tif"
            )
            adt_path = output_dir / "rasters/sealevel/adt/adt_20240108.tif"

            with (
                rasterio.open(thetao_path) as thetao,
                rasterio.open(so_path) as salinity,
                rasterio.open(ostia_path) as ostia,
                rasterio.open(adt_path) as adt,
            ):
                self.assertEqual(thetao.dtypes, ("uint8", "uint8"))
                self.assertEqual(thetao.nodata, 255.0)
                self.assertEqual(thetao.count, 2)
                self.assertEqual(thetao.descriptions[0], "depth_0_m")
                self.assertEqual(thetao.transform, ostia.transform)
                self.assertEqual(thetao.transform, adt.transform)
                self.assertEqual(thetao.crs, ostia.crs)
                self.assertEqual(thetao.crs, adt.crs)

                thetao_k = decode_stretched_uint8(
                    thetao.read(1),
                    STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
                )
                self.assertAlmostEqual(float(thetao_k[0, 0]), 285.15, delta=0.2)
                self.assertEqual(int(thetao.read(1)[1, 1]), 255)
                self.assertEqual(thetao.tags(1)["units"], "K")
                self.assertEqual(thetao.tags()["storage_dtype"], "uint8")
                self.assertEqual(thetao.tags()["valid_code_max"], "254")
                self.assertAlmostEqual(
                    float(thetao.tags()["max_abs_quantization_error"]),
                    0.0748,
                    delta=0.001,
                )

                salinity_psu = decode_stretched_uint8(
                    salinity.read(1),
                    STRETCH_SPECS[SALINITY_STRETCH],
                )
                self.assertAlmostEqual(float(salinity_psu[0, 0]), 35.2, delta=0.05)

                ostia_k = decode_stretched_uint8(
                    ostia.read(1),
                    STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
                )
                self.assertAlmostEqual(float(ostia_k[0, 0]), 292.0, delta=0.2)

                adt_m = decode_stretched_uint8(
                    adt.read(1),
                    STRETCH_SPECS[SEA_HEIGHT_STRETCH],
                )
                self.assertAlmostEqual(float(adt_m[0, 0]), 1.2, delta=0.03)

            argo = xr.open_zarr(
                output_dir / "argo/argo_profiles_on_grid.zarr",
                consolidated=None,
            )
            self.assertEqual(int(argo.sizes["profile"]), 1)
            self.assertEqual(int(argo["target_date"].values[0]), 20240108)
            self.assertEqual(int(argo["grid_row"].values[0]), 0)
            self.assertEqual(int(argo["grid_col"].values[0]), 0)
            argo_temp_k = decode_stretched_uint8(
                argo["argo_temp_kelvin_uint8"].values,
                STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
            )
            argo_psal = decode_stretched_uint8(
                argo["argo_psal_uint8"].values,
                STRETCH_SPECS[SALINITY_STRETCH],
            )
            self.assertAlmostEqual(float(argo_temp_k[0, 0]), 283.15, delta=0.2)
            self.assertAlmostEqual(float(argo_psal[0, 1]), 36.0, delta=0.05)
            self.assertTrue(bool(argo["argo_temp_valid"].values[0, 0]))
            argo.close()

            manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text())
            self.assertEqual(manifest["output_dir"], str(output_dir))
            self.assertEqual(manifest["stretch"]["temperature_kelvin"]["units"], "K")
            self.assertEqual(
                manifest["stretch"]["temperature_kelvin"]["storage_dtype"],
                "uint8",
            )
            self.assertAlmostEqual(
                manifest["stretch"]["salinity"]["max_abs_quantization_error"],
                0.0197,
                delta=0.001,
            )
            self.assertEqual(manifest["parallelism"]["raster_workers"], 2)
            self.assertEqual(manifest["argo"]["profile_count"], 1)

    def test_ostia_celsius_values_are_saved_as_kelvin(self) -> None:
        """Celsius-looking OSTIA values are converted before uint8 stretching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            glorys_dir = tmp_path / "glorys"
            ostia_dir = tmp_path / "ostia"
            sealevel_dir = tmp_path / "sealevel"
            land_mask_path = _write_land_mask(tmp_path / "land_mask.tif")
            _write_glorys(glorys_dir / "ignored.nc", date_value=20240108)
            _write_ostia(
                ostia_dir,
                date_value=20240108,
                base=15.0,
                values_are_kelvin=False,
            )
            _write_sealevel(sealevel_dir, date_value=20240108, base=0.0)

            output_dir = export_training_geotiff_dataset(
                glorys_dir=glorys_dir,
                ostia_dir=ostia_dir,
                sealevel_dir=sealevel_dir,
                land_mask_path=land_mask_path,
                output_dir=tmp_path / "geotiff_training",
                start_date=20240108,
                end_date=20240108,
                argo_source="none",
                workers=1,
                overwrite=True,
            )

            with rasterio.open(
                output_dir / "rasters/ostia/analysed_sst/analysed_sst_20240108.tif"
            ) as ostia:
                decoded = decode_stretched_uint8(
                    ostia.read(1),
                    STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
                )
                self.assertAlmostEqual(float(decoded[0, 0]), 290.15, delta=0.2)


if __name__ == "__main__":
    unittest.main()
