"""
Example:
 /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff \
   --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
   --ostia-dir /data1/datasets/depth_v2/ostia \
   --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
   --sss-dir /data1/datasets/depth_v2/sss_daily \
   --enriched-argo-zarr /work/data/depthdif/enriched_argo_profiles.zarr \
   --argo-dir /data1/datasets/depth_v2/en4_profiles \
   --land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
   --output-dir /work/data/depthdif \
   --start-date 20100101 \
   --end-date 20240731 \
   --surface-aggregate-days 7 \
   --argo-source enriched \
   --chunk-profile 50000 \
   --workers 12 \
   --rasters-only \
   --overwrite

Resume a partial raster export by replacing --overwrite with --skip-existing.

Export aligned uint8 GeoTIFF rasters and preprocessed ARGO profile inputs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence

from affine import Affine
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import xarray as xr
import yaml

SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from depth_recon.data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_LEVEL_QC_VARS,
    ARGO_PROFILE_QC_VARS,
    date_to_days_since_1950,
    filter_argo_files_by_date_range,
    scan_timed_files,
)
from depth_recon.data.netcdf_sources import (
    _align_argo_profile_to_glorys_depths,
)

DEFAULT_OUTPUT_DIR = Path("/work/data/depthdif")
DEFAULT_ENRICHED_ARGO_ZARR = (
    DEFAULT_OUTPUT_DIR / "aligned_argo" / "enriched_argo_profiles.zarr"
)
DEFAULT_LAND_MASK_PATH = (
    Path(__file__).resolve().parents[2]
    / "dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif"
)
DEFAULT_GLORYS_DIR = Path("/data1/datasets/depth_v2/glorys_weekly")
DEFAULT_OSTIA_DIR = Path("/data1/datasets/depth_v2/ostia")
DEFAULT_SEALEVEL_DIR = Path("/data1/datasets/depth_v2/sealevel_daily")
DEFAULT_SSS_DIR = Path("/data1/datasets/depth_v2/sss_daily")
DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_START_DATE = 20100101
DEFAULT_END_DATE = 20240731
DEFAULT_SURFACE_AGGREGATE_DAYS = 7
DEFAULT_CHUNK_PROFILE = 50000
DEFAULT_RASTER_WORKERS = max(1, min(4, os.cpu_count() or 1))

RASTER_DTYPE = "uint8"
VALID_CODE_MIN = 0
NODATA_CODE = np.uint8(255)
MISSING_QC_FLAG = np.int8(-1)
VALID_CODE_MAX = np.float32(254.0)
TEMPERATURE_KELVIN_STRETCH = "temperature_kelvin"
SALINITY_STRETCH = "salinity"
SEA_HEIGHT_STRETCH = "sea_height"
DENSITY_STRETCH = "density"
COMPACT_LEVEL_QC_NAMES = (
    "argo_depth_qc_on_glorys_depth",
    "argo_temp_qc_on_glorys_depth",
    "argo_psal_qc_on_glorys_depth",
)
COMPACT_TEMP_PROFILE_QC_NAME = "argo_temp_profile_qc"
COMPACT_PSAL_PROFILE_QC_NAME = "argo_psal_profile_qc"


@dataclass(frozen=True)
class StretchSpec:
    """Fixed uint8 encoding range for one physical variable family."""

    name: str
    minimum: float
    maximum: float
    units: str
    nodata: int = 255


@dataclass(frozen=True)
class EncodeStats:
    """Summary of one uint8 stretch operation."""

    valid_count: int
    nodata_count: int
    clipped_low_count: int
    clipped_high_count: int


@dataclass(frozen=True)
class TargetGrid:
    """Raster grid metadata and 1D pixel-center coordinate axes."""

    width: int
    height: int
    transform: Affine
    crs: Any
    lon_axis: np.ndarray
    lat_axis: np.ndarray


class DatasetCache:
    """Small LRU cache for NetCDF datasets opened during export."""

    def __init__(self, max_open: int = 8) -> None:
        """Initialize a bounded path-to-dataset cache."""
        self.max_open = int(max_open)
        self._items: OrderedDict[Path, xr.Dataset] = OrderedDict()

    def get(self, path: Path) -> xr.Dataset:
        """Return an open dataset for ``path``, opening it if necessary."""
        path = Path(path)
        if path in self._items:
            ds = self._items.pop(path)
            self._items[path] = ds
            return ds
        ds = xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        )
        self._items[path] = ds
        while len(self._items) > self.max_open:
            _, old = self._items.popitem(last=False)
            old.close()
        return ds

    def close(self) -> None:
        """Close all cached datasets."""
        for ds in self._items.values():
            ds.close()
        self._items.clear()


STRETCH_SPECS = {
    TEMPERATURE_KELVIN_STRETCH: StretchSpec(
        TEMPERATURE_KELVIN_STRETCH,
        270.15,
        308.15,
        "K",
    ),
    SALINITY_STRETCH: StretchSpec(SALINITY_STRETCH, 30.0, 40.0, "PSU"),
    SEA_HEIGHT_STRETCH: StretchSpec(SEA_HEIGHT_STRETCH, -2.0, 2.0, "m"),
    DENSITY_STRETCH: StretchSpec(DENSITY_STRETCH, 1000.0, 1035.0, "kg/m3"),
}
SSS_RASTER_VARS = ("sos", "dos")
SSS_SURFACE_STRETCHES = {
    "sos": SALINITY_STRETCH,
    "dos": DENSITY_STRETCH,
}


def _date_int_from_days_since_1950(day_value: float) -> int:
    """Convert numeric days since 1950-01-01 to compact YYYYMMDD."""
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


def _filter_timed_files(
    items: Sequence[Any],
    *,
    start_date: int | None,
    end_date: int | None,
) -> list[Any]:
    """Keep timed files whose parsed date falls in the requested range."""
    selected: list[Any] = []
    for item in items:
        date_value = _date_int_from_days_since_1950(float(item.day))
        if start_date is not None and date_value < int(start_date):
            continue
        if end_date is not None and date_value > int(end_date):
            continue
        selected.append(item)
    return selected


def _load_target_grid(land_mask_path: Path) -> TargetGrid:
    """Load the authoritative raster grid from the land-mask GeoTIFF."""
    with rasterio.open(land_mask_path) as src:
        transform = src.transform
        if not np.isclose(float(transform.b), 0.0) or not np.isclose(
            float(transform.d), 0.0
        ):
            raise RuntimeError("Rotated/sheared land-mask grids are not supported.")
        if float(transform.a) <= 0.0 or float(transform.e) >= 0.0:
            raise RuntimeError("Expected a north-up land-mask GeoTIFF grid.")
        width = int(src.width)
        height = int(src.height)
        lon_axis = transform.c + (
            (np.arange(width, dtype=np.float64) + 0.5) * transform.a
        )
        lat_axis = transform.f + (
            (np.arange(height, dtype=np.float64) + 0.5) * transform.e
        )
        return TargetGrid(
            width=width,
            height=height,
            transform=transform,
            crs=src.crs,
            lon_axis=lon_axis,
            lat_axis=lat_axis,
        )


def encode_stretched_uint8(
    values: np.ndarray,
    stretch: StretchSpec,
) -> tuple[np.ndarray, EncodeStats]:
    """Encode physical values into uint8 using a fixed stretch specification."""
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, int(stretch.nodata), dtype=np.uint8)
    clipped_low = finite & (arr < np.float32(stretch.minimum))
    clipped_high = finite & (arr > np.float32(stretch.maximum))
    if np.any(finite):
        clipped = np.clip(arr[finite], stretch.minimum, stretch.maximum)
        scaled = np.rint(
            (
                (clipped - np.float32(stretch.minimum))
                / np.float32(stretch.maximum - stretch.minimum)
            )
            * VALID_CODE_MAX
        )
        out[finite] = scaled.astype(np.uint8, copy=False)
    return (
        out,
        EncodeStats(
            valid_count=int(np.count_nonzero(finite)),
            nodata_count=int(arr.size - np.count_nonzero(finite)),
            clipped_low_count=int(np.count_nonzero(clipped_low)),
            clipped_high_count=int(np.count_nonzero(clipped_high)),
        ),
    )


def decode_stretched_uint8(values: np.ndarray, stretch: StretchSpec) -> np.ndarray:
    """Decode uint8 values back into their physical units."""
    arr = np.asarray(values, dtype=np.uint8)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = arr != int(stretch.nodata)
    out[valid] = np.float32(stretch.minimum) + (
        arr[valid].astype(np.float32)
        / VALID_CODE_MAX
        * np.float32(stretch.maximum - stretch.minimum)
    )
    return out


def _merge_stats(stats: Iterable[EncodeStats]) -> dict[str, int]:
    """Combine per-band encode statistics into one manifest-friendly mapping."""
    items = list(stats)
    return {
        "valid_count": int(sum(item.valid_count for item in items)),
        "nodata_count": int(sum(item.nodata_count for item in items)),
        "clipped_low_count": int(sum(item.clipped_low_count for item in items)),
        "clipped_high_count": int(sum(item.clipped_high_count for item in items)),
    }


def _quantization_step(stretch: StretchSpec) -> float:
    """Return the physical-unit distance between adjacent valid uint8 codes."""
    return float(stretch.maximum - stretch.minimum) / float(VALID_CODE_MAX)


def _max_quantization_error(stretch: StretchSpec) -> float:
    """Return the worst-case rounding error introduced by uint8 encoding."""
    return _quantization_step(stretch) / 2.0


def _stretch_manifest(stretch: StretchSpec) -> dict[str, Any]:
    """Return a YAML-safe representation of a stretch specification."""
    return {
        "name": stretch.name,
        "storage_dtype": RASTER_DTYPE,
        "minimum": float(stretch.minimum),
        "maximum": float(stretch.maximum),
        "units": stretch.units,
        "nodata": int(stretch.nodata),
        "valid_code_min": int(VALID_CODE_MIN),
        "valid_code_max": int(VALID_CODE_MAX),
        "valid_code_count": int(VALID_CODE_MAX) + 1,
        "quantization_step": _quantization_step(stretch),
        "max_abs_quantization_error": _max_quantization_error(stretch),
        "decode": "minimum + code / 254 * (maximum - minimum)",
    }


def _first_present_name(names: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Return the first candidate present in ``names``."""
    available = set(str(name) for name in names)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _lon_axis_for_source(
    da: xr.DataArray, lon_name: str, grid: TargetGrid
) -> np.ndarray:
    """Map output-grid longitudes into the source longitude convention."""
    source_lons = np.asarray(da[lon_name].values, dtype=np.float64)
    if source_lons.size > 0 and np.nanmin(source_lons) >= 0.0:
        return np.mod(grid.lon_axis, 360.0)
    return grid.lon_axis


def _coordinate_tolerance(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
) -> float:
    """Return a small coordinate tolerance for exact-grid label selection."""
    steps: list[float] = []
    for coord_name in (lat_name, lon_name):
        values = np.sort(
            np.asarray(da[coord_name].values, dtype=np.float64).reshape(-1)
        )
        diffs = np.diff(values[np.isfinite(values)])
        diffs = np.abs(diffs[diffs != 0.0])
        if diffs.size > 0:
            steps.append(float(np.nanmin(diffs)))
    return max(1.0e-8, min(steps) * 1.0e-6) if steps else 1.0e-8


def _axis_is_on_source_grid(
    source_values: np.ndarray,
    requested_values: np.ndarray,
    tolerance: float,
) -> bool:
    """Return true when every requested coordinate lands on the source axis."""
    source = np.sort(np.asarray(source_values, dtype=np.float64).reshape(-1))
    source = source[np.isfinite(source)]
    requested = np.asarray(requested_values, dtype=np.float64).reshape(-1)
    if source.size == 0 or requested.size == 0:
        return False
    if not np.all(np.isfinite(requested)):
        return False
    positions = np.searchsorted(source, requested)
    positions = np.clip(positions, 0, int(source.size) - 1)
    prev_positions = np.clip(positions - 1, 0, int(source.size) - 1)
    nearest = np.minimum(
        np.abs(source[positions] - requested),
        np.abs(source[prev_positions] - requested),
    )
    return bool(np.all(nearest <= float(tolerance)))


def _axes_are_on_source_grid(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
) -> bool:
    """Return true when requested horizontal axes are already source pixels."""
    tolerance = _coordinate_tolerance(da, lat_name, lon_name)
    return _axis_is_on_source_grid(
        da[lat_name].values,
        lat_axis,
        tolerance,
    ) and _axis_is_on_source_grid(
        da[lon_name].values,
        lon_axis,
        tolerance,
    )


def _read_dataarray_on_grid(
    ds: xr.Dataset,
    var_name: str,
    grid: TargetGrid,
    *,
    depth_index: int | None = None,
) -> np.ndarray:
    """Read one 2D source variable slice interpolated onto the output grid."""
    if var_name not in ds:
        return np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    if depth_index is not None:
        if "depth" not in da.dims:
            raise RuntimeError(f"Variable {var_name!r} has no depth dimension.")
        da = da.isel(depth=int(depth_index))
    elif "depth" in da.dims:
        if int(da.sizes["depth"]) != 1:
            raise RuntimeError(
                f"Variable {var_name!r} has a depth dimension; pass depth_index."
            )
        # SSS surface products keep a singleton depth axis; raster export is 2D.
        da = da.isel(depth=0)

    lat_name = _first_present_name(da.dims, ("latitude", "lat"))
    lon_name = _first_present_name(da.dims, ("longitude", "lon"))
    if lat_name is None or lon_name is None:
        raise RuntimeError(f"Variable {var_name!r} has no latitude/longitude dims.")

    lon_axis = _lon_axis_for_source(da, lon_name, grid)
    if _axes_are_on_source_grid(da, lat_name, lon_name, grid.lat_axis, lon_axis):
        sampled = da.sel(
            {lat_name: grid.lat_axis, lon_name: lon_axis},
            method="nearest",
            tolerance=_coordinate_tolerance(da, lat_name, lon_name),
        )
    else:
        sampled = da.interp(
            {
                lat_name: grid.lat_axis,
                lon_name: lon_axis,
            },
            method="linear",
        )
    sampled = sampled.transpose(lat_name, lon_name)
    return np.asarray(sampled.values, dtype=np.float32)


def _temperature_to_kelvin(
    values: np.ndarray, *, source_is_celsius: bool
) -> np.ndarray:
    """Return temperatures in Kelvin, preserving NaN values."""
    arr = np.asarray(values, dtype=np.float32)
    if source_is_celsius:
        return arr + np.float32(273.15)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and float(np.nanmedian(finite)) < 100.0:
        return arr + np.float32(273.15)
    return arr


def _geotiff_profile(
    path: Path,
    *,
    count: int,
    grid: TargetGrid,
    compress: str,
) -> dict[str, Any]:
    """Build the common rasterio profile for uint8 outputs."""
    profile = {
        "driver": "GTiff",
        "height": int(grid.height),
        "width": int(grid.width),
        "count": int(count),
        "dtype": RASTER_DTYPE,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": int(NODATA_CODE),
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
        "interleave": "band",
    }
    if int(grid.width) >= 16 and int(grid.height) >= 16:
        # GeoTIFF tile dimensions must be multiples of 16 for broad GDAL support.
        profile.update(
            {
                "tiled": True,
                "blockxsize": min(256, int(grid.width)),
                "blockysize": min(256, int(grid.height)),
            }
        )
    return profile


def _write_geotiff_with_fallback(
    path: Path,
    *,
    count: int,
    grid: TargetGrid,
    writer: Any,
) -> str:
    """Write a GeoTIFF using ZSTD when available, otherwise DEFLATE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for compress in ("ZSTD", "DEFLATE"):
        try:
            with rasterio.open(
                path,
                "w",
                **_geotiff_profile(path, count=count, grid=grid, compress=compress),
            ) as dst:
                writer(dst)
            return compress
        except (RasterioIOError, ValueError):
            if path.exists():
                path.unlink()
            if compress == "DEFLATE":
                raise
    raise RuntimeError(f"Could not write GeoTIFF: {path}")


def _set_common_tags(
    dst: rasterio.io.DatasetWriter,
    *,
    source_product: str,
    variable: str,
    stretch: StretchSpec,
) -> None:
    """Attach shared stretch metadata to a GeoTIFF dataset."""
    dst.update_tags(
        source_product=source_product,
        variable=variable,
        stretch_name=stretch.name,
        storage_dtype=RASTER_DTYPE,
        stretch_min=float(stretch.minimum),
        stretch_max=float(stretch.maximum),
        stretch_units=stretch.units,
        nodata=int(stretch.nodata),
        valid_code_min=int(VALID_CODE_MIN),
        valid_code_max=int(VALID_CODE_MAX),
        valid_code_count=int(VALID_CODE_MAX) + 1,
        quantization_step=_quantization_step(stretch),
        max_abs_quantization_error=_max_quantization_error(stretch),
        decode_formula="minimum + code / 254 * (maximum - minimum)",
    )


def _output_relative(path: Path, output_dir: Path) -> str:
    """Return a stable manifest path relative to the export root."""
    return str(Path(path).relative_to(output_dir)).replace("\\", "/")


def _raster_stats_from_tags(tag_groups: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Read encode statistics from existing GeoTIFF dataset or band tags."""
    stats = []
    for tags in tag_groups:
        if "valid_count" not in tags:
            continue
        stats.append(
            EncodeStats(
                valid_count=int(tags.get("valid_count", 0)),
                nodata_count=int(tags.get("nodata_count", 0)),
                clipped_low_count=int(tags.get("clipped_low_count", 0)),
                clipped_high_count=int(tags.get("clipped_high_count", 0)),
            )
        )
    return _merge_stats(stats)


def _existing_raster_metadata(
    path: Path,
    *,
    count: int,
    grid: TargetGrid,
) -> tuple[str, dict[str, int]]:
    """Validate and summarize an existing raster used by resume mode."""
    with rasterio.open(path) as src:
        if int(src.width) != int(grid.width) or int(src.height) != int(grid.height):
            raise RuntimeError(f"Existing raster has the wrong grid shape: {path}")
        if int(src.count) != int(count):
            raise RuntimeError(f"Existing raster has the wrong band count: {path}")
        if tuple(src.dtypes) != tuple([RASTER_DTYPE] * int(count)):
            raise RuntimeError(f"Existing raster has the wrong dtype: {path}")
        if src.nodata is None or int(src.nodata) != int(NODATA_CODE):
            raise RuntimeError(f"Existing raster has the wrong nodata value: {path}")
        if src.crs != grid.crs:
            raise RuntimeError(f"Existing raster has the wrong CRS: {path}")
        if not np.allclose(src.transform.to_gdal(), grid.transform.to_gdal()):
            raise RuntimeError(f"Existing raster has the wrong transform: {path}")

        compression = getattr(src, "compression", None)
        compression_name = getattr(compression, "value", None) or src.profile.get(
            "compress",
            "existing",
        )
        tag_groups = [src.tags()] + [src.tags(i) for i in range(1, src.count + 1)]
        return str(compression_name).upper(), _raster_stats_from_tags(tag_groups)


def _copy_land_mask_to_output(
    *, land_mask_path: Path, output_dir: Path, overwrite: bool, skip_existing: bool
) -> Path:
    """Copy the authoritative land-mask GeoTIFF into the export folder."""
    source_path = Path(land_mask_path)
    target_path = Path(output_dir) / "masks" / source_path.name
    if source_path.resolve() == target_path.resolve():
        return target_path
    if target_path.exists() and not overwrite:
        if skip_existing:
            return target_path
        raise FileExistsError(f"Land-mask copy already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path


def _glorys_depth_axis(ds: xr.Dataset) -> np.ndarray:
    """Read the GLORYS depth axis from a source dataset."""
    if "depth" not in ds:
        raise RuntimeError("GLORYS source is missing a depth coordinate.")
    depth = np.asarray(ds["depth"].values, dtype=np.float32).reshape(-1)
    depth = depth[np.isfinite(depth)]
    if depth.size == 0:
        raise RuntimeError("GLORYS source has an empty depth coordinate.")
    return depth.astype(np.float32, copy=False)


def _export_glorys_variable(
    *,
    item: Any,
    cache: DatasetCache,
    output_dir: Path,
    grid: TargetGrid,
    var_name: str,
    stretch: StretchSpec,
    source_is_temperature_celsius: bool,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Export one GLORYS variable/date to a multiband uint8 GeoTIFF."""
    ds = cache.get(item.path)
    if var_name not in ds:
        raise RuntimeError(f"GLORYS variable {var_name!r} is missing from {item.path}")
    depth = _glorys_depth_axis(ds)
    date_value = _date_int_from_days_since_1950(float(item.day))
    out_path = (
        output_dir / "rasters" / "glorys" / var_name / f"{var_name}_{date_value}.tif"
    )
    if skip_existing and out_path.exists():
        compression, stats = _existing_raster_metadata(
            out_path,
            count=int(depth.size),
            grid=grid,
        )
        return {
            "date": int(date_value),
            "path": _output_relative(out_path, output_dir),
            "source_file": Path(item.path).name,
            "compression": compression,
            "band_count": int(depth.size),
            "stats": stats,
            "skipped_existing": True,
        }
    per_band_stats: list[EncodeStats] = []

    def writer(dst: rasterio.io.DatasetWriter) -> None:
        """Write all depth bands into an opened GLORYS GeoTIFF."""
        per_band_stats.clear()
        _set_common_tags(
            dst,
            source_product="glorys",
            variable=var_name,
            stretch=stretch,
        )
        dst.update_tags(source_file=Path(item.path).name, date=int(date_value))
        depth_values = depth.tolist()
        band_iter = tqdm(
            depth_values,
            desc=f"GLORYS {var_name} {date_value}",
            unit="band",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for band_idx, depth_value in enumerate(band_iter, start=1):
            values = _read_dataarray_on_grid(
                ds,
                var_name,
                grid,
                depth_index=band_idx - 1,
            )
            if source_is_temperature_celsius:
                values = _temperature_to_kelvin(values, source_is_celsius=True)
            encoded, stats = encode_stretched_uint8(values, stretch)
            per_band_stats.append(stats)
            dst.write(encoded, band_idx)
            dst.set_band_description(band_idx, f"depth_{float(depth_value):g}_m")
            dst.update_tags(
                band_idx,
                depth_m=float(depth_value),
                **_stretch_manifest(stretch),
                **{
                    "valid_count": stats.valid_count,
                    "nodata_count": stats.nodata_count,
                    "clipped_low_count": stats.clipped_low_count,
                    "clipped_high_count": stats.clipped_high_count,
                },
            )

    compression = _write_geotiff_with_fallback(
        out_path,
        count=int(depth.size),
        grid=grid,
        writer=writer,
    )
    return {
        "date": int(date_value),
        "path": _output_relative(out_path, output_dir),
        "source_file": Path(item.path).name,
        "compression": compression,
        "band_count": int(depth.size),
        "stats": _merge_stats(per_band_stats),
    }


def _surface_files_for_target(
    items: Sequence[Any],
    *,
    target_day: float,
    aggregate_days: int,
) -> list[Any]:
    """Select source files inside the centered aggregation window."""
    radius = max(0.0, (float(aggregate_days) - 1.0) / 2.0)
    return [
        item
        for item in items
        if abs(float(item.day) - float(target_day)) <= radius + 1.0e-8
    ]


def _read_surface_mean_on_grid(
    *,
    items: Sequence[Any],
    cache: DatasetCache,
    var_name: str,
    grid: TargetGrid,
    source_is_temperature: bool,
    progress_label: str | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Read and average surface files onto the output grid."""
    total = np.zeros((grid.height, grid.width), dtype=np.float64)
    count = np.zeros((grid.height, grid.width), dtype=np.uint16)
    iterator = tqdm(
        items,
        desc=progress_label or f"Aggregating {var_name}",
        unit="file",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for item in iterator:
        ds = cache.get(item.path)
        values = _read_dataarray_on_grid(ds, var_name, grid)
        if source_is_temperature:
            # OSTIA is normally Kelvin, but tiny tests and derived products can be Celsius.
            values = _temperature_to_kelvin(values, source_is_celsius=False)
        valid = np.isfinite(values)
        total[valid] += values[valid].astype(np.float64)
        count[valid] += 1
    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    valid_mean = count > 0
    out[valid_mean] = (total[valid_mean] / count[valid_mean].astype(np.float64)).astype(
        np.float32,
        copy=False,
    )
    return out


def _export_surface_variable(
    *,
    source_name: str,
    source_items: Sequence[Any],
    target_item: Any,
    cache: DatasetCache,
    output_dir: Path,
    grid: TargetGrid,
    var_name: str,
    stretch: StretchSpec,
    aggregate_days: int,
    source_is_temperature: bool,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Export one surface variable/date to a single-band uint8 GeoTIFF."""
    target_date = _date_int_from_days_since_1950(float(target_item.day))
    selected_items = _surface_files_for_target(
        source_items,
        target_day=float(target_item.day),
        aggregate_days=aggregate_days,
    )
    out_path = (
        output_dir
        / "rasters"
        / source_name
        / var_name
        / f"{var_name}_{target_date}.tif"
    )
    if skip_existing and out_path.exists():
        compression, stats = _existing_raster_metadata(out_path, count=1, grid=grid)
        return {
            "date": int(target_date),
            "path": _output_relative(out_path, output_dir),
            "source_files": [Path(item.path).name for item in selected_items],
            "compression": compression,
            "stats": stats,
            "skipped_existing": True,
        }
    values = _read_surface_mean_on_grid(
        items=selected_items,
        cache=cache,
        var_name=var_name,
        grid=grid,
        source_is_temperature=source_is_temperature,
        progress_label=f"{source_name} {var_name} {target_date}",
        show_progress=show_progress,
    )
    encoded, stats = encode_stretched_uint8(values, stretch)

    def writer(dst: rasterio.io.DatasetWriter) -> None:
        """Write the encoded surface raster into an opened GeoTIFF."""
        _set_common_tags(
            dst,
            source_product=source_name,
            variable=var_name,
            stretch=stretch,
        )
        dst.update_tags(
            date=int(target_date),
            aggregate_days=int(aggregate_days),
            source_files=",".join(Path(item.path).name for item in selected_items),
            **{
                "valid_count": stats.valid_count,
                "nodata_count": stats.nodata_count,
                "clipped_low_count": stats.clipped_low_count,
                "clipped_high_count": stats.clipped_high_count,
            },
        )
        dst.set_band_description(1, var_name)
        dst.write(encoded, 1)

    compression = _write_geotiff_with_fallback(
        out_path,
        count=1,
        grid=grid,
        writer=writer,
    )
    return {
        "date": int(target_date),
        "path": _output_relative(out_path, output_dir),
        "source_files": [Path(item.path).name for item in selected_items],
        "compression": compression,
        "stats": _merge_stats([stats]),
    }


def _export_weekly_raster_date_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Export all dense raster variables for one GLORYS weekly target date."""
    item = task["item"]
    output_root = Path(task["output_dir"])
    grid = _load_target_grid(Path(task["land_mask_path"]))
    cache = DatasetCache(max_open=8)
    try:
        date_value = _date_int_from_days_since_1950(float(item.day))
        result: dict[str, Any] = {"date": int(date_value)}
        result["thetao"] = _export_glorys_variable(
            item=item,
            cache=cache,
            output_dir=output_root,
            grid=grid,
            var_name="thetao",
            stretch=STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
            source_is_temperature_celsius=True,
            skip_existing=bool(task.get("skip_existing", False)),
            show_progress=False,
        )
        result["so"] = _export_glorys_variable(
            item=item,
            cache=cache,
            output_dir=output_root,
            grid=grid,
            var_name="so",
            stretch=STRETCH_SPECS[SALINITY_STRETCH],
            source_is_temperature_celsius=False,
            skip_existing=bool(task.get("skip_existing", False)),
            show_progress=False,
        )
        result["analysed_sst"] = _export_surface_variable(
            source_name="ostia",
            source_items=task["ostia_items"],
            target_item=item,
            cache=cache,
            output_dir=output_root,
            grid=grid,
            var_name="analysed_sst",
            stretch=STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
            aggregate_days=int(task["surface_aggregate_days"]),
            source_is_temperature=True,
            skip_existing=bool(task.get("skip_existing", False)),
            show_progress=False,
        )
        result["adt"] = _export_surface_variable(
            source_name="sealevel",
            source_items=task["sealevel_items"],
            target_item=item,
            cache=cache,
            output_dir=output_root,
            grid=grid,
            var_name="adt",
            stretch=STRETCH_SPECS[SEA_HEIGHT_STRETCH],
            aggregate_days=int(task["surface_aggregate_days"]),
            source_is_temperature=False,
            skip_existing=bool(task.get("skip_existing", False)),
            show_progress=False,
        )
        result["sss"] = {}
        for var_name in SSS_RASTER_VARS:
            result["sss"][var_name] = _export_surface_variable(
                source_name="sss",
                source_items=task["sss_items"],
                target_item=item,
                cache=cache,
                output_dir=output_root,
                grid=grid,
                var_name=var_name,
                stretch=STRETCH_SPECS[SSS_SURFACE_STRETCHES[var_name]],
                aggregate_days=int(task["surface_aggregate_days"]),
                source_is_temperature=False,
                skip_existing=bool(task.get("skip_existing", False)),
                show_progress=False,
            )
        return result
    finally:
        cache.close()


def _record_weekly_raster_result(
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Append one weekly raster export result to the output manifest."""
    manifest["rasters"]["glorys"]["thetao"].append(result["thetao"])
    manifest["rasters"]["glorys"]["so"].append(result["so"])
    manifest["rasters"]["ostia"]["analysed_sst"].append(result["analysed_sst"])
    manifest["rasters"]["sealevel"]["adt"].append(result["adt"])
    for var_name in SSS_RASTER_VARS:
        manifest["rasters"]["sss"][var_name].append(result["sss"][var_name])


def _nearest_target_dates(
    profile_dates: np.ndarray,
    target_dates: np.ndarray,
    *,
    window_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map profile dates to nearest target dates inside the temporal window."""
    profile_dates = np.asarray(profile_dates, dtype=np.int64).reshape(-1)
    target_dates = np.asarray(target_dates, dtype=np.int64).reshape(-1)
    out = np.zeros(profile_dates.shape, dtype=np.int32)
    valid = profile_dates > 0
    if target_dates.size == 0 or not np.any(valid):
        return out, np.zeros(profile_dates.shape, dtype=bool)

    target_days = np.asarray(
        [date_to_days_since_1950(int(value)) for value in target_dates],
        dtype=np.float64,
    )
    profile_days = np.asarray(
        [
            date_to_days_since_1950(int(value)) if int(value) > 0 else np.nan
            for value in profile_dates
        ],
        dtype=np.float64,
    )
    positions = np.searchsorted(target_days, profile_days)
    right = np.clip(positions, 0, int(target_days.size) - 1)
    left = np.clip(positions - 1, 0, int(target_days.size) - 1)
    choose_right = np.abs(target_days[right] - profile_days) < np.abs(
        profile_days - target_days[left]
    )
    selected = np.where(choose_right, right, left)
    distance = np.abs(profile_days - target_days[selected])
    radius = max(0.0, (float(window_days) - 1.0) / 2.0)
    keep = np.isfinite(distance) & (distance <= radius + 1.0e-8)
    out[keep] = target_dates[selected[keep]].astype(np.int32, copy=False)
    return out, keep


def _normalize_lon_for_grid(lon: np.ndarray, grid: TargetGrid) -> np.ndarray:
    """Normalize longitudes into the output grid convention."""
    lon_values = np.asarray(lon, dtype=np.float64)
    left = float(grid.transform.c)
    right = left + (float(grid.width) * float(grid.transform.a))
    if left >= 0.0 and right > 180.0:
        return np.mod(lon_values, 360.0)
    return ((lon_values + 180.0) % 360.0) - 180.0


def _grid_indices_for_points(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    grid: TargetGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/column indices and in-grid flags for profile points."""
    lon_values = _normalize_lon_for_grid(lon, grid)
    lat_values = np.asarray(lat, dtype=np.float64)
    col = np.floor((lon_values - float(grid.transform.c)) / float(grid.transform.a))
    row = np.floor((lat_values - float(grid.transform.f)) / float(grid.transform.e))
    row_int = col.astype(np.int64, copy=False)
    col_int = row.astype(np.int64, copy=False)
    # The variables are intentionally swapped back below because row uses latitude
    # and col uses longitude; keeping the formulas separate makes the sign clear.
    row_int, col_int = col_int, row_int
    in_grid = (
        np.isfinite(lat_values)
        & np.isfinite(lon_values)
        & (row_int >= 0)
        & (row_int < int(grid.height))
        & (col_int >= 0)
        & (col_int < int(grid.width))
    )
    return row_int, col_int, in_grid


def _dataset_profile_size(ds: xr.Dataset) -> int:
    """Return the profile dimension size for an ARGO-like dataset."""
    for dim_name in ("profile", "N_PROF"):
        if dim_name in ds.sizes:
            return int(ds.sizes[dim_name])
    raise RuntimeError("ARGO profile dataset has no profile dimension.")


def _profile_dim_name(ds: xr.Dataset) -> str:
    """Return the profile dimension name for an ARGO-like dataset."""
    if "profile" in ds.sizes:
        return "profile"
    if "N_PROF" in ds.sizes:
        return "N_PROF"
    raise RuntimeError("ARGO profile dataset has no profile dimension.")


def _read_argo_chunk(
    ds: xr.Dataset,
    *,
    start: int,
    stop: int,
    names: dict[str, str],
) -> dict[str, np.ndarray]:
    """Read one ARGO profile chunk into numpy arrays."""
    dim = _profile_dim_name(ds)
    chunk = ds.isel({dim: slice(int(start), int(stop))})
    return {
        key: np.asarray(chunk[var_name].values)
        for key, var_name in names.items()
        if var_name in chunk
    }


def _worst_profile_qc_code(
    *,
    observed: np.ndarray,
    level_qc: Sequence[np.ndarray],
    profile_qc: Sequence[np.ndarray],
) -> np.ndarray:
    """Collapse available QC values to one worst non-missing code per profile."""
    observed_mask = np.asarray(observed, dtype=bool)
    if observed_mask.ndim != 2:
        raise ValueError("observed must have shape profile x depth.")
    profile_count = int(observed_mask.shape[0])
    worst = np.full(profile_count, int(MISSING_QC_FLAG), dtype=np.int16)
    for values in profile_qc:
        qc = np.asarray(values, dtype=np.int16).reshape(-1)
        if int(qc.size) != profile_count:
            raise ValueError("profile QC length does not match observed profiles.")
        present = qc >= 0
        worst[present] = np.maximum(worst[present], qc[present])
    for values in level_qc:
        qc = np.asarray(values, dtype=np.int16)
        if qc.shape != observed_mask.shape:
            raise ValueError("level QC shape does not match observed profiles.")
        # Only observed depths should influence the profile-level worst code.
        present = observed_mask & (qc >= 0)
        if np.any(present):
            masked = np.where(present, qc, int(MISSING_QC_FLAG))
            worst = np.maximum(worst, masked.max(axis=1))
    return worst.astype(np.int8, copy=False)


def _align_profile_values_to_depths(
    values: np.ndarray,
    depths: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    """Align one raw ARGO profile variable onto the target depth axis."""
    return _align_argo_profile_to_glorys_depths(
        temperature=values,
        depth=depths,
        glorys_depths=target_depths,
    )


def _project_raw_argo_dataset_to_depths(
    ds: xr.Dataset,
    *,
    variable_names: Sequence[str],
    depth_var_name: str,
    target_depths: np.ndarray,
) -> xr.Dataset:
    """Project raw ARGO variables onto the target GLORYS depth axis."""
    target_depths = np.asarray(target_depths, dtype=np.float32).reshape(-1)
    if target_depths.size == 0:
        raise RuntimeError(
            "Cannot project ARGO profiles to an empty GLORYS depth axis."
        )

    coords: dict[str, Any] = {"depth": target_depths}
    if "N_PROF" in ds.coords:
        coords["N_PROF"] = ds["N_PROF"]
    out = xr.Dataset(coords=coords, attrs=dict(ds.attrs))
    for name in ("JULD", "LATITUDE", "LONGITUDE"):
        if name in ds:
            out[name] = ds[name]

    target_depth_da = xr.DataArray(
        target_depths,
        dims=("depth",),
        coords={"depth": target_depths},
    )
    for name in variable_names:
        if name not in ds:
            continue
        valid = (
            np.isfinite(ds[name])
            & np.isfinite(ds[str(depth_var_name)])
            & (ds[str(depth_var_name)] >= 0.0)
        ).any(dim="N_LEVELS")
        out[f"HAS_VALID_{name}"] = valid.astype(bool)
        # Keep projection lazy across profile chunks instead of materializing
        # all source profiles before writing the compact ARGO store.
        projected = xr.apply_ufunc(
            _align_profile_values_to_depths,
            ds[name],
            ds[str(depth_var_name)],
            target_depth_da,
            input_core_dims=[["N_LEVELS"], ["N_LEVELS"], ["depth"]],
            output_core_dims=[["depth"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"depth": int(target_depths.size)}},
        )
        projected.attrs.update(ds[name].attrs)
        projected.attrs["source_depth_var"] = str(depth_var_name)
        projected.attrs["description"] = (
            f"{name} projected onto the GLORYS depth coordinate."
        )
        out[name] = projected
    return out


def _add_raw_argo_profile_helpers(
    ds: xr.Dataset,
    *,
    variable_names: Sequence[str],
) -> xr.Dataset:
    """Add profile date and validity helpers to a projected ARGO dataset."""
    if "JULD" in ds:
        juld = np.asarray(ds["JULD"].values, dtype=np.float64).reshape(-1)
        dates = np.zeros(juld.shape, dtype=np.int32)
        valid = np.isfinite(juld) & (juld < 90000.0) & (juld > -20000.0)
        if np.any(valid):
            days = np.datetime64("1950-01-01", "D") + np.floor(juld[valid]).astype(
                "timedelta64[D]"
            )
            # Encode dates as YYYYMMDD integers so the writer can assign profile
            # windows without reopening the original EN4 files.
            dates[valid] = np.char.replace(
                np.datetime_as_string(days, unit="D"),
                "-",
                "",
            ).astype(np.int32)
        ds["DATE"] = (("N_PROF",), dates)

    for name in variable_names:
        if name not in ds:
            continue
        if f"HAS_VALID_{name}" in ds:
            continue
        da = ds[name]
        depth_dim = "depth" if "depth" in da.dims else da.dims[-1]
        ds[f"HAS_VALID_{name}"] = np.isfinite(da).any(dim=depth_dim).astype(bool)
    return ds


def _open_projected_raw_argo_dataset(
    paths: Sequence[Path],
    *,
    variable_names: Sequence[str],
    depth_var_name: str,
    target_depths: np.ndarray,
    chunk_profile: int,
) -> xr.Dataset:
    """Open raw EN4/ARGO NetCDF files and project profile variables."""
    if not paths:
        raise RuntimeError("No ARGO NetCDF files were selected for GeoTIFF export.")

    selected = ("JULD", "LATITUDE", "LONGITUDE", str(depth_var_name)) + tuple(
        str(name) for name in variable_names
    )

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        """Keep only fields needed for raw ARGO projection."""
        present = [name for name in selected if name in ds]
        required = ("JULD", "LATITUDE", "LONGITUDE", str(depth_var_name))
        missing = [name for name in required if name not in ds]
        if missing:
            source = ds.encoding.get("source", "<unknown>")
            raise RuntimeError(
                f"ARGO source is missing required variables {missing}: {source}"
            )
        return ds[present]

    ds = xr.open_mfdataset(
        list(paths),
        combine="nested",
        concat_dim="N_PROF",
        preprocess=preprocess,
        decode_times=False,
        mask_and_scale=True,
        chunks={"N_PROF": int(chunk_profile), "N_LEVELS": -1},
        parallel=False,
    )
    chunk_map = {"N_PROF": int(chunk_profile)}
    if "N_LEVELS" in ds.dims:
        chunk_map["N_LEVELS"] = -1
    ds = ds.chunk({name: size for name, size in chunk_map.items() if name in ds.dims})
    projected = _project_raw_argo_dataset_to_depths(
        ds,
        variable_names=variable_names,
        depth_var_name=depth_var_name,
        target_depths=target_depths,
    )
    projected = projected.chunk(
        {
            name: size
            for name, size in {"N_PROF": int(chunk_profile), "depth": -1}.items()
            if name in projected.dims
        }
    )
    return _add_raw_argo_profile_helpers(
        projected,
        variable_names=variable_names,
    )


def _open_raw_argo_as_projected_dataset(
    *,
    argo_dir: Path,
    start_date: int | None,
    end_date: int | None,
    target_depths: np.ndarray,
    chunk_profile: int,
) -> xr.Dataset:
    """Open raw EN4/ARGO files and project TEMP/PSAL to the target depth axis."""
    argo_files = filter_argo_files_by_date_range(
        sorted(Path(argo_dir).glob("*.nc")),
        start_date=start_date,
        end_date=end_date,
    )
    return _open_projected_raw_argo_dataset(
        argo_files,
        variable_names=("TEMP", "PSAL_CORRECTED"),
        depth_var_name="DEPH_CORRECTED",
        target_depths=np.asarray(target_depths, dtype=np.float32),
        chunk_profile=int(chunk_profile),
    )


def _existing_argo_profile_store_metadata(
    output_zarr: Path,
    *,
    source_kind: str,
) -> dict[str, Any]:
    """Return manifest metadata for an existing ARGO profile store."""
    ds = xr.open_zarr(output_zarr, consolidated=None)
    try:
        return {
            "path": _output_relative(output_zarr, output_zarr.parent.parent),
            "profile_count": int(ds.sizes.get("profile", 0)),
            "source_kind": str(ds.attrs.get("source_kind", source_kind)),
            "skipped_existing": True,
        }
    finally:
        ds.close()


def _write_argo_profile_store(
    *,
    input_ds: xr.Dataset,
    output_zarr: Path,
    grid: TargetGrid,
    target_dates: np.ndarray,
    depth_axis: np.ndarray,
    source_kind: str,
    chunk_profile: int,
    overwrite: bool,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Write compact grid-indexed ARGO profile tensors from enriched/raw input."""
    if output_zarr.exists() and skip_existing and not overwrite:
        return _existing_argo_profile_store_metadata(
            output_zarr,
            source_kind=source_kind,
        )
    if output_zarr.exists() and not overwrite:
        raise FileExistsError(f"ARGO output zarr already exists: {output_zarr}")
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    if source_kind == "enriched":
        names = {
            "date": "profile_date",
            "lat": "latitude",
            "lon": "longitude",
            "temp": "argo_temp_on_glorys_depth",
            "salinity": "argo_psal_on_glorys_depth",
        }
        source_file_name = "profile_source_file"
        source_index_name = "profile_idx"
    else:
        names = {
            "date": "DATE",
            "lat": "LATITUDE",
            "lon": "LONGITUDE",
            "temp": "TEMP",
            "salinity": "PSAL_CORRECTED",
        }
        source_file_name = None
        source_index_name = None

    missing = [var_name for var_name in names.values() if var_name not in input_ds]
    if missing:
        raise RuntimeError(f"ARGO input is missing required variables: {missing}")

    profile_count = _dataset_profile_size(input_ds)
    temp_encoded_parts: list[np.ndarray] = []
    salinity_encoded_parts: list[np.ndarray] = []
    temp_valid_parts: list[np.ndarray] = []
    salinity_valid_parts: list[np.ndarray] = []
    profile_date_parts: list[np.ndarray] = []
    target_date_parts: list[np.ndarray] = []
    latitude_parts: list[np.ndarray] = []
    longitude_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    source_index_parts: list[np.ndarray] = []
    source_file_parts: list[np.ndarray] = []
    level_qc_names = COMPACT_LEVEL_QC_NAMES
    profile_qc_names = tuple(f"argo_{name}_qc" for name in ARGO_PROFILE_QC_VARS)
    level_qc_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in level_qc_names if name in input_ds
    }
    profile_qc_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in profile_qc_names if name in input_ds
    }
    temp_profile_qc_parts: list[np.ndarray] = []
    salinity_profile_qc_parts: list[np.ndarray] = []
    temp_stats: list[EncodeStats] = []
    salinity_stats: list[EncodeStats] = []

    optional_names = dict(names)
    if source_file_name is not None and source_file_name in input_ds:
        optional_names["source_file"] = source_file_name
    if source_index_name is not None and source_index_name in input_ds:
        optional_names["source_index"] = source_index_name
    for name in tuple(level_qc_parts) + tuple(profile_qc_parts):
        optional_names[name] = name

    profile_progress = tqdm(
        total=int(profile_count),
        desc=f"Preprocessing ARGO profiles ({source_kind})",
        unit="profile",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    kept_profiles = 0
    for start in range(0, int(profile_count), int(chunk_profile)):
        stop = min(int(profile_count), start + int(chunk_profile))
        chunk = _read_argo_chunk(input_ds, start=start, stop=stop, names=optional_names)
        profile_dates = np.asarray(chunk["date"], dtype=np.int64).reshape(-1)
        lat = np.asarray(chunk["lat"], dtype=np.float64).reshape(-1)
        lon = np.asarray(chunk["lon"], dtype=np.float64).reshape(-1)
        target_date, date_valid = _nearest_target_dates(
            profile_dates,
            target_dates,
            window_days=DEFAULT_SURFACE_AGGREGATE_DAYS,
        )
        grid_row, grid_col, grid_valid = _grid_indices_for_points(
            lat=lat,
            lon=lon,
            grid=grid,
        )
        temp_kelvin = _temperature_to_kelvin(
            np.asarray(chunk["temp"], dtype=np.float32),
            source_is_celsius=True,
        )
        salinity = np.asarray(chunk["salinity"], dtype=np.float32)
        temp_profile_valid = np.isfinite(temp_kelvin).any(axis=1)
        salinity_profile_valid = np.isfinite(salinity).any(axis=1)
        keep = date_valid & grid_valid & (temp_profile_valid | salinity_profile_valid)
        kept_profiles += int(np.count_nonzero(keep))
        profile_progress.update(int(stop) - int(start))
        profile_progress.set_postfix(kept=kept_profiles, refresh=False)
        if not np.any(keep):
            continue

        temp_encoded, temp_stat = encode_stretched_uint8(
            temp_kelvin[keep],
            STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
        )
        salinity_encoded, salinity_stat = encode_stretched_uint8(
            salinity[keep],
            STRETCH_SPECS[SALINITY_STRETCH],
        )
        temp_stats.append(temp_stat)
        salinity_stats.append(salinity_stat)
        temp_encoded_parts.append(temp_encoded)
        salinity_encoded_parts.append(salinity_encoded)
        temp_valid_parts.append(np.isfinite(temp_kelvin[keep]))
        salinity_valid_parts.append(np.isfinite(salinity[keep]))
        profile_date_parts.append(profile_dates[keep].astype(np.int32, copy=False))
        target_date_parts.append(target_date[keep].astype(np.int32, copy=False))
        latitude_parts.append(lat[keep].astype(np.float32, copy=False))
        longitude_parts.append(lon[keep].astype(np.float32, copy=False))
        row_parts.append(grid_row[keep].astype(np.int32, copy=False))
        col_parts.append(grid_col[keep].astype(np.int32, copy=False))
        if "source_index" in chunk:
            source_index_parts.append(
                np.asarray(chunk["source_index"], dtype=np.int32).reshape(-1)[keep]
            )
        if "source_file" in chunk:
            source_file_parts.append(np.asarray(chunk["source_file"]).reshape(-1)[keep])
        for name in level_qc_parts:
            level_qc_parts[name].append(
                np.asarray(chunk[name], dtype=np.int8)[keep].astype(np.int8, copy=False)
            )
        level_depth_qc = (
            [np.asarray(chunk["argo_depth_qc_on_glorys_depth"], dtype=np.int8)[keep]]
            if "argo_depth_qc_on_glorys_depth" in chunk
            else []
        )
        profile_common_qc = [
            np.asarray(chunk[name], dtype=np.int8).reshape(-1)[keep]
            for name in (
                "argo_juld_qc",
                "argo_position_qc",
                "argo_profile_depth_qc",
            )
            if name in chunk
        ]
        temp_profile_qc_parts.append(
            _worst_profile_qc_code(
                observed=np.isfinite(temp_kelvin[keep]),
                level_qc=level_depth_qc
                + (
                    [
                        np.asarray(
                            chunk["argo_temp_qc_on_glorys_depth"], dtype=np.int8
                        )[keep]
                    ]
                    if "argo_temp_qc_on_glorys_depth" in chunk
                    else []
                ),
                profile_qc=profile_common_qc,
            )
        )
        salinity_profile_qc_parts.append(
            _worst_profile_qc_code(
                observed=np.isfinite(salinity[keep]),
                level_qc=level_depth_qc
                + (
                    [
                        np.asarray(
                            chunk["argo_psal_qc_on_glorys_depth"], dtype=np.int8
                        )[keep]
                    ]
                    if "argo_psal_qc_on_glorys_depth" in chunk
                    else []
                ),
                profile_qc=profile_common_qc
                + (
                    [
                        np.asarray(
                            chunk["argo_profile_psal_qc"], dtype=np.int8
                        ).reshape(-1)[keep]
                    ]
                    if "argo_profile_psal_qc" in chunk
                    else []
                ),
            )
        )
    profile_progress.close()

    if temp_encoded_parts:
        temp_encoded_all = np.concatenate(temp_encoded_parts, axis=0)
        salinity_encoded_all = np.concatenate(salinity_encoded_parts, axis=0)
        temp_valid_all = np.concatenate(temp_valid_parts, axis=0)
        salinity_valid_all = np.concatenate(salinity_valid_parts, axis=0)
        profile_dates_all = np.concatenate(profile_date_parts, axis=0)
        target_dates_all = np.concatenate(target_date_parts, axis=0)
        lat_all = np.concatenate(latitude_parts, axis=0)
        lon_all = np.concatenate(longitude_parts, axis=0)
        rows_all = np.concatenate(row_parts, axis=0)
        cols_all = np.concatenate(col_parts, axis=0)
        temp_profile_qc_all = np.concatenate(temp_profile_qc_parts, axis=0)
        salinity_profile_qc_all = np.concatenate(salinity_profile_qc_parts, axis=0)
    else:
        depth_size = int(np.asarray(depth_axis).size)
        temp_encoded_all = np.zeros((0, depth_size), dtype=np.uint8)
        salinity_encoded_all = np.zeros((0, depth_size), dtype=np.uint8)
        temp_valid_all = np.zeros((0, depth_size), dtype=bool)
        salinity_valid_all = np.zeros((0, depth_size), dtype=bool)
        profile_dates_all = np.zeros((0,), dtype=np.int32)
        target_dates_all = np.zeros((0,), dtype=np.int32)
        lat_all = np.zeros((0,), dtype=np.float32)
        lon_all = np.zeros((0,), dtype=np.float32)
        rows_all = np.zeros((0,), dtype=np.int32)
        cols_all = np.zeros((0,), dtype=np.int32)
        temp_profile_qc_all = np.zeros((0,), dtype=np.int8)
        salinity_profile_qc_all = np.zeros((0,), dtype=np.int8)

    depth_size = int(np.asarray(depth_axis).size)
    level_qc_all = {
        name: (
            np.concatenate(parts, axis=0).astype(np.int8, copy=False)
            if parts
            else np.zeros((0, depth_size), dtype=np.int8)
        )
        for name, parts in level_qc_parts.items()
    }
    profile_axis = np.arange(int(profile_dates_all.size), dtype=np.int64)
    data_vars = {
        "profile_date": (("profile",), profile_dates_all),
        "target_date": (("profile",), target_dates_all),
        "latitude": (("profile",), lat_all),
        "longitude": (("profile",), lon_all),
        "grid_row": (("profile",), rows_all),
        "grid_col": (("profile",), cols_all),
        "argo_temp_kelvin_uint8": (
            ("profile", "glorys_depth"),
            temp_encoded_all,
        ),
        "argo_psal_uint8": (("profile", "glorys_depth"), salinity_encoded_all),
        "argo_temp_valid": (("profile", "glorys_depth"), temp_valid_all),
        "argo_psal_valid": (("profile", "glorys_depth"), salinity_valid_all),
        COMPACT_TEMP_PROFILE_QC_NAME: (("profile",), temp_profile_qc_all),
        COMPACT_PSAL_PROFILE_QC_NAME: (("profile",), salinity_profile_qc_all),
    }
    for name, values in level_qc_all.items():
        data_vars[name] = (("profile", "glorys_depth"), values)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "profile": profile_axis,
            "glorys_depth": np.asarray(depth_axis, dtype=np.float32),
        },
        attrs={
            "created_by": "depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff",
            "source_kind": source_kind,
            "temperature_units": "K",
            "temperature_stretch": _stretch_manifest(
                STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH]
            ),
            "salinity_stretch": _stretch_manifest(STRETCH_SPECS[SALINITY_STRETCH]),
            "temporal_assignment": "nearest GLORYS target date inside centered 7-day window",
        },
    )
    if source_index_parts:
        ds["source_profile_idx"] = (
            ("profile",),
            np.concatenate(source_index_parts, axis=0).astype(np.int32, copy=False),
        )
    if source_file_parts:
        ds["profile_source_file"] = (
            ("profile",),
            np.concatenate(source_file_parts, axis=0).astype(str),
        )

    for name in ("argo_temp_kelvin_uint8", "argo_psal_uint8"):
        ds[name].attrs["nodata"] = int(NODATA_CODE)
    ds["argo_temp_kelvin_uint8"].attrs.update(
        _stretch_manifest(STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH])
    )
    ds["argo_psal_uint8"].attrs.update(
        _stretch_manifest(STRETCH_SPECS[SALINITY_STRETCH])
    )

    encoding = {
        name: {
            "chunks": (min(int(chunk_profile), max(int(profile_dates_all.size), 1)),)
        }
        for name in ds.data_vars
        if ds[name].dims == ("profile",)
    }
    for name in (
        "argo_temp_kelvin_uint8",
        "argo_psal_uint8",
        "argo_temp_valid",
        "argo_psal_valid",
        *level_qc_all,
    ):
        encoding[name] = {
            "chunks": (
                min(int(chunk_profile), max(int(profile_dates_all.size), 1)),
                int(np.asarray(depth_axis).size),
            )
        }
    with tqdm(
        total=1,
        desc="Writing ARGO profile zarr",
        unit="store",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        ds.to_zarr(output_zarr, mode="w", encoding=encoding, zarr_format=2)
        progress.update(1)
    return {
        "path": _output_relative(output_zarr, output_zarr.parent.parent),
        "profile_count": int(profile_dates_all.size),
        "source_kind": source_kind,
        "temperature_stats": _merge_stats(temp_stats),
        "salinity_stats": _merge_stats(salinity_stats),
    }


def _open_argo_input_dataset(
    *,
    argo_source: str,
    enriched_argo_zarr: Path,
    argo_dir: Path,
    start_date: int | None,
    end_date: int | None,
    target_depths: np.ndarray,
    chunk_profile: int,
) -> xr.Dataset:
    """Open the configured ARGO source for preprocessing."""
    if argo_source == "none":
        return xr.Dataset(coords={"profile": np.arange(0, dtype=np.int64)})
    if argo_source == "enriched":
        if not enriched_argo_zarr.exists():
            raise FileNotFoundError(
                f"Enriched ARGO zarr does not exist: {enriched_argo_zarr}"
            )
        return xr.open_zarr(enriched_argo_zarr, consolidated=None)
    if argo_source == "raw":
        return _open_raw_argo_as_projected_dataset(
            argo_dir=argo_dir,
            start_date=start_date,
            end_date=end_date,
            target_depths=target_depths,
            chunk_profile=chunk_profile,
        )
    raise ValueError("argo_source must be one of: enriched, raw, none")


def _yaml_safe(value: Any) -> Any:
    """Convert numpy/rasterio values into YAML-safe Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_yaml_safe(item) for item in value.tolist()]
    if isinstance(value, Affine):
        return [float(v) for v in value.to_gdal()]
    if hasattr(value, "to_string"):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def export_training_geotiff_dataset(
    *,
    glorys_dir: str | Path = DEFAULT_GLORYS_DIR,
    ostia_dir: str | Path = DEFAULT_OSTIA_DIR,
    sealevel_dir: str | Path = DEFAULT_SEALEVEL_DIR,
    sss_dir: str | Path = DEFAULT_SSS_DIR,
    enriched_argo_zarr: str | Path = DEFAULT_ENRICHED_ARGO_ZARR,
    argo_dir: str | Path = DEFAULT_ARGO_DIR,
    land_mask_path: str | Path = DEFAULT_LAND_MASK_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: int | None = DEFAULT_START_DATE,
    end_date: int | None = DEFAULT_END_DATE,
    surface_aggregate_days: int = DEFAULT_SURFACE_AGGREGATE_DAYS,
    argo_source: str = "enriched",
    chunk_profile: int = DEFAULT_CHUNK_PROFILE,
    workers: int = DEFAULT_RASTER_WORKERS,
    overwrite: bool = False,
    skip_existing: bool = False,
    write_argo: bool = True,
    show_progress: bool = True,
) -> Path:
    """Export aligned uint8 raster training sources and preprocessed ARGO profiles."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    raster_workers = max(1, int(workers))
    skip_existing = bool(skip_existing and not overwrite)
    with tqdm(
        total=1,
        desc="Loading target grid",
        unit="grid",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        grid = _load_target_grid(Path(land_mask_path))
        progress.update(1)

    glorys_items = _filter_timed_files(
        scan_timed_files(Path(glorys_dir), show_progress=show_progress),
        start_date=start_date,
        end_date=end_date,
    )
    if not glorys_items:
        raise RuntimeError(f"No GLORYS files selected from: {glorys_dir}")
    ostia_items = _filter_timed_files(
        scan_timed_files(Path(ostia_dir), show_progress=show_progress),
        start_date=start_date,
        end_date=end_date,
    )
    sealevel_items = _filter_timed_files(
        scan_timed_files(Path(sealevel_dir), show_progress=show_progress),
        start_date=start_date,
        end_date=end_date,
    )
    sss_items = _filter_timed_files(
        scan_timed_files(Path(sss_dir), show_progress=show_progress),
        start_date=start_date,
        end_date=end_date,
    )
    if show_progress:
        tqdm.write(
            "Selected source files: "
            f"{len(glorys_items)} GLORYS weekly, "
            f"{len(ostia_items)} OSTIA daily, "
            f"{len(sealevel_items)} sea-level daily, "
            f"{len(sss_items)} SSS daily."
        )

    cache = DatasetCache(max_open=8)
    try:
        with xr.open_dataset(
            glorys_items[0].path,
            engine="h5netcdf",
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        ) as first_glorys:
            depth_axis = _glorys_depth_axis(first_glorys)
        exported_land_mask_path = _copy_land_mask_to_output(
            land_mask_path=Path(land_mask_path),
            output_dir=output_root,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        target_date_values = [
            _date_int_from_days_since_1950(float(item.day)) for item in glorys_items
        ]

        manifest: dict[str, Any] = {
            "created_by": "depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff",
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "output_dir": str(output_root),
            "requested_date_range": {
                "start_date": None if start_date is None else int(start_date),
                "end_date": None if end_date is None else int(end_date),
            },
            "grid": {
                "source": _output_relative(exported_land_mask_path, output_root),
                "source_original": str(land_mask_path),
                "width": int(grid.width),
                "height": int(grid.height),
                "crs": str(grid.crs),
                "transform_gdal": [float(v) for v in grid.transform.to_gdal()],
                "resolution": {
                    "x": float(grid.transform.a),
                    "y": abs(float(grid.transform.e)),
                },
            },
            "stretch": {
                name: _stretch_manifest(spec) for name, spec in STRETCH_SPECS.items()
            },
            "surface_temporal_aggregation": {
                "target": "glorys",
                "window_days": int(surface_aggregate_days),
            },
            "parallelism": {"raster_workers": int(raster_workers)},
            "resume": {"skip_existing": bool(skip_existing)},
            "depth_axis_m": [float(value) for value in depth_axis.tolist()],
            "target_dates": target_date_values,
            "rasters": {
                "glorys": {"thetao": [], "so": []},
                "ostia": {"analysed_sst": []},
                "sealevel": {"adt": []},
                "sss": {var_name: [] for var_name in SSS_RASTER_VARS},
            },
        }

        raster_steps_per_date = 4 + len(SSS_RASTER_VARS)
        total_steps = (len(glorys_items) * raster_steps_per_date) + 2
        if not write_argo or str(argo_source) == "none":
            total_steps -= 1
        with tqdm(
            total=total_steps,
            desc="Whole GeoTIFF dataset export",
            unit="step",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as export_progress:
            if raster_workers == 1:
                for item in tqdm(
                    glorys_items,
                    desc="Exporting weekly raster dates",
                    unit="date",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ):
                    date_value = _date_int_from_days_since_1950(float(item.day))
                    result: dict[str, Any] = {"date": int(date_value)}
                    export_progress.set_postfix(date=str(date_value), variable="thetao")
                    result["thetao"] = _export_glorys_variable(
                        item=item,
                        cache=cache,
                        output_dir=output_root,
                        grid=grid,
                        var_name="thetao",
                        stretch=STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
                        source_is_temperature_celsius=True,
                        skip_existing=skip_existing,
                        show_progress=show_progress,
                    )
                    export_progress.update(1)

                    export_progress.set_postfix(date=str(date_value), variable="so")
                    result["so"] = _export_glorys_variable(
                        item=item,
                        cache=cache,
                        output_dir=output_root,
                        grid=grid,
                        var_name="so",
                        stretch=STRETCH_SPECS[SALINITY_STRETCH],
                        source_is_temperature_celsius=False,
                        skip_existing=skip_existing,
                        show_progress=show_progress,
                    )
                    export_progress.update(1)

                    export_progress.set_postfix(
                        date=str(date_value),
                        variable="analysed_sst",
                    )
                    result["analysed_sst"] = _export_surface_variable(
                        source_name="ostia",
                        source_items=ostia_items,
                        target_item=item,
                        cache=cache,
                        output_dir=output_root,
                        grid=grid,
                        var_name="analysed_sst",
                        stretch=STRETCH_SPECS[TEMPERATURE_KELVIN_STRETCH],
                        aggregate_days=surface_aggregate_days,
                        source_is_temperature=True,
                        skip_existing=skip_existing,
                        show_progress=show_progress,
                    )
                    export_progress.update(1)

                    export_progress.set_postfix(date=str(date_value), variable="adt")
                    result["adt"] = _export_surface_variable(
                        source_name="sealevel",
                        source_items=sealevel_items,
                        target_item=item,
                        cache=cache,
                        output_dir=output_root,
                        grid=grid,
                        var_name="adt",
                        stretch=STRETCH_SPECS[SEA_HEIGHT_STRETCH],
                        aggregate_days=surface_aggregate_days,
                        source_is_temperature=False,
                        skip_existing=skip_existing,
                        show_progress=show_progress,
                    )
                    export_progress.update(1)

                    result["sss"] = {}
                    for var_name in SSS_RASTER_VARS:
                        export_progress.set_postfix(
                            date=str(date_value),
                            variable=f"sss_{var_name}",
                        )
                        result["sss"][var_name] = _export_surface_variable(
                            source_name="sss",
                            source_items=sss_items,
                            target_item=item,
                            cache=cache,
                            output_dir=output_root,
                            grid=grid,
                            var_name=var_name,
                            stretch=STRETCH_SPECS[SSS_SURFACE_STRETCHES[var_name]],
                            aggregate_days=surface_aggregate_days,
                            source_is_temperature=False,
                            skip_existing=skip_existing,
                            show_progress=show_progress,
                        )
                        export_progress.update(1)
                    _record_weekly_raster_result(manifest, result)
            else:
                results_by_date: dict[int, dict[str, Any]] = {}
                # Use processes instead of threads so Python-side encoding and
                # HDF5/xarray reads can actually occupy multiple CPU cores.
                with ProcessPoolExecutor(max_workers=raster_workers) as executor:
                    future_to_date = {
                        executor.submit(
                            _export_weekly_raster_date_worker,
                            {
                                "item": item,
                                "ostia_items": ostia_items,
                                "sealevel_items": sealevel_items,
                                "sss_items": sss_items,
                                "land_mask_path": str(land_mask_path),
                                "output_dir": str(output_root),
                                "surface_aggregate_days": int(surface_aggregate_days),
                                "skip_existing": bool(skip_existing),
                            },
                        ): int(date_value)
                        for item, date_value in zip(glorys_items, target_date_values)
                    }
                    for future in tqdm(
                        as_completed(future_to_date),
                        total=len(future_to_date),
                        desc=f"Exporting weekly raster dates ({raster_workers} workers)",
                        unit="date",
                        dynamic_ncols=True,
                        disable=not show_progress,
                    ):
                        date_value = future_to_date[future]
                        export_progress.set_postfix(
                            date=str(date_value),
                            variable="rasters",
                        )
                        result = future.result()
                        results_by_date[int(result["date"])] = result
                        export_progress.update(raster_steps_per_date)
                for date_value in target_date_values:
                    _record_weekly_raster_result(
                        manifest,
                        results_by_date[int(date_value)],
                    )

            argo_output_zarr = output_root / "argo" / "argo_profiles_on_grid.zarr"
            if not write_argo:
                if argo_output_zarr.exists():
                    manifest["argo"] = _existing_argo_profile_store_metadata(
                        argo_output_zarr,
                        source_kind=str(argo_source),
                    )
                else:
                    manifest["argo"] = {
                        "path": None,
                        "profile_count": 0,
                        "source_kind": "external",
                    }
            elif str(argo_source) == "none":
                manifest["argo"] = {
                    "path": None,
                    "profile_count": 0,
                    "source_kind": "none",
                }
            elif skip_existing and not overwrite and argo_output_zarr.exists():
                export_progress.set_postfix(date="", variable="argo")
                manifest["argo"] = _existing_argo_profile_store_metadata(
                    argo_output_zarr,
                    source_kind=str(argo_source),
                )
                export_progress.update(1)
            else:
                argo_input = _open_argo_input_dataset(
                    argo_source=str(argo_source),
                    enriched_argo_zarr=Path(enriched_argo_zarr),
                    argo_dir=Path(argo_dir),
                    start_date=start_date,
                    end_date=end_date,
                    target_depths=depth_axis,
                    chunk_profile=chunk_profile,
                )
                try:
                    export_progress.set_postfix(date="", variable="argo")
                    manifest["argo"] = _write_argo_profile_store(
                        input_ds=argo_input,
                        output_zarr=argo_output_zarr,
                        grid=grid,
                        target_dates=np.asarray(
                            manifest["target_dates"], dtype=np.int32
                        ),
                        depth_axis=depth_axis,
                        source_kind=str(argo_source),
                        chunk_profile=int(chunk_profile),
                        overwrite=overwrite,
                        skip_existing=skip_existing,
                        show_progress=show_progress,
                    )
                    export_progress.update(1)
                finally:
                    argo_input.close()

            export_progress.set_postfix(date="", variable="manifest")
            manifest_path = output_root / "manifest.yaml"
            if manifest_path.exists() and not overwrite and not skip_existing:
                raise FileExistsError(f"Manifest already exists: {manifest_path}")
            with manifest_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(_yaml_safe(manifest), f, sort_keys=False)
            export_progress.update(1)
            if show_progress:
                tqdm.write(
                    f"GeoTIFF dataset export complete: {output_root} "
                    f"({len(glorys_items)} weekly dates)."
                )
        return output_root
    finally:
        cache.close()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the GeoTIFF exporter."""
    parser = argparse.ArgumentParser(
        description="Export aligned uint8 GeoTIFF rasters and preprocessed ARGO profiles."
    )
    parser.add_argument("--glorys-dir", type=Path, default=DEFAULT_GLORYS_DIR)
    parser.add_argument("--ostia-dir", type=Path, default=DEFAULT_OSTIA_DIR)
    parser.add_argument("--sealevel-dir", type=Path, default=DEFAULT_SEALEVEL_DIR)
    parser.add_argument("--sss-dir", type=Path, default=DEFAULT_SSS_DIR)
    parser.add_argument(
        "--enriched-argo-zarr",
        type=Path,
        default=DEFAULT_ENRICHED_ARGO_ZARR,
    )
    parser.add_argument("--argo-dir", type=Path, default=DEFAULT_ARGO_DIR)
    parser.add_argument("--land-mask-path", type=Path, default=DEFAULT_LAND_MASK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", type=int, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=int, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--surface-aggregate-days",
        type=int,
        default=DEFAULT_SURFACE_AGGREGATE_DAYS,
    )
    parser.add_argument(
        "--argo-source",
        choices=("enriched", "raw", "none"),
        default="enriched",
        help="Use the enriched ARGO zarr, project raw EN4 files, or skip ARGO.",
    )
    parser.add_argument("--chunk-profile", type=int, default=DEFAULT_CHUNK_PROFILE)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_RASTER_WORKERS,
        help="Parallel worker processes for dense raster exports.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rasters-only",
        action="store_true",
        help="Only export dense GeoTIFF rasters; compact ARGO Zarrs are written by the ARGO exporter.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing raster/date outputs and write only missing files. Ignored when --overwrite is set.",
    )
    return parser


def main() -> None:
    """Run the command-line GeoTIFF export."""
    args = _build_parser().parse_args()
    output = export_training_geotiff_dataset(
        glorys_dir=args.glorys_dir,
        ostia_dir=args.ostia_dir,
        sealevel_dir=args.sealevel_dir,
        sss_dir=args.sss_dir,
        enriched_argo_zarr=args.enriched_argo_zarr,
        argo_dir=args.argo_dir,
        land_mask_path=args.land_mask_path,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        surface_aggregate_days=args.surface_aggregate_days,
        argo_source=args.argo_source,
        chunk_profile=args.chunk_profile,
        workers=args.workers,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        write_argo=not args.rasters_only,
    )
    print(f"Wrote GeoTIFF training dataset: {output}")


if __name__ == "__main__":
    main()
