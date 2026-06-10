# Example:
# /work/envs/depth/bin/python -m depth_recon.utils.visualization.create_paper_header_image \
#   --dataset-root /work/data/depthdif \
#   --manifest-path /work/data/depthdif/manifest.yaml \
#   --argo-path /work/data/depthdif/argo/argo_profiles_on_grid.zarr \
#   --ostia-raster-dir /work/data/depthdif/rasters/ostia/analysed_sst \
#   --glorys-raster-dir /work/data/depthdif/rasters/glorys/thetao \
#   --glorys-salinity-raster-dir /work/data/depthdif/rasters/glorys/so \
#   --ssh-raster-dir /work/data/depthdif/rasters/sealevel/adt \
#   --sss-raster-dir /work/data/depthdif/rasters/sss/sos \
#   --land-mask-path /work/data/depthdif/masks/world_land_mask_glorys_0p1.tif \
#   --world-geojson-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world.geojson \
#   --date 20180622 \
#   --region-bounds -82 -45 24 47 \
#   --glorys-max-depth-m 1000 \
#   --max-argo-profiles 0 \
#   --max-argo-profile-lines 180 \
#   --en4-profile-grid-size 9 \
#   --hex-gridsize 41 \
#   --land-stride 8 \
#   --argo-chunk-size 200000 \
#   --fig-width 9.6 \
#   --fig-height 6.4 \
#   --dpi 300 \
#   --output-png docs/assets/figures/depthdif_paper_header_gulf_stream_20180622.png \
#   --output-webp docs/assets/figures/depthdif_paper_header_gulf_stream_20180622.webp \
#   --metadata-json docs/assets/figures/depthdif_paper_header_gulf_stream_20180622.json
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from PIL import Image
import yaml
import zarr

CELSIUS_OFFSET = np.float32(273.15)
DEFAULT_DATASET_ROOT = Path("/work/data/depthdif")
DEFAULT_DATE = "20180622"
DEFAULT_REGION_BOUNDS = (-82.0, -45.0, 24.0, 47.0)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORLD_GEOJSON_PATH = (
    REPO_ROOT
    / "src/depth_recon/data/dataset_creation/data_download_raw/get_world/world.geojson"
)
DEFAULT_OUTPUT_STEM = Path(
    "docs/assets/figures/depthdif_paper_header_gulf_stream_20180622"
)
DEFAULT_FIG_WIDTH_IN = 9.6
DEFAULT_FIG_HEIGHT_IN = 6.4
DEFAULT_DPI = 300
EQUAL_EARTH = CRS.from_proj4("+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs")
WGS84 = CRS.from_epsg(4326)


def _resolve_path(path: str | Path | None, root: Path, default: str | Path) -> Path:
    """Resolve a CLI path against the dataset root when it is relative."""
    raw_path = Path(default if path is None else path)
    if raw_path.is_absolute():
        return raw_path
    return root / raw_path


def _resolve_repo_path(path: str | Path | None, default: str | Path) -> Path:
    """Resolve a repo-local CLI path while preserving absolute paths."""
    raw_path = Path(default if path is None else path)
    if raw_path.is_absolute():
        return raw_path
    cwd_path = Path.cwd() / raw_path
    if cwd_path.exists():
        return cwd_path
    return REPO_ROOT / raw_path


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the GeoTIFF export manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _decode_stretched_uint8(values: np.ndarray, stretch: dict[str, Any]) -> np.ndarray:
    """Decode uint8 raster/profile values into physical units."""
    arr = np.asarray(values, dtype=np.uint8)
    nodata = int(stretch.get("nodata", 255))
    valid_code_max = np.float32(stretch.get("valid_code_max", 254))
    minimum = np.float32(stretch["minimum"])
    maximum = np.float32(stretch["maximum"])

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = arr != nodata
    out[valid] = minimum + (
        arr[valid].astype(np.float32) / valid_code_max * np.float32(maximum - minimum)
    )
    return out


def _kelvin_to_celsius(values: np.ndarray) -> np.ndarray:
    """Convert a Kelvin array to Celsius."""
    return np.asarray(values, dtype=np.float32) - CELSIUS_OFFSET


def _window_for_bounds(
    src: rasterio.io.DatasetReader, bounds: tuple[float, ...]
) -> Window:
    """Return an integer raster window for lon/lat bounds."""
    lon_min, lon_max, lat_min, lat_max = [float(value) for value in bounds]
    window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
    window = window.round_offsets().round_lengths()
    col_off = max(int(window.col_off), 0)
    row_off = max(int(window.row_off), 0)
    width = min(int(window.width), int(src.width) - col_off)
    height = min(int(window.height), int(src.height) - row_off)
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounds {bounds} do not overlap raster {src.name}.")
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _grid_centers_from_window(
    src: rasterio.io.DatasetReader, window: Window
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return lon centers, lat centers, and plotting extent for a raster window."""
    transform = src.window_transform(window)
    width = int(window.width)
    height = int(window.height)
    lon = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    lat = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    left, bottom, right, top = src.window_bounds(window)
    return lon, lat, (float(left), float(right), float(bottom), float(top))


def _read_decoded_raster_region(
    raster_path: Path,
    *,
    bounds: tuple[float, ...],
    stretch: dict[str, Any],
    bands: list[int] | int = 1,
    convert_kelvin_to_celsius: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Read and decode one lon/lat raster region."""
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster does not exist: {raster_path}")
    with rasterio.open(raster_path) as src:
        window = _window_for_bounds(src, bounds)
        encoded = src.read(bands, window=window)
        lon, lat, extent = _grid_centers_from_window(src, window)

    decoded = _decode_stretched_uint8(encoded, stretch)
    if convert_kelvin_to_celsius:
        decoded = _kelvin_to_celsius(decoded)
    return decoded, lon, lat, extent


def _read_land_region(
    land_mask_path: Path, bounds: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a land-mask subset for coastline contours."""
    with rasterio.open(land_mask_path) as src:
        window = _window_for_bounds(src, bounds)
        land = src.read(1, window=window)
        lon, lat, _ = _grid_centers_from_window(src, window)
    return land, lon, lat


def _robust_limits(
    values: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    center: float | None = None,
) -> tuple[float, float]:
    """Return percentile limits while ignoring missing data."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot compute display limits for an all-missing array.")

    low, high = np.percentile(
        finite, [float(lower_percentile), float(upper_percentile)]
    )
    if center is not None:
        span = max(abs(float(center) - float(low)), abs(float(high) - float(center)))
        low = float(center) - span
        high = float(center) + span
    if not high > low:
        high = low + 1.0
    return float(low), float(high)


def _dated_raster_path(raster_dir: Path, prefix: str, date_text: str) -> Path:
    """Return the expected raster path for one exported weekly date."""
    return raster_dir / f"{prefix}_{date_text}.tif"


def _load_argo_density_points(
    argo_path: Path,
    *,
    chunk_size: int,
    max_profiles: int,
) -> dict[str, Any]:
    """Load valid ARGO point coordinates and date coverage from compact Zarr."""
    if not argo_path.exists():
        raise FileNotFoundError(f"ARGO Zarr does not exist: {argo_path}")

    group = zarr.open_group(argo_path, mode="r")
    required = {"latitude", "longitude", "target_date", "argo_temp_valid"}
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise RuntimeError(f"ARGO Zarr is missing required arrays {missing}.")

    profile_count = int(group["latitude"].shape[0])
    limit = (
        profile_count
        if int(max_profiles) <= 0
        else min(profile_count, int(max_profiles))
    )
    chunk_size = int(max(1, chunk_size))

    # The density normalization should reflect all years in the compact dataset,
    # even when max_profiles is used to speed up local smoke renders.
    target_dates = np.asarray(group["target_date"][:], dtype=np.int32)
    week_count = int(np.unique(target_dates).size)
    if week_count <= 0:
        raise RuntimeError("ARGO target_date array does not contain any weeks.")

    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    valid_profile_count = 0

    for start in range(0, limit, chunk_size):
        stop = min(start + chunk_size, limit)
        lat = np.asarray(group["latitude"][start:stop], dtype=np.float64)
        lon = np.asarray(group["longitude"][start:stop], dtype=np.float64)
        temp_valid = np.asarray(group["argo_temp_valid"][start:stop, :])
        # Profiles with no valid aligned temperature support should not enter the density map.
        valid = np.any(temp_valid > 0, axis=1) & np.isfinite(lat) & np.isfinite(lon)
        if np.any(valid):
            lat_parts.append(lat[valid])
            lon_parts.append(lon[valid])
            valid_profile_count += int(np.count_nonzero(valid))

    if not lat_parts:
        raise RuntimeError("No valid ARGO profile coordinates were found.")

    return {
        "latitude": np.concatenate(lat_parts),
        "longitude": np.concatenate(lon_parts),
        "profile_count": profile_count,
        "processed_profile_count": limit,
        "valid_profile_count": valid_profile_count,
        "week_count": week_count,
    }


def _sample_profile_indices(
    count: int, *, max_profile_lines: int, random_seed: int
) -> np.ndarray:
    """Return deterministic profile indices for 3D profile drawing."""
    count = int(count)
    max_profile_lines = int(max_profile_lines)
    if count <= 0 or max_profile_lines <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count <= max_profile_lines:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(random_seed))
    return np.sort(rng.choice(count, size=max_profile_lines, replace=False))


def _profile_payload(
    *,
    lon: list[np.ndarray],
    lat: list[np.ndarray],
    values: list[np.ndarray],
    valid: list[np.ndarray],
    depth_m: np.ndarray,
    bounds: tuple[float, ...],
    profile_grid_size: int,
    max_profile_lines: int,
    random_seed: int,
    convert_kelvin_to_celsius: bool,
    drop_low_tail_quantile: float | None = None,
) -> dict[str, Any]:
    """Build one coarsened sampled EN4-aligned profile payload for plotting."""
    if not lon:
        return {
            "longitude": np.zeros((0,), dtype=np.float32),
            "latitude": np.zeros((0,), dtype=np.float32),
            "values": np.zeros((0, depth_m.size), dtype=np.float32),
            "valid": np.zeros((0, depth_m.size), dtype=bool),
            "depth_m": depth_m,
            "filtered_count": 0,
            "column_count": 0,
            "outlier_column_count": 0,
            "plotted_count": 0,
        }

    lon_all = np.concatenate(lon).astype(np.float32, copy=False)
    lat_all = np.concatenate(lat).astype(np.float32, copy=False)
    values_all = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    valid_all = np.concatenate(valid, axis=0).astype(bool, copy=False)
    values_all[~valid_all] = np.nan
    if convert_kelvin_to_celsius:
        values_all = _kelvin_to_celsius(values_all)

    lon_min, lon_max, lat_min, lat_max = [float(value) for value in bounds]
    grid_size = int(max(1, profile_grid_size))
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1, dtype=np.float64)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1, dtype=np.float64)
    lon_bin = np.clip(
        np.searchsorted(lon_edges, lon_all, side="right") - 1, 0, grid_size - 1
    )
    lat_bin = np.clip(
        np.searchsorted(lat_edges, lat_all, side="right") - 1, 0, grid_size - 1
    )
    flat_bin = lat_bin * grid_size + lon_bin

    # Aggregate aligned profiles before plotting so the panel reads as thicker
    # regional columns instead of many needle-thin raw profile sticks.
    value_for_sum = np.where(valid_all, values_all, 0.0).astype(np.float64, copy=False)
    sums = np.zeros((grid_size * grid_size, depth_m.size), dtype=np.float64)
    counts = np.zeros((grid_size * grid_size, depth_m.size), dtype=np.int32)
    np.add.at(sums, flat_bin, value_for_sum)
    np.add.at(counts, flat_bin, valid_all.astype(np.int32, copy=False))
    column_valid = counts > 0
    column_has_profile = np.any(column_valid, axis=1)
    column_values = np.full(sums.shape, np.nan, dtype=np.float32)
    column_values[column_valid] = (sums[column_valid] / counts[column_valid]).astype(
        np.float32, copy=False
    )

    column_indices = np.flatnonzero(column_has_profile)
    column_lat_bin = column_indices // grid_size
    column_lon_bin = column_indices % grid_size
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_columns = lon_centers[column_lon_bin].astype(np.float32, copy=False)
    lat_columns = lat_centers[column_lat_bin].astype(np.float32, copy=False)
    values_columns = column_values[column_indices]
    valid_columns = column_valid[column_indices]
    column_count = int(lon_columns.size)

    column_low = np.nanpercentile(values_columns, 5.0, axis=1)
    column_high = np.nanpercentile(values_columns, 95.0, axis=1)
    finite_low = column_low[np.isfinite(column_low)]
    finite_high = column_high[np.isfinite(column_high)]
    keep_columns = np.ones((column_count,), dtype=bool)
    if finite_low.size >= 4 and finite_high.size >= 4:
        low_q1, low_q3 = np.percentile(finite_low, [25.0, 75.0])
        high_q1, high_q3 = np.percentile(finite_high, [25.0, 75.0])
        low_iqr = float(low_q3 - low_q1)
        high_iqr = float(high_q3 - high_q1)
        low_threshold = float(low_q1 - 1.5 * low_iqr)
        high_threshold = float(high_q3 + 1.5 * high_iqr)
        # Drop only whole coarsened columns with unusually extreme aligned values;
        # normal depth gradients remain because filtering happens after aggregation.
        keep_columns = (column_low >= low_threshold) & (column_high <= high_threshold)
        if drop_low_tail_quantile is not None:
            low_tail_threshold = float(
                np.percentile(finite_low, float(drop_low_tail_quantile))
            )
            keep_columns &= column_low >= low_tail_threshold
    outlier_column_count = int(column_count - np.count_nonzero(keep_columns))
    lon_columns = lon_columns[keep_columns]
    lat_columns = lat_columns[keep_columns]
    values_columns = values_columns[keep_columns]
    valid_columns = valid_columns[keep_columns]

    sample = _sample_profile_indices(
        lon_columns.size,
        max_profile_lines=max_profile_lines,
        random_seed=random_seed,
    )
    return {
        "longitude": lon_columns[sample],
        "latitude": lat_columns[sample],
        "values": values_columns[sample],
        "valid": valid_columns[sample],
        "depth_m": depth_m,
        "filtered_count": int(lon_all.size),
        "column_count": column_count,
        "outlier_column_count": outlier_column_count,
        "plotted_count": int(sample.size),
    }


def _load_argo_region_profiles(
    argo_path: Path,
    *,
    bounds: tuple[float, ...],
    date_text: str,
    depth_mask: np.ndarray,
    depth_axis_m: np.ndarray,
    temperature_stretch: dict[str, Any],
    salinity_stretch: dict[str, Any],
    chunk_size: int,
    max_profile_lines: int,
    profile_grid_size: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Load selected-week regional aligned EN4 temperature and salinity profiles."""
    if not argo_path.exists():
        raise FileNotFoundError(f"ARGO Zarr does not exist: {argo_path}")

    group = zarr.open_group(argo_path, mode="r")
    required = {
        "latitude",
        "longitude",
        "target_date",
        "argo_temp_kelvin_uint8",
        "argo_psal_uint8",
        "argo_temp_valid",
        "argo_psal_valid",
    }
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise RuntimeError(
            f"ARGO Zarr is missing required aligned EN4 arrays {missing}."
        )

    depth_mask = np.asarray(depth_mask, dtype=bool)
    profile_depth_count = int(group["argo_temp_kelvin_uint8"].shape[1])
    if profile_depth_count != depth_mask.size:
        raise RuntimeError(
            "Aligned EN4 profile depth count does not match manifest depth_axis_m: "
            f"{profile_depth_count} != {depth_mask.size}."
        )

    lon_min, lon_max, lat_min, lat_max = [float(value) for value in bounds]
    target_date = int(date_text)
    depth_m = np.asarray(depth_axis_m, dtype=np.float32)[depth_mask]
    deep_profile_mask = depth_m > 150.0
    profile_count = int(group["latitude"].shape[0])
    chunk_size = int(max(1, chunk_size))
    temp_lon_parts: list[np.ndarray] = []
    temp_lat_parts: list[np.ndarray] = []
    temp_value_parts: list[np.ndarray] = []
    temp_valid_parts: list[np.ndarray] = []
    salinity_lon_parts: list[np.ndarray] = []
    salinity_lat_parts: list[np.ndarray] = []
    salinity_value_parts: list[np.ndarray] = []
    salinity_valid_parts: list[np.ndarray] = []

    for start in range(0, profile_count, chunk_size):
        stop = min(start + chunk_size, profile_count)
        lat = np.asarray(group["latitude"][start:stop], dtype=np.float64)
        lon = np.asarray(group["longitude"][start:stop], dtype=np.float64)
        target_dates = np.asarray(group["target_date"][start:stop], dtype=np.int32)
        in_region = (
            (target_dates == target_date)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (lon >= lon_min)
            & (lon <= lon_max)
            & (lat >= lat_min)
            & (lat <= lat_max)
        )
        if not np.any(in_region):
            continue

        temp_valid = np.asarray(group["argo_temp_valid"][start:stop, :])[:, depth_mask]
        salinity_valid = np.asarray(group["argo_psal_valid"][start:stop, :])[
            :, depth_mask
        ]
        # Keep profile columns visually clean by requiring real subsurface support.
        temp_keep = in_region & np.any(temp_valid[:, deep_profile_mask] > 0, axis=1)
        salinity_keep = in_region & np.any(
            salinity_valid[:, deep_profile_mask] > 0, axis=1
        )
        if np.any(temp_keep):
            temp_encoded = np.asarray(
                group["argo_temp_kelvin_uint8"][start:stop, :], dtype=np.uint8
            )[:, depth_mask]
            temp_lon_parts.append(lon[temp_keep])
            temp_lat_parts.append(lat[temp_keep])
            temp_valid_parts.append(temp_valid[temp_keep])
            temp_value_parts.append(
                _decode_stretched_uint8(temp_encoded[temp_keep], temperature_stretch)
            )
        if np.any(salinity_keep):
            salinity_encoded = np.asarray(
                group["argo_psal_uint8"][start:stop, :], dtype=np.uint8
            )[:, depth_mask]
            salinity_lon_parts.append(lon[salinity_keep])
            salinity_lat_parts.append(lat[salinity_keep])
            salinity_valid_parts.append(salinity_valid[salinity_keep])
            salinity_value_parts.append(
                _decode_stretched_uint8(
                    salinity_encoded[salinity_keep], salinity_stretch
                )
            )

    return {
        "temperature": _profile_payload(
            lon=temp_lon_parts,
            lat=temp_lat_parts,
            values=temp_value_parts,
            valid=temp_valid_parts,
            depth_m=depth_m,
            bounds=bounds,
            profile_grid_size=profile_grid_size,
            max_profile_lines=max_profile_lines,
            random_seed=random_seed,
            convert_kelvin_to_celsius=True,
            drop_low_tail_quantile=15.0,
        ),
        "salinity": _profile_payload(
            lon=salinity_lon_parts,
            lat=salinity_lat_parts,
            values=salinity_value_parts,
            valid=salinity_valid_parts,
            depth_m=depth_m,
            bounds=bounds,
            profile_grid_size=profile_grid_size,
            max_profile_lines=max_profile_lines,
            random_seed=random_seed + 1,
            convert_kelvin_to_celsius=False,
        ),
    }


def _project_equal_earth(
    lon: np.ndarray, lat: np.ndarray, transformer: Transformer
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat arrays into Equal Earth meters."""
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _ocean_mask_for_points(
    lon: np.ndarray, lat: np.ndarray, land_mask_path: Path
) -> np.ndarray:
    """Return True for points that sample ocean in the exported land mask."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    normalized_lon = ((lon + 180.0) % 360.0) - 180.0
    ocean = np.zeros(lon.shape, dtype=bool)
    finite = np.isfinite(normalized_lon) & np.isfinite(lat)
    if not np.any(finite):
        return ocean

    with rasterio.open(land_mask_path) as src:
        land = src.read(1)
        col_float, row_float = (~src.transform) * (normalized_lon[finite], lat[finite])
        col = np.floor(col_float).astype(np.int64)
        row = np.floor(row_float).astype(np.int64)
        inside = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        finite_indices = np.flatnonzero(finite)
        sampled = finite_indices[inside]
        # The exported mask uses 0 for water and 1 for land. Only water enters hexbinning.
        ocean[sampled] = land[row[inside], col[inside]] == 0
    return ocean


def _global_projected_extent(
    transformer: Transformer,
) -> tuple[float, float, float, float]:
    """Approximate the full Equal Earth extent."""
    lons = np.linspace(-180.0, 180.0, 721)
    lats = np.linspace(-90.0, 90.0, 361)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    x, y = _project_equal_earth(lon_grid, lat_grid, transformer)
    return (
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )


def _draw_graticule(ax: plt.Axes, transformer: Transformer) -> None:
    """Draw subtle projected graticule lines on the global map."""
    for lon in np.arange(-150.0, 181.0, 30.0):
        lat_values = np.linspace(-85.0, 85.0, 300)
        lon_values = np.full_like(lat_values, float(lon))
        x, y = _project_equal_earth(lon_values, lat_values, transformer)
        ax.plot(x, y, color="#ffffff", linewidth=0.45, alpha=0.28, zorder=2)

    for lat in np.arange(-60.0, 61.0, 30.0):
        lon_values = np.linspace(-180.0, 180.0, 600)
        lat_values = np.full_like(lon_values, float(lat))
        x, y = _project_equal_earth(lon_values, lat_values, transformer)
        ax.plot(x, y, color="#ffffff", linewidth=0.45, alpha=0.28, zorder=2)


def _draw_region_box(
    ax: plt.Axes,
    transformer: Transformer,
    bounds: tuple[float, ...],
) -> None:
    """Draw the selected example-region footprint on the global map."""
    lon_min, lon_max, lat_min, lat_max = [float(value) for value in bounds]
    west_lon = np.full((80,), lon_min)
    east_lon = np.full((80,), lon_max)
    south_lat = np.full((100,), lat_min)
    north_lat = np.full((100,), lat_max)
    y_side = np.linspace(lat_min, lat_max, 80)
    x_side = np.linspace(lon_min, lon_max, 100)
    lon = np.concatenate([west_lon, x_side, east_lon, x_side[::-1], west_lon[:1]])
    lat = np.concatenate([y_side, north_lat, y_side[::-1], south_lat, y_side[:1]])
    x, y = _project_equal_earth(lon, lat, transformer)
    ax.plot(x, y, color="#f8fafc", linewidth=1.2, alpha=0.96, zorder=7)
    ax.plot(x, y, color="#111827", linewidth=2.4, alpha=0.38, zorder=6)


def _draw_global_land(
    ax: plt.Axes,
    *,
    land_mask_path: Path,
    transformer: Transformer,
    land_stride: int,
) -> None:
    """Draw a projected global land silhouette from the exported land mask."""
    land_stride = int(max(1, land_stride))
    with rasterio.open(land_mask_path) as src:
        out_height = max(2, int(np.ceil(src.height / land_stride)))
        out_width = max(2, int(np.ceil(src.width / land_stride)))
        land = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )

    lon_edges = np.linspace(-180.0, 180.0, land.shape[1] + 1, dtype=np.float64)
    lat_edges = np.linspace(90.0, -90.0, land.shape[0] + 1, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)
    x, y = _project_equal_earth(lon_grid, lat_grid, transformer)
    land_masked = np.ma.masked_where(land <= 0, land)
    ax.pcolormesh(
        x,
        y,
        land_masked,
        cmap=ListedColormap(["#9aa7a2"]),
        shading="auto",
        alpha=0.78,
        linewidth=0.0,
        rasterized=True,
        zorder=6,
    )


def _iter_geojson_exterior_rings(geometry: dict[str, Any]):
    """Yield exterior rings from GeoJSON polygon-like geometries."""
    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        if coordinates:
            yield coordinates[0]
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if polygon:
                yield polygon[0]
    elif geometry_type == "GeometryCollection":
        for child in geometry.get("geometries", []):
            if isinstance(child, dict):
                yield from _iter_geojson_exterior_rings(child)


def _draw_world_geojson_land(
    ax: plt.Axes,
    *,
    world_geojson_path: Path,
    transformer: Transformer,
) -> bool:
    """Draw colored world land polygons from the local Natural Earth-style GeoJSON."""
    if not world_geojson_path.exists():
        return False

    with world_geojson_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    payload_type = str(payload.get("type", ""))
    if payload_type == "FeatureCollection":
        features = payload.get("features", [])
    elif payload_type == "Feature":
        features = [payload]
    else:
        features = [{"geometry": payload, "properties": {}}]

    palette = [
        "#b8c7a2",
        "#c7b99a",
        "#a9bf9d",
        "#d1c3a0",
        "#b0c0aa",
        "#c4b08f",
    ]
    polygons: list[np.ndarray] = []
    colors: list[str] = []
    for feature_idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        for ring in _iter_geojson_exterior_rings(geometry):
            coords = np.asarray(ring, dtype=np.float64)
            if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
                continue
            x, y = _project_equal_earth(coords[:, 0], coords[:, 1], transformer)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(finite) < 3:
                continue
            polygons.append(np.column_stack([x[finite], y[finite]]))
            colors.append(palette[feature_idx % len(palette)])

    if not polygons:
        return False

    collection = PolyCollection(
        polygons,
        closed=True,
        facecolors=colors,
        edgecolors=(0.28, 0.34, 0.30, 0.42),
        linewidths=0.16,
        alpha=0.92,
        rasterized=True,
        zorder=6,
    )
    ax.add_collection(collection)
    return True


def _draw_argo_density_map(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    argo_points: dict[str, Any],
    land_mask_path: Path,
    world_geojson_path: Path,
    region_bounds: tuple[float, ...],
    hex_gridsize: int,
    land_stride: int,
) -> None:
    """Draw global ARGO profile density in Equal Earth projection."""
    transformer = Transformer.from_crs(WGS84, EQUAL_EARTH, always_xy=True)
    ax.set_facecolor("#e8f2f3")
    if not _draw_world_geojson_land(
        ax,
        world_geojson_path=world_geojson_path,
        transformer=transformer,
    ):
        _draw_global_land(
            ax,
            land_mask_path=land_mask_path,
            transformer=transformer,
            land_stride=land_stride,
        )
    _draw_graticule(ax, transformer)

    lon = np.asarray(argo_points["longitude"], dtype=np.float64)
    lat = np.asarray(argo_points["latitude"], dtype=np.float64)
    ocean = _ocean_mask_for_points(lon, lat, land_mask_path)
    argo_points["ocean_profile_count"] = int(np.count_nonzero(ocean))
    x, y = _project_equal_earth(lon[ocean], lat[ocean], transformer)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise RuntimeError(
            "No ocean ARGO profile coordinates were found after clipping."
        )

    hexes = ax.hexbin(
        x,
        y,
        gridsize=int(hex_gridsize),
        mincnt=1,
        linewidths=0.08,
        edgecolors=(1.0, 1.0, 1.0, 0.18),
        cmap="viridis",
        zorder=4,
    )
    mean_profiles = np.asarray(hexes.get_array(), dtype=np.float64) / float(
        argo_points["week_count"]
    )
    hexes.set_array(mean_profiles)
    positive = mean_profiles[mean_profiles > 0.0]
    vmax = float(np.percentile(positive, 99.7))
    vmin = float(max(positive.min(), 1.0 / float(argo_points["week_count"])))
    if vmax <= vmin:
        vmax = vmin * 10.0
    hexes.set_norm(LogNorm(vmin=vmin, vmax=vmax))

    _draw_region_box(ax, transformer, region_bounds)
    x_min, x_max, y_min, y_max = _global_projected_extent(transformer)
    ax.set_xlim(x_min * 1.02, x_max * 1.02)
    ax.set_ylim(y_min * 1.04, y_max * 1.04)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("EN4 Weekly Density", loc="left", pad=8)

    cax = inset_axes(
        ax,
        width="1.8%",
        height="72%",
        loc="center left",
        bbox_to_anchor=(1.025, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cbar = fig.colorbar(hexes, cax=cax, orientation="vertical")
    cbar.set_label("Average EN4 profiles / week / hex", rotation=90)
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: f"{value:.1f}" if value < 10.0 else f"{value:.0f}"
        )
    )
    cbar.outline.set_linewidth(0.35)


def _plot_land_contour(
    ax: plt.Axes,
    *,
    land_mask_path: Path,
    bounds: tuple[float, ...],
    linewidth: float = 0.45,
) -> None:
    """Overlay a coastline-like contour from the exported land mask."""
    land, lon, lat = _read_land_region(land_mask_path, bounds)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ax.contour(
        lon_grid,
        lat_grid,
        land,
        levels=[0.5],
        colors="#17231f",
        linewidths=float(linewidth),
        alpha=0.75,
    )


def _format_colorbar(
    cbar: Any,
    *,
    tick_format: str | None = None,
    nbins: int = 5,
) -> None:
    """Apply rounded, compact tick formatting to a Matplotlib colorbar."""
    cbar.locator = MaxNLocator(nbins=int(nbins), steps=[1, 2, 2.5, 5, 10])
    if tick_format is not None:
        cbar.formatter = FormatStrFormatter(str(tick_format))
    cbar.update_ticks()
    if getattr(cbar, "orientation", None) == "horizontal":
        cbar.ax.xaxis.labelpad = -0.75
    cbar.outline.set_linewidth(0.35)


def _rounded_limits_from_values(
    values: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    step: float = 1.0,
) -> tuple[float, float]:
    """Return robust display limits rounded outward to a fixed step."""
    low, high = _robust_limits(
        values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    step = float(max(step, 1.0e-6))
    low = float(np.floor(low / step) * step)
    high = float(np.ceil(high / step) * step)
    if high <= low:
        high = low + step
    return low, high


def _draw_raster_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    values: np.ndarray,
    extent: tuple[float, float, float, float],
    land_mask_path: Path,
    bounds: tuple[float, ...],
    title: str,
    cmap_name: str,
    colorbar_label: str,
    norm: Normalize,
    show_colorbar: bool = True,
    tick_format: str | None = None,
) -> Any:
    """Draw one regional 2D raster panel with an optional compact colorbar."""
    ax.set_facecolor("#d8dedb")
    image = ax.imshow(
        values,
        extent=extent,
        origin="upper",
        interpolation="bilinear",
        cmap=cmap_name,
        norm=norm,
    )
    _plot_land_contour(ax, land_mask_path=land_mask_path, bounds=bounds)
    ax.set_xlim(float(bounds[0]), float(bounds[1]))
    ax.set_ylim(float(bounds[2]), float(bounds[3]))
    ax.set_title(title, loc="left", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.grid(False)
    if show_colorbar:
        cbar = fig.colorbar(
            image, ax=ax, orientation="horizontal", fraction=0.075, pad=0.08
        )
        cbar.set_label(colorbar_label)
        _format_colorbar(cbar, tick_format=tick_format)
    return image


def _draw_glorys_cube(
    ax: plt.Axes,
    *,
    values_c: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    depth_m: np.ndarray,
    title: str,
    norm: Normalize,
    cmap: Any,
) -> None:
    """Draw a projected GLORYS temperature cube from the full lon/lat/depth block."""
    volume = np.asarray(values_c, dtype=np.float32)
    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        raise ValueError("GLORYS cube contains only missing values.")

    def _normalized_edges(values: np.ndarray) -> np.ndarray:
        """Return normalized cell edges for nonuniform physical coordinates."""
        centers = np.asarray(values, dtype=np.float64)
        if centers.size == 1:
            return np.asarray([0.0, 1.0], dtype=np.float64)
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - (edges[1] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - edges[-2])
        if edges[-1] == edges[0]:
            return np.linspace(0.0, 1.0, centers.size + 1, dtype=np.float64)
        return (edges - edges[0]) / (edges[-1] - edges[0])

    def _project_cube(
        u: np.ndarray, v: np.ndarray, d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project normalized cube coordinates into the 2D panel."""
        x = 0.08 + 0.68 * u + 0.18 * v
        y = 0.66 - 0.07 * u + 0.18 * v - 0.46 * d
        return x, y

    depth_edges = _normalized_edges(depth_m)
    row_edges = np.linspace(0.0, 1.0, volume.shape[1] + 1, dtype=np.float64)
    col_edges = np.linspace(0.0, 1.0, volume.shape[2] + 1, dtype=np.float64)

    ax.set_facecolor("#d8dedb")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#17231f")
        spine.set_linewidth(0.8)

    front = np.ma.masked_invalid(volume[:, -1, :])
    u_grid, d_grid = np.meshgrid(col_edges, depth_edges)
    x_front, y_front = _project_cube(u_grid, np.zeros_like(u_grid), d_grid)
    ax.pcolormesh(
        x_front,
        y_front,
        front,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.88,
        rasterized=True,
    )

    side = np.ma.masked_invalid(volume[:, :, -1])
    v_grid, d_grid = np.meshgrid(row_edges, depth_edges)
    x_side, y_side = _project_cube(np.ones_like(v_grid), v_grid, d_grid)
    ax.pcolormesh(
        x_side,
        y_side,
        side,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.84,
        rasterized=True,
    )

    top = np.ma.masked_invalid(volume[0])
    u_grid, v_grid = np.meshgrid(col_edges, row_edges)
    x_top, y_top = _project_cube(u_grid, v_grid, np.zeros_like(u_grid))
    ax.pcolormesh(
        x_top,
        y_top,
        top,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.96,
        rasterized=True,
    )

    def _draw_edge(
        start: tuple[float, float, float], end: tuple[float, float, float]
    ) -> None:
        """Draw one projected cube edge."""
        x0, y0 = _project_cube(
            np.asarray(start[0]), np.asarray(start[1]), np.asarray(start[2])
        )
        x1, y1 = _project_cube(
            np.asarray(end[0]), np.asarray(end[1]), np.asarray(end[2])
        )
        ax.plot(
            [float(x0), float(x1)],
            [float(y0), float(y1)],
            color="#17231f",
            linewidth=0.6,
            alpha=1.0,
        )

    corners = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    for a, b in (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ):
        _draw_edge(corners[a], corners[b])

    ax.set_title(title, loc="left", pad=5)


def _style_3d_profile_axes(
    ax: plt.Axes,
    *,
    bounds: tuple[float, ...],
    max_depth_m: float,
) -> None:
    """Apply compact styling to an EN4 3D profile axis."""
    ax.set_xlim(float(bounds[0]), float(bounds[1]))
    ax.set_ylim(float(bounds[2]), float(bounds[3]))
    ax.set_zlim(float(max_depth_m), 0.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.tick_params(length=0, labelsize=0)
    ax.view_init(elev=22.0, azim=-58.0)
    ax.set_box_aspect((1.0, 0.82, 0.9))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.86, 0.88, 0.86, 0.28))
        axis.pane.set_edgecolor((0.09, 0.14, 0.12, 0.35))
        axis._axinfo["grid"].update(color=(0.09, 0.14, 0.12, 0.16), linewidth=0.35)


def _draw_en4_profile_panel_3d(
    ax: plt.Axes,
    *,
    profile_data: dict[str, Any],
    bounds: tuple[float, ...],
    title: str,
    norm: Normalize,
    cmap: Any,
) -> None:
    """Draw coarsened selected-week aligned EN4 profiles as 3D columns."""
    lon = np.asarray(profile_data["longitude"], dtype=np.float32)
    lat = np.asarray(profile_data["latitude"], dtype=np.float32)
    values = np.asarray(profile_data["values"], dtype=np.float32)
    valid = np.asarray(profile_data["valid"], dtype=bool)
    depth_m = np.asarray(profile_data["depth_m"], dtype=np.float32)
    max_depth_m = float(np.nanmax(depth_m)) if depth_m.size else 1.0

    ax.set_facecolor("#ffffff")
    ax.set_title(title, loc="left", pad=1.5)
    _style_3d_profile_axes(ax, bounds=bounds, max_depth_m=max_depth_m)
    if lon.size == 0:
        ax.text2D(
            0.08,
            0.52,
            "No selected-week\nEN4 profiles",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#334155",
        )
        return

    segments: list[np.ndarray] = []
    segment_values: list[np.ndarray] = []
    point_lon: list[float] = []
    point_lat: list[float] = []
    point_depth: list[float] = []
    point_values: list[float] = []
    for profile_idx in range(lon.size):
        profile_valid = valid[profile_idx] & np.isfinite(values[profile_idx])
        if np.count_nonzero(profile_valid) == 0:
            continue
        profile_depth = depth_m[profile_valid]
        profile_values = values[profile_idx, profile_valid]
        profile_lon = np.full(profile_depth.shape, lon[profile_idx], dtype=np.float32)
        profile_lat = np.full(profile_depth.shape, lat[profile_idx], dtype=np.float32)
        coords = np.column_stack([profile_lon, profile_lat, profile_depth])
        if coords.shape[0] == 1:
            point_lon.append(float(coords[0, 0]))
            point_lat.append(float(coords[0, 1]))
            point_depth.append(float(coords[0, 2]))
            point_values.append(float(profile_values[0]))
            continue
        # Segment coloring follows the local mean value so vertical gradients remain visible.
        segments.append(np.stack([coords[:-1], coords[1:]], axis=1))
        segment_values.append(0.5 * (profile_values[:-1] + profile_values[1:]))

    if segments:
        collection = Line3DCollection(
            np.concatenate(segments, axis=0),
            cmap=cmap,
            norm=norm,
            linewidths=2.15,
            alpha=0.9,
        )
        collection.set_array(np.concatenate(segment_values, axis=0))
        ax.add_collection3d(collection)
    if point_values:
        ax.scatter(
            point_lon,
            point_lat,
            point_depth,
            c=point_values,
            cmap=cmap,
            norm=norm,
            s=12.0,
            alpha=0.82,
            depthshade=False,
        )


def _save_outputs(
    fig: plt.Figure,
    *,
    output_png: Path,
    output_webp: Path,
) -> tuple[int, int]:
    """Save PNG and WebP outputs, returning the PNG dimensions."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_webp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, facecolor=fig.get_facecolor())
    with Image.open(output_png) as img:
        width, height = img.size
        img.save(output_webp, "WEBP", quality=95, method=6)
    return int(width), int(height)


def create_header_image(args: argparse.Namespace) -> dict[str, Any]:
    """Create the paper header image and return reproducibility metadata."""
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = _resolve_path(args.manifest_path, dataset_root, "manifest.yaml")
    manifest = _load_manifest(manifest_path)

    argo_path = _resolve_path(
        args.argo_path, dataset_root, "argo/argo_profiles_on_grid.zarr"
    )
    ostia_dir = _resolve_path(
        args.ostia_raster_dir, dataset_root, "rasters/ostia/analysed_sst"
    )
    glorys_dir = _resolve_path(
        args.glorys_raster_dir, dataset_root, "rasters/glorys/thetao"
    )
    glorys_salinity_dir = _resolve_path(
        args.glorys_salinity_raster_dir, dataset_root, "rasters/glorys/so"
    )
    ssh_dir = _resolve_path(args.ssh_raster_dir, dataset_root, "rasters/sealevel/adt")
    sss_dir = _resolve_path(args.sss_raster_dir, dataset_root, "rasters/sss/sos")
    land_default = manifest.get("grid", {}).get(
        "source", "masks/world_land_mask_glorys_0p1.tif"
    )
    land_mask_path = _resolve_path(args.land_mask_path, dataset_root, land_default)
    world_geojson_path = _resolve_repo_path(
        args.world_geojson_path, DEFAULT_WORLD_GEOJSON_PATH
    )

    date_text = str(args.date)
    region_bounds = tuple(float(value) for value in args.region_bounds)
    if len(region_bounds) != 4:
        raise ValueError("region-bounds must contain lon_min lon_max lat_min lat_max.")

    argo_points = _load_argo_density_points(
        argo_path,
        chunk_size=int(args.argo_chunk_size),
        max_profiles=int(args.max_argo_profiles),
    )

    temperature_stretch = manifest["stretch"]["temperature_kelvin"]
    sea_height_stretch = manifest["stretch"]["sea_height"]
    salinity_stretch = manifest["stretch"]["salinity"]
    depth_axis_m = np.asarray(manifest["depth_axis_m"], dtype=np.float32)
    depth_mask = depth_axis_m <= float(args.glorys_max_depth_m)
    if not np.any(depth_mask):
        raise ValueError(
            f"No GLORYS depth levels are <= {float(args.glorys_max_depth_m)} m."
        )
    depth_indices = (np.flatnonzero(depth_mask) + 1).astype(int).tolist()
    depth_values_m = depth_axis_m[depth_mask]
    argo_profiles = _load_argo_region_profiles(
        argo_path,
        bounds=region_bounds,
        date_text=date_text,
        depth_mask=depth_mask,
        depth_axis_m=depth_axis_m,
        temperature_stretch=temperature_stretch,
        salinity_stretch=salinity_stretch,
        chunk_size=int(args.argo_chunk_size),
        max_profile_lines=int(args.max_argo_profile_lines),
        profile_grid_size=int(args.en4_profile_grid_size),
        random_seed=int(args.argo_profile_random_seed),
    )

    sst, _, _, sst_extent = _read_decoded_raster_region(
        _dated_raster_path(ostia_dir, "analysed_sst", date_text),
        bounds=region_bounds,
        stretch=temperature_stretch,
        convert_kelvin_to_celsius=True,
    )
    sst = np.asarray(sst, dtype=np.float32)

    glorys, glorys_lon, glorys_lat, _ = _read_decoded_raster_region(
        _dated_raster_path(glorys_dir, "thetao", date_text),
        bounds=region_bounds,
        stretch=temperature_stretch,
        bands=depth_indices,
        convert_kelvin_to_celsius=True,
    )

    glorys_salinity, glorys_salinity_lon, glorys_salinity_lat, _ = (
        _read_decoded_raster_region(
            _dated_raster_path(glorys_salinity_dir, "so", date_text),
            bounds=region_bounds,
            stretch=salinity_stretch,
            bands=depth_indices,
            convert_kelvin_to_celsius=False,
        )
    )

    ssh, _, _, ssh_extent = _read_decoded_raster_region(
        _dated_raster_path(ssh_dir, "adt", date_text),
        bounds=region_bounds,
        stretch=sea_height_stretch,
        convert_kelvin_to_celsius=False,
    )
    ssh = np.asarray(ssh, dtype=np.float32)

    sss, _, _, sss_extent = _read_decoded_raster_region(
        _dated_raster_path(sss_dir, "sos", date_text),
        bounds=region_bounds,
        stretch=salinity_stretch,
        convert_kelvin_to_celsius=False,
    )
    sss = np.asarray(sss, dtype=np.float32)

    glorys = np.asarray(glorys, dtype=np.float32)
    glorys_salinity = np.asarray(glorys_salinity, dtype=np.float32)
    temp_limit_values = np.concatenate(
        [
            sst[np.isfinite(sst)].reshape(-1),
            glorys[np.isfinite(glorys)].reshape(-1),
        ]
    )
    salinity_limit_values = np.concatenate(
        [
            glorys_salinity[np.isfinite(glorys_salinity)].reshape(-1),
            sss[np.isfinite(sss)].reshape(-1),
        ]
    )
    temp_vmin, temp_vmax = _rounded_limits_from_values(temp_limit_values, step=1.0)
    salinity_vmin, salinity_vmax = _rounded_limits_from_values(
        salinity_limit_values, step=0.2
    )
    temp_norm = Normalize(vmin=temp_vmin, vmax=temp_vmax)
    salinity_norm = Normalize(vmin=salinity_vmin, vmax=salinity_vmax)
    temp_cmap = plt.get_cmap("turbo")
    salinity_cmap = plt.get_cmap("YlGnBu_r")

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7,
            "axes.titleweight": "bold",
            "figure.facecolor": "#ffffff",
            "savefig.dpi": int(args.dpi),
        }
    )
    fig = plt.figure(
        figsize=(float(args.fig_width), float(args.fig_height)),
        dpi=int(args.dpi),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.055, right=0.965, top=0.955, bottom=0.065)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.62, 1.0],
        hspace=0.34,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.14)
    middle = outer[1].subgridspec(1, 3, width_ratios=[1.0, 2.5, 1.08], wspace=0.24)
    bottom = outer[2].subgridspec(1, 3, wspace=0.14)
    ax_glorys_temp = fig.add_subplot(top[0, 0])
    ax_glorys_salinity = fig.add_subplot(top[0, 1])
    ax_argo_temp = fig.add_subplot(middle[0, 0], projection="3d")
    ax_map = fig.add_subplot(middle[0, 1])
    ax_argo_salinity = fig.add_subplot(middle[0, 2], projection="3d")
    ax_sst = fig.add_subplot(bottom[0, 0])
    ax_ssh = fig.add_subplot(bottom[0, 1])
    ax_sss = fig.add_subplot(bottom[0, 2])

    _draw_en4_profile_panel_3d(
        ax_argo_temp,
        profile_data=argo_profiles["temperature"],
        bounds=region_bounds,
        title="EN4 Temperature",
        norm=temp_norm,
        cmap=temp_cmap,
    )
    _draw_argo_density_map(
        fig,
        ax_map,
        argo_points=argo_points,
        land_mask_path=land_mask_path,
        world_geojson_path=world_geojson_path,
        region_bounds=region_bounds,
        hex_gridsize=int(args.hex_gridsize),
        land_stride=int(args.land_stride),
    )
    _draw_en4_profile_panel_3d(
        ax_argo_salinity,
        profile_data=argo_profiles["salinity"],
        bounds=region_bounds,
        title="EN4 Salinity",
        norm=salinity_norm,
        cmap=salinity_cmap,
    )

    _draw_glorys_cube(
        ax_glorys_temp,
        values_c=glorys,
        lon=glorys_lon,
        lat=glorys_lat,
        depth_m=np.asarray(depth_values_m, dtype=np.float32),
        title="GLORYS Temperature",
        norm=temp_norm,
        cmap=temp_cmap,
    )
    temp_mappable = cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap)
    temp_mappable.set_array(temp_limit_values)
    temp_cbar = fig.colorbar(
        temp_mappable,
        ax=ax_glorys_temp,
        orientation="horizontal",
        fraction=0.075,
        pad=0.08,
    )
    temp_cbar.set_label("Temperature (deg C)")
    _format_colorbar(temp_cbar, tick_format="%.0f", nbins=6)

    _draw_glorys_cube(
        ax_glorys_salinity,
        values_c=glorys_salinity,
        lon=glorys_salinity_lon,
        lat=glorys_salinity_lat,
        depth_m=np.asarray(depth_values_m, dtype=np.float32),
        title="GLORYS Salinity",
        norm=salinity_norm,
        cmap=salinity_cmap,
    )
    salinity_mappable = cm.ScalarMappable(norm=salinity_norm, cmap=salinity_cmap)
    salinity_mappable.set_array(salinity_limit_values)
    salinity_cbar = fig.colorbar(
        salinity_mappable,
        ax=ax_glorys_salinity,
        orientation="horizontal",
        fraction=0.075,
        pad=0.08,
    )
    salinity_cbar.set_label("Salinity (PSU)")
    _format_colorbar(salinity_cbar, tick_format="%.1f", nbins=5)

    _draw_raster_panel(
        fig,
        ax_sst,
        values=sst,
        extent=sst_extent,
        land_mask_path=land_mask_path,
        bounds=region_bounds,
        title="OSTIA SST",
        cmap_name="turbo",
        colorbar_label="Temperature (deg C)",
        norm=temp_norm,
        tick_format="%.0f",
    )

    ssh_center = float(np.nanmedian(ssh))
    ssh_limits = _robust_limits(
        ssh,
        lower_percentile=2.0,
        upper_percentile=98.0,
        center=ssh_center,
    )
    ssh_step = 0.05
    ssh_vmin = float(np.floor(ssh_limits[0] / ssh_step) * ssh_step)
    ssh_vmax = float(np.ceil(ssh_limits[1] / ssh_step) * ssh_step)
    if ssh_vmin >= ssh_center:
        ssh_vmin = ssh_center - ssh_step
    if ssh_vmax <= ssh_center:
        ssh_vmax = ssh_center + ssh_step
    _draw_raster_panel(
        fig,
        ax_ssh,
        values=ssh,
        extent=ssh_extent,
        land_mask_path=land_mask_path,
        bounds=region_bounds,
        title="SSH ADT",
        cmap_name="RdBu_r",
        colorbar_label="ADT (m)",
        norm=TwoSlopeNorm(vmin=ssh_vmin, vcenter=ssh_center, vmax=ssh_vmax),
        tick_format="%.2f",
    )

    _draw_raster_panel(
        fig,
        ax_sss,
        values=sss,
        extent=sss_extent,
        land_mask_path=land_mask_path,
        bounds=region_bounds,
        title="SSS",
        cmap_name="YlGnBu_r",
        colorbar_label="Salinity (PSU)",
        norm=salinity_norm,
        tick_format="%.1f",
    )

    output_png = Path(args.output_png)
    output_webp = Path(args.output_webp)
    width, height = _save_outputs(fig, output_png=output_png, output_webp=output_webp)
    plt.close(fig)

    metadata = {
        "date": date_text,
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "argo_path": str(argo_path),
        "ostia_raster": str(_dated_raster_path(ostia_dir, "analysed_sst", date_text)),
        "glorys_temperature_raster": str(
            _dated_raster_path(glorys_dir, "thetao", date_text)
        ),
        "glorys_salinity_raster": str(
            _dated_raster_path(glorys_salinity_dir, "so", date_text)
        ),
        "ssh_raster": str(_dated_raster_path(ssh_dir, "adt", date_text)),
        "sss_raster": str(_dated_raster_path(sss_dir, "sos", date_text)),
        "land_mask_path": str(land_mask_path),
        "world_geojson_path": str(world_geojson_path),
        "region_bounds": {
            "lon_min": float(region_bounds[0]),
            "lon_max": float(region_bounds[1]),
            "lat_min": float(region_bounds[2]),
            "lat_max": float(region_bounds[3]),
        },
        "argo_profile_count": int(argo_points["profile_count"]),
        "argo_processed_profile_count": int(argo_points["processed_profile_count"]),
        "argo_valid_profile_count": int(argo_points["valid_profile_count"]),
        "argo_ocean_profile_count": int(argo_points.get("ocean_profile_count", 0)),
        "argo_week_count": int(argo_points["week_count"]),
        "argo_temperature_profile_filtered_count": int(
            argo_profiles["temperature"]["filtered_count"]
        ),
        "en4_temperature_profile_column_count": int(
            argo_profiles["temperature"]["column_count"]
        ),
        "en4_temperature_profile_outlier_column_count": int(
            argo_profiles["temperature"]["outlier_column_count"]
        ),
        "argo_temperature_profile_plotted_count": int(
            argo_profiles["temperature"]["plotted_count"]
        ),
        "argo_salinity_profile_filtered_count": int(
            argo_profiles["salinity"]["filtered_count"]
        ),
        "en4_salinity_profile_column_count": int(
            argo_profiles["salinity"]["column_count"]
        ),
        "en4_salinity_profile_outlier_column_count": int(
            argo_profiles["salinity"]["outlier_column_count"]
        ),
        "argo_salinity_profile_plotted_count": int(
            argo_profiles["salinity"]["plotted_count"]
        ),
        "rendering": {
            "projection": "Equal Earth",
            "hex_gridsize": int(args.hex_gridsize),
            "argo_density_units": "mean_profiles_per_week_per_hex",
            "max_argo_profile_lines": int(args.max_argo_profile_lines),
            "en4_profile_grid_size": int(args.en4_profile_grid_size),
            "en4_profile_min_valid_depth_m": 150.0,
            "en4_profile_outlier_filter": "column_5th_95th_percentile_iqr_1p5_temperature_low_tail_p15",
            "argo_profile_random_seed": int(args.argo_profile_random_seed),
            "land_stride": int(args.land_stride),
            "glorys_max_depth_m": float(args.glorys_max_depth_m),
            "glorys_render": "temperature_and_salinity_cubes",
            "temperature_vmin_c": float(temp_vmin),
            "temperature_vmax_c": float(temp_vmax),
            "glorys_salinity_vmin_psu": float(salinity_vmin),
            "glorys_salinity_vmax_psu": float(salinity_vmax),
            "ssh_vmin_m": float(ssh_vmin),
            "ssh_vmax_m": float(ssh_vmax),
            "sss_vmin_psu": float(salinity_vmin),
            "sss_vmax_psu": float(salinity_vmax),
            "figure_width_in": float(args.fig_width),
            "figure_height_in": float(args.fig_height),
            "dpi": int(args.dpi),
            "image_width_px": int(width),
            "image_height_px": int(height),
        },
        "outputs": {
            "png": str(output_png),
            "webp": str(output_webp),
        },
    }

    metadata_json = Path(args.metadata_json)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    metadata["outputs"]["metadata_json"] = str(metadata_json)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create the DepthDif paper header visualization."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--argo-path", type=Path, default=None)
    parser.add_argument("--ostia-raster-dir", type=Path, default=None)
    parser.add_argument("--glorys-raster-dir", type=Path, default=None)
    parser.add_argument("--glorys-salinity-raster-dir", type=Path, default=None)
    parser.add_argument("--ssh-raster-dir", type=Path, default=None)
    parser.add_argument("--sss-raster-dir", type=Path, default=None)
    parser.add_argument("--land-mask-path", type=Path, default=None)
    parser.add_argument("--world-geojson-path", type=Path, default=None)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--region-bounds",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=DEFAULT_REGION_BOUNDS,
    )
    parser.add_argument("--glorys-max-depth-m", type=float, default=1000.0)
    parser.add_argument(
        "--max-argo-profiles",
        type=int,
        default=0,
        help="Optional profile limit for smoke renders. Use 0 to process all profiles.",
    )
    parser.add_argument("--hex-gridsize", type=int, default=41)
    parser.add_argument(
        "--max-argo-profile-lines",
        type=int,
        default=180,
        help="Maximum coarsened selected-week regional EN4 columns to draw per 3D panel.",
    )
    parser.add_argument(
        "--en4-profile-grid-size",
        type=int,
        default=9,
        help="Coarse lon/lat grid size used to average aligned EN4 profiles into thicker columns.",
    )
    parser.add_argument(
        "--argo-profile-random-seed",
        type=int,
        default=0,
        help="Random seed for deterministic ARGO profile sampling.",
    )
    parser.add_argument("--land-stride", type=int, default=8)
    parser.add_argument("--argo-chunk-size", type=int, default=200000)
    parser.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH_IN)
    parser.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT_IN)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--output-png", type=Path, default=DEFAULT_OUTPUT_STEM.with_suffix(".png")
    )
    parser.add_argument(
        "--output-webp", type=Path, default=DEFAULT_OUTPUT_STEM.with_suffix(".webp")
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=DEFAULT_OUTPUT_STEM.with_suffix(".json"),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    metadata = create_header_image(parse_args())
    print(f"Wrote PNG: {metadata['outputs']['png']}")
    print(f"Wrote WebP: {metadata['outputs']['webp']}")
    print(f"Wrote metadata: {metadata['outputs']['metadata_json']}")


if __name__ == "__main__":
    main()
