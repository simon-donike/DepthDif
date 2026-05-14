from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from depth_recon.paths import resolve_package_path

MISSING_TEXT_VALUES = frozenset({"", "__missing__", "nan", "none", "null"})
DEFAULT_LAND_MASK_PATH = str(
    Path(__file__).resolve().parent
    / "dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif"
)


def _parse_date_int(value: Any) -> int:
    """Parse a model date integer while avoiding leap-day calendar issues."""
    raw = str(value).strip()
    if raw.isdigit():
        date_int = int(raw)
        month = (date_int // 100) % 100
        day = date_int % 100
        # Keep dataset dates compatible with the model's fixed non-leap calendar.
        if month == 2 and day == 29:
            return date_int - 1
        return date_int
    return 20100101


def _normalize_lon(lon: float) -> float:
    """Normalize longitude to the -180..180 degree range."""
    return float(((float(lon) + 180.0) % 360.0) - 180.0)


def _center_lon_deg(lon0: float, lon1: float) -> float:
    """Return the circular midpoint longitude in degrees."""
    lon0_rad = np.deg2rad(lon0)
    lon1_rad = np.deg2rad(lon1)
    sin_sum = np.sin(lon0_rad) + np.sin(lon1_rad)
    cos_sum = np.cos(lon0_rad) + np.cos(lon1_rad)
    return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))


@dataclass(frozen=True)
class _ForceIncludeRegion:
    """Named region that relaxes land-fraction filtering for patch centers."""

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    max_land_fraction: float


@dataclass(frozen=True)
class _GridParams:
    """Patch-grid construction parameters shared by dataset backends."""

    tile_size: int
    resolution_deg: float
    invalid_threshold: float
    invalid_mask_flags: tuple[str, ...]
    val_fraction: float
    val_year: int | None
    split_seed: int
    patch_grid_source: str = "land_mask"
    land_mask_path: str | Path | None = DEFAULT_LAND_MASK_PATH
    patch_stride: int | None = None
    max_land_fraction: float = 0.30
    force_include_regions: tuple[_ForceIncludeRegion, ...] = ()

    @property
    def effective_patch_stride(self) -> int:
        """Return the configured stride, defaulting to non-overlapping tiles."""
        return int(self.tile_size if self.patch_stride is None else self.patch_stride)


@dataclass(frozen=True)
class _PatchGridLookup:
    """Compact lookup from global pixel coordinates to retained patch ids."""

    patch_by_start: dict[tuple[int, int], int]
    y_starts: np.ndarray
    x_starts: np.ndarray
    grid_top: float
    grid_left: float
    tile_size: int
    resolution_deg: float


def _sanitize_cache_text(value: Any) -> str:
    """Sanitize arbitrary config text for use in cache filenames."""
    text = str(value).strip().lower().replace("\\", "/")
    for old, new in (("/", "-"), (".", "p"), (" ", ""), (":", "-")):
        text = text.replace(old, new)
    return text


def _path_cache_hash(path: str | Path | None) -> str:
    """Return a short stable hash for a path-like cache key."""
    if path is None:
        return "none"
    raw = str(Path(path)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def _deep_update_config(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy of a config mapping with nested override values applied."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update_config(out[key], value)
        else:
            out[key] = value
    return out


def _force_include_cache_hash(regions: Sequence[_ForceIncludeRegion]) -> str:
    """Return a short stable hash for force-include region settings."""
    if not regions:
        return "none"
    parts = [
        (
            region.name,
            f"{region.lon_min:.6f}",
            f"{region.lon_max:.6f}",
            f"{region.lat_min:.6f}",
            f"{region.lat_max:.6f}",
            f"{region.max_land_fraction:.6f}",
        )
        for region in regions
    ]
    raw = repr(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def _parse_force_include_regions(value: Any) -> tuple[_ForceIncludeRegion, ...]:
    """Parse optional force-include region mappings from dataset config."""
    if value is None:
        return ()
    if isinstance(value, str) and value.strip().lower() in MISSING_TEXT_VALUES:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("grid.force_include_regions must be a list of mappings.")

    regions: list[_ForceIncludeRegion] = []
    for idx, raw_region in enumerate(value):
        if not isinstance(raw_region, dict):
            raise ValueError("Each grid.force_include_regions item must be a mapping.")
        name = str(raw_region.get("name", f"region_{idx}"))
        lon_min = float(raw_region["lon_min"])
        lon_max = float(raw_region["lon_max"])
        lat_min = float(raw_region["lat_min"])
        lat_max = float(raw_region["lat_max"])
        max_land_fraction = float(raw_region.get("max_land_fraction", 1.0))
        regions.append(
            _ForceIncludeRegion(
                name=name,
                lon_min=min(lon_min, lon_max),
                lon_max=max(lon_min, lon_max),
                lat_min=min(lat_min, lat_max),
                lat_max=max(lat_min, lat_max),
                max_land_fraction=max_land_fraction,
            )
        )
    return tuple(regions)


def _grid_starts(size: int, tile: int, stride: int) -> list[int]:
    """Return grid start indices that always include the final valid tile."""
    if tile < 1:
        raise ValueError("grid.tile_size must be >= 1.")
    if stride < 1:
        raise ValueError("grid.patch_stride must be >= 1.")
    if int(size) < int(tile):
        raise RuntimeError("Source grid is smaller than the requested tile size.")

    last_start = int(size) - int(tile)
    starts = list(range(0, last_start + 1, int(stride)))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _summed_area_table(mask: np.ndarray) -> np.ndarray:
    """Build a summed-area table for fast rectangular mask sums."""
    values = np.asarray(mask, dtype=np.float64)
    table = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return table


def _window_sum(table: np.ndarray, *, y0: int, x0: int, tile: int) -> float:
    """Return a square-window sum from a summed-area table."""
    y1 = int(y0) + int(tile)
    x1 = int(x0) + int(tile)
    return float(
        table[y1, x1]
        - table[int(y0), x1]
        - table[y1, int(x0)]
        + table[int(y0), int(x0)]
    )


def _validate_grid_params(grid_params: _GridParams) -> None:
    """Validate patch-grid settings before building a registry."""
    tile = int(grid_params.tile_size)
    stride = int(grid_params.effective_patch_stride)
    if tile < 1:
        raise ValueError("grid.tile_size must be >= 1.")
    if stride < 1:
        raise ValueError("grid.patch_stride must be >= 1.")
    if stride < tile and grid_params.val_year is None:
        raise ValueError(
            "Overlapping patch grids require split.val_year to avoid spatial "
            "train/val leakage. Set split.val_year or use patch_stride >= tile_size."
        )
    if not (0.0 <= float(grid_params.max_land_fraction) <= 1.0):
        raise ValueError("grid.max_land_fraction must be in [0, 1].")
    for region in grid_params.force_include_regions:
        if not (0.0 <= float(region.max_land_fraction) <= 1.0):
            raise ValueError(
                "grid.force_include_regions[].max_land_fraction must be in [0, 1]."
            )
    source = str(grid_params.patch_grid_source).strip().lower()
    if source not in {"land_mask", "ostia_mask"}:
        raise ValueError("grid.patch_grid_source must be 'land_mask' or 'ostia_mask'.")


def _force_include_region_for_patch(
    *,
    lat_center: float,
    lon_center: float,
    land_fraction: float,
    regions: Sequence[_ForceIncludeRegion],
) -> _ForceIncludeRegion | None:
    """Return the matching force-include region for a patch, if any."""
    for region in regions:
        lon_value = _normalize_lon(float(lon_center))
        if (
            region.lat_min <= float(lat_center) <= region.lat_max
            and region.lon_min <= lon_value <= region.lon_max
            and float(land_fraction) <= float(region.max_land_fraction)
        ):
            return region
    return None


def _build_patch_lookup(
    patch_df: pd.DataFrame, grid_params: _GridParams
) -> _PatchGridLookup:
    """Build a compact lookup from retained patch starts to patch ids."""
    if patch_df.empty:
        raise RuntimeError("Cannot build patch lookup from an empty patch table.")

    records = patch_df.to_dict(orient="records")
    first = records[0]
    resolution = float(grid_params.resolution_deg)
    grid_top = max(float(first["lat0"]), float(first["lat1"])) + (
        int(first["grid_y0"]) * resolution
    )
    grid_left = min(float(first["lon0"]), float(first["lon1"])) - (
        int(first["grid_x0"]) * resolution
    )
    patch_by_start = {
        (int(row["grid_y0"]), int(row["grid_x0"])): int(row["patch_id"])
        for row in records
    }
    y_starts = np.asarray(
        sorted({int(row["grid_y0"]) for row in records}), dtype=np.int64
    )
    x_starts = np.asarray(
        sorted({int(row["grid_x0"]) for row in records}), dtype=np.int64
    )
    return _PatchGridLookup(
        patch_by_start=patch_by_start,
        y_starts=y_starts,
        x_starts=x_starts,
        grid_top=float(grid_top),
        grid_left=float(grid_left),
        tile_size=int(grid_params.tile_size),
        resolution_deg=resolution,
    )


def _candidate_starts_for_pixel(
    starts: np.ndarray, pixel_idx: int, tile: int
) -> np.ndarray:
    """Return patch start indices whose tile contains one pixel index."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return starts
    mask = (starts <= int(pixel_idx)) & (int(pixel_idx) < (starts + int(tile)))
    return starts[mask]


def _patch_ids_for_profile(
    lookup: _PatchGridLookup,
    *,
    lat: float,
    lon: float,
) -> list[int]:
    """Return all retained patch ids containing one profile location."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        return []

    row_idx = int(
        np.floor((float(lookup.grid_top) - float(lat)) / lookup.resolution_deg)
    )
    lon_value = _normalize_lon(float(lon))
    if float(lookup.grid_left) >= 0.0 and lon_value < float(lookup.grid_left):
        # Some legacy OSTIA grids use 0..360 longitude coordinates while ARGO
        # profile longitudes are normalized to -180..180.
        lon_value += 360.0
    col_idx = int(
        np.floor((lon_value - float(lookup.grid_left)) / lookup.resolution_deg)
    )
    y_candidates = _candidate_starts_for_pixel(
        lookup.y_starts,
        row_idx,
        lookup.tile_size,
    )
    x_candidates = _candidate_starts_for_pixel(
        lookup.x_starts,
        col_idx,
        lookup.tile_size,
    )
    patch_ids: list[int] = []
    for y0 in y_candidates.tolist():
        for x0 in x_candidates.tolist():
            patch_id = lookup.patch_by_start.get((int(y0), int(x0)))
            if patch_id is not None:
                patch_ids.append(int(patch_id))
    return patch_ids


def _build_land_mask_patch_table(grid_params: _GridParams) -> pd.DataFrame:
    """Build retained patch metadata from the authoritative land-mask GeoTIFF."""
    land_mask_path = resolve_package_path(
        DEFAULT_LAND_MASK_PATH
        if grid_params.land_mask_path is None
        else grid_params.land_mask_path
    )
    if not land_mask_path.exists():
        raise FileNotFoundError(f"Land-mask GeoTIFF does not exist: {land_mask_path}")

    with rasterio.open(land_mask_path) as src:
        land_mask = src.read(1)
        transform = src.transform
        width = int(src.width)
        height = int(src.height)

    tile = int(grid_params.tile_size)
    stride = int(grid_params.effective_patch_stride)
    resolution = float(grid_params.resolution_deg)
    if not np.isclose(
        float(transform.a), resolution, rtol=0.0, atol=1.0e-8
    ) or not np.isclose(
        abs(float(transform.e)),
        resolution,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise RuntimeError(
            "Land-mask GeoTIFF resolution does not match dataset.grid.resolution_deg: "
            f"{float(transform.a)} x {abs(float(transform.e))} != {resolution}"
        )

    y_starts = _grid_starts(height, tile, stride)
    x_starts = _grid_starts(width, tile, stride)
    land_bool = np.asarray(land_mask, dtype=np.float32) > 0.5
    table = _summed_area_table(land_bool)
    max_land_fraction = float(grid_params.max_land_fraction)

    records: list[dict[str, Any]] = []
    patch_id = 0
    for y0 in tqdm(
        y_starts,
        desc="Building land-mask patch grid",
        unit="row",
        dynamic_ncols=True,
    ):
        for x0 in x_starts:
            land_fraction = _window_sum(table, y0=y0, x0=x0, tile=tile) / float(
                tile * tile
            )
            left = float(transform.c) + (float(x0) * resolution)
            right = left + (float(tile) * resolution)
            top = float(transform.f) - (float(y0) * resolution)
            bottom = top - (float(tile) * resolution)
            lat_center = 0.5 * (float(bottom) + float(top))
            lon_center = _center_lon_deg(float(left), float(right))
            force_region = _force_include_region_for_patch(
                lat_center=lat_center,
                lon_center=lon_center,
                land_fraction=land_fraction,
                regions=grid_params.force_include_regions,
            )
            if land_fraction > max_land_fraction and force_region is None:
                continue
            records.append(
                {
                    "patch_id": int(patch_id),
                    "grid_y0": int(y0),
                    "grid_x0": int(x0),
                    "lat0": float(bottom),
                    "lat1": float(top),
                    "lon0": float(left),
                    "lon1": float(right),
                    "lat_center": lat_center,
                    "lon_center": lon_center,
                    "land_fraction": float(land_fraction),
                    "ocean_fraction": float(1.0 - land_fraction),
                    "invalid_fraction": float(land_fraction),
                    "force_included": bool(force_region is not None),
                    "force_include_region": (
                        "" if force_region is None else force_region.name
                    ),
                }
            )
            patch_id += 1

    if not records:
        raise RuntimeError("No valid patches were built from the land-mask grid.")
    return pd.DataFrame.from_records(records)
