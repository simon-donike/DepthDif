from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from depth_recon.data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_DEPTH_VAR,
    TimedFile,
    date_to_days_since_1950,
    open_argo_dataset,
    scan_timed_files,
)
from depth_recon.data.dataset_grid_utils import _normalize_lon

GLORYS_RELATIVE_DEPTH_CUTOFF = 0.10
GLORYS_MIN_ABSOLUTE_DEPTH_CUTOFF_M = 10.0


def _datetime_from_yyyymmdd(value: int) -> datetime:
    """Convert a compact YYYYMMDD integer to a datetime."""
    return datetime.strptime(str(int(value)), "%Y%m%d")


def _date_range_yyyymmdd(center_date: int, radius_days: int) -> list[int]:
    """Return compact dates around a center day."""
    center = _datetime_from_yyyymmdd(int(center_date))
    return [
        int((center + timedelta(days=offset)).strftime("%Y%m%d"))
        for offset in range(-int(radius_days), int(radius_days) + 1)
    ]


def _yyyymmdd_from_days_since_1950(day_value: float) -> int:
    """Convert days since 1950-01-01 to compact YYYYMMDD."""
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


def _juld_to_yyyymmdd(juld_days: np.ndarray) -> np.ndarray:
    """Convert EN4 JULD day offsets to compact YYYYMMDD integers."""
    out = np.zeros(juld_days.shape, dtype=np.int32)
    valid = np.isfinite(juld_days) & (juld_days < 90000.0) & (juld_days > -20000.0)
    if not np.any(valid):
        return out
    dates = np.datetime64("1950-01-01", "D") + np.floor(juld_days[valid]).astype(
        "timedelta64[D]"
    )
    compact = np.char.replace(np.datetime_as_string(dates, unit="D"), "-", "")
    out[valid] = compact.astype(np.int32)
    return out


def _first_present_name(names: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Return the first candidate name present in an ordered name collection."""
    available = set(names)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _collapse_duplicate_profile_depths(
    depth: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average values at duplicate profile depths before interpolation."""
    unique_depths, inverse = np.unique(depth, return_inverse=True)
    if unique_depths.size == depth.size:
        return depth, temperature
    # Interpolation expects one value per source depth; repeats are averaged.
    temp_sum = np.bincount(inverse, weights=temperature)
    temp_count = np.bincount(inverse)
    return (
        unique_depths.astype(np.float64, copy=False),
        (temp_sum / np.maximum(temp_count, 1)).astype(np.float64, copy=False),
    )


def _align_argo_profile_to_glorys_depths(
    *,
    temperature: np.ndarray,
    depth: np.ndarray,
    glorys_depths: np.ndarray,
) -> np.ndarray:
    """Project one ARGO profile onto the fixed GLORYS depth axis."""
    target_depths = np.asarray(glorys_depths, dtype=np.float64).reshape(-1)
    out = np.full(target_depths.shape, np.nan, dtype=np.float32)
    temp = np.asarray(temperature, dtype=np.float64).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(temp) & np.isfinite(depth) & (depth >= 0.0)
    if not np.any(valid):
        return out

    depth = depth[valid]
    temp = temp[valid]
    order = np.argsort(depth, kind="mergesort")
    depth = depth[order]
    temp = temp[order]
    depth, temp = _collapse_duplicate_profile_depths(depth, temp)
    if depth.size == 0:
        return out

    insert_idx = np.searchsorted(depth, target_depths, side="left")
    left_idx = np.clip(insert_idx - 1, 0, max(depth.size - 1, 0))
    right_idx = np.clip(insert_idx, 0, max(depth.size - 1, 0))
    nearest_depth_distance = np.minimum(
        np.abs(target_depths - depth[left_idx]),
        np.abs(target_depths - depth[right_idx]),
    )
    max_allowed_distance = np.maximum(
        GLORYS_RELATIVE_DEPTH_CUTOFF * target_depths,
        GLORYS_MIN_ABSOLUTE_DEPTH_CUTOFF_M,
    )
    valid_targets = (
        np.isfinite(target_depths)
        & (target_depths >= depth[0])
        & (target_depths <= depth[-1])
        & np.isfinite(nearest_depth_distance)
        & (nearest_depth_distance <= max_allowed_distance)
    )
    if not np.any(valid_targets):
        return out

    if depth.size == 1:
        exact = valid_targets & np.isclose(
            target_depths, depth[0], rtol=0.0, atol=1.0e-6
        )
        out[exact] = np.float32(temp[0])
        return out

    out[valid_targets] = np.interp(target_depths[valid_targets], depth, temp).astype(
        np.float32,
        copy=False,
    )
    return out


@dataclass(frozen=True)
class PatchAxes:
    """Latitude and longitude axes for one patch read."""

    lat_axis: np.ndarray
    lon_axis: np.ndarray


class DatasetCache:
    """Small LRU cache for xarray NetCDF datasets opened by one worker process."""

    def __init__(self, max_open: int = 8) -> None:
        """Initialize a bounded path-to-dataset cache."""
        self.max_open = int(max_open)
        self._items: OrderedDict[Path, xr.Dataset] = OrderedDict()

    def get(self, path: Path) -> xr.Dataset:
        """Return an open dataset for one path."""
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


class TimedNetCDFStore:
    """Date-indexed NetCDF source folder with lazy per-file reads."""

    LAT_CANDIDATES = ("latitude", "lat")
    LON_CANDIDATES = ("longitude", "lon")

    def __init__(self, root_dir: str | Path, *, cache_size: int = 8) -> None:
        """Initialize a date-indexed NetCDF source directory."""
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"NetCDF root does not exist: {self.root_dir}")
        self.index = scan_timed_files(self.root_dir, show_progress=False)
        if not self.index:
            raise RuntimeError(f"No readable NetCDF files found in: {self.root_dir}")
        self.cache = DatasetCache(max_open=cache_size)

    @property
    def dates(self) -> list[int]:
        """Return available source dates as compact YYYYMMDD integers."""
        return [_yyyymmdd_from_days_since_1950(item.day) for item in self.index]

    def depth_axis_m(self, coord_name: str = "depth") -> np.ndarray:
        """Read the first available depth axis from indexed source files."""
        for item in self.index:
            ds = self.cache.get(item.path)
            if coord_name in ds.coords or coord_name in ds.variables:
                depth = np.asarray(ds[coord_name].values, dtype=np.float32).reshape(-1)
                depth = depth[np.isfinite(depth)]
                if depth.size > 0:
                    return depth.astype(np.float32, copy=False)
        raise RuntimeError(
            f"No readable {coord_name!r} depth axis found in {self.root_dir}"
        )

    def bracket(self, target_date: int) -> tuple[TimedFile, TimedFile, float]:
        """Find source files bracketing a target date."""
        target_day = date_to_days_since_1950(int(target_date))
        days = np.asarray([item.day for item in self.index], dtype=np.float64)
        pos = int(np.searchsorted(days, float(target_day), side="left"))
        if pos < len(self.index) and np.isclose(
            days[pos], target_day, rtol=0.0, atol=1.0e-8
        ):
            return self.index[pos], self.index[pos], 0.0
        if pos == 0:
            return self.index[0], self.index[0], 0.0
        if pos >= len(self.index):
            return self.index[-1], self.index[-1], 0.0
        before = self.index[pos - 1]
        after = self.index[pos]
        span = after.day - before.day
        weight = 0.0 if span <= 0.0 else float((float(target_day) - before.day) / span)
        return before, after, weight

    def read_patch(
        self,
        *,
        target_date: int,
        var_name: str,
        axes: PatchAxes,
        categorical: bool = False,
    ) -> np.ndarray:
        """Read or interpolate one variable onto patch axes."""
        before, after, weight = self.bracket(int(target_date))
        if categorical or before.path == after.path:
            selected = before if weight <= 0.5 else after
            return self._read_one_patch(
                selected.path,
                var_name=var_name,
                axes=axes,
                categorical=categorical,
            )

        first = self._read_one_patch(
            before.path,
            var_name=var_name,
            axes=axes,
            categorical=False,
        )
        second = self._read_one_patch(
            after.path,
            var_name=var_name,
            axes=axes,
            categorical=False,
        )
        return (first + ((second - first) * np.float32(weight))).astype(
            np.float32, copy=False
        )

    def _read_one_patch(
        self,
        path: Path,
        *,
        var_name: str,
        axes: PatchAxes,
        categorical: bool,
    ) -> np.ndarray:
        """Read one source file and sample it on patch axes."""
        ds = self.cache.get(path)
        if var_name not in ds:
            raise RuntimeError(f"Variable {var_name!r} is missing from NetCDF: {path}")
        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=0)
        lat_name = _first_present_name(da.dims, self.LAT_CANDIDATES)
        lon_name = _first_present_name(da.dims, self.LON_CANDIDATES)
        if lat_name is None or lon_name is None:
            raise RuntimeError(
                f"Variable {var_name!r} in {path} does not have lat/lon dimensions."
            )

        lon_axis = self._lon_axis_for_source(da, lon_name, axes.lon_axis)
        method = "nearest" if categorical else "linear"
        sampled = da.interp(
            {lat_name: axes.lat_axis, lon_name: lon_axis},
            method=method,
        )
        if "depth" in sampled.dims:
            sampled = sampled.transpose("depth", lat_name, lon_name)
        else:
            sampled = sampled.transpose(lat_name, lon_name)
        return np.asarray(sampled.values, dtype=np.float32)

    @staticmethod
    def _lon_axis_for_source(
        da: xr.DataArray,
        lon_name: str,
        lon_axis: np.ndarray,
    ) -> np.ndarray:
        """Normalize target longitudes for the source coordinate convention."""
        source_lons = np.asarray(da[lon_name].values, dtype=np.float64)
        if source_lons.size == 0:
            return lon_axis
        if np.nanmin(source_lons) >= 0.0 and np.nanmax(source_lons) > 180.0:
            return np.mod(lon_axis, 360.0)
        return lon_axis


class ArgoNetCDFStore:
    """Raw EN4/ARGO NetCDF access plus profile/date filtering."""

    REQUIRED_VARS = ("JULD", "LATITUDE", "LONGITUDE")

    def __init__(
        self,
        root_dir: str | Path,
        *,
        depth_axis_m: np.ndarray,
        temp_var_name: str = "TEMP",
        depth_var_name: str = ARGO_DEPTH_VAR,
        cache_size: int = 8,
    ) -> None:
        """Initialize a raw ARGO profile source directory."""
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"ARGO NetCDF root does not exist: {self.root_dir}")
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        if self.depth_axis_m.size == 0:
            raise RuntimeError("ARGO NetCDF store needs a non-empty GLORYS depth axis.")
        self.temp_var_name = str(temp_var_name)
        self.depth_var_name = str(depth_var_name)
        self.files = self._discover_files()
        if not self.files:
            raise RuntimeError(f"No ARGO/EN4 NetCDF files found in: {self.root_dir}")

        self.cache_size = int(cache_size)
        self._cache: OrderedDict[Path, xr.Dataset] = OrderedDict()
        self._build_profile_index()

    @staticmethod
    def _normalize_profile_lon(value: Any) -> float:
        """Normalize one profile longitude to the project convention."""
        return _normalize_lon(float(value))

    def _discover_files(self) -> list[Path]:
        """Return candidate EN4/ARGO source files."""
        files = sorted(self.root_dir.glob("EN.4.2.2.f.profiles.g10.*.nc"))
        if files:
            return files
        return sorted(self.root_dir.glob("*.nc"))

    def _get_dataset(self, path: Path) -> xr.Dataset:
        """Return an open cached ARGO dataset."""
        path = Path(path)
        if path in self._cache:
            ds = self._cache.pop(path)
            self._cache[path] = ds
            return ds
        ds = open_argo_dataset(path)
        self._cache[path] = ds
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.close()
        return ds

    def _build_profile_index(self) -> None:
        """Build profile-level date and coordinate arrays."""
        dates: list[np.ndarray] = []
        latitudes: list[np.ndarray] = []
        longitudes: list[np.ndarray] = []
        file_indices: list[np.ndarray] = []
        profile_indices: list[np.ndarray] = []
        valid_temps: list[np.ndarray] = []

        for file_idx, path in enumerate(self.files):
            with open_argo_dataset(path) as ds:
                missing = [name for name in self.REQUIRED_VARS if name not in ds]
                missing += [
                    name
                    for name in (self.temp_var_name, self.depth_var_name)
                    if name not in ds
                ]
                if missing:
                    raise RuntimeError(
                        f"ARGO NetCDF file is missing required variables {missing}: {path}"
                    )

                juld = np.asarray(ds["JULD"].values, dtype=np.float64).reshape(-1)
                lat = np.asarray(ds["LATITUDE"].values, dtype=np.float64).reshape(-1)
                lon = np.asarray(ds["LONGITUDE"].values, dtype=np.float64).reshape(-1)
                n_prof = min(int(juld.size), int(lat.size), int(lon.size))
                if n_prof == 0:
                    continue
                temp = self._read_profile_matrix(
                    ds, self.temp_var_name, np.arange(n_prof)
                )
                depth = self._read_profile_matrix(
                    ds, self.depth_var_name, np.arange(n_prof)
                )
                valid_level = np.isfinite(temp) & np.isfinite(depth) & (depth >= 0.0)

                dates.append(_juld_to_yyyymmdd(juld[:n_prof]))
                latitudes.append(lat[:n_prof])
                longitudes.append(np.asarray([_normalize_lon(v) for v in lon[:n_prof]]))
                file_indices.append(np.full((n_prof,), int(file_idx), dtype=np.int32))
                profile_indices.append(np.arange(n_prof, dtype=np.int32))
                valid_temps.append(valid_level.any(axis=1))

        if not dates:
            raise RuntimeError(
                f"No profiles found in ARGO NetCDF root: {self.root_dir}"
            )
        self.profile_date = np.concatenate(dates).astype(np.int64, copy=False)
        self.latitude = np.concatenate(latitudes).astype(np.float64, copy=False)
        self.longitude = np.concatenate(longitudes).astype(np.float64, copy=False)
        self.file_index = np.concatenate(file_indices).astype(np.int32, copy=False)
        self.profile_index = np.concatenate(profile_indices).astype(
            np.int32, copy=False
        )
        self._has_valid_temp = np.concatenate(valid_temps).astype(bool, copy=False)
        self._indices_by_date = self._build_indices_by_date()

    def _build_indices_by_date(self) -> dict[int, np.ndarray]:
        """Group profile row indices by compact date."""
        out: dict[int, np.ndarray] = {}
        for date_value in np.unique(self.profile_date):
            if int(date_value) <= 0:
                continue
            out[int(date_value)] = np.flatnonzero(self.profile_date == int(date_value))
        return out

    @staticmethod
    def _replace_en4_fill_with_nan(values: np.ndarray) -> np.ndarray:
        """Replace EN4 fill and sentinel values with NaN."""
        out = np.asarray(values, dtype=np.float32)
        out[np.isclose(out, 99999.0)] = np.nan
        out[(~np.isfinite(out)) | (np.abs(out) > 1.0e6)] = np.nan
        return out

    @classmethod
    def _read_profile_matrix(
        cls,
        ds: xr.Dataset,
        var_name: str,
        profile_indices: np.ndarray,
    ) -> np.ndarray:
        """Read one profile-by-level variable as a float32 matrix."""
        da = ds[var_name]
        if da.ndim < 2:
            raise RuntimeError(f"ARGO variable {var_name!r} must be at least 2D.")
        profile_dim = da.dims[0]
        values = da.isel(
            {profile_dim: np.asarray(profile_indices, dtype=np.int64)}
        ).values
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            values = values.reshape((int(profile_indices.size), -1))
        return cls._replace_en4_fill_with_nan(values)

    def query_indices(
        self,
        *,
        target_date: int,
        temporal_window_days: int,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        """Return profile indices matching one date window and patch bounds."""
        radius = int(temporal_window_days) // 2
        date_values = _date_range_yyyymmdd(int(target_date), radius)
        date_indices = [
            self._indices_by_date[date_value]
            for date_value in date_values
            if date_value in self._indices_by_date
        ]
        if not date_indices:
            return np.zeros((0,), dtype=np.int64)

        indices = np.concatenate(date_indices).astype(np.int64, copy=False)
        lat_lo = min(float(lat0), float(lat1))
        lat_hi = max(float(lat0), float(lat1))
        lon_lo = _normalize_lon(min(float(lon0), float(lon1)))
        lon_hi = _normalize_lon(max(float(lon0), float(lon1)))
        lat = self.latitude[indices]
        lon = self.longitude[indices]
        mask = (
            self._has_valid_temp[indices]
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= lat_lo)
            & (lat < lat_hi)
        )
        if lon_lo <= lon_hi:
            mask &= (lon >= lon_lo) & (lon < lon_hi)
        else:
            mask &= (lon >= lon_lo) | (lon < lon_hi)
        return indices[mask]

    def load_temperature_profiles(self, indices: np.ndarray) -> np.ndarray:
        """Load selected profiles projected onto the configured depth axis."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.zeros((0, int(self.depth_axis_m.size)), dtype=np.float32)
        out = np.full(
            (int(indices.size), int(self.depth_axis_m.size)), np.nan, dtype=np.float32
        )
        selected_files = self.file_index[indices]
        selected_profiles = self.profile_index[indices]
        for file_idx in np.unique(selected_files):
            local_positions = np.flatnonzero(selected_files == int(file_idx))
            profile_rows = selected_profiles[local_positions].astype(
                np.int64, copy=False
            )
            ds = self._get_dataset(self.files[int(file_idx)])
            temp = self._read_profile_matrix(ds, self.temp_var_name, profile_rows)
            depth = self._read_profile_matrix(ds, self.depth_var_name, profile_rows)
            for local_idx, output_idx in enumerate(local_positions.tolist()):
                # Raw profile observations are projected onto the model target axis.
                out[int(output_idx)] = _align_argo_profile_to_glorys_depths(
                    temperature=temp[int(local_idx)],
                    depth=depth[int(local_idx)],
                    glorys_depths=self.depth_axis_m,
                )
        return out
