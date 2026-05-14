from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import xarray as xr
import yaml

from depth_recon.data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_DEPTH_VAR,
    TimedFile,
    date_to_days_since_1950,
    open_argo_dataset,
    scan_timed_files,
)
from depth_recon.data.dataset_grid_utils import (
    DEFAULT_LAND_MASK_PATH,
    MISSING_TEXT_VALUES,
    _GridParams,
    _build_land_mask_patch_table,
    _build_patch_lookup,
    _center_lon_deg,
    _deep_update_config,
    _force_include_cache_hash,
    _grid_starts,
    _normalize_lon,
    _parse_date_int,
    _parse_force_include_regions,
    _patch_ids_for_profile,
    _path_cache_hash,
    _sanitize_cache_text,
    _validate_grid_params,
)
from depth_recon.paths import config_path, resolve_config_path
from depth_recon.utils.normalizations import temperature_normalize

GLORYS_RELATIVE_DEPTH_CUTOFF = 0.10
GLORYS_MIN_ABSOLUTE_DEPTH_CUTOFF_M = 10.0


def _datetime_from_yyyymmdd(value: int) -> datetime:
    return datetime.strptime(str(int(value)), "%Y%m%d")


def _date_range_yyyymmdd(center_date: int, radius_days: int) -> list[int]:
    center = _datetime_from_yyyymmdd(int(center_date))
    return [
        int((center + timedelta(days=offset)).strftime("%Y%m%d"))
        for offset in range(-int(radius_days), int(radius_days) + 1)
    ]


def _yyyymmdd_from_days_since_1950(day_value: float) -> int:
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


def _juld_to_yyyymmdd(juld_days: np.ndarray) -> np.ndarray:
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
    available = set(names)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _collapse_duplicate_profile_depths(
    depth: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_depths, inverse = np.unique(depth, return_inverse=True)
    if unique_depths.size == depth.size:
        return depth, temperature
    # Average repeated source depths so interpolation receives one strictly increasing axis.
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
    lat_axis: np.ndarray
    lon_axis: np.ndarray


class DatasetCache:
    """Small LRU cache for xarray NetCDF datasets opened by one worker process."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = int(max_open)
        self._items: OrderedDict[Path, xr.Dataset] = OrderedDict()

    def get(self, path: Path) -> xr.Dataset:
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
        for ds in self._items.values():
            ds.close()
        self._items.clear()


class TimedNetCDFStore:
    """Date-indexed NetCDF source folder with lazy per-file reads."""

    LAT_CANDIDATES = ("latitude", "lat")
    LON_CANDIDATES = ("longitude", "lon")

    def __init__(self, root_dir: str | Path, *, cache_size: int = 8) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"NetCDF root does not exist: {self.root_dir}")
        self.index = scan_timed_files(self.root_dir, show_progress=False)
        if not self.index:
            raise RuntimeError(f"No readable NetCDF files found in: {self.root_dir}")
        self.cache = DatasetCache(max_open=cache_size)

    @property
    def dates(self) -> list[int]:
        return [_yyyymmdd_from_days_since_1950(item.day) for item in self.index]

    def depth_axis_m(self, coord_name: str = "depth") -> np.ndarray:
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
        return _normalize_lon(float(value))

    def _discover_files(self) -> list[Path]:
        files = sorted(self.root_dir.glob("EN.4.2.2.f.profiles.g10.*.nc"))
        if files:
            return files
        return sorted(self.root_dir.glob("*.nc"))

    def _get_dataset(self, path: Path) -> xr.Dataset:
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
        out: dict[int, np.ndarray] = {}
        for date_value in np.unique(self.profile_date):
            if int(date_value) <= 0:
                continue
            out[int(date_value)] = np.flatnonzero(self.profile_date == int(date_value))
        return out

    @staticmethod
    def _replace_en4_fill_with_nan(values: np.ndarray) -> np.ndarray:
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
                # Project raw profile observations onto the GLORYS target depth axis at read time.
                out[int(output_idx)] = _align_argo_profile_to_glorys_depths(
                    temperature=temp[int(local_idx)],
                    depth=depth[int(local_idx)],
                    glorys_depths=self.depth_axis_m,
                )
        return out


class VirtualPatchIndex:
    """Builds compact patch/date metadata rows without precomputing tensors."""

    CACHE_VERSION = 3

    def __init__(
        self,
        *,
        ostia_store: TimedNetCDFStore,
        glorys_store: TimedNetCDFStore,
        argo_store: ArgoNetCDFStore,
        optional_time_stores: Sequence[TimedNetCDFStore] = (),
        cache_dir: str | Path | None,
        grid_params: _GridParams,
        temporal_window_days: int,
    ) -> None:
        self.ostia_store = ostia_store
        self.glorys_store = glorys_store
        self.argo_store = argo_store
        self.optional_time_stores = tuple(optional_time_stores)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.grid_params = grid_params
        self.temporal_window_days = int(temporal_window_days)
        _validate_grid_params(self.grid_params)

    def load_rows(self) -> list[dict[str, Any]]:
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            return pd.read_csv(cache_path).to_dict(orient="records")

        patch_df = self._build_patch_table()
        dates = self._overlapping_dates()
        support_counts = self._build_support_counts(patch_df, dates)
        rows: list[dict[str, Any]] = []
        export_index = 0
        for date_value in dates:
            for patch in patch_df.to_dict(orient="records"):
                patch_id = int(patch["patch_id"])
                row = dict(patch)
                row["date"] = int(date_value)
                row["export_index"] = int(export_index)
                if self.grid_params.val_year is not None:
                    phase = self._phase_for_date(int(date_value))
                    row["split"] = phase
                    row["phase"] = phase
                row["argo_profile_count"] = int(
                    support_counts.get((patch_id, int(date_value)), 0)
                )
                rows.append(row)
                export_index += 1

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(rows).to_csv(cache_path, index=False)
        return rows

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        res_text = str(float(self.grid_params.resolution_deg)).replace(".", "p")
        land_text = str(float(self.grid_params.max_land_fraction)).replace(".", "p")
        grid_source = _sanitize_cache_text(self.grid_params.patch_grid_source)
        mask_hash = _path_cache_hash(self.grid_params.land_mask_path)
        force_hash = _force_include_cache_hash(self.grid_params.force_include_regions)
        split_text = (
            f"valyear{int(self.grid_params.val_year)}"
            if self.grid_params.val_year is not None
            else "patchsplit"
        )
        name = (
            f"argo_netcdf_gridded_v{self.CACHE_VERSION}_"
            f"tile{int(self.grid_params.tile_size)}_res{res_text}_"
            f"stride{int(self.grid_params.effective_patch_stride)}_"
            f"grid{grid_source}_land{land_text}_mask{mask_hash}_"
            f"force{force_hash}_"
            f"days{int(self.temporal_window_days)}_{split_text}.csv"
        )
        return self.cache_dir / name

    def _phase_for_date(self, date_value: int) -> str:
        year = int(date_value) // 10000
        return "val" if year == int(self.grid_params.val_year) else "train"

    def _overlapping_dates(self) -> list[int]:
        candidate_dates = sorted(set(int(v) for v in self.ostia_store.dates))
        source_stores = (self.glorys_store, *self.optional_time_stores)
        for store in source_stores:
            days = np.asarray([item.day for item in store.index], dtype=np.float64)
            if days.size == 0:
                continue
            day_lo = float(np.nanmin(days))
            day_hi = float(np.nanmax(days))
            candidate_dates = [
                date_value
                for date_value in candidate_dates
                if day_lo <= date_to_days_since_1950(int(date_value)) <= day_hi
            ]
        if not candidate_dates:
            raise RuntimeError(
                "No overlapping source dates found for OSTIA/GLORYS NetCDF stores."
            )
        return candidate_dates

    def _build_patch_table(self) -> pd.DataFrame:
        if str(self.grid_params.patch_grid_source).strip().lower() == "land_mask":
            patch_df = _build_land_mask_patch_table(self.grid_params)
            phases = self._split_phases(len(patch_df))
            records = patch_df.to_dict(orient="records")
            for rec, phase in zip(records, phases, strict=False):
                rec["split"] = phase
                rec["phase"] = phase
            return pd.DataFrame.from_records(records)

        path = self.ostia_store.index[0].path
        with xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        ) as ds:
            lat_name = _first_present_name(ds.coords, TimedNetCDFStore.LAT_CANDIDATES)
            lon_name = _first_present_name(ds.coords, TimedNetCDFStore.LON_CANDIDATES)
            if lat_name is None or lon_name is None:
                raise RuntimeError(f"OSTIA file is missing lat/lon coords: {path}")
            lat_values = np.asarray(ds[lat_name].values, dtype=np.float64)
            lon_values = np.asarray(ds[lon_name].values, dtype=np.float64)
            invalid_mask = self._load_ostia_invalid_mask(ds)

        tile = int(self.grid_params.tile_size)
        if tile < 1:
            raise ValueError("grid.tile_size must be >= 1.")
        if lat_values.size < tile or lon_values.size < tile:
            raise RuntimeError("OSTIA grid is smaller than the requested tile size.")

        records: list[dict[str, Any]] = []
        patch_id = 0
        stride = int(self.grid_params.effective_patch_stride)
        for y0 in _grid_starts(int(lat_values.size), tile, stride):
            for x0 in _grid_starts(int(lon_values.size), tile, stride):
                invalid_fraction = self._invalid_fraction(
                    invalid_mask,
                    y0=y0,
                    x0=x0,
                    tile=tile,
                )
                if invalid_fraction > float(self.grid_params.invalid_threshold):
                    continue
                lat_slice = lat_values[y0 : y0 + tile]
                lon_slice = lon_values[x0 : x0 + tile]
                lat0 = float(
                    np.nanmin(lat_slice) - 0.5 * self.grid_params.resolution_deg
                )
                lat1 = float(
                    np.nanmax(lat_slice) + 0.5 * self.grid_params.resolution_deg
                )
                lon0 = float(
                    np.nanmin(lon_slice) - 0.5 * self.grid_params.resolution_deg
                )
                lon1 = float(
                    np.nanmax(lon_slice) + 0.5 * self.grid_params.resolution_deg
                )
                records.append(
                    {
                        "patch_id": int(patch_id),
                        "grid_y0": int(y0),
                        "grid_x0": int(x0),
                        "lat0": lat0,
                        "lat1": lat1,
                        "lon0": lon0,
                        "lon1": lon1,
                        "lat_center": 0.5 * (lat0 + lat1),
                        "lon_center": _center_lon_deg(lon0, lon1),
                        "land_fraction": float(invalid_fraction),
                        "ocean_fraction": float(1.0 - invalid_fraction),
                        "invalid_fraction": float(invalid_fraction),
                    }
                )
                patch_id += 1

        if not records:
            raise RuntimeError("No valid patches were built from the OSTIA grid.")
        phases = self._split_phases(len(records))
        for rec, phase in zip(records, phases, strict=False):
            rec["split"] = phase
            rec["phase"] = phase
        return pd.DataFrame.from_records(records)

    def _load_ostia_invalid_mask(self, ds: xr.Dataset) -> np.ndarray | None:
        if "mask" not in ds:
            return None
        mask = ds["mask"]
        if "time" in mask.dims:
            mask = mask.isel(time=0)
        values = np.asarray(mask.values)
        raw_meanings = str(mask.attrs.get("flag_meanings", "")).strip()
        raw_masks = np.asarray(
            mask.attrs.get("flag_masks", ()), dtype=np.int64
        ).reshape(-1)
        if raw_meanings != "" and raw_masks.size > 0:
            flag_names = [part.strip() for part in raw_meanings.split() if part.strip()]
            if len(flag_names) == int(raw_masks.size):
                selected_masks = [
                    int(flag_mask)
                    for flag_name, flag_mask in zip(flag_names, raw_masks.tolist())
                    if flag_name in set(self.grid_params.invalid_mask_flags)
                ]
                if selected_masks:
                    invalid = np.zeros(values.shape, dtype=bool)
                    for flag_mask in selected_masks:
                        invalid |= (
                            values.astype(np.int64, copy=False) & flag_mask
                        ) != 0
                    return invalid
        return values != 0

    def _invalid_fraction(
        self,
        invalid_mask: np.ndarray | None,
        *,
        y0: int,
        x0: int,
        tile: int,
    ) -> float:
        if invalid_mask is None:
            return 0.0
        patch = np.asarray(invalid_mask[y0 : y0 + tile, x0 : x0 + tile], dtype=bool)
        if patch.size == 0:
            return 1.0
        return float(np.count_nonzero(patch) / float(patch.size))

    def _split_phases(self, n_patches: int) -> list[str]:
        phases = np.full((int(n_patches),), "train", dtype=object)
        val_len = int(round(int(n_patches) * float(self.grid_params.val_fraction)))
        if n_patches > 1:
            val_len = min(
                max(val_len, 1 if self.grid_params.val_fraction > 0.0 else 0),
                int(n_patches) - 1,
            )
        else:
            val_len = 0
        if val_len > 0:
            rng = np.random.default_rng(int(self.grid_params.split_seed))
            val_indices = rng.permutation(np.arange(int(n_patches)))[:val_len]
            phases[val_indices] = "val"
        return [str(value) for value in phases.tolist()]

    def _build_support_counts(
        self,
        patch_df: pd.DataFrame,
        dates: list[int],
    ) -> dict[tuple[int, int], int]:
        support_counts: dict[tuple[int, int], int] = {}
        if patch_df.empty or not dates:
            return support_counts

        date_set = set(int(v) for v in dates)
        # The support cache is built profile-first so each ARGO profile contributes
        # to at most temporal_window_days target dates instead of checking every
        # patch/date row independently.
        patch_lookup = _build_patch_lookup(patch_df, self.grid_params)
        for profile_idx in tqdm(
            range(int(self.argo_store.profile_date.size)),
            desc="Counting ARGO overlap support",
            unit="profile",
            dynamic_ncols=True,
        ):
            if not bool(self.argo_store._has_valid_temp[profile_idx]):
                continue
            profile_date = int(self.argo_store.profile_date[profile_idx])
            patch_ids = _patch_ids_for_profile(
                patch_lookup,
                lat=float(self.argo_store.latitude[profile_idx]),
                lon=float(self.argo_store.longitude[profile_idx]),
            )
            if not patch_ids:
                continue
            for date_value in _date_range_yyyymmdd(
                profile_date,
                int(self.temporal_window_days) // 2,
            ):
                if int(date_value) not in date_set:
                    continue
                for patch_id in patch_ids:
                    key = (int(patch_id), int(date_value))
                    support_counts[key] = support_counts.get(key, 0) + 1
        return support_counts

    @staticmethod
    def _patch_id_for_profile(
        patch_lookup: dict[int, dict[str, Any]],
        *,
        lat: float,
        lon: float,
    ) -> int | None:
        if not np.isfinite(lat) or not np.isfinite(lon):
            return None
        lon_norm = _normalize_lon(float(lon))
        for patch_id, patch in patch_lookup.items():
            lat_lo = min(float(patch["lat0"]), float(patch["lat1"]))
            lat_hi = max(float(patch["lat0"]), float(patch["lat1"]))
            lon_lo = _normalize_lon(min(float(patch["lon0"]), float(patch["lon1"])))
            lon_hi = _normalize_lon(max(float(patch["lon0"]), float(patch["lon1"])))
            if not (lat_lo <= float(lat) < lat_hi):
                continue
            if lon_lo <= lon_hi:
                in_lon = lon_lo <= lon_norm < lon_hi
            else:
                in_lon = lon_norm >= lon_lo or lon_norm < lon_hi
            if in_lon:
                return int(patch_id)
        return None


class ArgoNetCDFGriddedPatchDataset(Dataset):
    """Dataset that lazily builds OSTIA/ARGO/GLORYS patches from raw NetCDF."""

    DEFAULT_CONFIG_PATH = str(config_path("px_space", "data_ostia_argo_netcdf.yaml"))
    DEFAULT_ARGO_DIR = "/data1/datasets/depth_v2/en4_profiles"
    DEFAULT_GLORYS_DIR = "/data1/datasets/depth_v2/glorys_weekly"
    DEFAULT_OSTIA_DIR = "/data1/datasets/depth_v2/ostia"
    DEFAULT_SEALEVEL_DIR = "/data1/datasets/depth_v2/sealevel_daily"
    DEFAULT_METADATA_CACHE_DIR = "/data1/datasets/depth_v2/depthdif_cache"

    def __init__(
        self,
        *,
        argo_dir: str | Path = DEFAULT_ARGO_DIR,
        glorys_dir: str | Path = DEFAULT_GLORYS_DIR,
        ostia_dir: str | Path = DEFAULT_OSTIA_DIR,
        sealevel_dir: str | Path | None = DEFAULT_SEALEVEL_DIR,
        metadata_cache_dir: str | Path | None = DEFAULT_METADATA_CACHE_DIR,
        split: str = "all",
        tile_size: int = 128,
        resolution_deg: float = 0.1,
        patch_grid_source: str = "land_mask",
        land_mask_path: str | Path | None = DEFAULT_LAND_MASK_PATH,
        patch_stride: int | None = None,
        max_land_fraction: float = 0.30,
        force_include_regions: Sequence[dict[str, Any]] | None = None,
        temporal_window_days: int = 7,
        glorys_var_name: str = "thetao",
        ostia_var_name: str = "analysed_sst",
        argo_temp_var_name: str = "TEMP",
        argo_depth_var_name: str = ARGO_DEPTH_VAR,
        invalid_threshold: float = 0.5,
        invalid_mask_flags: Sequence[str] = ("land",),
        val_fraction: float = 0.2,
        val_year: int | None = None,
        require_argo_for_train: bool = True,
        require_argo_for_val: bool = True,
        require_argo_for_all: bool = False,
        synthetic_mode: bool = False,
        synthetic_pixel_count: int = 250,
        return_info: bool = True,
        return_coords: bool = True,
        random_seed: int = 7,
        cache_size: int = 8,
    ) -> None:
        self.split = str(split).strip().lower()
        if self.split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: 'all', 'train', 'val'")
        self.tile_size = int(tile_size)
        self.resolution_deg = float(resolution_deg)
        self.patch_grid_source = str(patch_grid_source)
        self.land_mask_path = None if land_mask_path is None else Path(land_mask_path)
        self.patch_stride = None if patch_stride is None else int(patch_stride)
        self.max_land_fraction = float(max_land_fraction)
        self.force_include_regions = _parse_force_include_regions(force_include_regions)
        self.temporal_window_days = int(temporal_window_days)
        self.glorys_var_name = str(glorys_var_name)
        self.ostia_var_name = str(ostia_var_name)
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
        self.random_seed = int(random_seed)
        self.require_argo_for_train = bool(require_argo_for_train)
        self.require_argo_for_val = bool(require_argo_for_val)
        self.require_argo_for_all = bool(require_argo_for_all)
        self.synthetic_mode = bool(synthetic_mode)
        self.synthetic_pixel_count = int(synthetic_pixel_count)
        if self.temporal_window_days < 1:
            raise ValueError("sampling.temporal_window_days must be >= 1.")
        if self.synthetic_pixel_count < 0:
            raise ValueError("synthetic.pixel_count must be >= 0.")

        self.glorys_store = TimedNetCDFStore(glorys_dir, cache_size=cache_size)
        self.ostia_store = TimedNetCDFStore(ostia_dir, cache_size=cache_size)
        self._depth_axis_m = self.glorys_store.depth_axis_m()
        self.argo_store = ArgoNetCDFStore(
            argo_dir,
            depth_axis_m=self._depth_axis_m,
            temp_var_name=argo_temp_var_name,
            depth_var_name=argo_depth_var_name,
            cache_size=cache_size,
        )
        self.sealevel_store = (
            None
            if sealevel_dir is None
            or str(sealevel_dir).strip().lower() in MISSING_TEXT_VALUES
            else TimedNetCDFStore(sealevel_dir, cache_size=cache_size)
        )

        grid_params = _GridParams(
            tile_size=self.tile_size,
            resolution_deg=self.resolution_deg,
            invalid_threshold=float(invalid_threshold),
            invalid_mask_flags=tuple(str(v) for v in invalid_mask_flags),
            val_fraction=float(val_fraction),
            val_year=None if val_year is None else int(val_year),
            split_seed=self.random_seed,
            patch_grid_source=self.patch_grid_source,
            land_mask_path=self.land_mask_path,
            patch_stride=self.patch_stride,
            max_land_fraction=self.max_land_fraction,
            force_include_regions=self.force_include_regions,
        )
        index = VirtualPatchIndex(
            ostia_store=self.ostia_store,
            glorys_store=self.glorys_store,
            argo_store=self.argo_store,
            optional_time_stores=(
                () if self.sealevel_store is None else (self.sealevel_store,)
            ),
            cache_dir=metadata_cache_dir,
            grid_params=grid_params,
            temporal_window_days=self.temporal_window_days,
        )
        rows = index.load_rows()
        rows = self._filter_rows(rows)
        if not rows:
            raise RuntimeError("Dataset is empty after split/ARGO filtering.")
        self._rows = rows

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def depth_axis_m(self) -> np.ndarray:
        return self._depth_axis_m.copy()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        split: str = "all",
        dataset_overrides: dict[str, Any] | None = None,
    ) -> "ArgoNetCDFGriddedPatchDataset":
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH
        with resolve_config_path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg.get("dataset", {})
        if dataset_overrides:
            ds_cfg = _deep_update_config(ds_cfg, dataset_overrides)
        return cls(
            argo_dir=cls._cfg_get(
                ds_cfg,
                "core.argo_dir",
                "argo_dir",
                default=cls.DEFAULT_ARGO_DIR,
            ),
            glorys_dir=cls._cfg_get(
                ds_cfg,
                "core.glorys_dir",
                "glorys_dir",
                default=cls.DEFAULT_GLORYS_DIR,
            ),
            ostia_dir=cls._cfg_get(
                ds_cfg,
                "core.ostia_dir",
                "ostia_dir",
                default=cls.DEFAULT_OSTIA_DIR,
            ),
            sealevel_dir=cls._cfg_get(
                ds_cfg,
                "core.sealevel_dir",
                "sealevel_dir",
                default=cls.DEFAULT_SEALEVEL_DIR,
            ),
            metadata_cache_dir=cls._cfg_get(
                ds_cfg,
                "core.metadata_cache_dir",
                "metadata_cache_dir",
                default=cls.DEFAULT_METADATA_CACHE_DIR,
            ),
            split=split,
            tile_size=int(
                cls._cfg_get(ds_cfg, "grid.tile_size", "tile_size", default=128)
            ),
            resolution_deg=float(
                cls._cfg_get(
                    ds_cfg, "grid.resolution_deg", "resolution_deg", default=0.1
                )
            ),
            patch_grid_source=str(
                cls._cfg_get(
                    ds_cfg,
                    "grid.patch_grid_source",
                    "patch_grid_source",
                    default="land_mask",
                )
            ),
            land_mask_path=cls._cfg_get(
                ds_cfg,
                "grid.land_mask_path",
                "land_mask_path",
                default=DEFAULT_LAND_MASK_PATH,
            ),
            patch_stride=cls._optional_int(
                cls._cfg_get(
                    ds_cfg,
                    "grid.patch_stride",
                    "patch_stride",
                    default=None,
                )
            ),
            max_land_fraction=float(
                cls._cfg_get(
                    ds_cfg,
                    "grid.max_land_fraction",
                    "max_land_fraction",
                    default=0.30,
                )
            ),
            force_include_regions=cls._cfg_get(
                ds_cfg,
                "grid.force_include_regions",
                "force_include_regions",
                default=None,
            ),
            temporal_window_days=int(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.temporal_window_days",
                    "temporal_window_days",
                    default=7,
                )
            ),
            glorys_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.glorys_var_name",
                    "glorys_var_name",
                    default="thetao",
                )
            ),
            ostia_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.ostia_var_name",
                    "ostia_var_name",
                    default="analysed_sst",
                )
            ),
            argo_temp_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.argo_temp_var_name",
                    "argo_temp_var_name",
                    default="TEMP",
                )
            ),
            argo_depth_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.argo_depth_var_name",
                    "argo_depth_var_name",
                    default=ARGO_DEPTH_VAR,
                )
            ),
            invalid_threshold=float(
                cls._cfg_get(
                    ds_cfg,
                    "grid.invalid_threshold",
                    "invalid_threshold",
                    default=0.5,
                )
            ),
            invalid_mask_flags=tuple(
                cls._cfg_get(
                    ds_cfg,
                    "grid.invalid_mask_flags",
                    "invalid_mask_flags",
                    default=("land",),
                )
            ),
            val_fraction=float(cfg.get("split", {}).get("val_fraction", 0.2)),
            val_year=cls._optional_int(cfg.get("split", {}).get("val_year", None)),
            require_argo_for_train=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_train",
                    "require_argo_for_train",
                    default=True,
                )
            ),
            require_argo_for_val=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_val",
                    "require_argo_for_val",
                    default=True,
                )
            ),
            require_argo_for_all=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_all",
                    "require_argo_for_all",
                    default=False,
                )
            ),
            synthetic_mode=bool(
                cls._cfg_get(
                    ds_cfg, "synthetic.enabled", "synthetic_enabled", default=False
                )
            ),
            synthetic_pixel_count=int(
                cls._cfg_get(
                    ds_cfg,
                    "synthetic.pixel_count",
                    "synthetic_pixel_count",
                    default=250,
                )
            ),
            return_info=bool(
                cls._cfg_get(ds_cfg, "output.return_info", "return_info", default=True)
            ),
            return_coords=bool(
                cls._cfg_get(
                    ds_cfg, "output.return_coords", "return_coords", default=True
                )
            ),
            random_seed=int(
                cls._cfg_get(ds_cfg, "runtime.random_seed", "random_seed", default=7)
            ),
            cache_size=int(
                cls._cfg_get(ds_cfg, "runtime.cache_size", "cache_size", default=8)
            ),
        )

    @staticmethod
    def _cfg_get(
        cfg: dict[str, Any],
        nested_key: str,
        flat_key: str,
        *,
        default: Any,
    ) -> Any:
        node: Any = cfg
        for part in nested_key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
        _ = flat_key
        return default

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in MISSING_TEXT_VALUES:
            return None
        return int(value)

    def _filter_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.split in {"train", "val"}:
            rows = [
                row
                for row in rows
                if str(row.get("split", row.get("phase", ""))).strip().lower()
                == self.split
            ]
        require_argo = self._require_argo_for_current_split()
        if require_argo:
            rows = [row for row in rows if int(row.get("argo_profile_count", 0)) > 0]
        return rows

    def _require_argo_for_current_split(self) -> bool:
        if self.synthetic_mode:
            return False
        if self.split == "train":
            return self.require_argo_for_train
        if self.split == "val":
            return self.require_argo_for_val
        return self.require_argo_for_all

    def __len__(self) -> int:
        return len(self._rows)

    def _patch_axes(self, row: dict[str, Any]) -> PatchAxes:
        top = max(float(row["lat0"]), float(row["lat1"]))
        left = min(float(row["lon0"]), float(row["lon1"]))
        half = 0.5 * float(self.resolution_deg)
        lat_axis = (
            top
            - half
            - (np.arange(self.tile_size, dtype=np.float64) * self.resolution_deg)
        )
        lon_axis = (
            left
            + half
            + (np.arange(self.tile_size, dtype=np.float64) * self.resolution_deg)
        )
        return PatchAxes(lat_axis=lat_axis, lon_axis=lon_axis)

    def _load_y_patch(self, row: dict[str, Any], axes: PatchAxes) -> np.ndarray:
        y_np = self.glorys_store.read_patch(
            target_date=int(row["date"]),
            var_name=self.glorys_var_name,
            axes=axes,
            categorical=False,
        )
        if y_np.ndim != 3:
            raise RuntimeError(
                f"Expected GLORYS patch shape (D,H,W), got {tuple(y_np.shape)}"
            )
        if int(y_np.shape[0]) != int(self._depth_axis_m.size):
            raise RuntimeError(
                "GLORYS depth channel count does not match the configured GLORYS depth axis: "
                f"{int(y_np.shape[0])} != {int(self._depth_axis_m.size)}"
            )
        return y_np.astype(np.float32, copy=False)

    def _load_eo_patch(self, row: dict[str, Any], axes: PatchAxes) -> np.ndarray:
        eo_np = self.ostia_store.read_patch(
            target_date=int(row["date"]),
            var_name=self.ostia_var_name,
            axes=axes,
            categorical=False,
        )
        if eo_np.ndim != 2:
            raise RuntimeError(
                f"Expected OSTIA patch shape (H,W), got {tuple(eo_np.shape)}"
            )
        eo_np = eo_np.astype(np.float32, copy=False)
        if self._ostia_values_look_kelvin(eo_np):
            eo_np = eo_np - np.float32(273.15)
        return eo_np[None, ...]

    @staticmethod
    def _ostia_values_look_kelvin(values: np.ndarray) -> bool:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            return False
        return float(np.nanmedian(finite)) > 100.0

    def _rasterize_argo_patch(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        depth_size = int(self.argo_store.depth_axis_m.size)
        x_sum = np.zeros((depth_size, self.tile_size, self.tile_size), dtype=np.float64)
        x_count = np.zeros(
            (depth_size, self.tile_size, self.tile_size), dtype=np.uint16
        )
        indices = self.argo_store.query_indices(
            target_date=int(row["date"]),
            temporal_window_days=self.temporal_window_days,
            lat0=float(row["lat0"]),
            lat1=float(row["lat1"]),
            lon0=float(row["lon0"]),
            lon1=float(row["lon1"]),
        )
        if indices.size == 0:
            return (
                np.full(x_sum.shape, np.nan, dtype=np.float32),
                np.zeros(x_sum.shape, dtype=bool),
            )

        values = self.argo_store.load_temperature_profiles(indices)
        top = max(float(row["lat0"]), float(row["lat1"]))
        left = min(float(row["lon0"]), float(row["lon1"]))
        for local_idx, profile_idx in enumerate(indices.tolist()):
            lat = float(self.argo_store.latitude[int(profile_idx)])
            lon = _normalize_lon(float(self.argo_store.longitude[int(profile_idx)]))
            row_idx = int(np.floor((top - lat) / float(self.resolution_deg)))
            col_idx = int(
                np.floor((_normalize_lon(lon) - left) / float(self.resolution_deg))
            )
            if (
                row_idx < 0
                or row_idx >= self.tile_size
                or col_idx < 0
                or col_idx >= self.tile_size
            ):
                continue
            profile = values[int(local_idx)]
            valid = np.isfinite(profile)
            if not np.any(valid):
                continue
            x_sum[valid, row_idx, col_idx] += profile[valid].astype(np.float64)
            x_count[valid, row_idx, col_idx] += 1

        x_np = np.full(x_sum.shape, np.nan, dtype=np.float32)
        x_valid = x_count > 0
        x_np[x_valid] = (x_sum[x_valid] / x_count[x_valid].astype(np.float64)).astype(
            np.float32,
            copy=False,
        )
        return x_np, x_valid

    def _synthetic_rng_for_row(
        self,
        row: dict[str, Any],
        *,
        idx: int,
    ) -> np.random.Generator:
        seed = np.random.SeedSequence(
            [
                int(self.random_seed),
                int(row.get("patch_id", 0)),
                int(row.get("date", 0)),
                int(idx),
            ]
        )
        return np.random.default_rng(seed)

    def _build_synthetic_x_from_glorys(
        self,
        y_np: np.ndarray,
        y_valid_mask_np: np.ndarray,
        row: dict[str, Any],
        *,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_np = np.full(y_np.shape, np.nan, dtype=np.float32)
        x_valid = np.zeros(y_valid_mask_np.shape, dtype=bool)
        if self.synthetic_pixel_count == 0:
            return x_np, x_valid

        valid_columns = np.asarray(y_valid_mask_np, dtype=bool).any(axis=0)
        flat_valid_columns = np.flatnonzero(valid_columns.reshape(-1))
        if flat_valid_columns.size == 0:
            return x_np, x_valid

        sample_count = min(
            int(self.synthetic_pixel_count), int(flat_valid_columns.size)
        )
        rng = self._synthetic_rng_for_row(row, idx=idx)
        selected = rng.choice(flat_valid_columns, size=sample_count, replace=False)
        row_indices, col_indices = np.unravel_index(
            selected,
            valid_columns.shape,
        )
        for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
            depth_valid = y_valid_mask_np[:, int(row_idx), int(col_idx)]
            if not np.any(depth_valid):
                continue
            # Synthetic mode uses GLORYS itself as sparse input at sampled columns.
            x_np[depth_valid, int(row_idx), int(col_idx)] = y_np[
                depth_valid,
                int(row_idx),
                int(col_idx),
            ]
            x_valid[depth_valid, int(row_idx), int(col_idx)] = True
        return x_np, x_valid

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        axes = self._patch_axes(row)
        eo_np = self._load_eo_patch(row, axes)
        y_np = self._load_y_patch(row, axes)
        y_valid_mask_np = np.isfinite(y_np)
        if self.synthetic_mode:
            x_np, x_valid_mask_np = self._build_synthetic_x_from_glorys(
                y_np,
                y_valid_mask_np,
                row,
                idx=int(idx),
            )
        else:
            x_np, x_valid_mask_np = self._rasterize_argo_patch(row)

        land_mask_np = y_valid_mask_np[:1].astype(np.float32, copy=False)
        eo = temperature_normalize(mode="norm", tensor=torch.from_numpy(eo_np))
        x = temperature_normalize(mode="norm", tensor=torch.from_numpy(x_np))
        y = temperature_normalize(mode="norm", tensor=torch.from_numpy(y_np))
        eo = torch.nan_to_num(eo, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x_valid_mask = torch.from_numpy(x_valid_mask_np.astype(np.bool_, copy=False))
        y_valid_mask = torch.from_numpy(y_valid_mask_np.astype(np.bool_, copy=False))
        land_mask = torch.from_numpy(land_mask_np)
        x_valid_mask_1d = x_valid_mask.any(dim=0, keepdim=True)

        sample: dict[str, Any] = {
            "eo": eo,
            "x": x,
            "y": y,
            "x_valid_mask": x_valid_mask,
            "y_valid_mask": y_valid_mask,
            "x_valid_mask_1d": x_valid_mask_1d,
            "land_mask": land_mask,
            "date": _parse_date_int(row.get("date", 19700115)),
        }
        if self.return_coords:
            sample["coords"] = torch.tensor(
                [
                    0.5 * (float(row["lat0"]) + float(row["lat1"])),
                    _center_lon_deg(float(row["lon0"]), float(row["lon1"])),
                ],
                dtype=torch.float32,
            )
        if self.return_info:
            info = dict(row)
            info["x_source"] = "glorys_synthetic" if self.synthetic_mode else "argo"
            info["synthetic_pixel_count"] = (
                int(self.synthetic_pixel_count) if self.synthetic_mode else 0
            )
            sample["info"] = info
        return sample
