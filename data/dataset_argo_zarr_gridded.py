from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr
import yaml

from data.dataset_argo_netcdf_gridded import (
    DEFAULT_LAND_MASK_PATH,
    MISSING_TEXT_VALUES,
    PatchAxes,
    _GridParams,
    _align_argo_profile_to_glorys_depths,
    _build_land_mask_patch_table,
    _build_patch_lookup,
    _center_lon_deg,
    _date_range_yyyymmdd,
    _first_present_name,
    _force_include_cache_hash,
    _grid_starts,
    _juld_to_yyyymmdd,
    _normalize_lon,
    _patch_ids_for_profile,
    _path_cache_hash,
    _parse_force_include_regions,
    _parse_date_int,
    _sanitize_cache_text,
    _validate_grid_params,
)
from data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_DEPTH_VAR,
    TimedFile,
    date_to_days_since_1950,
)
from utils.normalizations import temperature_normalize


def _date_int_from_days_since_1950(day_value: float) -> int:
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


def _date_ints_from_time_coord(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if raw.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if np.issubdtype(raw.dtype, np.datetime64):
        return np.char.replace(np.datetime_as_string(raw, unit="D"), "-", "").astype(
            np.int32
        )

    numeric = raw.astype(np.float64, copy=False)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.zeros(raw.shape, dtype=np.int32)
    if float(np.nanmedian(finite)) > 10_000_000.0:
        return numeric.astype(np.int32, copy=False)

    out = np.zeros(raw.shape, dtype=np.int32)
    valid = np.isfinite(numeric)
    out[valid] = np.asarray(
        [_date_int_from_days_since_1950(day) for day in numeric[valid]],
        dtype=np.int32,
    )
    return out


def _first_available_var(
    available: Sequence[str],
    preferred: str | None,
    aliases: Sequence[str],
    *,
    required: bool,
    role: str,
) -> str | None:
    names = set(str(name) for name in available)
    candidates = tuple([preferred] if preferred is not None else []) + tuple(aliases)
    for candidate in candidates:
        if candidate is not None and str(candidate) in names:
            return str(candidate)
    if required:
        raise RuntimeError(f"No zarr variable found for required modality: {role}")
    return None


def _tensor_with_mask(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    mask = np.isfinite(values)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor, torch.from_numpy(mask.astype(np.bool_, copy=False))


@dataclass(frozen=True)
class _ZarrPaths:
    argo: Path
    glorys: Path
    ostia: Path
    sealevel: Path | None


class TimedZarrStore:
    """Date-indexed zarr source with lazy xarray reads."""

    LAT_CANDIDATES = ("latitude", "lat")
    LON_CANDIDATES = ("longitude", "lon")

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Zarr source does not exist: {self.root_dir}")
        self.ds = xr.open_zarr(
            self.root_dir,
            consolidated=None,
            decode_times=False,
            mask_and_scale=True,
        )
        self._dates = self._load_dates()
        self.index = [
            TimedFile(path=self.root_dir, day=date_to_days_since_1950(int(date_value)))
            for date_value in self._dates
        ]

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.ds.data_vars)

    @property
    def dates(self) -> list[int]:
        return [int(value) for value in self._dates.tolist()]

    def _load_dates(self) -> np.ndarray:
        if "time" not in self.ds.coords:
            raise RuntimeError(
                f"Timed zarr source is missing a time coordinate: {self.root_dir}"
            )
        dates = _date_ints_from_time_coord(np.asarray(self.ds["time"].values))
        if dates.size == 0:
            raise RuntimeError(
                f"Timed zarr source has an empty time coordinate: {self.root_dir}"
            )
        return dates.astype(np.int32, copy=False)

    def depth_axis_m(self, coord_name: str = "depth") -> np.ndarray:
        if coord_name in self.ds.coords or coord_name in self.ds.variables:
            depth = np.asarray(self.ds[coord_name].values, dtype=np.float32).reshape(-1)
            depth = depth[np.isfinite(depth)]
            if depth.size > 0:
                return depth.astype(np.float32, copy=False)
        raise RuntimeError(
            f"No readable {coord_name!r} depth axis found in {self.root_dir}"
        )

    def bracket(self, target_date: int) -> tuple[int, int, float]:
        target_day = date_to_days_since_1950(int(target_date))
        days = np.asarray([item.day for item in self.index], dtype=np.float64)
        pos = int(np.searchsorted(days, float(target_day), side="left"))
        if pos < days.size and np.isclose(days[pos], target_day, rtol=0.0, atol=1.0e-8):
            return pos, pos, 0.0
        if pos == 0:
            return 0, 0, 0.0
        if pos >= days.size:
            last = int(days.size - 1)
            return last, last, 0.0
        before = pos - 1
        after = pos
        span = days[after] - days[before]
        weight = (
            0.0 if span <= 0.0 else float((float(target_day) - days[before]) / span)
        )
        return before, after, weight

    def horizontal_grid(
        self, var_name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        da = self.ds[var_name] if var_name is not None and var_name in self.ds else None
        names = da.dims if da is not None else self.ds.coords
        lat_name = _first_present_name(names, self.LAT_CANDIDATES)
        lon_name = _first_present_name(names, self.LON_CANDIDATES)
        if lat_name is None or lon_name is None:
            raise RuntimeError(
                f"Zarr source is missing horizontal coordinates: {self.root_dir}"
            )
        return (
            np.asarray(self.ds[lat_name].values, dtype=np.float64),
            np.asarray(self.ds[lon_name].values, dtype=np.float64),
        )

    def read_patch(
        self,
        *,
        target_date: int,
        var_name: str,
        axes: PatchAxes,
        categorical: bool = False,
    ) -> np.ndarray:
        before, after, weight = self.bracket(int(target_date))
        if categorical or before == after:
            selected = before if weight <= 0.5 else after
            return self._read_one_patch(
                selected,
                var_name=var_name,
                axes=axes,
                categorical=categorical,
            )

        first = self._read_one_patch(
            before,
            var_name=var_name,
            axes=axes,
            categorical=False,
        )
        second = self._read_one_patch(
            after,
            var_name=var_name,
            axes=axes,
            categorical=False,
        )
        return (first + ((second - first) * np.float32(weight))).astype(
            np.float32,
            copy=False,
        )

    def _read_one_patch(
        self,
        time_index: int,
        *,
        var_name: str,
        axes: PatchAxes,
        categorical: bool,
    ) -> np.ndarray:
        if var_name not in self.ds:
            raise RuntimeError(
                f"Variable {var_name!r} is missing from zarr: {self.root_dir}"
            )
        da = self.ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=int(time_index))
        lat_name = _first_present_name(da.dims, self.LAT_CANDIDATES)
        lon_name = _first_present_name(da.dims, self.LON_CANDIDATES)
        if lat_name is None or lon_name is None:
            raise RuntimeError(
                f"Variable {var_name!r} in {self.root_dir} does not have lat/lon dimensions."
            )

        lon_axis = self._lon_axis_for_source(da, lon_name, axes.lon_axis)
        if self._axes_are_on_source_grid(
            da,
            lat_name,
            lon_name,
            axes.lat_axis,
            lon_axis,
        ):
            sampled = da.sel(
                {lat_name: axes.lat_axis, lon_name: lon_axis},
                method="nearest",
                tolerance=self._coordinate_tolerance(da, lat_name, lon_name),
            )
        else:
            sampled = da.interp(
                {lat_name: axes.lat_axis, lon_name: lon_axis},
                method="nearest" if categorical else "linear",
            )
        if "depth" in sampled.dims:
            sampled = sampled.transpose("depth", lat_name, lon_name)
        else:
            sampled = sampled.transpose(lat_name, lon_name)
        return np.asarray(sampled.values, dtype=np.float32)

    def invalid_mask(
        self,
        *,
        mask_var_name: str = "mask",
        invalid_mask_flags: Sequence[str] = ("land",),
    ) -> np.ndarray | None:
        if mask_var_name not in self.ds:
            return None
        mask = self.ds[mask_var_name]
        if "time" in mask.dims:
            mask = mask.isel(time=0)
        if "depth" in mask.dims:
            mask = mask.isel(depth=0)
        values = np.asarray(mask.values)
        raw_meanings = str(mask.attrs.get("flag_meanings", "")).strip()
        raw_masks = np.asarray(
            mask.attrs.get("flag_masks", ()), dtype=np.int64
        ).reshape(-1)
        if raw_meanings != "" and raw_masks.size > 0:
            flag_names = [part.strip() for part in raw_meanings.split() if part.strip()]
            if len(flag_names) == int(raw_masks.size):
                selected = [
                    int(flag_mask)
                    for flag_name, flag_mask in zip(flag_names, raw_masks.tolist())
                    if flag_name in set(str(v) for v in invalid_mask_flags)
                ]
                if selected:
                    invalid = np.zeros(values.shape, dtype=bool)
                    for flag_mask in selected:
                        invalid |= (
                            values.astype(np.int64, copy=False) & flag_mask
                        ) != 0
                    return invalid
        return values != 0

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

    @staticmethod
    def _coordinate_tolerance(
        da: xr.DataArray,
        lat_name: str,
        lon_name: str,
    ) -> float:
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

    @classmethod
    def _axes_are_on_source_grid(
        cls,
        da: xr.DataArray,
        lat_name: str,
        lon_name: str,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
    ) -> bool:
        tolerance = cls._coordinate_tolerance(da, lat_name, lon_name)
        return cls._axis_is_on_source_grid(
            da[lat_name].values,
            lat_axis,
            tolerance,
        ) and cls._axis_is_on_source_grid(
            da[lon_name].values,
            lon_axis,
            tolerance,
        )

    @staticmethod
    def _axis_is_on_source_grid(
        source_values: np.ndarray,
        requested_values: np.ndarray,
        tolerance: float,
    ) -> bool:
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
        # Exact-grid Zarr exports can use label selection instead of interpolation.
        return bool(np.all(nearest <= float(tolerance)))


class ArgoZarrStore:
    """Profile-indexed ARGO zarr access for sparse gridded training inputs."""

    REQUIRED_VARS = ("JULD", "LATITUDE", "LONGITUDE")

    def __init__(
        self,
        root_dir: str | Path,
        *,
        depth_axis_m: np.ndarray,
        temp_var_name: str = "TEMP",
        salinity_var_name: str | None = "PSAL_CORRECTED",
        depth_var_name: str = ARGO_DEPTH_VAR,
    ) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"ARGO zarr source does not exist: {self.root_dir}")
        self.ds = xr.open_zarr(
            self.root_dir,
            consolidated=None,
            decode_times=False,
            mask_and_scale=True,
        )
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        if self.depth_axis_m.size == 0:
            raise RuntimeError("ARGO zarr store needs a non-empty GLORYS depth axis.")
        self.temp_var_name = str(temp_var_name)
        self.salinity_var_name = (
            None if salinity_var_name is None else str(salinity_var_name)
        )
        self.depth_var_name = str(depth_var_name)
        self._build_profile_index()

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.ds.data_vars)

    def _build_profile_index(self) -> None:
        missing = [name for name in self.REQUIRED_VARS if name not in self.ds]
        missing += [name for name in (self.temp_var_name,) if name not in self.ds]
        if (
            self.temp_var_name in self.ds
            and not self._is_projected_profile_var(self.temp_var_name)
            and self.depth_var_name not in self.ds
        ):
            missing.append(self.depth_var_name)
        if missing:
            raise RuntimeError(
                f"ARGO zarr is missing required variables {missing}: {self.root_dir}"
            )

        juld = np.asarray(self.ds["JULD"].values, dtype=np.float64).reshape(-1)
        lat = np.asarray(self.ds["LATITUDE"].values, dtype=np.float64).reshape(-1)
        lon = np.asarray(self.ds["LONGITUDE"].values, dtype=np.float64).reshape(-1)
        n_prof = min(int(juld.size), int(lat.size), int(lon.size))
        if n_prof == 0:
            raise RuntimeError(f"No profiles found in ARGO zarr: {self.root_dir}")

        if "DATE" in self.ds:
            dates = np.asarray(self.ds["DATE"].values, dtype=np.int64).reshape(-1)[
                :n_prof
            ]
        else:
            dates = _juld_to_yyyymmdd(juld[:n_prof]).astype(np.int64, copy=False)
        self.profile_date = dates.astype(np.int64, copy=False)
        self.latitude = lat[:n_prof].astype(np.float64, copy=False)
        self.longitude = np.asarray(
            [_normalize_lon(v) for v in lon[:n_prof]], dtype=np.float64
        )
        self.profile_index = np.arange(n_prof, dtype=np.int64)
        self._has_valid_temp = self._load_profile_validity(self.temp_var_name, n_prof)
        self._indices_by_date = self._build_indices_by_date()

    def _load_profile_validity(self, var_name: str, n_prof: int) -> np.ndarray:
        helper_name = f"HAS_VALID_{var_name}"
        if helper_name in self.ds:
            return np.asarray(self.ds[helper_name].values, dtype=bool).reshape(-1)[
                :n_prof
            ]
        values = self._read_profile_matrix(var_name, np.arange(n_prof, dtype=np.int64))
        if (
            self._is_projected_profile_var(var_name)
            or self.depth_var_name not in self.ds
        ):
            return np.isfinite(values).any(axis=1)
        depth = self._read_profile_matrix(
            self.depth_var_name, np.arange(n_prof, dtype=np.int64)
        )
        return (np.isfinite(values) & np.isfinite(depth) & (depth >= 0.0)).any(axis=1)

    def _is_projected_profile_var(self, var_name: str) -> bool:
        if var_name not in self.ds:
            return False
        da = self.ds[var_name]
        return "depth" in da.dims and da.dims[0] in {"N_PROF", "profile"}

    def _build_indices_by_date(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for date_value in np.unique(self.profile_date):
            if int(date_value) <= 0:
                continue
            out[int(date_value)] = np.flatnonzero(self.profile_date == int(date_value))
        return out

    @staticmethod
    def _replace_fill_with_nan(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32)
        out[np.isclose(out, 99999.0)] = np.nan
        out[(~np.isfinite(out)) | (np.abs(out) > 1.0e6)] = np.nan
        return out

    def _read_profile_matrix(
        self,
        var_name: str,
        profile_indices: np.ndarray,
    ) -> np.ndarray:
        da = self.ds[var_name]
        if da.ndim < 2:
            raise RuntimeError(f"ARGO variable {var_name!r} must be at least 2D.")
        profile_dim = da.dims[0]
        values = da.isel(
            {profile_dim: np.asarray(profile_indices, dtype=np.int64)}
        ).values
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            values = values.reshape((int(profile_indices.size), -1))
        return self._replace_fill_with_nan(values)

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

    def load_aligned_profiles(self, indices: np.ndarray, var_name: str) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.zeros((0, int(self.depth_axis_m.size)), dtype=np.float32)
        out = np.full(
            (int(indices.size), int(self.depth_axis_m.size)), np.nan, dtype=np.float32
        )
        values = self._read_profile_matrix(var_name, indices)
        if self._is_projected_profile_var(var_name):
            if int(values.shape[1]) != int(self.depth_axis_m.size):
                raise RuntimeError(
                    "Projected ARGO profile depth count does not match GLORYS depth axis: "
                    f"{int(values.shape[1])} != {int(self.depth_axis_m.size)}"
                )
            return values.astype(np.float32, copy=False)
        depth = self._read_profile_matrix(self.depth_var_name, indices)
        for local_idx in range(int(indices.size)):
            # ARGO variables share the corrected-depth samples, so each profile is projected
            # independently onto the GLORYS depth axis used by the gridded targets.
            out[local_idx] = _align_argo_profile_to_glorys_depths(
                temperature=values[local_idx],
                depth=depth[local_idx],
                glorys_depths=self.depth_axis_m,
            )
        return out

    def load_temperature_profiles(self, indices: np.ndarray) -> np.ndarray:
        return self.load_aligned_profiles(indices, self.temp_var_name)


class ZarrPatchIndex:
    """Build compact patch/date metadata rows from zarr stores."""

    CACHE_VERSION = 2

    def __init__(
        self,
        *,
        ostia_store: TimedZarrStore,
        glorys_store: TimedZarrStore,
        argo_store: ArgoZarrStore,
        optional_time_stores: Sequence[TimedZarrStore] = (),
        cache_dir: str | Path | None,
        grid_params: _GridParams,
        temporal_window_days: int,
        ostia_var_name: str,
    ) -> None:
        self.ostia_store = ostia_store
        self.glorys_store = glorys_store
        self.argo_store = argo_store
        self.optional_time_stores = tuple(optional_time_stores)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.grid_params = grid_params
        self.temporal_window_days = int(temporal_window_days)
        self.ostia_var_name = str(ostia_var_name)
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
            f"argo_zarr_gridded_v{self.CACHE_VERSION}_"
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
            raise RuntimeError("No overlapping source dates found for zarr stores.")
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

        lat_values, lon_values = self.ostia_store.horizontal_grid(self.ostia_var_name)
        invalid_mask = self.ostia_store.invalid_mask(
            invalid_mask_flags=self.grid_params.invalid_mask_flags
        )

        tile = int(self.grid_params.tile_size)
        if tile < 1:
            raise ValueError("grid.tile_size must be >= 1.")
        if lat_values.size < tile or lon_values.size < tile:
            raise RuntimeError(
                "OSTIA zarr grid is smaller than the requested tile size."
            )

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
            raise RuntimeError("No valid patches were built from the OSTIA zarr grid.")
        phases = self._split_phases(len(records))
        for rec, phase in zip(records, phases, strict=False):
            rec["split"] = phase
            rec["phase"] = phase
        return pd.DataFrame.from_records(records)

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
        patch_lookup = _build_patch_lookup(patch_df, self.grid_params)
        for profile_idx in range(int(self.argo_store.profile_date.size)):
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


class ArgoZarrGriddedPatchDataset(Dataset):
    """Dataset that lazily builds training patches from compact zarr sources."""

    DEFAULT_CONFIG_PATH = "configs/px_space/data_ostia_argo_zarr.yaml"
    DEFAULT_ZARR_ROOT_DIR = "/data1/datasets/depth_v2/zarr_training"
    DEFAULT_METADATA_CACHE_DIR = "/data1/datasets/depth_v2/depthdif_cache"

    def __init__(
        self,
        *,
        zarr_root_dir: str | Path = DEFAULT_ZARR_ROOT_DIR,
        argo_zarr: str | Path | None = None,
        glorys_zarr: str | Path | None = None,
        ostia_zarr: str | Path | None = None,
        sealevel_zarr: str | Path | None = None,
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
        glorys_salinity_var_name: str = "so",
        glorys_ssh_var_name: str = "zos",
        ostia_var_name: str = "analysed_sst",
        argo_temp_var_name: str = "TEMP",
        argo_salinity_var_name: str = "PSAL_CORRECTED",
        argo_depth_var_name: str = ARGO_DEPTH_VAR,
        sealevel_ssh_var_name: str = "adt",
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
        return_modalities: bool = False,
        random_seed: int = 7,
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
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
        self.return_modalities = bool(return_modalities)
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

        paths = self._resolve_paths(
            zarr_root_dir=zarr_root_dir,
            argo_zarr=argo_zarr,
            glorys_zarr=glorys_zarr,
            ostia_zarr=ostia_zarr,
            sealevel_zarr=sealevel_zarr,
        )
        self.glorys_store = TimedZarrStore(paths.glorys)
        self.ostia_store = TimedZarrStore(paths.ostia)
        self.sealevel_store = (
            None if paths.sealevel is None else TimedZarrStore(paths.sealevel)
        )
        self._depth_axis_m = self.glorys_store.depth_axis_m()

        self.glorys_var_name = _first_available_var(
            self.glorys_store.variable_names,
            glorys_var_name,
            ("thetao",),
            required=True,
            role="GLORYS temperature",
        )
        self.glorys_salinity_var_name = _first_available_var(
            self.glorys_store.variable_names,
            glorys_salinity_var_name,
            ("so",),
            required=False,
            role="GLORYS salinity",
        )
        self.glorys_ssh_var_name = _first_available_var(
            self.glorys_store.variable_names,
            glorys_ssh_var_name,
            ("zos",),
            required=False,
            role="GLORYS sea surface height",
        )
        self.ostia_var_name = _first_available_var(
            self.ostia_store.variable_names,
            ostia_var_name,
            ("analysed_sst",),
            required=True,
            role="OSTIA SST",
        )
        sealevel_names = (
            () if self.sealevel_store is None else self.sealevel_store.variable_names
        )
        self.sealevel_ssh_var_name = _first_available_var(
            sealevel_names,
            sealevel_ssh_var_name,
            ("adt", "sla"),
            required=False,
            role="altimetry sea surface height",
        )

        argo_variable_names = self._read_zarr_variable_names(paths.argo)
        argo_temp_name = _first_available_var(
            argo_variable_names,
            argo_temp_var_name,
            ("TEMP", "POTM_CORRECTED"),
            required=True,
            role="ARGO temperature",
        )
        argo_salinity_name = _first_available_var(
            argo_variable_names,
            argo_salinity_var_name,
            ("PSAL_CORRECTED", "PSAL"),
            required=False,
            role="ARGO salinity",
        )
        self.argo_store = ArgoZarrStore(
            paths.argo,
            depth_axis_m=self._depth_axis_m,
            temp_var_name=str(argo_temp_name),
            salinity_var_name=argo_salinity_name,
            depth_var_name=argo_depth_var_name,
        )
        self.argo_temp_var_name = self.argo_store.temp_var_name
        self.argo_salinity_var_name = self.argo_store.salinity_var_name

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
        index = ZarrPatchIndex(
            ostia_store=self.ostia_store,
            glorys_store=self.glorys_store,
            argo_store=self.argo_store,
            optional_time_stores=(
                () if self.sealevel_store is None else (self.sealevel_store,)
            ),
            cache_dir=metadata_cache_dir,
            grid_params=grid_params,
            temporal_window_days=self.temporal_window_days,
            ostia_var_name=self.ostia_var_name,
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

    @property
    def modality_variables(self) -> dict[str, list[str]]:
        out = {
            "ostia": [self.ostia_var_name],
            "argo": [self.argo_temp_var_name],
            "glorys": [self.glorys_var_name],
        }
        if self.argo_salinity_var_name is not None:
            out["argo"].append(self.argo_salinity_var_name)
        if self.glorys_salinity_var_name is not None:
            out["glorys"].append(self.glorys_salinity_var_name)
        if self.glorys_ssh_var_name is not None:
            out["glorys"].append(self.glorys_ssh_var_name)
        if self.sealevel_ssh_var_name is not None:
            out["sealevel"] = [self.sealevel_ssh_var_name]
        return out

    @staticmethod
    def _resolve_paths(
        *,
        zarr_root_dir: str | Path,
        argo_zarr: str | Path | None,
        glorys_zarr: str | Path | None,
        ostia_zarr: str | Path | None,
        sealevel_zarr: str | Path | None,
    ) -> _ZarrPaths:
        root = Path(zarr_root_dir)

        def required(value: str | Path | None, name: str) -> Path:
            if isinstance(value, str) and value.strip().lower() in MISSING_TEXT_VALUES:
                value = None
            path = root / name if value is None else Path(value)
            if not path.exists():
                raise FileNotFoundError(f"Required zarr source does not exist: {path}")
            return path

        if (
            isinstance(sealevel_zarr, str)
            and sealevel_zarr.strip().lower() in MISSING_TEXT_VALUES
        ):
            sealevel_path = None
        else:
            sealevel_path = (
                root / "sealevel.zarr" if sealevel_zarr is None else Path(sealevel_zarr)
            )
        if sealevel_path is not None and not sealevel_path.exists():
            sealevel_path = None
        return _ZarrPaths(
            argo=required(argo_zarr, "argo.zarr"),
            glorys=required(glorys_zarr, "glorys.zarr"),
            ostia=required(ostia_zarr, "ostia.zarr"),
            sealevel=sealevel_path,
        )

    @staticmethod
    def _read_zarr_variable_names(path: Path) -> tuple[str, ...]:
        ds = xr.open_zarr(
            path,
            consolidated=None,
            decode_times=False,
            mask_and_scale=True,
        )
        try:
            return tuple(str(name) for name in ds.data_vars)
        finally:
            ds.close()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        split: str = "all",
    ) -> "ArgoZarrGriddedPatchDataset":
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH
        with Path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg.get("dataset", {})
        return cls(
            zarr_root_dir=cls._cfg_get(
                ds_cfg,
                "core.zarr_root_dir",
                "zarr_root_dir",
                default=cls.DEFAULT_ZARR_ROOT_DIR,
            ),
            argo_zarr=cls._cfg_get(ds_cfg, "core.argo_zarr", "argo_zarr", default=None),
            glorys_zarr=cls._cfg_get(
                ds_cfg,
                "core.glorys_zarr",
                "glorys_zarr",
                default=None,
            ),
            ostia_zarr=cls._cfg_get(
                ds_cfg, "core.ostia_zarr", "ostia_zarr", default=None
            ),
            sealevel_zarr=cls._cfg_get(
                ds_cfg,
                "core.sealevel_zarr",
                "sealevel_zarr",
                default=None,
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
            glorys_salinity_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.glorys_salinity_var_name",
                    "glorys_salinity_var_name",
                    default="so",
                )
            ),
            glorys_ssh_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.glorys_ssh_var_name",
                    "glorys_ssh_var_name",
                    default="zos",
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
            argo_salinity_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.argo_salinity_var_name",
                    "argo_salinity_var_name",
                    default="PSAL_CORRECTED",
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
            sealevel_ssh_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.sealevel_ssh_var_name",
                    "sealevel_ssh_var_name",
                    default="adt",
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
            return_modalities=bool(
                cls._cfg_get(
                    ds_cfg,
                    "output.return_modalities",
                    "return_modalities",
                    default=False,
                )
            ),
            random_seed=int(
                cls._cfg_get(ds_cfg, "runtime.random_seed", "random_seed", default=7)
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
        if flat_key in cfg:
            return cfg[flat_key]
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
        return self._rasterize_argo_variable_patch(row, self.argo_temp_var_name)

    def _rasterize_argo_variable_patch(
        self,
        row: dict[str, Any],
        var_name: str,
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

        values = self.argo_store.load_aligned_profiles(indices, var_name)
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
            x_np[depth_valid, int(row_idx), int(col_idx)] = y_np[
                depth_valid,
                int(row_idx),
                int(col_idx),
            ]
            x_valid[depth_valid, int(row_idx), int(col_idx)] = True
        return x_np, x_valid

    def _add_modalities(
        self,
        sample: dict[str, Any],
        row: dict[str, Any],
        axes: PatchAxes,
        *,
        eo_np: np.ndarray,
        x_np: np.ndarray,
        y_np: np.ndarray,
    ) -> None:
        modalities: dict[str, dict[str, torch.Tensor]] = {}
        masks: dict[str, dict[str, torch.Tensor]] = {}

        modalities["ostia"] = {}
        masks["ostia"] = {}
        (
            modalities["ostia"][self.ostia_var_name],
            masks["ostia"][self.ostia_var_name],
        ) = _tensor_with_mask(eo_np)

        modalities["argo"] = {}
        masks["argo"] = {}
        (
            modalities["argo"][self.argo_temp_var_name],
            masks["argo"][self.argo_temp_var_name],
        ) = _tensor_with_mask(x_np)
        if self.argo_salinity_var_name is not None:
            salinity_np, _ = self._rasterize_argo_variable_patch(
                row,
                self.argo_salinity_var_name,
            )
            (
                modalities["argo"][self.argo_salinity_var_name],
                masks["argo"][self.argo_salinity_var_name],
            ) = _tensor_with_mask(salinity_np)

        modalities["glorys"] = {}
        masks["glorys"] = {}
        (
            modalities["glorys"][self.glorys_var_name],
            masks["glorys"][self.glorys_var_name],
        ) = _tensor_with_mask(y_np)
        if self.glorys_salinity_var_name is not None:
            salinity_np = self.glorys_store.read_patch(
                target_date=int(row["date"]),
                var_name=self.glorys_salinity_var_name,
                axes=axes,
                categorical=False,
            )
            (
                modalities["glorys"][self.glorys_salinity_var_name],
                masks["glorys"][self.glorys_salinity_var_name],
            ) = _tensor_with_mask(salinity_np)
        if self.glorys_ssh_var_name is not None:
            ssh_np = self.glorys_store.read_patch(
                target_date=int(row["date"]),
                var_name=self.glorys_ssh_var_name,
                axes=axes,
                categorical=False,
            )
            if ssh_np.ndim == 2:
                ssh_np = ssh_np[None, ...]
            (
                modalities["glorys"][self.glorys_ssh_var_name],
                masks["glorys"][self.glorys_ssh_var_name],
            ) = _tensor_with_mask(ssh_np)

        if self.sealevel_store is not None and self.sealevel_ssh_var_name is not None:
            ssh_np = self.sealevel_store.read_patch(
                target_date=int(row["date"]),
                var_name=self.sealevel_ssh_var_name,
                axes=axes,
                categorical=False,
            )
            if ssh_np.ndim == 2:
                ssh_np = ssh_np[None, ...]
            modalities["sealevel"] = {}
            masks["sealevel"] = {}
            (
                modalities["sealevel"][self.sealevel_ssh_var_name],
                masks["sealevel"][self.sealevel_ssh_var_name],
            ) = _tensor_with_mask(ssh_np)

        sample["modalities"] = modalities
        sample["modality_valid_masks"] = masks

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
        if self.return_modalities:
            self._add_modalities(
                sample,
                row,
                axes,
                eo_np=eo_np,
                x_np=x_np,
                y_np=y_np,
            )
        if self.return_info:
            info = dict(row)
            info["x_source"] = "glorys_synthetic" if self.synthetic_mode else "argo"
            info["synthetic_pixel_count"] = (
                int(self.synthetic_pixel_count) if self.synthetic_mode else 0
            )
            info["modality_variables"] = self.modality_variables
            sample["info"] = info
        return sample
