"""
Production-range enriched ARGO export:
/work/envs/depth/bin/python data/dataset_creation/export_aligned_argo/b_export_enriched_argo_profiles.py \
  --start-date 20100101 \
  --end-date 20240731 \
  --output-zarr /data1/datasets/depth_v2/enriched_argo_profiles.zarr

Small smoke export:
/work/envs/depth/bin/python data/dataset_creation/export_aligned_argo/b_export_enriched_argo_profiles.py \
  --start-date 20100101 \
  --end-date 20100101 \
  --max-profiles 4 \
  --batch-size 2 \
  --output-zarr /tmp/depthdif_enriched_argo_profiles_smoke.zarr \
  --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_argo_netcdf_gridded import (
    GLORYS_MIN_ABSOLUTE_DEPTH_CUTOFF_M,
    GLORYS_RELATIVE_DEPTH_CUTOFF,
    _align_argo_profile_to_glorys_depths,
)
from data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_DEPTH_VAR,
    ARGO_LEVEL_QC_VARS,
    ARGO_PROFILE_VARS,
    ARGO_PROFILE_QC_VARS,
    GLORYS_2D_VARS,
    GLORYS_3D_VARS,
    OSTIA_VARS,
    SEALEVEL_VARS,
    SOURCE_VARIABLES,
    TimedFile,
    _filter_argo_files_by_date_range,
    _open_argo_dataset,
    scan_timed_files,
)
CATEGORICAL_VARS = {"mask", "flag_ice"}
SOURCE_PRODUCTS = {
    "argo": {
        "provider": "UK Met Office Hadley Centre",
        "product": "EN4.2.2 profile archive",
        "role": "In-situ profile observations projected onto the GLORYS depth grid.",
    },
    "glorys": {
        "provider": "Copernicus Marine Service",
        "product": "Global Ocean Physics Reanalysis / GLORYS12V1",
        "role": "3D ocean reanalysis fields and 2D model surface/ice fields sampled at profile points.",
    },
    "ostia": {
        "provider": "Copernicus Marine Service / UK Met Office OSTIA",
        "product": "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
        "role": "Daily analysed sea-surface temperature and mask fields sampled at profile points.",
    },
    "sealevel": {
        "provider": "Copernicus Marine Service",
        "product": "SEALEVEL_GLO_PHY_L4_MY_008_047",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "role": "Daily sea-level, geostrophic current, and ice-flag fields sampled at profile points.",
    },
}
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?P<prefix>^|[\s=:'\"(\[{])(?P<path>/(?!/)[^\s,;)\]\}]+)"
)

MISSING_STATUS = np.int8(2)
INTERPOLATED_STATUS = np.int8(0)
NEAREST_EDGE_STATUS = np.int8(1)
MISSING_QC_FLAG = np.int8(-1)
ARGO_LEVEL_QC_VALUE_KEYS = {
    "depth": "depth",
    "temp": "temp",
    "potm": "potm",
    "psal": "psal",
}


class DatasetCache:
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


def _normalize_lon(lon: float) -> float:
    return float(((float(lon) + 180.0) % 360.0) - 180.0)


def bracket_timed_files(index: list[TimedFile], target_day: float) -> tuple[TimedFile | None, TimedFile | None, float, np.int8]:
    if not index:
        return None, None, np.nan, MISSING_STATUS
    days = np.asarray([item.day for item in index], dtype=np.float64)
    pos = int(np.searchsorted(days, float(target_day), side="left"))
    if pos < len(index) and np.isclose(days[pos], target_day, rtol=0.0, atol=1.0e-8):
        return index[pos], index[pos], 0.0, INTERPOLATED_STATUS
    if pos == 0:
        return index[0], index[0], 0.0, NEAREST_EDGE_STATUS
    if pos >= len(index):
        return index[-1], index[-1], 0.0, NEAREST_EDGE_STATUS
    prev_item = index[pos - 1]
    next_item = index[pos]
    span = next_item.day - prev_item.day
    if span <= 0.0:
        return prev_item, next_item, 0.0, INTERPOLATED_STATUS
    weight = float((float(target_day) - prev_item.day) / span)
    return prev_item, next_item, weight, INTERPOLATED_STATUS


def project_argo_profile_to_glorys_depths(
    values: np.ndarray,
    depths: np.ndarray,
    glorys_depths: np.ndarray,
) -> np.ndarray:
    # Keep the export projection identical to the active NetCDF patch dataset.
    return _align_argo_profile_to_glorys_depths(
        temperature=np.asarray(values, dtype=np.float32),
        depth=np.asarray(depths, dtype=np.float32),
        glorys_depths=np.asarray(glorys_depths, dtype=np.float32),
    )


def _qc_to_int_array(qc: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    out = np.full(shape, MISSING_QC_FLAG, dtype=np.int8)
    if qc is None:
        return out
    arr = np.asarray(qc)
    if arr.size != out.size:
        return out
    arr = arr.reshape(shape)
    if np.issubdtype(arr.dtype, np.number):
        valid = np.isfinite(arr)
        out[valid] = arr[valid].astype(np.int8, copy=False)
        return out

    text = np.char.strip(arr.astype("U8"))
    for code in range(10):
        out[text == str(code)] = np.int8(code)
    return out


def _qc_scalar_to_int(qc: np.ndarray | np.generic | str | bytes | None) -> np.int8:
    return _qc_to_int_array(None if qc is None else np.asarray([qc]), (1,))[0]


def _collapse_duplicate_profile_qc(
    depth: np.ndarray,
    qc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_depths, starts = np.unique(depth, return_index=True)
    collapsed_qc = np.maximum.reduceat(qc, starts)
    return unique_depths, collapsed_qc.astype(np.int8, copy=False)


def _project_argo_qc_to_glorys_depths(
    qc: np.ndarray | None,
    *,
    values: np.ndarray,
    depth: np.ndarray,
    glorys_depths: np.ndarray,
) -> np.ndarray:
    target_depths = np.asarray(glorys_depths, dtype=np.float64).reshape(-1)
    out = np.full(target_depths.shape, MISSING_QC_FLAG, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    qc_values = _qc_to_int_array(qc, depth.shape).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(depth) & (depth >= 0.0)
    if not np.any(valid):
        return out

    depth = depth[valid]
    qc_values = qc_values[valid]
    order = np.argsort(depth, kind="mergesort")
    depth = depth[order]
    qc_values = qc_values[order]
    depth, qc_values = _collapse_duplicate_profile_qc(depth, qc_values)
    if depth.size == 0:
        return out

    insert_idx = np.searchsorted(depth, target_depths, side="left")
    left_idx = np.clip(insert_idx - 1, 0, max(depth.size - 1, 0))
    right_idx = np.clip(insert_idx, 0, max(depth.size - 1, 0))

    left_depth = depth[left_idx]
    right_depth = depth[right_idx]
    nearest_depth_distance = np.minimum(
        np.abs(target_depths - left_depth),
        np.abs(target_depths - right_depth),
    )
    max_allowed_distance = np.maximum(
        GLORYS_RELATIVE_DEPTH_CUTOFF * target_depths,
        GLORYS_MIN_ABSOLUTE_DEPTH_CUTOFF_M,
    )
    in_range = (
        np.isfinite(target_depths)
        & (target_depths >= depth[0])
        & (target_depths <= depth[-1])
    )
    within_cutoff = np.isfinite(nearest_depth_distance) & (
        nearest_depth_distance <= max_allowed_distance
    )
    valid_targets = in_range & within_cutoff
    if not np.any(valid_targets):
        return out

    # Exact target-depth matches keep the source QC flag. Interpolated targets
    # keep the worst available code from the bracketing source levels.
    exact_right = valid_targets & np.isclose(
        target_depths,
        right_depth,
        rtol=0.0,
        atol=1.0e-6,
    )
    if np.any(exact_right):
        out[exact_right] = qc_values[right_idx[exact_right]]
    interpolated = valid_targets & ~exact_right
    if np.any(interpolated):
        out[interpolated] = np.maximum(
            qc_values[left_idx[interpolated]],
            qc_values[right_idx[interpolated]],
        )
    return out


def _sample_dataarray_at_point(
    da: xr.DataArray,
    *,
    lat: float,
    lon: float,
    method: str,
) -> np.ndarray:
    if "time" in da.dims:
        da = da.isel(time=0)
    if "latitude" in da.dims and "longitude" in da.dims:
        lat_name = "latitude"
        lon_name = "longitude"
    elif "lat" in da.dims and "lon" in da.dims:
        lat_name = "lat"
        lon_name = "lon"
    else:
        return np.asarray(da.values, dtype=np.float32)

    lon_values = np.asarray(da[lon_name].values, dtype=np.float64)
    sample_lon = _normalize_lon(lon)
    if lon_values.size > 0 and np.nanmin(lon_values) >= 0.0:
        sample_lon = sample_lon % 360.0

    if method == "nearest":
        sampled = da.sel({lat_name: float(lat), lon_name: sample_lon}, method="nearest")
    else:
        # Linear interpolation is only meaningful for continuous gridded fields.
        sampled = da.interp({lat_name: float(lat), lon_name: sample_lon}, method="linear")
    return np.asarray(sampled.values, dtype=np.float32)


def sample_spatial_value(
    ds: xr.Dataset,
    var_name: str,
    *,
    lat: float,
    lon: float,
    categorical: bool = False,
) -> np.ndarray:
    if var_name not in ds:
        return np.asarray(np.nan, dtype=np.float32)
    method = "nearest" if categorical else "linear"
    return _sample_dataarray_at_point(ds[var_name], lat=lat, lon=lon, method=method)


def _nearest_time_item(before: TimedFile, after: TimedFile, target_day: float) -> TimedFile:
    if abs(before.day - target_day) <= abs(after.day - target_day):
        return before
    return after


def sample_temporal_value(
    index: list[TimedFile],
    cache: DatasetCache,
    var_name: str,
    *,
    target_day: float,
    lat: float,
    lon: float,
    categorical: bool = False,
) -> tuple[np.ndarray, np.int8]:
    before, after, weight, status = bracket_timed_files(index, target_day)
    if before is None or after is None:
        return np.asarray(np.nan, dtype=np.float32), status
    if categorical:
        item = _nearest_time_item(before, after, target_day)
        return sample_spatial_value(
            cache.get(item.path), var_name, lat=lat, lon=lon, categorical=True
        ), status
    first = sample_spatial_value(cache.get(before.path), var_name, lat=lat, lon=lon)
    if before.path == after.path:
        return first, status
    second = sample_spatial_value(cache.get(after.path), var_name, lat=lat, lon=lon)
    return (
        (np.asarray(first, dtype=np.float32) * np.float32(1.0 - weight))
        + (np.asarray(second, dtype=np.float32) * np.float32(weight))
    ), status


def _source_file_label(path: Path) -> str:
    return Path(path).name


def _yyyymmdd_or_none(day: float | None) -> int | None:
    if day is None or not np.isfinite(day):
        return None
    date = np.datetime64("1950-01-01", "D") + np.timedelta64(int(np.floor(day)), "D")
    return int(np.datetime_as_string(date, unit="D").replace("-", ""))


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _sanitize_metadata_value(value.item())
    if isinstance(value, np.ndarray):
        return [_sanitize_metadata_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _sanitize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, str):
        text = value.strip()
        # Do not persist local absolute filesystem paths in portable Zarr metadata.
        # The negative // check keeps URL schemes such as https:// untouched.
        return _ABSOLUTE_PATH_PATTERN.sub(
            lambda match: f"{match.group('prefix')}{Path(match.group('path')).name}",
            text,
        )
    if isinstance(value, (int, bool)) or value is None:
        return value
    return str(value)


def _sanitize_attrs(attrs: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_metadata_value(value) for key, value in attrs.items()}


def _open_source_metadata_dataset(path: Path, *, kind: str) -> xr.Dataset:
    if kind == "argo":
        return _open_argo_dataset(path)
    return xr.open_dataset(
        path,
        engine="h5netcdf",
        decode_times=False,
        mask_and_scale=True,
        cache=False,
    )


def _extract_source_metadata(
    *,
    kind: str,
    files: list[Path],
    variables: tuple[str, ...],
) -> dict[str, Any]:
    if not files:
        return {
            "file_count": 0,
            "representative_file": None,
            "global_attrs": {},
            "dimensions": {},
            "variables": {},
            **SOURCE_PRODUCTS.get(kind, {}),
        }

    representative = files[0]
    with _open_source_metadata_dataset(representative, kind=kind) as ds:
        variable_attrs: dict[str, Any] = {}
        for name in variables:
            if name in ds:
                variable_attrs[name] = {
                    "dims": list(ds[name].dims),
                    "dtype": str(ds[name].dtype),
                    "attrs": _sanitize_attrs(dict(ds[name].attrs)),
                }
        for coord_name in ("time", "depth", "latitude", "longitude", "lat", "lon"):
            if coord_name in ds.coords and coord_name not in variable_attrs:
                variable_attrs[coord_name] = {
                    "dims": list(ds[coord_name].dims),
                    "dtype": str(ds[coord_name].dtype),
                    "attrs": _sanitize_attrs(dict(ds[coord_name].attrs)),
                }
        return {
            **SOURCE_PRODUCTS.get(kind, {}),
            "file_count": len(files),
            "representative_file": representative.name,
            "first_file": files[0].name,
            "last_file": files[-1].name,
            "global_attrs": _sanitize_attrs(dict(ds.attrs)),
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "variables": variable_attrs,
        }


def _metadata_file_list(index: list[TimedFile]) -> list[Path]:
    return [item.path for item in index]


def _argo_metadata_variables() -> tuple[str, ...]:
    optional_qc = tuple(ARGO_LEVEL_QC_VARS.values()) + tuple(ARGO_PROFILE_QC_VARS.values())
    return SOURCE_VARIABLES["argo"] + tuple(
        name for name in optional_qc if name not in SOURCE_VARIABLES["argo"]
    )


def _build_export_metadata(
    *,
    argo_files: list[Path],
    glorys_index: list[TimedFile],
    ostia_index: list[TimedFile],
    sealevel_index: list[TimedFile],
    start_date: int | None,
    end_date: int | None,
    batch_size: int,
    max_profiles: int | None,
) -> dict[str, Any]:
    source_file_summaries = {
        "argo": {
            "file_count": len(argo_files),
            "first_file": argo_files[0].name if argo_files else None,
            "last_file": argo_files[-1].name if argo_files else None,
        },
        "glorys": _timed_index_summary(glorys_index),
        "ostia": _timed_index_summary(ostia_index),
        "sealevel": _timed_index_summary(sealevel_index),
    }
    return {
        "description": "ARGO profiles enriched with freshly collocated GLORYS, OSTIA, and sea-level fields.",
        "created_by": "data/dataset_creation/export_aligned_argo/b_export_enriched_argo_profiles.py",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "requested_date_range": {"start_date": start_date, "end_date": end_date},
        "batch_size": int(batch_size),
        "max_profiles": None if max_profiles is None else int(max_profiles),
        "path_policy": "No absolute source filesystem paths are stored. profile_source_file stores source filenames only.",
        "profile_axis": "One row per valid EN4/ARGO profile passing date and coordinate filters.",
        "depth_axis": "GLORYS native depth coordinate, in meters, loaded from the first readable GLORYS file.",
        "source_file_summary": source_file_summaries,
        "source_metadata": {
            "argo": _extract_source_metadata(
                kind="argo",
                files=argo_files,
                variables=_argo_metadata_variables(),
            ),
            "glorys": _extract_source_metadata(
                kind="glorys",
                files=_metadata_file_list(glorys_index),
                variables=SOURCE_VARIABLES["glorys"],
            ),
            "ostia": _extract_source_metadata(
                kind="ostia",
                files=_metadata_file_list(ostia_index),
                variables=SOURCE_VARIABLES["ostia"],
            ),
            "sealevel": _extract_source_metadata(
                kind="sealevel",
                files=_metadata_file_list(sealevel_index),
                variables=SOURCE_VARIABLES["sealevel"],
            ),
        },
        "processing": {
            "argo_depth_projection": (
                "ARGO TEMP, POTM_CORRECTED, and PSAL_CORRECTED are interpolated onto "
                "the GLORYS depth coordinate with duplicate-depth collapse and the "
                "existing GLORYS depth cutoff rules."
            ),
            "spatial_collocation": (
                "Continuous gridded fields use xarray linear interpolation at profile "
                "latitude/longitude. Categorical variables use nearest-neighbor sampling."
            ),
            "temporal_collocation": (
                "Continuous fields are linearly interpolated between bracketing source "
                "files. Categorical variables use the nearest bracketing source file."
            ),
            "argo_quality_flags": (
                "Optional EN4/ARGO QC variables are stored as int8 QC codes. Depth-level "
                "QC variables are projected onto the GLORYS depth coordinate by keeping "
                "the exact source-depth code where possible, otherwise the worst code "
                "from the bracketing source levels. A value of -1 means unavailable or "
                "unsupported at that target depth."
            ),
            "longitude_handling": (
                "Profile longitudes are normalized to [-180, 180), then converted to "
                "0..360 only when the source grid uses non-negative longitudes."
            ),
        },
        "status_values": {
            "0": "interpolated_or_exact",
            "1": "nearest_edge",
            "2": "missing",
        },
        "primary_external_sea_surface_height": "sealevel_adt",
        "known_source_notes": {
            "sealevel_tpa_correction": (
                "The Copernicus source metadata marks this field as not implemented in "
                "the current product version, so values may be NaN."
            )
        },
    }


def _timed_index_summary(index: list[TimedFile]) -> dict[str, Any]:
    if not index:
        return {
            "file_count": 0,
            "first_file": None,
            "last_file": None,
            "first_date": None,
            "last_date": None,
        }
    return {
        "file_count": len(index),
        "first_file": index[0].path.name,
        "last_file": index[-1].path.name,
        "first_date": _yyyymmdd_or_none(index[0].day),
        "last_date": _yyyymmdd_or_none(index[-1].day),
    }


def _load_glorys_depths(glorys_index: list[TimedFile]) -> np.ndarray:
    if not glorys_index:
        raise RuntimeError("Cannot export enriched profiles without readable GLORYS files.")
    with xr.open_dataset(
        glorys_index[0].path,
        engine="h5netcdf",
        decode_times=False,
        mask_and_scale=True,
        cache=False,
    ) as ds:
        if "depth" not in ds:
            raise RuntimeError(f"GLORYS file is missing depth coordinate: {glorys_index[0].path}")
        return np.asarray(ds["depth"].values, dtype=np.float32)


def _empty_batch() -> dict[str, list[Any]]:
    keys = [
        "profile_source_file",
        "profile_idx",
        "profile_date",
        "profile_juld",
        "latitude",
        "longitude",
        "valid_observed_depth_count",
        "argo_temp_on_glorys_depth",
        "argo_potm_on_glorys_depth",
        "argo_psal_on_glorys_depth",
        "argo_temp_valid_on_glorys_depth",
        "argo_potm_valid_on_glorys_depth",
        "argo_psal_valid_on_glorys_depth",
        "glorys_temporal_status",
        "ostia_temporal_status",
        "sealevel_temporal_status",
    ]
    keys.extend(f"argo_{name}_qc_on_glorys_depth" for name in ARGO_LEVEL_QC_VARS)
    keys.extend(f"argo_{name}_qc" for name in ARGO_PROFILE_QC_VARS)
    keys.extend(f"glorys_{name}" for name in GLORYS_3D_VARS + GLORYS_2D_VARS)
    keys.extend(f"ostia_{name}" for name in OSTIA_VARS)
    keys.extend(f"sealevel_{name}" for name in SEALEVEL_VARS)
    return {key: [] for key in keys}


def _append_profile_to_batch(
    batch: dict[str, list[Any]],
    *,
    argo_path: Path,
    profile_idx: int,
    profile_date: int,
    profile_juld: float,
    lat: float,
    lon: float,
    depth: np.ndarray,
    temp: np.ndarray,
    potm: np.ndarray,
    psal: np.ndarray,
    argo_level_qc: dict[str, np.ndarray | None],
    argo_profile_qc: dict[str, np.int8],
    glorys_depths: np.ndarray,
    glorys_index: list[TimedFile],
    ostia_index: list[TimedFile],
    sealevel_index: list[TimedFile],
    cache: DatasetCache,
) -> None:
    target_day = float(profile_juld)
    valid_depth_count = int(np.count_nonzero(np.isfinite(depth) & (depth >= 0.0)))
    projected_temp = project_argo_profile_to_glorys_depths(temp, depth, glorys_depths)
    projected_potm = project_argo_profile_to_glorys_depths(potm, depth, glorys_depths)
    projected_psal = project_argo_profile_to_glorys_depths(psal, depth, glorys_depths)

    batch["profile_source_file"].append(_source_file_label(argo_path))
    batch["profile_idx"].append(int(profile_idx))
    batch["profile_date"].append(int(profile_date))
    batch["profile_juld"].append(float(profile_juld))
    batch["latitude"].append(float(lat))
    batch["longitude"].append(float(lon))
    batch["valid_observed_depth_count"].append(valid_depth_count)
    batch["argo_temp_on_glorys_depth"].append(projected_temp)
    batch["argo_potm_on_glorys_depth"].append(projected_potm)
    batch["argo_psal_on_glorys_depth"].append(projected_psal)
    batch["argo_temp_valid_on_glorys_depth"].append(np.isfinite(projected_temp))
    batch["argo_potm_valid_on_glorys_depth"].append(np.isfinite(projected_potm))
    batch["argo_psal_valid_on_glorys_depth"].append(np.isfinite(projected_psal))
    source_profile_values = {
        "depth": depth,
        "temp": temp,
        "potm": potm,
        "psal": psal,
    }
    for name in ARGO_LEVEL_QC_VARS:
        value_key = ARGO_LEVEL_QC_VALUE_KEYS.get(name, "depth")
        batch[f"argo_{name}_qc_on_glorys_depth"].append(
            _project_argo_qc_to_glorys_depths(
                argo_level_qc.get(name),
                values=source_profile_values[value_key],
                depth=depth,
                glorys_depths=glorys_depths,
            )
        )
    for name in ARGO_PROFILE_QC_VARS:
        batch[f"argo_{name}_qc"].append(
            np.int8(argo_profile_qc.get(name, MISSING_QC_FLAG))
        )

    source_status: dict[str, np.int8] = {}
    for source_name, index, names in (
        ("glorys", glorys_index, GLORYS_3D_VARS + GLORYS_2D_VARS),
        ("ostia", ostia_index, OSTIA_VARS),
        ("sealevel", sealevel_index, SEALEVEL_VARS),
    ):
        status_values: list[np.int8] = []
        for name in names:
            value, status = sample_temporal_value(
                index,
                cache,
                name,
                target_day=target_day,
                lat=lat,
                lon=lon,
                categorical=name in CATEGORICAL_VARS,
            )
            batch[f"{source_name}_{name}"].append(value)
            status_values.append(status)
        source_status[source_name] = (
            MISSING_STATUS if not status_values else np.max(status_values).astype(np.int8)
        )
    batch["glorys_temporal_status"].append(source_status["glorys"])
    batch["ostia_temporal_status"].append(source_status["ostia"])
    batch["sealevel_temporal_status"].append(source_status["sealevel"])


def _batch_to_dataset(
    batch: dict[str, list[Any]],
    *,
    profile_start: int,
    glorys_depths: np.ndarray,
) -> xr.Dataset:
    n = len(batch["profile_idx"])
    profile_coord = np.arange(profile_start, profile_start + n, dtype=np.int64)
    data_vars: dict[str, tuple[tuple[str, ...], Any]] = {}

    scalar_int = {"profile_idx", "profile_date", "valid_observed_depth_count"}
    scalar_float = {"profile_juld", "latitude", "longitude"}
    depth_vars = {
        "argo_temp_on_glorys_depth",
        "argo_potm_on_glorys_depth",
        "argo_psal_on_glorys_depth",
        "argo_temp_valid_on_glorys_depth",
        "argo_potm_valid_on_glorys_depth",
        "argo_psal_valid_on_glorys_depth",
    }
    depth_int8_vars = {
        f"argo_{name}_qc_on_glorys_depth" for name in ARGO_LEVEL_QC_VARS
    }
    scalar_int8_vars = {f"argo_{name}_qc" for name in ARGO_PROFILE_QC_VARS}
    depth_vars.update(depth_int8_vars)
    for name in GLORYS_3D_VARS:
        depth_vars.add(f"glorys_{name}")

    for key, values in batch.items():
        if key == "profile_source_file":
            data_vars[key] = (("profile",), np.asarray(values, dtype=str))
        elif key in depth_vars:
            arr = np.asarray(values)
            if key in depth_int8_vars:
                arr = arr.astype(np.int8, copy=False)
            elif arr.dtype == bool:
                arr = arr.astype(bool, copy=False)
            else:
                arr = arr.astype(np.float32, copy=False)
            data_vars[key] = (("profile", "glorys_depth"), arr)
        elif key in scalar_int8_vars:
            data_vars[key] = (("profile",), np.asarray(values, dtype=np.int8))
        elif key in scalar_int:
            data_vars[key] = (("profile",), np.asarray(values, dtype=np.int64))
        elif key in scalar_float:
            data_vars[key] = (("profile",), np.asarray(values, dtype=np.float64))
        elif key.endswith("_temporal_status"):
            data_vars[key] = (("profile",), np.asarray(values, dtype=np.int8))
        else:
            data_vars[key] = (("profile",), np.asarray(values, dtype=np.float32).reshape(n))

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "profile": profile_coord,
            "glorys_depth": np.asarray(glorys_depths, dtype=np.float32),
        },
    )


def _source_variable_metadata(
    export_metadata: dict[str, Any],
    *,
    kind: str,
    var_name: str,
) -> dict[str, Any]:
    return (
        export_metadata.get("source_metadata", {})
        .get(kind, {})
        .get("variables", {})
        .get(var_name, {})
    )


def _source_variable_attrs(
    export_metadata: dict[str, Any],
    *,
    kind: str,
    var_name: str,
) -> dict[str, Any]:
    return _source_variable_metadata(
        export_metadata,
        kind=kind,
        var_name=var_name,
    ).get("attrs", {})


def _source_units(
    export_metadata: dict[str, Any],
    *,
    kind: str,
    var_name: str,
    default: str | None = None,
) -> str | None:
    attrs = _source_variable_attrs(export_metadata, kind=kind, var_name=var_name)
    return attrs.get("units", default)


def _set_attrs(ds: xr.Dataset, name: str, attrs: dict[str, Any]) -> None:
    if name in ds:
        ds[name].attrs.update(_sanitize_attrs(attrs))


def _argo_qc_attrs(
    export_metadata: dict[str, Any],
    *,
    source_var: str,
    description: str,
) -> dict[str, Any]:
    return {
        "description": description,
        "source_product": SOURCE_PRODUCTS["argo"]["product"],
        "source_variable": source_var,
        "source_attrs": _source_variable_attrs(
            export_metadata,
            kind="argo",
            var_name=source_var,
        ),
        "flag_values": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "missing_flag_value": -1,
    }


def _apply_output_metadata(
    ds: xr.Dataset,
    *,
    export_metadata: dict[str, Any],
) -> xr.Dataset:
    ds.attrs.update(_sanitize_metadata_value(export_metadata))
    ds["profile"].attrs.update(
        {
            "long_name": "export profile row index",
            "description": "Contiguous row index assigned during this Zarr export.",
        }
    )
    ds["glorys_depth"].attrs.update(
        {
            "long_name": "GLORYS depth",
            "standard_name": "depth",
            "units": _source_units(
                export_metadata,
                kind="glorys",
                var_name="depth",
                default="m",
            ),
            "positive": "down",
            "source_variable": "depth",
            "source_attrs": _source_variable_attrs(
                export_metadata,
                kind="glorys",
                var_name="depth",
            ),
        }
    )

    _set_attrs(
        ds,
        "profile_source_file",
        {
            "long_name": "EN4 profile source filename",
            "description": "Source EN4/ARGO monthly filename only; absolute paths are intentionally not stored.",
            "source_product": SOURCE_PRODUCTS["argo"]["product"],
        },
    )
    _set_attrs(
        ds,
        "profile_idx",
        {
            "long_name": "profile index within source EN4 file",
            "description": "Zero-based profile row index in profile_source_file.",
        },
    )
    _set_attrs(
        ds,
        "profile_date",
        {
            "long_name": "profile date",
            "description": "Profile date derived from EN4 JULD.",
            "format": "YYYYMMDD",
        },
    )
    juld_source_units = _source_units(export_metadata, kind="argo", var_name="JULD")
    _set_attrs(
        ds,
        "profile_juld",
        {
            "long_name": "EN4 profile Julian day",
            # Keep this numeric instead of CF-time-decodable so appending batches
            # does not depend on optional cftime support in the runtime env.
            "value_units": "days",
            "reference_epoch": "1950-01-01T00:00:00Z",
            "source_units": juld_source_units,
            "source_variable": "JULD",
            "source_attrs": _source_variable_attrs(
                export_metadata,
                kind="argo",
                var_name="JULD",
            ),
        },
    )
    for output_name, source_name, standard_name, units in (
        ("latitude", "LATITUDE", "latitude", "degrees_north"),
        ("longitude", "LONGITUDE", "longitude", "degrees_east"),
    ):
        _set_attrs(
            ds,
            output_name,
            {
                "long_name": f"profile {output_name}",
                "standard_name": standard_name,
                "units": _source_units(
                    export_metadata,
                    kind="argo",
                    var_name=source_name,
                    default=units,
                ),
                "source_variable": source_name,
                "source_attrs": _source_variable_attrs(
                    export_metadata,
                    kind="argo",
                    var_name=source_name,
                ),
            },
        )
    _set_attrs(
        ds,
        "valid_observed_depth_count",
        {
            "long_name": "valid observed EN4 depth count",
            "description": "Number of finite, non-negative DEPH_CORRECTED samples in the source profile.",
            "source_variable": ARGO_DEPTH_VAR,
        },
    )

    for output_prefix, source_var in (
        ("argo_temp", "TEMP"),
        ("argo_potm", "POTM_CORRECTED"),
        ("argo_psal", "PSAL_CORRECTED"),
    ):
        _set_attrs(
            ds,
            f"{output_prefix}_on_glorys_depth",
            {
                "long_name": f"{source_var} projected onto GLORYS depth",
                "description": "EN4/ARGO profile variable vertically interpolated onto the GLORYS depth coordinate.",
                "source_product": SOURCE_PRODUCTS["argo"]["product"],
                "source_variable": source_var,
                "source_depth_variable": ARGO_DEPTH_VAR,
                "vertical_interpolation": export_metadata["processing"]["argo_depth_projection"],
                "units": _source_units(export_metadata, kind="argo", var_name=source_var),
                "source_attrs": _source_variable_attrs(
                    export_metadata,
                    kind="argo",
                    var_name=source_var,
                ),
            },
        )
        _set_attrs(
            ds,
            f"{output_prefix}_valid_on_glorys_depth",
            {
                "long_name": f"{source_var} finite validity mask on GLORYS depth",
                "description": "True where the projected ARGO variable has finite support at this GLORYS depth.",
                "source_variable": source_var,
            },
        )

    for name, source_var in ARGO_LEVEL_QC_VARS.items():
        _set_attrs(
            ds,
            f"argo_{name}_qc_on_glorys_depth",
            {
                "long_name": f"{source_var} projected QC code on GLORYS depth",
                **_argo_qc_attrs(
                    export_metadata,
                    source_var=source_var,
                    description=(
                        "Optional EN4/ARGO depth-level QC code carried onto the "
                        "GLORYS depth coordinate. Exact depth matches keep the "
                        "source code; interpolated targets use the worst bracketing "
                        "source-level code. -1 means unavailable or unsupported."
                    ),
                ),
                "projection": export_metadata["processing"]["argo_quality_flags"],
            },
        )

    for name, source_var in ARGO_PROFILE_QC_VARS.items():
        _set_attrs(
            ds,
            f"argo_{name}_qc",
            {
                "long_name": f"{source_var} profile QC code",
                **_argo_qc_attrs(
                    export_metadata,
                    source_var=source_var,
                    description=(
                        "Optional EN4/ARGO profile-level QC code copied from the "
                        "source profile. -1 means the source QC variable was absent."
                    ),
                ),
            },
        )

    for source_name, variables in (
        ("glorys", GLORYS_3D_VARS + GLORYS_2D_VARS),
        ("ostia", OSTIA_VARS),
        ("sealevel", SEALEVEL_VARS),
    ):
        product = SOURCE_PRODUCTS[source_name]["product"]
        for var_name in variables:
            output_name = f"{source_name}_{var_name}"
            interpolation = (
                "nearest spatial and temporal sample"
                if var_name in CATEGORICAL_VARS
                else "linear spatial interpolation and linear temporal interpolation"
            )
            _set_attrs(
                ds,
                output_name,
                {
                    "long_name": f"{source_name} {var_name} collocated at profile point",
                    "description": (
                        f"{var_name} from {product}, sampled at the EN4 profile "
                        "latitude, longitude, and JULD."
                    ),
                    "source_product": product,
                    "source_variable": var_name,
                    "collocation": interpolation,
                    "units": _source_units(
                        export_metadata,
                        kind=source_name,
                        var_name=var_name,
                    ),
                    "source_attrs": _source_variable_attrs(
                        export_metadata,
                        kind=source_name,
                        var_name=var_name,
                    ),
                },
            )

    for source_name in ("glorys", "ostia", "sealevel"):
        _set_attrs(
            ds,
            f"{source_name}_temporal_status",
            {
                "long_name": f"{source_name} temporal collocation status",
                "description": "Worst temporal status across variables sampled from this source for the profile.",
                "flag_values": [0, 1, 2],
                "flag_meanings": "interpolated_or_exact nearest_edge missing",
            },
        )
    return ds


def _zarr_encoding(ds: xr.Dataset, chunk_size: int) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        if da.dims == ("profile", "glorys_depth"):
            encoding[name] = {"chunks": (min(int(chunk_size), da.shape[0]), da.shape[1])}
        elif da.dims == ("profile",):
            encoding[name] = {"chunks": (min(int(chunk_size), da.shape[0]),)}
    return encoding


def _write_batch(
    batch: dict[str, list[Any]],
    *,
    output_zarr: Path,
    profile_start: int,
    glorys_depths: np.ndarray,
    chunk_size: int,
    first_write: bool,
    export_metadata: dict[str, Any],
) -> int:
    if not batch["profile_idx"]:
        return 0
    ds = _batch_to_dataset(
        batch,
        profile_start=profile_start,
        glorys_depths=glorys_depths,
    )
    ds = _apply_output_metadata(ds, export_metadata=export_metadata)
    if first_write:
        ds.to_zarr(
            output_zarr,
            mode="w",
            encoding=_zarr_encoding(ds, chunk_size),
            zarr_format=2,
        )
    else:
        ds.to_zarr(output_zarr, mode="a", append_dim="profile", zarr_format=2)
    return int(ds.sizes["profile"])


def export_enriched_argo_profiles(
    *,
    argo_dir: Path,
    glorys_dir: Path,
    ostia_dir: Path,
    sealevel_dir: Path,
    output_zarr: Path,
    start_date: int | None = None,
    end_date: int | None = None,
    batch_size: int = 2048,
    cache_size: int = 8,
    overwrite: bool = False,
    max_profiles: int | None = None,
) -> Path:
    output_zarr = Path(output_zarr)
    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"Output Zarr already exists: {output_zarr}")
        shutil.rmtree(output_zarr)

    glorys_index = scan_timed_files(glorys_dir, show_progress=True)
    ostia_index = scan_timed_files(ostia_dir, show_progress=True)
    sealevel_index = scan_timed_files(sealevel_dir, show_progress=True)
    glorys_depths = _load_glorys_depths(glorys_index)
    argo_files = _filter_argo_files_by_date_range(
        sorted(Path(argo_dir).glob("EN.4.2.2.f.profiles.g10.*.nc")),
        start_date=start_date,
        end_date=end_date,
    )
    if not argo_files:
        raise RuntimeError(f"No ARGO/EN4 NetCDF files found in: {argo_dir}")
    export_metadata = _build_export_metadata(
        argo_files=argo_files,
        glorys_index=glorys_index,
        ostia_index=ostia_index,
        sealevel_index=sealevel_index,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        max_profiles=max_profiles,
    )

    cache = DatasetCache(max_open=cache_size)
    written = 0
    first_write = True
    batch = _empty_batch()
    try:
        profile_total = int(max_profiles) if max_profiles is not None else None
        file_progress = tqdm(
            argo_files,
            desc="ARGO source months",
            unit="month",
            dynamic_ncols=True,
        )
        profile_progress = tqdm(
            total=profile_total,
            desc="Profiles collocated",
            unit="profile",
            dynamic_ncols=True,
        )
        with file_progress, profile_progress:
            for argo_path in file_progress:
                file_progress.set_postfix_str(argo_path.name, refresh=False)
                with _open_argo_dataset(argo_path) as ds:
                    required = ("JULD", "LATITUDE", "LONGITUDE", ARGO_DEPTH_VAR) + ARGO_PROFILE_VARS
                    missing = [name for name in required if name not in ds]
                    if missing:
                        raise RuntimeError(f"ARGO file {argo_path} is missing variables: {missing}")

                    juld = np.asarray(ds["JULD"].values, dtype=np.float64)
                    dates = _juld_to_yyyymmdd(juld)
                    lat = np.asarray(ds["LATITUDE"].values, dtype=np.float64)
                    lon = np.asarray(ds["LONGITUDE"].values, dtype=np.float64)
                    depth = np.asarray(ds[ARGO_DEPTH_VAR].values, dtype=np.float32)
                    temp = np.asarray(ds["TEMP"].values, dtype=np.float32)
                    potm = np.asarray(ds["POTM_CORRECTED"].values, dtype=np.float32)
                    psal = np.asarray(ds["PSAL_CORRECTED"].values, dtype=np.float32)
                    level_qc_arrays = {
                        name: np.asarray(ds[source_name].values)
                        if source_name in ds
                        else None
                        for name, source_name in ARGO_LEVEL_QC_VARS.items()
                    }
                    profile_qc_arrays = {
                        name: np.asarray(ds[source_name].values)
                        if source_name in ds
                        else None
                        for name, source_name in ARGO_PROFILE_QC_VARS.items()
                    }

                    for profile_idx in range(int(juld.size)):
                        date = int(dates[profile_idx])
                        if date <= 0:
                            continue
                        if start_date is not None and date < int(start_date):
                            continue
                        if end_date is not None and date > int(end_date):
                            continue
                        if not (np.isfinite(lat[profile_idx]) and np.isfinite(lon[profile_idx])):
                            continue

                        _append_profile_to_batch(
                            batch,
                            argo_path=argo_path,
                            profile_idx=profile_idx,
                            profile_date=date,
                            profile_juld=float(juld[profile_idx]),
                            lat=float(lat[profile_idx]),
                            lon=float(lon[profile_idx]),
                            depth=depth[profile_idx],
                            temp=temp[profile_idx],
                            potm=potm[profile_idx],
                            psal=psal[profile_idx],
                            argo_level_qc={
                                name: values[profile_idx]
                                if values is not None
                                else None
                                for name, values in level_qc_arrays.items()
                            },
                            argo_profile_qc={
                                name: _qc_scalar_to_int(values[profile_idx])
                                if values is not None
                                else MISSING_QC_FLAG
                                for name, values in profile_qc_arrays.items()
                            },
                            glorys_depths=glorys_depths,
                            glorys_index=glorys_index,
                            ostia_index=ostia_index,
                            sealevel_index=sealevel_index,
                            cache=cache,
                        )
                        profile_progress.update(1)
                        if profile_progress.n == 1 or profile_progress.n % 100 == 0:
                            profile_progress.set_postfix(
                                written=written,
                                queued=len(batch["profile_idx"]),
                                refresh=False,
                            )

                        reached_profile_cap = (
                            max_profiles is not None
                            and written + len(batch["profile_idx"]) >= int(max_profiles)
                        )
                        if len(batch["profile_idx"]) >= int(batch_size) or reached_profile_cap:
                            count = _write_batch(
                                batch,
                                output_zarr=output_zarr,
                                profile_start=written,
                                glorys_depths=glorys_depths,
                                chunk_size=batch_size,
                                first_write=first_write,
                                export_metadata=export_metadata,
                            )
                            written += count
                            first_write = False
                            batch = _empty_batch()
                            profile_progress.set_postfix(
                                written=written,
                                queued=0,
                                refresh=False,
                            )
                            if reached_profile_cap:
                                file_progress.update(1)
                                return output_zarr

                    if max_profiles is not None and written >= int(max_profiles):
                        file_progress.update(1)
                        return output_zarr

            if batch["profile_idx"]:
                count = _write_batch(
                    batch,
                    output_zarr=output_zarr,
                    profile_start=written,
                    glorys_depths=glorys_depths,
                    chunk_size=batch_size,
                    first_write=first_write,
                    export_metadata=export_metadata,
                )
                written += count
                profile_progress.set_postfix(written=written, queued=0, refresh=False)
    finally:
        cache.close()

    return output_zarr


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ARGO profiles enriched with collocated GLORYS, OSTIA, and sea-level fields."
    )
    parser.add_argument("--argo-dir", type=Path, default=Path("/data1/datasets/depth_v2/en4_profiles"))
    parser.add_argument("--glorys-dir", type=Path, default=Path("/data1/datasets/depth_v2/glorys_weekly"))
    parser.add_argument("--ostia-dir", type=Path, default=Path("/data1/datasets/depth_v2/ostia"))
    parser.add_argument("--sealevel-dir", type=Path, default=Path("/data1/datasets/depth_v2/sealevel_daily"))
    parser.add_argument("--output-zarr", type=Path, default=Path("/data1/datasets/depth_v2/enriched_argo_profiles.zarr"))
    parser.add_argument("--start-date", type=int, default=None, help="Optional YYYYMMDD inclusive start.")
    parser.add_argument("--end-date", type=int, default=None, help="Optional YYYYMMDD inclusive end.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--max-profiles", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out = export_enriched_argo_profiles(
        argo_dir=args.argo_dir,
        glorys_dir=args.glorys_dir,
        ostia_dir=args.ostia_dir,
        sealevel_dir=args.sealevel_dir,
        output_zarr=args.output_zarr,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        overwrite=args.overwrite,
        max_profiles=args.max_profiles,
    )
    print(f"Wrote enriched ARGO profile Zarr: {out}")


if __name__ == "__main__":
    main()
