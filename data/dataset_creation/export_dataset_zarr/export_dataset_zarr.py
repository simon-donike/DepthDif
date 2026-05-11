# Example:
# /work/envs/depth/bin/python data/dataset_creation/export_dataset_zarr/export_dataset_zarr.py --argo-dir /data1/datasets/depth_v2/en4_profiles --glorys-dir /data1/datasets/depth_v2/glorys --ostia-dir /data1/datasets/depth_v2/ostia --sealevel-dir /data1/datasets/depth_v2/sealevel_daily --output-dir /data1/datasets/depth_v2/zarr_training --source-variable-config data/dataset_creation/export_dataset_zarr/source_variables.yaml --start-date 20100101 --end-date 20240731 --target-resolution-deg 0.1 --surface-aggregate-days 7 --ostia-vars analysed_sst mask --argo-vars TEMP PSAL_CORRECTED --argo-depth-var DEPH_CORRECTED --glorys-vars thetao so zos --sealevel-vars adt --chunk-time 1 --chunk-profile 20000 --chunk-lat 256 --chunk-lon 256 --dask-scheduler threads --dask-num-workers 8 --overwrite
"""Export a compact ML-facing zarr dataset from the raw NetCDF source folders."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import dask
import numpy as np
from numcodecs import Blosc
import xarray as xr
import yaml

from data.dataset_argo_netcdf_gridded import _align_argo_profile_to_glorys_depths
from data.dataset_creation.export_aligned_argo.source_files import (
    filter_argo_files_by_date_range,
    TimedFile,
    scan_timed_files,
)

SOURCE_VARIABLE_CONFIG_PATH = Path(__file__).with_name("source_variables.yaml")
DEFAULT_TARGET_RESOLUTION_DEG = 0.1
DEFAULT_SURFACE_AGGREGATE_DAYS = 7
DEFAULT_DASK_SCHEDULER = None
PACKED_FILL_VALUE = -32768
MASK_FILL_VALUE = -128
ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)
PACKED_SCALE_FACTORS = {
    "analysed_sst": 0.01,
    "thetao": 0.01,
    "TEMP": 0.01,
    "POTM_CORRECTED": 0.01,
    "so": 0.01,
    "PSAL_CORRECTED": 0.01,
    "zos": 0.001,
    "adt": 0.001,
    "sla": 0.001,
}


@dataclass(frozen=True)
class ZarrSourceVariables:
    ostia_vars: tuple[str, ...]
    argo_vars: tuple[str, ...]
    argo_depth_var: str
    glorys_vars: tuple[str, ...]
    sealevel_vars: tuple[str, ...]


@dataclass(frozen=True)
class RasterTargetGrid:
    latitude: np.ndarray
    longitude: np.ndarray


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"zarr source variable config is missing section: {name}")
    return value


def _string_value(section: dict[str, Any], key: str, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise RuntimeError(
            f"zarr source variable config value must be a string: {path}"
        )
    return value


def _string_tuple(section: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(
            f"zarr source variable config value must be a string list: {path}"
        )
    return tuple(value)


def load_zarr_source_variables(
    path: Path = SOURCE_VARIABLE_CONFIG_PATH,
) -> ZarrSourceVariables:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"zarr source variable config must be a mapping: {path}")

    argo_section = _section(payload, "argo")
    return ZarrSourceVariables(
        ostia_vars=_string_tuple(_section(payload, "ostia"), "vars", "ostia.vars"),
        argo_vars=_string_tuple(argo_section, "vars", "argo.vars"),
        argo_depth_var=_string_value(argo_section, "depth_var", "argo.depth_var"),
        glorys_vars=_string_tuple(_section(payload, "glorys"), "vars", "glorys.vars"),
        sealevel_vars=_string_tuple(
            _section(payload, "sealevel"),
            "vars",
            "sealevel.vars",
        ),
    )


_DEFAULT_SOURCE_VARIABLES = load_zarr_source_variables()
DEFAULT_OSTIA_VARS = _DEFAULT_SOURCE_VARIABLES.ostia_vars
DEFAULT_ARGO_VARS = _DEFAULT_SOURCE_VARIABLES.argo_vars
DEFAULT_ARGO_DEPTH_VAR = _DEFAULT_SOURCE_VARIABLES.argo_depth_var
DEFAULT_GLORYS_VARS = _DEFAULT_SOURCE_VARIABLES.glorys_vars
DEFAULT_SEALEVEL_VARS = _DEFAULT_SOURCE_VARIABLES.sealevel_vars


def _optional_float(text: str) -> float | None:
    normalized = str(text).strip().lower()
    if normalized in {"none", "null", "native", "off"}:
        return None
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


@contextmanager
def _timed_step(label: str):
    """Print elapsed wall time for one export phase."""
    start = time.perf_counter()
    print(f"[zarr export] {label} started", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[zarr export] {label} finished in {elapsed:.1f}s", flush=True)


@contextmanager
def _dask_settings(scheduler: str | None, num_workers: int | None):
    """Temporarily apply dask scheduler settings for this export."""
    config: dict[str, Any] = {}
    if scheduler is not None:
        config["scheduler"] = scheduler
    if num_workers is not None:
        config["num_workers"] = int(num_workers)
    if not config:
        yield
        return
    with dask.config.set(config):
        yield


def _date_int_from_days_since_1950(day_value: float) -> int:
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


def _days_since_1950_from_date_int(date_value: int) -> float:
    text = str(int(date_value))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float(
        (day - np.datetime64("1950-01-01", "D")).astype("timedelta64[D]").astype(int)
    )


def _filter_timed_files(
    paths: Sequence[TimedFile],
    *,
    start_date: int | None,
    end_date: int | None,
) -> list[TimedFile]:
    filtered: list[TimedFile] = []
    for item in paths:
        date_value = _date_int_from_days_since_1950(float(item.day))
        if start_date is not None and date_value < int(start_date):
            continue
        if end_date is not None and date_value > int(end_date):
            continue
        filtered.append(item)
    return filtered


def _horizontal_coord_names(ds: xr.Dataset) -> tuple[str, str] | None:
    lat_name = next((name for name in ("lat", "latitude") if name in ds.dims), None)
    lon_name = next((name for name in ("lon", "longitude") if name in ds.dims), None)
    if lat_name is None or lon_name is None:
        return None
    return lat_name, lon_name


def _target_axis(values: np.ndarray, resolution_deg: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = source[np.isfinite(source)]
    if finite.size < 2:
        return source

    increasing = bool(source[-1] >= source[0])
    start = float(np.nanmin(finite))
    stop = float(np.nanmax(finite))
    count = int(np.floor((stop - start) / float(resolution_deg))) + 1
    axis = start + (np.arange(max(count, 1), dtype=np.float64) * float(resolution_deg))
    axis = np.round(axis, decimals=10)
    return axis if increasing else axis[::-1]


def _is_categorical_raster(name: str, da: xr.DataArray) -> bool:
    if str(name).lower() in {"mask"}:
        return True
    return bool(
        np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.bool_)
    )


def _resample_horizontal_grid(
    ds: xr.Dataset,
    *,
    target_resolution_deg: float | None,
    target_grid: RasterTargetGrid | None,
) -> xr.Dataset:
    if target_resolution_deg is None and target_grid is None:
        return ds

    coord_names = _horizontal_coord_names(ds)
    if coord_names is None:
        return ds
    lat_name, lon_name = coord_names
    if target_grid is None:
        target_coords = {
            lat_name: _target_axis(ds[lat_name].values, float(target_resolution_deg)),
            lon_name: _target_axis(ds[lon_name].values, float(target_resolution_deg)),
        }
    else:
        # Reusing the GLORYS coordinates keeps all saved raster pixels aligned.
        target_coords = {
            lat_name: np.asarray(target_grid.latitude, dtype=np.float64),
            lon_name: np.asarray(target_grid.longitude, dtype=np.float64),
        }

    data_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if lat_name not in da.dims or lon_name not in da.dims:
            data_vars[name] = da
            continue
        method = "nearest" if _is_categorical_raster(str(name), da) else "linear"
        resampled = da.interp(
            target_coords,
            method=method,
            kwargs={"fill_value": "extrapolate"},
        )
        # Categorical rasters such as OSTIA mask must remain encoded labels, not floats.
        if method == "nearest":
            resampled = resampled.astype(da.dtype, copy=False)
        resampled.attrs.update(da.attrs)
        data_vars[name] = resampled

    out = xr.Dataset(data_vars, attrs=dict(ds.attrs))
    for coord_name in (lat_name, lon_name):
        out[coord_name].attrs.update(ds[coord_name].attrs)
    if target_resolution_deg is not None:
        out.attrs["target_resolution_deg"] = float(target_resolution_deg)
    if target_grid is not None:
        out.attrs["horizontal_grid_source"] = "glorys"
    return out


def _raster_target_grid_from_dataset(ds: xr.Dataset) -> RasterTargetGrid:
    coord_names = _horizontal_coord_names(ds)
    if coord_names is None:
        raise RuntimeError(
            "GLORYS zarr export source is missing horizontal coordinates."
        )
    lat_name, lon_name = coord_names
    return RasterTargetGrid(
        latitude=np.asarray(ds[lat_name].values, dtype=np.float64).copy(),
        longitude=np.asarray(ds[lon_name].values, dtype=np.float64).copy(),
    )


def _target_dates_from_files(target_timed_files: Sequence[TimedFile]) -> np.ndarray:
    dates = np.asarray(
        [
            _date_int_from_days_since_1950(float(item.day))
            for item in target_timed_files
        ],
        dtype=np.int32,
    )
    return np.unique(dates)


def _nearest_target_date_labels(
    source_dates: np.ndarray,
    target_dates: np.ndarray,
    *,
    aggregate_days: int,
) -> np.ndarray:
    source_days = np.asarray(
        [_days_since_1950_from_date_int(int(value)) for value in source_dates],
        dtype=np.float64,
    )
    target_days = np.asarray(
        [_days_since_1950_from_date_int(int(value)) for value in target_dates],
        dtype=np.float64,
    )
    labels = np.zeros(source_days.shape, dtype=np.int32)
    if source_days.size == 0 or target_days.size == 0:
        return labels

    radius_days = max(0.0, (float(aggregate_days) - 1.0) / 2.0)
    positions = np.searchsorted(target_days, source_days)
    right = np.clip(positions, 0, int(target_days.size) - 1)
    left = np.clip(positions - 1, 0, int(target_days.size) - 1)
    left_dist = np.abs(source_days - target_days[left])
    right_dist = np.abs(source_days - target_days[right])
    nearest = np.where(right_dist < left_dist, right, left)
    nearest_dist = np.minimum(left_dist, right_dist)
    valid = nearest_dist <= (radius_days + 1.0e-8)
    labels[valid] = target_dates[nearest[valid]]
    return labels


def _aggregate_to_target_dates(
    ds: xr.Dataset,
    *,
    target_timed_files: Sequence[TimedFile] | None,
    aggregate_days: int,
) -> xr.Dataset:
    if target_timed_files is None:
        return ds
    if "time" not in ds.dims:
        return ds

    target_dates = _target_dates_from_files(target_timed_files)
    source_dates = np.asarray(ds["time"].values, dtype=np.int32).reshape(-1)
    labels = _nearest_target_date_labels(
        source_dates,
        target_dates,
        aggregate_days=aggregate_days,
    )
    valid = labels > 0
    if not np.any(valid):
        raise RuntimeError(
            "No source raster dates fall inside the target aggregate windows."
        )

    grouped_parts: list[xr.Dataset] = []
    continuous_names = [
        name
        for name, da in ds.data_vars.items()
        if "time" in da.dims and not _is_categorical_raster(str(name), da)
    ]
    categorical_names = [
        name
        for name, da in ds.data_vars.items()
        if "time" in da.dims and _is_categorical_raster(str(name), da)
    ]
    passthrough_names = [
        name for name, da in ds.data_vars.items() if "time" not in da.dims
    ]

    if continuous_names:
        grouped = (
            ds[continuous_names]
            .isel(time=valid)
            .assign_coords(target_time=("time", labels[valid]))
            .groupby("target_time")
            .mean(dim="time", skipna=True)
            .rename({"target_time": "time"})
        )
        grouped = grouped.assign_coords(time=grouped["time"].astype(np.int32))
        for name in continuous_names:
            grouped[name].attrs.update(ds[name].attrs)
            grouped[name].attrs[
                "temporal_aggregation"
            ] = f"Centered {int(aggregate_days)}-day mean around GLORYS timestep."
        grouped_parts.append(grouped)

    if categorical_names:
        source_days = np.asarray(
            [_days_since_1950_from_date_int(int(value)) for value in source_dates],
            dtype=np.float64,
        )
        target_days = np.asarray(
            [_days_since_1950_from_date_int(int(value)) for value in target_dates],
            dtype=np.float64,
        )
        radius_days = max(0.0, (float(aggregate_days) - 1.0) / 2.0)
        nearest_indices: list[int] = []
        nearest_dates: list[int] = []
        for target_date, target_day in zip(target_dates.tolist(), target_days.tolist()):
            distances = np.abs(source_days - float(target_day))
            if (
                distances.size == 0
                or float(np.nanmin(distances)) > radius_days + 1.0e-8
            ):
                continue
            nearest_indices.append(int(np.nanargmin(distances)))
            nearest_dates.append(int(target_date))
        if nearest_indices:
            categorical = ds[categorical_names].isel(
                time=xr.DataArray(nearest_indices, dims=("time",))
            )
            categorical = categorical.assign_coords(
                time=np.asarray(nearest_dates, dtype=np.int32)
            )
            for name in categorical_names:
                categorical[name].attrs.update(ds[name].attrs)
                categorical[name].attrs["temporal_aggregation"] = (
                    f"Nearest source date within centered {int(aggregate_days)}-day "
                    "window around GLORYS timestep."
                )
            grouped_parts.append(categorical)

    if passthrough_names:
        grouped_parts.append(ds[passthrough_names])
    if not grouped_parts:
        raise RuntimeError("No raster variables remained after temporal aggregation.")

    out = xr.merge(grouped_parts, compat="override", join="outer")
    out.attrs.update(ds.attrs)
    out.attrs["temporal_aggregation_days"] = int(aggregate_days)
    out.attrs["temporal_aggregation_target"] = "glorys"
    return out


def _depth_axis_from_dataset(ds: xr.Dataset) -> np.ndarray:
    if "depth" not in ds.coords and "depth" not in ds.variables:
        raise RuntimeError("GLORYS zarr export source is missing a depth coordinate.")
    depth = np.asarray(ds["depth"].values, dtype=np.float32).reshape(-1)
    depth = depth[np.isfinite(depth)]
    if depth.size == 0:
        raise RuntimeError("GLORYS zarr export source has an empty depth coordinate.")
    return depth.astype(np.float32, copy=False)


def _open_time_series_dataset(
    timed_files: Sequence[TimedFile],
    *,
    variable_names: Sequence[str],
    target_resolution_deg: float | None,
    target_grid: RasterTargetGrid | None,
    target_timed_files: Sequence[TimedFile] | None,
    aggregate_days: int,
    chunk_time: int,
    chunk_lat: int,
    chunk_lon: int,
) -> xr.Dataset:
    if not timed_files:
        raise RuntimeError("No timed NetCDF files were selected for zarr export.")

    paths = [Path(item.path) for item in timed_files]
    dates = [_date_int_from_days_since_1950(float(item.day)) for item in timed_files]
    selected = tuple(str(name) for name in variable_names)

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        present = [name for name in selected if name in ds]
        if not present:
            source = ds.encoding.get("source", "<unknown>")
            raise RuntimeError(f"No requested variables found in source file: {source}")
        out = ds[present]
        if "time" in out.dims and int(out.sizes["time"]) == 1:
            # Each raw file is one analysis time; concat below adds the canonical time axis.
            out = out.isel(time=0, drop=True)
        return out

    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        decode_times=False,
        mask_and_scale=True,
        chunks={
            "time": int(chunk_time),
            "lat": int(chunk_lat),
            "latitude": int(chunk_lat),
            "lon": int(chunk_lon),
            "longitude": int(chunk_lon),
            "depth": -1,
        },
        parallel=False,
    )
    ds = ds.assign_coords(time=np.asarray(dates, dtype=np.int32))
    ds["time"].attrs["description"] = "Source date encoded as YYYYMMDD."
    ds = _aggregate_to_target_dates(
        ds,
        target_timed_files=target_timed_files,
        aggregate_days=aggregate_days,
    )
    ds["time"].attrs["description"] = "Source date encoded as YYYYMMDD."
    ds = _resample_horizontal_grid(
        ds,
        target_resolution_deg=target_resolution_deg,
        target_grid=target_grid,
    )
    chunk_map = {"time": int(chunk_time)}
    for lat_name in ("lat", "latitude"):
        if lat_name in ds.dims:
            chunk_map[lat_name] = int(chunk_lat)
    for lon_name in ("lon", "longitude"):
        if lon_name in ds.dims:
            chunk_map[lon_name] = int(chunk_lon)
    if "depth" in ds.dims:
        chunk_map["depth"] = -1
    return ds.chunk({name: size for name, size in chunk_map.items() if name in ds.dims})


def _open_argo_dataset(
    paths: Sequence[Path],
    *,
    variable_names: Sequence[str],
    depth_var_name: str,
    target_depths: np.ndarray,
    chunk_profile: int,
) -> xr.Dataset:
    if not paths:
        raise RuntimeError("No ARGO NetCDF files were selected for zarr export.")

    selected = ("JULD", "LATITUDE", "LONGITUDE", str(depth_var_name)) + tuple(
        str(name) for name in variable_names
    )

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        present = [name for name in selected if name in ds]
        missing = [name for name in ("JULD", "LATITUDE", "LONGITUDE") if name not in ds]
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
    projected = _project_argo_dataset_to_depths(
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
    return _add_argo_profile_helpers(
        projected,
        variable_names=variable_names,
    )


def _align_profile_values_to_depths(
    values: np.ndarray,
    depths: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    return _align_argo_profile_to_glorys_depths(
        temperature=values,
        depth=depths,
        glorys_depths=target_depths,
    )


def _project_argo_dataset_to_depths(
    ds: xr.Dataset,
    *,
    variable_names: Sequence[str],
    depth_var_name: str,
    target_depths: np.ndarray,
) -> xr.Dataset:
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
            f"{name} projected onto the GLORYS depth coordinate during Zarr export."
        )
        out[name] = projected
    return out


def _add_argo_profile_helpers(
    ds: xr.Dataset,
    *,
    variable_names: Sequence[str],
) -> xr.Dataset:
    if "JULD" in ds:
        juld = np.asarray(ds["JULD"].values, dtype=np.float64).reshape(-1)
        dates = np.zeros(juld.shape, dtype=np.int32)
        valid = np.isfinite(juld) & (juld < 90000.0) & (juld > -20000.0)
        if np.any(valid):
            days = np.datetime64("1950-01-01", "D") + np.floor(juld[valid]).astype(
                "timedelta64[D]"
            )
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
        # The profile-level flags let loaders build row metadata without scanning
        # full profile matrices every time a training job starts.
        valid = np.isfinite(da).any(dim=depth_dim)
        ds[f"HAS_VALID_{name}"] = valid.astype(bool)
    return ds


def _write_dataset(ds: xr.Dataset, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output zarr already exists: {path}")
    ds.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding=_zarr_encoding(ds),
        zarr_format=2,
    )
    ds.close()


def _zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in ds.variables.items():
        item: dict[str, Any] = {"compressor": ZARR_COMPRESSOR}
        if name in PACKED_SCALE_FACTORS:
            item.update(
                {
                    "dtype": "int16",
                    "scale_factor": float(PACKED_SCALE_FACTORS[name]),
                    "_FillValue": np.int16(PACKED_FILL_VALUE),
                }
            )
        elif str(name).lower() == "mask":
            item.update({"dtype": "int8", "_FillValue": np.int8(MASK_FILL_VALUE)})
        elif name in {
            "JULD",
            "LATITUDE",
            "LONGITUDE",
            "lat",
            "lon",
            "latitude",
            "longitude",
            "depth",
        }:
            item["dtype"] = "float32"
        elif name == "DATE":
            item["dtype"] = "int32"
        elif name == "N_PROF":
            item["dtype"] = "int32"
        elif np.issubdtype(variable.dtype, np.floating):
            item["dtype"] = "float32"
        encoding[str(name)] = item
    return encoding


def export_training_zarr_dataset(
    *,
    argo_dir: str | Path,
    glorys_dir: str | Path,
    ostia_dir: str | Path,
    sealevel_dir: str | Path | None,
    output_dir: str | Path,
    start_date: int | None = None,
    end_date: int | None = None,
    ostia_vars: Sequence[str] = DEFAULT_OSTIA_VARS,
    argo_vars: Sequence[str] = DEFAULT_ARGO_VARS,
    argo_depth_var: str = DEFAULT_ARGO_DEPTH_VAR,
    glorys_vars: Sequence[str] = DEFAULT_GLORYS_VARS,
    sealevel_vars: Sequence[str] = DEFAULT_SEALEVEL_VARS,
    target_resolution_deg: float | None = DEFAULT_TARGET_RESOLUTION_DEG,
    surface_aggregate_days: int = DEFAULT_SURFACE_AGGREGATE_DAYS,
    chunk_time: int = 1,
    chunk_profile: int = 20000,
    chunk_lat: int = 256,
    chunk_lon: int = 256,
    dask_scheduler: str | None = DEFAULT_DASK_SCHEDULER,
    dask_num_workers: int | None = None,
    overwrite: bool = False,
) -> Path:
    with _dask_settings(dask_scheduler, dask_num_workers):
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        with _timed_step("source file scan"):
            ostia_files = _filter_timed_files(
                scan_timed_files(Path(ostia_dir), show_progress=True),
                start_date=start_date,
                end_date=end_date,
            )
            glorys_files = _filter_timed_files(
                scan_timed_files(Path(glorys_dir), show_progress=True),
                start_date=start_date,
                end_date=end_date,
            )
            argo_files = filter_argo_files_by_date_range(
                sorted(Path(argo_dir).glob("*.nc")),
                start_date=start_date,
                end_date=end_date,
            )

        with _timed_step("GLORYS open, resample, and write"):
            glorys_ds = _open_time_series_dataset(
                glorys_files,
                variable_names=glorys_vars,
                target_resolution_deg=target_resolution_deg,
                target_grid=None,
                target_timed_files=None,
                aggregate_days=1,
                chunk_time=chunk_time,
                chunk_lat=chunk_lat,
                chunk_lon=chunk_lon,
            )
            raster_target_grid = _raster_target_grid_from_dataset(glorys_ds)
            glorys_depth_axis = _depth_axis_from_dataset(glorys_ds)
            _write_dataset(
                glorys_ds,
                output_root / "glorys.zarr",
                overwrite=overwrite,
            )

        with _timed_step("OSTIA open, aggregate, resample, and write"):
            ostia_ds = _open_time_series_dataset(
                ostia_files,
                variable_names=ostia_vars,
                target_resolution_deg=target_resolution_deg,
                target_grid=raster_target_grid,
                target_timed_files=glorys_files,
                aggregate_days=surface_aggregate_days,
                chunk_time=chunk_time,
                chunk_lat=chunk_lat,
                chunk_lon=chunk_lon,
            )
            _write_dataset(
                ostia_ds,
                output_root / "ostia.zarr",
                overwrite=overwrite,
            )

        with _timed_step("ARGO open, project, and write"):
            _write_dataset(
                _open_argo_dataset(
                    argo_files,
                    variable_names=argo_vars,
                    depth_var_name=argo_depth_var,
                    target_depths=glorys_depth_axis,
                    chunk_profile=chunk_profile,
                ),
                output_root / "argo.zarr",
                overwrite=overwrite,
            )

        wrote_sealevel = False
        if sealevel_dir is not None and str(sealevel_dir).strip() != "":
            with _timed_step("sea-level file scan"):
                sealevel_files = _filter_timed_files(
                    scan_timed_files(Path(sealevel_dir), show_progress=True),
                    start_date=start_date,
                    end_date=end_date,
                )
            if sealevel_files:
                with _timed_step("sea-level open, aggregate, resample, and write"):
                    _write_dataset(
                        _open_time_series_dataset(
                            sealevel_files,
                            variable_names=sealevel_vars,
                            target_resolution_deg=target_resolution_deg,
                            target_grid=raster_target_grid,
                            target_timed_files=glorys_files,
                            aggregate_days=surface_aggregate_days,
                            chunk_time=chunk_time,
                            chunk_lat=chunk_lat,
                            chunk_lon=chunk_lon,
                        ),
                        output_root / "sealevel.zarr",
                        overwrite=overwrite,
                    )
                wrote_sealevel = True

        manifest = {
            "format": "depthdif_training_zarr",
            "version": 1,
            "date_range": {"start_date": start_date, "end_date": end_date},
            "raster_target_resolution_deg": target_resolution_deg,
            "raster_target_grid": "glorys",
            "surface_temporal_aggregation": {
                "target": "glorys",
                "window_days": int(surface_aggregate_days),
            },
            "dask": {
                "scheduler": dask_scheduler,
                "num_workers": dask_num_workers,
            },
            "groups": {
                "ostia": {"path": "ostia.zarr", "variables": list(ostia_vars)},
                "glorys": {"path": "glorys.zarr", "variables": list(glorys_vars)},
                "argo": {
                    "path": "argo.zarr",
                    "variables": list(argo_vars),
                    "source_depth_var": str(argo_depth_var),
                    "depth_axis": "depth",
                    "projected_to_glorys_depth": True,
                },
            },
        }
        if wrote_sealevel:
            manifest["groups"]["sealevel"] = {
                "path": "sealevel.zarr",
                "variables": list(sealevel_vars),
            }
        with (output_root / "manifest.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact DepthDif zarr sources."
    )
    parser.add_argument("--argo-dir", type=Path, required=True)
    parser.add_argument("--glorys-dir", type=Path, required=True)
    parser.add_argument("--ostia-dir", type=Path, required=True)
    parser.add_argument("--sealevel-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-variable-config",
        type=Path,
        default=SOURCE_VARIABLE_CONFIG_PATH,
    )
    parser.add_argument("--start-date", type=int, default=None)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument(
        "--target-resolution-deg",
        type=_optional_float,
        default=DEFAULT_TARGET_RESOLUTION_DEG,
    )
    parser.add_argument(
        "--surface-aggregate-days",
        type=_positive_int,
        default=DEFAULT_SURFACE_AGGREGATE_DAYS,
    )
    parser.add_argument("--ostia-vars", nargs="+", default=None)
    parser.add_argument("--argo-vars", nargs="+", default=None)
    parser.add_argument("--argo-depth-var", default=None)
    parser.add_argument("--glorys-vars", nargs="+", default=None)
    parser.add_argument("--sealevel-vars", nargs="+", default=None)
    parser.add_argument("--chunk-time", type=int, default=1)
    parser.add_argument("--chunk-profile", type=int, default=20000)
    parser.add_argument("--chunk-lat", type=int, default=256)
    parser.add_argument("--chunk-lon", type=int, default=256)
    parser.add_argument(
        "--dask-scheduler",
        choices=("threads", "processes", "single-threaded", "synchronous"),
        default=DEFAULT_DASK_SCHEDULER,
    )
    parser.add_argument("--dask-num-workers", type=_positive_int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_variables = load_zarr_source_variables(args.source_variable_config)
    export_training_zarr_dataset(
        argo_dir=args.argo_dir,
        glorys_dir=args.glorys_dir,
        ostia_dir=args.ostia_dir,
        sealevel_dir=args.sealevel_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        ostia_vars=args.ostia_vars or source_variables.ostia_vars,
        argo_vars=args.argo_vars or source_variables.argo_vars,
        argo_depth_var=args.argo_depth_var or source_variables.argo_depth_var,
        glorys_vars=args.glorys_vars or source_variables.glorys_vars,
        sealevel_vars=args.sealevel_vars or source_variables.sealevel_vars,
        target_resolution_deg=args.target_resolution_deg,
        surface_aggregate_days=args.surface_aggregate_days,
        chunk_time=args.chunk_time,
        chunk_profile=args.chunk_profile,
        chunk_lat=args.chunk_lat,
        chunk_lon=args.chunk_lon,
        dask_scheduler=args.dask_scheduler,
        dask_num_workers=args.dask_num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
