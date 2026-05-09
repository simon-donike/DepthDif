# Example:
# /work/envs/depth/bin/python data/dataset_creation/export_dataset_zarr/export_dataset_zarr.py --argo-dir /data1/datasets/depth_v2/en4_profiles --glorys-dir /data1/datasets/depth_v2/glorys --ostia-dir /data1/datasets/depth_v2/ostia --sealevel-dir /data1/datasets/depth_v2/sealevel_daily --output-dir /data1/datasets/depth_v2/zarr_training --source-variable-config data/dataset_creation/export_dataset_zarr/source_variables.yaml --start-date 20100101 --end-date 20240731 --target-resolution-deg 0.1 --ostia-vars analysed_sst mask --argo-vars TEMP PSAL_CORRECTED --argo-depth-var DEPH_CORRECTED --glorys-vars thetao so zos --sealevel-vars adt --chunk-time 16 --chunk-profile 20000 --chunk-lat 256 --chunk-lon 256 --overwrite
"""Export a compact ML-facing zarr dataset from the raw NetCDF source folders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
import yaml

from data.dataset_creation.export_aligned_argo.source_files import (
    filter_argo_files_by_date_range,
    TimedFile,
    scan_timed_files,
)

SOURCE_VARIABLE_CONFIG_PATH = Path(__file__).with_name("source_variables.yaml")
DEFAULT_TARGET_RESOLUTION_DEG = 0.1


@dataclass(frozen=True)
class ZarrSourceVariables:
    ostia_vars: tuple[str, ...]
    argo_vars: tuple[str, ...]
    argo_depth_var: str
    glorys_vars: tuple[str, ...]
    sealevel_vars: tuple[str, ...]


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"zarr source variable config is missing section: {name}")
    return value


def _string_value(section: dict[str, Any], key: str, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"zarr source variable config value must be a string: {path}")
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


def _date_int_from_days_since_1950(day_value: float) -> int:
    day = np.datetime64("1950-01-01", "D") + np.timedelta64(
        int(round(float(day_value))), "D"
    )
    return int(np.datetime_as_string(day, unit="D").replace("-", ""))


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
    return bool(np.issubdtype(da.dtype, np.integer) or np.issubdtype(da.dtype, np.bool_))


def _resample_horizontal_grid(
    ds: xr.Dataset,
    *,
    target_resolution_deg: float | None,
) -> xr.Dataset:
    if target_resolution_deg is None:
        return ds

    coord_names = _horizontal_coord_names(ds)
    if coord_names is None:
        return ds
    lat_name, lon_name = coord_names
    target_coords = {
        lat_name: _target_axis(ds[lat_name].values, float(target_resolution_deg)),
        lon_name: _target_axis(ds[lon_name].values, float(target_resolution_deg)),
    }

    data_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if lat_name not in da.dims or lon_name not in da.dims:
            data_vars[name] = da
            continue
        method = "nearest" if _is_categorical_raster(str(name), da) else "linear"
        resampled = da.interp(target_coords, method=method)
        # Categorical rasters such as OSTIA mask must remain encoded labels, not floats.
        if method == "nearest":
            resampled = resampled.astype(da.dtype, copy=False)
        resampled.attrs.update(da.attrs)
        data_vars[name] = resampled

    out = xr.Dataset(data_vars, attrs=dict(ds.attrs))
    for coord_name in (lat_name, lon_name):
        out[coord_name].attrs.update(ds[coord_name].attrs)
    out.attrs["target_resolution_deg"] = float(target_resolution_deg)
    return out


def _open_time_series_dataset(
    timed_files: Sequence[TimedFile],
    *,
    variable_names: Sequence[str],
    target_resolution_deg: float | None,
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
        parallel=False,
    )
    ds = ds.assign_coords(time=np.asarray(dates, dtype=np.int32))
    ds["time"].attrs["description"] = "Source date encoded as YYYYMMDD."
    ds = _resample_horizontal_grid(
        ds,
        target_resolution_deg=target_resolution_deg,
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
        parallel=False,
    )
    chunk_map = {"N_PROF": int(chunk_profile)}
    if "N_LEVELS" in ds.dims:
        chunk_map["N_LEVELS"] = -1
    ds = ds.chunk({name: size for name, size in chunk_map.items() if name in ds.dims})
    return _add_argo_profile_helpers(ds, depth_var_name=depth_var_name)


def _add_argo_profile_helpers(ds: xr.Dataset, *, depth_var_name: str) -> xr.Dataset:
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

    depth = ds.get(str(depth_var_name))
    if depth is not None:
        finite_depth = np.isfinite(depth)
        for name in ("TEMP", "PSAL_CORRECTED"):
            if name not in ds:
                continue
            # The profile-level flags let loaders build row metadata without scanning
            # full profile matrices every time a training job starts.
            valid = np.asarray(
                (np.isfinite(ds[name]) & finite_depth).any(dim="N_LEVELS")
            )
            ds[f"HAS_VALID_{name}"] = (("N_PROF",), valid.astype(bool))
    return ds


def _write_dataset(ds: xr.Dataset, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output zarr already exists: {path}")
    ds.to_zarr(
        path,
        mode="w",
        consolidated=True,
        zarr_format=2,
    )
    ds.close()


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
    chunk_time: int = 16,
    chunk_profile: int = 20000,
    chunk_lat: int = 256,
    chunk_lon: int = 256,
    overwrite: bool = False,
) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

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

    _write_dataset(
        _open_time_series_dataset(
            ostia_files,
            variable_names=ostia_vars,
            target_resolution_deg=target_resolution_deg,
            chunk_time=chunk_time,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
        ),
        output_root / "ostia.zarr",
        overwrite=overwrite,
    )
    _write_dataset(
        _open_time_series_dataset(
            glorys_files,
            variable_names=glorys_vars,
            target_resolution_deg=target_resolution_deg,
            chunk_time=chunk_time,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
        ),
        output_root / "glorys.zarr",
        overwrite=overwrite,
    )
    _write_dataset(
        _open_argo_dataset(
            argo_files,
            variable_names=argo_vars,
            depth_var_name=argo_depth_var,
            chunk_profile=chunk_profile,
        ),
        output_root / "argo.zarr",
        overwrite=overwrite,
    )

    wrote_sealevel = False
    if sealevel_dir is not None and str(sealevel_dir).strip() != "":
        sealevel_files = _filter_timed_files(
            scan_timed_files(Path(sealevel_dir), show_progress=True),
            start_date=start_date,
            end_date=end_date,
        )
        if sealevel_files:
            _write_dataset(
                _open_time_series_dataset(
                    sealevel_files,
                    variable_names=sealevel_vars,
                    target_resolution_deg=target_resolution_deg,
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
        "groups": {
            "ostia": {"path": "ostia.zarr", "variables": list(ostia_vars)},
            "glorys": {"path": "glorys.zarr", "variables": list(glorys_vars)},
            "argo": {
                "path": "argo.zarr",
                "variables": list(argo_vars),
                "depth_var": str(argo_depth_var),
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
    parser.add_argument("--ostia-vars", nargs="+", default=None)
    parser.add_argument("--argo-vars", nargs="+", default=None)
    parser.add_argument("--argo-depth-var", default=None)
    parser.add_argument("--glorys-vars", nargs="+", default=None)
    parser.add_argument("--sealevel-vars", nargs="+", default=None)
    parser.add_argument("--chunk-time", type=int, default=16)
    parser.add_argument("--chunk-profile", type=int, default=20000)
    parser.add_argument("--chunk-lat", type=int, default=256)
    parser.add_argument("--chunk-lon", type=int, default=256)
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
        chunk_time=args.chunk_time,
        chunk_profile=args.chunk_profile,
        chunk_lat=args.chunk_lat,
        chunk_lon=args.chunk_lon,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
