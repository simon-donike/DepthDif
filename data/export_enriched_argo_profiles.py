from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ostia_argo import OstiaArgoTileDataset


ARGO_PROFILE_VARS = ("TEMP", "POTM_CORRECTED", "PSAL_CORRECTED")
ARGO_DEPTH_VAR = "DEPH_CORRECTED"
GLORYS_3D_VARS = ("thetao", "so", "uo", "vo")
GLORYS_2D_VARS = ("zos", "mlotst", "bottomT", "sithick", "siconc", "usi", "vsi")
OSTIA_VARS = ("analysed_sst", "analysis_error", "sea_ice_fraction", "mask")
SEALEVEL_VARS = (
    "sla",
    "err_sla",
    "ugosa",
    "err_ugosa",
    "vgosa",
    "err_vgosa",
    "adt",
    "ugos",
    "vgos",
    "flag_ice",
    "tpa_correction",
)
CATEGORICAL_VARS = {"mask", "flag_ice"}

MISSING_STATUS = np.int8(2)
INTERPOLATED_STATUS = np.int8(0)
NEAREST_EDGE_STATUS = np.int8(1)


@dataclass(frozen=True)
class TimedFile:
    path: Path
    day: float


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


def _date_to_days_since_1950(date_yyyymmdd: int) -> float:
    text = str(int(date_yyyymmdd))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float((day - np.datetime64("1950-01-01", "D")).astype("timedelta64[D]").astype(int))


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


def _parse_first_date(path: Path) -> int | None:
    match = re.search(r"(\d{8})", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _time_day_from_file(path: Path) -> float:
    parsed = _parse_first_date(path)
    if parsed is not None:
        # Source filenames encode the valid observation/model date as the first YYYYMMDD.
        # Reading NetCDF time is kept as a fallback, but opening thousands of files just
        # for indexing makes startup unnecessarily expensive on the full raw archive.
        return _date_to_days_since_1950(parsed)
    try:
        with xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
            mask_and_scale=False,
            cache=False,
        ) as ds:
            if "time" in ds:
                values = np.asarray(ds["time"].values, dtype=np.float64).reshape(-1)
                if values.size > 0 and np.isfinite(values[0]):
                    units = str(ds["time"].attrs.get("units", "")).lower()
                    if "hours since 1950-01-01" in units:
                        return float(values[0] / 24.0)
                    if "days since 1950-01-01" in units:
                        return float(values[0])
    except Exception:
        pass
    raise RuntimeError(f"Could not determine date for source file: {path}")


def scan_timed_files(root: Path, pattern: str = "*.nc") -> list[TimedFile]:
    files = sorted(Path(root).glob(pattern))
    out: list[TimedFile] = []
    for path in files:
        try:
            out.append(TimedFile(path=path, day=_time_day_from_file(path)))
        except Exception:
            # Download directories may contain partial or unrelated files; skip unreadable inputs.
            continue
    out.sort(key=lambda item: item.day)
    return out


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
    return OstiaArgoTileDataset._align_argo_profile_to_glorys_depths(
        temperature=np.asarray(values, dtype=np.float32),
        depth=np.asarray(depths, dtype=np.float32),
        glorys_depths=np.asarray(glorys_depths, dtype=np.float32),
    )


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


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().as_posix()
    except Exception:
        return path.as_posix()


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

    batch["profile_source_file"].append(_relative_or_absolute(argo_path))
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
    for name in GLORYS_3D_VARS:
        depth_vars.add(f"glorys_{name}")

    for key, values in batch.items():
        if key == "profile_source_file":
            data_vars[key] = (("profile",), np.asarray(values, dtype=str))
        elif key in depth_vars:
            arr = np.asarray(values)
            if arr.dtype == bool:
                arr = arr.astype(bool, copy=False)
            else:
                arr = arr.astype(np.float32, copy=False)
            data_vars[key] = (("profile", "glorys_depth"), arr)
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
        attrs={
            "description": "ARGO profiles enriched with freshly collocated GLORYS, OSTIA, and sea-level fields.",
            "temporal_status_values": "0=interpolated_or_exact, 1=nearest_edge, 2=missing",
            "primary_external_sea_surface_height": "sealevel_adt",
        },
    )


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
) -> int:
    if not batch["profile_idx"]:
        return 0
    ds = _batch_to_dataset(
        batch,
        profile_start=profile_start,
        glorys_depths=glorys_depths,
    )
    if first_write:
        ds.to_zarr(output_zarr, mode="w", encoding=_zarr_encoding(ds, chunk_size))
    else:
        ds.to_zarr(output_zarr, mode="a", append_dim="profile")
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

    glorys_index = scan_timed_files(glorys_dir)
    ostia_index = scan_timed_files(ostia_dir)
    sealevel_index = scan_timed_files(sealevel_dir)
    glorys_depths = _load_glorys_depths(glorys_index)
    argo_files = sorted(Path(argo_dir).glob("EN.4.2.2.f.profiles.g10.*.nc"))
    if not argo_files:
        raise RuntimeError(f"No ARGO/EN4 NetCDF files found in: {argo_dir}")

    cache = DatasetCache(max_open=cache_size)
    written = 0
    first_write = True
    batch = _empty_batch()
    try:
        for argo_path in tqdm(argo_files, desc="Exporting enriched ARGO profiles"):
            with xr.open_dataset(
                argo_path,
                engine="h5netcdf",
                decode_times=False,
                mask_and_scale=True,
                cache=False,
            ) as ds:
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
                        glorys_depths=glorys_depths,
                        glorys_index=glorys_index,
                        ostia_index=ostia_index,
                        sealevel_index=sealevel_index,
                        cache=cache,
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
                        )
                        written += count
                        first_write = False
                        batch = _empty_batch()
                        if reached_profile_cap:
                            return output_zarr

                if max_profiles is not None and written >= int(max_profiles):
                    return output_zarr

        if batch["profile_idx"]:
            _write_batch(
                batch,
                output_zarr=output_zarr,
                profile_start=written,
                glorys_depths=glorys_depths,
                chunk_size=batch_size,
                first_write=first_write,
            )
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
