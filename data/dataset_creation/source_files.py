from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm


SOURCE_VARIABLE_CONFIG_PATH = Path(__file__).with_name("source_variables.yaml")


def load_source_variable_config(
    path: Path = SOURCE_VARIABLE_CONFIG_PATH,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"source variable config must be a mapping: {path}")
    return payload


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"source variable config is missing section: {name}")
    return value


def _string_value(section: dict[str, Any], key: str, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"source variable config value must be a string: {path}")
    return value


def _string_tuple(section: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"source variable config value must be a string list: {path}")
    return tuple(value)


def _string_mapping(section: dict[str, Any], key: str, path: str) -> dict[str, str]:
    value = section.get(key, {})
    if not isinstance(value, dict) or not all(
        isinstance(item_key, str) and isinstance(item_value, str)
        for item_key, item_value in value.items()
    ):
        raise RuntimeError(f"source variable config value must be a string mapping: {path}")
    return dict(value)


_SOURCE_VARIABLE_CONFIG = load_source_variable_config()
_ARGO_SECTION = _section(_SOURCE_VARIABLE_CONFIG, "argo")
_GLORYS_SECTION = _section(_SOURCE_VARIABLE_CONFIG, "glorys")
_OSTIA_SECTION = _section(_SOURCE_VARIABLE_CONFIG, "ostia")
_SEALEVEL_SECTION = _section(_SOURCE_VARIABLE_CONFIG, "sealevel")

# Keep configured variable groups as tuples because downstream code concatenates
# them into fixed ordered variable lists during validation and export.
ARGO_PROFILE_VARS = _string_tuple(_ARGO_SECTION, "profile_vars", "argo.profile_vars")
ARGO_DEPTH_VAR = _string_value(_ARGO_SECTION, "depth_var", "argo.depth_var")
ARGO_LEVEL_QC_VARS = _string_mapping(
    _ARGO_SECTION,
    "level_qc_vars",
    "argo.level_qc_vars",
)
ARGO_PROFILE_QC_VARS = _string_mapping(
    _ARGO_SECTION,
    "profile_qc_vars",
    "argo.profile_qc_vars",
)
GLORYS_3D_VARS = _string_tuple(_GLORYS_SECTION, "vars_3d", "glorys.vars_3d")
GLORYS_2D_VARS = _string_tuple(_GLORYS_SECTION, "vars_2d", "glorys.vars_2d")
OSTIA_VARS = _string_tuple(_OSTIA_SECTION, "vars", "ostia.vars")
SEALEVEL_VARS = _string_tuple(_SEALEVEL_SECTION, "vars", "sealevel.vars")
SOURCE_VARIABLES = {
    "argo": ("JULD", "LATITUDE", "LONGITUDE", ARGO_DEPTH_VAR) + ARGO_PROFILE_VARS,
    "glorys": GLORYS_3D_VARS + GLORYS_2D_VARS,
    "ostia": OSTIA_VARS,
    "sealevel": SEALEVEL_VARS,
}


@dataclass(frozen=True)
class TimedFile:
    path: Path
    day: float


def date_to_days_since_1950(date_yyyymmdd: int) -> float:
    text = str(int(date_yyyymmdd))
    day = np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}", "D")
    return float(
        (day - np.datetime64("1950-01-01", "D"))
        .astype("timedelta64[D]")
        .astype(int)
    )


def _parse_first_date(path: Path) -> int | None:
    match = re.search(r"(\d{8})", path.name)
    if match is None:
        return None
    return int(match.group(1))


def parse_argo_file_month(path: Path) -> int | None:
    match = re.search(r"\.(\d{6})\.nc$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def filter_argo_files_by_date_range(
    argo_files: list[Path],
    *,
    start_date: int | None,
    end_date: int | None,
) -> list[Path]:
    start_month = int(str(int(start_date))[:6]) if start_date is not None else None
    end_month = int(str(int(end_date))[:6]) if end_date is not None else None
    if start_month is None and end_month is None:
        return argo_files

    filtered: list[Path] = []
    for path in argo_files:
        month = parse_argo_file_month(path)
        if month is None:
            filtered.append(path)
            continue
        # Profile-level day filtering still happens after opening the matching month.
        if start_month is not None and month < start_month:
            continue
        if end_month is not None and month > end_month:
            continue
        filtered.append(path)
    return filtered


def open_argo_dataset(path: Path) -> xr.Dataset:
    # EN4 archives can mix NetCDF4/HDF5 and NetCDF3 months.
    # Xarray backend autodetection picks the usable reader.
    return xr.open_dataset(
        path,
        decode_times=False,
        mask_and_scale=True,
        cache=False,
    )


def _time_day_from_file(path: Path) -> float:
    parsed = _parse_first_date(path)
    if parsed is not None:
        # Source filenames encode the valid observation/model date as the first YYYYMMDD.
        # Reading NetCDF time is kept as a fallback, but opening thousands of files just
        # for indexing makes startup unnecessarily expensive on the full raw archive.
        return date_to_days_since_1950(parsed)
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


def scan_timed_files(
    root: Path,
    pattern: str = "*.nc",
    *,
    show_progress: bool = False,
) -> list[TimedFile]:
    files = sorted(Path(root).glob(pattern))
    out: list[TimedFile] = []
    iterator = tqdm(
        files,
        desc=f"Scanning {Path(root).name}",
        unit="file",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for path in iterator:
        try:
            out.append(TimedFile(path=path, day=_time_day_from_file(path)))
        except Exception:
            # Download directories may contain partial or unrelated files; skip unreadable inputs.
            continue
    out.sort(key=lambda item: item.day)
    return out
