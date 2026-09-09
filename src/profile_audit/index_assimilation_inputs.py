from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
import xarray as xr

from profile_audit.acquire_assimilation_inputs import _is_tar_member, _is_xbt_member, _member_year, _target_days
from profile_audit.common import decode_char_array, first_existing, normalize_longitude_180, read_table, valid_coordinates


OBSERVATION_COLUMNS = [
    "profile_id", "level_index", "depth_m", "temperature_c", "salinity", "temperature_qc", "salinity_qc",
    "source_vertical_variable",
]


def pressure_dbar_to_depth_m(pressure: np.ndarray, latitude: float) -> np.ndarray:
    """Convert sea pressure to positive depth using the UNESCO 1983 formula."""
    p = np.asarray(pressure, dtype=np.float64)
    sine_squared = np.sin(np.radians(float(latitude))) ** 2
    gravity = 9.780318 * (1 + 5.2788e-3 * sine_squared + 2.36e-5 * sine_squared**2) + 1.092e-6 * p
    geopotential = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return (geopotential / gravity).astype(np.float32)


def _profile_dim(ds: xr.Dataset) -> str:
    for name in ("N_PROF", "profile", "PROFILE", "N_OBS"):
        if name in ds.sizes:
            return name
    for name in ("JULD", "TIME", "LATITUDE"):
        if name in ds and ds[name].dims:
            return ds[name].dims[0]
    raise ValueError("Could not identify the profile dimension.")


def _values(ds: xr.Dataset, names: Sequence[str], profile_dim: str, default: Any = None) -> np.ndarray | None:
    name = first_existing(names, ds.variables)
    if name is None:
        return default
    data = ds[name]
    if profile_dim in data.dims and data.dims[0] != profile_dim:
        data = data.transpose(profile_dim, ...)
    return np.asarray(data.values)


def _profile_vector(values: np.ndarray | None, count: int, *, dtype: Any = float) -> np.ndarray:
    """Normalize per-profile metadata, including scalar platform coordinates."""
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.size == 1 and count != 1:
        return np.full(count, array.item(), dtype=dtype)
    return array


def _profile_matrix(ds: xr.Dataset, name: str | None, profile_dim: str, count: int, *, dtype: Any) -> np.ndarray | None:
    """Return a vertical variable with its profile axis first, broadcasting shared levels."""
    if name is None:
        return None
    data = ds[name]
    if profile_dim in data.dims and data.dims[0] != profile_dim:
        data = data.transpose(profile_dim, ...)
    values = np.asarray(data.values, dtype=dtype)
    if profile_dim not in data.dims:
        return np.broadcast_to(values, (count, *values.shape))
    if values.ndim == 0:
        return np.full((count, 1), values.item(), dtype=dtype)
    if values.shape[0] == 1 and count != 1:
        return np.broadcast_to(values, (count, *values.shape[1:]))
    if values.shape[0] != count:
        raise ValueError(f"{name} has {values.shape[0]} profiles; expected {count}.")
    return values


def _decode_time(ds: xr.Dataset, profile_dim: str) -> pd.DatetimeIndex:
    name = first_existing(("JULD", "TIME", "time"), ds.variables)
    if name is None:
        return pd.DatetimeIndex([pd.NaT] * ds.sizes[profile_dim], tz="UTC")
    values = np.asarray(ds[name].values, dtype=np.float64).reshape(-1)
    raw_units = str(ds[name].attrs.get("units", "days since 1950-01-01"))
    units = raw_units.lower()
    unit = "h" if units.startswith("hour") else "s" if units.startswith("second") else "D"
    if "since" not in units:
        origin = "1950-01-01"
    else:
        origin = raw_units[units.index("since") + len("since"):].strip().replace(" utc", "")
        reference_name = next((variable for variable in ds.variables if variable.lower() == origin.lower()), None)
        if reference_name is not None:
            reference = np.asarray(ds[reference_name].values).reshape(-1)[0]
            if isinstance(reference, bytes):
                reference = reference.decode("ascii", "replace")
            origin = str(reference).strip()
    origin_timestamp = pd.Timestamp(origin)
    if origin_timestamp.tzinfo is not None:
        origin_timestamp = origin_timestamp.tz_convert("UTC").tz_localize(None)
    return pd.to_datetime(values, unit=unit, origin=origin_timestamp, utc=True, errors="coerce")


def _qc(values: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    output = np.full(shape, -1, dtype=np.int8)
    if values is None:
        return output
    array = np.asarray(values)
    if array.shape != shape:
        try:
            array = array.reshape(shape)
        except ValueError:
            return output
    if array.dtype.kind in {"S", "U", "O"}:
        text = np.char.strip(array.astype(str))
        for code in range(10):
            output[text == str(code)] = code
    else:
        finite = np.isfinite(array)
        output[finite] = array[finite].astype(np.int8)
    return output


def _open_netcdf(path: Path) -> xr.Dataset | None:
    """Open both NetCDF4/HDF5 and classic NetCDF files when backends vary."""
    with path.open("rb") as stream:
        signature = stream.read(8)
        # CORA encodes unavailable products as a 32-byte all-zero .nc member.
        if path.stat().st_size == 32 and signature == b"\0" * 8 and stream.read() == b"\0" * 24:
            return None
    is_hdf5 = signature == b"\x89HDF\r\n\x1a\n"
    if not is_hdf5:
        return xr.open_dataset(
            path,
            engine="scipy",
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        )
    try:
        return xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
            mask_and_scale=True,
            cache=False,
        )
    except ImportError as error:
        raise ImportError(
            "Reading NetCDF4/HDF5 inputs requires the h5py dependency. "
            "Install the project dependencies and retry."
        ) from error


def iter_archive_file(
    path: Path,
    archive_version: str,
    chunk_size: int = 10_000,
    *,
    metadata_only: bool = False,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    ds = _open_netcdf(path)
    if ds is None:
        return
    with ds:
        try:
            profile_dim = _profile_dim(ds)
        except ValueError:
            # CORA tarballs also contain gridded field products; those are not profile inputs.
            return
        count = ds.sizes[profile_dim]
        times = _decode_time(ds, profile_dim)
        lat_all = _profile_vector(_values(ds, ("LATITUDE", "latitude", "lat"), profile_dim), count)
        lon_all = _profile_vector(_values(ds, ("LONGITUDE", "longitude", "lon"), profile_dim), count)
        id_names = {
            "dc_reference": ("DC_REFERENCE", "INST_REFERENCE"),
            "platform_code": ("PLATFORM_CODE", "PLATFORM_NUMBER"),
            "project_name": ("PROJECT_NAME",),
            "instrument_reference": ("INST_REFERENCE",),
            "wmo_inst_type": ("WMO_INST_TYPE",),
            "probe_type": ("PROBE_TYPE", "DATA_TYPE"),
            "data_mode": ("DATA_MODE",),
        }
        identifiers = {}
        for target, names in id_names.items():
            value = _values(ds, names, profile_dim)
            decoded = decode_char_array(value) if value is not None else np.full(count, "")
            decoded = np.asarray(decoded)
            identifiers[target] = (
                np.full(count, str(decoded.item()))
                if decoded.ndim == 0
                else decoded.reshape(-1)
            )
        depth_name = first_existing(("DEPH_CORRECTED", "DEPH", "PRES_ADJUSTED", "PRES", "depth"), ds.variables)
        temp_name = first_existing(("TEMP_ADJUSTED", "POTM_CORRECTED", "TEMP", "temperature"), ds.variables)
        sal_name = first_existing(("PSAL_ADJUSTED", "PSAL_CORRECTED", "PSAL", "salinity"), ds.variables)
        depth = _profile_matrix(ds, depth_name, profile_dim, count, dtype=np.float32) if not metadata_only else None
        temp = _profile_matrix(ds, temp_name, profile_dim, count, dtype=np.float32) if not metadata_only else None
        sal = _profile_matrix(ds, sal_name, profile_dim, count, dtype=np.float32) if not metadata_only else None
        temp_qc_name = first_existing((f"{temp_name}_QC" if temp_name else "", "TEMP_QC"), ds.variables)
        sal_qc_name = first_existing((f"{sal_name}_QC" if sal_name else "", "PSAL_QC"), ds.variables)
        temp_qc = _profile_matrix(ds, temp_qc_name, profile_dim, count, dtype=object) if not metadata_only else None
        sal_qc = _profile_matrix(ds, sal_qc_name, profile_dim, count, dtype=object) if not metadata_only else None
        profile_qc = _values(ds, ("PROFILE_QC", "PROFILE_POTM_QC", "PROFILE_TEMP_QC"), profile_dim)
        position_qc = _values(ds, ("POSITION_QC",), profile_dim)

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            indexes = np.arange(start, stop)
            lat, lon = lat_all[start:stop], normalize_longitude_180(lon_all[start:stop])
            profiles = pd.DataFrame({
                "archive_version": archive_version,
                "source_file": Path(path).name,
                "source_profile_index": indexes,
                "profile_id": [f"{Path(path).name}:{index}" for index in indexes],
                "datetime_utc": times[start:stop],
                "latitude": lat,
                "longitude": lon,
                "valid_coordinate": valid_coordinates(lat, lon),
                **{name: values[start:stop].astype(str) for name, values in identifiers.items()},
                "profile_qc": _qc(profile_qc, (count,))[start:stop],
                "position_qc": _qc(position_qc, (count,))[start:stop],
                "selected_depth_variable": depth_name or "",
                "selected_temperature_variable": temp_name or "",
                "selected_salinity_variable": sal_name or "",
            })
            profiles["calendar_day"] = profiles["datetime_utc"].dt.strftime("%Y%m%d")
            profiles["latitude_cell"] = np.floor((profiles["latitude"] + 90.0) / 0.25).astype("Int32")
            profiles["longitude_cell"] = np.floor((profiles["longitude"] + 180.0) / 0.25).astype("Int32")
            profiles["instrument_class"] = profiles["probe_type"].where(profiles["probe_type"].str.strip() != "", profiles["wmo_inst_type"])
            observations: list[dict[str, Any]] = []
            if depth is not None:
                for local, source_index in enumerate(indexes):
                    row_depth = np.asarray(depth[source_index]).reshape(-1)
                    if depth_name and depth_name.upper().startswith("PRES"):
                        row_depth = pressure_dbar_to_depth_m(row_depth, lat_all[source_index])
                    row_temp = np.asarray(temp[source_index]).reshape(-1) if temp is not None else np.full(row_depth.shape, np.nan)
                    row_sal = np.asarray(sal[source_index]).reshape(-1) if sal is not None else np.full(row_depth.shape, np.nan)
                    tq = _qc(np.asarray(temp_qc[source_index]).reshape(-1) if temp_qc is not None else None, row_depth.shape)
                    sq = _qc(np.asarray(sal_qc[source_index]).reshape(-1) if sal_qc is not None else None, row_depth.shape)
                    valid = np.isfinite(row_depth) & (row_depth >= 0) & (np.isfinite(row_temp) | np.isfinite(row_sal))
                    for level in np.flatnonzero(valid):
                        observations.append({
                            "profile_id": profiles.iloc[local]["profile_id"], "level_index": int(level),
                            "depth_m": float(row_depth[level]), "temperature_c": float(row_temp[level]),
                            "salinity": float(row_sal[level]), "temperature_qc": int(tq[level]), "salinity_qc": int(sq[level]),
                            "source_vertical_variable": depth_name,
                        })
            # Preserve the observation schema for profiles with no valid T/S levels.
            yield profiles, pd.DataFrame.from_records(observations, columns=OBSERVATION_COLUMNS)


def index_assimilation_inputs(
    inputs: Sequence[Path],
    *,
    archive_version: str,
    profiles_output: Path,
    observations_output: Path | None,
    chunk_size: int = 10_000,
    metadata_only: bool = False,
    target_dates: Sequence[pd.Timestamp] | None = None,
    target_profiles: pd.DataFrame | None = None,
    xbt_only: bool = False,
    date_margin_days: int = 1,
) -> tuple[Path, Path | None]:
    profile_writer = observation_writer = None
    Path(profiles_output).parent.mkdir(parents=True, exist_ok=True)
    if observations_output is not None:
        Path(observations_output).parent.mkdir(parents=True, exist_ok=True)
    if not metadata_only and observations_output is None:
        raise ValueError("observations_output is required unless metadata_only=True.")
    proximity = TargetProximityFilter(target_profiles) if target_profiles is not None else None
    try:
        paths = (
            iter_targeted_xbt_inputs(inputs, target_dates or (), date_margin_days)
            if xbt_only
            else iter_targeted_inputs(inputs, target_dates or (), date_margin_days)
            if target_dates is not None
            else iter_netcdf_inputs(inputs)
        )
        for path in paths:
            for profiles, observations in iter_archive_file(path, archive_version, chunk_size, metadata_only=metadata_only):
                if proximity is not None:
                    profiles = profiles.loc[proximity.mask(profiles)].reset_index(drop=True)
                    if profiles.empty:
                        continue
                    observations = observations[observations["profile_id"].isin(set(profiles["profile_id"]))]
                profile_table = pa.Table.from_pandas(profiles, preserve_index=False)
                if profile_writer is None:
                    profile_writer = pq.ParquetWriter(profiles_output, profile_table.schema, compression="zstd")
                profile_writer.write_table(profile_table)
                if not metadata_only and not observations.empty:
                    observation_table = pa.Table.from_pandas(observations, preserve_index=False)
                    if observation_writer is None:
                        observation_writer = pq.ParquetWriter(observations_output, observation_table.schema, compression="zstd")
                    observation_writer.write_table(observation_table)
    finally:
        if profile_writer:
            profile_writer.close()
        if observation_writer:
            observation_writer.close()
    if profile_writer is None:
        raise ValueError("No profiles were indexed.")
    if observation_writer is None and not metadata_only and observations_output is not None:
        pd.DataFrame(columns=OBSERVATION_COLUMNS).to_parquet(observations_output, index=False)
    return profiles_output, observations_output


def expand_input_paths(inputs: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in map(Path, inputs):
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.nc")))
        else:
            expanded.append(path)
    return expanded


def _iter_tar_netcdf_members(
    bundle: tarfile.TarFile,
    temporary_root: Path,
    *,
    member_filter: Any = None,
    container_filter: Any = None,
) -> Iterator[Path]:
    for member in bundle:
        if not member.isfile():
            continue
        member_name = member.name.lower()
        source = bundle.extractfile(member)
        if source is None:
            continue
        if _is_tar_member(member.name):
            if container_filter is not None and not container_filter(member.name):
                source.close()
                continue
            with source:
                try:
                    with tarfile.open(fileobj=source, mode="r:*") as nested_bundle:
                        yield from _iter_tar_netcdf_members(
                            nested_bundle,
                            temporary_root,
                            member_filter=member_filter,
                            container_filter=container_filter,
                        )
                except tarfile.ReadError:
                    continue
        elif member_name.endswith(".nc"):
            if member_filter is not None and not member_filter(member.name):
                source.close()
                continue
            destination = temporary_root / Path(member.name).name
            with source, destination.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
            try:
                yield destination
            finally:
                destination.unlink(missing_ok=True)
def iter_netcdf_inputs(
    inputs: Sequence[Path],
    *,
    member_filter: Any = None,
    container_filter: Any = None,
) -> Iterator[Path]:
    """Yield NetCDF paths, streaming tar members through bounded temporary storage."""
    for path in map(Path, inputs):
        if not tarfile.is_tarfile(path):
            yield path
            continue
        with tempfile.TemporaryDirectory(prefix="profile-audit-netcdf-") as temporary:
            temporary_root = Path(temporary)
            with tarfile.open(path, mode="r:*") as bundle:
                yield from _iter_tar_netcdf_members(
                    bundle,
                    temporary_root,
                    member_filter=member_filter,
                    container_filter=container_filter,
                )


def iter_targeted_xbt_inputs(
    inputs: Sequence[Path],
    target_dates: Sequence[pd.Timestamp],
    margin_days: int = 1,
) -> Iterator[Path]:
    """Stream only date-matching XBT NetCDF members from archive inputs."""
    selected_days = _target_days(target_dates, margin_days)
    selected_years = {date.year for date in target_dates if not pd.isna(date)}

    def member_filter(name: str) -> bool:
        if not _is_xbt_member(name):
            return False
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", name)
        return not selected_days or match is None or match.group(1) in selected_days

    def container_filter(name: str) -> bool:
        year = _member_year(name)
        return not selected_years or year is None or year in selected_years

    yield from iter_netcdf_inputs(
        inputs,
        member_filter=member_filter,
        container_filter=container_filter,
    )


def iter_targeted_inputs(
    inputs: Sequence[Path],
    target_dates: Sequence[pd.Timestamp],
    margin_days: int = 1,
) -> Iterator[Path]:
    """Stream only archive members near target dates across all instrument classes."""
    selected_days = _target_days(target_dates, margin_days)
    selected_years = {date.year for date in target_dates if not pd.isna(date)}

    def member_filter(name: str) -> bool:
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", name)
        return not selected_days or match is None or match.group(1) in selected_days

    def container_filter(name: str) -> bool:
        year = _member_year(name)
        return not selected_years or year is None or year in selected_years

    yield from iter_netcdf_inputs(
        inputs,
        member_filter=member_filter,
        container_filter=container_filter,
    )


def _unit_sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(latitude, dtype=np.float64))
    lon = np.radians(np.asarray(longitude, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


class TargetProximityFilter:
    """Retain CORA profiles only when they share the audit's broad duplicate window."""

    def __init__(self, targets: pd.DataFrame) -> None:
        if "datetime_utc" not in targets:
            raise ValueError("Target profiles need datetime_utc for candidate filtering.")
        valid = valid_coordinates(targets["latitude"], targets["longitude"])
        targets = targets.loc[valid].copy()
        targets["datetime_utc"] = pd.to_datetime(targets["datetime_utc"], utc=True, errors="coerce")
        self.targets = targets.loc[targets["datetime_utc"].notna()].reset_index(drop=True)
        self.tree = cKDTree(_unit_sphere(self.targets["latitude"], self.targets["longitude"]))
        self.chord_radius = 2.0 * np.sin(25.0 / (2.0 * 6371.0088))

    def mask(self, candidates: pd.DataFrame) -> np.ndarray:
        if candidates.empty:
            return np.zeros(0, dtype=bool)
        valid = valid_coordinates(candidates["latitude"], candidates["longitude"])
        times = pd.to_datetime(candidates["datetime_utc"], utc=True, errors="coerce")
        valid &= times.notna().to_numpy()
        retained = np.zeros(len(candidates), dtype=bool)
        positions = np.flatnonzero(valid)
        if not len(positions):
            return retained
        neighbors = self.tree.query_ball_point(
            _unit_sphere(candidates.iloc[positions]["latitude"], candidates.iloc[positions]["longitude"]),
            self.chord_radius,
        )
        for position, target_positions in zip(positions, neighbors):
            if not target_positions:
                continue
            target_rows = self.targets.iloc[target_positions]
            minutes = (target_rows["datetime_utc"] - times.iloc[position]).abs().dt.total_seconds() / 60.0
            if (minutes <= 24 * 60).any():
                retained[position] = True
        return retained


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index frozen CORA, NRT, or EN4 profile NetCDF files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--archive-version", required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--target-profiles", type=Path, help="Target profile table for targeted archive streaming.")
    parser.add_argument("--xbt-only", action="store_true", help="Stream only target-day XBT members from tar archives.")
    parser.add_argument("--date-margin-days", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    inputs = expand_input_paths(args.inputs)
    if not inputs:
        raise SystemExit("No NetCDF files found in the supplied paths.")
    if args.xbt_only and args.target_profiles is None:
        raise SystemExit("--xbt-only requires --target-profiles.")
    target_dates = None
    target_profiles = None
    if args.target_profiles is not None:
        target_profiles = read_table(args.target_profiles)
        column = "datetime_utc" if "datetime_utc" in target_profiles else "profile_date"
        if column not in target_profiles:
            raise SystemExit("Target profiles need datetime_utc or profile_date.")
        if column != "datetime_utc":
            target_profiles = target_profiles.rename(columns={column: "datetime_utc"})
        target_dates = list(pd.to_datetime(target_profiles["datetime_utc"], utc=True, errors="coerce"))
    outputs = index_assimilation_inputs(
        inputs,
        archive_version=args.archive_version,
        profiles_output=args.profiles_output,
        observations_output=args.observations_output,
        chunk_size=args.chunk_size,
        metadata_only=args.metadata_only,
        target_dates=target_dates,
        target_profiles=target_profiles,
        xbt_only=args.xbt_only,
        date_margin_days=args.date_margin_days,
    )
    print(*(path for path in outputs if path is not None), sep="\n")


if __name__ == "__main__":
    main()
