from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from profile_audit.common import read_table, write_table


def _write_source(
    profiles: list[dict[str, object]],
    observations: list[dict[str, object]],
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    profiles_path = output_dir / f"{name}_profiles.parquet"
    observations_path = output_dir / f"{name}_observations.parquet"
    write_table(pd.DataFrame.from_records(profiles), profiles_path)
    write_table(pd.DataFrame.from_records(observations), observations_path)
    return profiles_path, observations_path


def _parse_cugn(inputs: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    profiles: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for line in (66, 80, 90):
        with xr.open_dataset(inputs / "cugn" / f"CUGN_line_{line}.nc") as dataset:
            times = pd.to_datetime(dataset["time"].values, utc=True)
            selected = np.flatnonzero((times >= start) & (times < end))
            depth = np.asarray(dataset["depth"].values, dtype=float)
            temperature = np.asarray(dataset["temperature"].isel(profile=selected).values, dtype=float)
            salinity = np.asarray(dataset["salinity"].isel(profile=selected).values, dtype=float)
            latitudes = np.asarray(dataset["lat"].isel(profile=selected).values, dtype=float)
            longitudes = np.asarray(dataset["lon"].isel(profile=selected).values, dtype=float)
            missions = np.asarray(dataset["mission"].isel(profile=selected).values)
        for output_index, source_index in enumerate(selected):
            profile_id = f"cugn:{line}:{int(source_index)}"
            profiles.append({
                "profile_id": profile_id,
                "source_name": "CUGN",
                "line": int(line),
                "source_file": f"CUGN_line_{line}.nc",
                "source_profile_index": int(source_index),
                "mission": f"cugn:{line}:{missions[output_index]}",
                "datetime_utc": times[source_index],
                "latitude": latitudes[output_index],
                "longitude": longitudes[output_index],
            })
            valid = np.isfinite(depth) & (np.isfinite(temperature[:, output_index]) | np.isfinite(salinity[:, output_index]))
            for level in np.flatnonzero(valid):
                observations.append({
                    "profile_id": profile_id,
                    "level_index": int(level),
                    "depth_m": depth[level],
                    "temperature_c": temperature[level, output_index],
                    "salinity_psu": salinity[level, output_index],
                })
    return profiles, observations


def _parse_bgep(inputs: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = read_table(inputs / "beaufort_bgep" / "2018_station_metadata.csv").set_index("file")
    profiles: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    with zipfile.ZipFile(inputs / "beaufort_bgep" / "LSSL_ctd2018.zip") as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in metadata.index:
                continue
            lines = archive.read(member).decode("latin-1").splitlines()
            names = {
                line.split("=")[1].split(":", 1)[0].strip(): int(line.split("=")[0].split()[-1])
                for line in lines if line.startswith("# name ")
            }
            required = {"depSM", "t090C", "sal00"}
            if not required.issubset(names):
                raise ValueError(f"{name} is missing required CTD variables: {sorted(required - set(names))}")
            data_start = next(index for index, line in enumerate(lines) if line.strip() == "*END*") + 1
            values = np.loadtxt(io.StringIO("\n".join(lines[data_start:])), ndmin=2)
            row = metadata.loc[name]
            profile_id = f"bgep-2018:{name}"
            profiles.append({
                "profile_id": profile_id,
                "source_name": "BGEP",
                "cruise": "2018-81",
                "station": str(row["station"]),
                "source_file": name,
                "datetime_utc": pd.to_datetime(row["time"], utc=True),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "bottom_depth_m": float(row["bottom_depth_m"]),
            })
            for level, value in enumerate(values):
                depth = value[names["depSM"]]
                temperature = value[names["t090C"]]
                salinity = value[names["sal00"]]
                if not np.isfinite(depth) or depth < 0 or not (np.isfinite(temperature) or np.isfinite(salinity)):
                    continue
                observations.append({
                    "profile_id": profile_id,
                    "level_index": level,
                    "depth_m": depth,
                    "temperature_c": temperature,
                    "salinity_psu": salinity,
                })
    return profiles, observations


def _parse_itp(inputs: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = read_table(inputs / "beaufort_itp" / "itp107_2018-09_profile_metadata.csv")
    profiles: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        values = np.loadtxt(inputs / "beaufort_itp" / "itp107" / row.file, comments="%", ndmin=2)
        profile_id = f"itp-107:{int(row.profile_number)}"
        profiles.append({
            "profile_id": profile_id,
            "source_name": "ITP",
            "platform_id": "ITP-107",
            "mission": "ITP-107",
            "source_file": str(row.file),
            "source_profile_index": int(row.profile_number),
            "datetime_utc": pd.to_datetime(row.time, utc=True),
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        })
        for level, value in enumerate(values):
            if not np.isfinite(value[0]) or value[0] < 0 or not (np.isfinite(value[1]) or np.isfinite(value[2])):
                continue
            observations.append({
                "profile_id": profile_id,
                "level_index": level,
                "pressure_dbar": value[0],
                "depth_m": value[0],  # Preserve pressure; latitude-aware depth conversion belongs to E5 scoring.
                "temperature_c": value[1],
                "salinity_psu": value[2],
            })
    return profiles, observations


def parse_e4_e5_profiles(inputs: Path, output_dir: Path, cugn_start: str = "2007-01-01", cugn_end: str = "2019-01-01") -> list[Path]:
    """Normalize local E4/E5 sources for existing archive and EN4 audits."""
    start, end = pd.Timestamp(cugn_start, tz="UTC"), pd.Timestamp(cugn_end, tz="UTC")
    outputs: list[Path] = []
    for name, parser in (
        ("cugn", lambda: _parse_cugn(inputs, start, end)),
        ("bgep_2018", lambda: _parse_bgep(inputs)),
        ("itp107_2018_09", lambda: _parse_itp(inputs)),
    ):
        outputs.extend(_write_source(*parser(), output_dir, name))
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize local E4 CUGN and E5 BGEP/ITP profiles for auditing.")
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cugn-start", default="2007-01-01")
    parser.add_argument("--cugn-end", default="2019-01-01")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    for output in parse_e4_e5_profiles(args.inputs, args.output_dir, args.cugn_start, args.cugn_end):
        print(output)


if __name__ == "__main__":
    main()
