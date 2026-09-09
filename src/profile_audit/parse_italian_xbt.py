from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

from profile_audit.common import normalize_longitude_180, normalize_longitude_360, sha256_file, write_table

COLUMNS = (
    "cruise", "station", "instrument_type", "date", "time", "longitude_raw",
    "latitude", "bottom_depth", "elapsed_time", "depth_1", "depth_2", "depth_3",
    "temperature_1", "temperature_2", "qf",
)
PNRA_XXXII_SHA256 = "d3131feedf029654d6623759d0f7f1f6589adb620d2be27ff9fae0fad764605b"
PNRA_XXXII_SWAPPED_DATES = frozenset(
    {"02/01/2017", "03/01/2017", "04/01/2017", "05/01/2017"}
)


def download_zenodo_xbt(output_dir: Path, record_id: int = 14848849) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    metadata.raise_for_status()
    paths = []
    for item in metadata.json()["files"]:
        path = output_dir / item["key"]
        expected_md5 = item["checksum"].removeprefix("md5:")
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected_md5:
            with requests.get(item["links"]["self"], timeout=300, stream=True) as response:
                response.raise_for_status()
                with path.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected_md5:
            raise ValueError(f"MD5 mismatch after downloading {path.name}")
        paths.append(path)
    return paths


def _metadata_and_rows(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    metadata: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    header_seen = False
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith("//"):
                fields = line[2:].strip().split("\t", maxsplit=1)
                if fields and fields[0]:
                    metadata[fields[0].strip()] = fields[1].strip() if len(fields) == 2 else ""
                continue
            parsed = next(csv.reader([line], delimiter="\t"))
            if not header_seen:
                if parsed and parsed[0].strip().lower() == "cruise":
                    if len(parsed) != 15:
                        raise ValueError(f"{path}:{line_number}: expected 15 header columns, got {len(parsed)}")
                    header_seen = True
                continue
            if not parsed or all(not item.strip() for item in parsed):
                continue
            if len(parsed) != 15:
                raise ValueError(f"{path}:{line_number}: expected 15 columns, got {len(parsed)}")
            rows.append(dict(zip(COLUMNS, (item.strip() for item in parsed), strict=True)))
    if not header_seen:
        raise ValueError(f"No 15-column data header found in {path}")
    return metadata, rows


def _depth_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    depths = np.asarray([float(row["depth_3"]) for row in rows], dtype=np.float64)
    finite = depths[np.isfinite(depths)]
    differences = np.diff(finite)
    reversals = differences[differences < 0]
    return {
        "finite_level_count": int(finite.size),
        "depth_reversal_count": int(reversals.size),
        "maximum_depth_reversal_m": float(-reversals.min()) if reversals.size else 0.0,
        "corrected_depth_monotonic": bool(reversals.size == 0),
    }


def _usable_row(row: dict[str, Any], accepted_qf: frozenset[int]) -> bool:
    return (
        int(row["qf"]) in accepted_qf
        and np.isfinite(float(row["depth_3"]))
        and np.isfinite(float(row["temperature_2"]))
    )


def _parse_profile_datetime(
    date_text: str,
    time_text: str,
    *,
    source_file: str,
    source_checksum: str,
) -> tuple[pd.Timestamp, str, str | None]:
    correction = None
    date_format = "%m/%d/%Y %H:%M"
    format_label = "MDY"
    if (
        source_file == "xbt_PNRA_XXXII.txt"
        and source_checksum == PNRA_XXXII_SHA256
        and date_text in PNRA_XXXII_SWAPPED_DATES
    ):
        date_format = "%d/%m/%Y %H:%M"
        format_label = "DMY_source_correction"
        correction = (
            "PNRA XXXII provider spreadsheet date-format defect; corrected using "
            "Zenodo checksum, declared cruise coverage, NCEI accession 0287163, "
            "and Aulicino et al. (2025) Table 1."
        )
    timestamp = pd.to_datetime(
        f"{date_text} {time_text}",
        format=date_format,
        utc=True,
        errors="raise",
    )
    return timestamp, format_label, correction


def _coverage_offset_minutes(
    timestamp: pd.Timestamp,
    coverage_start: pd.Timestamp | None,
    coverage_end: pd.Timestamp | None,
) -> float:
    if coverage_start is not None and pd.notna(coverage_start) and timestamp < coverage_start:
        return float((timestamp - coverage_start).total_seconds() / 60)
    if coverage_end is not None and pd.notna(coverage_end) and timestamp > coverage_end:
        return float((timestamp - coverage_end).total_seconds() / 60)
    return 0.0


def parse_xbt_files(
    paths: Sequence[Path],
    *,
    profiles_output: Path,
    observations_output: Path,
    sensitivity_output: Path | None = None,
    primary_qf: frozenset[int] = frozenset({1}),
    sensitivity_qf: frozenset[int] = frozenset({1, 2}),
) -> tuple[Path, Path]:
    profile_records: list[dict[str, Any]] = []
    observation_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    seen_keys: dict[tuple[str, str], Path] = {}

    for path in map(Path, paths):
        metadata, raw_rows = _metadata_and_rows(path)
        checksum = sha256_file(path)
        coverage_start = pd.to_datetime(metadata.get("Time Coverage Start"), utc=True, errors="coerce")
        coverage_end = pd.to_datetime(metadata.get("Time Coverage End"), utc=True, errors="coerce")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            grouped[(row["cruise"], row["station"])].append(row)
        for key, rows in grouped.items():
            if key in seen_keys:
                raise ValueError(f"Duplicate cruise/station {key} in {seen_keys[key]} and {path}")
            seen_keys[key] = path
            cruise, station = key
            first = rows[0]
            timestamp, datetime_format, datetime_correction = _parse_profile_datetime(
                first["date"],
                first["time"],
                source_file=path.name,
                source_checksum=checksum,
            )
            coverage_offset_minutes = _coverage_offset_minutes(timestamp, coverage_start, coverage_end)
            lat = float(first["latitude"])
            lon = float(first["longitude_raw"])
            if not -90 <= lat <= 90 or not -360 <= lon <= 360:
                raise ValueError(f"Impossible coordinates for {cruise}/{station}: {lat}, {lon}")
            for row in rows:
                if (row["date"], row["time"], row["latitude"], row["longitude_raw"]) != (
                    first["date"], first["time"], first["latitude"], first["longitude_raw"]
                ):
                    raise ValueError(f"Inconsistent profile metadata for {cruise}/{station}")
            profile_id = f"italian-xbt:{cruise}:{station}"
            good = [row for row in rows if _usable_row(row, primary_qf)]
            raw_depth_diagnostics = _depth_diagnostics(rows)
            primary_depth_diagnostics = _depth_diagnostics(good)
            profile_records.append({
                "profile_id": profile_id, "cruise": cruise, "station": station,
                "datetime_utc": timestamp, "latitude": lat, "longitude_raw": lon,
                "source_date_text": first["date"], "source_time_text": first["time"],
                "datetime_format_inferred": datetime_format,
                "datetime_correction_applied": datetime_correction is not None,
                "datetime_correction_reason": datetime_correction,
                "coverage_offset_minutes": coverage_offset_minutes,
                "inside_declared_time_coverage": coverage_offset_minutes == 0.0,
                "longitude_180": normalize_longitude_180(lon), "longitude_360": normalize_longitude_360(lon),
                "probe_type": metadata.get("Probe Type", first["instrument_type"]),
                "ship_name": metadata.get("Platform Name"), "project_name": metadata.get("Scientific Project"),
                "source_file": path.name, "source_checksum": checksum,
                "maximum_depth": max((float(row["depth_3"]) for row in good), default=float("nan")),
                "number_of_good_levels": len(good),
                "raw_finite_level_count": raw_depth_diagnostics["finite_level_count"],
                "raw_depth_reversal_count": raw_depth_diagnostics["depth_reversal_count"],
                "raw_maximum_depth_reversal_m": raw_depth_diagnostics["maximum_depth_reversal_m"],
                "raw_corrected_depth_monotonic": raw_depth_diagnostics["corrected_depth_monotonic"],
                "primary_depth_reversal_count": primary_depth_diagnostics["depth_reversal_count"],
                "primary_maximum_depth_reversal_m": primary_depth_diagnostics["maximum_depth_reversal_m"],
                "primary_corrected_depth_monotonic": primary_depth_diagnostics["corrected_depth_monotonic"],
            })
            for level_index, row in enumerate(rows):
                record = {
                    "profile_id": profile_id, "level_index": level_index,
                    "elapsed_time_s": float(row["elapsed_time"]), "depth_1": float(row["depth_1"]),
                    "depth_2": float(row["depth_2"]), "depth_3": float(row["depth_3"]),
                    "temperature_1": float(row["temperature_1"]), "temperature_2": float(row["temperature_2"]),
                    "qf": int(row["qf"]),
                }
                if _usable_row(row, primary_qf):
                    observation_records.append(record)
                if _usable_row(row, sensitivity_qf):
                    sensitivity_records.append(record)

    write_table(pd.DataFrame.from_records(profile_records), profiles_output)
    write_table(pd.DataFrame.from_records(observation_records), observations_output)
    if sensitivity_output is not None:
        write_table(pd.DataFrame.from_records(sensitivity_records), sensitivity_output)
    return profiles_output, observations_output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and parse the Italian PNRA XBT profiles.")
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--record", type=int, default=14848849)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path, required=True)
    parser.add_argument("--sensitivity-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    inputs = args.inputs or (download_zenodo_xbt(args.download_dir, args.record) if args.download_dir else [])
    if not inputs:
        raise SystemExit("Provide input files or --download-dir.")
    outputs = parse_xbt_files(inputs, profiles_output=args.profiles_output, observations_output=args.observations_output, sensitivity_output=args.sensitivity_output)
    print(*(str(path) for path in outputs), sep="\n")


if __name__ == "__main__":
    main()
