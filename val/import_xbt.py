# /work/envs/depth/bin/python -m val.import_xbt evaluations/focus_evaluation.tar.gz val/results/xbt_import/profiles.csv --metadata evaluations/outputs/italian_en4_no_match_profiles.parquet --manifest val/results/xbt_import/manifest.json --year 2016 --accepted-qc-flag 1
"""Import Italian PNRA XBT text files from the supplied evaluation archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import pandas as pd

CANONICAL_COLUMNS = (
    "profile_id",
    "cluster_id",
    "source",
    "observation_time",
    "longitude",
    "latitude",
    "depth_m",
    "variable",
    "observation",
    "independence",
)
_SOURCE_PREFIX = "inputs/italian_xbt/"
_REQUIRED_METADATA = {
    "profile_id",
    "cruise",
    "station",
    "datetime_utc",
    "source_file",
    "latitude",
    "longitude_raw",
}


def _sha256(path: Path) -> str:
    """Hash a file without loading the complete archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256_bytes(value: bytes) -> str:
    """Hash one archive member."""
    return hashlib.sha256(value).hexdigest()


def _identifier(value: Any) -> str:
    """Normalize integer-like archive identifiers without changing text IDs."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _metadata(path: Path | None, year: int | None) -> pd.DataFrame | None:
    """Load and validate the optional set of archive-screened profiles."""
    if path is None:
        return None
    frame = pd.read_parquet(path)
    missing = sorted(_REQUIRED_METADATA - set(frame.columns))
    if missing:
        raise ValueError("metadata parquet missing columns: " + ", ".join(missing))
    if frame[list(_REQUIRED_METADATA)].isna().any().any():
        raise ValueError("metadata parquet contains missing identifiers or coordinates")
    frame = frame.copy()
    frame["datetime_utc"] = pd.to_datetime(
        frame["datetime_utc"], utc=True, errors="coerce"
    )
    if frame["datetime_utc"].isna().any():
        raise ValueError("metadata datetime_utc contains invalid timestamps")
    if year is not None:
        frame = frame.loc[frame["datetime_utc"].dt.year.eq(year)].copy()
    if frame.empty:
        raise ValueError(f"metadata contains no profiles for year {year}")
    for column in ("profile_id", "cruise", "station", "source_file"):
        frame[column] = frame[column].map(_identifier)
        if frame[column].eq("").any():
            raise ValueError(f"metadata {column} must be populated")
    if frame.profile_id.duplicated().any():
        raise ValueError("metadata profile_id values must be unique")
    keys = ["source_file", "cruise", "station"]
    if frame.duplicated(keys).any():
        raise ValueError("metadata source_file/cruise/station keys must be unique")
    return frame


def _metadata_index(
    metadata: pd.DataFrame | None,
) -> dict[tuple[str, str, str], tuple[str, pd.Timestamp, float, float]] | None:
    """Index selected profile IDs once instead of filtering metadata per data row."""
    if metadata is None:
        return None
    return {
        (row.source_file, row.cruise, row.station): (
            row.profile_id,
            row.datetime_utc,
            float(row.latitude),
            float(row.longitude_raw),
        )
        for row in metadata.itertuples(index=False)
    }


def _column_indices(row: list[str], source_file: str) -> dict[str, int]:
    """Resolve the published PNRA columns from the actual tabular header."""
    normalized = [value.strip().lower() for value in row]
    prefixes = {
        "cruise": "cruise",
        "station": "station",
        "date": "mon/day/yr",
        "time": "hh:mm",
        "longitude": "longitude",
        "latitude": "latitude",
        "depth": "depth 3",
        "temperature": "temperature 2",
        "qc": "qf",
    }
    indices: dict[str, int] = {}
    for name, prefix in prefixes.items():
        matches = [
            index for index, value in enumerate(normalized) if value.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(f"{source_file} header must contain one {prefix!r} column")
        indices[name] = matches[0]
    return indices


def _parse_member(
    raw: bytes,
    member_name: str,
    selected_profiles: (
        dict[tuple[str, str, str], tuple[str, pd.Timestamp, float, float]] | None
    ),
    accepted_qc_flags: tuple[int, ...],
    year: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], set[str]]:
    """Parse QF, Depth 3, and Temperature 2 from one PNRA text member."""
    source_file = Path(member_name).name
    records: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    indices: dict[str, int] | None = None
    raw_rows = 0
    for row in csv.reader(
        io.StringIO(raw.decode("utf-8-sig", errors="replace")), delimiter="\t"
    ):
        if not row or not row[0] or row[0].startswith("//"):
            continue
        if indices is None:
            indices = _column_indices(row, source_file)
            continue
        raw_rows += 1
        try:
            cruise = _identifier(row[indices["cruise"]])
            station = _identifier(row[indices["station"]])
            timestamp = pd.to_datetime(
                f"{row[indices['date']].strip()} {row[indices['time']].strip()}",
                format="%m/%d/%Y %H:%M",
                utc=True,
                errors="raise",
            )
            longitude = float(row[indices["longitude"]])
            latitude = float(row[indices["latitude"]])
            depth = float(row[indices["depth"]])
            temperature = float(row[indices["temperature"]])
            qc_number = float(row[indices["qc"]])
        except (IndexError, TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (longitude, latitude, depth, temperature, qc_number)
        ):
            continue
        if not qc_number.is_integer() or not 0 <= qc_number <= 4:
            raise ValueError(f"{source_file} contains a QF code outside [0, 4]")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90 or depth < 0:
            raise ValueError(f"{source_file} contains invalid coordinates or depth")
        qc = int(qc_number)
        key = (source_file, cruise, station)
        if selected_profiles is not None:
            selected = selected_profiles.get(key)
            # The metadata is an accepted-profile set; unlisted raw casts are expected.
            if selected is None:
                continue
            profile, metadata_time, metadata_latitude, metadata_longitude = selected
            if profile not in seen_profiles and (
                timestamp != metadata_time
                or not math.isclose(latitude, metadata_latitude, abs_tol=1e-6)
                or not math.isclose(longitude, metadata_longitude, abs_tol=1e-6)
            ):
                raise ValueError(
                    f"metadata does not match raw time/coordinate for {profile}"
                )
        else:
            profile = f"italian-xbt:{source_file}:{cruise}:{station}"
        seen_profiles.add(profile)
        if (year is not None and timestamp.year != year) or qc not in accepted_qc_flags:
            continue
        records.append(
            {
                "profile_id": profile,
                "cluster_id": cruise,
                "source": "italian_xbt",
                "observation_time": timestamp,
                "longitude": longitude,
                "latitude": latitude,
                "depth_m": depth,
                "variable": "temperature",
                "observation": temperature,
                "independence": "unknown",
                "split_role": "external_holdout",
                "source_file": source_file,
                "cruise": cruise,
                "station": station,
                "qc_flag": qc,
                "temperature_kind": "in_situ",
            }
        )
    if indices is None:
        raise ValueError(f"{source_file} contains no PNRA table header")
    return (
        pd.DataFrame.from_records(records),
        {
            "source_file": source_file,
            "rows_in_table": raw_rows,
            "rows_selected": len(records),
            "sha256": _sha256_bytes(raw),
        },
        seen_profiles,
    )


def import_xbt_archive(
    archive_path: str | Path,
    output_csv: str | Path,
    *,
    metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    accepted_qc_flags: tuple[int, ...] = (1,),
    year: int | None = 2016,
) -> dict[str, Any]:
    """Create canonical raw XBT observations and a provenance manifest."""
    archive = Path(archive_path).resolve()
    output = Path(output_csv)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_suffix(".manifest.json")
    )
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists() or manifest_file.exists():
        raise FileExistsError("refusing to overwrite existing XBT output or manifest")
    accepted_qc = tuple(int(value) for value in accepted_qc_flags)
    if not accepted_qc or any(value < 0 or value > 4 for value in accepted_qc):
        raise ValueError("accepted_qc_flags must contain PNRA QF codes in [0, 4]")
    metadata = _metadata(None if metadata_path is None else Path(metadata_path), year)
    selected_profiles = _metadata_index(metadata)
    frames: list[pd.DataFrame] = []
    members: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            if not member.name.startswith(_SOURCE_PREFIX) or not member.name.endswith(
                ".txt"
            ):
                continue
            extracted = stream.extractfile(member)
            if extracted is None:
                continue
            frame, summary, seen = _parse_member(
                extracted.read(),
                member.name,
                selected_profiles,
                accepted_qc,
                year,
            )
            if not frame.empty:
                frames.append(frame)
            members.append(summary)
            seen_profiles.update(seen)
    if not frames:
        raise ValueError("archive contains no selected Italian XBT rows")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["profile_id", "depth_m"]).any():
        raise ValueError("selected XBT data contain duplicate profile/depth rows")
    expected_profiles = set(metadata.profile_id) if metadata is not None else set()
    missing_profiles = sorted(expected_profiles - seen_profiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "source_prefix": _SOURCE_PREFIX,
        "members": members,
        "year": year,
        "rows_parsed": int(len(result)),
        "profiles": int(result.profile_id.nunique()),
        "metadata_profiles_requested": len(expected_profiles),
        "metadata_profiles_missing_from_archive": len(missing_profiles),
        "metadata_missing_profile_ids": missing_profiles,
        "accepted_qc_flags": list(accepted_qc),
        "temperature_kind": "in_situ",
        "salinity_available": False,
        "provenance_status": "archive_screened_not_verified",
        "independence": "unknown",
        "conversion_required": "potential_temperature_for_primary_metric",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Run the importer from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument(
        "--accepted-qc-flag", type=int, action="append", dest="accepted_qc_flags"
    )
    args = parser.parse_args()
    import_xbt_archive(
        args.archive,
        args.output_csv,
        metadata_path=args.metadata,
        manifest_path=args.manifest,
        accepted_qc_flags=tuple(args.accepted_qc_flags or (1,)),
        year=args.year,
    )


if __name__ == "__main__":
    main()
