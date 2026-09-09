from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import re
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
from tqdm import tqdm

from profile_audit.common import read_table, sha256_file, utc_now, write_json


@dataclass(frozen=True)
class ArchiveSource:
    key: str
    archive_version: str
    start_year: int
    end_year: int
    url: str | None
    filename: str | None
    expected_size: int | None
    doi: str | None
    status: str
    note: str


ARCHIVE_SOURCES = {
    "cora41": ArchiveSource(
        key="cora41",
        archive_version="CORA4.1",
        start_year=1993,
        end_year=2013,
        url="https://www.seanoe.org/data/00351/46219/data/45999.tar.gz",
        filename="45999.tar.gz",
        expected_size=51_397_219_166,
        doi="10.17882/46219#45999",
        status="public_archive",
        note="GLORYS QUID says 2003; 2013 is the probable correction and still requires producer confirmation.",
    ),
    "cora50": ArchiveSource(
        key="cora50",
        archive_version="CORA5.0",
        start_year=2014,
        end_year=2015,
        url="https://www.seanoe.org/data/00351/46219/data/56697.tar",
        filename="56697.tar",
        expected_size=18_416_660_480,
        doi="10.17882/46219#56697",
        status="public_archive",
        note="Documented GLORYS input for 2014-2015.",
    ),
    "cora51": ArchiveSource(
        key="cora51",
        archive_version="CORA5.1",
        start_year=2016,
        end_year=2016,
        url="https://www.seanoe.org/data/00351/46219/data/56698.tar",
        filename="56698.tar",
        expected_size=16_786_544_640,
        doi="10.17882/46219#56698",
        status="public_archive",
        note="Documented GLORYS input for 2016.",
    ),
    "2017_202106": ArchiveSource(
        key="2017_202106",
        archive_version="unresolved",
        start_year=2017,
        end_year=2021,
        url=None,
        filename=None,
        expected_size=None,
        doi=None,
        status="blocked_unresolved_source",
        note="No authoritative GLORYS input archive has been identified for 2017 through June 2021.",
    ),
    "nrt013047": ArchiveSource(
        key="nrt013047",
        archive_version="INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047",
        start_year=2021,
        end_year=2100,
        url=None,
        filename=None,
        expected_size=None,
        doi=None,
        status="blocked_internal_product",
        note=(
            "Internal CMEMS T/S assimilation feed documented from July 2021. "
            "Also named INSITU_GLO_TS_ASSIM_NRT_OBSERVATIONS_013_047; derived "
            "from the Coriolis/In Situ TAC ecosystem, but no public endpoint or "
            "exact selection/thinning specification is available."
        ),
    ),
}

AUTHORITATIVE_REFERENCES = {
    "glorys_quid": "https://documentation.marine.copernicus.eu/QUID/CMEMS-GLO-QUID-001-030.pdf",
    "cora_landing_page": "https://www.seanoe.org/data/00351/46219/",
    "cora_doi": "https://doi.org/10.17882/46219",
    "copernicus_support": "https://marine.copernicus.eu/contact",
    "mercator_contact": "https://www.mercator-ocean.eu/contact-us/",
}

DOWNLOAD_ATTEMPTS = 4


def sources_for_dates(dates: Iterable[pd.Timestamp]) -> list[ArchiveSource]:
    keys: set[str] = set()
    for date in dates:
        if pd.isna(date):
            continue
        if date.year <= 2013:
            keys.add("cora41")
        elif date.year <= 2015:
            keys.add("cora50")
        elif date.year == 2016:
            keys.add("cora51")
        elif date < pd.Timestamp("2021-07-01", tz="UTC"):
            keys.add("2017_202106")
        else:
            keys.add("nrt013047")
    return [ARCHIVE_SOURCES[key] for key in ARCHIVE_SOURCES if key in keys]


def dates_from_profiles(profiles: pd.DataFrame) -> pd.DatetimeIndex:
    if "datetime_utc" in profiles:
        return pd.DatetimeIndex(pd.to_datetime(profiles["datetime_utc"], utc=True, errors="coerce"))
    if "profile_date" in profiles:
        return pd.DatetimeIndex(pd.to_datetime(profiles["profile_date"].astype(str), format="%Y%m%d", utc=True, errors="coerce"))
    raise ValueError("Target profiles need datetime_utc or profile_date.")


def verify_remote_source(source: ArchiveSource) -> dict[str, Any]:
    if source.url is None:
        return {"available": False, "reason": source.status}
    response = requests.head(source.url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    remote_size = int(response.headers["content-length"]) if response.headers.get("content-length", "").isdigit() else None
    return {
        "available": True,
        "resolved_url": response.url,
        "remote_size": remote_size,
        "expected_size_matches": remote_size is None or source.expected_size is None or remote_size == source.expected_size,
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
    }


def _check_disk_space(output_dir: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(output_dir).free
    if free < required_bytes:
        raise OSError(f"Insufficient free space in {output_dir}: need {required_bytes:,} bytes, have {free:,}.")


def download_archive(source: ArchiveSource, output_dir: Path, *, overwrite: bool = False) -> Path:
    if source.url is None or source.filename is None or source.expected_size is None:
        raise ValueError(f"{source.archive_version} has no public frozen download endpoint: {source.note}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and not overwrite:
        if destination.stat().st_size != source.expected_size:
            raise ValueError(f"Existing archive has the wrong size: {destination}")
        return destination
    partial_size = partial.stat().st_size if partial.exists() and not overwrite else 0
    if partial_size > source.expected_size:
        raise ValueError(f"Partial archive is larger than expected: {partial}")
    if partial_size == source.expected_size and not overwrite:
        partial.replace(destination)
        return destination
    _check_disk_space(output_dir, source.expected_size - partial_size)
    for attempt in range(DOWNLOAD_ATTEMPTS):
        offset = partial.stat().st_size if partial.exists() and not overwrite else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(source.url, headers=headers, stream=True, timeout=(60, 300)) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if offset and not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as stream, tqdm(
                    total=source.expected_size,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    desc=source.archive_version,
                    dynamic_ncols=True,
                ) as progress:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            stream.write(block)
                            progress.update(len(block))
            break
        except requests.RequestException as exc:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise requests.ConnectionError(
                    f"Download failed after {DOWNLOAD_ATTEMPTS} attempts; partial file retained at {partial}. "
                    "Rerun the command to resume it."
                ) from exc
            time.sleep(2 ** attempt)
    if partial.stat().st_size != source.expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {source.archive_version}: "
            f"expected {source.expected_size:,}, got {partial.stat().st_size:,}. Partial file retained."
        )
    partial.replace(destination)
    return destination


def _target_days(dates: Iterable[pd.Timestamp], margin_days: int) -> set[str]:
    selected: set[str] = set()
    for date in dates:
        if pd.isna(date):
            continue
        day = date.date()
        for offset in range(-margin_days, margin_days + 1):
            selected.add((day + timedelta(days=offset)).strftime("%Y%m%d"))
    return selected


def _is_xbt_member(name: str) -> bool:
    upper = name.upper()
    return upper.endswith(".NC") and ("PR_XB" in upper or "/XB/" in upper or "_XB_" in upper)


def _is_tar_member(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".tar", ".tar.gz", ".tgz"))


def _archive_stem(name: str) -> str:
    filename = PurePosixPath(name).name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if filename.lower().endswith(suffix):
            return filename[:-len(suffix)]
    return filename


def _member_year(name: str) -> int | None:
    matches = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name)
    return int(matches[-1]) if matches else None


def _safe_relative_path(name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe archive member path: {name}")
    return Path(*parts)


def _write_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    *,
    overwrite: bool,
) -> bool:
    if destination.exists() and not overwrite and destination.stat().st_size == member.size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    source_stream = bundle.extractfile(member)
    if source_stream is None:
        return False
    with source_stream, partial.open("wb") as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream, length=8 * 1024 * 1024)
    partial.replace(destination)
    return True


def _extract_xbt_members(
    bundle: tarfile.TarFile,
    output_dir: Path,
    *,
    selected_days: set[str],
    selected_years: set[int],
    overwrite: bool,
    prefix: Path = Path(),
) -> list[Path]:
    extracted: list[Path] = []
    for member in bundle:
        if not member.isfile():
            continue
        if _is_tar_member(member.name):
            year = _member_year(member.name)
            if selected_years and year is not None and year not in selected_years:
                continue
            source_stream = bundle.extractfile(member)
            if source_stream is None:
                continue
            nested_prefix = prefix / _safe_relative_path(member.name).parent / _archive_stem(member.name)
            with source_stream, tarfile.open(fileobj=source_stream, mode="r|*") as nested:
                extracted.extend(
                    _extract_xbt_members(
                        nested,
                        output_dir,
                        selected_days=selected_days,
                        selected_years=selected_years,
                        overwrite=overwrite,
                        prefix=nested_prefix,
                    )
                )
            continue
        if not _is_xbt_member(member.name):
            continue
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", member.name)
        if selected_days and match is not None and match.group(1) not in selected_days:
            continue
        destination = output_dir / prefix / _safe_relative_path(member.name)
        _write_member(bundle, member, destination, overwrite=overwrite)
        extracted.append(destination)
    return extracted


def extract_archive(
    archive: Path,
    output_dir: Path,
    *,
    mode: str,
    target_dates: Iterable[pd.Timestamp] = (),
    margin_days: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    if mode not in {"all", "xbt"}:
        raise ValueError("Extraction mode must be 'all' or 'xbt'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [date for date in target_dates if not pd.isna(date)]
    selected_days = _target_days(dates, margin_days)
    selected_years = {date.year for date in dates}
    extracted: list[Path] = []
    with tarfile.open(archive, mode="r:*") as bundle:
        if mode == "xbt":
            return _extract_xbt_members(
                bundle,
                output_dir,
                selected_days=selected_days,
                selected_years=selected_years,
                overwrite=overwrite,
            )
        for member in bundle:
            if not member.isfile():
                continue
            relative = _safe_relative_path(member.name)
            destination = output_dir / relative
            _write_member(bundle, member, destination, overwrite=overwrite)
            extracted.append(destination)
    return extracted


def acquire_inputs(
    sources: Sequence[ArchiveSource],
    *,
    output_dir: Path,
    manifest_path: Path,
    target_dates: Iterable[pd.Timestamp] = (),
    extract: str = "none",
    margin_days: int = 1,
    download: bool = True,
    overwrite: bool = False,
    workers: int = 1,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    dates = list(target_dates)
    def acquire_source(source: ArchiveSource) -> dict[str, Any]:
        source_dates = [date for date in dates if not pd.isna(date) and source.start_year <= date.year <= source.end_year]
        record = asdict(source)
        try:
            record["remote"] = verify_remote_source(source)
        except requests.RequestException as exc:
            record["remote"] = {"available": False, "reason": f"remote_check_failed: {exc}"}
        if source.url is None or not download:
            record["acquisition_status"] = source.status if source.url is None else "discovery_only"
            return record
        archive = download_archive(source, output_dir / "archives", overwrite=overwrite)
        record.update({
            "acquisition_status": "downloaded",
            "local_archive": str(archive.resolve()),
            "local_size": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "acquired_utc": utc_now(),
        })
        if extract != "none":
            extracted = extract_archive(
                archive,
                output_dir / "extracted" / source.key,
                mode=extract,
                target_dates=source_dates,
                margin_days=margin_days,
                overwrite=overwrite,
            )
            record["extraction"] = {
                "mode": extract,
                "file_count": len(extracted),
                "root": str((output_dir / "extracted" / source.key).resolve()),
            }
        return record

    selected_sources = list(sources)
    if workers == 1 or len(selected_sources) < 2:
        records = [acquire_source(source) for source in selected_sources]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="profile-audit") as executor:
            futures = [executor.submit(acquire_source, source) for source in selected_sources]
            # Reading in submission order keeps the manifest deterministic.
            records = [future.result() for future in futures]
    return write_json(
        {
            "created_utc": utc_now(),
            "warning": "Public release archives establish candidate input membership, not profile-level GLORYS acceptance.",
            "authoritative_references": AUTHORITATIVE_REFERENCES,
            "unresolved_action": "Ask Mercator Ocean for the 2017-June 2021 source, frozen 013_047 vintages, and profile-level used/rejected feedback.",
            "sources": records,
        },
        manifest_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and acquire frozen GLORYS in-situ input archives.")
    parser.add_argument("--release", choices=tuple(ARCHIVE_SOURCES), action="append")
    parser.add_argument("--from-profiles", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract", choices=("none", "xbt", "all"), default="none")
    parser.add_argument("--date-margin-days", type=int, default=1)
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=3, help="Process independent archives concurrently (default: 3).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dates = dates_from_profiles(read_table(args.from_profiles)) if args.from_profiles else pd.DatetimeIndex([])
    selected = [ARCHIVE_SOURCES[key] for key in args.release] if args.release else sources_for_dates(dates)
    if not selected:
        raise SystemExit("Select --release or provide --from-profiles with valid dates.")
    manifest = args.manifest or args.output_dir / "assimilation_input_inventory.json"
    print(acquire_inputs(
        selected,
        output_dir=args.output_dir,
        manifest_path=manifest,
        target_dates=dates,
        extract=args.extract,
        margin_days=args.date_margin_days,
        download=not args.discovery_only,
        overwrite=args.overwrite,
        workers=args.workers,
    ))


if __name__ == "__main__":
    main()
