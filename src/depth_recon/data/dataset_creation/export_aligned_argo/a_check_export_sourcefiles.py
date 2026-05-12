"""
Default overlap-range check using the standard /data1/datasets/depth_v2 paths:
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles

Explicit full-path equivalent:
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles \
  --argo-dir /data1/datasets/depth_v2/en4_profiles \
  --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
  --start-date 20100101 \
  --end-date 20240731

Quick ARGO-only check:
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles \
  --include argo

Repair broken files after confirmation:
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles --repair
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from depth_recon.data.dataset_creation.export_aligned_argo.source_files import (
    ARGO_DEPTH_VAR,
    ARGO_PROFILE_VARS,
    GLORYS_2D_VARS,
    GLORYS_3D_VARS,
    OSTIA_VARS,
    SEALEVEL_VARS,
    TimedFile,
    _date_to_days_since_1950,
    _filter_argo_files_by_date_range,
    _open_argo_dataset,
    _parse_argo_file_month,
    scan_timed_files,
)

SOURCE_KINDS = ("argo", "glorys", "ostia", "sealevel")
ARGO_BASE_URL = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1"
GLORYS_DATASET_CANDIDATES = (
    "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311",
    "global-reanalysis-phy-001-030-daily",
)
OSTIA_DATASET_CANDIDATES = (
    "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    "METOFFICE-GLO-SST-L4-REP-OBS-SST-V2",
    "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
)
SEALEVEL_DATASET_CANDIDATES = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
)


@dataclass(frozen=True)
class SourceFile:
    kind: str
    path: Path
    date: int | None


@dataclass(frozen=True)
class BrokenFile:
    source: SourceFile
    reason: str


def _yyyymmdd_to_iso(date_yyyymmdd: int) -> str:
    text = str(int(date_yyyymmdd))
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _day_to_yyyymmdd(day: float) -> int:
    date = np.datetime64("1950-01-01", "D") + np.timedelta64(int(np.floor(day)), "D")
    return int(np.datetime_as_string(date, unit="D").replace("-", ""))


def _parse_date_from_timed_file(item: TimedFile) -> int:
    return _day_to_yyyymmdd(item.day)


def _select_timed_files_for_range(
    index: list[TimedFile],
    *,
    start_date: int,
    end_date: int,
) -> list[TimedFile]:
    if not index:
        return []
    start_day = _date_to_days_since_1950(start_date)
    end_day = _date_to_days_since_1950(end_date)
    selected: list[TimedFile] = [
        item for item in index if start_day <= item.day <= end_day
    ]
    before = [item for item in index if item.day < start_day]
    after = [item for item in index if item.day > end_day]
    if before:
        selected.insert(0, before[-1])
    if after:
        selected.append(after[0])

    seen: set[Path] = set()
    out: list[TimedFile] = []
    for item in selected:
        if item.path in seen:
            continue
        seen.add(item.path)
        out.append(item)
    return out


def _read_probe(da: xr.DataArray) -> None:
    indexers = {dim: 0 for dim in da.dims if int(da.sizes[dim]) > 0}
    if len(indexers) != len(da.dims):
        raise RuntimeError(f"{da.name} has an empty dimension")
    np.asarray(da.isel(indexers).values)


def _check_required_vars(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise RuntimeError(f"missing variables: {missing}")
    for name in names:
        _read_probe(ds[name])


def check_argo_file(path: Path) -> None:
    with _open_argo_dataset(path) as ds:
        required = ("JULD", "LATITUDE", "LONGITUDE", ARGO_DEPTH_VAR) + ARGO_PROFILE_VARS
        _check_required_vars(ds, required)


def check_gridded_file(
    path: Path,
    names: tuple[str, ...],
    *,
    needs_depth: bool = False,
) -> None:
    with xr.open_dataset(
        path,
        engine="h5netcdf",
        decode_times=False,
        mask_and_scale=True,
        cache=False,
    ) as ds:
        if needs_depth and "depth" not in ds:
            raise RuntimeError("missing depth coordinate")
        _check_required_vars(ds, names)


def check_source_file(source: SourceFile) -> None:
    if source.kind == "argo":
        check_argo_file(source.path)
    elif source.kind == "glorys":
        check_gridded_file(
            source.path,
            GLORYS_3D_VARS + GLORYS_2D_VARS,
            needs_depth=True,
        )
    elif source.kind == "ostia":
        check_gridded_file(source.path, OSTIA_VARS)
    elif source.kind == "sealevel":
        check_gridded_file(source.path, SEALEVEL_VARS)
    else:
        raise RuntimeError(f"unknown source kind: {source.kind}")


def collect_sources(args: argparse.Namespace) -> list[SourceFile]:
    sources: list[SourceFile] = []
    requested = set(args.include)

    if "argo" in requested:
        argo_files = _filter_argo_files_by_date_range(
            sorted(args.argo_dir.glob("EN.4.2.2.f.profiles.g10.*.nc")),
            start_date=args.start_date,
            end_date=args.end_date,
        )
        for path in argo_files:
            month = _parse_argo_file_month(path)
            date = int(f"{month}01") if month is not None else None
            sources.append(SourceFile("argo", path, date))

    for kind, root in (
        ("glorys", args.glorys_dir),
        ("ostia", args.ostia_dir),
        ("sealevel", args.sealevel_dir),
    ):
        if kind not in requested:
            continue
        selected = _select_timed_files_for_range(
            scan_timed_files(root),
            start_date=args.start_date,
            end_date=args.end_date,
        )
        sources.extend(
            SourceFile(kind, item.path, _parse_date_from_timed_file(item))
            for item in selected
        )

    if args.max_files is not None:
        sources = sources[: int(args.max_files)]
    return sources


def check_sources(sources: list[SourceFile]) -> list[BrokenFile]:
    broken: list[BrokenFile] = []
    totals = {kind: 0 for kind in SOURCE_KINDS}
    progress = tqdm(
        sources,
        desc="Checking source files",
        unit="file",
        dynamic_ncols=True,
    )
    for source in progress:
        totals[source.kind] += 1
        progress.set_postfix(kind=source.kind, broken=len(broken), refresh=False)
        try:
            check_source_file(source)
        except Exception as exc:
            broken.append(BrokenFile(source, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"BROKEN\t{source.kind}\t{source.path}\t{broken[-1].reason}")

    print()
    print("Checked source files:")
    for kind in SOURCE_KINDS:
        if totals[kind]:
            print(f"- {kind}: {totals[kind]}")
    print(f"Broken files: {len(broken)}")
    for item in broken:
        print(
            "BROKEN\t"
            f"{item.source.kind}\t{item.source.date}\t"
            f"{item.source.path}\t{item.reason}"
        )
    return broken


def _copernicus_cmd() -> str:
    found = shutil.which("copernicusmarine")
    if found is not None:
        return found
    fallback = Path("/work/envs/depth/bin/copernicusmarine")
    if fallback.exists():
        return fallback.as_posix()
    raise RuntimeError("could not find copernicusmarine CLI")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True)


def _replace_checked_file(
    tmp_path: Path, target_path: Path, source: SourceFile
) -> None:
    tmp_source = SourceFile(source.kind, tmp_path, source.date)
    check_source_file(tmp_source)
    tmp_path.replace(target_path)


def repair_argo_file(source: SourceFile, *, base_url: str) -> None:
    month = _parse_argo_file_month(source.path)
    if month is None:
        raise RuntimeError(f"cannot infer EN4 month from filename: {source.path.name}")
    year = str(month)[:4]
    archive_name = f"EN.4.2.2.profiles.g10.{year}.zip"
    url = f"{base_url.rstrip('/')}/{archive_name}"
    member_suffix = source.path.name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        archive_path = tmp_root / archive_name
        extracted_path = tmp_root / source.path.name
        tqdm.write(f"  downloading {url}")
        _run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--progress-bar",
                url,
                "-o",
                archive_path.as_posix(),
            ]
        )
        with zipfile.ZipFile(archive_path) as zf:
            matches = [
                name for name in zf.namelist() if Path(name).name == member_suffix
            ]
            if not matches:
                raise RuntimeError(f"{archive_name} did not contain {member_suffix}")
            with zf.open(matches[0]) as src, extracted_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        _replace_checked_file(extracted_path, source.path, source)


def _date_filter(source: SourceFile, kind: str) -> str:
    if source.date is None:
        raise RuntimeError(f"cannot infer date for {source.path}")
    day = str(int(source.date))
    year = day[:4]
    month = day[4:6]
    if kind == "ostia":
        return (
            f"*/{year}/{month}/*{day}120000-"
            "UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc"
        )
    return f"*/{year}/{month}/*{day}*.nc"


def _dataset_candidates(kind: str) -> tuple[str, ...]:
    if kind == "glorys":
        return GLORYS_DATASET_CANDIDATES
    if kind == "ostia":
        return OSTIA_DATASET_CANDIDATES
    if kind == "sealevel":
        return SEALEVEL_DATASET_CANDIDATES
    raise RuntimeError(f"no Copernicus dataset candidates for {kind}")


def repair_copernicus_file(source: SourceFile) -> None:
    cmd = _copernicus_cmd()
    file_filter = _date_filter(source, source.kind)
    day_tag = str(int(source.date)) if source.date is not None else ""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for dataset_id in _dataset_candidates(source.kind):
            tqdm.write(
                f"  trying {source.kind} dataset {dataset_id} with filter {file_filter}"
            )
            result = subprocess.run(
                [
                    cmd,
                    "get",
                    "-i",
                    dataset_id,
                    "--filter",
                    file_filter,
                    "-o",
                    tmp_root.as_posix(),
                    "-nd",
                    "--log-level",
                    "ERROR",
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                continue
            downloaded = sorted(tmp_root.glob(f"*{day_tag}*.nc"))
            if not downloaded:
                continue
            _replace_checked_file(downloaded[0], source.path, source)
            return
        raise RuntimeError(f"could not redownload {source.kind} file for {source.path}")


def repair_broken_files(broken: list[BrokenFile], *, args: argparse.Namespace) -> None:
    if not broken:
        return
    print()
    print(f"About to redownload and overwrite {len(broken)} broken files.")
    answer = input("Type 'yes' to continue: ").strip()
    if answer != "yes":
        print("Repair cancelled.")
        return

    progress = tqdm(
        broken,
        desc="Repairing source files",
        unit="file",
        dynamic_ncols=True,
    )
    for item in progress:
        source = item.source
        progress.set_postfix(kind=source.kind, refresh=False)
        tqdm.write(f"repairing {source.kind}: {source.path}")
        if source.kind == "argo":
            repair_argo_file(source, base_url=args.argo_base_url)
        else:
            repair_copernicus_file(source)
        tqdm.write("  repaired")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check enriched-profile export source NetCDF files and optionally redownload broken files."
    )
    parser.add_argument(
        "--argo-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/en4_profiles"),
    )
    parser.add_argument(
        "--glorys-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/glorys_weekly"),
    )
    parser.add_argument(
        "--ostia-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/ostia"),
    )
    parser.add_argument(
        "--sealevel-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/sealevel_daily"),
    )
    parser.add_argument(
        "--start-date",
        type=int,
        default=20100101,
        help="YYYYMMDD inclusive start.",
    )
    parser.add_argument(
        "--end-date",
        type=int,
        default=20240731,
        help="YYYYMMDD inclusive end.",
    )
    parser.add_argument(
        "--include",
        choices=SOURCE_KINDS,
        nargs="+",
        default=list(SOURCE_KINDS),
        help="Source groups to check.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional smoke-test cap.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Prompt to redownload and overwrite broken files.",
    )
    parser.add_argument("--argo-base-url", default=ARGO_BASE_URL)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be >= --start-date")
    # Validate date spelling before doing any filesystem or network work.
    datetime.fromisoformat(_yyyymmdd_to_iso(args.start_date))
    datetime.fromisoformat(_yyyymmdd_to_iso(args.end_date))

    sources = collect_sources(args)
    broken = check_sources(sources)
    if args.repair:
        repair_broken_files(broken, args=args)


if __name__ == "__main__":
    main()
