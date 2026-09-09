from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from profile_audit.common import utc_now, write_json

PRODUCT_ID = "INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030"
DATASET_ID = "cmems_obs-ins_glo_phybgcwav_mynrt_na_irr"
DATASET_VERSION = "202311"


def _run_copernicusmarine(arguments: list[str]) -> None:
    executable = shutil.which("copernicusmarine")
    if executable is None:
        raise RuntimeError("copernicusmarine is not installed in the active environment.")
    subprocess.run([executable, *arguments], check=True)


def download_indexes(output_dir: Path, *, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        "get",
        "--dataset-id", DATASET_ID,
        "--dataset-version", DATASET_VERSION,
        "--index-parts",
        "--no-directories",
        "--output-directory", str(output_dir),
    ]
    arguments.append("--overwrite" if overwrite else "--skip-existing")
    _run_copernicusmarine(arguments)
    return output_dir


def _profile_columns(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        available = set(pd.read_parquet(path, columns=[]).columns)
        # PyArrow returns no names for columns=[]; inspect the schema directly.
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in ("datetime_utc", "profile_date", "latitude", "longitude", "longitude_180") if name in available]
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path)


def _target_frame(path: Path) -> pd.DataFrame:
    profiles = _profile_columns(path)
    if "datetime_utc" in profiles:
        timestamp = pd.to_datetime(profiles["datetime_utc"], utc=True, errors="coerce")
    elif "profile_date" in profiles:
        timestamp = pd.to_datetime(profiles["profile_date"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    else:
        raise ValueError("Target profiles need datetime_utc or profile_date.")
    longitude_name = "longitude_180" if "longitude_180" in profiles else "longitude"
    if longitude_name not in profiles or "latitude" not in profiles:
        raise ValueError("Target profiles need latitude and longitude.")
    targets = pd.DataFrame({
        "time": timestamp,
        "latitude": pd.to_numeric(profiles["latitude"], errors="coerce"),
        "longitude": ((pd.to_numeric(profiles[longitude_name], errors="coerce") + 180) % 360) - 180,
    }).dropna()
    return targets[targets["time"] >= pd.Timestamp("2017-01-01", tz="UTC")].reset_index(drop=True)


def _index_chunks(path: Path, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    column_names: list[str] | None = None
    with Path(path).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.lower().startswith("# product_id,"):
                column_names = [name.strip() for name in stripped[1:].strip().split(",")]
                break
            if stripped and not stripped.startswith("#"):
                break
    yield from pd.read_csv(
        path,
        comment="#",
        names=column_names,
        header=None if column_names is not None else "infer",
        chunksize=chunk_size,
    )


def select_proxy_files(index_path: Path, targets: pd.DataFrame, output: Path) -> list[str]:
    if targets.empty:
        raise ValueError("No target profiles on or after 2017 were supplied for proxy screening.")
    selected: set[str] = set()
    global_screen = len(targets) > 20_000
    target_start = targets["time"].min() - pd.Timedelta(days=1)
    target_end = targets["time"].max() + pd.Timedelta(days=1)

    for chunk in _index_chunks(index_path):
        required = {
            "file_name", "geospatial_lat_min", "geospatial_lat_max",
            "geospatial_lon_min", "geospatial_lon_max", "time_coverage_start",
            "time_coverage_end", "parameters",
        }
        missing = required - set(chunk)
        if missing:
            raise ValueError(f"013_030 index is missing columns: {sorted(missing)}")
        names = chunk["file_name"].astype(str)
        starts = pd.to_datetime(chunk["time_coverage_start"], utc=True, errors="coerce")
        ends = pd.to_datetime(chunk["time_coverage_end"], utc=True, errors="coerce")
        mask = (
            names.str.contains("_PR_", regex=False)
            & chunk["parameters"].fillna("").str.contains(r"(?:^|\s)TEMP(?:\s|$)", regex=True)
            & (ends >= target_start)
            & (starts <= target_end)
        ).to_numpy()
        if not global_screen and np.any(mask):
            lat_min = pd.to_numeric(chunk["geospatial_lat_min"], errors="coerce").to_numpy(float)
            lat_max = pd.to_numeric(chunk["geospatial_lat_max"], errors="coerce").to_numpy(float)
            lon_min = pd.to_numeric(chunk["geospatial_lon_min"], errors="coerce").to_numpy(float)
            lon_max = pd.to_numeric(chunk["geospatial_lon_max"], errors="coerce").to_numpy(float)
            starts_ns = starts.astype("int64").to_numpy()
            ends_ns = ends.astype("int64").to_numpy()
            spatial_temporal = np.zeros(len(chunk), dtype=bool)
            for target in targets.itertuples(index=False):
                target_ns = pd.Timestamp(target.time).value
                time_match = (starts_ns <= target_ns + 86_400 * 10**9) & (ends_ns >= target_ns - 86_400 * 10**9)
                lat_match = (lat_min <= target.latitude + 0.25) & (lat_max >= target.latitude - 0.25)
                direct_lon = (lon_min <= target.longitude + 0.5) & (lon_max >= target.longitude - 0.5)
                wrapped_lon = (lon_min <= target.longitude + 360.5) & (lon_max >= target.longitude + 359.5) | (lon_min <= target.longitude - 359.5) & (lon_max >= target.longitude - 360.5)
                spatial_temporal |= time_match & lat_match & (direct_lon | wrapped_lon)
            mask &= spatial_temporal
        selected.update(names[mask])
    paths = sorted(selected)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    return paths


def download_selected_files(file_list: Path, output_dir: Path, *, part: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_copernicusmarine(
        [
            "get",
            "--dataset-id", DATASET_ID,
            "--dataset-version", DATASET_VERSION,
            "--dataset-part", part,
            "--file-list", str(file_list),
            "--output-directory", str(output_dir),
            "--skip-existing",
        ]
    )
    return output_dir


def acquire_proxy(
    *,
    profiles: Path,
    output_dir: Path,
    part: str = "history",
    download_files: bool = False,
    refresh_indexes: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    index_dir = output_dir / "indexes"
    index_path = index_dir / f"index_{part}.txt"
    if refresh_indexes or not index_path.exists():
        download_indexes(index_dir, overwrite=refresh_indexes)
    targets = _target_frame(profiles)
    file_list = output_dir / f"selected_{part}_profile_files.txt"
    selected = select_proxy_files(index_path, targets, file_list)
    files_dir = output_dir / "files" / part
    if download_files and selected:
        download_selected_files(file_list, files_dir, part=part)
    return write_json(
        {
            "created_utc": utc_now(),
            "product_id": PRODUCT_ID,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_part": part,
            "doi": "10.48670/moi-00036",
            "index_path": str(index_path.resolve()),
            "selected_file_list": str(file_list.resolve()),
            "selected_file_count": len(selected),
            "files_downloaded": bool(download_files and selected),
            "files_root": str(files_dir.resolve()),
            "target_profile_count": len(targets),
            "target_start": targets["time"].min().isoformat() if not targets.empty else None,
            "target_end": targets["time"].max().isoformat() if not targets.empty else None,
            "evidence_scope": "current_013030_proxy_nonhistorical",
            "warning": (
                "This evolving current product is not a frozen parent snapshot of internal 013_047. "
                "Positive matches are supporting evidence; negative matches do not prove historical absence."
            ),
        },
        output_dir / "proxy_inventory.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire an explicitly non-historical current 013_030 proxy screen.")
    parser.add_argument("--from-profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", choices=("latest", "monthly", "history"), default="history")
    parser.add_argument("--download-files", action="store_true")
    parser.add_argument("--refresh-indexes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    print(acquire_proxy(
        profiles=args.from_profiles,
        output_dir=args.output_dir,
        part=args.part,
        download_files=args.download_files,
        refresh_indexes=args.refresh_indexes,
    ))


if __name__ == "__main__":
    main()
