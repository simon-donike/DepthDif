from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import xarray as xr
from h5py import is_hdf5


def _path_from_depth_v2(path: Path) -> str:
    """Return path string anchored at the `depth_v2` segment when present."""
    resolved = path.resolve()
    parts = resolved.parts
    if "depth_v2" not in parts:
        # Fallback keeps current behavior when datasets are stored outside the expected tree.
        return resolved.as_posix()
    depth_v2_idx = parts.index("depth_v2")
    return Path(*parts[depth_v2_idx:]).as_posix()


def _scan_ostia_by_month(ostia_dir: Path) -> dict[str, list[tuple[int, str]]]:
    files = sorted(
        ostia_dir.glob("*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc")
    )
    if not files:
        raise RuntimeError(f"No OSTIA files found in: {ostia_dir}")

    out: dict[str, list[tuple[int, str]]] = {}
    for path in files:
        m = re.match(r"(\d{8})\d{6}-", path.name)
        if m is None:
            continue
        date_yyyymmdd = int(m.group(1))
        month_key = m.group(1)[:6]
        out.setdefault(month_key, []).append((date_yyyymmdd, _path_from_depth_v2(path)))

    for month_key in out:
        out[month_key].sort(key=lambda x: x[0])
    return out


def _parse_argo_month(path: Path) -> str | None:
    m = re.search(r"g10\.(\d{6})\.nc$", path.name)
    return m.group(1) if m is not None else None


def _juld_to_yyyymmdd(juld_days: np.ndarray) -> np.ndarray:
    out = np.zeros(juld_days.shape, dtype=np.int32)
    valid = np.isfinite(juld_days) & (juld_days < 90000.0) & (juld_days > -20000.0)
    if not np.any(valid):
        return out

    base = np.datetime64("1950-01-01", "D")
    days = np.floor(juld_days[valid]).astype(np.int64)
    dates = base + days.astype("timedelta64[D]")
    date_str = np.datetime_as_string(dates, unit="D")
    compact = np.char.replace(date_str, "-", "")
    out[valid] = compact.astype(np.int32)
    return out


def _match_ostia_days(
    profile_dates: np.ndarray,
    ostia_rows: list[tuple[int, str]],
) -> dict[int, tuple[int, str]]:
    if not ostia_rows:
        return {}
    ostia_days = np.asarray([r[0] for r in ostia_rows], dtype=np.int32)
    ostia_paths = [r[1] for r in ostia_rows]

    mapping: dict[int, tuple[int, str]] = {}
    unique_dates = np.unique(profile_dates[profile_dates > 0])
    for d in unique_dates.tolist():
        # Per-month nearest-date match keeps the temporal lookup deterministic.
        nearest_idx = int(np.argmin(np.abs(ostia_days.astype(np.int64) - int(d))))
        mapping[int(d)] = (int(ostia_days[nearest_idx]), ostia_paths[nearest_idx])
    return mapping


def build_argo_datetime_match_csv(
    *,
    argo_dir: Path,
    ostia_dir: Path,
    output_csv: Path,
) -> Path:
    argo_files = sorted(argo_dir.glob("EN.4.2.2.f.profiles.g10.*.nc"))
    if not argo_files:
        raise RuntimeError(f"No EN4 profile files found in: {argo_dir}")

    ostia_by_month = _scan_ostia_by_month(ostia_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row_id = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "argo_row_id",
                "argo_month_key",
                "argo_profile_date",
                "profile_idx",
                "argo_file_path",
                "matched_ostia_date",
                "matched_ostia_file_path",
            ]
        )

        for argo_path in argo_files:
            if not is_hdf5(str(argo_path)):
                # Skip corrupted/non-HDF5 .nc files to keep indexing robust.
                continue

            month_key = _parse_argo_month(argo_path)
            if month_key is None:
                continue

            with xr.open_dataset(
                argo_path,
                engine="h5netcdf",
                decode_times=False,
                cache=False,
            ) as ds:
                if "JULD" not in ds:
                    continue
                profile_dates = _juld_to_yyyymmdd(
                    np.asarray(ds["JULD"].to_numpy(), dtype=np.float64)
                )

            date_to_ostia = _match_ostia_days(
                profile_dates=profile_dates,
                ostia_rows=ostia_by_month.get(month_key, []),
            )

            for profile_idx, d in enumerate(profile_dates.tolist()):
                if int(d) <= 0:
                    continue
                matched_date, matched_path = date_to_ostia.get(int(d), (0, ""))
                writer.writerow(
                    [
                        row_id,
                        month_key,
                        int(d),
                        int(profile_idx),
                        _path_from_depth_v2(argo_path),
                        int(matched_date),
                        matched_path,
                    ]
                )
                row_id += 1

    return output_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CSV of EN4/Argo profile datetimes and nearest matching OSTIA daily files."
        )
    )
    parser.add_argument(
        "--argo-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/en4_profiles"),
        help="Directory containing EN4 profile files.",
    )
    parser.add_argument(
        "--ostia-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/ostia"),
        help="Directory containing OSTIA daily files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/data1/datasets/depth_v2/argo_profile_datetime_match.csv"),
        help="Output CSV path for Argo datetime <-> OSTIA daily matching.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    out = build_argo_datetime_match_csv(
        argo_dir=args.argo_dir,
        ostia_dir=args.ostia_dir,
        output_csv=args.output_csv,
    )
    print(f"Wrote Argo datetime match CSV: {out}")


if __name__ == "__main__":
    main()
