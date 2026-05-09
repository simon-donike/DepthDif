from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

MISSING_TEXT = "__missing__"


def _parse_mask_flag_metadata(mask_attrs: dict[str, object]) -> dict[str, int]:
    """Map OSTIA mask flag names to bit values from the NetCDF metadata."""
    raw_meanings = str(mask_attrs.get("flag_meanings", "")).strip()
    raw_masks = np.asarray(mask_attrs.get("flag_masks", ()), dtype=np.int64).reshape(-1)
    if raw_meanings == "" or raw_masks.size == 0:
        raise RuntimeError(
            "OSTIA mask variable is missing flag_meanings/flag_masks metadata."
        )

    flag_names = [part.strip() for part in raw_meanings.split() if part.strip()]
    if len(flag_names) != int(raw_masks.size):
        raise RuntimeError(
            "OSTIA mask flag metadata is inconsistent: "
            f"{len(flag_names)} meanings vs {int(raw_masks.size)} masks."
        )
    return {
        flag_name: int(flag_mask)
        for flag_name, flag_mask in zip(flag_names, raw_masks.tolist())
    }


def _parse_invalid_mask_flags(
    raw: str,
    *,
    allowed_flags: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Parse comma-separated OSTIA mask flags that should count as invalid."""
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError("invalid_mask_flags cannot be empty.")
    if allowed_flags is not None:
        invalid_names = [part for part in parts if part not in allowed_flags]
        if invalid_names:
            allowed = ", ".join(sorted(allowed_flags))
            invalid = ", ".join(invalid_names)
            raise ValueError(
                f"Unknown OSTIA mask flag(s): {invalid}. Allowed values: {allowed}."
            )
    return parts


@dataclass
class PatchRecord:
    """Patch-level metadata that is replicated across all OSTIA timesteps."""

    patch_id: int
    lat0: float
    lat1: float
    lon0: float
    lon1: float
    lat_center: float
    lon_center: float
    invalid_fraction: float
    invalid_percentage: float
    phase: str


def _ostia_files(ostia_dir: Path) -> list[Path]:
    files = sorted(
        ostia_dir.glob("*-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.nc")
    )
    if not files:
        raise RuntimeError(f"No OSTIA files found in: {ostia_dir}")
    return files


def _parse_ostia_date_yyyymmdd(path: Path) -> int:
    m = re.match(r"(\d{8})\d{6}-", path.name)
    if m is None:
        raise RuntimeError(f"Cannot parse OSTIA date from filename: {path.name}")
    return int(m.group(1))


def _parse_glorys_date_yyyymmdd(path: Path) -> int:
    m = re.search(r"_(\d{8})(?:_R\d{8})?\.nc$", path.name)
    if m is None:
        raise RuntimeError(f"Cannot parse GLORYS date from filename: {path.name}")
    return int(m.group(1))


def _path_from_depth_v2(path: Path) -> str:
    """Return path string anchored at the `depth_v2` segment when present."""
    resolved = path.resolve()
    parts = resolved.parts
    if "depth_v2" not in parts:
        # Fallback keeps current behavior when datasets are stored outside the expected tree.
        return resolved.as_posix()
    depth_v2_idx = parts.index("depth_v2")
    return Path(*parts[depth_v2_idx:]).as_posix()


def _stable_text(value: str) -> str:
    """Write one explicit sentinel for missing text to avoid mixed CSV dtype inference."""
    text = str(value).strip()
    return text if text else MISSING_TEXT


def _scan_glorys_weekly(glorys_dir: Path) -> list[tuple[int, str]]:
    files = sorted(glorys_dir.glob("*.nc"))
    if not files:
        raise RuntimeError(f"No GLORYS weekly files found in: {glorys_dir}")

    out: list[tuple[int, str]] = []
    for path in files:
        out.append((_parse_glorys_date_yyyymmdd(path), _path_from_depth_v2(path)))
    out.sort(key=lambda item: item[0])
    return out


def _match_nearest_days(
    query_dates: np.ndarray,
    candidate_rows: list[tuple[int, str]],
) -> dict[int, tuple[int, str, int]]:
    if not candidate_rows:
        return {}

    candidate_days = np.asarray([row[0] for row in candidate_rows], dtype=np.int32)
    candidate_paths = [row[1] for row in candidate_rows]

    mapping: dict[int, tuple[int, str, int]] = {}
    unique_dates = np.unique(query_dates[query_dates > 0])
    for day in unique_dates.tolist():
        nearest_idx = int(np.argmin(np.abs(candidate_days.astype(np.int64) - int(day))))
        matched_day = int(candidate_days[nearest_idx])
        mapping[int(day)] = (
            matched_day,
            candidate_paths[nearest_idx],
            int(abs(matched_day - int(day))),
        )
    return mapping


def _build_patch_grid(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    patch_span_deg: float,
) -> list[tuple[float, float, float, float]]:
    grid: list[tuple[float, float, float, float]] = []

    lat0 = float(lat_min)
    while lat0 + patch_span_deg <= float(lat_max) + 1e-9:
        lon0 = float(lon_min)
        while lon0 + patch_span_deg <= float(lon_max) + 1e-9:
            lat1 = lat0 + patch_span_deg
            lon1 = lon0 + patch_span_deg
            grid.append((lat0, lat1, lon0, lon1))
            lon0 += patch_span_deg
        lat0 += patch_span_deg

    if not grid:
        raise RuntimeError(
            "Patch grid is empty. Check span/resolution and lat/lon bounds."
        )
    return grid


def _patch_axes(
    *,
    lat0: float,
    lon0: float,
    tile_size: int,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Axes are pixel centers; each pixel represents resolution_deg x resolution_deg.
    half = 0.5 * float(resolution_deg)
    lat_axis = (
        lat0 + half + (np.arange(tile_size, dtype=np.float64) * float(resolution_deg))
    )
    lon_axis = (
        lon0 + half + (np.arange(tile_size, dtype=np.float64) * float(resolution_deg))
    )
    return lat_axis, lon_axis


def _split_phases(
    *,
    invalid_fractions: np.ndarray,
    invalid_threshold: float,
    val_fraction: float,
    split_seed: int,
) -> list[str]:
    phases = np.full(invalid_fractions.shape, "invalid", dtype=object)
    water_idx = np.flatnonzero(invalid_fractions <= float(invalid_threshold))
    if water_idx.size == 0:
        return phases.tolist()

    n_val = int(round(float(water_idx.size) * float(val_fraction)))
    if water_idx.size > 1:
        n_val = min(max(n_val, 1 if val_fraction > 0.0 else 0), int(water_idx.size) - 1)
    else:
        n_val = 0

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(water_idx)
    val_set = set(perm[:n_val].tolist())

    for idx in water_idx.tolist():
        phases[idx] = "val" if idx in val_set else "train"
    return phases.tolist()


def _classify_patches_from_reference(
    *,
    reference_ostia_path: Path,
    patch_grid: Iterable[tuple[float, float, float, float]],
    tile_size: int,
    resolution_deg: float,
    invalid_threshold: float,
    val_fraction: float,
    split_seed: int,
    invalid_mask_flags: tuple[str, ...] = ("land",),
) -> list[PatchRecord]:
    with xr.open_dataset(
        reference_ostia_path,
        engine="h5netcdf",
        decode_times=False,
        cache=False,
    ) as ds:
        lat = np.asarray(ds["lat"].to_numpy(), dtype=np.float32)
        lon = np.asarray(ds["lon"].to_numpy(), dtype=np.float32)
        mask = np.asarray(ds["mask"].isel(time=0).to_numpy(), dtype=np.float32)
        mask_flag_bits = _parse_mask_flag_metadata(dict(ds["mask"].attrs))
        invalid_mask_flags = _parse_invalid_mask_flags(
            ",".join(invalid_mask_flags),
            allowed_flags=mask_flag_bits,
        )

    mask_interp = RegularGridInterpolator(
        (lat, lon),
        mask,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    invalid_fractions: list[float] = []
    patch_boxes = list(patch_grid)
    invalid_flag_mask = np.int64(0)
    for flag_name in invalid_mask_flags:
        invalid_flag_mask |= np.int64(mask_flag_bits[flag_name])

    for lat0, _, lon0, _ in patch_boxes:
        lat_axis, lon_axis = _patch_axes(
            lat0=lat0,
            lon0=lon0,
            tile_size=tile_size,
            resolution_deg=resolution_deg,
        )
        mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
        query_points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

        mask_patch = mask_interp(query_points).reshape(tile_size, tile_size)
        mask_patch_i = np.rint(mask_patch).astype(np.int64, copy=False)

        invalid = (~np.isfinite(mask_patch)) | ((mask_patch_i & invalid_flag_mask) != 0)
        invalid_fractions.append(float(np.mean(invalid)))

    invalid_fractions_np = np.asarray(invalid_fractions, dtype=np.float64)
    phases = _split_phases(
        invalid_fractions=invalid_fractions_np,
        invalid_threshold=invalid_threshold,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    records: list[PatchRecord] = []
    for patch_id, (bbox, inv_frac, phase) in enumerate(
        zip(patch_boxes, invalid_fractions_np.tolist(), phases)
    ):
        lat0, lat1, lon0, lon1 = bbox
        records.append(
            PatchRecord(
                patch_id=int(patch_id),
                lat0=float(lat0),
                lat1=float(lat1),
                lon0=float(lon0),
                lon1=float(lon1),
                lat_center=float(0.5 * (lat0 + lat1)),
                lon_center=float(0.5 * (lon0 + lon1)),
                invalid_fraction=float(inv_frac),
                invalid_percentage=float(inv_frac * 100.0),
                phase=str(phase),
            )
        )
    return records


def _write_patch_spatial_csv(path: Path, records: list[PatchRecord]) -> None:
    def _bbox_wkt_wgs84(rec: PatchRecord) -> str:
        lat_lo = min(rec.lat0, rec.lat1)
        lat_hi = max(rec.lat0, rec.lat1)
        lon_lo = min(rec.lon0, rec.lon1)
        lon_hi = max(rec.lon0, rec.lon1)
        # WKT coordinate order is x/y, i.e., lon/lat in EPSG:4326 (WGS84).
        return (
            "POLYGON (("
            f"{lon_lo:.6f} {lat_lo:.6f}, "
            f"{lon_hi:.6f} {lat_lo:.6f}, "
            f"{lon_hi:.6f} {lat_hi:.6f}, "
            f"{lon_lo:.6f} {lat_hi:.6f}, "
            f"{lon_lo:.6f} {lat_lo:.6f}"
            "))"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patch_id",
                "lat_center",
                "lon_center",
                "lat0",
                "lat1",
                "lon0",
                "lon1",
                "invalid_fraction",
                "invalid_percentage",
                "phase",
                "wkt",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.patch_id,
                    f"{rec.lat_center:.6f}",
                    f"{rec.lon_center:.6f}",
                    f"{rec.lat0:.6f}",
                    f"{rec.lat1:.6f}",
                    f"{rec.lon0:.6f}",
                    f"{rec.lon1:.6f}",
                    f"{rec.invalid_fraction:.6f}",
                    f"{rec.invalid_percentage:.4f}",
                    rec.phase,
                    _bbox_wkt_wgs84(rec),
                ]
            )


def _write_patch_daily_csv(
    path: Path,
    records: list[PatchRecord],
    ostia_files: list[Path],
    glorys_rows: list[tuple[int, str]],
) -> None:
    ostia_days = np.asarray(
        [_parse_ostia_date_yyyymmdd(ostia_path) for ostia_path in ostia_files],
        dtype=np.int32,
    )
    date_to_glorys = _match_nearest_days(ostia_days, glorys_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patch_id",
                "date",
                "lat_center",
                "lon_center",
                "lat0",
                "lat1",
                "lon0",
                "lon1",
                "invalid_fraction",
                "invalid_percentage",
                "phase",
                "ostia_file_path",
                "matched_glorys_date",
                "matched_glorys_file_path",
                "matched_glorys_abs_day_delta",
            ]
        )
        # Stream rows directly to disk to keep memory stable for large (patches x timesteps).
        for ostia_path in ostia_files:
            day = _parse_ostia_date_yyyymmdd(ostia_path)
            day_path = _path_from_depth_v2(ostia_path)
            matched_glorys_date, matched_glorys_path, matched_glorys_delta = (
                date_to_glorys.get(int(day), (0, "", 0))
            )
            for rec in records:
                writer.writerow(
                    [
                        rec.patch_id,
                        day,
                        f"{rec.lat_center:.6f}",
                        f"{rec.lon_center:.6f}",
                        f"{rec.lat0:.6f}",
                        f"{rec.lat1:.6f}",
                        f"{rec.lon0:.6f}",
                        f"{rec.lon1:.6f}",
                        f"{rec.invalid_fraction:.6f}",
                        f"{rec.invalid_percentage:.4f}",
                        rec.phase,
                        day_path,
                        int(matched_glorys_date),
                        _stable_text(matched_glorys_path),
                        int(matched_glorys_delta),
                    ]
                )


def build_ostia_patch_daily_index(
    *,
    ostia_dir: Path,
    output_daily_csv: Path,
    output_spatial_csv: Path | None = None,
    glorys_dir: Path = Path("/data1/datasets/depth_v2/glorys_weekly"),
    tile_size: int = 128,
    resolution_deg: float | None = None,
    invalid_threshold: float = 0.2,
    val_fraction: float = 0.15,
    split_seed: int = 7,
    invalid_mask_flags: tuple[str, ...] = ("land",),
    include_invalid: bool = False,
) -> tuple[Path, Path | None]:
    if tile_size < 2:
        raise ValueError("tile_size must be >= 2.")
    ostia_files = _ostia_files(ostia_dir)
    glorys_rows = _scan_glorys_weekly(glorys_dir)
    reference_path = ostia_files[0]

    with xr.open_dataset(
        reference_path,
        engine="h5netcdf",
        decode_times=False,
        cache=False,
    ) as ds:
        lat = np.asarray(ds["lat"].to_numpy(), dtype=np.float64)
        lon = np.asarray(ds["lon"].to_numpy(), dtype=np.float64)

    if resolution_deg is None:
        lat_step = float(np.median(np.abs(np.diff(lat))))
        lon_step = float(np.median(np.abs(np.diff(lon))))
        if (
            not np.isfinite(lat_step)
            or not np.isfinite(lon_step)
            or lat_step <= 0.0
            or lon_step <= 0.0
        ):
            raise RuntimeError(
                "Could not infer native OSTIA grid resolution from lat/lon axes."
            )
        if abs(lat_step - lon_step) > 1e-4:
            raise RuntimeError(
                f"OSTIA native grid is not isotropic enough for one resolution value: "
                f"lat_step={lat_step}, lon_step={lon_step}"
            )
        resolution_deg = lat_step
    if resolution_deg <= 0.0:
        raise ValueError("resolution_deg must be > 0.")

    patch_span_deg = float(tile_size) * float(resolution_deg)
    patch_grid = _build_patch_grid(
        lat_min=float(np.min(lat)),
        lat_max=float(np.max(lat)),
        lon_min=float(np.min(lon)),
        lon_max=float(np.max(lon)),
        patch_span_deg=patch_span_deg,
    )

    records = _classify_patches_from_reference(
        reference_ostia_path=reference_path,
        patch_grid=patch_grid,
        tile_size=tile_size,
        resolution_deg=resolution_deg,
        invalid_threshold=invalid_threshold,
        val_fraction=val_fraction,
        split_seed=split_seed,
        invalid_mask_flags=invalid_mask_flags,
    )
    if not include_invalid:
        records = [r for r in records if r.phase != "invalid"]
        if not records:
            raise RuntimeError("No valid water patches remain after invalid filtering.")
        for new_patch_id, rec in enumerate(records):
            rec.patch_id = int(new_patch_id)

    if output_spatial_csv is not None:
        _write_patch_spatial_csv(output_spatial_csv, records)
    _write_patch_daily_csv(output_daily_csv, records, ostia_files, glorys_rows)

    return output_daily_csv, output_spatial_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build OSTIA patch index CSV on a fixed 128x128@0.1° grid, classify "
            "patches into train/val/invalid from invalid coverage, and expand to "
            "daily rows (patches x timesteps)."
        )
    )
    parser.add_argument(
        "--ostia-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/ostia"),
        help="Directory containing daily OSTIA NetCDF files.",
    )
    parser.add_argument(
        "--output-daily-csv",
        type=Path,
        default=Path("/data1/datasets/depth_v2/ostia_patch_index_daily.csv"),
        help="Output CSV with one row per (patch, day).",
    )
    parser.add_argument(
        "--output-spatial-csv",
        type=Path,
        default=Path("/data1/datasets/depth_v2/ostia_patch_index_spatial.csv"),
        help="Optional patch-only CSV (written before daily expansion).",
    )
    parser.add_argument(
        "--glorys-dir",
        type=Path,
        default=Path("/data1/datasets/depth_v2/glorys_weekly"),
        help="Directory containing weekly GLORYS NetCDF files used for nearest-date matching.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Patch width/height in pixels.",
    )
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=None,
        help="Patch pixel size in degrees. If omitted, infer native OSTIA resolution.",
    )
    parser.add_argument(
        "--invalid-threshold",
        type=float,
        default=0.2,
        help="Patch is invalid if the fraction of pixels matching invalid mask flags exceeds this threshold.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of water patches assigned to validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=7,
        help="Random seed for water train/val assignment.",
    )
    parser.add_argument(
        "--invalid-mask-flags",
        type=str,
        default="land",
        help=(
            "Comma-separated OSTIA mask flags treated as invalid coverage. "
            "Default keeps sea ice and rejects only land."
        ),
    )
    parser.add_argument(
        "--valid-mask-values",
        type=str,
        default="1",
        help=(
            "Deprecated compatibility option. Exact-value mask filtering is no longer used; "
            "patch filtering is now controlled by --invalid-mask-flags."
        ),
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="If set, keep invalid patches in output CSVs. Default excludes invalid patches.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    invalid_mask_flags = _parse_invalid_mask_flags(args.invalid_mask_flags)

    daily_csv, spatial_csv = build_ostia_patch_daily_index(
        ostia_dir=args.ostia_dir,
        output_daily_csv=args.output_daily_csv,
        output_spatial_csv=args.output_spatial_csv,
        glorys_dir=args.glorys_dir,
        tile_size=args.tile_size,
        resolution_deg=args.resolution_deg,
        invalid_threshold=args.invalid_threshold,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        invalid_mask_flags=invalid_mask_flags,
        include_invalid=bool(args.include_invalid),
    )

    print(f"Wrote daily index CSV: {daily_csv}")
    if spatial_csv is not None:
        print(f"Wrote spatial index CSV: {spatial_csv}")


if __name__ == "__main__":
    main()
