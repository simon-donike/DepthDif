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
        raise RuntimeError("Patch grid is empty. Check span/resolution and lat/lon bounds.")
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
    lat_axis = lat0 + half + (np.arange(tile_size, dtype=np.float64) * float(resolution_deg))
    lon_axis = lon0 + half + (np.arange(tile_size, dtype=np.float64) * float(resolution_deg))
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
    valid_mask_values: tuple[float, ...] = (1.0,),
) -> list[PatchRecord]:
    with xr.open_dataset(
        reference_ostia_path,
        engine="h5netcdf",
        decode_times=False,
        cache=False,
    ) as ds:
        lat = np.asarray(ds["lat"].to_numpy(), dtype=np.float32)
        lon = np.asarray(ds["lon"].to_numpy(), dtype=np.float32)
        sst = np.asarray(ds["analysed_sst"].isel(time=0).to_numpy(), dtype=np.float32)
        mask = np.asarray(ds["mask"].isel(time=0).to_numpy(), dtype=np.float32)

    sst[(~np.isfinite(sst)) | (sst > 1.0e6)] = np.nan

    sst_interp = RegularGridInterpolator(
        (lat, lon),
        sst,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
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
    valid_mask_values_np = np.asarray(valid_mask_values, dtype=np.float32)

    for lat0, _, lon0, _ in patch_boxes:
        lat_axis, lon_axis = _patch_axes(
            lat0=lat0,
            lon0=lon0,
            tile_size=tile_size,
            resolution_deg=resolution_deg,
        )
        mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
        query_points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

        sst_patch = sst_interp(query_points).reshape(tile_size, tile_size)
        mask_patch = mask_interp(query_points).reshape(tile_size, tile_size)

        invalid = (
            (~np.isfinite(sst_patch))
            | np.isclose(sst_patch, 0.0, atol=1e-8, rtol=0.0)
            | (~np.isfinite(mask_patch))
            | (~np.isin(mask_patch, valid_mask_values_np))
        )
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
                ]
            )


def _write_patch_daily_csv(
    path: Path,
    records: list[PatchRecord],
    ostia_files: list[Path],
) -> None:
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
            ]
        )
        # Stream rows directly to disk to keep memory stable for large (patches x timesteps).
        for ostia_path in ostia_files:
            day = _parse_ostia_date_yyyymmdd(ostia_path)
            day_path = str(ostia_path.resolve())
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
                    ]
                )


def build_ostia_patch_daily_index(
    *,
    ostia_dir: Path,
    output_daily_csv: Path,
    output_spatial_csv: Path | None = None,
    tile_size: int = 128,
    resolution_deg: float | None = None,
    invalid_threshold: float = 0.2,
    val_fraction: float = 0.15,
    split_seed: int = 7,
    valid_mask_values: tuple[float, ...] = (1.0,),
    include_invalid: bool = False,
) -> tuple[Path, Path | None]:
    if tile_size < 2:
        raise ValueError("tile_size must be >= 2.")
    ostia_files = _ostia_files(ostia_dir)
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
        if not np.isfinite(lat_step) or not np.isfinite(lon_step) or lat_step <= 0.0 or lon_step <= 0.0:
            raise RuntimeError("Could not infer native OSTIA grid resolution from lat/lon axes.")
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
        valid_mask_values=valid_mask_values,
    )
    if not include_invalid:
        records = [r for r in records if r.phase != "invalid"]
        if not records:
            raise RuntimeError("No valid water patches remain after invalid filtering.")
        for new_patch_id, rec in enumerate(records):
            rec.patch_id = int(new_patch_id)

    if output_spatial_csv is not None:
        _write_patch_spatial_csv(output_spatial_csv, records)
    _write_patch_daily_csv(output_daily_csv, records, ostia_files)

    return output_daily_csv, output_spatial_csv


def _parse_valid_mask_values(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("valid_mask_values cannot be empty.")
    return tuple(float(p) for p in parts)


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
        help="Patch is invalid if invalid_fraction > threshold.",
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
        "--valid-mask-values",
        type=str,
        default="1",
        help="Comma-separated OSTIA mask values treated as valid water (default: 1).",
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
    valid_mask_values = _parse_valid_mask_values(args.valid_mask_values)

    daily_csv, spatial_csv = build_ostia_patch_daily_index(
        ostia_dir=args.ostia_dir,
        output_daily_csv=args.output_daily_csv,
        output_spatial_csv=args.output_spatial_csv,
        tile_size=args.tile_size,
        resolution_deg=args.resolution_deg,
        invalid_threshold=args.invalid_threshold,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        valid_mask_values=valid_mask_values,
        include_invalid=bool(args.include_invalid),
    )

    print(f"Wrote daily index CSV: {daily_csv}")
    if spatial_csv is not None:
        print(f"Wrote spatial index CSV: {spatial_csv}")


if __name__ == "__main__":
    main()
