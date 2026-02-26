from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _parse_month_key(source_name: str) -> str | None:
    """Extract YYYYMM from a source filename."""
    matches = re.findall(r"(\d{8}|\d{6})", str(source_name))
    if not matches:
        return None
    return matches[-1][:6]


def _scan_ostia_month_files(ostia_root: Path) -> dict[str, Path]:
    """Build mapping YYYYMM -> OSTIA monthly file path."""
    nc_files = sorted(ostia_root.rglob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No OSTIA .nc files found under: {ostia_root}")

    month_to_file: dict[str, Path] = {}
    for nc_path in nc_files:
        stamp_match = re.search(r"(\d{14})", nc_path.name)
        if stamp_match is None:
            continue
        month_key = stamp_match.group(1)[:6]
        prev = month_to_file.get(month_key)
        if prev is None or nc_path.name > prev.name:
            month_to_file[month_key] = nc_path
    return month_to_file


def _normalize_lon_scalar(lon_deg: float) -> float:
    """Normalize one longitude to [-180, 180)."""
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


def _nearest_scalar_index(sorted_values: np.ndarray, query: float) -> int:
    """Return nearest-neighbor index for a sorted 1D coordinate array."""
    if sorted_values.size == 1:
        return 0
    pos = int(np.searchsorted(sorted_values, query, side="left"))
    pos = max(1, min(pos, int(sorted_values.size) - 1))
    left = float(sorted_values[pos - 1])
    right = float(sorted_values[pos])
    return pos if abs(query - right) < abs(query - left) else pos - 1


def _lat_bounds_indices(lat_grid: np.ndarray, lat0: float, lat1: float) -> tuple[int, int]:
    """Compute inclusive nearest-neighbor lat index bounds."""
    lat_lo = min(float(lat0), float(lat1))
    lat_hi = max(float(lat0), float(lat1))
    i0 = _nearest_scalar_index(lat_grid, lat_lo)
    i1 = _nearest_scalar_index(lat_grid, lat_hi)
    return (min(i0, i1), max(i0, i1))


def _lon_bounds_segments(lon_grid: np.ndarray, lon0: float, lon1: float) -> list[tuple[int, int]]:
    """Compute inclusive lon index segment(s), handling dateline crossing."""
    a = _normalize_lon_scalar(float(lon0))
    b = _normalize_lon_scalar(float(lon1))

    lo = min(a, b)
    hi = max(a, b)
    span = hi - lo

    if span <= 180.0:
        i0 = _nearest_scalar_index(lon_grid, lo)
        i1 = _nearest_scalar_index(lon_grid, hi)
        return [(min(i0, i1), max(i0, i1))]

    i_lo = _nearest_scalar_index(lon_grid, lo)
    i_hi = _nearest_scalar_index(lon_grid, hi)
    return [(i_hi, int(lon_grid.size) - 1), (0, i_lo)]


def _extract_ostia_tile(
    sst_2d: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    lat0: float,
    lat1: float,
    lon0: float,
    lon1: float,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Extract one OSTIA tile for given geographic bounds."""
    lat_i0, lat_i1 = _lat_bounds_indices(lat_grid, lat0, lat1)
    lon_segments = _lon_bounds_segments(lon_grid, lon0, lon1)

    lat_slice = slice(lat_i0, lat_i1 + 1)
    lon_pieces = [
        sst_2d[lat_slice, seg_start : seg_end + 1]
        for seg_start, seg_end in lon_segments
    ]
    # Dateline-crossing windows become two lon slices that are stitched together.
    tile = lon_pieces[0] if len(lon_pieces) == 1 else np.concatenate(lon_pieces, axis=1)
    return tile, (lat_i0, lat_i1), lon_segments


def _format_output_tile_path(
    base_dir: Path,
    sample_idx: int,
    csv_parent: Path,
) -> str:
    """Build CSV-friendly path for one OSTIA tile file."""
    abs_path = base_dir / f"{sample_idx:08d}.npy"
    try:
        return abs_path.relative_to(csv_parent).as_posix()
    except ValueError:
        return abs_path.as_posix()


def build_overlap_csv_with_tiles(
    input_csv: Path,
    output_csv: Path,
    ostia_root: Path,
    ostia_tile_dir: Path,
    *,
    save_units: str = "celsius",
) -> Path:
    """Extract overlap OSTIA tiles and write merged depth+OSTIA index CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    month_to_file = _scan_ostia_month_files(ostia_root)
    if not month_to_file:
        raise RuntimeError(f"No parsable OSTIA monthly files found under: {ostia_root}")

    df = pd.read_csv(input_csv)
    if "source_file" not in df.columns:
        raise RuntimeError("Input CSV is missing required column: source_file")
    if "y_npy_path" not in df.columns:
        raise RuntimeError("Input CSV is missing required column: y_npy_path")

    required_geo_cols = {"lat0", "lat1", "lon0", "lon1"}
    missing_geo_cols = required_geo_cols.difference(df.columns)
    if missing_geo_cols:
        raise RuntimeError(f"Input CSV is missing required geo columns: {sorted(missing_geo_cols)}")

    month_keys = df["source_file"].astype(str).map(_parse_month_key)
    overlap_mask = month_keys.notna() & month_keys.isin(month_to_file)
    overlap_df = df.loc[overlap_mask].copy()

    if overlap_df.empty:
        raise RuntimeError("No overlapping months found between depth CSV and OSTIA files.")

    overlap_df["ostia_month_key"] = month_keys.loc[overlap_mask].to_numpy()
    overlap_df["ostia_nc_path"] = ""
    overlap_df["ostia_timestamp_utc"] = ""
    overlap_df["ostia_npy_path"] = ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ostia_tile_dir.mkdir(parents=True, exist_ok=True)

    if save_units not in {"kelvin", "celsius"}:
        raise ValueError("save_units must be one of: {'kelvin', 'celsius'}")

    for month_key, month_rows in overlap_df.groupby("ostia_month_key", sort=True):
        nc_path = month_to_file[str(month_key)]
        stamp_match = re.search(r"(\d{14})", nc_path.name)
        stamp = stamp_match.group(1) if stamp_match is not None else ""

        with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
            lat_grid = np.asarray(ds["lat"].to_numpy(), dtype=np.float64)
            lon_grid = np.asarray(ds["lon"].to_numpy(), dtype=np.float64)
            sst_2d = np.asarray(ds["analysed_sst"].isel(time=0).to_numpy(), dtype=np.float32)

        if save_units == "celsius":
            sst_2d = sst_2d - np.float32(273.15)

        for row_idx, row in month_rows.iterrows():
            sample_idx = (
                int(row["sample_idx"])
                if "sample_idx" in overlap_df.columns and pd.notna(row.get("sample_idx"))
                else int(row_idx)
            )

            tile, _, _ = _extract_ostia_tile(
                sst_2d=sst_2d,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                lat0=float(row["lat0"]),
                lat1=float(row["lat1"]),
                lon0=float(row["lon0"]),
                lon1=float(row["lon1"]),
            )

            rel_tile_path = _format_output_tile_path(
                base_dir=ostia_tile_dir,
                sample_idx=sample_idx,
                csv_parent=output_csv.parent,
            )
            np.save(ostia_tile_dir / f"{sample_idx:08d}.npy", tile.astype(np.float32, copy=False))

            overlap_df.at[row_idx, "ostia_nc_path"] = str(nc_path)
            overlap_df.at[row_idx, "ostia_timestamp_utc"] = stamp
            overlap_df.at[row_idx, "ostia_npy_path"] = rel_tile_path

    overlap_df.to_csv(output_csv, index=False)
    return output_csv


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "For overlapping months, extract OSTIA SST tiles matching each depth tile area "
            "and write a CSV with depth row info plus OSTIA tile paths."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/work/data/depth/4_bands_v2/patch_index_with_paths_split.csv"),
        help="Path to existing depth dataset CSV.",
    )
    parser.add_argument(
        "--ostia-root",
        type=Path,
        default=Path(
            "/data1/datasets/depth/ostia/"
            "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/"
            "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
        ),
        help="Root folder containing monthly OSTIA NetCDF files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/work/data/depth/4_bands_v2/patch_index_with_ostia_overlap.csv"),
        help="Path to output overlap CSV.",
    )
    parser.add_argument(
        "--ostia-tile-dir",
        type=Path,
        default=Path("/work/data/depth/4_bands_v2/ostia_npy"),
        help="Directory where extracted OSTIA tile .npy files are written.",
    )
    parser.add_argument(
        "--save-units",
        type=str,
        default="celsius",
        choices=["kelvin", "celsius"],
        help="Units used for saved OSTIA tile values.",
    )
    return parser


def main() -> None:
    """Run CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    out_path = build_overlap_csv_with_tiles(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        ostia_root=args.ostia_root,
        ostia_tile_dir=args.ostia_tile_dir,
        save_units=str(args.save_units).lower(),
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
