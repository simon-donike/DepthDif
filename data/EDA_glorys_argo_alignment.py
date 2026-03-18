from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_ARGO_GLOB = "EN.4.2.2.f.profiles.g10.*.nc"
DEFAULT_GLORYS_DIR = Path("/data1/datasets/depth_v2/glorys_weekly")
DEFAULT_GLORYS_GLOB = "*.nc"
DEFAULT_OUTPUT_DIR = Path("data/glorys_argo_alignment")


def open_dataset_with_fallback(nc_path: Path) -> xr.Dataset:
    """Open a NetCDF file with a backend fallback for mixed encodings."""
    try:
        return xr.open_dataset(
            nc_path,
            engine="h5netcdf",
            decode_times=False,
            cache=False,
        )
    except Exception:
        return xr.open_dataset(nc_path, decode_times=False, cache=False)


def find_nc_files(root_dir: Path, glob_pattern: str) -> list[Path]:
    """Find matching NetCDF files under a directory."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    nc_files = sorted(root_dir.glob(glob_pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {root_dir} with pattern {glob_pattern!r}."
        )
    return nc_files


def parse_month_key(path: Path) -> str:
    """Extract a month key from an EN4 file name when possible."""
    match = re.search(r"g10\.(\d{6})\.nc$", path.name)
    return match.group(1) if match is not None else path.stem


def sanitize_float_array(values: np.ndarray) -> np.ndarray:
    """Replace extreme or non-finite values with NaN for robust EDA."""
    out = np.asarray(values, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1.0e10] = np.nan
    return out


def extract_glorys_depths(glorys_path: Path, depth_var_name: str) -> np.ndarray:
    """Load the fixed GLORYS depth coordinate."""
    with open_dataset_with_fallback(glorys_path) as ds:
        if depth_var_name not in ds.coords and depth_var_name not in ds.variables:
            raise RuntimeError(
                f"GLORYS file is missing depth coordinate/variable '{depth_var_name}': {glorys_path}"
            )
        depth_values = sanitize_float_array(np.asarray(ds[depth_var_name].values))
    depth_values = depth_values[np.isfinite(depth_values)]
    if depth_values.size == 0:
        raise RuntimeError(f"No valid GLORYS depths found in {glorys_path}")
    return np.sort(depth_values.astype(np.float64, copy=False))


def sample_argo_depths(
    argo_files: list[Path],
    *,
    depth_var_name: str,
    max_files: int | None,
    max_profiles_per_file: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample ARGO profile depths across monthly EN4 files."""
    selected_files = argo_files if max_files is None else argo_files[: max(int(max_files), 0)]
    records: list[pd.DataFrame] = []
    file_summaries: list[dict[str, Any]] = []

    for argo_path in selected_files:
        with open_dataset_with_fallback(argo_path) as ds:
            if depth_var_name not in ds.variables:
                raise RuntimeError(
                    f"ARGO file is missing depth variable '{depth_var_name}': {argo_path}"
                )

            depth = sanitize_float_array(np.asarray(ds[depth_var_name].values))
            if depth.ndim != 2:
                raise RuntimeError(
                    f"Expected ARGO depth array with shape (N_PROF, N_LEVELS), got {depth.shape} "
                    f"in {argo_path}"
                )

            n_prof_total, n_levels = depth.shape
            if max_profiles_per_file is not None:
                n_prof_keep = min(n_prof_total, max(int(max_profiles_per_file), 0))
                depth = depth[:n_prof_keep]
            else:
                n_prof_keep = n_prof_total

            profile_idx, level_idx = np.where(np.isfinite(depth) & (depth >= 0.0))
            valid_depths = depth[profile_idx, level_idx]

            if valid_depths.size > 0:
                records.append(
                    pd.DataFrame(
                        {
                            "argo_file": str(argo_path),
                            "argo_month_key": parse_month_key(argo_path),
                            "profile_idx": profile_idx.astype(np.int64, copy=False),
                            "argo_level_index": level_idx.astype(np.int64, copy=False),
                            "argo_depth_m": valid_depths.astype(np.float64, copy=False),
                        }
                    )
                )

            file_summaries.append(
                {
                    "argo_file": str(argo_path),
                    "argo_month_key": parse_month_key(argo_path),
                    "profiles_total": int(n_prof_total),
                    "profiles_sampled": int(n_prof_keep),
                    "n_levels": int(n_levels),
                    "valid_depth_count": int(valid_depths.size),
                    "depth_min_m": float(np.nanmin(valid_depths)) if valid_depths.size else np.nan,
                    "depth_max_m": float(np.nanmax(valid_depths)) if valid_depths.size else np.nan,
                }
            )

    if not records:
        raise RuntimeError("No valid ARGO depth observations were found in the selected files.")

    total_levels_in_source_files = sorted(
        {int(summary["n_levels"]) for summary in file_summaries if int(summary["n_levels"]) > 0}
    )

    return pd.concat(records, ignore_index=True), {
        "files_scanned": int(len(selected_files)),
        "file_summaries": file_summaries,
        "argo_n_levels_in_source_files": total_levels_in_source_files,
    }


def build_alignment_frame(
    argo_depth_df: pd.DataFrame,
    glorys_depths: np.ndarray,
    *,
    exact_tolerance_m: float,
) -> pd.DataFrame:
    """Map every ARGO depth observation to a GLORYS depth bracket."""
    argo_depths = argo_depth_df["argo_depth_m"].to_numpy(dtype=np.float64, copy=True)
    insert_right = np.searchsorted(glorys_depths, argo_depths, side="left")
    lower_idx = insert_right - 1
    upper_idx = insert_right

    in_range = (lower_idx >= 0) & (upper_idx < glorys_depths.size)
    exact_idx = np.full(argo_depths.shape, -1, dtype=np.int64)

    lower_depth = np.full(argo_depths.shape, np.nan, dtype=np.float64)
    upper_depth = np.full(argo_depths.shape, np.nan, dtype=np.float64)
    lower_depth[lower_idx >= 0] = glorys_depths[lower_idx[lower_idx >= 0]]
    upper_depth[upper_idx < glorys_depths.size] = glorys_depths[upper_idx[upper_idx < glorys_depths.size]]

    lower_diff = np.abs(argo_depths - lower_depth)
    upper_diff = np.abs(argo_depths - upper_depth)
    lower_exact = (lower_idx >= 0) & np.isfinite(lower_diff) & (lower_diff <= exact_tolerance_m)
    upper_exact = (upper_idx < glorys_depths.size) & np.isfinite(upper_diff) & (
        upper_diff <= exact_tolerance_m
    )

    exact_idx[lower_exact] = lower_idx[lower_exact]
    exact_idx[(exact_idx < 0) & upper_exact] = upper_idx[(exact_idx < 0) & upper_exact]
    exact_match = exact_idx >= 0

    interp_weight_upper = np.full(argo_depths.shape, np.nan, dtype=np.float64)
    interp_weight_lower = np.full(argo_depths.shape, np.nan, dtype=np.float64)
    span = upper_depth - lower_depth
    interp_mask = in_range & (~exact_match) & np.isfinite(span) & (span > 0.0)
    interp_weight_upper[interp_mask] = (
        (argo_depths[interp_mask] - lower_depth[interp_mask]) / span[interp_mask]
    )
    interp_weight_lower[interp_mask] = 1.0 - interp_weight_upper[interp_mask]

    status = np.full(argo_depths.shape, "between_levels", dtype=object)
    status[exact_match] = "exact_glorys_level"
    status[argo_depths < glorys_depths[0]] = "shallower_than_glorys_min"
    status[argo_depths > glorys_depths[-1]] = "deeper_than_glorys_max"

    out = argo_depth_df.copy()
    out["glorys_lower_idx"] = np.where(lower_idx >= 0, lower_idx, -1).astype(np.int64, copy=False)
    out["glorys_upper_idx"] = np.where(upper_idx < glorys_depths.size, upper_idx, -1).astype(
        np.int64,
        copy=False,
    )
    out["glorys_lower_depth_m"] = lower_depth
    out["glorys_upper_depth_m"] = upper_depth
    out["glorys_exact_idx"] = exact_idx
    out["alignment_status"] = status
    out["interp_weight_lower"] = interp_weight_lower
    out["interp_weight_upper"] = interp_weight_upper
    out["distance_to_lower_m"] = argo_depths - lower_depth
    out["distance_to_upper_m"] = upper_depth - argo_depths
    return out


def summarize_per_argo_level(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize depth alignment per ARGO vertical level index."""
    grouped = alignment_df.groupby("argo_level_index", sort=True)
    records: list[dict[str, Any]] = []
    for level_idx, group in grouped:
        depths = group["argo_depth_m"].to_numpy(dtype=np.float64, copy=False)
        between_mask = group["alignment_status"].eq("between_levels").to_numpy()
        lower_mode = (
            int(group.loc[between_mask, "glorys_lower_idx"].mode().iloc[0])
            if np.any(between_mask)
            else -1
        )
        upper_mode = (
            int(group.loc[between_mask, "glorys_upper_idx"].mode().iloc[0])
            if np.any(between_mask)
            else -1
        )
        records.append(
            {
                "argo_level_index": int(level_idx),
                "count": int(len(group)),
                "depth_min_m": float(np.nanmin(depths)),
                "depth_p10_m": float(np.nanpercentile(depths, 10)),
                "depth_p50_m": float(np.nanpercentile(depths, 50)),
                "depth_p90_m": float(np.nanpercentile(depths, 90)),
                "depth_max_m": float(np.nanmax(depths)),
                "exact_match_fraction": float(group["alignment_status"].eq("exact_glorys_level").mean()),
                "between_fraction": float(group["alignment_status"].eq("between_levels").mean()),
                "shallower_fraction": float(
                    group["alignment_status"].eq("shallower_than_glorys_min").mean()
                ),
                "deeper_fraction": float(
                    group["alignment_status"].eq("deeper_than_glorys_max").mean()
                ),
                "most_common_glorys_lower_idx": lower_mode,
                "most_common_glorys_upper_idx": upper_mode,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_glorys_coverage(alignment_df: pd.DataFrame, glorys_depths: np.ndarray) -> pd.DataFrame:
    """Count how strongly each GLORYS layer participates in ARGO interpolation."""
    records: list[dict[str, Any]] = []
    for glorys_idx, glorys_depth in enumerate(glorys_depths.tolist()):
        exact_count = int((alignment_df["glorys_exact_idx"] == glorys_idx).sum())
        lower_count = int((alignment_df["glorys_lower_idx"] == glorys_idx).sum())
        upper_count = int((alignment_df["glorys_upper_idx"] == glorys_idx).sum())
        records.append(
            {
                "glorys_level_index": int(glorys_idx),
                "glorys_depth_m": float(glorys_depth),
                "exact_match_count": exact_count,
                "used_as_lower_count": lower_count,
                "used_as_upper_count": upper_count,
                # This gives a quick sense of which GLORYS levels are most relevant for ARGO alignment.
                "total_alignment_touch_count": int(exact_count + lower_count + upper_count),
            }
        )
    return pd.DataFrame.from_records(records)


def build_representative_argo_level_summary(
    argo_path: Path,
    *,
    depth_var_name: str,
) -> pd.DataFrame:
    """Summarize all ARGO level indices across every profile in one representative file."""
    with open_dataset_with_fallback(argo_path) as ds:
        if depth_var_name not in ds.variables:
            raise RuntimeError(
                f"ARGO file is missing depth variable '{depth_var_name}': {argo_path}"
            )

        depth = sanitize_float_array(np.asarray(ds[depth_var_name].values))
        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected ARGO depth array with shape (N_PROF, N_LEVELS), got {depth.shape} "
                f"in {argo_path}"
            )

    valid_mask = np.isfinite(depth) & (depth >= 0.0)
    depth[~valid_mask] = np.nan
    valid_counts = np.sum(valid_mask, axis=0, dtype=np.int64)
    valid_level_idx = np.flatnonzero(valid_counts > 0).astype(np.int64)

    records: list[dict[str, Any]] = []
    for level_idx in valid_level_idx.tolist():
        level_values = depth[:, level_idx]
        records.append(
            {
                "argo_level_index": int(level_idx),
                "valid_profile_count": int(valid_counts[level_idx]),
                "argo_depth_min_m": float(np.nanmin(level_values)),
                "argo_depth_p10_m": float(np.nanpercentile(level_values, 10)),
                "argo_depth_m": float(np.nanmedian(level_values)),
                "argo_depth_p90_m": float(np.nanpercentile(level_values, 90)),
                "argo_depth_max_m": float(np.nanmax(level_values)),
            }
        )

    return pd.DataFrame.from_records(records)


def build_one_to_one_channel_mapping(
    representative_argo_level_df: pd.DataFrame,
    glorys_depths: np.ndarray,
    *,
    argo_n_levels_total: int | None,
    argo_reference_file: Path | None,
) -> dict[str, Any]:
    """Assign each ARGO level to the nearest GLORYS depth level."""
    mapping_records: list[dict[str, Any]] = []
    for row in representative_argo_level_df.itertuples(index=False):
        representative_depth = float(row.argo_depth_m)
        nearest_glorys_idx = int(np.argmin(np.abs(glorys_depths - representative_depth)))
        nearest_glorys_depth = float(glorys_depths[nearest_glorys_idx])
        mapping_records.append(
            {
                "argo_level_index": int(row.argo_level_index),
                "argo_depth_m": representative_depth,
                "argo_valid_profile_count": int(row.valid_profile_count),
                "glorys_level_index": nearest_glorys_idx,
                "glorys_depth_m": nearest_glorys_depth,
                "absolute_depth_difference_m": float(abs(representative_depth - nearest_glorys_depth)),
            }
        )
    return {
        "argo_reference_file": str(argo_reference_file) if argo_reference_file is not None else None,
        "argo_n_levels_total_in_source_file": (
            int(argo_n_levels_total) if argo_n_levels_total is not None else None
        ),
        "argo_n_levels_with_any_valid_depth_in_reference_file": int(len(mapping_records)),
        "glorys_n_levels": int(glorys_depths.size),
        "mapping_method": "nearest_glorys_depth_per_argo_level_median_depth_across_all_profiles_in_reference_file",
        "mapping": mapping_records,
    }


def interpolate_glorys_profile_to_argo_depths(
    glorys_temperature: np.ndarray,
    glorys_depths: np.ndarray,
    argo_depths: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a 1D GLORYS temperature profile onto ARGO depths."""
    glorys_temperature = np.asarray(glorys_temperature, dtype=np.float64)
    glorys_depths = np.asarray(glorys_depths, dtype=np.float64)
    argo_depths = np.asarray(argo_depths, dtype=np.float64)

    if glorys_temperature.ndim != 1 or glorys_depths.ndim != 1:
        raise ValueError("glorys_temperature and glorys_depths must both be 1D arrays.")
    if glorys_temperature.shape[0] != glorys_depths.shape[0]:
        raise ValueError("glorys_temperature and glorys_depths must have the same length.")

    valid = np.isfinite(glorys_temperature) & np.isfinite(glorys_depths)
    if valid.sum() < 2:
        return np.full(argo_depths.shape, np.nan, dtype=np.float64)

    x = glorys_depths[valid]
    y = glorys_temperature[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.full(argo_depths.shape, np.nan, dtype=np.float64)
    in_range = np.isfinite(argo_depths) & (argo_depths >= x[0]) & (argo_depths <= x[-1])
    out[in_range] = np.interp(argo_depths[in_range], x, y)
    return out


def build_report_lines(
    *,
    glorys_file: Path,
    glorys_depths: np.ndarray,
    argo_summary: dict[str, Any],
    alignment_df: pd.DataFrame,
    per_level_df: pd.DataFrame,
    representative_argo_level_df: pd.DataFrame,
    representative_argo_file: Path | None,
) -> list[str]:
    """Build a human-readable alignment report."""
    status_counts = alignment_df["alignment_status"].value_counts().to_dict()
    argo_depths = alignment_df["argo_depth_m"].to_numpy(dtype=np.float64, copy=False)

    lines: list[str] = [
        f"GLORYS file inspected: {glorys_file}",
        f"GLORYS depth level count: {glorys_depths.size}",
        f"GLORYS depth range: {glorys_depths[0]:.3f} m .. {glorys_depths[-1]:.3f} m",
        f"ARGO files scanned: {argo_summary['files_scanned']}",
        f"ARGO source-file N_LEVELS values: {argo_summary.get('argo_n_levels_in_source_files', [])}",
        f"ARGO valid depth observations in sampled profile EDA: {len(alignment_df)}",
        f"ARGO sampled-profile depth range: {np.nanmin(argo_depths):.3f} m .. {np.nanmax(argo_depths):.3f} m",
        f"ARGO representative file for 400-level mapping: {representative_argo_file}",
        "ARGO level-count note: EN4 stores 400 possible level slots per profile, but each individual "
        "profile only fills the slots it actually observed.",
        f"ARGO level indices with any valid depth in representative file: {len(representative_argo_level_df)}",
        "",
        "=== GLORYS DEPTH LEVELS ===",
    ]
    for idx, depth_m in enumerate(glorys_depths.tolist()):
        lines.append(f"- level {idx}: {depth_m:.3f} m")

    lines.extend(
        [
            "",
            "=== ALIGNMENT STATUS COUNTS ===",
            f"- exact_glorys_level: {int(status_counts.get('exact_glorys_level', 0))}",
            f"- between_levels: {int(status_counts.get('between_levels', 0))}",
            f"- shallower_than_glorys_min: {int(status_counts.get('shallower_than_glorys_min', 0))}",
            f"- deeper_than_glorys_max: {int(status_counts.get('deeper_than_glorys_max', 0))}",
            "",
            "=== ALIGNMENT STATUS FRACTIONS ===",
            f"- exact_glorys_level: {alignment_df['alignment_status'].eq('exact_glorys_level').mean():.4f}",
            f"- between_levels: {alignment_df['alignment_status'].eq('between_levels').mean():.4f}",
            f"- shallower_than_glorys_min: {alignment_df['alignment_status'].eq('shallower_than_glorys_min').mean():.4f}",
            f"- deeper_than_glorys_max: {alignment_df['alignment_status'].eq('deeper_than_glorys_max').mean():.4f}",
            "",
            "=== REPRESENTATIVE ARGO LEVEL MAPPING (MEDIAN DEPTH PER EN4 LEVEL) ===",
        ]
    )

    for row in per_level_df.itertuples(index=False):
        lines.append(
            "- "
            f"ARGO level {int(row.argo_level_index)}: "
            f"p10/p50/p90={row.depth_p10_m:.3f}/{row.depth_p50_m:.3f}/{row.depth_p90_m:.3f} m, "
            f"common GLORYS bracket=({int(row.most_common_glorys_lower_idx)}, {int(row.most_common_glorys_upper_idx)}), "
            f"exact={row.exact_match_fraction:.3f}, between={row.between_fraction:.3f}"
        )

    return lines


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run EDA for ARGO-vs-GLORYS depth alignment and summarize how ARGO profile depths "
            "map onto the fixed GLORYS vertical grid."
        )
    )
    parser.add_argument(
        "--argo-dir",
        type=Path,
        default=DEFAULT_ARGO_DIR,
        help="Directory containing EN4/ARGO monthly profile files.",
    )
    parser.add_argument(
        "--argo-glob",
        type=str,
        default=DEFAULT_ARGO_GLOB,
        help="Glob used to select EN4/ARGO monthly files.",
    )
    parser.add_argument(
        "--glorys-dir",
        type=Path,
        default=DEFAULT_GLORYS_DIR,
        help="Directory containing weekly GLORYS NetCDF files.",
    )
    parser.add_argument(
        "--glorys-glob",
        type=str,
        default=DEFAULT_GLORYS_GLOB,
        help="Glob used to select GLORYS NetCDF files.",
    )
    parser.add_argument(
        "--glorys-file",
        type=Path,
        default=None,
        help="Optional explicit GLORYS file to inspect. If omitted, the first match in --glorys-dir is used.",
    )
    parser.add_argument(
        "--argo-depth-var",
        type=str,
        default="DEPH_CORRECTED",
        help="ARGO depth variable name.",
    )
    parser.add_argument(
        "--glorys-depth-var",
        type=str,
        default="depth",
        help="GLORYS depth coordinate name.",
    )
    parser.add_argument(
        "--max-argo-files",
        type=int,
        default=1,
        help="Maximum number of ARGO monthly files to scan. Use <= 0 for all files.",
    )
    parser.add_argument(
        "--max-profiles-per-file",
        type=int,
        default=1,
        help="Maximum ARGO profiles to sample per file. Use <= 0 for all profiles.",
    )
    parser.add_argument(
        "--exact-tolerance-m",
        type=float,
        default=0.25,
        help="Depth tolerance in meters for treating an ARGO depth as an exact GLORYS level match.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where EDA outputs are written.",
    )
    return parser


def main() -> None:
    """Run the ARGO/GLORYS alignment EDA."""
    args = build_parser().parse_args()

    argo_files = find_nc_files(args.argo_dir, args.argo_glob)
    glorys_file = (
        args.glorys_file
        if args.glorys_file is not None
        else find_nc_files(args.glorys_dir, args.glorys_glob)[0]
    )
    max_argo_files = None if int(args.max_argo_files) <= 0 else int(args.max_argo_files)
    max_profiles_per_file = (
        None if int(args.max_profiles_per_file) <= 0 else int(args.max_profiles_per_file)
    )

    glorys_depths = extract_glorys_depths(glorys_file, args.glorys_depth_var)
    argo_depth_df, argo_summary = sample_argo_depths(
        argo_files,
        depth_var_name=args.argo_depth_var,
        max_files=max_argo_files,
        max_profiles_per_file=max_profiles_per_file,
    )
    alignment_df = build_alignment_frame(
        argo_depth_df,
        glorys_depths,
        exact_tolerance_m=float(args.exact_tolerance_m),
    )
    per_level_df = summarize_per_argo_level(alignment_df)
    glorys_coverage_df = summarize_glorys_coverage(alignment_df, glorys_depths)
    representative_argo_file = argo_files[0] if len(argo_files) > 0 else None
    representative_argo_level_df = build_representative_argo_level_summary(
        representative_argo_file,
        depth_var_name=args.argo_depth_var,
    )
    argo_n_levels_values = argo_summary.get("argo_n_levels_in_source_files", [])
    channel_mapping = build_one_to_one_channel_mapping(
        representative_argo_level_df,
        glorys_depths,
        argo_n_levels_total=(int(argo_n_levels_values[0]) if len(argo_n_levels_values) == 1 else None),
        argo_reference_file=representative_argo_file,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment_csv_path = args.output_dir / "argo_glorys_depth_alignment.csv"
    per_level_csv_path = args.output_dir / "argo_depth_level_summary.csv"
    glorys_coverage_csv_path = args.output_dir / "glorys_depth_coverage_summary.csv"
    report_path = args.output_dir / "glorys_argo_alignment_report.txt"
    channel_mapping_json_path = args.output_dir / "argo_to_glorys_channel_mapping.json"

    alignment_df.to_csv(alignment_csv_path, index=False)
    per_level_df.to_csv(per_level_csv_path, index=False)
    glorys_coverage_df.to_csv(glorys_coverage_csv_path, index=False)
    channel_mapping_json_path.write_text(json.dumps(channel_mapping, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        "\n".join(
            build_report_lines(
                glorys_file=glorys_file,
                glorys_depths=glorys_depths,
                argo_summary=argo_summary,
                alignment_df=alignment_df,
                per_level_df=per_level_df,
                representative_argo_level_df=representative_argo_level_df,
                representative_argo_file=representative_argo_file,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote report: {report_path}")
    print(f"Wrote per-observation alignment CSV: {alignment_csv_path}")
    print(f"Wrote ARGO-level summary CSV: {per_level_csv_path}")
    print(f"Wrote GLORYS-level coverage CSV: {glorys_coverage_csv_path}")
    print(f"Wrote 1-to-1 channel mapping JSON: {channel_mapping_json_path}")


if __name__ == "__main__":
    main()
