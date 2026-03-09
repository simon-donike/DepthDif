from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_GLOB_PATTERN = "EN.4.2.2.f.profiles.g10.*.nc"
DEFAULT_OUTPUT_PATH = Path("data/depth_profiles_info.txt")


def open_dataset_with_fallback(nc_path: Path) -> tuple[xr.Dataset, str]:
    """Open dataset with backend fallback for mixed EN4 NetCDF encodings.

    Args:
        nc_path (Path): Input EN4 monthly profile file.

    Returns:
        tuple[xr.Dataset, str]: Open dataset plus backend label used.
    """
    try:
        return (
            xr.open_dataset(
                nc_path,
                engine="h5netcdf",
                decode_times=False,
                cache=False,
            ),
            "h5netcdf",
        )
    except Exception:
        # Some EN4 files are not HDF5-backed; fallback keeps inspection coverage complete.
        return xr.open_dataset(nc_path, decode_times=False, cache=False), "default"


def find_profile_files(argo_dir: Path, glob_pattern: str) -> list[Path]:
    """Find monthly EN4 profile files in a directory.

    Args:
        argo_dir (Path): Directory containing downloaded EN4 monthly files.
        glob_pattern (str): File glob pattern used to select monthly files.

    Returns:
        list[Path]: Sorted list of matching files.
    """
    if not argo_dir.exists():
        raise FileNotFoundError(f"Argo directory not found: {argo_dir}")

    nc_files = sorted(argo_dir.glob(glob_pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No EN4 monthly files found in {argo_dir} with pattern {glob_pattern!r}."
        )
    return nc_files


def parse_month_key(path: Path) -> str | None:
    """Extract YYYYMM month key from a monthly EN4 file name.

    Args:
        path (Path): Path to a monthly EN4 .nc file.

    Returns:
        str | None: Extracted month key or None when parsing fails.
    """
    match = re.search(r"g10\.(\d{6})\.nc$", path.name)
    return match.group(1) if match is not None else None


def build_expected_months(first_month: str, last_month: str) -> list[str]:
    """Build inclusive YYYYMM sequence from start month to end month.

    Args:
        first_month (str): Lower bound month key (YYYYMM).
        last_month (str): Upper bound month key (YYYYMM).

    Returns:
        list[str]: Ordered month keys spanning the full interval.
    """
    y = int(first_month[:4])
    m = int(first_month[4:6])
    end_y = int(last_month[:4])
    end_m = int(last_month[4:6])

    out: list[str] = []
    while (y < end_y) or (y == end_y and m <= end_m):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def summarize_collection(nc_files: list[Path]) -> dict[str, Any]:
    """Summarize structure and schema consistency across EN4 monthly files.

    Args:
        nc_files (list[Path]): Monthly EN4 profile files.

    Returns:
        dict[str, Any]: Aggregated collection metadata.
    """
    n_prof_values: list[int] = []
    n_levels_values: list[int] = []
    variable_counts: list[int] = []
    attribute_counts: list[int] = []
    dim_name_signatures: set[tuple[str, ...]] = set()
    variable_name_signatures: set[tuple[str, ...]] = set()
    unreadable_files: list[str] = []
    backend_use_counts: dict[str, int] = {"h5netcdf": 0, "default": 0}

    for nc_path in nc_files:
        try:
            ds, backend = open_dataset_with_fallback(nc_path)
            try:
                backend_use_counts[backend] = backend_use_counts.get(backend, 0) + 1
                sizes = dict(ds.sizes)
                n_prof_values.append(int(sizes.get("N_PROF", 0)))
                n_levels_values.append(int(sizes.get("N_LEVELS", 0)))
                variable_counts.append(len(ds.data_vars))
                attribute_counts.append(len(ds.attrs))

                # Compare schema by names only, because N_PROF naturally changes per month.
                dim_name_signatures.add(tuple(sorted(sizes.keys())))
                variable_name_signatures.add(tuple(sorted(ds.data_vars.keys())))
            finally:
                ds.close()
        except Exception:
            unreadable_files.append(str(nc_path))

    month_keys = [mk for mk in (parse_month_key(p) for p in nc_files) if mk is not None]
    month_keys_sorted = sorted(month_keys)
    missing_months: list[str] = []
    if month_keys_sorted:
        expected = set(build_expected_months(month_keys_sorted[0], month_keys_sorted[-1]))
        missing_months = sorted(expected.difference(set(month_keys_sorted)))

    return {
        "file_count": len(nc_files),
        "first_file": str(nc_files[0]),
        "last_file": str(nc_files[-1]),
        "total_size_bytes": int(sum(path.stat().st_size for path in nc_files)),
        "month_start": month_keys_sorted[0] if month_keys_sorted else "n/a",
        "month_end": month_keys_sorted[-1] if month_keys_sorted else "n/a",
        "missing_months": missing_months,
        "n_prof_min": min(n_prof_values) if n_prof_values else None,
        "n_prof_max": max(n_prof_values) if n_prof_values else None,
        "n_prof_avg": (sum(n_prof_values) / len(n_prof_values)) if n_prof_values else None,
        "n_levels_unique": sorted(set(n_levels_values)),
        "variable_count_unique": sorted(set(variable_counts)),
        "attribute_count_unique": sorted(set(attribute_counts)),
        "dim_signatures": sorted(dim_name_signatures),
        "variable_signatures_count": len(variable_name_signatures),
        "unreadable_files": unreadable_files,
        "backend_use_counts": backend_use_counts,
    }


def describe_sample_file(nc_path: Path) -> list[str]:
    """Build detailed sample-file structure text for manual inspection.

    Args:
        nc_path (Path): Path to one EN4 monthly file.

    Returns:
        list[str]: Formatted text lines.
    """
    ds, backend = open_dataset_with_fallback(nc_path)
    lines: list[str] = [
        f"\n=== SAMPLE FILE SUMMARY ===",
        f"Opening: {nc_path}",
        f"Backend used: {backend}",
    ]
    try:
        lines.append("\n=== DATASET SUMMARY ===")
        lines.append(str(ds))
        lines.extend(summarize_target_variables(ds))
        lines.extend(summarize_depth_levels(ds))

        lines.append("\n=== GLOBAL ATTRIBUTES (TAGS) ===")
        if ds.attrs:
            for key, value in ds.attrs.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("(none)")

        lines.append("\n=== DIMENSIONS ===")
        for dim, size in ds.sizes.items():
            lines.append(f"- {dim}: {size}")

        lines.append("\n=== COORDINATES ===")
        if ds.coords:
            for name, coord in ds.coords.items():
                lines.append(
                    f"- {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}"
                )
                if coord.attrs:
                    lines.append("  attrs:")
                    for key, value in coord.attrs.items():
                        lines.append(f"    - {key}: {value}")
        else:
            lines.append("(none)")

        lines.append("\n=== DATA VARIABLES ===")
        for name, var in ds.data_vars.items():
            lines.append(f"- {name}: dims={var.dims}, shape={var.shape}, dtype={var.dtype}")
            if var.attrs:
                lines.append("  attrs:")
                for key, value in var.attrs.items():
                    lines.append(f"    - {key}: {value}")
    finally:
        ds.close()
    return lines


def resolve_fill_value(var: xr.DataArray) -> float | None:
    """Resolve common NetCDF fill-value attributes for numeric filtering.

    Args:
        var (xr.DataArray): Variable to inspect.

    Returns:
        float | None: Parsed fill value, if available.
    """
    for key in ("_FillValue", "_fillvalue", "missing_value"):
        if key in var.attrs:
            raw_value = np.asarray(var.attrs[key]).reshape(-1)
            if raw_value.size > 0:
                try:
                    return float(raw_value[0])
                except (TypeError, ValueError):
                    return None
    return None


def summarize_target_variables(ds: xr.Dataset) -> list[str]:
    """Summarize temperature/salinity value ranges for one sample file.

    Args:
        ds (xr.Dataset): Open sample dataset.

    Returns:
        list[str]: Formatted lines with value statistics.
    """
    lines: list[str] = ["\n=== TARGET VARIABLE VALUE SUMMARY ==="]
    target_vars = ("TEMP", "PSAL_CORRECTED", "POTM_CORRECTED")
    for name in target_vars:
        if name not in ds.data_vars:
            lines.append(f"- {name}: not present in this file.")
            continue

        var = ds[name]
        arr = np.asarray(var.values, dtype=np.float64)
        fill_value = resolve_fill_value(var)

        valid_mask = np.isfinite(arr)
        if fill_value is not None:
            # EN4 uses sentinel fill values (e.g. 99999.0); exclude them from stats.
            valid_mask &= arr != fill_value

        valid_values = arr[valid_mask]
        total_count = int(arr.size)
        valid_count = int(valid_values.size)

        lines.append(
            f"- {name}: dims={var.dims}, shape={arr.shape}, dtype={var.dtype}, valid={valid_count}/{total_count}"
        )
        lines.append(f"  units={var.attrs.get('units', 'n/a')}, fill_value={fill_value}")
        if valid_count == 0:
            lines.append("  stats: no valid numeric values")
            continue

        lines.append(
            "  stats: "
            f"min={float(np.min(valid_values)):.4f}, "
            f"max={float(np.max(valid_values)):.4f}, "
            f"mean={float(np.mean(valid_values)):.4f}, "
            f"p01={float(np.percentile(valid_values, 1.0)):.4f}, "
            f"p50={float(np.percentile(valid_values, 50.0)):.4f}, "
            f"p99={float(np.percentile(valid_values, 99.0)):.4f}"
        )
    return lines


def summarize_depth_levels(ds: xr.Dataset) -> list[str]:
    """Summarize the depth-dimension level indices and representative depths.

    Args:
        ds (xr.Dataset): Open sample dataset.

    Returns:
        list[str]: Formatted lines for depth-level details.
    """
    lines: list[str] = ["\n=== DEPTH LEVELS IN N_LEVELS DIMENSION ==="]
    if "N_LEVELS" not in ds.sizes:
        lines.append("- N_LEVELS not present in this file.")
        return lines

    n_levels = int(ds.sizes["N_LEVELS"])
    lines.append(f"- N_LEVELS count: {n_levels}")
    lines.append(f"- Level index range: 0 .. {n_levels - 1}")
    lines.append(f"- Level indices: {', '.join(str(i) for i in range(n_levels))}")

    if "DEPH_CORRECTED" not in ds.data_vars:
        lines.append("- DEPH_CORRECTED not present; no physical-depth values to summarize.")
        return lines

    depth = ds["DEPH_CORRECTED"]
    depth_arr = np.asarray(depth.values, dtype=np.float64)
    fill_value = resolve_fill_value(depth)

    valid_mask = np.isfinite(depth_arr)
    if fill_value is not None:
        valid_mask &= depth_arr != fill_value
    # Convert invalid values to NaN so per-level median depth can be computed safely.
    depth_arr = np.where(valid_mask, depth_arr, np.nan)
    level_medians = np.nanmedian(depth_arr, axis=0)

    lines.append("- Representative median physical depth per level (m):")
    for level_idx, depth_m in enumerate(level_medians):
        if np.isnan(depth_m):
            lines.append(f"  level {level_idx}: nan")
        else:
            lines.append(f"  level {level_idx}: {float(depth_m):.3f}")
    return lines


def main() -> None:
    """Run EN4 profile structure inspection and write report to disk.

    Args:
        None: This callable takes no explicit input arguments.

    Returns:
        None: No value is returned.
    """
    parser = argparse.ArgumentParser(
        description="Inspect downloaded EN4 depth-profile NetCDF file structure."
    )
    parser.add_argument(
        "--argo-dir",
        type=str,
        default=str(DEFAULT_ARGO_DIR),
        help="Directory containing monthly EN4 profile .nc files.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help="Glob pattern used to select EN4 monthly files.",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Optional explicit sample file path. Defaults to first matched file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output txt report.",
    )
    args = parser.parse_args()

    argo_dir = Path(args.argo_dir)
    output_path = Path(args.output_path)
    nc_files = find_profile_files(argo_dir, args.glob_pattern)
    sample_path = Path(args.sample_path) if args.sample_path else nc_files[0]

    summary = summarize_collection(nc_files)
    lines: list[str] = [
        f"EN4 depth-profile directory: {argo_dir}",
        f"Matched files: {summary['file_count']}",
        f"Total size (GB): {summary['total_size_bytes'] / (1024 ** 3):.2f}",
        f"First file: {summary['first_file']}",
        f"Last file: {summary['last_file']}",
        f"Month coverage: {summary['month_start']} -> {summary['month_end']}",
        f"Missing month count in covered range: {len(summary['missing_months'])}",
    ]

    if summary["missing_months"]:
        lines.append(f"Missing months: {', '.join(summary['missing_months'])}")

    lines.extend(
        [
            "\n=== COLLECTION STRUCTURE SUMMARY ===",
            f"- N_PROF min/max: {summary['n_prof_min']} / {summary['n_prof_max']}",
            f"- N_PROF average: {summary['n_prof_avg']:.2f}"
            if summary["n_prof_avg"] is not None
            else "- N_PROF average: n/a",
            f"- Unique N_LEVELS values: {summary['n_levels_unique']}",
            f"- Unique data-variable counts: {summary['variable_count_unique']}",
            f"- Unique global-attribute counts: {summary['attribute_count_unique']}",
            f"- Unique dimension-name signatures: {len(summary['dim_signatures'])}",
            f"- Unique variable-name signatures: {summary['variable_signatures_count']}",
            f"- Unreadable files: {len(summary['unreadable_files'])}",
            f"- Backend use counts: {summary['backend_use_counts']}",
        ]
    )

    if summary["dim_signatures"]:
        lines.append("\nDimension signatures:")
        for sig in summary["dim_signatures"]:
            lines.append(f"- {sig}")

    if summary["unreadable_files"]:
        lines.append("\nUnreadable files:")
        for bad_path in summary["unreadable_files"]:
            lines.append(f"- {bad_path}")

    lines.extend(describe_sample_file(sample_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote EN4 profile structure report to {output_path}")


if __name__ == "__main__":
    main()
