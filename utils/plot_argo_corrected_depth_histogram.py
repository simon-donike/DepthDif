from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_ARGO_GLOB = "EN.4.2.2.f.profiles.g10.*.nc"
DEFAULT_GLORYS_DIR = Path("/data1/datasets/depth_v2/glorys_weekly")
DEFAULT_GLORYS_GLOB = "*.nc"
DEFAULT_OUTPUT_PATH = Path("data/glorys_argo_alignment/argo_corrected_depth_histogram.png")


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


def sanitize_depth_array(depth: np.ndarray) -> np.ndarray:
    """Replace invalid EN4 depth entries with NaN before aggregation."""
    out = np.asarray(depth, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1.0e10] = np.nan
    out[out < 0.0] = np.nan
    return out


def find_nc_files(root_dir: Path, glob_pattern: str, *, label: str) -> list[Path]:
    """Find matching NetCDF files in a directory."""
    if not root_dir.exists():
        raise FileNotFoundError(f"{label} directory not found: {root_dir}")
    nc_files = sorted(root_dir.glob(glob_pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No {label} files found in {root_dir} with pattern {glob_pattern!r}."
        )
    return nc_files


def load_glorys_depth_levels(
    *,
    glorys_dir: Path,
    glorys_glob: str,
    depth_var_name: str,
    max_depth_m: float,
) -> np.ndarray:
    """Load the shared GLORYS depth axis up to the plotting limit."""
    glorys_file = find_nc_files(glorys_dir, glorys_glob, label="GLORYS")[0]
    with open_dataset_with_fallback(glorys_file) as ds:
        if depth_var_name not in ds.coords and depth_var_name not in ds.variables:
            raise RuntimeError(
                f"GLORYS file is missing depth coordinate/variable '{depth_var_name}': {glorys_file}"
            )
        depth_values = np.asarray(ds[depth_var_name].values, dtype=np.float64)

    depth_values = depth_values[np.isfinite(depth_values)]
    depth_values = depth_values[(depth_values >= 0.0) & (depth_values <= float(max_depth_m))]
    if depth_values.size == 0:
        raise RuntimeError(
            f"No GLORYS depths are within [0, {max_depth_m}] in {glorys_file}"
        )
    return np.sort(depth_values)


def aggregate_corrected_depth_histogram(
    *,
    argo_files: list[Path],
    depth_var_name: str,
    max_depth_m: float,
    depth_bin_size_m: float,
) -> dict[str, np.ndarray | int]:
    """Aggregate one histogram across all valid corrected-depth samples in the archive."""
    n_depth_bins = int(np.ceil(float(max_depth_m) / float(depth_bin_size_m)))
    depth_edges = np.linspace(0.0, n_depth_bins * float(depth_bin_size_m), n_depth_bins + 1)
    depth_hist = np.zeros((n_depth_bins,), dtype=np.int64)
    total_profiles = 0
    total_valid_depths = 0

    for argo_path in tqdm(argo_files, desc="Scanning EN4 files", unit="file"):
        with open_dataset_with_fallback(argo_path) as ds:
            if depth_var_name not in ds.variables:
                raise RuntimeError(
                    f"ARGO file is missing depth variable '{depth_var_name}': {argo_path}"
                )

            depth = sanitize_depth_array(np.asarray(ds[depth_var_name].values))
            if depth.ndim != 2:
                raise RuntimeError(
                    f"Expected depth array with shape (N_PROF, N_LEVELS), got {depth.shape} in {argo_path}"
                )

            total_profiles += int(depth.shape[0])
            valid_depths = depth[np.isfinite(depth)]
            if valid_depths.size == 0:
                continue

            total_valid_depths += int(valid_depths.size)
            depth_hist += np.histogram(valid_depths, bins=depth_edges)[0].astype(np.int64, copy=False)

    if total_valid_depths == 0:
        raise RuntimeError("No valid ARGO corrected depths were found in the selected files.")

    return {
        "depth_edges": depth_edges,
        "depth_hist": depth_hist,
        "total_profiles": int(total_profiles),
        "total_valid_depths": int(total_valid_depths),
        "n_files": int(len(argo_files)),
    }


def plot_corrected_depth_histogram(
    aggregates: dict[str, np.ndarray | int],
    *,
    glorys_depth_levels: np.ndarray,
    output_path: Path,
) -> Path:
    """Render one archive-wide corrected-depth histogram with GLORYS depth references."""
    depth_edges = np.asarray(aggregates["depth_edges"], dtype=np.float64)
    depth_hist = np.asarray(aggregates["depth_hist"], dtype=np.int64)
    total_profiles = int(aggregates["total_profiles"])
    total_valid_depths = int(aggregates["total_valid_depths"])
    n_files = int(aggregates["n_files"])
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.bar(
        depth_centers,
        depth_hist,
        width=np.diff(depth_edges),
        align="center",
        color="#0b4f6c",
        alpha=0.9,
        edgecolor="none",
    )

    for depth_m in glorys_depth_levels:
        # Keep GLORYS levels visually subordinate so the histogram remains the primary signal.
        ax.axvline(depth_m, color="#9ca3af", linestyle=":", linewidth=0.9, alpha=0.8)

    ax.set_xlim(depth_edges[0], depth_edges[-1])
    ax.set_xlabel("Corrected Depth (m)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Archive-Wide Histogram of EN4 Corrected Depth Values\n"
        f"{n_files} files, {total_profiles} profiles, {total_valid_depths} valid depth samples"
    )
    ax.grid(True, alpha=0.2, linestyle="--")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    """Build and save one archive-wide histogram of EN4 corrected depths."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan all EN4/ARGO NetCDF files, aggregate every valid DEPH_CORRECTED value, "
            "and plot one archive-wide depth histogram with GLORYS depth references."
        )
    )
    parser.add_argument(
        "--argo-dir",
        type=Path,
        default=DEFAULT_ARGO_DIR,
        help="Directory containing EN4 monthly profile NetCDF files.",
    )
    parser.add_argument(
        "--argo-glob",
        type=str,
        default=DEFAULT_ARGO_GLOB,
        help="Glob pattern used to select EN4 monthly profile files.",
    )
    parser.add_argument(
        "--depth-var",
        type=str,
        default="DEPH_CORRECTED",
        help="Corrected depth variable to aggregate.",
    )
    parser.add_argument(
        "--glorys-dir",
        type=Path,
        default=DEFAULT_GLORYS_DIR,
        help="Directory containing GLORYS NetCDF files used to load depth levels.",
    )
    parser.add_argument(
        "--glorys-glob",
        type=str,
        default=DEFAULT_GLORYS_GLOB,
        help="Glob pattern used to select one GLORYS NetCDF file for depth references.",
    )
    parser.add_argument(
        "--glorys-depth-var",
        type=str,
        default="depth",
        help="GLORYS depth coordinate/variable name used for horizontal reference lines.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=6000.0,
        help="Upper depth limit used for histogram binning and plotting.",
    )
    parser.add_argument(
        "--depth-bin-size-m",
        type=float,
        default=10.0,
        help="Fixed depth-bin width in meters used for the archive-wide histogram.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the histogram plot is written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of EN4 files to process.",
    )
    args = parser.parse_args()

    argo_files = find_nc_files(args.argo_dir, args.argo_glob, label="EN4")
    if args.max_files is not None and int(args.max_files) > 0:
        argo_files = argo_files[: int(args.max_files)]

    glorys_depth_levels = load_glorys_depth_levels(
        glorys_dir=args.glorys_dir,
        glorys_glob=args.glorys_glob,
        depth_var_name=args.glorys_depth_var,
        max_depth_m=float(args.max_depth_m),
    )
    aggregates = aggregate_corrected_depth_histogram(
        argo_files=argo_files,
        depth_var_name=args.depth_var,
        max_depth_m=float(args.max_depth_m),
        depth_bin_size_m=float(args.depth_bin_size_m),
    )
    output_path = plot_corrected_depth_histogram(
        aggregates,
        glorys_depth_levels=glorys_depth_levels,
        output_path=args.output_path,
    )
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
