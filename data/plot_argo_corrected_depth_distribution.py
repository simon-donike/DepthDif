from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm


DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_ARGO_GLOB = "EN.4.2.2.f.profiles.g10.*.nc"
DEFAULT_OUTPUT_PATH = Path("data/glorys_argo_alignment/argo_corrected_depth_distribution.png")


def open_dataset_with_fallback(nc_path: Path) -> xr.Dataset:
    """Open a NetCDF file with a backend fallback for mixed EN4 encodings."""
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


def find_argo_files(argo_dir: Path, glob_pattern: str) -> list[Path]:
    """Find EN4 monthly profile files."""
    if not argo_dir.exists():
        raise FileNotFoundError(f"ARGO directory not found: {argo_dir}")
    nc_files = sorted(argo_dir.glob(glob_pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No EN4 monthly files found in {argo_dir} with pattern {glob_pattern!r}."
        )
    return nc_files


def _quantile_from_hist(
    hist_counts: np.ndarray,
    depth_edges: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Approximate a quantile curve from per-level depth histograms."""
    out = np.full((hist_counts.shape[0],), np.nan, dtype=np.float64)
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    cumulative = np.cumsum(hist_counts, axis=1)
    total = cumulative[:, -1]

    for level_idx in range(hist_counts.shape[0]):
        if total[level_idx] <= 0:
            continue
        threshold = float(total[level_idx]) * float(quantile)
        bin_idx = int(np.searchsorted(cumulative[level_idx], threshold, side="left"))
        bin_idx = min(max(bin_idx, 0), depth_centers.size - 1)
        out[level_idx] = depth_centers[bin_idx]
    return out


def aggregate_corrected_depth_distribution(
    argo_files: list[Path],
    *,
    depth_var_name: str,
    max_depth_m: float,
    n_depth_bins: int,
    live_output_path: Path | None,
) -> dict[str, np.ndarray | int]:
    """Aggregate archive-wide EN4 corrected-depth distributions without storing all samples."""
    n_levels: int | None = None
    depth_edges = np.geomspace(1.0e-1, float(max_depth_m), int(n_depth_bins) + 1)
    level_depth_hist: np.ndarray | None = None
    level_valid_count: np.ndarray | None = None
    all_depth_hist = np.zeros((int(n_depth_bins),), dtype=np.int64)
    total_profiles = 0

    processed_files = 0
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

            if n_levels is None:
                n_levels = int(depth.shape[1])
                level_depth_hist = np.zeros((n_levels, int(n_depth_bins)), dtype=np.int64)
                level_valid_count = np.zeros((n_levels,), dtype=np.int64)
            elif int(depth.shape[1]) != n_levels:
                raise RuntimeError(
                    f"Found inconsistent N_LEVELS across files: expected {n_levels}, got {depth.shape[1]} "
                    f"in {argo_path}"
                )

            total_profiles += int(depth.shape[0])
            valid_mask = np.isfinite(depth)
            if not np.any(valid_mask):
                continue

            valid_depths = depth[valid_mask]
            all_depth_hist += np.histogram(valid_depths, bins=depth_edges)[0].astype(np.int64, copy=False)

            assert level_depth_hist is not None
            assert level_valid_count is not None
            for level_idx in range(n_levels):
                level_values = depth[:, level_idx]
                level_values = level_values[np.isfinite(level_values)]
                if level_values.size == 0:
                    continue
                level_valid_count[level_idx] += int(level_values.size)
                level_depth_hist[level_idx] += np.histogram(level_values, bins=depth_edges)[0].astype(
                    np.int64,
                    copy=False,
                )

        processed_files += 1
        if live_output_path is not None:
            plot_corrected_depth_distribution(
                {
                    "depth_edges": depth_edges,
                    "level_depth_hist": level_depth_hist,
                    "level_valid_count": level_valid_count,
                    "all_depth_hist": all_depth_hist,
                    "n_levels": int(n_levels),
                    "total_profiles": int(total_profiles),
                    "n_files": int(len(argo_files)),
                    "processed_files": int(processed_files),
                },
                output_path=live_output_path,
            )

    if n_levels is None or level_depth_hist is None or level_valid_count is None:
        raise RuntimeError("No valid ARGO corrected depths were found in the selected files.")

    return {
        "depth_edges": depth_edges,
        "level_depth_hist": level_depth_hist,
        "level_valid_count": level_valid_count,
        "all_depth_hist": all_depth_hist,
        "n_levels": int(n_levels),
        "total_profiles": int(total_profiles),
        "n_files": int(len(argo_files)),
        "processed_files": int(processed_files),
    }


def plot_corrected_depth_distribution(
    aggregates: dict[str, np.ndarray | int],
    *,
    output_path: Path,
) -> Path:
    """Render one combined archive-wide figure for EN4 corrected-depth distributions."""
    depth_edges = np.asarray(aggregates["depth_edges"], dtype=np.float64)
    level_depth_hist = np.asarray(aggregates["level_depth_hist"], dtype=np.int64)
    level_valid_count = np.asarray(aggregates["level_valid_count"], dtype=np.int64)
    all_depth_hist = np.asarray(aggregates["all_depth_hist"], dtype=np.int64)
    n_levels = int(aggregates["n_levels"])
    total_profiles = int(aggregates["total_profiles"])
    n_files = int(aggregates["n_files"])
    processed_files = int(aggregates.get("processed_files", n_files))

    q10 = _quantile_from_hist(level_depth_hist, depth_edges, 0.10)
    q50 = _quantile_from_hist(level_depth_hist, depth_edges, 0.50)
    q90 = _quantile_from_hist(level_depth_hist, depth_edges, 0.90)

    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    level_edges = np.arange(n_levels + 1, dtype=np.float64) - 0.5

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), dpi=180)
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[5.5, 1.3],
        height_ratios=[1.2, 4.0],
        wspace=0.08,
        hspace=0.12,
    )

    ax_count = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_count)
    ax_hist = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_count.plot(np.arange(n_levels), level_valid_count, color="#166534", linewidth=1.7)
    ax_count.set_ylabel("Valid\nProfiles")
    ax_count.set_title(
        "EN4 Corrected Depth Distribution Across Archive\n"
        f"{processed_files}/{n_files} files processed, {total_profiles} profiles, {n_levels} EN4 level slots"
    )
    ax_count.grid(True, alpha=0.25, linestyle="--")

    # Log1p color scaling keeps sparse deep levels visible without blowing out dense shallow bins.
    density_image = np.log1p(level_depth_hist.T.astype(np.float64))
    mesh = ax_main.pcolormesh(
        level_edges,
        depth_edges,
        density_image,
        shading="auto",
        cmap="viridis",
    )
    ax_main.plot(np.arange(n_levels), q50, color="white", linewidth=2.0, label="Median depth")
    ax_main.plot(np.arange(n_levels), q10, color="#fca5a5", linewidth=1.2, linestyle="--", label="P10 / P90")
    ax_main.plot(np.arange(n_levels), q90, color="#fca5a5", linewidth=1.2, linestyle="--")
    ax_main.set_yscale("log")
    ax_main.set_xlabel("EN4 Level Index")
    ax_main.set_ylabel("Corrected Depth (m, log scale)")
    ax_main.grid(True, which="both", alpha=0.18, linestyle="--")
    ax_main.legend(loc="upper left")

    ax_hist.fill_betweenx(depth_centers, 0.0, all_depth_hist, color="#0b4f6c", alpha=0.85)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Count")
    ax_hist.grid(True, which="both", alpha=0.18, linestyle="--")
    ax_hist.set_title("All Valid Depths")

    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    """Build and save archive-wide EN4 corrected-depth distribution plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan all EN4/ARGO NetCDF files, aggregate DEPH_CORRECTED across the archive, "
            "and plot the corrected-depth distribution as a common figure."
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
        "--max-depth-m",
        type=float,
        default=6000.0,
        help="Upper depth limit used for histogram binning and plotting.",
    )
    parser.add_argument(
        "--n-depth-bins",
        type=int,
        default=240,
        help="Number of logarithmic depth bins used for the archive-wide distribution.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the combined plot is written.",
    )
    parser.add_argument(
        "--live-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite the output plot after each processed EN4 file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of EN4 files to process.",
    )
    args = parser.parse_args()

    argo_files = find_argo_files(args.argo_dir, args.argo_glob)
    if args.max_files is not None and int(args.max_files) > 0:
        argo_files = argo_files[: int(args.max_files)]
    aggregates = aggregate_corrected_depth_distribution(
        argo_files,
        depth_var_name=args.depth_var,
        max_depth_m=float(args.max_depth_m),
        n_depth_bins=int(args.n_depth_bins),
        live_output_path=(args.output_path if bool(args.live_update) else None),
    )
    output_path = plot_corrected_depth_distribution(aggregates, output_path=args.output_path)
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
