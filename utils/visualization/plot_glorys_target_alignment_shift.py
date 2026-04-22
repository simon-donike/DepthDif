from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

DEFAULT_ARGO_DIR = Path("/data1/datasets/depth_v2/en4_profiles")
DEFAULT_ARGO_GLOB = "EN.4.2.2.f.profiles.g10.*.nc"
DEFAULT_GLORYS_DIR = Path("/data1/datasets/depth_v2/glorys_weekly")
DEFAULT_GLORYS_GLOB = "*.nc"
DEFAULT_OUTPUT_DIR = Path("data/glorys_argo_alignment/figures")


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
    """Replace invalid EN4 depth entries with NaN before analysis."""
    out = np.asarray(depth, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1.0e10] = np.nan
    out[out < 0.0] = np.nan
    return out


def find_nc_files(root_dir: Path, glob_pattern: str) -> list[Path]:
    """Find matching NetCDF files in a directory."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    nc_files = sorted(root_dir.glob(glob_pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {root_dir} with pattern {glob_pattern!r}."
        )
    return nc_files


def load_retained_glorys_targets(
    *,
    glorys_dir: Path,
    glorys_glob: str,
    depth_var_name: str,
    max_glorys_depth_m: float,
) -> pd.DataFrame:
    """Load the retained GLORYS target depths up to the requested depth cutoff."""
    glorys_file = find_nc_files(glorys_dir, glorys_glob)[0]
    with open_dataset_with_fallback(glorys_file) as ds:
        if depth_var_name not in ds.coords and depth_var_name not in ds.variables:
            raise RuntimeError(
                f"GLORYS file is missing depth coordinate/variable '{depth_var_name}': {glorys_file}"
            )
        depth_values = np.asarray(ds[depth_var_name].values, dtype=np.float64)

    depth_values = depth_values[np.isfinite(depth_values)]
    retained_mask = depth_values <= float(max_glorys_depth_m)
    retained_depths = depth_values[retained_mask]
    retained_idx = np.flatnonzero(retained_mask).astype(np.int64)
    if retained_depths.size == 0:
        raise RuntimeError(
            f"No GLORYS depths are <= max_glorys_depth_m={max_glorys_depth_m} in {glorys_file}"
        )

    return pd.DataFrame(
        {
            "glorys_level_index": retained_idx,
            "glorys_depth_m": retained_depths,
        }
    )


def _nearest_depth_differences_for_profile(
    profile_depths: np.ndarray,
    glorys_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest ARGO depth and signed GLORYS-minus-ARGO shift per GLORYS target."""
    if profile_depths.size == 0:
        nan_vec = np.full(glorys_depths.shape, np.nan, dtype=np.float64)
        return nan_vec, nan_vec

    sorted_depths = np.sort(profile_depths.astype(np.float64, copy=False))
    insert_idx = np.searchsorted(sorted_depths, glorys_depths, side="left")

    left_idx = np.clip(insert_idx - 1, 0, max(sorted_depths.size - 1, 0))
    right_idx = np.clip(insert_idx, 0, max(sorted_depths.size - 1, 0))

    left_depth = sorted_depths[left_idx]
    right_depth = sorted_depths[right_idx]

    left_diff = np.abs(glorys_depths - left_depth)
    right_diff = np.abs(glorys_depths - right_depth)
    use_right = right_diff < left_diff
    nearest_depth = np.where(use_right, right_depth, left_depth)
    signed_shift = glorys_depths - nearest_depth
    return nearest_depth, signed_shift


def summarize_nearest_profile_alignment(
    *,
    argo_files: list[Path],
    argo_depth_var_name: str,
    glorys_targets_df: pd.DataFrame,
    max_valid_distance_m: float | None,
) -> pd.DataFrame:
    """Aggregate nearest-profile-depth offsets for each retained GLORYS target over all EN4 profiles."""
    glorys_depths = glorys_targets_df["glorys_depth_m"].to_numpy(
        dtype=np.float64, copy=False
    )
    n_targets = glorys_depths.size

    nearest_depth_sum = np.zeros((n_targets,), dtype=np.float64)
    nearest_depth_sq_sum = np.zeros((n_targets,), dtype=np.float64)
    signed_shift_sum = np.zeros((n_targets,), dtype=np.float64)
    signed_shift_sq_sum = np.zeros((n_targets,), dtype=np.float64)
    abs_shift_sum = np.zeros((n_targets,), dtype=np.float64)
    abs_shift_sq_sum = np.zeros((n_targets,), dtype=np.float64)
    exact_match_count = np.zeros((n_targets,), dtype=np.int64)
    evaluated_profile_count = np.zeros((n_targets,), dtype=np.int64)
    valid_match_count = np.zeros((n_targets,), dtype=np.int64)

    total_profiles = 0

    for argo_path in tqdm(argo_files, desc="Scanning EN4 profiles", unit="file"):
        with open_dataset_with_fallback(argo_path) as ds:
            if argo_depth_var_name not in ds.variables:
                raise RuntimeError(
                    f"ARGO file is missing depth variable '{argo_depth_var_name}': {argo_path}"
                )

            depth = sanitize_depth_array(np.asarray(ds[argo_depth_var_name].values))
            if depth.ndim != 2:
                raise RuntimeError(
                    f"Expected depth array with shape (N_PROF, N_LEVELS), got {depth.shape} in {argo_path}"
                )

            for profile_idx in range(depth.shape[0]):
                profile_depths = depth[profile_idx]
                profile_depths = profile_depths[np.isfinite(profile_depths)]
                if profile_depths.size == 0:
                    continue

                nearest_depth, signed_shift = _nearest_depth_differences_for_profile(
                    profile_depths,
                    glorys_depths,
                )
                abs_shift = np.abs(signed_shift)
                if max_valid_distance_m is None:
                    valid_match_mask = np.isfinite(abs_shift)
                else:
                    # Rejected nearest-depth matches do not contribute to any summary statistic.
                    valid_match_mask = np.isfinite(abs_shift) & (
                        abs_shift <= float(max_valid_distance_m)
                    )

                total_profiles += 1
                evaluated_profile_count += 1
                valid_match_count += valid_match_mask.astype(np.int64)
                nearest_depth_sum += np.where(valid_match_mask, nearest_depth, 0.0)
                nearest_depth_sq_sum += np.where(
                    valid_match_mask, np.square(nearest_depth), 0.0
                )
                signed_shift_sum += np.where(valid_match_mask, signed_shift, 0.0)
                signed_shift_sq_sum += np.where(
                    valid_match_mask, np.square(signed_shift), 0.0
                )
                abs_shift_sum += np.where(valid_match_mask, abs_shift, 0.0)
                abs_shift_sq_sum += np.where(
                    valid_match_mask, np.square(abs_shift), 0.0
                )
                exact_match_count += (abs_shift <= 1.0e-6).astype(np.int64)

    if total_profiles == 0:
        raise RuntimeError(
            "No valid ARGO profiles were found for nearest-depth alignment analysis."
        )

    valid_denom = np.maximum(valid_match_count.astype(np.float64), 1.0)
    evaluated_denom = np.maximum(evaluated_profile_count.astype(np.float64), 1.0)
    mean_nearest = nearest_depth_sum / valid_denom
    mean_signed = signed_shift_sum / valid_denom
    mean_abs = abs_shift_sum / valid_denom

    # Use E[x^2] - E[x]^2 so the summary can be streamed across all profiles without
    # storing per-profile arrays in memory.
    var_nearest = np.maximum(
        (nearest_depth_sq_sum / valid_denom) - np.square(mean_nearest), 0.0
    )
    var_signed = np.maximum(
        (signed_shift_sq_sum / valid_denom) - np.square(mean_signed), 0.0
    )
    var_abs = np.maximum((abs_shift_sq_sum / valid_denom) - np.square(mean_abs), 0.0)

    no_valid_match_mask = valid_match_count == 0
    mean_nearest[no_valid_match_mask] = np.nan
    mean_signed[no_valid_match_mask] = np.nan
    mean_abs[no_valid_match_mask] = np.nan
    var_nearest[no_valid_match_mask] = np.nan
    var_signed[no_valid_match_mask] = np.nan
    var_abs[no_valid_match_mask] = np.nan

    out = glorys_targets_df.copy()
    out["evaluated_profile_count"] = evaluated_profile_count
    out["contributing_profile_count"] = valid_match_count
    out["mean_nearest_argo_depth_m"] = mean_nearest
    out["std_nearest_argo_depth_m"] = np.sqrt(var_nearest)
    out["mean_signed_shift_m"] = mean_signed
    out["std_signed_shift_m"] = np.sqrt(var_signed)
    out["mean_absolute_shift_m"] = mean_abs
    out["std_absolute_shift_m"] = np.sqrt(var_abs)
    out["exact_match_fraction"] = exact_match_count / evaluated_denom
    out["within_cutoff_fraction"] = (
        valid_match_count / evaluated_denom
        if max_valid_distance_m is not None
        else np.nan
    )
    return out


def _save_depth_summary_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    max_valid_distance_m: float | None,
) -> Path:
    """Plot depth alignment and average nearest-distance mismatch on one shared GLORYS-level axis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=180, sharex=True)
    nearest_depth_lower = np.maximum(
        summary_df["mean_nearest_argo_depth_m"]
        - summary_df["std_nearest_argo_depth_m"],
        1.0e-3,
    )

    axes[0].plot(
        summary_df["glorys_level_index"],
        summary_df["glorys_depth_m"],
        color="#d97706",
        linewidth=2.0,
        label="GLORYS target depth",
    )
    axes[0].plot(
        summary_df["glorys_level_index"],
        summary_df["mean_nearest_argo_depth_m"],
        color="#0b4f6c",
        linewidth=2.0,
        label="Mean nearest ARGO depth",
    )
    axes[0].fill_between(
        summary_df["glorys_level_index"],
        nearest_depth_lower,
        summary_df["mean_nearest_argo_depth_m"]
        + summary_df["std_nearest_argo_depth_m"],
        color="#0b4f6c",
        alpha=0.18,
        label="Nearest ARGO depth mean ± 1 std",
    )
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_yscale("log")
    if max_valid_distance_m is None:
        axes[0].set_title("GLORYS Depth And Mean Nearest ARGO Depth Across Profiles")
    else:
        axes[0].set_title(
            "GLORYS Depth And Mean Nearest ARGO Depth Across Profiles\n"
            f"Only nearest matches within {float(max_valid_distance_m):.2f} m are included"
        )
    axes[0].grid(True, alpha=0.25, linestyle="--")
    axes[0].legend(loc="best")

    axes[1].plot(
        summary_df["glorys_level_index"],
        summary_df["mean_absolute_shift_m"],
        color="#b91c1c",
        linewidth=1.8,
        label="Mean absolute nearest-depth distance",
    )
    axes[1].fill_between(
        summary_df["glorys_level_index"],
        np.maximum(
            summary_df["mean_absolute_shift_m"] - summary_df["std_absolute_shift_m"],
            0.0,
        ),
        summary_df["mean_absolute_shift_m"] + summary_df["std_absolute_shift_m"],
        color="#ef4444",
        alpha=0.18,
        label="Absolute distance mean ± 1 std",
    )
    if max_valid_distance_m is not None:
        axes[1].axhline(
            float(max_valid_distance_m),
            color="#1d4ed8",
            linewidth=1.2,
            linestyle="--",
            label=f"Validity cutoff = {float(max_valid_distance_m):.2f} m",
        )
    axes[1].set_xlabel("Retained GLORYS Level Index")
    axes[1].set_ylabel("Distance To Nearest ARGO Depth (m)")
    if max_valid_distance_m is None:
        axes[1].set_title(
            "Average Distance Between Each GLORYS Target And The Closest ARGO Observation"
        )
    else:
        axes[1].set_title(
            "Average Distance Between Each GLORYS Target And The Closest Accepted ARGO Observation\n"
            f"Matches farther than {float(max_valid_distance_m):.2f} m are excluded"
        )
    axes[1].grid(True, alpha=0.25, linestyle="--")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def _save_within_cutoff_fraction_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    max_valid_distance_m: float,
) -> Path:
    """Plot what fraction of profiles have a nearest ARGO depth within the cutoff."""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=180)
    ax.bar(
        summary_df["glorys_level_index"].astype(int),
        summary_df["within_cutoff_fraction"].astype(float),
        color="#166534",
        alpha=0.85,
        width=0.8,
    )
    ax.set_xlabel("Retained GLORYS Level Index")
    ax.set_ylabel("Fraction Within Cutoff")
    ax.set_title(
        "Fraction Of Profiles Whose Nearest ARGO Observation Is Close Enough To Be Treated As Valid\n"
        f"Cutoff = {float(max_valid_distance_m):.2f} m"
    )
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    """Visualize nearest-profile-depth offsets between EN4 / ARGO and retained GLORYS target depths."""
    parser = argparse.ArgumentParser(
        description=(
            "Iterate over all EN4 / ARGO profiles and, for each retained GLORYS target depth, "
            "measure how far away the nearest observed ARGO depth is on average."
        )
    )
    parser.add_argument(
        "--argo-dir",
        type=Path,
        default=DEFAULT_ARGO_DIR,
        help="Directory containing EN4 monthly profile files.",
    )
    parser.add_argument(
        "--argo-glob",
        type=str,
        default=DEFAULT_ARGO_GLOB,
        help="Glob used to select EN4 monthly files.",
    )
    parser.add_argument(
        "--argo-depth-var",
        type=str,
        default="DEPH_CORRECTED",
        help="EN4 / ARGO corrected depth variable name.",
    )
    parser.add_argument(
        "--glorys-dir",
        type=Path,
        default=DEFAULT_GLORYS_DIR,
        help="Directory containing GLORYS weekly files.",
    )
    parser.add_argument(
        "--glorys-glob",
        type=str,
        default=DEFAULT_GLORYS_GLOB,
        help="Glob used to select GLORYS files.",
    )
    parser.add_argument(
        "--glorys-depth-var",
        type=str,
        default="depth",
        help="GLORYS depth coordinate name.",
    )
    parser.add_argument(
        "--max-glorys-depth-m",
        type=float,
        default=1000.0,
        help="Only retain GLORYS target levels shallower than or equal to this depth.",
    )
    parser.add_argument(
        "--max-valid-distance-m",
        type=float,
        default=None,
        help=(
            "Optional validity cutoff in meters. If set, distances larger than this threshold "
            "can be treated as not having a valid nearby ARGO observation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated figures are written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of EN4 files to process.",
    )
    args = parser.parse_args()

    argo_files = find_nc_files(args.argo_dir, args.argo_glob)
    if args.max_files is not None and int(args.max_files) > 0:
        argo_files = argo_files[: int(args.max_files)]

    glorys_targets_df = load_retained_glorys_targets(
        glorys_dir=args.glorys_dir,
        glorys_glob=args.glorys_glob,
        depth_var_name=args.glorys_depth_var,
        max_glorys_depth_m=float(args.max_glorys_depth_m),
    )
    summary_df = summarize_nearest_profile_alignment(
        argo_files=argo_files,
        argo_depth_var_name=args.argo_depth_var,
        glorys_targets_df=glorys_targets_df,
        max_valid_distance_m=(
            None
            if args.max_valid_distance_m is None
            else float(args.max_valid_distance_m)
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = args.output_dir / "glorys_target_alignment_shift_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    summary_plot_path = _save_depth_summary_plot(
        summary_df,
        args.output_dir / "glorys_target_alignment_depth_summary.png",
        max_valid_distance_m=(
            None
            if args.max_valid_distance_m is None
            else float(args.max_valid_distance_m)
        ),
    )

    print(f"Wrote summary CSV: {summary_csv_path.resolve()}")
    print(f"Wrote plot: {summary_plot_path}")
    if args.max_valid_distance_m is not None:
        cutoff_plot_path = _save_within_cutoff_fraction_plot(
            summary_df,
            args.output_dir / "glorys_target_alignment_within_cutoff_fraction.png",
            max_valid_distance_m=float(args.max_valid_distance_m),
        )
        print(f"Wrote plot: {cutoff_plot_path}")


if __name__ == "__main__":
    main()
