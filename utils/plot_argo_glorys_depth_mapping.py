from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MAPPING_JSON = Path("data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json")
DEFAULT_OUTPUT_DIR = Path("data/glorys_argo_alignment/figures")


def _load_mapping_rows(mapping_json_path: Path) -> list[dict[str, float | int]]:
    """Load the saved ARGO-to-GLORYS mapping rows from JSON."""
    payload = json.loads(mapping_json_path.read_text(encoding="utf-8"))
    rows = payload.get("mapping", [])
    if not isinstance(rows, list) or len(rows) == 0:
        raise RuntimeError(f"Mapping JSON does not contain a non-empty 'mapping' list: {mapping_json_path}")
    return rows


def _mapping_arrays(rows: list[dict[str, float | int]]) -> dict[str, np.ndarray]:
    """Convert JSON mapping rows into sorted numpy arrays for plotting."""
    rows_sorted = sorted(rows, key=lambda row: int(row["argo_level_index"]))
    return {
        "argo_idx": np.asarray([int(row["argo_level_index"]) for row in rows_sorted], dtype=np.int64),
        "argo_depth_m": np.asarray([float(row["argo_depth_m"]) for row in rows_sorted], dtype=np.float64),
        "glorys_idx": np.asarray([int(row["glorys_level_index"]) for row in rows_sorted], dtype=np.int64),
        "glorys_depth_m": np.asarray([float(row["glorys_depth_m"]) for row in rows_sorted], dtype=np.float64),
        "abs_diff_m": np.asarray(
            [float(row["absolute_depth_difference_m"]) for row in rows_sorted],
            dtype=np.float64,
        ),
        "argo_valid_profile_count": np.asarray(
            [int(row.get("argo_valid_profile_count", 0)) for row in rows_sorted],
            dtype=np.int64,
        ),
    }


def _save_depth_vs_index_plot(arrays: dict[str, np.ndarray], output_path: Path) -> Path:
    """Save a line plot comparing representative ARGO depths and chosen GLORYS depths."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.plot(
        arrays["argo_idx"],
        arrays["argo_depth_m"],
        color="#0b4f6c",
        linewidth=2.0,
        label="ARGO representative depth",
    )
    ax.plot(
        arrays["argo_idx"],
        arrays["glorys_depth_m"],
        color="#d97706",
        linewidth=2.0,
        label="Matched GLORYS depth",
    )
    # A log depth axis makes the shallow and deep structure readable in one figure.
    ax.set_yscale("log")
    ax.set_xlabel("ARGO Level Index")
    ax.set_ylabel("Depth (m, log scale)")
    ax.set_title("ARGO vs GLORYS Depth Levels by ARGO Channel Index")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def _save_absolute_difference_plot(arrays: dict[str, np.ndarray], output_path: Path) -> Path:
    """Save the absolute depth mismatch per ARGO channel."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=180)
    ax.plot(
        arrays["argo_idx"],
        arrays["abs_diff_m"],
        color="#b91c1c",
        linewidth=1.8,
    )
    ax.fill_between(
        arrays["argo_idx"],
        0.0,
        arrays["abs_diff_m"],
        color="#ef4444",
        alpha=0.18,
    )
    ax.set_xlabel("ARGO Level Index")
    ax.set_ylabel("Absolute Depth Difference (m)")
    ax.set_title("Absolute ARGO-GLORYS Depth Mismatch by ARGO Channel Index")
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def _save_depth_scatter_plot(arrays: dict[str, np.ndarray], output_path: Path) -> Path:
    """Save a parity-style scatter showing ARGO depth against matched GLORYS depth."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    scatter = ax.scatter(
        arrays["argo_depth_m"],
        arrays["glorys_depth_m"],
        c=arrays["abs_diff_m"],
        cmap="viridis",
        s=28.0,
        alpha=0.9,
        edgecolors="none",
    )
    max_depth = float(max(np.nanmax(arrays["argo_depth_m"]), np.nanmax(arrays["glorys_depth_m"])))
    ax.plot([0.0, max_depth], [0.0, max_depth], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("ARGO Representative Depth (m)")
    ax.set_ylabel("Matched GLORYS Depth (m)")
    ax.set_title("Representative ARGO Depth vs Matched GLORYS Depth")
    ax.grid(True, alpha=0.25, linestyle="--")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Absolute Difference (m)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def _save_profile_count_plot(arrays: dict[str, np.ndarray], output_path: Path) -> Path:
    """Save how many profiles contributed to each representative ARGO depth."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=180)
    ax.plot(
        arrays["argo_idx"],
        arrays["argo_valid_profile_count"],
        color="#166534",
        linewidth=1.8,
    )
    ax.set_xlabel("ARGO Level Index")
    ax.set_ylabel("Valid Profile Count")
    ax.set_title("Number of Profiles Contributing to Each Representative ARGO Level")
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    """Render plots for the saved ARGO-to-GLORYS depth mapping JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the saved ARGO-to-GLORYS depth mapping with depth-index curves, "
            "absolute mismatch, and parity scatter plots."
        )
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=DEFAULT_MAPPING_JSON,
        help="Path to the saved ARGO-to-GLORYS mapping JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated figures are written.",
    )
    args = parser.parse_args()

    rows = _load_mapping_rows(args.mapping_json)
    arrays = _mapping_arrays(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_vs_index_path = _save_depth_vs_index_plot(arrays, args.output_dir / "argo_glorys_depth_vs_index.png")
    abs_diff_path = _save_absolute_difference_plot(
        arrays,
        args.output_dir / "argo_glorys_absolute_difference.png",
    )
    scatter_path = _save_depth_scatter_plot(arrays, args.output_dir / "argo_glorys_depth_scatter.png")
    profile_count_path = _save_profile_count_plot(
        arrays,
        args.output_dir / "argo_level_valid_profile_count.png",
    )

    print(f"Wrote plot: {depth_vs_index_path}")
    print(f"Wrote plot: {abs_diff_path}")
    print(f"Wrote plot: {scatter_path}")
    print(f"Wrote plot: {profile_count_path}")


if __name__ == "__main__":
    main()
