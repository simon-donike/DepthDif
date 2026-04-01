from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset
from utils.normalizations import temperature_normalize


DEFAULT_DATA_CONFIG = "configs/px_space/data_ostia_argo_disk.yaml"
DEFAULT_OUTPUT_PATH = Path("temp/argo_glorys_profile_comparison.png")


def _resolve_index_path(csv_dir: Path, path_value: str) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else csv_dir / path


def _load_depth_axis_from_glorys_export(dataset: OstiaArgoTiffDataset, sample_idx: int) -> np.ndarray:
    row = dataset._rows[int(sample_idx)]
    glorys_path = _resolve_index_path(dataset.csv_dir, str(row[dataset._glorys_path_col]))
    with rasterio.open(glorys_path) as ds:
        tags = ds.tags()
    depth_text = str(tags.get("glorys_depth_m", "")).strip()
    if depth_text == "":
        return np.arange(50, dtype=np.float64)

    depth_values: list[float] = []
    for token in depth_text.split("|"):
        token_stripped = token.strip()
        if token_stripped == "":
            continue
        depth_values.append(float(token_stripped))
    if not depth_values:
        return np.arange(50, dtype=np.float64)
    return np.asarray(depth_values, dtype=np.float64)


def _build_standard_dataset(config_path: str, split: str) -> OstiaArgoTiffDataset:
    dataset = OstiaArgoTiffDataset.from_config(config_path=config_path, split=split)
    # Force the real on-disk Argo-vs-GLORYS comparison, not the synthetic sparse-GLORYS path.
    dataset.synthetic_mode = False
    return dataset


def _select_sample_index(dataset_len: int, sample_idx: int | None, seed: int) -> int:
    if dataset_len <= 0:
        raise RuntimeError("Dataset is empty.")
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= dataset_len:
            raise ValueError(f"sample_idx={sample_idx} is out of range for dataset length {dataset_len}.")
        return int(sample_idx)
    rng = np.random.default_rng(int(seed))
    return int(rng.integers(0, dataset_len))


def _plot_profiles(
    *,
    dataset: OstiaArgoTiffDataset,
    sample_idx: int,
    num_profiles: int,
    seed: int,
    output_path: Path,
    max_depth_value: float | None = None,
) -> Path:
    sample = dataset[int(sample_idx)]
    x = temperature_normalize(mode="denorm", tensor=sample["x"]).detach().float()
    y = temperature_normalize(mode="denorm", tensor=sample["y"]).detach().float()
    valid_mask = sample["x_valid_mask"].detach().bool()

    if x.ndim != 3 or y.ndim != 3 or valid_mask.ndim != 3:
        raise RuntimeError(
            "Expected x, y, and valid_mask to have shape (C,H,W), got "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}, valid_mask={tuple(valid_mask.shape)}."
        )
    if x.shape != y.shape or x.shape != valid_mask.shape:
        raise RuntimeError(
            "Expected x, y, and valid_mask to share the same shape, got "
            f"x={tuple(x.shape)}, y={tuple(y.shape)}, valid_mask={tuple(valid_mask.shape)}."
        )

    observed_map = valid_mask.any(dim=0)
    observed_coords = torch.nonzero(observed_map, as_tuple=False)
    if int(observed_coords.size(0)) <= 0:
        raise RuntimeError(f"Sample {sample_idx} has no observed Argo pixels to plot.")

    num_profiles = max(1, min(int(num_profiles), 9, int(observed_coords.size(0))))
    rng = np.random.default_rng(int(seed) + int(sample_idx))
    chosen_idx = rng.choice(int(observed_coords.size(0)), size=num_profiles, replace=False)
    chosen = observed_coords[torch.as_tensor(chosen_idx, dtype=torch.long)]

    depth_axis = _load_depth_axis_from_glorys_export(dataset, sample_idx)
    if int(depth_axis.size) != int(y.size(0)):
        # Keep plotting robust even if depth metadata is missing or malformed for a specific export.
        depth_axis = np.arange(int(y.size(0)), dtype=np.float64)
        depth_label = "GLORYS depth band"
    else:
        depth_label = "Depth (m)"

    deepest_observed_depth_value: float | None = None
    if max_depth_value is None:
        chosen_depth_values: list[float] = []
        for plot_idx in range(num_profiles):
            row_i = int(chosen[plot_idx, 0].item())
            col_i = int(chosen[plot_idx, 1].item())
            observed_profile = valid_mask[:, row_i, col_i].cpu().numpy()
            if bool(np.any(observed_profile)):
                chosen_depth_values.append(float(np.max(depth_axis[observed_profile])))
        if chosen_depth_values:
            deepest_observed_depth_value = float(max(chosen_depth_values))
    else:
        deepest_observed_depth_value = float(max_depth_value)

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.0), squeeze=False)
    axes_flat = axes.reshape(-1)
    try:
        for plot_idx, ax in enumerate(axes_flat):
            if plot_idx >= num_profiles:
                ax.set_axis_off()
                continue

            row_i = int(chosen[plot_idx, 0].item())
            col_i = int(chosen[plot_idx, 1].item())
            x_profile = x[:, row_i, col_i].cpu().numpy()
            y_profile = y[:, row_i, col_i].cpu().numpy()
            observed_profile = valid_mask[:, row_i, col_i].cpu().numpy()
            depth_mask = np.ones(depth_axis.shape, dtype=bool)
            if deepest_observed_depth_value is not None:
                depth_mask = depth_axis <= deepest_observed_depth_value

            ax.plot(
                y_profile[depth_mask],
                depth_axis[depth_mask],
                label="GLORYS",
                color="tab:blue",
                linewidth=2.0,
            )
            if bool(np.any(observed_profile)):
                # Plot only the truly observed Argo levels so zero-filled placeholders never appear.
                ax.plot(
                    x_profile[observed_profile],
                    depth_axis[observed_profile],
                    label="Argo x",
                    color="tab:orange",
                    marker="o",
                    linewidth=1.4,
                    markersize=3.5,
                )
            ax.invert_yaxis()
            if deepest_observed_depth_value is not None:
                ax.set_ylim(deepest_observed_depth_value, float(depth_axis[0]))
            ax.set_xlabel("Temperature (deg C)")
            ax.set_ylabel(depth_label)
            ax.set_title(f"Pixel ({row_i}, {col_i})")
            ax.grid(True, alpha=0.25)
            if plot_idx == 0:
                ax.legend(loc="best")

        coords = sample.get("coords")
        coords_text = ""
        if coords is not None:
            coords_np = coords.detach().cpu().numpy()
            coords_text = f", lat={coords_np[0]:.3f}, lon={coords_np[1]:.3f}"
        fig.suptitle(
            f"Standard Argo vs GLORYS profiles, sample {sample_idx}, date={int(sample['date'])}{coords_text}",
            fontsize=14,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 9 random observed Argo-vs-GLORYS temperature profiles from the standard "
            "OstiaArgoTiffDataset at the same pixel locations."
        )
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="Path to the OSTIA/ARGO/GLORYS disk dataset config.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("all", "train", "val"),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Optional explicit dataset sample index. If omitted, one sample is drawn from the seed.",
    )
    parser.add_argument(
        "--num-profiles",
        type=int,
        default=9,
        help="Number of observed pixels to plot, capped at 9.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for sample and pixel selection.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save the output PNG.",
    )
    args = parser.parse_args()

    dataset = _build_standard_dataset(config_path=str(args.data_config), split=str(args.split))
    sample_idx = _select_sample_index(
        dataset_len=len(dataset),
        sample_idx=args.sample_idx,
        seed=int(args.seed),
    )
    output_path = _plot_profiles(
        dataset=dataset,
        sample_idx=sample_idx,
        num_profiles=int(args.num_profiles),
        seed=int(args.seed),
        output_path=Path(args.output_path),
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()


"""
output_dir = Path('temp/argo_glorys_profile_comparison_batch')
output_dir.mkdir(parents=True, exist_ok=True)

dataset = _build_standard_dataset('configs/px_space/data_ostia_argo_disk.yaml', 'val')
rng = np.random.default_rng(31)
indices = rng.choice(len(dataset), size=15, replace=False)

for run_idx, sample_idx in enumerate(indices.tolist(), start=1):
    out_path = output_dir / f'profile_comparison_{run_idx:02d}_sample_{int(sample_idx)}.png'
    _plot_profiles(
        dataset=dataset,
        sample_idx=int(sample_idx),
        num_profiles=9,
        seed=31 + run_idx,
        output_path=out_path,
    )

print('output_dir', output_dir)
print('sample_indices', ','.join(str(int(v)) for v in indices.tolist()))
"""
