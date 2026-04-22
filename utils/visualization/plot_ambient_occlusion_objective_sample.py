from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset

DEFAULT_DATA_CONFIG = Path("configs/px_space/data_ostia_argo_disk.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/px_space/model_config.yaml")
DEFAULT_OUTPUT_PATH = Path("temp/ambient_occlusion_sample.png")
DEFAULT_VISUALIZATION_WITHHOLD_FRACTION = 0.33


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected a mapping at {path}, got {type(payload).__name__}."
        )
    return payload


def _resolve_dataset_defaults(data_config_path: Path) -> tuple[Path, bool, bool]:
    cfg = _load_yaml(data_config_path)
    ds_cfg = cfg.get("dataset", {})
    core_cfg = ds_cfg.get("core", {})
    output_cfg = ds_cfg.get("output", {})
    csv_path = Path(
        core_cfg.get("manifest_csv_path", OstiaArgoTiffDataset.DEFAULT_CSV_PATH)
    )
    return (
        csv_path,
        bool(output_cfg.get("return_info", True)),
        bool(output_cfg.get("return_coords", True)),
    )


def _resolve_ambient_defaults(model_config_path: Path) -> tuple[float, bool, int]:
    cfg = _load_yaml(model_config_path)
    model_cfg = cfg.get("model", {})
    ambient_cfg = model_cfg.get("ambient_occlusion", {})
    return (
        float(ambient_cfg.get("further_drop_prob", 0.1)),
        bool(ambient_cfg.get("shared_spatial_mask", True)),
        int(ambient_cfg.get("min_kept_observed_pixels", 50)),
    )


def _project_max(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    if values.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(
            f"Expected (C,H,W) tensors for projection, got {tuple(values.shape)} and {tuple(mask.shape)}."
        )
    masked = torch.where(mask, values, torch.full_like(values, float("-inf")))
    projected = masked.amax(dim=0)
    projected = torch.where(
        torch.isfinite(projected), projected, torch.zeros_like(projected)
    )
    return projected.detach().cpu().numpy().astype(np.float32, copy=False)


def _build_ambient_further_valid_mask(
    valid_mask: torch.Tensor,
    *,
    further_drop_prob: float,
    shared_spatial_mask: bool,
    min_kept_observed_pixels: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if valid_mask.ndim != 3:
        raise RuntimeError(
            f"Expected valid_mask with shape (C,H,W), got {tuple(valid_mask.shape)}."
        )

    base_mask = (valid_mask > 0).to(dtype=torch.float32)
    if further_drop_prob <= 0.0:
        return base_mask > 0.5

    keep_prob = 1.0 - float(further_drop_prob)
    channels, height, width = base_mask.shape
    if shared_spatial_mask:
        spatial_observed = base_mask.amax(dim=0) > 0.5
        observed_idx = torch.nonzero(
            spatial_observed.reshape(-1), as_tuple=False
        ).squeeze(1)
        spatial_keep = torch.zeros((height * width,), dtype=torch.float32)
        if observed_idx.numel() > 0:
            keep_count = int(round(float(observed_idx.numel()) * keep_prob))
            keep_count = max(0, min(int(observed_idx.numel()), keep_count))
            if keep_count > 0:
                chosen = observed_idx[
                    torch.randperm(int(observed_idx.numel()), generator=generator)[
                        :keep_count
                    ]
                ]
                spatial_keep[chosen] = 1.0
        keep_draw = spatial_keep.view(1, height, width).expand(channels, -1, -1)
    else:
        keep_draw = (
            torch.rand(
                (channels, height, width),
                generator=generator,
                dtype=torch.float32,
            )
            < keep_prob
        )

    further_mask = base_mask * keep_draw.to(dtype=base_mask.dtype)
    if int(min_kept_observed_pixels) <= 0:
        return further_mask > 0.5

    flat_base = base_mask.reshape(-1)
    flat_further = further_mask.reshape(-1)
    observed_idx = torch.nonzero(flat_base > 0.5, as_tuple=False).squeeze(1)
    if observed_idx.numel() == 0:
        return further_mask > 0.5

    min_keep = min(int(min_kept_observed_pixels), int(observed_idx.numel()))
    kept = int((flat_further > 0.5).sum().item())
    if kept < min_keep:
        needed = min_keep - kept
        choose = observed_idx[
            torch.randperm(int(observed_idx.numel()), generator=generator)[:needed]
        ]
        flat_further[choose] = 1.0
    return flat_further.view_as(further_mask) > 0.5


def _make_seen_withheld_rgb(
    seen_mask_1d: torch.Tensor,
    withheld_mask_1d: torch.Tensor,
) -> np.ndarray:
    if seen_mask_1d.ndim != 2 or withheld_mask_1d.ndim != 2:
        raise RuntimeError(
            "Expected 2D masks for RGB visualization, "
            f"got {tuple(seen_mask_1d.shape)} and {tuple(withheld_mask_1d.shape)}."
        )

    out = np.zeros((seen_mask_1d.shape[0], seen_mask_1d.shape[1], 3), dtype=np.float32)
    seen_np = seen_mask_1d.detach().cpu().numpy().astype(bool, copy=False)
    withheld_np = withheld_mask_1d.detach().cpu().numpy().astype(bool, copy=False)
    out[withheld_np, 0] = 1.0
    out[seen_np, 1] = 1.0
    return out


def _interpolate_zero_regions_for_display(image: np.ndarray) -> np.ndarray:
    """Fill zero-valued display holes with local neighbor averages for plotting only."""
    out = np.asarray(image, dtype=np.float32).copy()
    zero_mask = np.isfinite(out) & np.isclose(out, 0.0)
    if not np.any(zero_mask):
        return out

    # Iteratively diffuse neighboring non-zero values into zero-only display artifacts.
    for _ in range(out.shape[0] + out.shape[1]):
        pending = np.isfinite(out) & np.isclose(out, 0.0)
        if not np.any(pending):
            break

        neighbor_sum = np.zeros_like(out, dtype=np.float32)
        neighbor_count = np.zeros_like(out, dtype=np.int32)
        for row_shift, col_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(out, shift=(row_shift, col_shift), axis=(0, 1))
            shifted_valid = np.roll(
                ~np.isclose(out, 0.0), shift=(row_shift, col_shift), axis=(0, 1)
            )

            if row_shift == -1:
                shifted[-1, :] = 0.0
                shifted_valid[-1, :] = False
            elif row_shift == 1:
                shifted[0, :] = 0.0
                shifted_valid[0, :] = False
            if col_shift == -1:
                shifted[:, -1] = 0.0
                shifted_valid[:, -1] = False
            elif col_shift == 1:
                shifted[:, 0] = 0.0
                shifted_valid[:, 0] = False

            neighbor_sum += np.where(shifted_valid, shifted, 0.0)
            neighbor_count += shifted_valid.astype(np.int32)

        fillable = pending & (neighbor_count > 0)
        if not np.any(fillable):
            break
        out[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]

    return out


def _plot_sample_figure(
    *,
    sample: dict,
    further_valid_mask: torch.Tensor,
    output_path: Path,
    sample_index: int,
    further_drop_prob: float,
    seed: int,
) -> Path:
    x = sample["x"].detach().float().cpu()
    y = sample["y"].detach().float().cpu()
    eo = sample["eo"].detach().float().cpu()
    valid_mask = sample["x_valid_mask"].detach().cpu().bool()
    valid_mask_1d = valid_mask.any(dim=0)
    further_valid_mask = further_valid_mask.detach().cpu().bool()
    seen_mask_1d = further_valid_mask.any(dim=0)
    withheld_mask = valid_mask & ~further_valid_mask
    withheld_mask_1d = withheld_mask.any(dim=0)

    argo_projection = _project_max(x, valid_mask)
    seen_projection = _project_max(x, further_valid_mask)
    withheld_projection = _project_max(x, withheld_mask)
    # Use the shallowest GLORYS level so the panel matches the surface reference the user asked for.
    glorys_projection = _interpolate_zero_regions_for_display(
        y[0].numpy().astype(np.float32, copy=False)
    )
    rgb_mask = _make_seen_withheld_rgb(seen_mask_1d, withheld_mask_1d)
    ostia_img = eo[0].numpy().astype(np.float32, copy=False)

    finite_value_panels = [
        arr[np.isfinite(arr)]
        for arr in (
            argo_projection,
            seen_projection,
            withheld_projection,
            glorys_projection,
        )
    ]
    finite_value_panels = [arr for arr in finite_value_panels if arr.size > 0]
    if finite_value_panels:
        stacked = np.concatenate(finite_value_panels)
        value_vmin = float(np.percentile(stacked, 2.0))
        value_vmax = float(np.percentile(stacked, 98.0))
        if (
            not np.isfinite(value_vmin)
            or not np.isfinite(value_vmax)
            or value_vmin >= value_vmax
        ):
            value_vmin = float(np.nanmin(stacked))
            value_vmax = float(np.nanmax(stacked))
    else:
        value_vmin, value_vmax = 0.0, 1.0

    surface_finite_panels = [
        arr[np.isfinite(arr)]
        for arr in (ostia_img, glorys_projection)
        if np.isfinite(arr).any()
    ]
    if surface_finite_panels:
        surface_values = np.concatenate(surface_finite_panels)
        surface_vmin = float(np.percentile(surface_values, 2.0))
        surface_vmax = float(np.percentile(surface_values, 98.0))
        if (
            not np.isfinite(surface_vmin)
            or not np.isfinite(surface_vmax)
            or surface_vmin >= surface_vmax
        ):
            surface_vmin = float(np.nanmin(surface_values))
            surface_vmax = float(np.nanmax(surface_values))
    else:
        surface_vmin, surface_vmax = 0.0, 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180, constrained_layout=True)
    axes_flat = axes.reshape(-1)

    image_panels = [
        (
            argo_projection,
            "viridis",
            "Argo max over channels\n(original 1D observed mask)",
            value_vmin,
            value_vmax,
        ),
        (
            seen_projection,
            "viridis",
            "Points model still sees\n(after ambient further occlusion)",
            value_vmin,
            value_vmax,
        ),
        (
            withheld_projection,
            "viridis",
            "Points withheld from model",
            value_vmin,
            value_vmax,
        ),
        (
            rgb_mask,
            None,
            "Seen/withheld mask\ngreen=seen, red=withheld, black=empty",
            None,
            None,
        ),
        (
            ostia_img,
            "coolwarm",
            "OSTIA",
            surface_vmin,
            surface_vmax,
        ),
        (
            glorys_projection,
            "coolwarm",
            "GLORYS surface level",
            surface_vmin,
            surface_vmax,
        ),
    ]

    for ax, (img, cmap, title, vmin, vmax) in zip(axes_flat, image_panels):
        if cmap is None:
            ax.imshow(img)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_axis_off()

    observed_pixels = int(valid_mask_1d.sum().item())
    seen_pixels = int(seen_mask_1d.sum().item())
    withheld_pixels = int(withheld_mask_1d.sum().item())
    sample_date = int(sample.get("date", 0))
    fig.suptitle(
        "Ambient occlusion sample visualization\n"
        f"idx={sample_index} date={sample_date} seed={seed} further_drop_prob={further_drop_prob:.3f} "
        f"observed_1d={observed_pixels} seen_1d={seen_pixels} withheld_1d={withheld_pixels}",
        fontsize=13,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one real OSTIA/ARGO/GLORYS sample together with the ambient "
            "diffusion further-occlusion split into seen and withheld observations."
        )
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Dataset config used to resolve the manifest CSV and dataset output flags.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Model config used to resolve ambient-occlusion defaults.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional manifest CSV override. Defaults to the dataset config value.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("all", "train", "val"),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset index to visualize within the selected split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for the ambient further-occlusion mask sampling.",
    )
    parser.add_argument(
        "--further-drop-prob",
        type=float,
        default=DEFAULT_VISUALIZATION_WITHHOLD_FRACTION,
        help="Fraction of observed points to withhold in the visualization.",
    )
    parser.add_argument(
        "--shared-spatial-mask",
        dest="shared_spatial_mask",
        action="store_true",
        help="Use one spatial further-mask shared across all depth channels.",
    )
    parser.add_argument(
        "--per-channel-mask",
        dest="shared_spatial_mask",
        action="store_false",
        help="Sample the further-mask independently per depth channel.",
    )
    parser.add_argument(
        "--min-kept-observed-pixels",
        type=int,
        default=None,
        help="Optional override for the minimum number of originally observed channel-pixels kept.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path for the generated figure.",
    )
    parser.set_defaults(shared_spatial_mask=None)
    args = parser.parse_args()

    csv_path_default, return_info, return_coords = _resolve_dataset_defaults(
        args.data_config
    )
    ambient_drop_default, shared_mask_default, min_keep_default = (
        _resolve_ambient_defaults(args.model_config)
    )

    dataset = OstiaArgoTiffDataset(
        csv_path=args.csv_path if args.csv_path is not None else csv_path_default,
        split=args.split,
        return_info=return_info,
        return_coords=return_coords,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} is out of range for split={args.split!r} with {len(dataset)} samples."
        )

    sample = dataset[int(args.sample_index)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    further_drop_prob = float(
        ambient_drop_default
        if args.further_drop_prob is None
        else args.further_drop_prob
    )
    shared_spatial_mask = (
        shared_mask_default
        if args.shared_spatial_mask is None
        else bool(args.shared_spatial_mask)
    )
    min_kept_observed_pixels = (
        min_keep_default
        if args.min_kept_observed_pixels is None
        else int(args.min_kept_observed_pixels)
    )
    further_valid_mask = _build_ambient_further_valid_mask(
        sample["x_valid_mask"],
        further_drop_prob=further_drop_prob,
        shared_spatial_mask=shared_spatial_mask,
        min_kept_observed_pixels=min_kept_observed_pixels,
        generator=generator,
    )

    out_path = _plot_sample_figure(
        sample=sample,
        further_valid_mask=further_valid_mask,
        output_path=args.output_path,
        sample_index=int(args.sample_index),
        further_drop_prob=float(further_drop_prob),
        seed=int(args.seed),
    )
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
