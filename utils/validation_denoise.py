from __future__ import annotations

from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from utils.normalizations import temperature_normalize
from utils.stretching import minmax_stretch


def build_capture_indices(
    total_steps: int,
    intermediate_step_indices: list[int] | None,
) -> set[int]:
    if total_steps < 0:
        return set()
    if intermediate_step_indices is None:
        return set(range(0, int(total_steps) + 1))
    return {
        int(step)
        for step in intermediate_step_indices
        if 0 <= int(step) <= int(total_steps)
    }


def build_evenly_spaced_capture_steps(total_steps: int, num_frames: int) -> list[int]:
    if total_steps <= 0 or num_frames <= 0:
        return []
    # Include start-noise (step 0) and final sample (step total_steps).
    raw = torch.linspace(0, total_steps, steps=min(total_steps + 1, num_frames))
    rounded = raw.round().long().tolist()
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for step in rounded:
        step_i = int(step)
        if step_i in seen:
            continue
        seen.add(step_i)
        ordered_unique.append(step_i)
    return ordered_unique


def step_to_sampler_timestep_label(
    *,
    step_index: int,
    total_steps: int,
    sampler: Any,
) -> int:
    if total_steps <= 0:
        return 0
    step_index = int(max(0, min(step_index, total_steps)))
    if hasattr(sampler, "ddim_train_steps"):
        ddim_train_steps = sampler.ddim_train_steps.detach().long().cpu().tolist()
        if not ddim_train_steps:
            return 0
        if step_index >= total_steps:
            return 0
        reverse_idx = max(
            0,
            min(len(ddim_train_steps) - 1 - step_index, len(ddim_train_steps) - 1),
        )
        return int(ddim_train_steps[reverse_idx])
    if step_index >= total_steps:
        return 0
    return int(max(0, total_steps - 1 - step_index))


def log_wandb_denoise_timestep_grid(
    *,
    logger: Any,
    denoise_samples: list[tuple[int, torch.Tensor]],
    total_steps: int,
    sampler: Any,
    conditioning_image: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    prefix: str = "val",
    cmap: str = "turbo",
    nrows: int = 4,
    ncols: int = 4,
    tile_size_px: int = 128,
    tile_pad_px: int = 2,
) -> None:
    if not denoise_samples:
        return
    if logger is None or not hasattr(logger, "experiment"):
        return
    experiment = logger.experiment
    if not hasattr(experiment, "log"):
        return

    try:
        import wandb
    except Exception:
        return

    max_plots = nrows * ncols
    tile_size_px = int(max(16, tile_size_px))
    tile_pad_px = int(max(0, tile_pad_px))
    canvas_h = (nrows * tile_size_px) + ((nrows - 1) * tile_pad_px)
    canvas_w = (ncols * tile_size_px) + ((ncols - 1) * tile_pad_px)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    cmap_fn = cm.get_cmap(cmap)
    timestep_labels: list[str] = []

    sorted_samples = sorted(denoise_samples, key=lambda item: int(item[0]))
    final_step = int(sorted_samples[-1][0])
    intermediate_candidates = [
        (int(step_idx), sample_t)
        for step_idx, sample_t in sorted_samples
        if int(step_idx) != 0 and int(step_idx) != final_step
    ]
    if len(intermediate_candidates) >= 14:
        pick_positions = np.linspace(
            0, len(intermediate_candidates) - 1, num=14
        ).round().astype(int)
        picked_intermediates = [intermediate_candidates[int(i)] for i in pick_positions]
    else:
        picked_intermediates = intermediate_candidates

    plot_entries: list[tuple[str, int | None, torch.Tensor]] = []
    if conditioning_image is not None:
        plot_entries.append(("cond", None, conditioning_image))

    for step_idx, sample_t in picked_intermediates:
        plot_entries.append(("intermediate", step_idx, sample_t))
        if len(plot_entries) >= 15:
            break

    while len(plot_entries) < 15 and picked_intermediates:
        plot_entries.append(
            (
                "intermediate",
                int(picked_intermediates[-1][0]),
                picked_intermediates[-1][1],
            )
        )

    plot_entries.append(("final", final_step, sorted_samples[-1][1]))

    for plot_idx in range(max_plots):
        if plot_idx >= len(plot_entries):
            continue

        entry_kind, step_idx, sample_t = plot_entries[plot_idx]
        mask_i: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.ndim == 4:
                mask_i = valid_mask[0, 0]
            elif valid_mask.ndim == 3:
                mask_i = valid_mask[0]
            elif valid_mask.ndim == 2:
                mask_i = valid_mask

        image_t = sample_t[0, 0].detach().float()
        image_t = torch.nan_to_num(image_t, nan=0.0, posinf=0.0, neginf=0.0)
        image_t = temperature_normalize(mode="denorm", tensor=image_t)
        image_plot = minmax_stretch(image_t, mask=mask_i, nodata_value=None)
        image_plot = F.interpolate(
            image_plot.unsqueeze(0).unsqueeze(0),
            size=(tile_size_px, tile_size_px),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        image_np = image_plot.cpu().numpy()
        rgb = (cmap_fn(image_np)[..., :3] * 255.0).astype(np.uint8)

        row = plot_idx // ncols
        col = plot_idx % ncols
        y0 = row * (tile_size_px + tile_pad_px)
        x0 = col * (tile_size_px + tile_pad_px)
        canvas[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px, :] = rgb

        if entry_kind == "cond":
            timestep_labels.append(f"{plot_idx + 1}:cond")
        else:
            sampler_t = step_to_sampler_timestep_label(
                step_index=int(step_idx),
                total_steps=total_steps,
                sampler=sampler,
            )
            timestep_labels.append(f"{plot_idx + 1}:t={sampler_t}/s={int(step_idx)}")

    caption = "conditioning + 14 intermediates + final"
    experiment.log(
        {
            f"{prefix}/denoise_timestep_grid_4x4": wandb.Image(
                canvas,
                caption=caption,
            )
        }
    )
