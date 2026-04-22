from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils.normalizations import temperature_normalize, temperature_to_plot_unit


def build_capture_indices(
    total_steps: int,
    intermediate_step_indices: list[int] | None,
) -> set[int]:
    """Build and return capture indices.

    Args:
        total_steps (int): Step or timestep value.
        intermediate_step_indices (list[int] | None): Input value.

    Returns:
        set[int]: Computed output value.
    """
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
    """Build and return evenly spaced capture steps.

    Args:
        total_steps (int): Step or timestep value.
        num_frames (int): Size/count parameter.

    Returns:
        list[int]: List containing computed outputs.
    """
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
    """Compute step to sampler timestep label and return the result.

    Args:
        step_index (int): Input value.
        total_steps (int): Step or timestep value.
        sampler (Any): Sampler instance used for reverse diffusion.

    Returns:
        int: Computed scalar output.
    """
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
    mae_samples: list[tuple[int, torch.Tensor]] | None = None,
    total_steps: int,
    sampler: Any,
    conditioning_image: torch.Tensor | None = None,
    eo_conditioning_image: torch.Tensor | None = None,
    ground_truth: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    land_mask: torch.Tensor | None = None,
    prefix: str = "val_imgs",
    cmap: str = "turbo",
    nrows: int = 4,
    ncols: int = 4,
    tile_size_px: int = 128,
    tile_pad_px: int = 2,
) -> None:
    """Log wandb denoise timestep grid for monitoring.

    Args:
        logger (Any): Logger instance used for experiment tracking.
        denoise_samples (list[tuple[int, torch.Tensor]]): Tensor input for the computation.
        mae_samples (list[tuple[int, torch.Tensor]] | None): Tensor input for the computation.
        total_steps (int): Step or timestep value.
        sampler (Any): Sampler instance used for reverse diffusion.
        conditioning_image (torch.Tensor | None): Tensor input for the computation.
        eo_conditioning_image (torch.Tensor | None): Tensor input for the computation.
        ground_truth (torch.Tensor | None): Tensor input for the computation.
        valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        prefix (str): Input value.
        cmap (str): Input value.
        nrows (int): Input value.
        ncols (int): Input value.
        tile_size_px (int): Input value.
        tile_pad_px (int): Input value.

    Returns:
        None: No value is returned.
    """
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
        pick_positions = (
            np.linspace(0, len(intermediate_candidates) - 1, num=14).round().astype(int)
        )
        picked_intermediates = [intermediate_candidates[int(i)] for i in pick_positions]
    else:
        picked_intermediates = intermediate_candidates

    plot_entries: list[tuple[str, int | None, torch.Tensor]] = []
    if conditioning_image is not None:
        plot_entries.append(("cond", None, conditioning_image))
    if eo_conditioning_image is not None:
        plot_entries.append(("eo", None, eo_conditioning_image))

    max_entries_before_final = max(0, max_plots - 1)
    for step_idx, sample_t in picked_intermediates:
        plot_entries.append(("intermediate", step_idx, sample_t))
        if len(plot_entries) >= max_entries_before_final:
            break

    while len(plot_entries) < max_entries_before_final and picked_intermediates:
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
        # valid_mask is intentionally not applied here so generated and observed pixels
        # remain visible together in denoising previews.
        _ = valid_mask
        mask_i: torch.Tensor | None = _mask_for_sample(land_mask, 0)
        if mask_i is not None and mask_i.ndim == 3:
            mask_i = mask_i[0]

        image_t = sample_t[0, 0].detach().float()
        image_t = temperature_normalize(mode="denorm", tensor=image_t)
        image_plot = torch.from_numpy(
            _temperature_band_to_plot_image(image_t, mask=mask_i)
        ).to(device=image_t.device, dtype=image_t.dtype)
        image_plot = (
            F.interpolate(
                image_plot.unsqueeze(0).unsqueeze(0),
                size=(tile_size_px, tile_size_px),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        image_np = image_plot.cpu().numpy()
        rgb = (cmap_fn(image_np)[..., :3] * 255.0).astype(np.uint8)

        row = plot_idx // ncols
        col = plot_idx % ncols
        y0 = row * (tile_size_px + tile_pad_px)
        x0 = col * (tile_size_px + tile_pad_px)
        canvas[y0 : y0 + tile_size_px, x0 : x0 + tile_size_px, :] = rgb

        if entry_kind == "cond":
            timestep_labels.append(f"{plot_idx + 1}:cond")
        elif entry_kind == "eo":
            timestep_labels.append(f"{plot_idx + 1}:eo")
        else:
            sampler_t = step_to_sampler_timestep_label(
                step_index=int(step_idx),
                total_steps=total_steps,
                sampler=sampler,
            )
            timestep_labels.append(f"{plot_idx + 1}:t={sampler_t}/s={int(step_idx)}")

    if conditioning_image is not None and eo_conditioning_image is not None:
        caption = "conditioning + eo + intermediates + final"
    elif conditioning_image is not None:
        caption = "conditioning + intermediates + final"
    else:
        caption = "intermediates + final"
    experiment.log(
        {
            f"{prefix}/denoise_timestep_grid_4x4": wandb.Image(
                canvas,
                caption=caption,
            )
        }
    )

    mae_source = denoise_samples if mae_samples is None else mae_samples
    mae_steps: list[int] = []
    mae_vals: list[float] = []
    if ground_truth is not None:
        gt = ground_truth.detach().float()
        for step_idx, sample_t in mae_source:
            sample = sample_t.detach().float()
            if sample.shape != gt.shape:
                continue
            # Intentionally unmasked: MAE is computed over the full image tensor.
            mae = torch.mean(torch.abs(sample - gt))
            mae_steps.append(int(step_idx))
            mae_vals.append(float(mae.item()))
        if mae_steps:
            by_step: dict[int, list[float]] = {}
            for step_i, mae_i in zip(mae_steps, mae_vals):
                by_step.setdefault(int(step_i), []).append(float(mae_i))
            mae_steps = sorted(by_step.keys())
            mae_vals = [
                float(sum(by_step[step_i]) / max(1, len(by_step[step_i])))
                for step_i in mae_steps
            ]
            fig_mae, ax_mae = plt.subplots(figsize=(5, 3), dpi=150)
            mae_line = ax_mae.plot(
                np.asarray(mae_steps, dtype=np.int32),
                np.asarray(mae_vals, dtype=np.float32),
                linewidth=1.5,
                color="#d62728",
                marker="o",
                markersize=2.5,
                label=(
                    "MAE (x0_pred vs target)"
                    if mae_samples is not None
                    else "MAE (intermediate vs target)"
                ),
            )
            ax_mae.set_xlabel("Reverse step")
            ax_mae.set_ylabel("MAE")
            ax_mae.set_title("Intermediate MAE vs Reverse Diff. Step")
            ax_mae.invert_xaxis()
            ax_mae.grid(True, alpha=0.3, linewidth=0.5)
            handles = list(mae_line)
            labels = [h.get_label() for h in handles]
            ax_mae.legend(handles, labels, loc="best")
            ax_mae.text(
                0.01,
                0.02,
                f"MAE start={mae_vals[-1]:.3f}, end={mae_vals[0]:.3f}",
                transform=ax_mae.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                color="#333333",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2.0),
            )
            fig_mae.tight_layout()
            experiment.log({f"{prefix}/intermediate_mae_vs_step": wandb.Image(fig_mae)})
            plt.close(fig_mae)


def log_wandb_diffusion_schedule_profile(
    *,
    logger: Any,
    sampler: Any,
    total_steps: int,
    prefix: str = "val_imgs",
    eps: float = 1e-12,
) -> None:
    """Log wandb diffusion schedule profile for monitoring.

    Args:
        logger (Any): Logger instance used for experiment tracking.
        sampler (Any): Sampler instance used for reverse diffusion.
        total_steps (int): Step or timestep value.
        prefix (str): Input value.
        eps (float): Input value.

    Returns:
        None: No value is returned.
    """
    if total_steps <= 0:
        return
    if sampler is None:
        return
    if not hasattr(sampler, "alphas_cumprod") or not hasattr(sampler, "betas"):
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

    alpha_cumprod = sampler.alphas_cumprod.detach().float().cpu()
    betas = sampler.betas.detach().float().cpu()
    if alpha_cumprod.ndim != 1 or alpha_cumprod.numel() == 0:
        return
    if betas.ndim != 1 or betas.numel() == 0:
        return

    step_indices = list(range(max(0, int(total_steps))))
    if not step_indices:
        return

    reverse_t_list = [
        step_to_sampler_timestep_label(
            step_index=step_idx,
            total_steps=int(total_steps),
            sampler=sampler,
        )
        for step_idx in step_indices
    ]
    reverse_t = torch.as_tensor(reverse_t_list, dtype=torch.long).clamp(
        min=0, max=int(alpha_cumprod.numel() - 1)
    )
    # For DDIM, reverse steps live on a sparse subset of the training-time schedule.
    # Reuse the same mapped timesteps in forward order so both panels describe the
    # same trajectory rather than comparing sparse reverse steps to dense early train steps.
    if hasattr(sampler, "ddim_train_steps"):
        forward_t = torch.flip(reverse_t, dims=[0])
    else:
        forward_t = torch.arange(0, int(total_steps), dtype=torch.long).clamp(
            min=0, max=int(alpha_cumprod.numel() - 1)
        )

    def _schedule_terms(t_idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Helper that computes schedule terms.

        Args:
            t_idx (torch.Tensor): Tensor input for the computation.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing computed outputs.
        """
        alpha_bar_t = alpha_cumprod[t_idx]
        beta_t = betas[t_idx]
        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=0.0))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))
        snr_t = alpha_bar_t / torch.clamp(1.0 - alpha_bar_t, min=float(eps))
        log_snr_t = torch.log10(torch.clamp(snr_t, min=float(eps)))

        prev_t = torch.clamp(t_idx - 1, min=0)
        alpha_bar_prev = alpha_cumprod[prev_t]
        alpha_bar_prev = torch.where(
            t_idx > 0,
            alpha_bar_prev,
            torch.ones_like(alpha_bar_prev),
        )
        beta_tilde_t = (
            beta_t
            * (1.0 - alpha_bar_prev)
            / torch.clamp(1.0 - alpha_bar_t, min=float(eps))
        )
        beta_tilde_t = torch.clamp(beta_tilde_t, min=0.0)
        return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t, beta_tilde_t, log_snr_t

    rev_sqrt_ab, rev_sqrt_1mab, rev_beta_tilde, rev_log_snr = _schedule_terms(reverse_t)
    fwd_sqrt_ab, fwd_sqrt_1mab, fwd_beta_tilde, fwd_log_snr = _schedule_terms(forward_t)

    x_rev = np.asarray(step_indices, dtype=np.int32)
    x_fwd = np.asarray(step_indices, dtype=np.int32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    def _plot_panel(
        ax: Any,
        x_vals: np.ndarray,
        sqrt_ab: torch.Tensor,
        sqrt_1mab: torch.Tensor,
        beta_tilde: torch.Tensor,
        log_snr: torch.Tensor,
        *,
        title: str,
        xlabel: str,
    ) -> None:
        """Helper that computes plot panel.

        Args:
            ax (Any): Input value.
            x_vals (np.ndarray): Input value.
            sqrt_ab (torch.Tensor): Tensor input for the computation.
            sqrt_1mab (torch.Tensor): Tensor input for the computation.
            beta_tilde (torch.Tensor): Tensor input for the computation.
            log_snr (torch.Tensor): Tensor input for the computation.
            title (str): Input value.
            xlabel (str): Input value.

        Returns:
            None: No value is returned.
        """
        l_sqrt_ab = ax.plot(
            x_vals,
            sqrt_ab.numpy(),
            color="#1f77b4",
            linewidth=1.2,
            label="sqrt(alpha_bar_t)",
        )
        l_sqrt_1mab = ax.plot(
            x_vals,
            sqrt_1mab.numpy(),
            color="#2ca02c",
            linewidth=1.2,
            label="sqrt(1-alpha_bar_t)",
        )
        l_beta_tilde = ax.plot(
            x_vals,
            beta_tilde.numpy(),
            color="#9467bd",
            linewidth=1.2,
            label="beta_tilde_t",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Schedule values")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_title(title)

        ax_log_snr = ax.twinx()
        l_log_snr = ax_log_snr.plot(
            x_vals,
            log_snr.numpy(),
            color="#d62728",
            linewidth=1.5,
            linestyle="--",
            label="log10(SNR+eps)",
        )
        ax_log_snr.set_ylabel("log10(SNR + eps)", color="#d62728")
        ax_log_snr.tick_params(axis="y", labelcolor="#d62728")

        handles = l_sqrt_ab + l_sqrt_1mab + l_beta_tilde + l_log_snr
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="best")

    _plot_panel(
        axes[0],
        x_rev,
        rev_sqrt_ab,
        rev_sqrt_1mab,
        rev_beta_tilde,
        rev_log_snr,
        title="Reverse Process",
        xlabel="Reverse step",
    )
    _plot_panel(
        axes[1],
        x_fwd,
        fwd_sqrt_ab,
        fwd_sqrt_1mab,
        fwd_beta_tilde,
        fwd_log_snr,
        title="Forward Process",
        xlabel="Forward step",
    )

    descriptor_text = (
        "β̃ₜ  “How violent is this step?”\n"
        "√ᾱₜ  “Is the original image still visible?”\n"
        "√(1−ᾱₜ)  “Is noise dominating the pixels?”\n"
        "log-SNR  “How difficult is denoising here?”"
    )
    fig.text(0.02, 0.01, descriptor_text, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    experiment.log({f"{prefix}/diffusion_schedule_vs_step": wandb.Image(fig)})
    plt.close(fig)


def _mask_for_sample(
    mask: torch.Tensor | None,
    sample_idx: int,
) -> torch.Tensor | None:
    """Helper that computes mask for sample.

    Args:
        mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        sample_idx (int): Input value.

    Returns:
        torch.Tensor | None: Tensor output produced by this call.
    """
    if mask is None:
        return None
    if mask.ndim == 4:
        sample_mask = mask[sample_idx]
        if sample_mask.size(0) == 1:
            return sample_mask[0]
        # Keep per-band masks so caller can pick the plotted channel explicitly.
        return sample_mask
    if mask.ndim == 3:
        return mask[sample_idx]
    if mask.ndim == 2:
        return mask
    return None


def _plot_band_image(
    tensor: torch.Tensor,
    sample_idx: int,
    *,
    band_idx: int = 0,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Helper that computes plot band image.

    Args:
        tensor (torch.Tensor): Tensor input for the computation.
        sample_idx (int): Input value.
        band_idx (int): Input value.
        mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.

    Returns:
        np.ndarray: Computed output value.
    """
    if tensor.ndim == 4:
        channel_idx = int(max(0, min(int(band_idx), int(tensor.size(1)) - 1)))
        image_t = tensor[sample_idx, channel_idx].detach().float()
    elif tensor.ndim == 3:
        image_t = tensor[sample_idx].detach().float()
    elif tensor.ndim == 2:
        image_t = tensor.detach().float()
    else:
        raise RuntimeError(
            f"Expected tensor ndim in {{2,3,4}} for plotting, got {int(tensor.ndim)}."
        )
    return _temperature_band_to_plot_image(image_t, mask=mask)


def _temperature_band_to_plot_image(
    image_t: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Convert one denormalized temperature band into the shared plot color scale."""
    image_t = image_t.detach().float()
    finite_mask = torch.isfinite(image_t)
    if mask is not None:
        finite_mask = finite_mask & (mask > 0.5).to(device=image_t.device)
    # Use one global Celsius plotting range so EO, GLORYS, and reconstructions share colors.
    image_plot = temperature_to_plot_unit(image_t, tensor_is_normalized=False)
    image_plot = torch.where(
        finite_mask,
        image_plot,
        torch.zeros_like(image_plot),
    )
    return image_plot.cpu().numpy().astype(np.float32)


def log_wandb_conditional_reconstruction_grid(
    *,
    logger: Any,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    y_hat: torch.Tensor,
    y_target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    land_mask: torch.Tensor | None = None,
    eo: torch.Tensor | None = None,
    prefix: str = "val_imgs",
    image_key: str = "x_y_full_reconstruction",
    cmap: str = "turbo",
    show_valid_mask_panel: bool = True,
) -> None:
    """Log wandb conditional reconstruction grid for monitoring.

    Args:
        logger (Any): Logger instance used for experiment tracking.
        x (torch.Tensor): Tensor input for the computation.
        y (torch.Tensor | None): Tensor input for the computation.
        y_hat (torch.Tensor): Tensor input for the computation.
        y_target (torch.Tensor): Tensor input for the computation.
        valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        eo (torch.Tensor | None): Tensor input for the computation.
        prefix (str): Input value.
        image_key (str): Input value.
        cmap (str): Input value.
        show_valid_mask_panel (bool): Controls whether valid mask is shown as a panel.

    Returns:
        None: No value is returned.
    """
    if logger is None or not hasattr(logger, "experiment"):
        return
    experiment = logger.experiment
    if not hasattr(experiment, "log"):
        return

    try:
        import wandb
    except Exception:
        return

    num_to_plot = min(5, int(x.size(0)))
    if num_to_plot <= 0:
        return
    # Show one representative band per sample by default. Plotting all bands can
    # look like repeated images when channels are highly correlated.
    channels_to_plot = 1

    fig = None
    try:
        total_rows = num_to_plot * channels_to_plot
        show_valid_panel = bool(show_valid_mask_panel and valid_mask is not None)
        show_target_panel = True
        if y is not None and y_target.shape == y.shape:
            # Avoid duplicating the GLORYS panel in the common production validation path.
            show_target_panel = not torch.equal(y_target.detach(), y.detach())
        ncols = 2
        if y is not None:
            ncols += 1
        if eo is not None:
            ncols += 1
        if show_target_panel:
            ncols += 1
        if show_valid_panel:
            ncols += 1
        if land_mask is not None:
            ncols += 1
        fig, axes = plt.subplots(
            total_rows, ncols, figsize=(4 * ncols, 2.8 * total_rows), squeeze=False
        )

        for i in range(num_to_plot):
            valid_mask_i = _mask_for_sample(valid_mask, i)
            land_mask_i = _mask_for_sample(land_mask, i)
            for band_idx in range(channels_to_plot):
                row_idx = (i * channels_to_plot) + band_idx
                valid_band = valid_mask_i
                if valid_band is not None and valid_band.ndim == 3:
                    valid_band = valid_band[min(band_idx, int(valid_band.size(0)) - 1)]
                land_band = land_mask_i
                if land_band is not None and land_band.ndim == 3:
                    land_band = land_band[min(band_idx, int(land_band.size(0)) - 1)]

                x_img = _plot_band_image(x, i, band_idx=band_idx, mask=land_band)
                y_hat_img = _plot_band_image(
                    y_hat, i, band_idx=band_idx, mask=land_band
                )
                y_target_img = _plot_band_image(
                    y_target, i, band_idx=band_idx, mask=land_band
                )
                if valid_band is not None:
                    # Keep full-panel x visualization sparse by zeroing invalid pixels at render time.
                    valid_np = valid_band.detach().cpu().numpy() > 0.5
                    x_img[~valid_np] = 0.0
                if y is not None:
                    y_img = _plot_band_image(y, i, band_idx=band_idx, mask=land_band)
                else:
                    y_img = None
                if land_band is not None:
                    # Zero land pixels right before rendering full reconstruction panels.
                    ocean_np = land_band.detach().cpu().numpy() > 0.5
                    x_img[~ocean_np] = 0.0
                    if y_img is not None:
                        y_img[~ocean_np] = 0.0
                    y_hat_img[~ocean_np] = 0.0
                    y_target_img[~ocean_np] = 0.0

                col = 0
                axes[row_idx, col].imshow(x_img, cmap=cmap, vmin=0.0, vmax=1.0)
                axes[row_idx, col].set_axis_off()
                if row_idx == 0:
                    axes[row_idx, col].set_title("Input")
                col += 1

                if y_img is not None:
                    axes[row_idx, col].imshow(y_img, cmap=cmap, vmin=0.0, vmax=1.0)
                    axes[row_idx, col].set_axis_off()
                    if row_idx == 0:
                        axes[row_idx, col].set_title("GLORYS")
                    col += 1

                if eo is not None:
                    eo_img = _plot_band_image(eo, i, band_idx=band_idx, mask=land_band)
                    if land_band is not None:
                        ocean_np = land_band.detach().cpu().numpy() > 0.5
                        eo_img[~ocean_np] = 0.0
                    axes[row_idx, col].imshow(eo_img, cmap=cmap, vmin=0.0, vmax=1.0)
                    axes[row_idx, col].set_axis_off()
                    if row_idx == 0:
                        axes[row_idx, col].set_title("EO condition")
                    col += 1

                axes[row_idx, col].imshow(y_hat_img, cmap=cmap, vmin=0.0, vmax=1.0)
                axes[row_idx, col].set_axis_off()
                if row_idx == 0:
                    axes[row_idx, col].set_title("Reconstruction")
                col += 1

                if show_target_panel:
                    axes[row_idx, col].imshow(
                        y_target_img, cmap=cmap, vmin=0.0, vmax=1.0
                    )
                    axes[row_idx, col].set_axis_off()
                    if row_idx == 0:
                        axes[row_idx, col].set_title("Target")
                    col += 1

                if show_valid_panel:
                    if valid_band is not None:
                        axes[row_idx, col].imshow(
                            valid_band.detach().float().cpu().numpy(),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0,
                        )
                        axes[row_idx, col].set_axis_off()
                        if row_idx == 0:
                            axes[row_idx, col].set_title("Valid mask")
                    col += 1

                if land_mask is not None and land_band is not None:
                    axes[row_idx, col].imshow(
                        land_band.detach().float().cpu().numpy(),
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axes[row_idx, col].set_axis_off()
                    if row_idx == 0:
                        axes[row_idx, col].set_title("Land mask")

                axes[row_idx, 0].set_ylabel(f"s{i} b{band_idx}", rotation=90)

        fig.tight_layout()
        experiment.log({f"{prefix}/{image_key}": wandb.Image(fig)})
    finally:
        if fig is not None:
            plt.close(fig)

    # Log denormalized reconstruction L1 (in degrees) over generated pixels only.
    try:
        y_hat_t = y_hat.detach().float()
        y_target_t = y_target.detach().float()
        if y_hat_t.ndim == 3:
            y_hat_t = y_hat_t.unsqueeze(1)
        if y_target_t.ndim == 3:
            y_target_t = y_target_t.unsqueeze(1)
        if y_hat_t.ndim != 4 or y_target_t.ndim != 4:
            return
        if y_hat_t.shape != y_target_t.shape:
            return

        if valid_mask is None:
            return
        generated_mask = (valid_mask.detach().float() <= 0.5).to(device=y_hat_t.device)
        if generated_mask.ndim == 3:
            generated_mask = generated_mask.unsqueeze(1)
        if generated_mask.ndim != 4:
            return
        if (
            generated_mask.shape[0] != y_hat_t.shape[0]
            or generated_mask.shape[2:] != y_hat_t.shape[2:]
        ):
            return
        if generated_mask.size(1) == 1 and y_hat_t.size(1) > 1:
            generated_mask = generated_mask.expand(-1, y_hat_t.size(1), -1, -1)
        elif generated_mask.size(1) != y_hat_t.size(1):
            return

        if land_mask is not None:
            ocean_mask = (land_mask.detach().float() > 0.5).to(device=y_hat_t.device)
            if ocean_mask.ndim == 3:
                ocean_mask = ocean_mask.unsqueeze(1)
            if ocean_mask.ndim != 4:
                return
            if (
                ocean_mask.shape[0] != y_hat_t.shape[0]
                or ocean_mask.shape[2:] != y_hat_t.shape[2:]
            ):
                return
            if ocean_mask.size(1) == 1 and y_hat_t.size(1) > 1:
                ocean_mask = ocean_mask.expand(-1, y_hat_t.size(1), -1, -1)
            elif ocean_mask.size(1) != y_hat_t.size(1):
                return
            generated_mask = generated_mask * ocean_mask

        abs_diff = torch.abs(y_hat_t - y_target_t)
        numer_per_band = (abs_diff * generated_mask).sum(dim=(0, 2, 3))
        denom_per_band = generated_mask.sum(dim=(0, 2, 3))
        valid_bands = denom_per_band > 0
        if not bool(torch.any(valid_bands)):
            return

        l1_per_band = torch.zeros_like(numer_per_band)
        l1_per_band[valid_bands] = (
            numer_per_band[valid_bands] / denom_per_band[valid_bands]
        )

        # Keep per-band error metrics out of the image namespace in W&B.
        metric_prefix = "val_absolute_band_error"
        l1_logs: dict[str, float] = {}
        band_x: list[int] = []
        band_y: list[float] = []
        for band_idx in range(int(l1_per_band.numel())):
            if not bool(valid_bands[band_idx].item()):
                continue
            band_val = float(l1_per_band[band_idx].item())
            l1_logs[f"{metric_prefix}/recon_l1_generated_deg_band_{int(band_idx)}"] = (
                band_val
            )
            band_x.append(int(band_idx))
            band_y.append(band_val)

        if not l1_logs:
            return
        # One scalar per depth level for standard W&B metric tracking.
        experiment.log(l1_logs)
        # Optional compact view: all bands in a single plot panel for this validation pass.
        experiment.log(
            {
                f"{metric_prefix}/recon_l1_generated_deg_by_band": wandb.plot.line_series(
                    xs=band_x,
                    ys=[band_y],
                    keys=["L1 (deg)"],
                    title="Generated-Pixel L1 by Band",
                    xname="Band index",
                )
            }
        )
    except Exception:
        # Auxiliary scalar logging must never block validation image logging.
        pass


def log_wandb_depth_level_reconstruction_grid(
    *,
    logger: Any,
    y_hat: torch.Tensor,
    y_target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eo: torch.Tensor | None = None,
    land_mask: torch.Tensor | None = None,
    prefix: str = "val_imgs",
    image_key: str = "depth_level_reconstruction_grid",
    band_indices: tuple[int, ...] = (0, 1, 3),
    sample_idx: int = 0,
    cmap: str = "turbo",
) -> None:
    """Log wandb depth-level reconstruction grid for monitoring.

    Args:
        logger (Any): Logger instance used for experiment tracking.
        y_hat (torch.Tensor): Tensor input for the computation.
        y_target (torch.Tensor): Tensor input for the computation.
        valid_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        eo (torch.Tensor | None): Tensor input for the computation.
        land_mask (torch.Tensor | None): Mask tensor controlling valid or known pixels.
        prefix (str): Input value.
        image_key (str): Input value.
        band_indices (tuple[int, ...]): Input value.
        sample_idx (int): Input value.
        cmap (str): Input value.

    Returns:
        None: No value is returned.
    """
    if logger is None or not hasattr(logger, "experiment"):
        return
    experiment = logger.experiment
    if not hasattr(experiment, "log"):
        return

    try:
        import wandb
    except Exception:
        return

    if int(y_hat.size(0)) <= 0 or int(y_target.size(0)) <= 0:
        return
    if int(y_hat.size(0)) != int(y_target.size(0)):
        return
    if not band_indices:
        return

    sample_i = int(max(0, min(int(sample_idx), int(y_hat.size(0)) - 1)))
    available_bands = int(y_hat.size(1)) if y_hat.ndim == 4 else 1
    max_band_idx = max(0, available_bands - 1)

    fig = None
    try:
        fig, axes = plt.subplots(
            len(band_indices),
            4,
            figsize=(16, 2.8 * len(band_indices)),
            squeeze=False,
        )
        valid_mask_i = _mask_for_sample(valid_mask, sample_i)
        land_mask_i = _mask_for_sample(land_mask, sample_i)

        for row_idx, requested_band_idx in enumerate(band_indices):
            # Clamp requested indices so this view still renders for 1/3/4-band setups.
            band_idx = int(max(0, min(int(requested_band_idx), max_band_idx)))
            valid_band = valid_mask_i
            if valid_band is not None and valid_band.ndim == 3:
                valid_band = valid_band[min(band_idx, int(valid_band.size(0)) - 1)]
            land_band = land_mask_i
            if land_band is not None and land_band.ndim == 3:
                land_band = land_band[min(band_idx, int(land_band.size(0)) - 1)]

            recon_img = _plot_band_image(
                y_hat, sample_i, band_idx=band_idx, mask=land_band
            )
            target_img = _plot_band_image(
                y_target, sample_i, band_idx=band_idx, mask=land_band
            )
            if eo is not None:
                eo_img = _plot_band_image(
                    eo, sample_i, band_idx=band_idx, mask=land_band
                )
            else:
                eo_img = np.zeros_like(recon_img, dtype=np.float32)
            if valid_band is not None:
                valid_img = valid_band.detach().float().cpu().numpy()
            else:
                valid_img = np.zeros_like(recon_img, dtype=np.float32)

            if land_band is not None:
                ocean_np = land_band.detach().cpu().numpy() > 0.5
                eo_img[~ocean_np] = 0.0
                recon_img[~ocean_np] = 0.0
                target_img[~ocean_np] = 0.0

            axes[row_idx, 0].imshow(valid_img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, 0].set_axis_off()
            axes[row_idx, 1].imshow(eo_img, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row_idx, 1].set_axis_off()
            axes[row_idx, 2].imshow(recon_img, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row_idx, 2].set_axis_off()
            axes[row_idx, 3].imshow(target_img, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row_idx, 3].set_axis_off()

            if row_idx == 0:
                axes[row_idx, 0].set_title("Valid mask")
                axes[row_idx, 1].set_title("EO condition")
                axes[row_idx, 2].set_title("Reconstruction")
                axes[row_idx, 3].set_title("Ground truth")

            if int(requested_band_idx) == band_idx:
                axes[row_idx, 0].set_ylabel(f"band {band_idx}", rotation=90)
            else:
                axes[row_idx, 0].set_ylabel(
                    f"band {int(requested_band_idx)} -> {band_idx}",
                    rotation=90,
                )

        fig.tight_layout()
        experiment.log({f"{prefix}/{image_key}": wandb.Image(fig)})
    finally:
        if fig is not None:
            plt.close(fig)


def _resolve_profile_depth_axis(
    *,
    profile_size: int,
    depth_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return the plotting depth axis for one vertical profile."""
    if depth_axis is None:
        return np.arange(int(profile_size), dtype=np.int32), "GLORYS depth band"

    depth_axis_np = np.asarray(depth_axis, dtype=np.float64).reshape(-1)
    if int(depth_axis_np.size) != int(profile_size):
        raise ValueError(
            "depth_axis must match the profile depth dimension: "
            f"{int(depth_axis_np.size)} != {int(profile_size)}"
        )
    return depth_axis_np, "Depth (m)"


def plot_glorys_profile_comparison_axis(
    ax: Any,
    *,
    x_profile: np.ndarray,
    y_hat_profile: np.ndarray,
    y_target_profile: np.ndarray,
    observed_profile: np.ndarray,
    depth_axis: np.ndarray | None = None,
    title: str | None = None,
    show_legend: bool = False,
) -> None:
    """Draw one validation-style profile comparison axis."""
    x_profile_np = np.asarray(x_profile, dtype=np.float64).reshape(-1)
    y_hat_profile_np = np.asarray(y_hat_profile, dtype=np.float64).reshape(-1)
    y_target_profile_np = np.asarray(y_target_profile, dtype=np.float64).reshape(-1)
    observed_profile_np = np.asarray(observed_profile, dtype=bool).reshape(-1)
    if (
        x_profile_np.size != y_hat_profile_np.size
        or x_profile_np.size != y_target_profile_np.size
        or x_profile_np.size != observed_profile_np.size
    ):
        raise ValueError("All profile inputs must share the same depth dimension.")

    depth_values, depth_label = _resolve_profile_depth_axis(
        profile_size=int(y_target_profile_np.size),
        depth_axis=depth_axis,
    )
    ax.plot(
        y_target_profile_np,
        depth_values,
        label="GLORYS target",
        color="black",
        linewidth=2.0,
    )
    ax.plot(
        y_hat_profile_np,
        depth_values,
        label="Reconstruction",
        color="tab:orange",
        linewidth=1.8,
    )
    if bool(np.any(observed_profile_np)):
        # Keep the sparse conditioning profile visually identical to validation logging.
        ax.plot(
            x_profile_np[observed_profile_np],
            depth_values[observed_profile_np],
            label="Argo conditioning",
            color="tab:blue",
            marker="o",
            linewidth=1.4,
            markersize=3.5,
        )
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel(depth_label)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best")


def save_glorys_profile_comparison_plot(
    *,
    output_path: str | Path,
    x_profile: np.ndarray,
    y_hat_profile: np.ndarray,
    y_target_profile: np.ndarray,
    observed_profile: np.ndarray,
    depth_axis: np.ndarray | None = None,
    title: str | None = None,
    figure_title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Save one validation-style profile comparison plot to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.0), squeeze=False)
        plot_glorys_profile_comparison_axis(
            ax[0, 0],
            x_profile=x_profile,
            y_hat_profile=y_hat_profile,
            y_target_profile=y_target_profile,
            observed_profile=observed_profile,
            depth_axis=depth_axis,
            title=title,
            show_legend=True,
        )
        if figure_title is not None:
            fig.suptitle(figure_title, fontsize=13)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        else:
            fig.tight_layout()
        fig.savefig(output_path, dpi=int(dpi))
    finally:
        if fig is not None:
            plt.close(fig)
    return output_path


def log_wandb_glorys_profile_comparison(
    *,
    logger: Any,
    x: torch.Tensor,
    y_hat: torch.Tensor,
    y_target: torch.Tensor,
    conditioning_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    prefix: str = "val_imgs",
    image_key: str = "glorys_profile_comparison",
    sample_idx: int = 0,
) -> None:
    """Log full-depth profile comparisons at generated-only validation pixels.

    Args:
        logger (Any): Logger instance used for experiment tracking.
        x (torch.Tensor): Conditioning tensor containing sparse Argo-aligned profiles.
        y_hat (torch.Tensor): Reconstructed tensor in denormalized space.
        y_target (torch.Tensor): GLORYS target tensor in denormalized space.
        conditioning_mask (torch.Tensor | None): Mask tensor marking known x pixels.
        candidate_mask (torch.Tensor | None): Mask tensor selecting generated-only pixels.
        prefix (str): Input value.
        image_key (str): Input value.
        sample_idx (int): Zero-based index for selecting a sample or batch.

    Returns:
        None: No value is returned.
    """
    if logger is None or not hasattr(logger, "experiment"):
        return
    experiment = logger.experiment
    if not hasattr(experiment, "log"):
        return

    try:
        import wandb
    except Exception:
        return

    if x.ndim != 4 or y_hat.ndim != 4 or y_target.ndim != 4:
        return
    if int(x.size(0)) <= 0 or int(x.size(1)) <= 0:
        return
    if x.shape != y_hat.shape or x.shape != y_target.shape:
        return

    sample_i = int(max(0, min(int(sample_idx), int(x.size(0)) - 1)))
    candidate_mask_i = _mask_for_sample(candidate_mask, sample_i)
    conditioning_mask_i = _mask_for_sample(conditioning_mask, sample_i)
    if candidate_mask_i is None or conditioning_mask_i is None:
        return
    if candidate_mask_i.ndim == 3:
        candidate_map = candidate_mask_i.detach().bool().any(dim=0)
    elif candidate_mask_i.ndim == 2:
        candidate_map = candidate_mask_i.detach().bool()
    else:
        return

    candidate_coords = torch.nonzero(candidate_map, as_tuple=False)
    # Skip the plot when no generated-only pixels exist; falling back to known pixels
    # would make the diagnostic contradict the reconstruction task being visualized.
    if int(candidate_coords.size(0)) <= 0:
        return

    num_profiles = min(9, int(candidate_coords.size(0)))
    # Randomly subsample generated-only locations so the figure covers different profile shapes.
    chosen = candidate_coords[
        torch.randperm(int(candidate_coords.size(0)), device=candidate_coords.device)[
            :num_profiles
        ]
    ]

    depth_idx = np.arange(int(y_target.size(1)), dtype=np.int32)
    fig = None
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.0), squeeze=False)
        axes_flat = axes.reshape(-1)
        for plot_idx, ax in enumerate(axes_flat):
            if plot_idx >= num_profiles:
                ax.set_axis_off()
                continue

            row_i = int(chosen[plot_idx, 0].item())
            col_i = int(chosen[plot_idx, 1].item())
            x_profile = x[sample_i, :, row_i, col_i].detach().float().cpu().numpy()
            y_hat_profile = (
                y_hat[sample_i, :, row_i, col_i].detach().float().cpu().numpy()
            )
            y_target_profile = (
                y_target[sample_i, :, row_i, col_i].detach().float().cpu().numpy()
            )
            observed_profile = (
                conditioning_mask_i[:, row_i, col_i].detach().bool().cpu().numpy()
            )
            plot_glorys_profile_comparison_axis(
                ax,
                x_profile=x_profile,
                y_hat_profile=y_hat_profile,
                y_target_profile=y_target_profile,
                observed_profile=observed_profile,
                depth_axis=depth_idx,
                title=f"Pixel ({row_i}, {col_i})",
                show_legend=(plot_idx == 0),
            )

        fig.suptitle(
            f"Sample {sample_i} generated-only profile comparisons",
            fontsize=14,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
        experiment.log({f"{prefix}/{image_key}": wandb.Image(fig)})
    finally:
        if fig is not None:
            plt.close(fig)
