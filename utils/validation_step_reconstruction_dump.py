from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.normalizations import temperature_normalize
from utils.stretching import minmax_stretch


def _sample_band_2d(
    tensor: torch.Tensor | None,
    sample_idx: int,
    *,
    band_idx: int = 0,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 4:
        channel_idx = int(max(0, min(int(band_idx), int(tensor.size(1)) - 1)))
        return tensor[sample_idx, channel_idx].detach().float()
    if tensor.ndim == 3:
        return tensor[sample_idx].detach().float()
    if tensor.ndim == 2:
        return tensor.detach().float()
    return None


def _to_plot_np(
    tensor_2d: torch.Tensor | None,
    *,
    mask_2d: torch.Tensor | None = None,
    denorm_temperature: bool = False,
) -> np.ndarray | None:
    if tensor_2d is None:
        return None
    image_t = tensor_2d
    if denorm_temperature:
        image_t = temperature_normalize(mode="denorm", tensor=image_t)
    image_t = torch.nan_to_num(image_t, nan=0.0, posinf=0.0, neginf=0.0)
    stretched = minmax_stretch(image_t, mask=mask_2d, nodata_value=None)
    return stretched.cpu().numpy().astype(np.float32)


def _save_panel(
    output_path: Path,
    panels: list[tuple[str, np.ndarray | None, str]],
) -> None:
    valid_panels = [item for item in panels if item[1] is not None]
    if not valid_panels:
        return
    fig, axes = plt.subplots(1, len(valid_panels), figsize=(4 * len(valid_panels), 4))
    if len(valid_panels) == 1:
        axes = [axes]
    for ax, (title, image_np, cmap_name) in zip(axes, valid_panels):
        ax.imshow(image_np, cmap=cmap_name, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def dump_validation_reconstruction_sequence(
    *,
    output_dir: str,
    epoch: int,
    global_step: int,
    batch_idx: int,
    x_context: torch.Tensor,
    denoise_samples: list[tuple[int, torch.Tensor]],
    y_target: torch.Tensor | None = None,
    eo: torch.Tensor | None = None,
    valid_mask_for_loss: torch.Tensor | None = None,
    valid_mask_context: torch.Tensor | None = None,
    land_mask: torch.Tensor | None = None,
    max_samples: int = 1,
) -> None:
    """Persist per-step validation reconstructions and supervision masks to disk.

    Args:
        output_dir (str): Directory where image dumps are written.
        epoch (int): Current trainer epoch.
        global_step (int): Current global optimizer step.
        batch_idx (int): Validation batch index.
        x_context (torch.Tensor): Input x tensor after objective masking.
        denoise_samples (list[tuple[int, torch.Tensor]]): Intermediate and final reconstructions.
        y_target (torch.Tensor | None): Optional target y tensor for side-by-side reference.
        eo (torch.Tensor | None): Optional EO condition tensor.
        valid_mask_for_loss (torch.Tensor | None): Exact mask used for masked-loss supervision.
        valid_mask_context (torch.Tensor | None): Context valid mask provided to model condition.
        land_mask (torch.Tensor | None): Optional land/ocean mask for rendering.
        max_samples (int): Maximum number of batch samples to dump.

    Returns:
        None: No value is returned.
    """
    if not denoise_samples:
        return
    out_root = Path(output_dir)
    # Keep dumps grouped per validation call for quick chronological inspection.
    call_dir = out_root / (
        f"epoch_{int(epoch):04d}_step_{int(global_step):08d}_batch_{int(batch_idx):04d}"
    )
    call_dir.mkdir(parents=True, exist_ok=True)

    sorted_samples = sorted(denoise_samples, key=lambda item: int(item[0]))
    batch_size = int(sorted_samples[0][1].size(0))
    num_samples = int(max(1, min(int(max_samples), batch_size)))

    for sample_idx in range(num_samples):
        sample_dir = call_dir / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        land_mask_2d = _sample_band_2d(land_mask, sample_idx)
        context_mask_2d = _sample_band_2d(valid_mask_context, sample_idx)
        loss_mask_2d = _sample_band_2d(valid_mask_for_loss, sample_idx)

        x_np = _to_plot_np(
            _sample_band_2d(x_context, sample_idx),
            mask_2d=land_mask_2d,
            denorm_temperature=True,
        )
        eo_np = _to_plot_np(
            _sample_band_2d(eo, sample_idx),
            mask_2d=land_mask_2d,
            denorm_temperature=True,
        )
        y_np = _to_plot_np(
            _sample_band_2d(y_target, sample_idx),
            mask_2d=land_mask_2d,
            denorm_temperature=True,
        )
        context_mask_np = _to_plot_np(context_mask_2d, denorm_temperature=False)
        loss_mask_np = _to_plot_np(loss_mask_2d, denorm_temperature=False)

        _save_panel(
            sample_dir / "context_overview.png",
            [
                ("x_context", x_np, "turbo"),
                ("eo", eo_np, "turbo"),
                ("y_target", y_np, "turbo"),
                ("valid_mask_context", context_mask_np, "gray"),
                ("loss_mask", loss_mask_np, "gray"),
            ],
        )

        for step_idx, sample_t in sorted_samples:
            # The model outputs are in standardized temperature space; denorm for interpretability.
            recon_np = _to_plot_np(
                _sample_band_2d(sample_t, sample_idx),
                mask_2d=land_mask_2d,
                denorm_temperature=True,
            )
            _save_panel(
                sample_dir / f"recon_step_{int(step_idx):04d}.png",
                [
                    (f"recon s={int(step_idx)}", recon_np, "turbo"),
                    ("y_target", y_np, "turbo"),
                    ("loss_mask", loss_mask_np, "gray"),
                ],
            )
