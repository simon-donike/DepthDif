from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize


def save_argo_profile_3d_plot(
    profile_tensor: torch.Tensor,
    output_path: str | Path,
    *,
    elev: float = 20.0,
    azim: float = -65.0,
    max_depth_levels: int = 400,
    dpi: int = 200,
) -> Path:
    """Render and save a 3D voxel view of a rasterized Argo profile tensor.

    Args:
        profile_tensor (torch.Tensor): Input tensor with shape (depth_levels, H, W).
        output_path (str | Path): Destination path for the saved figure.
        elev (float): Matplotlib 3D elevation angle in degrees.
        azim (float): Matplotlib 3D azimuth angle in degrees.
        max_depth_levels (int): Fixed depth-axis limit used for visualization.
        dpi (int): Saved figure resolution.

    Returns:
        Path: Resolved output path for the saved figure.
    """
    if not isinstance(profile_tensor, torch.Tensor):
        raise TypeError("profile_tensor must be a torch.Tensor.")
    if profile_tensor.ndim != 3:
        raise ValueError(
            "profile_tensor must have shape (depth_levels, H, W), "
            f"got {tuple(profile_tensor.shape)}."
        )

    volume = profile_tensor.detach().float().cpu()
    volume = torch.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    voxel_values = volume.numpy()
    # Reorder from (depth, H, W) to (H, W, depth) so the vertical axis is depth.
    display_values = np.transpose(voxel_values, (1, 2, 0))
    filled = display_values != 0.0
    if not np.any(filled):
        raise ValueError("profile_tensor does not contain any nonzero voxels to plot.")

    filled_coords = np.argwhere(filled)
    valid_values = display_values[filled]
    norm = Normalize(vmin=float(valid_values.min()), vmax=float(valid_values.max()))
    cmap = cm.get_cmap("viridis")
    point_colors = cmap(norm(valid_values))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_levels, height, width = voxel_values.shape
    max_depth_levels = int(max(1, max_depth_levels))
    fig = plt.figure(figsize=(10, 8), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    # Plot only observed cells. This stays fast for sparse Argo tensors where a full
    # voxel-grid render would waste work on millions of empty locations.
    ax.scatter(
        filled_coords[:, 0] + 0.5,
        filled_coords[:, 1] + 0.5,
        filled_coords[:, 2] + 0.5,
        c=point_colors,
        marker="o",
        s=10.0,
        depthshade=False,
    )

    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth Level")
    ax.set_xlim(0, height)
    ax.set_ylim(0, width)
    # Keep a fixed depth view so Argo plots remain visually comparable across samples.
    ax.set_zlim(max_depth_levels, 0)
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_box_aspect((max(height, 1), max(width, 1), max(max_depth_levels, 1)))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(valid_values)
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Value")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def save_argo_profile_3d_flyaround_gif(
    profile_tensor: torch.Tensor,
    output_path: str | Path,
    *,
    elev: float = 20.0,
    azim_start: float = -65.0,
    azim_end: float = 295.0,
    num_frames: int = 48,
    frame_duration_ms: int = 80,
    max_depth_levels: int = 400,
    dpi: int = 160,
) -> Path:
    """Render and save a 3D flyaround GIF of a rasterized Argo profile tensor.

    Args:
        profile_tensor (torch.Tensor): Input tensor with shape (depth_levels, H, W).
        output_path (str | Path): Destination path for the saved GIF.
        elev (float): Matplotlib 3D elevation angle in degrees.
        azim_start (float): Starting azimuth angle in degrees.
        azim_end (float): Ending azimuth angle in degrees.
        num_frames (int): Number of animation frames.
        frame_duration_ms (int): Per-frame GIF duration in milliseconds.
        max_depth_levels (int): Fixed depth-axis limit used for visualization.
        dpi (int): Saved animation resolution.

    Returns:
        Path: Resolved output path for the saved GIF.
    """
    if not isinstance(profile_tensor, torch.Tensor):
        raise TypeError("profile_tensor must be a torch.Tensor.")
    if profile_tensor.ndim != 3:
        raise ValueError(
            "profile_tensor must have shape (depth_levels, H, W), "
            f"got {tuple(profile_tensor.shape)}."
        )

    volume = profile_tensor.detach().float().cpu()
    volume = torch.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    voxel_values = volume.numpy()
    display_values = np.transpose(voxel_values, (1, 2, 0))
    filled = display_values != 0.0
    if not np.any(filled):
        raise ValueError("profile_tensor does not contain any nonzero voxels to plot.")

    filled_coords = np.argwhere(filled)
    valid_values = display_values[filled]
    norm = Normalize(vmin=float(valid_values.min()), vmax=float(valid_values.max()))
    cmap = cm.get_cmap("viridis")
    point_colors = cmap(norm(valid_values))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_levels, height, width = voxel_values.shape
    max_depth_levels = int(max(1, max_depth_levels))
    num_frames = int(max(2, num_frames))
    frame_duration_ms = int(max(20, frame_duration_ms))

    fig = plt.figure(figsize=(10, 8), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        filled_coords[:, 0] + 0.5,
        filled_coords[:, 1] + 0.5,
        filled_coords[:, 2] + 0.5,
        c=point_colors,
        marker="o",
        s=10.0,
        depthshade=False,
    )
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel("Depth Level")
    ax.set_xlim(0, height)
    ax.set_ylim(0, width)
    ax.set_zlim(max_depth_levels, 0)
    ax.set_box_aspect((max(height, 1), max(width, 1), max(max_depth_levels, 1)))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(valid_values)
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Value")

    azimuths = np.linspace(float(azim_start), float(azim_end), num=num_frames, endpoint=False)

    def _update(frame_idx: int) -> tuple[plt.Axes]:
        ax.view_init(elev=float(elev), azim=float(azimuths[int(frame_idx)]))
        return (ax,)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=num_frames,
        interval=frame_duration_ms,
        blit=False,
    )
    fig.tight_layout()
    anim.save(output_path, writer="pillow", dpi=int(dpi))
    plt.close(fig)
    return output_path.resolve()
