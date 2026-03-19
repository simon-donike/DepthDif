from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize

VERTICAL_ASPECT_STRETCH = 1.4


def _resolve_depth_axis_m(
    profile_tensor: torch.Tensor,
    *,
    depth_axis_m: np.ndarray | torch.Tensor | None,
    max_depth_levels: int,
) -> tuple[np.ndarray, str]:
    depth_levels = int(profile_tensor.shape[0])
    if depth_axis_m is None:
        return np.arange(depth_levels, dtype=np.float64), "Depth Level"

    depth_values = np.asarray(depth_axis_m, dtype=np.float64).reshape(-1)
    if depth_values.size != depth_levels:
        raise ValueError(
            "depth_axis_m must match the profile depth dimension: "
            f"{depth_values.size} != {depth_levels}"
        )
    if not np.all(np.isfinite(depth_values)):
        raise ValueError("depth_axis_m must contain only finite values.")
    _ = max_depth_levels
    return depth_values.astype(np.float64, copy=False), "GLORYS Depth (m)"


def _prepare_argo_profile_scatter(
    profile_tensor: torch.Tensor,
    *,
    depth_axis_m: np.ndarray | torch.Tensor | None,
    max_depth_levels: int,
) -> tuple[np.ndarray, np.ndarray, Normalize, np.ndarray, int, int, np.ndarray, str]:
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
    depth_values_m, depth_label = _resolve_depth_axis_m(
        profile_tensor,
        depth_axis_m=depth_axis_m,
        max_depth_levels=max_depth_levels,
    )
    height = int(voxel_values.shape[1])
    width = int(voxel_values.shape[2])
    return filled_coords, valid_values, norm, voxel_values, height, width, depth_values_m, depth_label


def save_argo_profile_3d_plot(
    profile_tensor: torch.Tensor,
    output_path: str | Path,
    *,
    elev: float = 20.0,
    azim: float = -65.0,
    depth_axis_m: np.ndarray | torch.Tensor | None = None,
    max_depth_levels: int = 400,
    dpi: int = 200,
) -> Path:
    """Render and save a 3D voxel view of a rasterized Argo profile tensor.

    Args:
        profile_tensor (torch.Tensor): Input tensor with shape (depth_levels, H, W).
        output_path (str | Path): Destination path for the saved figure.
        elev (float): Matplotlib 3D elevation angle in degrees.
        azim (float): Matplotlib 3D azimuth angle in degrees.
        depth_axis_m (np.ndarray | torch.Tensor | None): Optional physical depth coordinates
            in meters for the profile depth axis. When provided, the plot keeps the vertical
            axis on the 50 depth levels and labels those z ticks with the GLORYS depth values.
        max_depth_levels (int): Fixed depth-axis limit used for visualization.
        dpi (int): Saved figure resolution.

    Returns:
        Path: Resolved output path for the saved figure.
    """
    filled_coords, valid_values, norm, voxel_values, height, width, depth_values_m, depth_label = (
        _prepare_argo_profile_scatter(
            profile_tensor,
            depth_axis_m=depth_axis_m,
            max_depth_levels=max_depth_levels,
        )
    )
    cmap = cm.get_cmap("viridis")
    point_colors = cmap(norm(valid_values))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_levels = int(voxel_values.shape[0])
    max_depth_levels = int(max(1, max_depth_levels))
    fig = plt.figure(figsize=(10, 8), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    point_x = filled_coords[:, 0] + 0.5
    point_y = filled_coords[:, 1] + 0.5
    point_z = filled_coords[:, 2] + 0.5
    # Plot only observed cells. This stays fast for sparse Argo tensors where a full
    # voxel-grid render would waste work on millions of empty locations.
    ax.scatter(
        point_x,
        point_y,
        point_z,
        c=point_colors,
        marker="o",
        s=10.0,
        depthshade=False,
    )

    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel(depth_label)
    ax.set_xlim(0, height)
    ax.set_ylim(0, width)
    if depth_axis_m is None:
        ax.set_zlim(float(max_depth_levels), 0.0)
    else:
        ax.set_zlim(float(depth_levels), 0.0)
        ax.set_zticks((np.arange(depth_levels, dtype=np.float64) + 0.5).tolist())
        ax.set_zticklabels([f"{depth_value:.3f}" for depth_value in depth_values_m.tolist()])
        ax.tick_params(axis="z", labelsize=6, pad=1)
    ax.view_init(elev=float(elev), azim=float(azim))
    depth_display_span = int(depth_levels if depth_axis_m is not None else max_depth_levels)
    ax.set_box_aspect(
        (
            max(height, 1),
            max(width, 1),
            max(VERTICAL_ASPECT_STRETCH * depth_display_span, 1.0),
        )
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(valid_values)
    fig.colorbar(mappable, ax=ax, shrink=0.78, pad=0.02, label="Value")

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.04, top=0.98)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path.resolve()


def save_argo_profile_3d_flyaround_gif(
    profile_tensor: torch.Tensor,
    output_path: str | Path,
    *,
    elev: float = 20.0,
    elev_wave_amplitude: float = 12.0,
    azim_start: float = -65.0,
    azim_end: float = 295.0,
    num_frames: int = 64,
    frame_duration_ms: int = 60,
    depth_axis_m: np.ndarray | torch.Tensor | None = None,
    max_depth_levels: int = 400,
    dpi: int = 160,
) -> Path:
    """Render and save a 3D flyaround GIF of a rasterized Argo profile tensor.

    Args:
        profile_tensor (torch.Tensor): Input tensor with shape (depth_levels, H, W).
        output_path (str | Path): Destination path for the saved GIF.
        elev (float): Matplotlib 3D elevation angle in degrees.
        elev_wave_amplitude (float): Base amplitude of the vertical camera wave in degrees.
        azim_start (float): Starting azimuth angle in degrees.
        azim_end (float): Ending azimuth angle in degrees.
        num_frames (int): Number of animation frames.
        frame_duration_ms (int): Per-frame GIF duration in milliseconds.
        depth_axis_m (np.ndarray | torch.Tensor | None): Optional physical depth coordinates
            in meters for the profile depth axis. When provided, the animation keeps the vertical
            axis on the 50 depth levels and labels those z ticks with the GLORYS depth values.
        max_depth_levels (int): Fixed depth-axis limit used for visualization.
        dpi (int): Saved animation resolution.

    Returns:
        Path: Resolved output path for the saved GIF.
    """
    filled_coords, valid_values, norm, voxel_values, height, width, depth_values_m, depth_label = (
        _prepare_argo_profile_scatter(
            profile_tensor,
            depth_axis_m=depth_axis_m,
            max_depth_levels=max_depth_levels,
        )
    )
    cmap = cm.get_cmap("viridis")
    point_colors = cmap(norm(valid_values))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_levels = int(voxel_values.shape[0])
    max_depth_levels = int(max(1, max_depth_levels))
    num_frames = int(max(2, num_frames))
    frame_duration_ms = int(max(20, frame_duration_ms))
    elev_wave_amplitude = float(max(0.0, elev_wave_amplitude))

    fig = plt.figure(figsize=(10, 8), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    point_x = filled_coords[:, 0] + 0.5
    point_y = filled_coords[:, 1] + 0.5
    point_z = filled_coords[:, 2] + 0.5
    ax.scatter(
        point_x,
        point_y,
        point_z,
        c=point_colors,
        marker="o",
        s=10.0,
        depthshade=False,
    )
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")
    ax.set_zlabel(depth_label)
    ax.set_xlim(0, height)
    ax.set_ylim(0, width)
    if depth_axis_m is None:
        ax.set_zlim(float(max_depth_levels), 0.0)
    else:
        ax.set_zlim(float(depth_levels), 0.0)
        ax.set_zticks((np.arange(depth_levels, dtype=np.float64) + 0.5).tolist())
        ax.set_zticklabels([f"{depth_value:.3f}" for depth_value in depth_values_m.tolist()])
        ax.tick_params(axis="z", labelsize=6, pad=1)
    depth_display_span = int(depth_levels if depth_axis_m is not None else max_depth_levels)
    ax.set_box_aspect(
        (
            max(height, 1),
            max(width, 1),
            max(VERTICAL_ASPECT_STRETCH * depth_display_span, 1.0),
        )
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(valid_values)
    fig.colorbar(mappable, ax=ax, shrink=0.78, pad=0.02, label="Value")

    azimuths = np.linspace(float(azim_start), float(azim_end), num=num_frames, endpoint=False)
    phase = np.linspace(0.0, 2.0 * np.pi, num=num_frames, endpoint=False)
    # Add a smooth asymmetric up/down motion so the flyaround reveals depth structure
    # better than a flat orbit and spends more time dipping below the base elevation
    # than rising above it.
    wave = np.sin(phase)
    up_wave = np.clip(wave, 0.0, None)
    down_wave = np.clip(wave, None, 0.0)
    elevations = (
        float(elev)
        + (0.75 * elev_wave_amplitude * up_wave)
        + (1.50 * elev_wave_amplitude * down_wave)
    )

    def _update(frame_idx: int) -> tuple[plt.Axes]:
        ax.view_init(
            elev=float(elevations[int(frame_idx)]),
            azim=float(azimuths[int(frame_idx)]),
        )
        return (ax,)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=num_frames,
        interval=frame_duration_ms,
        blit=False,
    )
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.04, top=0.98)
    anim.save(output_path, writer="pillow", dpi=int(dpi))
    plt.close(fig)
    return output_path.resolve()
