# Example: /work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_loss_explanations --export-dir inference/outputs/global_variables_2018_W25_v2/temperature --output-dir docs/assets/figures
"""Generate data-driven documentation figures for the auxiliary loss stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

BLUE = "#176B9C"
ORANGE = "#D95F02"
GREEN = "#1B9E77"
PURPLE = "#7570B3"
GRAY = "#5C6770"
PANEL = "#F4F7F9"


def _load_raster_crop(
    path: Path, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Load one geographic crop and convert nodata pixels to NaN."""
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        field = src.read(1, window=window, boundless=True).astype(np.float64)
        if src.nodata is not None:
            field[field == src.nodata] = np.nan
    return field


def _load_representative_profile(path: Path) -> dict[str, np.ndarray]:
    """Select a well-observed real ARGO profile from the exported profile collection."""
    collection = json.loads(path.read_text())
    candidates = []
    for feature in collection["features"]:
        properties = feature["properties"]
        observed = np.asarray(properties["argo_profile_c"], dtype=np.float64)
        candidates.append((np.isfinite(observed).sum(), properties))
    properties = max(candidates, key=lambda item: item[0])[1]
    return {
        "depth": np.asarray(properties["depth_m"], dtype=np.float64),
        "argo": np.asarray(properties["argo_profile_c"], dtype=np.float64),
        "prediction": np.asarray(properties["prediction_profile_c"], dtype=np.float64),
        "glorys": np.asarray(properties["glorys_profile_c"], dtype=np.float64),
    }


def _fill_for_filter(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill land locally before filtering while retaining the original ocean mask."""
    valid = np.isfinite(field)
    values = np.where(valid, field, 0.0)
    weights = gaussian_filter(valid.astype(np.float64), sigma=5.0)
    filled = gaussian_filter(values, sigma=5.0) / np.maximum(weights, 1.0e-8)
    return np.where(valid, field, filled), valid


def _degrade(field: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Apply a controlled Gaussian smoothing degradation over ocean support."""
    filled, valid = _fill_for_filter(field)
    degraded = gaussian_filter(filled, sigma=sigma)
    return np.where(valid, degraded, np.nan)


def _radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a radial mean power spectrum for one real-valued 2D field."""
    filled, valid = _fill_for_filter(field)
    centered = np.where(valid, filled - np.nanmean(field), 0.0)
    window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(xx - power.shape[1] // 2, yy - power.shape[0] // 2)
    bins = np.arange(1, int(radius.max()) + 1)
    radial = np.asarray(
        [power[(radius >= low) & (radius < low + 1)].mean() for low in bins]
    )
    return bins / max(field.shape), radial


def _structure_function(
    field: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the second-order structure function along both grid axes."""
    estimates = []
    for distance in distances:
        horizontal = (field[:, distance:] - field[:, :-distance]) ** 2
        vertical = (field[distance:, :] - field[:-distance, :]) ** 2
        estimates.append(
            np.nanmean(np.concatenate((horizontal.ravel(), vertical.ravel())))
        )
    return distances, np.asarray(estimates)


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared documentation plot styling."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.7)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a tightly cropped, publication-resolution PNG."""
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_overview(
    glorys: np.ndarray,
    prediction: np.ndarray,
    profile: dict[str, np.ndarray],
    output: Path,
) -> None:
    """Show the distinct discrepancy inspected by every active loss term."""
    degraded = _degrade(glorys)
    distances = np.unique(np.geomspace(1, min(glorys.shape) // 3, 18).astype(int))
    freq, power_true = _radial_spectrum(glorys)
    _, power_smooth = _radial_spectrum(degraded)
    _, s2_true = _structure_function(glorys, distances)
    _, s2_smooth = _structure_function(degraded, distances)
    vmin, vmax = np.nanpercentile(glorys, (2, 98))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("What each loss looks for", fontsize=20, fontweight="bold")
    axes[0, 0].imshow(glorys, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Base: denoising target")
    axes[0, 0].axis("off")
    error = np.abs(prediction - glorys)
    axes[0, 1].imshow(error, cmap="magma", vmin=0, vmax=np.nanpercentile(error, 98))
    axes[0, 1].set_title("Base: pixelwise discrepancy")
    axes[0, 1].axis("off")

    valid = np.isfinite(profile["argo"]) & (profile["depth"] <= 2000)
    axes[0, 2].plot(
        profile["prediction"], profile["depth"], color=BLUE, label="Prediction"
    )
    axes[0, 2].scatter(
        profile["argo"][valid],
        profile["depth"][valid],
        color=ORANGE,
        s=28,
        zorder=3,
        label="ARGO",
    )
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_title("Observation: match ARGO points")
    axes[0, 2].set_xlabel("Temperature (°C)")
    axes[0, 2].set_ylabel("Depth (m)")
    axes[0, 2].legend(frameon=False)
    _style_axis(axes[0, 2])

    observed_indices = np.flatnonzero(valid)
    increments_obs = np.diff(profile["argo"][observed_indices])
    increments_pred = np.diff(profile["prediction"][observed_indices])
    axes[1, 0].scatter(increments_obs, increments_pred, color=ORANGE, alpha=0.8)
    limits = np.nanpercentile(np.r_[increments_obs, increments_pred], (2, 98))
    axes[1, 0].plot(limits, limits, "--", color=GRAY)
    axes[1, 0].set(
        title="Increment: preserve vertical changes",
        xlabel="ARGO adjacent-depth change (°C)",
        ylabel="Predicted change (°C)",
    )
    _style_axis(axes[1, 0])

    axes[1, 1].loglog(distances, s2_true, color=GREEN, label="GLORYS")
    axes[1, 1].loglog(distances, s2_smooth, color=PURPLE, label="Smoothed")
    axes[1, 1].set(
        title="Structure: variance across distance",
        xlabel="Separation (grid cells)",
        ylabel=r"$S_2$ (°C²)",
    )
    axes[1, 1].legend(frameon=False)
    _style_axis(axes[1, 1])

    axes[1, 2].loglog(freq, power_true, color=GREEN, label="GLORYS")
    axes[1, 2].loglog(freq, power_smooth, color=PURPLE, label="Smoothed")
    axes[1, 2].fill_between(
        freq,
        power_smooth,
        power_true,
        where=power_true > power_smooth,
        color=ORANGE,
        alpha=0.25,
        label="Missing energy",
    )
    axes[1, 2].set(
        title="Spectral: retain fine-scale energy",
        xlabel="Spatial frequency (cycles/grid cell)",
        ylabel="Power",
    )
    axes[1, 2].legend(frameon=False)
    _style_axis(axes[1, 2])
    _save(fig, output)


def _plot_argo(profile: dict[str, np.ndarray], output: Path) -> None:
    """Contrast pointwise observation and adjacent-depth increment consistency."""
    mask = np.isfinite(profile["argo"]) & (profile["depth"] <= 2000)
    indices = np.flatnonzero(mask)
    depth = profile["depth"][indices]
    observed = profile["argo"][indices]
    predicted = profile["prediction"][indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    fig.suptitle(
        "The ARGO losses inspect different errors", fontsize=18, fontweight="bold"
    )
    axes[0].plot(
        profile["prediction"], profile["depth"], color=BLUE, label="Prediction"
    )
    axes[0].plot(
        profile["glorys"], profile["depth"], color=GREEN, alpha=0.7, label="GLORYS"
    )
    axes[0].scatter(
        observed, depth, color=ORANGE, s=35, zorder=3, label="ARGO observations"
    )
    for obs, pred, dep in zip(observed, predicted, depth):
        axes[0].plot([obs, pred], [dep, dep], color=ORANGE, alpha=0.45)
    axes[0].invert_yaxis()
    axes[0].set(
        title="Observation consistency: horizontal gaps",
        xlabel="Temperature (°C)",
        ylabel="Depth (m)",
    )
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    obs_increment = np.diff(observed)
    pred_increment = np.diff(predicted)
    mid_depth = (depth[:-1] + depth[1:]) / 2
    axes[1].plot(obs_increment, mid_depth, "o-", color=ORANGE, label="ARGO increment")
    axes[1].plot(
        pred_increment, mid_depth, "o-", color=BLUE, label="Predicted increment"
    )
    axes[1].fill_betweenx(
        mid_depth, obs_increment, pred_increment, color=ORANGE, alpha=0.2
    )
    axes[1].axvline(0, color=GRAY, linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set(
        title="Increment consistency: vertical-change gaps",
        xlabel="Adjacent-depth temperature change (°C)",
        ylabel="Midpoint depth (m)",
    )
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    _save(fig, output)


def _plot_structure_and_spectrum(field: np.ndarray, output: Path) -> None:
    """Demonstrate how smoothing changes multiscale structure and spectral energy."""
    smooth = _degrade(field)
    distances = np.unique(np.geomspace(1, min(field.shape) // 3, 22).astype(int))
    freq, original_power = _radial_spectrum(field)
    _, smooth_power = _radial_spectrum(smooth)
    _, original_s2 = _structure_function(field, distances)
    _, smooth_s2 = _structure_function(smooth, distances)
    vmin, vmax = np.nanpercentile(field, (2, 98))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(
        "Smoothing removes signals seen by two auxiliary losses",
        fontsize=18,
        fontweight="bold",
    )
    axes[0, 0].imshow(field, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Real GLORYS field")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(smooth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Controlled Gaussian smoothing")
    axes[0, 1].axis("off")
    axes[1, 0].loglog(distances, original_s2, color=GREEN, label="GLORYS")
    axes[1, 0].loglog(distances, smooth_s2, color=PURPLE, label="Smoothed")
    axes[1, 0].set(
        title="Structure-function loss sees reduced increments",
        xlabel="Separation (grid cells)",
        ylabel=r"$S_2$ (°C²)",
    )
    axes[1, 0].legend(frameon=False)
    _style_axis(axes[1, 0])
    axes[1, 1].loglog(freq, original_power, color=GREEN, label="GLORYS")
    axes[1, 1].loglog(freq, smooth_power, color=PURPLE, label="Smoothed")
    axes[1, 1].fill_between(
        freq,
        smooth_power,
        original_power,
        where=original_power > smooth_power,
        color=ORANGE,
        alpha=0.25,
        label="Energy-floor penalty region",
    )
    axes[1, 1].set(
        title="Spectral loss sees missing high-frequency energy",
        xlabel="Spatial frequency (cycles/grid cell)",
        ylabel="Power",
    )
    axes[1, 1].legend(frameon=False)
    _style_axis(axes[1, 1])
    _save(fig, output)


def _plot_timestep_weighting(output: Path) -> None:
    """Plot the configured linear and bounded SNR auxiliary timestep weights."""
    timestep = np.arange(1000)
    linear = 0.1 + (1.0 - timestep / 999.0) * 0.9
    beta = np.linspace(1.0e-4, 2.0e-2, timestep.size)
    alpha_bar = np.cumprod(1.0 - beta)
    snr = alpha_bar / np.maximum(1.0 - alpha_bar, 1.0e-12)
    snr_weight = np.clip(np.minimum(snr, 5.0) / 5.0, 0.1, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(
        timestep,
        np.ones_like(timestep),
        color=GRAY,
        linestyle="--",
        label="Base loss: unchanged",
    )
    ax.plot(timestep, linear, color=BLUE, linewidth=2.5, label="Auxiliary: linear")
    ax.plot(
        timestep,
        snr_weight,
        color=ORANGE,
        linewidth=2.5,
        label="Auxiliary: bounded SNR",
    )
    ax.fill_between(timestep, 0, snr_weight, color=ORANGE, alpha=0.12)
    ax.annotate("clean / high SNR", (15, 0.04), ha="left")
    ax.annotate("noisy / low SNR", (985, 0.04), ha="right")
    ax.set(
        title="Timestep weighting changes only the auxiliary contribution",
        xlabel="Diffusion timestep",
        ylabel="Loss multiplier",
        ylim=(0, 1.08),
    )
    ax.legend(frameon=False, ncol=3, loc="lower center")
    _style_axis(ax)
    _save(fig, output)


def _plot_target_modes(field: np.ndarray, output: Path) -> None:
    """Contrast archive-reference and paired-target statistical comparisons."""
    height, width = field.shape
    tiles = []
    for row in range(0, height - 64, 48):
        for col in range(0, width - 64, 48):
            tile = field[row : row + 64, col : col + 64]
            if np.isfinite(tile).mean() > 0.85:
                tiles.append(_radial_spectrum(tile)[1])
    common = min(map(len, tiles))
    archive = np.stack([tile[:common] for tile in tiles])
    frequency = _radial_spectrum(field[:64, :64])[0][:common]
    paired = archive[len(archive) // 2]
    prediction = gaussian_filter(_fill_for_filter(field[:64, :64])[0], sigma=2.5)
    prediction_power = _radial_spectrum(prediction)[1][:common]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        "The GLORYS modes differ in where the target statistic comes from",
        fontsize=17,
        fontweight="bold",
    )
    median = np.median(archive, axis=0)
    low, high = np.percentile(archive, (20, 80), axis=0)
    axes[0].fill_between(
        frequency, low, high, color=GREEN, alpha=0.2, label="Archive spread"
    )
    axes[0].loglog(frequency, median, color=GREEN, label="Precomputed reference")
    axes[0].loglog(frequency, prediction_power, color=BLUE, label="Prediction")
    axes[0].set_title("target: reference")
    axes[1].loglog(frequency, paired, color=GREEN, label="Same-sample GLORYS")
    axes[1].loglog(frequency, prediction_power, color=BLUE, label="Prediction")
    axes[1].set_title("target: paired_glorys")
    for axis in axes:
        axis.set(xlabel="Spatial frequency", ylabel="Power")
        axis.legend(frameon=False)
        _style_axis(axis)
    _save(fig, output)


def _plot_degradation_sensitivity(field: np.ndarray, output: Path) -> None:
    """Compare which statistical losses react as smoothing becomes stronger."""
    sigmas = np.asarray([0.0, 1.0, 2.0, 4.0, 7.0])
    distances = np.unique(np.geomspace(1, min(field.shape) // 3, 18).astype(int))
    _, reference_s2 = _structure_function(field, distances)
    _, reference_power = _radial_spectrum(field)
    s2_errors = []
    spectral_errors = []
    pixel_errors = []
    for sigma in sigmas:
        candidate = field if sigma == 0 else _degrade(field, sigma=sigma)
        _, candidate_s2 = _structure_function(candidate, distances)
        _, candidate_power = _radial_spectrum(candidate)
        pixel_errors.append(np.nanmean((candidate - field) ** 2))
        s2_errors.append(
            np.nanmean(np.abs(np.log(candidate_s2) - np.log(reference_s2)))
        )
        spectral_errors.append(
            np.nanmean(
                np.maximum(
                    np.log(reference_power + 1.0e-8) - np.log(candidate_power + 1.0e-8),
                    0,
                )
            )
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    metrics = (
        ("Pixel MSE", pixel_errors, BLUE),
        ("Structure-function discrepancy", s2_errors, GREEN),
        ("Missing spectral energy", spectral_errors, ORANGE),
    )
    for axis, (title, values, color) in zip(axes, metrics):
        axis.plot(sigmas, values, "o-", color=color, linewidth=2.5)
        axis.set(
            title=title, xlabel="Gaussian smoothing σ", ylabel="Measured discrepancy"
        )
        _style_axis(axis)
    fig.suptitle(
        "Controlled degradation reveals what each loss detects",
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, output)


def generate_figures(export_dir: Path, output_dir: Path) -> None:
    """Load real exports and generate every loss-explanation figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = (-75.0, 30.0, -45.0, 50.0)
    glorys = _load_raster_crop(export_dir / "temperature_glorys_10m.tif", bounds)
    prediction = _load_raster_crop(
        export_dir / "temperature_prediction_10m.tif", bounds
    )
    profile = _load_representative_profile(
        export_dir / "temperature_full_sample_locations.geojson"
    )
    _plot_overview(
        glorys, prediction, profile, output_dir / "loss-signals-overview.png"
    )
    _plot_argo(profile, output_dir / "loss-argo-observation-increment.png")
    _plot_structure_and_spectrum(
        glorys, output_dir / "loss-structure-spectral-smoothing.png"
    )
    _plot_timestep_weighting(output_dir / "loss-timestep-weighting.png")
    _plot_target_modes(glorys, output_dir / "loss-reference-vs-paired.png")
    _plot_degradation_sensitivity(
        glorys, output_dir / "loss-degradation-sensitivity.png"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate Matplotlib explanations of the auxiliary loss signals."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    """Run the documentation figure generator."""
    args = _build_parser().parse_args()
    generate_figures(args.export_dir, args.output_dir)


if __name__ == "__main__":
    main()
