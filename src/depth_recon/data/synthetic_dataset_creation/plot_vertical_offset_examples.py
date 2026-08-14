# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.plot_vertical_offset_examples \
#   --geotiff-root-dir /work/data/OceanVariableReconstruction \
#   --statistics-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
#   --output-dir outputs/vertical_offset_examples --metadata-cache-dir \
#   /work/data/OceanVariableReconstruction/depthdif_cache --validation-year 2018 \
#   --tile-size 128 --patch-stride 128 --max-land-fraction 0.30 \
#   --depths-m 0 50 100 250 500 1000 --candidate-count 24 --random-seed 7 \
#   --dpi 170
"""Plot deterministic surface-plus-depth-offset targets and held-out GLORYS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depth_recon.data.dataset_argo_geotiff_gridded import (
    ArgoGeoTIFFGriddedPatchDataset,
)
from depth_recon.data.synthetic_dataset_creation.vertical_offset_prior import (
    VerticalOffsetPrior,
)
from depth_recon.utils.normalizations import CELSIUS_TO_KELVIN_OFFSET

REGIONS = (
    ("gulf_stream", 38.0, -60.0),
    ("kuroshio", 35.0, 145.0),
    ("southern_ocean", -50.0, 20.0),
)


def _wrapped_lon_distance(values: np.ndarray, target: float) -> np.ndarray:
    """Return shortest signed longitude differences in degrees."""
    return (np.asarray(values) - float(target) + 180.0) % 360.0 - 180.0


def _select_textured_row(
    dataset: ArgoGeoTIFFGriddedPatchDataset,
    *,
    latitude: float,
    longitude: float,
    candidate_count: int,
) -> dict[str, Any]:
    """Choose the strongest-SST-contrast validation patch near a region."""
    rows = dataset._rows
    lon_delta = _wrapped_lon_distance(rows["lon_center"].to_numpy(), longitude)
    distance = (rows["lat_center"].to_numpy() - latitude) ** 2 + (
        lon_delta * np.cos(np.deg2rad(latitude))
    ) ** 2
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for index in np.argsort(distance)[: max(1, int(candidate_count))]:
        row = rows.iloc[int(index)].to_dict()
        sst = dataset._load_surface_fields(row)["sst"]
        score = float(np.nanstd(sst))
        if score > best_score:
            best, best_score = row, score
    if best is None:
        raise RuntimeError("No valid regional example patch was found.")
    return best


def _patch_coordinates(
    row: Mapping[str, Any], *, tile_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return patch pixel centers while preserving dateline wrapping."""
    fractions = (np.arange(int(tile_size), dtype=np.float32) + 0.5) / float(tile_size)
    latitude = float(row["lat0"]) + fractions * (
        float(row["lat1"]) - float(row["lat0"])
    )
    lon0 = float(row["lon0"])
    span = (float(row["lon1"]) - lon0) % 360.0
    if span == 0.0:
        span = float(row["lon1"]) - lon0
    longitude = ((lon0 + fractions * span + 180.0) % 360.0) - 180.0
    return np.meshgrid(latitude, longitude, indexing="ij")


def _profile_pixel(valid_mask: np.ndarray) -> tuple[int, int]:
    """Choose the valid profile nearest the center of a diagnostic patch."""
    spatial_valid = np.asarray(valid_mask, dtype=bool).any(axis=0)
    candidates = np.argwhere(spatial_valid)
    if candidates.size == 0:
        raise RuntimeError("Diagnostic patch contains no valid GLORYS profile.")
    center = (np.asarray(spatial_valid.shape, dtype=np.float64) - 1.0) / 2.0
    index = int(np.argmin(np.square(candidates - center).sum(axis=1)))
    return tuple(int(value) for value in candidates[index])


def _plot_profile_comparison(
    *,
    synthetic_temperature: np.ndarray,
    glorys_temperature: np.ndarray,
    synthetic_salinity: np.ndarray,
    glorys_salinity: np.ndarray,
    depths: np.ndarray,
    pixel: tuple[int, int],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Write validation-style synthetic-versus-GLORYS profile line charts."""
    row, column = pixel
    figure, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    for axis, synthetic, glorys, label in (
        (
            axes[0],
            synthetic_temperature[:, row, column],
            glorys_temperature[:, row, column],
            "Temperature (degrees C)",
        ),
        (
            axes[1],
            synthetic_salinity[:, row, column],
            glorys_salinity[:, row, column],
            "Salinity (PSU)",
        ),
    ):
        valid = np.isfinite(synthetic) & np.isfinite(glorys)
        axis.plot(
            synthetic[valid], depths[valid], label="Synthetic target", linewidth=2.0
        )
        axis.plot(
            glorys[valid], depths[valid], label="GLORYS diagnostic", linewidth=1.6
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Depth (m)")
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"{title} | profile pixel ({row}, {column})")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _depth_indices(axis: np.ndarray, requested: Sequence[float]) -> list[int]:
    """Map requested depths to unique nearest stored levels."""
    indices: list[int] = []
    for depth in requested:
        index = int(np.argmin(np.abs(np.asarray(axis) - float(depth))))
        if index not in indices:
            indices.append(index)
    return indices


def _plot_mosaic(
    *,
    synthetic: np.ndarray,
    glorys: np.ndarray,
    depths: np.ndarray,
    indices: Sequence[int],
    title: str,
    units: str,
    cmap: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot synthetic and held-out diagnostic fields with shared depth scales."""
    figure, axes = plt.subplots(
        2,
        len(indices),
        figsize=(3.0 * len(indices), 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    for column, depth_index in enumerate(indices):
        values = np.concatenate(
            (
                synthetic[depth_index][np.isfinite(synthetic[depth_index])],
                glorys[depth_index][np.isfinite(glorys[depth_index])],
            )
        )
        low, high = np.percentile(values, (2.0, 98.0))
        for row_index, field in enumerate((synthetic, glorys)):
            image = axes[row_index, column].imshow(
                field[depth_index],
                cmap=cmap,
                vmin=float(low),
                vmax=float(high),
                interpolation="nearest",
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            if row_index == 0:
                axes[row_index, column].set_title(f"{depths[depth_index]:g} m")
            if column == 0:
                axes[row_index, column].set_ylabel(
                    "EO + smooth GLORYS delta" if row_index == 0 else "held-out GLORYS"
                )
        figure.colorbar(
            image,
            ax=axes[:, column],
            location="bottom",
            shrink=0.82,
            pad=0.02,
            label=units,
        )
    figure.suptitle(f"{title}\nGLORYS row is diagnostic only")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def plot_examples(
    *,
    geotiff_root_dir: str | Path,
    statistics_path: str | Path,
    output_dir: str | Path,
    metadata_cache_dir: str | Path | None,
    validation_year: int,
    tile_size: int,
    patch_stride: int,
    max_land_fraction: float,
    depths_m: Sequence[float],
    candidate_count: int,
    random_seed: int,
    dpi: int,
) -> list[Path]:
    """Generate coefficient and regional depth-comparison PNGs."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = ArgoGeoTIFFGriddedPatchDataset(
        geotiff_root_dir=geotiff_root_dir,
        metadata_cache_dir=metadata_cache_dir,
        split="val",
        tile_size=int(tile_size),
        patch_stride=int(patch_stride),
        max_land_fraction=float(max_land_fraction),
        require_argo_for_val=True,
        surface_conditioning={"sources": ["sst", "sss", "adt"]},
        synthetic_target={"enabled": False},
        output_fields=("temperature", "salinity"),
        val_year=int(validation_year),
        random_seed=int(random_seed),
    )
    prior = VerticalOffsetPrior.from_npz(
        statistics_path, expected_depth_axis_m=dataset.depth_axis_m
    )
    indices = _depth_indices(dataset.depth_axis_m, depths_m)
    written: list[Path] = []

    summary_path = output_root / "vertical_offset_summary.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    if prior.is_spatial:
        for month_index in range(12):
            label = f"{month_index + 1:02d}"
            axes[0].plot(
                prior.temperature_offset_c[month_index].mean(axis=(0, 1)),
                prior.depth_axis_m,
                label=label,
            )
            axes[1].plot(
                prior.salinity_offset_psu[month_index].mean(axis=(0, 1)),
                prior.depth_axis_m,
                label=label,
            )
        axes[1].legend(title="Month", ncol=2, fontsize=8)
        figure.suptitle("Monthly mean smooth GLORYS depth-minus-surface deltas")
    else:
        axes[0].plot(prior.temperature_offset_c, prior.depth_axis_m)
        axes[1].plot(prior.salinity_offset_psu, prior.depth_axis_m)
        figure.suptitle("Mean GLORYS depth-minus-surface coefficients")
    axes[0].set_xlabel("temperature delta (degrees C)")
    axes[1].set_xlabel("salinity delta (PSU)")
    for axis in axes:
        axis.set_ylabel("depth (m)")
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
    figure.savefig(summary_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    written.append(summary_path)

    for site_name, latitude, longitude in REGIONS:
        row = _select_textured_row(
            dataset,
            latitude=latitude,
            longitude=longitude,
            candidate_count=candidate_count,
        )
        surface = dataset._load_surface_fields(row)
        depth_valid_mask = dataset._load_prior_depth_valid_mask(row)
        latitude_deg, longitude_deg = _patch_coordinates(
            row, tile_size=dataset.tile_size
        )
        sample = prior.sample(
            {
                "sst": surface["sst"] + np.float32(CELSIUS_TO_KELVIN_OFFSET),
                "sss": surface["sss"],
            },
            depth_valid_mask=depth_valid_mask,
            date=int(row["date"]),
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
        )
        title = (
            f"{site_name.replace('_', ' ').title()} | {int(row['date'])} | "
            f"{float(row['lat_center']):.1f} deg N, "
            f"{float(row['lon_center']):.1f} deg E"
        )
        for suffix, synthetic, glorys, units, cmap in (
            (
                "temperature",
                sample.temperature_k - np.float32(CELSIUS_TO_KELVIN_OFFSET),
                dataset._load_y_patch(row),
                "degrees C",
                "turbo",
            ),
            (
                "salinity",
                sample.salinity_psu,
                dataset._load_y_salinity_patch(row),
                "PSU",
                "viridis",
            ),
        ):
            output_path = output_root / f"{site_name}_{suffix}_depths.png"
            _plot_mosaic(
                synthetic=synthetic,
                glorys=glorys,
                depths=dataset.depth_axis_m,
                indices=indices,
                title=title,
                units=units,
                cmap=cmap,
                output_path=output_path,
                dpi=dpi,
            )
            written.append(output_path)
        profile_path = output_root / f"{site_name}_synthetic_vs_glorys_profiles.png"
        _plot_profile_comparison(
            synthetic_temperature=sample.temperature_k
            - np.float32(CELSIUS_TO_KELVIN_OFFSET),
            glorys_temperature=dataset._load_y_patch(row),
            synthetic_salinity=sample.salinity_psu,
            glorys_salinity=dataset._load_y_salinity_patch(row),
            depths=dataset.depth_axis_m,
            pixel=_profile_pixel(depth_valid_mask),
            title=title,
            output_path=profile_path,
            dpi=dpi,
        )
        written.append(profile_path)
    dataset.raster_cache.close()
    return written


def _build_parser() -> argparse.ArgumentParser:
    """Build the plotting command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geotiff-root-dir", type=Path, required=True)
    parser.add_argument("--statistics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-cache-dir", type=Path, default=None)
    parser.add_argument("--validation-year", type=int, default=2018)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--max-land-fraction", type=float, default=0.30)
    parser.add_argument(
        "--depths-m",
        type=float,
        nargs="+",
        default=(0, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000),
    )
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=170)
    return parser


def main() -> None:
    """Generate plots from command-line arguments."""
    args = _build_parser().parse_args()
    paths = plot_examples(
        geotiff_root_dir=args.geotiff_root_dir,
        statistics_path=args.statistics_path,
        output_dir=args.output_dir,
        metadata_cache_dir=args.metadata_cache_dir,
        validation_year=args.validation_year,
        tile_size=args.tile_size,
        patch_stride=args.patch_stride,
        max_land_fraction=args.max_land_fraction,
        depths_m=args.depths_m,
        candidate_count=args.candidate_count,
        random_seed=args.random_seed,
        dpi=args.dpi,
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
