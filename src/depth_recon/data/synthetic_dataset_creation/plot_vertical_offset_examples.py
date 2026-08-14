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
from typing import Any, Sequence

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
                    "surface + mean offset" if row_index == 0 else "held-out GLORYS"
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
        pretraining_prior={"enabled": False},
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
    figure, axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    axes[0].plot(prior.temperature_offset_c, prior.depth_axis_m)
    axes[0].set_xlabel("temperature offset (degrees C)")
    axes[1].plot(prior.salinity_offset_psu, prior.depth_axis_m)
    axes[1].set_xlabel("salinity offset (PSU)")
    for axis in axes:
        axis.set_ylabel("depth (m)")
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
    figure.suptitle("Mean GLORYS depth-minus-surface coefficients")
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
        sample = prior.sample(
            {
                "sst": surface["sst"] + np.float32(CELSIUS_TO_KELVIN_OFFSET),
                "sss": surface["sss"],
            },
            depth_valid_mask=dataset._load_prior_depth_valid_mask(row),
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
        "--depths-m", type=float, nargs="+", default=(0, 50, 100, 250, 500, 1000)
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
