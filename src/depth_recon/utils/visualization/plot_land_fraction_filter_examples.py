# Run example:
# /work/envs/depth/bin/python src/depth_recon/utils/visualization/plot_land_fraction_filter_examples.py --config src/depth_recon/configs/px_space/training_super_config.yaml --output docs/assets/data/patch_grid/land_fraction_filter_examples.webp --dpi 200 --quality 95

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from depth_recon.data.dataset_argo_geotiff_gridded import (  # noqa: E402
    ArgoGeoTIFFGriddedPatchDataset,
)
from depth_recon.data.dataset_grid_utils import (  # noqa: E402
    _GridParams,
    _center_lon_deg,
    _force_include_region_for_patch,
    _grid_starts,
    _parse_force_include_regions,
    _summed_area_table,
    _window_sum,
)


@dataclass(frozen=True)
class PatchCandidate:
    """One land-mask patch candidate with filter metadata."""

    y0: int
    x0: int
    lat0: float
    lat1: float
    lon0: float
    lon1: float
    lat_center: float
    lon_center: float
    land_fraction: float
    force_include_region: str
    retained: bool


@dataclass(frozen=True)
class PanelSpec:
    """Plot settings for one panel in the land-fraction example figure."""

    label: str
    title: str
    subtitle: str
    candidate: PatchCandidate


@dataclass(frozen=True)
class PanelData:
    """Raster and point data plotted for one example panel."""

    spec: PanelSpec
    sst: np.ndarray
    ocean_mask: np.ndarray
    argo_x: np.ndarray
    argo_y: np.ndarray
    date: int


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return value


def _grid_params_from_config(config_path: Path) -> tuple[_GridParams, Path]:
    """Parse grid plotting settings from the training config."""
    cfg = _load_yaml(config_path)
    dataset_cfg = cfg["data"]["dataset"]
    core_cfg = dataset_cfg["core"]
    grid_cfg = dataset_cfg["grid"]
    geotiff_root = Path(core_cfg["geotiff_root_dir"])
    land_mask_path = Path(grid_cfg["land_mask_path"])
    if not land_mask_path.is_absolute():
        land_mask_path = geotiff_root / land_mask_path

    grid_params = _GridParams(
        tile_size=int(grid_cfg["tile_size"]),
        resolution_deg=float(grid_cfg["resolution_deg"]),
        invalid_threshold=0.0,
        invalid_mask_flags=(),
        val_fraction=0.0,
        val_year=2015,
        split_seed=0,
        patch_grid_source=str(grid_cfg["patch_grid_source"]),
        land_mask_path=land_mask_path,
        patch_stride=int(grid_cfg["patch_stride"]),
        max_land_fraction=float(grid_cfg["max_land_fraction"]),
        force_include_regions=_parse_force_include_regions(
            grid_cfg.get("force_include_regions")
        ),
    )
    return grid_params, land_mask_path


def _build_candidates(
    *,
    land_mask: np.ndarray,
    transform: Any,
    width: int,
    height: int,
    grid_params: _GridParams,
) -> list[PatchCandidate]:
    """Build retained and rejected patch candidates from the land mask."""
    tile = int(grid_params.tile_size)
    stride = int(grid_params.effective_patch_stride)
    resolution = float(grid_params.resolution_deg)
    table = _summed_area_table(np.asarray(land_mask) > 0.5)
    candidates: list[PatchCandidate] = []

    for y0 in _grid_starts(height, tile, stride):
        for x0 in _grid_starts(width, tile, stride):
            land_fraction = _window_sum(table, y0=y0, x0=x0, tile=tile) / float(
                tile * tile
            )
            lon0 = float(transform.c) + (float(x0) * resolution)
            lon1 = lon0 + (float(tile) * resolution)
            lat1 = float(transform.f) - (float(y0) * resolution)
            lat0 = lat1 - (float(tile) * resolution)
            lat_center = 0.5 * (lat0 + lat1)
            lon_center = _center_lon_deg(lon0, lon1)
            force_region = _force_include_region_for_patch(
                lat_center=lat_center,
                lon_center=lon_center,
                land_fraction=land_fraction,
                regions=grid_params.force_include_regions,
            )
            retained = (
                float(land_fraction) <= float(grid_params.max_land_fraction)
                or force_region is not None
            )
            candidates.append(
                PatchCandidate(
                    y0=int(y0),
                    x0=int(x0),
                    lat0=float(lat0),
                    lat1=float(lat1),
                    lon0=float(lon0),
                    lon1=float(lon1),
                    lat_center=float(lat_center),
                    lon_center=float(lon_center),
                    land_fraction=float(land_fraction),
                    force_include_region=(
                        "" if force_region is None else force_region.name
                    ),
                    retained=bool(retained),
                )
            )
    return candidates


def _nearest_candidate(
    candidates: list[PatchCandidate],
    *,
    target_land_fraction: float,
    target_lat: float,
    target_lon: float,
    predicate: Any,
) -> PatchCandidate:
    """Select one deterministic candidate nearest to the requested target."""
    matches = [candidate for candidate in candidates if predicate(candidate)]
    if not matches:
        raise RuntimeError("No patch candidate matched the requested example filter.")
    return min(
        matches,
        key=lambda candidate: (
            abs(candidate.land_fraction - float(target_land_fraction)),
            abs(candidate.lat_center - float(target_lat)),
            abs(candidate.lon_center - float(target_lon)),
        ),
    )


def _panel_specs(
    candidates: list[PatchCandidate],
    grid_params: _GridParams,
) -> list[PanelSpec]:
    """Choose deterministic examples for the three filter outcomes."""
    max_land = float(grid_params.max_land_fraction)
    accepted = _nearest_candidate(
        candidates,
        target_land_fraction=0.28,
        target_lat=35.0,
        target_lon=15.0,
        predicate=lambda item: (
            item.retained
            and not item.force_include_region
            and 0.20 <= item.land_fraction <= max_land
            and abs(item.lat_center) < 70.0
        ),
    )
    rejected = _nearest_candidate(
        candidates,
        target_land_fraction=0.45,
        target_lat=35.0,
        target_lon=15.0,
        predicate=lambda item: (
            not item.retained
            and max_land < item.land_fraction <= 0.65
            and abs(item.lat_center) < 70.0
        ),
    )
    force_included = _nearest_candidate(
        candidates,
        target_land_fraction=0.45,
        target_lat=38.0,
        target_lon=24.0,
        predicate=lambda item: (
            item.retained
            and item.force_include_region == "mediterranean"
            and item.land_fraction > max_land
        ),
    )

    return [
        PanelSpec(
            label="a",
            title="Kept by default",
            subtitle=f"land fraction {accepted.land_fraction:.2f} <= {max_land:.2f}",
            candidate=accepted,
        ),
        PanelSpec(
            label="b",
            title="Rejected by land cap",
            subtitle=f"land fraction {rejected.land_fraction:.2f} > {max_land:.2f}",
            candidate=rejected,
        ),
        PanelSpec(
            label="c",
            title="Kept by regional override",
            subtitle=(
                f"{force_included.force_include_region.replace('_', ' ')}: "
                f"{force_included.land_fraction:.2f} > {max_land:.2f}"
            ),
            candidate=force_included,
        ),
    ]


def _candidate_row(
    candidate: PatchCandidate, *, date: int, patch_id: int
) -> dict[str, Any]:
    """Build a dataset-compatible row for a patch candidate and date."""
    return {
        "patch_id": int(patch_id),
        "grid_y0": int(candidate.y0),
        "grid_x0": int(candidate.x0),
        "lat0": float(candidate.lat0),
        "lat1": float(candidate.lat1),
        "lon0": float(candidate.lon0),
        "lon1": float(candidate.lon1),
        "lat_center": float(candidate.lat_center),
        "lon_center": float(candidate.lon_center),
        "land_fraction": float(candidate.land_fraction),
        "date": int(date),
        "split": "all",
        "phase": "all",
    }


def _best_date_for_candidate(
    dataset: ArgoGeoTIFFGriddedPatchDataset,
    candidate: PatchCandidate,
) -> int:
    """Pick the available raster date with the most ARGO support for a patch."""
    best_date = int(dataset.available_dates[0])
    best_count = -1
    for date_value in dataset.available_dates:
        if dataset.argo_store is None:
            break
        indices = dataset.argo_store.query_indices(
            target_date=int(date_value),
            grid_y0=int(candidate.y0),
            grid_x0=int(candidate.x0),
            tile_size=int(dataset.tile_size),
        )
        count = int(indices.size)
        if count > best_count:
            best_date = int(date_value)
            best_count = count
        if count >= 3:
            # A few dots are enough for the small paper panel.
            break
    return best_date


def _panel_data(
    dataset: ArgoGeoTIFFGriddedPatchDataset,
    panels: list[PanelSpec],
) -> list[PanelData]:
    """Load SST, land, and ARGO overlays for each panel."""
    data: list[PanelData] = []
    for patch_id, panel in enumerate(panels):
        date_value = _best_date_for_candidate(dataset, panel.candidate)
        row = _candidate_row(panel.candidate, date=date_value, patch_id=patch_id)
        sst = dataset._load_eo_patch(row)[0]
        ocean_mask = dataset._load_land_mask_patch(row)[0] > 0.5
        _, x_valid_mask = dataset._rasterize_argo_patch(row)
        argo_y, argo_x = np.nonzero(np.asarray(x_valid_mask, dtype=bool).any(axis=0))
        data.append(
            PanelData(
                spec=panel,
                sst=np.asarray(sst, dtype=np.float32),
                ocean_mask=np.asarray(ocean_mask, dtype=bool),
                argo_x=argo_x.astype(np.float32, copy=False),
                argo_y=argo_y.astype(np.float32, copy=False),
                date=int(date_value),
            )
        )
    return data


def _rounded_coord_text(candidate: PatchCandidate) -> str:
    """Return a compact rounded patch-center coordinate label."""
    lat_value = int(round(abs(float(candidate.lat_center))))
    lon_value = int(round(abs(float(candidate.lon_center))))
    lat_suffix = "N" if float(candidate.lat_center) >= 0.0 else "S"
    lon_suffix = "E" if float(candidate.lon_center) >= 0.0 else "W"
    return f"{lat_value} deg {lat_suffix}, {lon_value} deg {lon_suffix}"


def _draw_panel(
    ax: plt.Axes,
    *,
    panel_data: PanelData,
    vmin: float,
    vmax: float,
) -> None:
    """Draw one labeled SST patch panel."""
    panel = panel_data.spec
    patch = np.ma.masked_where(~panel_data.ocean_mask, panel_data.sst)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#000000")
    ax.imshow(
        patch,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    if panel_data.argo_x.size > 0:
        ax.scatter(
            panel_data.argo_x,
            panel_data.argo_y,
            s=34,
            c="white",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=3,
        )
    ax.set_title(
        f"{panel.title}\n{panel.subtitle}",
        fontsize=11.5,
        fontweight="semibold",
        pad=8,
        color="#111111",
    )
    ax.text(
        0.035,
        0.965,
        panel.label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111111",
        bbox={
            "facecolor": "white",
            "edgecolor": "#222222",
            "boxstyle": "round,pad=0.18,rounding_size=0.04",
            "linewidth": 0.8,
        },
    )
    ax.text(
        0.5,
        -0.055,
        _rounded_coord_text(panel.candidate),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#111111",
        bbox={
            "facecolor": "white",
            "edgecolor": "#333333",
            "boxstyle": "round,pad=0.22,rounding_size=0.04",
            "linewidth": 0.7,
        },
    )
    ax.set_axis_off()


def _save_webp(fig: plt.Figure, output_path: Path, *, dpi: int, quality: int) -> None:
    """Save a Matplotlib figure as WebP with stable dimensions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=int(dpi), facecolor=fig.get_facecolor())
        with Image.open(tmp.name) as image:
            image.save(output_path, format="WEBP", quality=int(quality), method=6)


def render_figure(
    *,
    config_path: Path,
    output_path: Path,
    dpi: int,
    quality: int,
) -> Path:
    """Render the land-fraction filter example figure."""
    grid_params, land_mask_path = _grid_params_from_config(config_path)
    with rasterio.open(land_mask_path) as src:
        land_mask = src.read(1)
        transform = src.transform
        width = int(src.width)
        height = int(src.height)

    candidates = _build_candidates(
        land_mask=land_mask,
        transform=transform,
        width=width,
        height=height,
        grid_params=grid_params,
    )
    panels = _panel_specs(candidates, grid_params)
    cfg_for_dataset = _load_yaml(config_path)
    if "split" not in cfg_for_dataset and isinstance(cfg_for_dataset.get("data"), dict):
        # The dataset factory reads split settings at the top level, while this
        # super-config stores them under data.split.
        cfg_for_dataset = dict(cfg_for_dataset)
        cfg_for_dataset["split"] = dict(cfg_for_dataset["data"].get("split", {}))
    with NamedTemporaryFile("w", suffix=".yaml") as tmp_config:
        yaml.safe_dump(cfg_for_dataset, tmp_config)
        tmp_config.flush()
        dataset = ArgoGeoTIFFGriddedPatchDataset.from_config(
            tmp_config.name,
            split="all",
            dataset_overrides={
                "selection": {"require_argo_for_all": False},
                "finetune_sampling": {"enabled": False},
            },
        )
    panel_rasters = _panel_data(dataset, panels)
    ocean_values = np.concatenate(
        [
            item.sst[item.ocean_mask & np.isfinite(item.sst)].reshape(-1)
            for item in panel_rasters
        ]
    )
    vmin = float(np.nanpercentile(ocean_values, 2.0))
    vmax = float(np.nanpercentile(ocean_values, 98.0))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(2338 / float(dpi), 855 / float(dpi)),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.018, right=0.982, top=0.865, bottom=0.105, wspace=0.035)

    for ax, panel_data_item in zip(axes, panel_rasters, strict=True):
        _draw_panel(
            ax,
            panel_data=panel_data_item,
            vmin=vmin,
            vmax=vmax,
        )

    _save_webp(fig, output_path, dpi=dpi, quality=quality)
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render land-fraction filtering examples for the dataset docs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "src/depth_recon/configs/px_space/training_super_config.yaml",
        help="Training config that defines the land-mask patch grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "docs/assets/data/patch_grid/land_fraction_filter_examples.webp",
        help="Output WebP path.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--quality", type=int, default=95, help="WebP output quality.")
    return parser.parse_args()


def main() -> None:
    """Render the configured figure and report the output path."""
    args = _parse_args()
    output_path = render_figure(
        config_path=Path(args.config),
        output_path=Path(args.output),
        dpi=int(args.dpi),
        quality=int(args.quality),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
