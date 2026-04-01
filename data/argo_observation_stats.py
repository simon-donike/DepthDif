from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from tqdm import tqdm

try:
    from data.dataset_ostia_argo import OstiaArgoTileDataset
except ModuleNotFoundError:
    # Allow running this file directly via `python data/argo_observation_stats.py`.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.dataset_ostia_argo import OstiaArgoTileDataset


def _parse_year(value: Any) -> int | None:
    """Parse year from CSV date-like values (expected YYYYMMDD)."""
    if value is None:
        return None
    try:
        date_int = int(str(value).strip())
    except ValueError:
        return None
    date_str = str(date_int)
    if len(date_str) < 4:
        return None
    year = int(date_str[:4])
    if year < 1900 or year > 2100:
        return None
    return year


def _patch_bounds_from_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return normalized lat/lon bounds as (lat_lo, lat_hi, lon_lo, lon_hi)."""
    lat0 = float(row["lat0"])
    lat1 = float(row["lat1"])
    lon0 = float(row["lon0"])
    lon1 = float(row["lon1"])
    return min(lat0, lat1), max(lat0, lat1), min(lon0, lon1), max(lon0, lon1)


def _patch_key_from_row(row: dict[str, Any]) -> str:
    """Build a stable patch key for aggregation and GeoJSON feature identity."""
    if "patch_id" in row and str(row.get("patch_id", "")).strip() != "":
        return str(row["patch_id"])
    lat_lo, lat_hi, lon_lo, lon_hi = _patch_bounds_from_row(row)
    return f"{lat_lo:.6f}|{lat_hi:.6f}|{lon_lo:.6f}|{lon_hi:.6f}"


def _load_world_outline(world_shapefile: Path | None = None) -> gpd.GeoDataFrame:
    """Load world polygons for map background."""
    if world_shapefile is not None:
        if not world_shapefile.exists():
            raise FileNotFoundError(f"World shapefile not found: {world_shapefile}")
        return gpd.read_file(world_shapefile)

    # GeoPandas <=0.14 exposed this directly; newer versions removed datasets.get_path.
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass

    # Use pyogrio's bundled Natural Earth fixture as an offline fallback.
    try:
        import pyogrio  # type: ignore

        fixture = (
            Path(pyogrio.__file__).resolve().parent
            / "tests"
            / "fixtures"
            / "naturalearth_lowres"
            / "naturalearth_lowres.shp"
        )
        if fixture.exists():
            return gpd.read_file(fixture)
    except Exception:
        pass

    # Last-resort fallback so map plotting still works without external downloads.
    return gpd.GeoDataFrame(
        {"name": ["world_bounds"]},
        geometry=[box(-180.0, -90.0, 180.0, 90.0)],
        crs="EPSG:4326",
    )


def _write_geojson(
    *,
    out_path: Path,
    patch_meta: dict[str, dict[str, Any]],
    patch_obs_fraction_sum: dict[str, float],
    patch_sample_count: dict[str, int],
) -> tuple[dict[str, Any], int]:
    """Write per-patch observation aggregates as GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    for patch_key, meta in patch_meta.items():
        lat_lo = float(meta["lat_lo"])
        lat_hi = float(meta["lat_hi"])
        lon_lo = float(meta["lon_lo"])
        lon_hi = float(meta["lon_hi"])

        n_days_all = int(patch_sample_count.get(patch_key, 0))
        avg_obs_fraction_per_day = (
            float(patch_obs_fraction_sum.get(patch_key, 0.0) / n_days_all) if n_days_all > 0 else 0.0
        )

        props: dict[str, Any] = {
            "patch_key": patch_key,
            "patch_id": meta["patch_id"],
            "lat0": meta["lat0"],
            "lat1": meta["lat1"],
            "lon0": meta["lon0"],
            "lon1": meta["lon1"],
            "days_with_sample": n_days_all,
            "avg_observation_fraction_per_day": avg_obs_fraction_per_day,
        }

        # GeoJSON polygon ring is (lon, lat) and must be closed.
        ring = [
            [lon_lo, lat_lo],
            [lon_hi, lat_lo],
            [lon_hi, lat_hi],
            [lon_lo, lat_hi],
            [lon_lo, lat_lo],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return fc, len(features)


def _plot_histogram(
    *,
    out_path: Path,
    patch_day_obs_fractions: list[float],
) -> None:
    """Plot distribution of observed-pixel fractions over patch-day samples."""
    if not patch_day_obs_fractions:
        raise RuntimeError("No patch-day fractions available; cannot draw histogram.")

    vals = np.asarray(patch_day_obs_fractions, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=40, color="#1f77b4", alpha=0.9)
    ax.set_title("Observed Pixel Fraction per Patch-Day (valid_mask_1d)")
    ax.set_xlabel("Observed fraction (0-1)")
    ax.set_ylabel("Patch-day sample count")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_world_map(
    *,
    out_path: Path,
    feature_collection: dict[str, Any],
    year_start: int,
    year_end: int,
    world_shapefile: Path | None,
) -> None:
    """Plot world map with patch polygons color-coded by avg observation fraction/day."""
    if not feature_collection["features"]:
        raise RuntimeError("GeoJSON is empty; cannot draw map.")

    world = _load_world_outline(world_shapefile=world_shapefile)
    patch_gdf = gpd.GeoDataFrame.from_features(feature_collection["features"], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(16, 8))
    world.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.8)
    patch_gdf.plot(
        ax=ax,
        column="avg_observation_fraction_per_day",
        cmap="viridis",
        linewidth=0.0,
        alpha=0.9,
        legend=True,
        legend_kwds={"label": "Average observed fraction per day per patch"},
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Average Argo Observation Fraction per Patch per Day ({year_start}-{year_end})"
    )
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate OstiaArgoTileDataset samples and compute Argo observation stats, "
            "GeoJSON patch aggregates, and plots."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/work/data/depth_v2/ostia_patch_index_daily_0p1.csv"),
        help="Merged daily CSV used by OstiaArgoTileDataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["all", "train", "val"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Patch side length used by dataset rasterization.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help=(
            "Temporal window length passed to OstiaArgoTileDataset "
            "(even values auto-adjust to odd in dataset)."
            "Set 1 for daily."
        ),
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2019,
        help="First year to include (inclusive).",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2019,
        help="Last year to include (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/code/DepthDif/temp/ostia_argo_EDA/"),
        help="Directory for GeoJSON and plot outputs.",
    )
    parser.add_argument(
        "--quiet-init",
        action="store_true",
        help="Disable dataset init progress logs.",
    )
    parser.add_argument(
        "--world-shapefile",
        type=Path,
        default=None,
        help="Optional local world-outline shapefile path (.shp).",
    )
    parser.add_argument(
        "--intermediate-every-n-samples",
        type=int,
        default=500,
        help="Overwrite *_intermediate outputs every N processed samples (0 disables).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.year_end < args.year_start:
        raise ValueError("--year-end must be >= --year-start.")

    years = list(range(int(args.year_start), int(args.year_end) + 1))
    year_set = set(years)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OstiaArgoTileDataset(
        csv_path=args.csv_path,
        # Keep dataset-root resolution explicit for depth_v2-anchored CSV paths.
        root_path=Path("/work/data/depth_v2/"),
        split=args.split,
        tile_size=int(args.tile_size),
        days=int(args.days),
        verbose_init=not bool(args.quiet_init),
    )

    # Pre-select relevant indices by year to avoid unnecessary __getitem__ calls.
    selected_indices: list[int] = []
    for idx, row in enumerate(dataset._rows):
        year = _parse_year(row.get("date"))
        if year in year_set:
            selected_indices.append(idx)

    if not selected_indices:
        raise RuntimeError(
            f"No dataset rows found for years {args.year_start}-{args.year_end}."
        )

    print(f"Selected samples for year window {args.year_start}-{args.year_end}: {len(selected_indices)}")

    patch_day_obs_fractions: list[float] = []
    patch_obs_fraction_sum: dict[str, float] = defaultdict(float)
    patch_sample_count: dict[str, int] = defaultdict(int)
    patch_meta: dict[str, dict[str, Any]] = {}
    intermediate_every_n = max(int(args.intermediate_every_n_samples), 0)
    intermediate_hist_path = output_dir / "argo_observations_histogram_intermediate.png"
    intermediate_geojson_path = output_dir / "argo_observations_per_patch_intermediate.geojson"
    intermediate_map_path = output_dir / "argo_observations_map_intermediate.png"

    for sample_i, idx in enumerate(
        tqdm(selected_indices, desc="Iterating dataset", unit="sample"),
        start=1,
    ):
        row = dataset._rows[idx]
        year = _parse_year(row.get("date"))
        if year not in year_set:
            continue

        sample = dataset[idx]
        # Use collapsed 1D validity so fractions are computed per patch pixel per day.
        valid_mask_1d = sample["x_valid_mask_1d"]
        obs_fraction = (
            float(valid_mask_1d.sum().item() / int(valid_mask_1d.numel()))
            if int(valid_mask_1d.numel()) > 0
            else 0.0
        )
        patch_day_obs_fractions.append(obs_fraction)

        patch_key = _patch_key_from_row(row)
        patch_obs_fraction_sum[patch_key] += obs_fraction
        patch_sample_count[patch_key] += 1

        if patch_key not in patch_meta:
            lat_lo, lat_hi, lon_lo, lon_hi = _patch_bounds_from_row(row)
            patch_meta[patch_key] = {
                "patch_id": row.get("patch_id"),
                "lat0": float(row["lat0"]),
                "lat1": float(row["lat1"]),
                "lon0": float(row["lon0"]),
                "lon1": float(row["lon1"]),
                "lat_lo": lat_lo,
                "lat_hi": lat_hi,
                "lon_lo": lon_lo,
                "lon_hi": lon_hi,
            }

        if intermediate_every_n > 0 and (sample_i % intermediate_every_n == 0):
            # Intermediate checkpoints overwrite fixed files so the latest progress is always visible.
            _plot_histogram(
                out_path=intermediate_hist_path,
                patch_day_obs_fractions=patch_day_obs_fractions,
            )
            intermediate_fc, _ = _write_geojson(
                out_path=intermediate_geojson_path,
                patch_meta=patch_meta,
                patch_obs_fraction_sum=patch_obs_fraction_sum,
                patch_sample_count=patch_sample_count,
            )
            _plot_world_map(
                out_path=intermediate_map_path,
                feature_collection=intermediate_fc,
                year_start=int(args.year_start),
                year_end=int(args.year_end),
                world_shapefile=args.world_shapefile,
            )

    hist_path = output_dir / f"argo_observations_histogram_{args.year_start}_{args.year_end}.png"
    geojson_path = (
        output_dir / f"argo_observations_per_patch_{args.year_start}_{args.year_end}.geojson"
    )
    geojson_stable_path = output_dir / "argo_observations_per_patch.geojson"
    map_path = output_dir / f"argo_observations_map_{args.year_start}_{args.year_end}.png"

    _plot_histogram(
        out_path=hist_path,
        patch_day_obs_fractions=patch_day_obs_fractions,
    )
    feature_collection, n_features = _write_geojson(
        out_path=geojson_path,
        patch_meta=patch_meta,
        patch_obs_fraction_sum=patch_obs_fraction_sum,
        patch_sample_count=patch_sample_count,
    )
    # Also persist a stable filename in output_dir for easier downstream lookup.
    with geojson_stable_path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False)
    _plot_world_map(
        out_path=map_path,
        feature_collection=feature_collection,
        year_start=int(args.year_start),
        year_end=int(args.year_end),
        world_shapefile=args.world_shapefile,
    )

    print("Done.")
    print(f"- Histogram: {hist_path}")
    print(f"- GeoJSON: {geojson_path} ({n_features} features)")
    print(f"- GeoJSON (stable): {geojson_stable_path}")
    print(f"- Map: {map_path}")
    print(f"- Patch-day samples: {len(patch_day_obs_fractions)}")
    if patch_sample_count:
        patch_avg_fraction_per_day = {
            k: (float(patch_obs_fraction_sum[k]) / float(v))
            for k, v in patch_sample_count.items()
            if v > 0
        }
        global_patch_avg = float(
            np.mean(np.asarray(list(patch_avg_fraction_per_day.values()), dtype=np.float64))
        )
        print(f"- Mean patch avg observation fraction per day: {global_patch_avg:.6f}")


if __name__ == "__main__":
    main()
