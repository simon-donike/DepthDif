from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box


def _normalize_patch_bounds(row: pd.Series) -> tuple[float, float, float, float]:
    """Return patch bounds as (lat_lo, lat_hi, lon_lo, lon_hi)."""
    lat0 = float(row["lat0"])
    lat1 = float(row["lat1"])
    lon0 = float(row["lon0"])
    lon1 = float(row["lon1"])
    return min(lat0, lat1), max(lat0, lat1), min(lon0, lon1), max(lon0, lon1)


def _patch_key_from_row(row: pd.Series) -> str:
    """Build a stable patch key even when patch_id is missing."""
    patch_id = str(row.get("patch_id", "")).strip()
    if patch_id != "":
        return patch_id
    lat_lo, lat_hi, lon_lo, lon_hi = _normalize_patch_bounds(row)
    return f"{lat_lo:.6f}|{lat_hi:.6f}|{lon_lo:.6f}|{lon_hi:.6f}"


def _build_feature(
    *,
    patch_key: str,
    patch_id: str,
    lat0: float,
    lat1: float,
    lon0: float,
    lon1: float,
    sample_count: int,
    median_valid_pixels: float,
) -> dict[str, Any]:
    lat_lo = min(lat0, lat1)
    lat_hi = max(lat0, lat1)
    lon_lo = min(lon0, lon1)
    lon_hi = max(lon0, lon1)
    ring = [
        [lon_lo, lat_lo],
        [lon_hi, lat_lo],
        [lon_hi, lat_hi],
        [lon_lo, lat_hi],
        [lon_lo, lat_lo],
    ]
    return {
        "type": "Feature",
        "properties": {
            "patch_key": patch_key,
            "patch_id": patch_id,
            "lat0": lat0,
            "lat1": lat1,
            "lon0": lon0,
            "lon1": lon1,
            "sample_count": int(sample_count),
            "median_valid_pixels": float(median_valid_pixels),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [ring],
        },
    }


def _load_world_outline(world_shapefile: Path | None = None) -> gpd.GeoDataFrame:
    """Load world polygons for a simple map background."""
    if world_shapefile is not None:
        if not world_shapefile.exists():
            raise FileNotFoundError(f"World shapefile not found: {world_shapefile}")
        return gpd.read_file(world_shapefile)

    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass

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

    return gpd.GeoDataFrame(
        {"name": ["world_bounds"]},
        geometry=[box(-180.0, -90.0, 180.0, 90.0)],
        crs="EPSG:4326",
    )


def _plot_world_map(
    *,
    out_path: Path,
    feature_collection: dict[str, Any],
    world_shapefile: Path | None,
) -> None:
    """Plot patch polygons color-coded by median valid-pixel count."""
    if not feature_collection["features"]:
        raise RuntimeError("GeoJSON is empty; cannot draw map.")

    world = _load_world_outline(world_shapefile=world_shapefile)
    patch_gdf = gpd.GeoDataFrame.from_features(feature_collection["features"], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(16, 8))
    world.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.8)
    patch_gdf.plot(
        ax=ax,
        column="median_valid_pixels",
        cmap="viridis",
        linewidth=0.0,
        alpha=0.9,
        legend=True,
        legend_kwds={"label": "Median valid pixels per patch"},
    )
    for feature in feature_collection["features"]:
        props = feature["properties"]
        centroid_lat = 0.5 * (float(props["lat0"]) + float(props["lat1"]))
        centroid_lon = 0.5 * (float(props["lon0"]) + float(props["lon1"]))
        # Keep labels compact and neutral so the heatmap remains readable.
        ax.text(
            centroid_lon,
            centroid_lat,
            f"{int(round(float(props['median_valid_pixels'])))}",
            fontsize=6,
            color="black",
            ha="center",
            va="center",
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#d9d9d9",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Median Valid Argo Pixels per Spatial Patch")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the median number of valid Argo pixels per spatial patch "
            "from the depth_prod manifest and save it as GeoJSON."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/work/data/depth_prod/ostia_argo_tiff_index.csv"),
        help="Path to the depth_prod manifest CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/work/data/depth_prod/argo_valid_pixels_per_patch.geojson"),
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("/work/code/DepthDif/data/glorys_argo_alignment/figures/argo_valid_pixels_per_patch.png"),
        help="Output PNG heatmap path.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help=(
            "Include rows with a non-empty export_skipped_reason. "
            "By default those rows are excluded because they did not produce usable exports."
        ),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "train", "val"],
        help="Optional phase filter applied to the manifest rows before aggregation.",
    )
    parser.add_argument(
        "--world-shapefile",
        type=Path,
        default=None,
        help="Optional local world-outline shapefile path (.shp).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required_columns = {"lat0", "lat1", "lon0", "lon1", "argo_valid_spatial_observation_count"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Manifest is missing required columns: {', '.join(missing_columns)}"
        )

    if args.phase != "all":
        if "phase" not in df.columns:
            raise RuntimeError("Manifest does not contain a 'phase' column for --phase filtering.")
        df = df[df["phase"].astype(str) == str(args.phase)].copy()

    if not args.include_skipped and "export_skipped_reason" in df.columns:
        # Treat NaN and empty strings as "not skipped" so normal manifest rows remain included.
        skipped_reason = df["export_skipped_reason"].fillna("").astype(str).str.strip()
        df = df[skipped_reason == ""].copy()

    metric = pd.to_numeric(df["argo_valid_spatial_observation_count"], errors="coerce")
    df = df[metric.notna()].copy()
    df["argo_valid_spatial_observation_count"] = metric.loc[df.index].astype(float)

    if df.empty:
        raise RuntimeError("No manifest rows remain after filtering; nothing to aggregate.")

    patch_aggregates: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        patch_key = _patch_key_from_row(row)
        patch_id = str(row.get("patch_id", "")).strip()
        stats = patch_aggregates.get(patch_key)
        if stats is None:
            stats = {
                "patch_id": patch_id,
                "lat0": float(row["lat0"]),
                "lat1": float(row["lat1"]),
                "lon0": float(row["lon0"]),
                "lon1": float(row["lon1"]),
                "value_sum": 0.0,
                "sample_count": 0,
            }
            patch_aggregates[patch_key] = stats

        stats.setdefault("values", []).append(float(row["argo_valid_spatial_observation_count"]))
        stats["sample_count"] += 1

    features: list[dict[str, Any]] = []
    for patch_key in sorted(patch_aggregates):
        stats = patch_aggregates[patch_key]
        sample_count = int(stats["sample_count"])
        values = stats.get("values", [])
        median_valid_pixels = float(pd.Series(values, dtype=float).median()) if values else 0.0
        features.append(
            _build_feature(
                patch_key=patch_key,
                patch_id=str(stats["patch_id"]),
                lat0=float(stats["lat0"]),
                lat1=float(stats["lat1"]),
                lon0=float(stats["lon0"]),
                lon1=float(stats["lon1"]),
                sample_count=sample_count,
                median_valid_pixels=median_valid_pixels,
            )
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_collection = {"type": "FeatureCollection", "features": features}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False)

    _plot_world_map(
        out_path=Path(args.plot_path),
        feature_collection=feature_collection,
        world_shapefile=args.world_shapefile,
    )

    print(f"Wrote {len(features)} patch features to {output_path}")
    print(f"Wrote heatmap PNG to {Path(args.plot_path)}")


if __name__ == "__main__":
    main()
