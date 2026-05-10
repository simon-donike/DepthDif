# Example:
# /work/envs/depth/bin/python data/dataset_creation/data_download_raw/get_world/download_manipulate_world_file.py --overwrite
"""Download world polygons and rasterize them as a GLORYS-aligned land mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import requests

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/simon-donike/file_hosting/"
    "89a324d74f362a9820452fba87a64dbff99f53df/world.geojson"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_GEOJSON_NAME = "world.geojson"
DEFAULT_GEOTIFF_NAME = "world_land_mask_glorys_0p1.tif"
DEFAULT_RESOLUTION_DEG = 0.1
DEFAULT_LEFT = -180.0
DEFAULT_TOP = 90.0
DEFAULT_WIDTH = 3600
DEFAULT_HEIGHT = 1800


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    timeout_seconds: int = 120,
) -> Path:
    """Download ``url`` to ``output_path`` unless an existing file can be reused."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=int(timeout_seconds)) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(output_path)
    return output_path


def load_geojson_geometries(geojson_path: Path) -> list[dict[str, Any]]:
    """Load GeoJSON geometries accepted by ``rasterio.features.rasterize``."""
    with Path(geojson_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    payload_type = str(payload.get("type", ""))
    if payload_type == "FeatureCollection":
        geometries = [
            feature.get("geometry")
            for feature in payload.get("features", [])
            if isinstance(feature, dict) and feature.get("geometry") is not None
        ]
    elif payload_type == "Feature":
        geometries = [payload.get("geometry")]
    else:
        geometries = [payload]

    geometries = [geom for geom in geometries if isinstance(geom, dict)]
    if not geometries:
        raise RuntimeError(f"No rasterizable geometries found in: {geojson_path}")
    return geometries


def rasterize_land_mask(
    geometries: list[dict[str, Any]],
    *,
    resolution_deg: float = DEFAULT_RESOLUTION_DEG,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    left: float = DEFAULT_LEFT,
    top: float = DEFAULT_TOP,
    all_touched: bool = False,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Rasterize land polygons to a global uint8 mask where land is 1 and water is 0."""
    transform = from_origin(
        float(left),
        float(top),
        float(resolution_deg),
        float(resolution_deg),
    )
    shapes = ((geom, np.uint8(1)) for geom in geometries)
    mask = rasterize(
        shapes=shapes,
        out_shape=(int(height), int(width)),
        fill=np.uint8(0),
        transform=transform,
        dtype="uint8",
        all_touched=bool(all_touched),
    )
    return mask.astype(np.uint8, copy=False), transform


def write_land_mask_geotiff(
    output_path: Path,
    mask: np.ndarray,
    transform: rasterio.Affine,
    *,
    resolution_deg: float,
    source_url: str,
    source_geojson: Path,
    overwrite: bool = False,
) -> Path:
    """Write the land mask as a compressed EPSG:4326 GeoTIFF."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=int(mask.shape[0]),
        width=int(mask.shape[1]),
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(mask, 1)
        dst.update_tags(
            grid="glorys_global_0p1_degree",
            resolution_deg=str(float(resolution_deg)),
            source_url=str(source_url),
            source_geojson=str(source_geojson),
            land_value="1",
            water_value="0",
        )
        dst.set_band_description(1, "land_mask")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the world-mask export utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Download a world GeoJSON file and rasterize it to a global "
            "0.1-degree GLORYS-style land mask GeoTIFF."
        )
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="GeoJSON URL to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the downloaded GeoJSON and output GeoTIFF are written.",
    )
    parser.add_argument(
        "--geojson-name",
        default=DEFAULT_GEOJSON_NAME,
        help="Downloaded GeoJSON filename inside --output-dir.",
    )
    parser.add_argument(
        "--geotiff-name",
        default=DEFAULT_GEOTIFF_NAME,
        help="Output GeoTIFF filename inside --output-dir.",
    )
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=DEFAULT_RESOLUTION_DEG,
        help="Output grid resolution in degrees.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Output raster width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Output raster height in pixels.",
    )
    parser.add_argument(
        "--left",
        type=float,
        default=DEFAULT_LEFT,
        help="Western grid edge in degrees.",
    )
    parser.add_argument(
        "--top",
        type=float,
        default=DEFAULT_TOP,
        help="Northern grid edge in degrees.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Mark every pixel touched by a land polygon instead of center-hit pixels only.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the GeoJSON even if the local file already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output GeoTIFF if it already exists.",
    )
    return parser


def main() -> None:
    """Download the source GeoJSON and write a GLORYS-aligned land-mask GeoTIFF."""
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    geojson_path = output_dir / str(args.geojson_name)
    geotiff_path = output_dir / str(args.geotiff_name)

    downloaded_path = download_file(
        str(args.source_url),
        geojson_path,
        force=bool(args.force_download),
    )
    geometries = load_geojson_geometries(downloaded_path)
    mask, transform = rasterize_land_mask(
        geometries,
        resolution_deg=float(args.resolution_deg),
        width=int(args.width),
        height=int(args.height),
        left=float(args.left),
        top=float(args.top),
        all_touched=bool(args.all_touched),
    )
    output_path = write_land_mask_geotiff(
        geotiff_path,
        mask,
        transform,
        resolution_deg=float(args.resolution_deg),
        source_url=str(args.source_url),
        source_geojson=downloaded_path,
        overwrite=bool(args.overwrite),
    )
    land_fraction = float(np.count_nonzero(mask) / mask.size)
    print(f"Downloaded GeoJSON: {downloaded_path}")
    print(f"Wrote land mask: {output_path}")
    print(f"Shape: {mask.shape[0]} rows x {mask.shape[1]} cols")
    print(f"Values: 1=land, 0=water")
    print(f"Land fraction: {land_fraction:.4f}")


if __name__ == "__main__":
    main()
