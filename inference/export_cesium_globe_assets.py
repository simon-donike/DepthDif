"""Convert one global inference export run into Cesium-hostable assets.

Typical CLI:
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_20150615 \
  --public-base-url https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/inference_production/globe \
  --rclone-remote r2:depth-data/inference_production/globe \
  --rclone-sync-scope globe

The exported globe bundle uses fixed filenames and a fixed `globe/` prefix so
production uploads can be overwritten in place without changing week- or
year-specific run names.


"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import rasterio
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TEMPLATE_PATH = Path("inference/transforms/globe-config.template.json")
DEFAULT_COLOR_RAMP_PATH = Path("inference/transforms/temperature_blue_red_ramp.txt")
DEFAULT_COLOR_SCALE_MIN_C = 0.0
DEFAULT_COLOR_SCALE_MAX_C = 30.0
DEFAULT_EXTRA_ZOOM_LEVELS = 0
DEFAULT_RCLONE_SYNC_SCOPE = "globe"
DEFAULT_TILE_SIZE = 256
DEFAULT_CAMERA_LON = -38.55
DEFAULT_CAMERA_LAT = 34.50
DEFAULT_CAMERA_HEIGHT = 11_500_000.0
DEFAULT_GEOJSON_COORD_PRECISION = 4
ARGO_POINT_PROPERTY_KEYS = (
    "date",
    "patch_id",
    "export_index",
    "pixel_row",
    "pixel_col",
)
FULL_SAMPLE_PROPERTY_KEYS = (
    "date",
    "graph_png_path",
    "location_id",
    "patch_id",
    "pixel_row",
    "pixel_col",
)
ARGO_SAMPLE_LOCATION_PROPERTY_KEYS = tuple(
    dict.fromkeys(
        ARGO_POINT_PROPERTY_KEYS
        + FULL_SAMPLE_PROPERTY_KEYS
        + ("marker_kind", "has_full_depth_graph")
    )
)
PATCH_SPLIT_PROPERTY_KEYS = ("split",)


def _surface_depth_export(
    *,
    prediction_path: Path,
    ground_truth_path: Path | None,
) -> dict[str, Any]:
    return {
        "suffix": "surface",
        "label": "Surface",
        "requested_depth_m": 0.0,
        "actual_depth_m": 0.0,
        "channel_index": 0,
        "prediction_path": prediction_path,
        "ground_truth_path": ground_truth_path,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_existing_path(path_value: str | None, *, run_dir: Path) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        run_relative = run_dir / candidate.name
        if run_relative.exists():
            return run_relative
    return None


def _resolve_run_artifacts(
    run_dir: Path,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, Path | None, dict[str, Any]]:
    run_summary_path = run_dir / "run_summary.yaml"
    run_summary = _load_yaml(run_summary_path) if run_summary_path.exists() else {}

    prediction_path = _coerce_existing_path(
        run_summary.get("prediction_tif_path"),
        run_dir=run_dir,
    )
    if prediction_path is None:
        matches = sorted(run_dir.glob("*_prediction.tif"))
        if not matches:
            raise FileNotFoundError(
                "Could not locate the prediction GeoTIFF. "
                "Expected run_summary.yaml or one '*_prediction.tif' inside the run directory."
            )
        prediction_path = matches[0]

    ground_truth_path = _coerce_existing_path(
        run_summary.get("ground_truth_tif_path"),
        run_dir=run_dir,
    )
    if ground_truth_path is None:
        matches = sorted(run_dir.glob("*_glorys_top_band.tif"))
        ground_truth_path = matches[0] if matches else None

    points_path = _coerce_existing_path(
        run_summary.get("argo_points_geojson_path"),
        run_dir=run_dir,
    )
    if points_path is None:
        matches = sorted(run_dir.glob("*_argo_points.geojson"))
        points_path = matches[0] if matches else None

    patch_splits_path = _coerce_existing_path(
        run_summary.get("patch_splits_geojson_path"),
        run_dir=run_dir,
    )
    if patch_splits_path is None:
        matches = sorted(run_dir.glob("*_patch_splits.geojson"))
        patch_splits_path = matches[0] if matches else None

    full_sample_points_path = _coerce_existing_path(
        run_summary.get("full_sample_locations_geojson_path"),
        run_dir=run_dir,
    )
    if full_sample_points_path is None:
        matches = sorted(run_dir.glob("*_full_sample_locations.geojson"))
        full_sample_points_path = matches[0] if matches else None

    graphs_dir_path = _coerce_existing_path(
        run_summary.get("graphs_dir_path"),
        run_dir=run_dir,
    )
    if graphs_dir_path is None:
        candidate = run_dir / "graphs"
        graphs_dir_path = candidate if candidate.exists() else None

    return (
        prediction_path,
        ground_truth_path,
        points_path,
        patch_splits_path,
        full_sample_points_path,
        graphs_dir_path,
        run_summary,
    )


def _resolve_depth_export_artifacts(
    *,
    run_dir: Path,
    run_summary: dict[str, Any],
    prediction_path: Path,
    ground_truth_path: Path | None,
) -> list[dict[str, Any]]:
    raw_exports = run_summary.get("depth_exports")
    if not isinstance(raw_exports, list) or not raw_exports:
        return [
            _surface_depth_export(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
            )
        ]

    depth_exports: list[dict[str, Any]] = []
    for raw_export in raw_exports:
        if not isinstance(raw_export, dict):
            continue
        prediction_export_path = _coerce_existing_path(
            raw_export.get("prediction_tif_path"),
            run_dir=run_dir,
        )
        if prediction_export_path is None:
            raise FileNotFoundError(
                "Could not locate prediction GeoTIFF for depth export "
                f"{raw_export.get('label', raw_export.get('suffix', 'unknown'))!r}."
            )
        ground_truth_export_path = _coerce_existing_path(
            raw_export.get("ground_truth_tif_path"),
            run_dir=run_dir,
        )
        depth_exports.append(
            {
                "suffix": str(raw_export.get("suffix", prediction_export_path.stem)),
                "label": str(raw_export.get("label", raw_export.get("suffix", ""))),
                "requested_depth_m": float(raw_export.get("requested_depth_m", 0.0)),
                "actual_depth_m": float(raw_export.get("actual_depth_m", 0.0)),
                "channel_index": int(raw_export.get("channel_index", 0)),
                "prediction_path": prediction_export_path,
                "ground_truth_path": ground_truth_export_path,
            }
        )

    if not depth_exports:
        return [
            _surface_depth_export(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
            )
        ]
    return depth_exports


def _read_raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        bounds = ds.bounds
        tags = ds.tags()
        return {
            "west": float(bounds.left),
            "south": float(bounds.bottom),
            "east": float(bounds.right),
            "north": float(bounds.top),
            "default_camera_destination": {
                "lon": DEFAULT_CAMERA_LON,
                "lat": DEFAULT_CAMERA_LAT,
                "height": DEFAULT_CAMERA_HEIGHT,
            },
            "credit": str(tags.get("source", path.name)),
        }


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _colorize_raster(
    input_path: Path,
    output_path: Path,
    *,
    color_ramp_path: Path,
) -> None:
    gdaldem_exe = shutil.which("gdaldem")
    if gdaldem_exe is None:
        raise RuntimeError("gdaldem was not found on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gdaldem_exe,
        "color-relief",
        "-alpha",
        str(input_path),
        str(color_ramp_path),
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _estimate_native_zoom_level(input_path: Path) -> int:
    with rasterio.open(input_path) as ds:
        bounds = ds.bounds
        span_x = abs(float(bounds.right) - float(bounds.left))
        span_y = abs(float(bounds.top) - float(bounds.bottom))
        if ds.width <= 0 or ds.height <= 0 or span_x <= 0.0 or span_y <= 0.0:
            return 0

        # Pick the first XYZ zoom whose world width meets or exceeds the raster's
        # native world-equivalent width. Higher zooms only overzoom the source.
        degrees_per_pixel = min(span_x / float(ds.width), span_y / float(ds.height))
        world_pixels = 360.0 / degrees_per_pixel
        return max(
            0, int(math.ceil(math.log2(world_pixels / float(DEFAULT_TILE_SIZE))))
        )


def _build_gdal2tiles_command(
    input_path: Path,
    output_dir: Path,
    *,
    extra_zoom_levels: int,
) -> list[str]:
    gdal2tiles_exe = shutil.which("gdal2tiles.py")
    if gdal2tiles_exe is None:
        raise RuntimeError("gdal2tiles.py was not found on PATH.")

    max_zoom = _estimate_native_zoom_level(input_path) + max(0, int(extra_zoom_levels))
    return [
        gdal2tiles_exe,
        "-p",
        "mercator",
        "-r",
        "near",
        "-z",
        f"0-{max_zoom}",
        "-w",
        "none",
        str(input_path),
        str(output_dir),
    ]


def _run_gdal2tiles(
    input_path: Path, output_dir: Path, *, extra_zoom_levels: int
) -> None:
    _ensure_clean_directory(output_dir)
    command = _build_gdal2tiles_command(
        input_path,
        output_dir,
        extra_zoom_levels=extra_zoom_levels,
    )
    subprocess.run(command, check=True)


def _sync_with_rclone(local_dir: Path, remote: str) -> tuple[bool, str]:
    rclone_exe = shutil.which("rclone")
    if rclone_exe is None:
        return False, "rclone was not found on PATH. Skipping upload."

    command = [
        rclone_exe,
        "sync",
        "--progress",
        str(local_dir),
        str(remote),
    ]
    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, f"rclone sync failed with exit code {exc.returncode}."

    return True, "rclone sync completed successfully."


def _resolve_rclone_sync_source(
    *,
    run_dir: Path,
    globe_dir: Path,
    sync_scope: str,
) -> tuple[Path, str]:
    if sync_scope == "run":
        return run_dir, "run directory"
    return globe_dir, "globe assets"


def _round_geojson_coordinates(value: Any, *, decimals: int) -> Any:
    if isinstance(value, list):
        return [
            _round_geojson_coordinates(item, decimals=decimals) for item in value
        ]
    if isinstance(value, float):
        return round(value, decimals)
    return value


def _freeze_geojson_coordinates(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_geojson_coordinates(item) for item in value)
    return value


def _rewrite_geojson(
    source_path: Path,
    destination_path: Path,
    *,
    allowed_property_keys: tuple[str, ...],
    coordinate_precision: int = DEFAULT_GEOJSON_COORD_PRECISION,
) -> None:
    with source_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    features = []
    for feature in payload.get("features", []):
        geometry = dict(feature.get("geometry", {}))
        geometry["coordinates"] = _round_geojson_coordinates(
            geometry.get("coordinates", []),
            decimals=coordinate_precision,
        )
        rewritten_feature = {
            key: value
            for key, value in feature.items()
            if key not in {"geometry", "properties"}
        }
        rewritten_feature["geometry"] = geometry

        filtered_properties = {
            key: feature.get("properties", {}).get(key)
            for key in allowed_property_keys
            if key in feature.get("properties", {})
        }
        if filtered_properties:
            rewritten_feature["properties"] = filtered_properties
        features.append(rewritten_feature)

    rewritten_payload = dict(payload)
    rewritten_payload["features"] = features
    with destination_path.open("w", encoding="utf-8") as f:
        json.dump(rewritten_payload, f, separators=(",", ":"))
        f.write("\n")


def _feature_identity(feature: dict[str, Any], *, coordinate_precision: int) -> tuple[Any, ...]:
    properties = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coordinates = _freeze_geojson_coordinates(
        _round_geojson_coordinates(
            geometry.get("coordinates", []),
            decimals=coordinate_precision,
        )
    )
    return (
        coordinates,
        properties.get("date"),
        properties.get("patch_id"),
        properties.get("pixel_row"),
        properties.get("pixel_col"),
    )


def _rewrite_argo_sample_locations_geojson(
    destination_path: Path,
    *,
    points_path: Path | None,
    full_sample_points_path: Path | None,
    coordinate_precision: int = DEFAULT_GEOJSON_COORD_PRECISION,
) -> None:
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    features: list[dict[str, Any]] = []
    full_sample_identities: set[tuple[Any, ...]] = set()

    if full_sample_points_path is not None:
        with full_sample_points_path.open("r", encoding="utf-8") as f:
            full_sample_payload = json.load(f)
        for feature in full_sample_payload.get("features", []):
            if feature.get("geometry", {}).get("type") != "Point":
                continue
            full_sample_identities.add(
                _feature_identity(feature, coordinate_precision=coordinate_precision)
            )

    sources = (
        (points_path, "argo", False, full_sample_identities),
        (full_sample_points_path, "full_depth_profile", True, set()),
    )
    for source_path, marker_kind, has_full_depth_graph, skip_identities in sources:
        if source_path is None:
            continue
        with source_path.open("r", encoding="utf-8") as f:
            source_payload = json.load(f)
        if not payload["features"] and source_payload.get("type"):
            payload["type"] = source_payload["type"]

        for feature in source_payload.get("features", []):
            if feature.get("geometry", {}).get("type") != "Point":
                continue
            identity = _feature_identity(
                feature,
                coordinate_precision=coordinate_precision,
            )
            if identity in skip_identities:
                continue

            geometry = dict(feature.get("geometry", {}))
            geometry["coordinates"] = _round_geojson_coordinates(
                geometry.get("coordinates", []),
                decimals=coordinate_precision,
            )
            rewritten_feature = {
                key: value
                for key, value in feature.items()
                if key not in {"geometry", "properties"}
            }
            rewritten_feature["geometry"] = geometry

            # The viewer uses marker_kind to choose between the two ARGO marker
            # symbols while keeping both sample types in one toggleable layer.
            raw_properties = feature.get("properties", {})
            filtered_properties = {
                key: raw_properties.get(key)
                for key in ARGO_SAMPLE_LOCATION_PROPERTY_KEYS
                if key in raw_properties
            }
            filtered_properties["marker_kind"] = marker_kind
            filtered_properties["has_full_depth_graph"] = has_full_depth_graph
            rewritten_feature["properties"] = filtered_properties
            features.append(rewritten_feature)

    payload["features"] = features
    with destination_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")


def _resolve_layer_url(name: str, *, public_base_url: str | None) -> str:
    if public_base_url is None:
        return f"./{name}"
    return f"{public_base_url.rstrip('/')}/{name}"


def build_globe_config(
    *,
    selected_date: int | None,
    prediction_tiles_url: str,
    ground_truth_tiles_url: str | None,
    depth_levels: list[dict[str, Any]],
    argo_sample_locations_url: str | None,
    argo_points_url: str | None,
    patch_splits_url: str | None,
    full_sample_points_url: str | None,
    bounds: dict[str, Any],
    prediction_credit: str,
    ground_truth_credit: str | None,
    points_credit: str | None,
    patch_splits_credit: str | None,
    full_sample_points_credit: str | None,
    color_scale_min_c: float,
    color_scale_max_c: float,
    color_palette: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    config = dict(template)
    config.update(
        {
            "selected_date": selected_date,
            "prediction_tiles_url": prediction_tiles_url,
            "ground_truth_tiles_url": ground_truth_tiles_url,
            "depth_levels": depth_levels,
            "argo_sample_locations_url": argo_sample_locations_url,
            "argo_points_url": argo_points_url,
            "patch_splits_url": patch_splits_url,
            "full_sample_points_url": full_sample_points_url,
            "west": float(bounds["west"]),
            "south": float(bounds["south"]),
            "east": float(bounds["east"]),
            "north": float(bounds["north"]),
            "default_camera_destination": dict(bounds["default_camera_destination"]),
            "color_scale_min_c": float(color_scale_min_c),
            "color_scale_max_c": float(color_scale_max_c),
            "color_palette": str(color_palette),
        }
    )
    credits = dict(config.get("credits", {}))
    credits["prediction"] = prediction_credit
    if ground_truth_credit is not None:
        credits["ground_truth"] = ground_truth_credit
    if points_credit is not None:
        credits["points"] = points_credit
    if patch_splits_credit is not None:
        credits["patch_splits"] = patch_splits_credit
    if full_sample_points_credit is not None:
        credits["full_sample_points"] = full_sample_points_credit
    config["credits"] = credits
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one exported global inference run into Cesium-hostable TMS imagery "
            "plus GeoJSON and a globe-config.json manifest."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="One completed inference run directory under inference/outputs/.",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help=(
            "Optional public base URL where the generated globe directory will be hosted. "
            "When omitted, the config uses local relative URLs."
        ),
    )
    parser.add_argument(
        "--globe-dir-name",
        type=str,
        default="globe",
        help="Subdirectory name created inside the run directory for the hosted assets.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to the JSON template used as the starting point for globe-config.json.",
    )
    parser.add_argument(
        "--color-ramp-path",
        type=Path,
        default=DEFAULT_COLOR_RAMP_PATH,
        help="GDAL color-relief ramp applied to the Celsius rasters before tiling.",
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help=(
            "Optional rclone destination such as 'r2:depth-data/inference_production/globe'. "
            "When provided, the selected sync scope is mirrored after export."
        ),
    )
    parser.add_argument(
        "--rclone-sync-scope",
        type=str,
        choices=("globe", "run"),
        default=DEFAULT_RCLONE_SYNC_SCOPE,
        help=(
            "Choose whether rclone sync uploads only the generated globe directory "
            "or the full run directory."
        ),
    )
    parser.add_argument(
        "--extra-zoom-levels",
        type=int,
        default=DEFAULT_EXTRA_ZOOM_LEVELS,
        help=(
            "Number of extra on-disk zoom levels written beyond the raster's estimated "
            "native max zoom. Tiling always uses nearest-neighbor resampling."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    (
        prediction_path,
        ground_truth_path,
        points_path,
        patch_splits_path,
        full_sample_points_path,
        graphs_dir_path,
        run_summary,
    ) = _resolve_run_artifacts(run_dir)

    globe_dir = run_dir / str(args.globe_dir_name)
    globe_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = globe_dir / ".tmp_colorized_rasters"
    _ensure_clean_directory(temp_dir)
    color_ramp_path = Path(args.color_ramp_path)
    if not color_ramp_path.exists():
        raise FileNotFoundError(f"Color ramp not found: {color_ramp_path}")

    depth_exports = _resolve_depth_export_artifacts(
        run_dir=run_dir,
        run_summary=run_summary,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
    )
    config_depth_levels: list[dict[str, Any]] = []
    prediction_tiles_dir: Path | None = None
    ground_truth_tiles_dir: Path | None = None
    for depth_export in depth_exports:
        suffix = str(depth_export["suffix"])
        prediction_export_path = Path(depth_export["prediction_path"])
        prediction_colorized_path = (
            temp_dir / f"{prediction_export_path.stem}_colorized.tif"
        )
        _colorize_raster(
            prediction_export_path,
            prediction_colorized_path,
            color_ramp_path=color_ramp_path,
        )
        prediction_tiles_dir_for_depth = globe_dir / f"prediction_tiles_{suffix}"
        _run_gdal2tiles(
            prediction_colorized_path,
            prediction_tiles_dir_for_depth,
            extra_zoom_levels=args.extra_zoom_levels,
        )
        if prediction_tiles_dir is None:
            prediction_tiles_dir = prediction_tiles_dir_for_depth

        ground_truth_tiles_dir_for_depth: Path | None = None
        ground_truth_export_path = depth_export.get("ground_truth_path")
        if ground_truth_export_path is not None:
            ground_truth_export_path = Path(ground_truth_export_path)
            ground_truth_colorized_path = (
                temp_dir / f"{ground_truth_export_path.stem}_colorized.tif"
            )
            _colorize_raster(
                ground_truth_export_path,
                ground_truth_colorized_path,
                color_ramp_path=color_ramp_path,
            )
            ground_truth_tiles_dir_for_depth = globe_dir / f"ground_truth_tiles_{suffix}"
            _run_gdal2tiles(
                ground_truth_colorized_path,
                ground_truth_tiles_dir_for_depth,
                extra_zoom_levels=args.extra_zoom_levels,
            )
            if ground_truth_tiles_dir is None:
                ground_truth_tiles_dir = ground_truth_tiles_dir_for_depth

        config_depth_levels.append(
            {
                "suffix": suffix,
                "label": str(depth_export["label"]),
                "requested_depth_m": float(depth_export["requested_depth_m"]),
                "actual_depth_m": float(depth_export["actual_depth_m"]),
                "channel_index": int(depth_export["channel_index"]),
                "prediction_tiles_url": _resolve_layer_url(
                    prediction_tiles_dir_for_depth.name,
                    public_base_url=args.public_base_url,
                ),
                "ground_truth_tiles_url": (
                    None
                    if ground_truth_tiles_dir_for_depth is None
                    else _resolve_layer_url(
                        ground_truth_tiles_dir_for_depth.name,
                        public_base_url=args.public_base_url,
                    )
                ),
            }
        )

    if prediction_tiles_dir is None:
        raise RuntimeError("No prediction depth tiles were generated.")
    copied_points_path: Path | None = None
    if points_path is not None:
        copied_points_path = globe_dir / "argo_points.geojson"
        _rewrite_geojson(
            points_path,
            copied_points_path,
            allowed_property_keys=ARGO_POINT_PROPERTY_KEYS,
        )

    copied_patch_splits_path: Path | None = None
    if patch_splits_path is not None:
        copied_patch_splits_path = globe_dir / "patch_splits.geojson"
        _rewrite_geojson(
            patch_splits_path,
            copied_patch_splits_path,
            allowed_property_keys=PATCH_SPLIT_PROPERTY_KEYS,
        )

    copied_full_sample_points_path: Path | None = None
    if full_sample_points_path is not None:
        copied_full_sample_points_path = globe_dir / "full_sample_locations.geojson"
        _rewrite_geojson(
            full_sample_points_path,
            copied_full_sample_points_path,
            allowed_property_keys=FULL_SAMPLE_PROPERTY_KEYS,
        )

    copied_argo_sample_locations_path: Path | None = None
    if points_path is not None or full_sample_points_path is not None:
        copied_argo_sample_locations_path = globe_dir / "argo_sample_locations.geojson"
        _rewrite_argo_sample_locations_geojson(
            copied_argo_sample_locations_path,
            points_path=points_path,
            full_sample_points_path=full_sample_points_path,
        )

    copied_graphs_dir_path: Path | None = None
    if graphs_dir_path is not None and graphs_dir_path.exists():
        copied_graphs_dir_path = globe_dir / "graphs"
        if copied_graphs_dir_path.exists():
            shutil.rmtree(copied_graphs_dir_path)
        shutil.copytree(graphs_dir_path, copied_graphs_dir_path)

    prediction_meta = _read_raster_metadata(prediction_path)
    ground_truth_meta = (
        _read_raster_metadata(ground_truth_path)
        if ground_truth_path is not None
        else None
    )
    template = _load_template(Path(args.template_path))
    config = build_globe_config(
        selected_date=run_summary.get("selected_date"),
        prediction_tiles_url=str(config_depth_levels[0]["prediction_tiles_url"]),
        ground_truth_tiles_url=(
            None
            if ground_truth_tiles_dir is None
            else config_depth_levels[0]["ground_truth_tiles_url"]
        ),
        depth_levels=config_depth_levels,
        argo_sample_locations_url=(
            None
            if copied_argo_sample_locations_path is None
            else _resolve_layer_url(
                copied_argo_sample_locations_path.name,
                public_base_url=args.public_base_url,
            )
        ),
        argo_points_url=(
            None
            if copied_points_path is None
            else _resolve_layer_url(
                copied_points_path.name,
                public_base_url=args.public_base_url,
            )
        ),
        patch_splits_url=(
            None
            if copied_patch_splits_path is None
            else _resolve_layer_url(
                copied_patch_splits_path.name,
                public_base_url=args.public_base_url,
            )
        ),
        full_sample_points_url=(
            None
            if copied_full_sample_points_path is None
            else _resolve_layer_url(
                copied_full_sample_points_path.name,
                public_base_url=args.public_base_url,
            )
        ),
        bounds=prediction_meta,
        prediction_credit=prediction_meta["credit"],
        ground_truth_credit=(
            None if ground_truth_meta is None else ground_truth_meta["credit"]
        ),
        points_credit=None if copied_points_path is None else "Observed Argo points",
        patch_splits_credit=(
            None if copied_patch_splits_path is None else "Train/val patch split grid"
        ),
        full_sample_points_credit=(
            None
            if copied_full_sample_points_path is None
            else "Random full-depth profile locations"
        ),
        color_scale_min_c=DEFAULT_COLOR_SCALE_MIN_C,
        color_scale_max_c=DEFAULT_COLOR_SCALE_MAX_C,
        color_palette="temperature_blue_red",
        template=template,
    )
    config_path = globe_dir / "globe-config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Wrote globe assets to: {globe_dir}")
    print(f"- prediction depth tile sets: {len(config_depth_levels)}")
    print(f"- first prediction tiles: {prediction_tiles_dir}")
    if ground_truth_tiles_dir is not None:
        print(f"- first ground-truth tiles: {ground_truth_tiles_dir}")
    if copied_argo_sample_locations_path is not None:
        print(
            "- combined ARGO sample locations GeoJSON: "
            f"{copied_argo_sample_locations_path}"
        )
    if copied_points_path is not None:
        print(f"- points GeoJSON: {copied_points_path}")
    if copied_patch_splits_path is not None:
        print(f"- patch splits GeoJSON: {copied_patch_splits_path}")
    if copied_full_sample_points_path is not None:
        print(f"- full-sample locations GeoJSON: {copied_full_sample_points_path}")
    if copied_graphs_dir_path is not None:
        print(f"- full-sample graphs: {copied_graphs_dir_path}")
    print(f"- config: {config_path}")
    print(
        "- fixed Celsius color scale: "
        f"[{DEFAULT_COLOR_SCALE_MIN_C:.1f}, {DEFAULT_COLOR_SCALE_MAX_C:.1f}]"
    )
    print(f"- color ramp: {color_ramp_path}")
    print(f"- extra zoom levels: {max(0, int(args.extra_zoom_levels))}")
    print("- tile resampling: nearest-neighbor")
    if args.rclone_remote is not None:
        sync_source, sync_scope_label = _resolve_rclone_sync_source(
            run_dir=run_dir,
            globe_dir=globe_dir,
            sync_scope=str(args.rclone_sync_scope),
        )
        ok, message = _sync_with_rclone(sync_source, args.rclone_remote)
        if ok:
            print(f"- rclone upload ({sync_scope_label}): {args.rclone_remote}")
            print(f"  source: {sync_source}")
            print(f"  {message}")
        else:
            print(f"WARNING: {message}")
            print(
                "WARNING: Globe assets were still created locally and can be uploaded later with:"
            )
            print(f"  rclone sync {sync_source} {args.rclone_remote}")


if __name__ == "__main__":
    main()
