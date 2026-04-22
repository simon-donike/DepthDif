"""Convert one global inference export run into Cesium-hostable assets.

Typical CLI:
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_20150615 \
  --public-base-url https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/inference_production/globe \
  --rclone-remote r2:depth-data/inference_production \
  --rclone-sync-scope run


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
DEFAULT_CAMERA_LON = -38.56452881619089
DEFAULT_CAMERA_LAT = 34.53988238358822
DEFAULT_CAMERA_HEIGHT = 9_500_000.0


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
) -> tuple[Path, Path | None, Path | None, Path | None, dict[str, Any]]:
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

    return (
        prediction_path,
        ground_truth_path,
        points_path,
        patch_splits_path,
        run_summary,
    )


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

        # One extra level reduces visible overzoom artifacts while keeping the
        # tile count close to the native pyramid size.
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


def _resolve_layer_url(name: str, *, public_base_url: str | None) -> str:
    if public_base_url is None:
        return f"./{name}"
    return f"{public_base_url.rstrip('/')}/{name}"


def build_globe_config(
    *,
    selected_date: int | None,
    prediction_tiles_url: str,
    ground_truth_tiles_url: str | None,
    argo_points_url: str | None,
    patch_splits_url: str | None,
    bounds: dict[str, Any],
    prediction_credit: str,
    ground_truth_credit: str | None,
    points_credit: str | None,
    patch_splits_credit: str | None,
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
            "argo_points_url": argo_points_url,
            "patch_splits_url": patch_splits_url,
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
            "Optional rclone destination such as 'r2:depth-data/inference_production'. "
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

    prediction_path, ground_truth_path, points_path, patch_splits_path, run_summary = (
        _resolve_run_artifacts(run_dir)
    )

    globe_dir = run_dir / str(args.globe_dir_name)
    globe_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = globe_dir / ".tmp_colorized_rasters"
    _ensure_clean_directory(temp_dir)
    color_ramp_path = Path(args.color_ramp_path)
    if not color_ramp_path.exists():
        raise FileNotFoundError(f"Color ramp not found: {color_ramp_path}")

    prediction_colorized_path = temp_dir / f"{prediction_path.stem}_colorized.tif"
    _colorize_raster(
        prediction_path,
        prediction_colorized_path,
        color_ramp_path=color_ramp_path,
    )
    prediction_tiles_dir = globe_dir / "prediction_tiles"
    _run_gdal2tiles(
        prediction_colorized_path,
        prediction_tiles_dir,
        extra_zoom_levels=args.extra_zoom_levels,
    )

    ground_truth_tiles_dir: Path | None = None
    if ground_truth_path is not None:
        ground_truth_colorized_path = (
            temp_dir / f"{ground_truth_path.stem}_colorized.tif"
        )
        _colorize_raster(
            ground_truth_path,
            ground_truth_colorized_path,
            color_ramp_path=color_ramp_path,
        )
        ground_truth_tiles_dir = globe_dir / "ground_truth_tiles"
        _run_gdal2tiles(
            ground_truth_colorized_path,
            ground_truth_tiles_dir,
            extra_zoom_levels=args.extra_zoom_levels,
        )

    copied_points_path: Path | None = None
    if points_path is not None:
        copied_points_path = globe_dir / "argo_points.geojson"
        shutil.copy2(points_path, copied_points_path)

    copied_patch_splits_path: Path | None = None
    if patch_splits_path is not None:
        copied_patch_splits_path = globe_dir / "patch_splits.geojson"
        shutil.copy2(patch_splits_path, copied_patch_splits_path)

    prediction_meta = _read_raster_metadata(prediction_path)
    ground_truth_meta = (
        _read_raster_metadata(ground_truth_path)
        if ground_truth_path is not None
        else None
    )
    template = _load_template(Path(args.template_path))
    config = build_globe_config(
        selected_date=run_summary.get("selected_date"),
        prediction_tiles_url=_resolve_layer_url(
            "prediction_tiles",
            public_base_url=args.public_base_url,
        ),
        ground_truth_tiles_url=(
            None
            if ground_truth_tiles_dir is None
            else _resolve_layer_url(
                "ground_truth_tiles",
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
        bounds=prediction_meta,
        prediction_credit=prediction_meta["credit"],
        ground_truth_credit=(
            None if ground_truth_meta is None else ground_truth_meta["credit"]
        ),
        points_credit=None if copied_points_path is None else "Observed Argo points",
        patch_splits_credit=(
            None if copied_patch_splits_path is None else "Train/val patch split grid"
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
    print(f"- prediction tiles: {prediction_tiles_dir}")
    if ground_truth_tiles_dir is not None:
        print(f"- ground-truth tiles: {ground_truth_tiles_dir}")
    if copied_points_path is not None:
        print(f"- points GeoJSON: {copied_points_path}")
    if copied_patch_splits_path is not None:
        print(f"- patch splits GeoJSON: {copied_patch_splits_path}")
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
