# /work/envs/depth/bin/python -m depth_recon.inference.export_temporal_cesium_globe_assets --temperature-run-dir inference/outputs/temporal_variables_2018/runs/temperature/2018_W01 --salinity-run-dir inference/outputs/temporal_variables_2018/runs/salinity/2018_W01 --output-dir inference/outputs/temporal_variables_2018/temporal-globe --validation-year 2018 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/temporal-globe
"""Package weekly 10m temporal exports into a lightweight Cesium globe animation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import (  # noqa: E402
    DEFAULT_ABSOLUTE_ERROR_COLOR_PALETTE,
    DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
    DEFAULT_COLOR_RAMP_PATH,
    DEFAULT_EXTRA_ZOOM_LEVELS,
    DEFAULT_REPO_ROOT,
    DEFAULT_SALINITY_COLOR_RAMP_PATH,
    DEFAULT_TILE_DRIVER,
    _absolute_error_color_scale,
    _colorize_raster,
    _ensure_clean_directory,
    _export_base_map_tiles,
    _read_raster_metadata,
    _resolve_layer_url,
    _rounded_absolute_error_legend_max,
    _run_gdal2tiles,
    _run_variable_metadata,
    _sync_with_rclone,
    _validate_raster_transparency_contract,
    _write_absolute_error_color_ramp,
)
from depth_recon.inference.export_temporal_consistency_dashboard import (  # noqa: E402
    DEFAULT_TEMPORAL_VALIDATION_YEAR,
    _ordered_variable_items,
    _resolve_temporal_run,
)

DEFAULT_TEMPORAL_GLOBE_DIR_NAME = "temporal-globe"
DEFAULT_TEMPORAL_GLOBE_CONFIG_NAME = "temporal-globe-config.json"
DEFAULT_TEMPORAL_GLOBE_FRAME_INTERVAL_MS = 1000
DEFAULT_TEMPORAL_GLOBE_DEPTH_SUFFIX = "10m"
DEFAULT_TEMPORAL_GLOBE_WEBP_QUALITY = 80
DEFAULT_TEMPORAL_GLOBE_MAX_ZOOM_LEVEL = 4
DEFAULT_TEMPORAL_GLOBE_TILE_DRIVER = DEFAULT_TILE_DRIVER
DEFAULT_TEMPORAL_GLOBE_URL = (
    "https://globe-assets.hyperalislabs.com/inference_production/temporal-globe"
)
TEMPORAL_GLOBE_PAGE_FILES = {
    DEFAULT_REPO_ROOT / "docs" / "temporal-globe" / "index.html": Path("index.html"),
    DEFAULT_REPO_ROOT
    / "docs"
    / "javascripts"
    / "load-temporal-globe.js": Path("javascripts/load-temporal-globe.js"),
    DEFAULT_REPO_ROOT
    / "docs"
    / "javascripts"
    / "temporal-globe.js": Path("javascripts/temporal-globe.js"),
    DEFAULT_REPO_ROOT
    / "docs"
    / "stylesheets"
    / "globe.css": Path("stylesheets/globe.css"),
    DEFAULT_REPO_ROOT
    / "docs"
    / "assets"
    / "branding"
    / "website_icon.png": Path("assets/branding/website_icon.png"),
}


def _safe_frame_key(run: Any) -> str:
    """Return a stable folder key for one weekly frame."""
    if run.iso_year is not None and run.iso_week is not None:
        return f"{int(run.iso_year):04d}_W{int(run.iso_week):02d}"
    return str(run.selected_date)


def _frame_label(run: Any) -> str:
    """Return the compact label displayed by the temporal globe controls."""
    if run.iso_year is not None and run.iso_week is not None:
        return f"{int(run.iso_year):04d}-W{int(run.iso_week):02d}"
    return str(run.selected_date)


def _color_ramp_for_variable(variable_metadata: dict[str, Any]) -> Path:
    """Return the prediction color ramp matching the exported variable."""
    variable = str(variable_metadata["name"])
    if variable == "salinity":
        return DEFAULT_SALINITY_COLOR_RAMP_PATH
    return DEFAULT_COLOR_RAMP_PATH


def _resolve_temporal_globe_runs(run_dirs: Sequence[Path]) -> list[Any]:
    """Resolve and sort weekly temporal run directories for one variable."""
    runs = sorted(
        [_resolve_temporal_run(Path(run_dir)) for run_dir in run_dirs],
        key=lambda run: run.selected_date,
    )
    if not runs:
        raise ValueError(
            "At least one weekly run is required for temporal globe export."
        )
    variable = str(runs[0].variable_metadata["name"])
    for run in runs[1:]:
        if str(run.variable_metadata["name"]) != variable:
            raise ValueError(
                "All temporal globe runs in one series must export the same variable."
            )
        if (
            str(run.ten_meter_artifact.get("suffix"))
            != DEFAULT_TEMPORAL_GLOBE_DEPTH_SUFFIX
        ):
            raise ValueError("Temporal globe export requires retained 10m artifacts.")
    return runs


def _combined_error_scale(runs: Sequence[Any]) -> dict[str, float | int]:
    """Return one stable absolute-error color scale across all weekly frames."""
    scales = []
    for run in runs:
        error_path = Path(run.ten_meter_artifact["absolute_error_path"])
        scales.append(_absolute_error_color_scale(error_path))
    color_scale_max = max(float(scale["color_scale_max_c"]) for scale in scales)
    valid_max = max(float(scale["valid_max_c"]) for scale in scales)
    return {
        "color_scale_min_c": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
        "color_scale_max_c": color_scale_max,
        "valid_max_c": valid_max,
        "legend_min_c": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
        "legend_max_c": _rounded_absolute_error_legend_max(color_scale_max),
    }


def _tile_prediction_frame(
    *,
    input_path: Path,
    output_dir: Path,
    temp_dir: Path,
    color_ramp_path: Path,
    extra_zoom_levels: int,
    max_zoom_level: int | None,
    webp_quality: int,
    raster_edge_erosion_pixels: int,
    raster_edge_feather_pixels: int,
) -> None:
    """Colorize and tile one 10m prediction raster."""
    _validate_raster_transparency_contract(input_path)
    colorized_path = temp_dir / f"{output_dir.name}_colorized.tif"
    _colorize_raster(
        input_path,
        colorized_path,
        color_ramp_path=color_ramp_path,
        raster_edge_erosion_pixels=raster_edge_erosion_pixels,
        raster_edge_feather_pixels=raster_edge_feather_pixels,
    )
    _run_gdal2tiles(
        colorized_path,
        output_dir,
        extra_zoom_levels=extra_zoom_levels,
        max_zoom_level=max_zoom_level,
        webp_quality=webp_quality,
    )


def _tile_error_frame(
    *,
    input_path: Path,
    output_dir: Path,
    temp_dir: Path,
    error_ramp_path: Path,
    extra_zoom_levels: int,
    max_zoom_level: int | None,
    webp_quality: int,
    raster_edge_erosion_pixels: int,
    raster_edge_feather_pixels: int,
) -> None:
    """Colorize and tile one 10m absolute-error raster."""
    _validate_raster_transparency_contract(input_path)
    colorized_path = temp_dir / f"{output_dir.name}_colorized.tif"
    _colorize_raster(
        input_path,
        colorized_path,
        color_ramp_path=error_ramp_path,
        raster_edge_erosion_pixels=raster_edge_erosion_pixels,
        raster_edge_feather_pixels=raster_edge_feather_pixels,
    )
    _run_gdal2tiles(
        colorized_path,
        output_dir,
        extra_zoom_levels=extra_zoom_levels,
        max_zoom_level=max_zoom_level,
        webp_quality=webp_quality,
    )


def _copy_temporal_globe_page(output_dir: Path) -> None:
    """Copy the standalone temporal-globe viewer files into the hosted bundle."""
    for source_path, relative_path in TEMPORAL_GLOBE_PAGE_FILES.items():
        if not source_path.exists():
            continue
        destination_path = output_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".html":
            html = source_path.read_text(encoding="utf-8")
            # The docs page lives one directory below docs/, while the exported
            # bundle serves these files from its own root.
            html = html.replace("../stylesheets/globe.css", "stylesheets/globe.css")
            html = html.replace(
                "../javascripts/load-temporal-globe.js",
                "javascripts/load-temporal-globe.js",
            )
            html = html.replace(
                "../assets/branding/website_icon.png",
                "assets/branding/website_icon.png",
            )
            destination_path.write_text(html, encoding="utf-8")
        else:
            shutil.copy2(source_path, destination_path)


def _build_variable_frames(
    *,
    runs: Sequence[Any],
    output_dir: Path,
    temp_dir: Path,
    public_base_url: str | None,
    extra_zoom_levels: int,
    max_zoom_level: int | None,
    webp_quality: int,
    raster_edge_erosion_pixels: int,
    raster_edge_feather_pixels: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tile every weekly 10m frame for one variable and return manifest data."""
    variable_metadata = _run_variable_metadata(runs[0].run_summary)
    variable = str(variable_metadata["name"])
    color_ramp_path = _color_ramp_for_variable(variable_metadata)
    error_scale = _combined_error_scale(runs)
    error_ramp_path = temp_dir / f"{variable}_absolute_error_ramp.txt"
    _write_absolute_error_color_ramp(
        error_ramp_path,
        color_scale_min_c=float(error_scale["color_scale_min_c"]),
        color_scale_max_c=float(error_scale["color_scale_max_c"]),
        valid_max_c=float(error_scale["valid_max_c"]),
    )

    frames = []
    bounds_metadata = None
    for run in runs:
        frame_key = _safe_frame_key(run)
        artifact = run.ten_meter_artifact
        prediction_path = Path(artifact["prediction_path"])
        error_path = Path(artifact["absolute_error_path"])
        if bounds_metadata is None:
            bounds_metadata = _read_raster_metadata(prediction_path)

        frame_root = Path("frames") / variable / frame_key
        prediction_relative = frame_root / f"prediction_tiles_{artifact['suffix']}"
        error_relative = frame_root / f"absolute_error_tiles_{artifact['suffix']}"
        _tile_prediction_frame(
            input_path=prediction_path,
            output_dir=output_dir / prediction_relative,
            temp_dir=temp_dir,
            color_ramp_path=color_ramp_path,
            extra_zoom_levels=extra_zoom_levels,
            max_zoom_level=max_zoom_level,
            webp_quality=webp_quality,
            raster_edge_erosion_pixels=raster_edge_erosion_pixels,
            raster_edge_feather_pixels=raster_edge_feather_pixels,
        )
        _tile_error_frame(
            input_path=error_path,
            output_dir=output_dir / error_relative,
            temp_dir=temp_dir,
            error_ramp_path=error_ramp_path,
            extra_zoom_levels=extra_zoom_levels,
            max_zoom_level=max_zoom_level,
            webp_quality=webp_quality,
            raster_edge_erosion_pixels=raster_edge_erosion_pixels,
            raster_edge_feather_pixels=raster_edge_feather_pixels,
        )
        frames.append(
            {
                "label": _frame_label(run),
                "selected_date": int(run.selected_date),
                "iso_year": None if run.iso_year is None else int(run.iso_year),
                "iso_week": None if run.iso_week is None else int(run.iso_week),
                "prediction_tiles_url": _resolve_layer_url(
                    prediction_relative.as_posix(), public_base_url=public_base_url
                ),
                "absolute_error_tiles_url": _resolve_layer_url(
                    error_relative.as_posix(), public_base_url=public_base_url
                ),
            }
        )

    depth_level = dict(runs[0].ten_meter_artifact)
    depth_level.pop("prediction_path", None)
    depth_level.pop("absolute_error_path", None)
    variable_config = {
        "variable": variable,
        "variable_label": variable_metadata["label"],
        "value_units": variable_metadata["value_units"],
        "value_unit_label": variable_metadata["value_unit_label"],
        "color_scale_min": float(variable_metadata["color_scale_min"]),
        "color_scale_max": float(variable_metadata["color_scale_max"]),
        "color_palette": variable_metadata["color_palette"],
        "depth_level": depth_level,
        "frame_count": len(frames),
        "frames": frames,
        "absolute_error_color_palette": DEFAULT_ABSOLUTE_ERROR_COLOR_PALETTE,
        "absolute_error_value_units": variable_metadata["value_units"],
        "absolute_error_value_unit_label": variable_metadata["value_unit_label"],
        "absolute_error_color_scale_min": float(error_scale["color_scale_min_c"]),
        "absolute_error_color_scale_max": float(error_scale["color_scale_max_c"]),
        "absolute_error_legend_min": float(error_scale["legend_min_c"]),
        "absolute_error_legend_max": int(error_scale["legend_max_c"]),
    }
    return variable_config, bounds_metadata or {}


def _default_variable_key(variables: dict[str, Any]) -> str:
    """Return the preferred initial variable key for the temporal globe."""
    if "temperature" in variables:
        return "temperature"
    return next(iter(variables))


def export_temporal_cesium_globe_assets(
    *,
    variable_run_dirs: dict[str, Sequence[Path]],
    output_dir: Path,
    public_base_url: str | None = None,
    rclone_remote: str | None = None,
    copy_viewer: bool = True,
    validation_year: int = DEFAULT_TEMPORAL_VALIDATION_YEAR,
    frame_interval_ms: int = DEFAULT_TEMPORAL_GLOBE_FRAME_INTERVAL_MS,
    extra_zoom_levels: int = DEFAULT_EXTRA_ZOOM_LEVELS,
    max_zoom_level: int | None = DEFAULT_TEMPORAL_GLOBE_MAX_ZOOM_LEVEL,
    webp_quality: int = DEFAULT_TEMPORAL_GLOBE_WEBP_QUALITY,
    raster_edge_erosion_pixels: int = 2,
    raster_edge_feather_pixels: int = 4,
    include_base_map: bool = True,
) -> dict[str, Any]:
    """Write a lightweight temporal Cesium globe bundle from weekly 10m rasters."""
    output_dir = Path(output_dir).resolve()
    _ensure_clean_directory(output_dir)
    temp_dir = output_dir / ".tmp_colorized_rasters"
    _ensure_clean_directory(temp_dir)

    variables: dict[str, Any] = {}
    results: dict[str, Any] = {}
    bounds_metadata: dict[str, Any] | None = None
    for variable, raw_run_dirs in _ordered_variable_items(variable_run_dirs):
        run_dirs = [Path(run_dir) for run_dir in raw_run_dirs]
        if not run_dirs:
            continue
        runs = _resolve_temporal_globe_runs(run_dirs)
        variable_config, candidate_bounds = _build_variable_frames(
            runs=runs,
            output_dir=output_dir,
            temp_dir=temp_dir,
            public_base_url=public_base_url,
            extra_zoom_levels=extra_zoom_levels,
            max_zoom_level=max_zoom_level,
            webp_quality=webp_quality,
            raster_edge_erosion_pixels=max(0, int(raster_edge_erosion_pixels)),
            raster_edge_feather_pixels=max(0, int(raster_edge_feather_pixels)),
        )
        payload_variable = str(variable_config["variable"])
        if payload_variable != str(variable):
            raise ValueError(
                f"Variable key {variable!r} does not match run payload variable "
                f"{payload_variable!r}."
            )
        variables[variable] = variable_config
        results[variable] = {
            "frame_count": int(variable_config["frame_count"]),
            "depth_suffix": str(variable_config["depth_level"]["suffix"]),
        }
        if bounds_metadata is None:
            bounds_metadata = candidate_bounds

    if not variables:
        raise ValueError("No temporal variable run directories were provided.")

    base_map_tiles_dir = None
    base_map_tiles_url = None
    base_map_credit = None
    if include_base_map:
        base_map_tiles_dir, base_map_tiles_url, base_map_credit = (
            _export_base_map_tiles(
                output_dir,
                public_base_url=public_base_url,
            )
        )

    default_variable = _default_variable_key(variables)
    bounds_metadata = bounds_metadata or {}
    config = {
        "schema_version": 1,
        "title": "DepthDif Temporal Globe",
        "validation_year": int(validation_year),
        "depth_suffix": DEFAULT_TEMPORAL_GLOBE_DEPTH_SUFFIX,
        "default_variable": default_variable,
        "available_variables": list(variables.keys()),
        "default_layer": "prediction",
        "frame_interval_ms": int(frame_interval_ms),
        "tile_driver": DEFAULT_TEMPORAL_GLOBE_TILE_DRIVER,
        "webp_quality": int(webp_quality),
        "extra_zoom_levels": int(extra_zoom_levels),
        "max_zoom_level": None if max_zoom_level is None else int(max_zoom_level),
        "base_map_tiles_url": base_map_tiles_url,
        "base_map_credit": base_map_credit,
        "default_camera_destination": bounds_metadata.get("default_camera_destination"),
        "west": bounds_metadata.get("west", -180.0),
        "south": bounds_metadata.get("south", -90.0),
        "east": bounds_metadata.get("east", 180.0),
        "north": bounds_metadata.get("north", 90.0),
        "credits": {
            "prediction": "DepthDif weekly 10m prediction overlay",
            "absolute_error": "DepthDif weekly 10m absolute error overlay",
            "base_map": base_map_credit or "Natural Earth II",
        },
        "variables": variables,
    }
    config_path = output_dir / DEFAULT_TEMPORAL_GLOBE_CONFIG_NAME
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    if copy_viewer:
        _copy_temporal_globe_page(output_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

    summary = {
        "temporal_globe": {
            "schema_version": 1,
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "public_base_url": public_base_url,
            "default_variable": default_variable,
            "validation_year": int(validation_year),
            "variables": results,
            "base_map_tiles_dir": (
                None if base_map_tiles_dir is None else str(base_map_tiles_dir)
            ),
        }
    }
    summary_path = output_dir / "run_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    upload_ok: bool | None = None
    upload_message: str | None = None
    if rclone_remote is not None:
        ok, message = _sync_with_rclone(output_dir, rclone_remote)
        upload_ok = bool(ok)
        upload_message = str(message)

    return {
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "variables": list(variables.keys()),
        "default_variable": default_variable,
        "validation_year": int(validation_year),
        "frame_interval_ms": int(frame_interval_ms),
        "webp_quality": int(webp_quality),
        "max_zoom_level": None if max_zoom_level is None else int(max_zoom_level),
        "variable_results": results,
        "upload_requested": rclone_remote is not None,
        "upload_ok": upload_ok,
        "upload_message": upload_message,
        "upload_remote": rclone_remote,
    }


def _collect_variable_run_dirs(args: argparse.Namespace) -> dict[str, list[Path]]:
    """Collect CLI run-dir arguments into a variable-keyed mapping."""
    variable_run_dirs: dict[str, list[Path]] = {}
    if args.temperature_run_dir:
        variable_run_dirs["temperature"] = [
            Path(path) for path in args.temperature_run_dir
        ]
    if args.salinity_run_dir:
        variable_run_dirs["salinity"] = [Path(path) for path in args.salinity_run_dir]
    for run_dir in args.run_dir or []:
        run = _resolve_temporal_run(Path(run_dir))
        variable = str(run.variable_metadata["name"])
        variable_run_dirs.setdefault(variable, []).append(Path(run_dir))
    return variable_run_dirs


def _build_parser() -> argparse.ArgumentParser:
    """Build the temporal Cesium globe packaging CLI parser."""
    parser = argparse.ArgumentParser(
        description="Tile weekly 10m temporal rasters into a Cesium globe animation."
    )
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--temperature-run-dir", action="append", default=[])
    parser.add_argument("--salinity-run-dir", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where temporal globe assets will be written.",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=DEFAULT_TEMPORAL_VALIDATION_YEAR,
        help="Validation year represented by the temporal globe.",
    )
    parser.add_argument("--public-base-url", default=None)
    parser.add_argument("--rclone-remote", default=None)
    parser.add_argument(
        "--frame-interval-ms",
        type=int,
        default=DEFAULT_TEMPORAL_GLOBE_FRAME_INTERVAL_MS,
    )
    parser.add_argument(
        "--extra-zoom-levels",
        type=int,
        default=DEFAULT_EXTRA_ZOOM_LEVELS,
        help="Additional zoom levels above native resolution. Keep 0 for light bundles.",
    )
    parser.add_argument(
        "--max-zoom-level",
        type=int,
        default=DEFAULT_TEMPORAL_GLOBE_MAX_ZOOM_LEVEL,
        help="Maximum Cesium tile zoom level retained for weekly frames.",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=DEFAULT_TEMPORAL_GLOBE_WEBP_QUALITY,
        help="WebP quality passed to gdal2tiles for weekly frames.",
    )
    parser.add_argument("--raster-edge-erosion-pixels", type=int, default=2)
    parser.add_argument("--raster-edge-feather-pixels", type=int, default=4)
    parser.add_argument(
        "--base-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="include_base_map",
    )
    parser.add_argument(
        "--copy-viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="copy_viewer",
    )
    return parser


def main() -> None:
    """Run the temporal Cesium globe packaging CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    result = export_temporal_cesium_globe_assets(
        variable_run_dirs=_collect_variable_run_dirs(args),
        output_dir=args.output_dir,
        public_base_url=args.public_base_url,
        rclone_remote=args.rclone_remote,
        copy_viewer=bool(args.copy_viewer),
        validation_year=int(args.validation_year),
        frame_interval_ms=int(args.frame_interval_ms),
        extra_zoom_levels=int(args.extra_zoom_levels),
        max_zoom_level=args.max_zoom_level,
        webp_quality=int(args.webp_quality),
        raster_edge_erosion_pixels=int(args.raster_edge_erosion_pixels),
        raster_edge_feather_pixels=int(args.raster_edge_feather_pixels),
        include_base_map=bool(args.include_base_map),
    )
    print(f"Wrote temporal globe assets to: {result['output_dir']}")
    print(f"- config: {result['config_path']}")
    print(f"- variables: {', '.join(result['variables'])}")


if __name__ == "__main__":
    main()
