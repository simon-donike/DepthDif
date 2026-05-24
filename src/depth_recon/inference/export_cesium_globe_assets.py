"""Convert one global inference export run into Cesium-hostable assets.

Typical CLI:
/work/envs/depth/bin/python -m depth_recon.inference.export_cesium_globe_assets \
  --run-dir src/depth_recon/inference/outputs/global_top_band_20150615 \
  --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe \
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

import numpy as np
import rasterio
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "transforms" / "globe-config.template.json"
)
DEFAULT_COLOR_RAMP_PATH = (
    Path(__file__).resolve().parent / "transforms" / "temperature_blue_red_ramp.txt"
)
DEFAULT_SALINITY_COLOR_RAMP_PATH = (
    Path(__file__).resolve().parent / "transforms" / "salinity_blue_green_ramp.txt"
)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRANSPARENT_ALPHA = 0
DEFAULT_OPAQUE_ALPHA = 255
DEFAULT_COLOR_SCALE_MIN_C = 0.0
DEFAULT_COLOR_SCALE_MAX_C = 30.0
DEFAULT_SALINITY_COLOR_SCALE_MIN = 30.0
DEFAULT_SALINITY_COLOR_SCALE_MAX = 40.0
DEFAULT_ABSOLUTE_ERROR_SCALE_MIN_PERCENTILE = 2.0
DEFAULT_ABSOLUTE_ERROR_SCALE_MAX_PERCENTILE = 98.0
DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C = 0.0
DEFAULT_ABSOLUTE_ERROR_COLOR_PALETTE = "absolute_error_green_red"
DEFAULT_EXTRA_ZOOM_LEVELS = 0
DEFAULT_RCLONE_SYNC_SCOPE = "globe"
DEFAULT_ERROR_ANALYSIS_JSON_NAME = "error-analysis.json"
DEFAULT_ANALYSIS_GRID_GEOJSON_NAME = "analysis-grid.geojson"
DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_DRIVER = "WEBP"
DEFAULT_WEBP_QUALITY = 95
DEFAULT_BASE_MAP_RASTER_PATH = (
    DEFAULT_REPO_ROOT / "src/depth_recon/data/local_data/NE2_LR_LC_SR_W_DR.tif"
)
DEFAULT_BASE_MAP_TILES_PATH = Path("basemaps") / "natural_earth_ii_webp_q95"
DEFAULT_BASE_MAP_CREDIT = "Natural Earth II"
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


VARIABLE_GLOBE_DEFAULTS: dict[str, dict[str, Any]] = {
    "temperature": {
        "label": "Temperature",
        "value_units": "degree_Celsius",
        "value_unit_label": "°C",
        "color_scale_min": DEFAULT_COLOR_SCALE_MIN_C,
        "color_scale_max": DEFAULT_COLOR_SCALE_MAX_C,
        "color_palette": "temperature_blue_red",
        "color_ramp_path": DEFAULT_COLOR_RAMP_PATH,
    },
    "salinity": {
        "label": "Salinity",
        "value_units": "PSU",
        "value_unit_label": "PSU",
        "color_scale_min": DEFAULT_SALINITY_COLOR_SCALE_MIN,
        "color_scale_max": DEFAULT_SALINITY_COLOR_SCALE_MAX,
        "color_palette": "salinity_blue_green",
        "color_ramp_path": DEFAULT_SALINITY_COLOR_RAMP_PATH,
    },
}


def _run_variable_metadata(run_summary: dict[str, Any]) -> dict[str, Any]:
    """Return display and color metadata for a single exported variable run."""
    variable = str(run_summary.get("variable", "temperature")).strip().lower()
    defaults = VARIABLE_GLOBE_DEFAULTS.get(
        variable, VARIABLE_GLOBE_DEFAULTS["temperature"]
    )
    return {
        "name": variable,
        "label": str(run_summary.get("variable_label", defaults["label"])),
        "value_units": str(run_summary.get("value_units", defaults["value_units"])),
        "value_unit_label": str(
            run_summary.get("value_unit_label", defaults["value_unit_label"])
        ),
        "color_scale_min": float(
            run_summary.get("color_scale_min", defaults["color_scale_min"])
        ),
        "color_scale_max": float(
            run_summary.get("color_scale_max", defaults["color_scale_max"])
        ),
        "color_palette": str(
            run_summary.get("color_palette", defaults["color_palette"])
        ),
        "color_ramp_path": Path(defaults["color_ramp_path"]),
    }


def _surface_depth_export(
    *,
    prediction_path: Path,
    ground_truth_path: Path | None,
    absolute_error_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "suffix": "surface",
        "label": "Surface",
        "requested_depth_m": 0.0,
        "actual_depth_m": 0.0,
        "channel_index": 0,
        "prediction_path": prediction_path,
        "ground_truth_path": ground_truth_path,
        "absolute_error_path": absolute_error_path,
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
        run_relative = run_dir / candidate
        if run_relative.exists():
            return run_relative
        repo_relative = DEFAULT_REPO_ROOT / candidate
        if repo_relative.exists():
            return repo_relative
        run_relative = run_dir / candidate.name
        if run_relative.exists():
            return run_relative
    return None


def _copy_precomputed_error_analysis_json(
    *,
    run_dir: Path,
    globe_dir: Path,
    run_summary: dict[str, Any],
) -> Path | None:
    """Copy inference-generated error analysis JSON into the globe bundle."""
    path_value = run_summary.get("error_analysis_json_path")
    if not isinstance(path_value, str) or path_value.strip() == "":
        return None

    source_path = _coerce_existing_path(path_value, run_dir=run_dir)
    if source_path is None:
        return None

    globe_dir.mkdir(parents=True, exist_ok=True)
    output_path = globe_dir / DEFAULT_ERROR_ANALYSIS_JSON_NAME
    if source_path.resolve() != output_path.resolve():
        shutil.copy2(source_path, output_path)
    return output_path


def _copy_precomputed_analysis_grid_geojson(
    *,
    run_dir: Path,
    globe_dir: Path,
    run_summary: dict[str, Any],
) -> Path | None:
    """Copy inference-generated coast-clipped analysis grid GeoJSON."""
    path_value = run_summary.get("error_analysis_grid_geojson_path")
    if not isinstance(path_value, str) or path_value.strip() == "":
        return None

    source_path = _coerce_existing_path(path_value, run_dir=run_dir)
    if source_path is None:
        return None

    globe_dir.mkdir(parents=True, exist_ok=True)
    output_path = globe_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
    if source_path.resolve() != output_path.resolve():
        shutil.copy2(source_path, output_path)
    return output_path


def _error_analysis_grid_size(error_analysis_json_path: Path | None) -> float | None:
    """Read the dashboard grid size from an error-analysis payload."""
    if error_analysis_json_path is None or not Path(error_analysis_json_path).exists():
        return None
    with Path(error_analysis_json_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    grouping = payload.get("grouping", {})
    if not isinstance(grouping, dict):
        return None
    value = grouping.get("grid_size_degrees")
    if value is None:
        return None
    return float(value)


def _write_analysis_grid_geojson_from_mask(
    *,
    globe_dir: Path,
    land_mask_path: Path | None,
    error_analysis_json_path: Path | None,
) -> Path | None:
    """Create coast-clipped analysis grid geometry from the run land mask."""
    if land_mask_path is None:
        return None

    from depth_recon.inference.export_error_analysis_dashboard import (
        DEFAULT_GRID_SIZE_DEGREES,
        write_analysis_grid_geojson,
    )

    grid_size = _error_analysis_grid_size(error_analysis_json_path)
    globe_dir.mkdir(parents=True, exist_ok=True)
    return write_analysis_grid_geojson(
        output_path=globe_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
        land_mask_path=land_mask_path,
        grid_size_degrees=(
            DEFAULT_GRID_SIZE_DEGREES if grid_size is None else float(grid_size)
        ),
    )


def _resolve_land_mask_path(
    run_summary: dict[str, Any],
    *,
    run_dir: Path,
) -> Path | None:
    """Resolve the land-mask path recorded by the inference export."""
    path_value = run_summary.get("land_mask_path")
    inference_grid = run_summary.get("inference_grid")
    if path_value is None and isinstance(inference_grid, dict):
        path_value = inference_grid.get("land_mask_path")
    return _coerce_existing_path(path_value, run_dir=run_dir)


def _resolve_run_artifacts(
    run_dir: Path,
) -> tuple[
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    dict[str, Any],
]:
    run_summary_path = run_dir / "run_summary.yaml"
    run_summary = _load_yaml(run_summary_path) if run_summary_path.exists() else {}

    prediction_path = _coerce_existing_path(
        run_summary.get("prediction_tif_path"),
        run_dir=run_dir,
    )
    if prediction_path is None:
        matches = sorted(run_dir.glob("*_prediction.tif"))
        prediction_path = matches[0] if matches else None

    ground_truth_path = _coerce_existing_path(
        run_summary.get("ground_truth_tif_path"),
        run_dir=run_dir,
    )
    if ground_truth_path is None:
        matches = sorted(run_dir.glob("*_glorys_top_band.tif"))
        ground_truth_path = matches[0] if matches else None

    absolute_error_path = _coerce_existing_path(
        run_summary.get("absolute_error_tif_path"),
        run_dir=run_dir,
    )

    uncertainty_path = _coerce_existing_path(
        run_summary.get("uncertainty_tif_path"),
        run_dir=run_dir,
    )
    if uncertainty_path is None:
        matches = sorted(run_dir.glob("*_uncertainty.tif"))
        uncertainty_path = matches[0] if matches else None
    if prediction_path is None and uncertainty_path is None:
        raise FileNotFoundError(
            "Could not locate prediction or uncertainty GeoTIFFs. Expected "
            "run_summary.yaml, '*_prediction.tif', or '*_uncertainty.tif' inside "
            "the run directory."
        )

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
    if full_sample_points_path is None:
        candidate = run_dir / "globe" / "full_sample_locations.geojson"
        full_sample_points_path = candidate if candidate.exists() else None

    graphs_dir_path = _coerce_existing_path(
        run_summary.get("graphs_dir_path"),
        run_dir=run_dir,
    )
    if graphs_dir_path is None:
        candidate = run_dir / "graphs"
        graphs_dir_path = candidate if candidate.exists() else None
    if graphs_dir_path is None:
        candidate = run_dir / "globe" / "graphs"
        graphs_dir_path = candidate if candidate.exists() else None

    return (
        prediction_path,
        ground_truth_path,
        absolute_error_path,
        points_path,
        patch_splits_path,
        full_sample_points_path,
        graphs_dir_path,
        uncertainty_path,
        run_summary,
    )


def _resolve_depth_export_artifacts(
    *,
    run_dir: Path,
    run_summary: dict[str, Any],
    prediction_path: Path | None,
    ground_truth_path: Path | None,
) -> list[dict[str, Any]]:
    if prediction_path is None:
        return []
    raw_exports = run_summary.get("depth_exports")
    if not isinstance(raw_exports, list) or not raw_exports:
        return [
            _surface_depth_export(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                absolute_error_path=_coerce_existing_path(
                    run_summary.get("absolute_error_tif_path"),
                    run_dir=run_dir,
                ),
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
        absolute_error_export_path = _coerce_existing_path(
            raw_export.get("absolute_error_tif_path"),
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
                "absolute_error_path": absolute_error_export_path,
            }
        )

    if not depth_exports:
        return [
            _surface_depth_export(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                absolute_error_path=_coerce_existing_path(
                    run_summary.get("absolute_error_tif_path"),
                    run_dir=run_dir,
                ),
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
    """Colorize one single-band raster and make nodata/land pixels transparent."""
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
    _apply_alpha_mask_to_colorized_raster(input_path, output_path)


def _valid_raster_values(path: Path) -> np.ndarray:
    """Read finite, non-nodata values from a single-band raster."""
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=False)
        valid_mask = np.isfinite(data)
        if ds.nodata is not None and np.isfinite(float(ds.nodata)):
            valid_mask &= ~np.isclose(data, float(ds.nodata), atol=0.0, rtol=0.0)
    return data[valid_mask].astype(np.float64, copy=False)


def _rounded_absolute_error_legend_max(value: float) -> int:
    """Round an absolute-error legend maximum to the nearest display integer."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return 0
    return int(max(1, round(value)))


def _absolute_error_color_scale(path: Path) -> dict[str, float | int]:
    """Compute robust color-scale metadata for an absolute-error raster."""
    values = _valid_raster_values(path)
    if values.size == 0:
        return {
            "valid_min_c": 0.0,
            "valid_max_c": 0.0,
            "color_scale_min_c": 0.0,
            "color_scale_max_c": 1.0,
            "legend_min_c": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
            "legend_max_c": 0,
        }

    valid_min = float(np.min(values))
    valid_max = float(np.max(values))
    scale_min = float(
        np.percentile(values, DEFAULT_ABSOLUTE_ERROR_SCALE_MIN_PERCENTILE)
    )
    scale_max = float(
        np.percentile(values, DEFAULT_ABSOLUTE_ERROR_SCALE_MAX_PERCENTILE)
    )
    scale_min = max(0.0, scale_min)
    if not np.isfinite(scale_max) or scale_max <= scale_min:
        scale_max = max(float(valid_max), scale_min + 1.0)
    if scale_max <= scale_min:
        scale_max = scale_min + 1.0
    return {
        "valid_min_c": valid_min,
        "valid_max_c": valid_max,
        "color_scale_min_c": scale_min,
        "color_scale_max_c": float(scale_max),
        "legend_min_c": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
        "legend_max_c": _rounded_absolute_error_legend_max(scale_max),
    }


def _write_absolute_error_color_ramp(
    output_path: Path,
    *,
    color_scale_min_c: float,
    color_scale_max_c: float,
    valid_max_c: float,
) -> None:
    """Write a GDAL color-relief ramp for absolute-error visualization."""
    scale_min = max(0.0, float(color_scale_min_c))
    scale_max = max(scale_min + 1.0e-6, float(color_scale_max_c))
    valid_max = max(scale_max, float(valid_max_c))
    stops: list[tuple[float, tuple[int, int, int]]] = [
        (0.0, (34, 197, 94)),
    ]
    if scale_min > 0.0:
        # Values below the lower percentile stay fully green, while the middle
        # range is stretched between the requested robust percentiles.
        stops.append((scale_min, (34, 197, 94)))
    stops.extend(
        [
            ((scale_min + scale_max) / 2.0, (250, 204, 21)),
            (scale_max, (220, 38, 38)),
        ]
    )
    if valid_max > scale_max:
        stops.append((valid_max, (220, 38, 38)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    previous_value: float | None = None
    for value, rgb in sorted(stops, key=lambda item: item[0]):
        if previous_value is not None and np.isclose(value, previous_value):
            continue
        previous_value = float(value)
        lines.append(f"{float(value):.6f}    {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}")
    lines.append("nv   0 0 0 0")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _transparent_pixel_mask(
    input_path: Path,
) -> np.ndarray:
    """Build a boolean mask for pixels that should be transparent in Cesium."""
    with rasterio.open(input_path) as ds:
        data = ds.read(1, masked=False)
        transparent = ~np.isfinite(data)
        if ds.nodata is not None:
            transparent |= np.isclose(data, float(ds.nodata))
    return transparent


def _apply_alpha_mask_to_colorized_raster(
    input_path: Path,
    output_path: Path,
    *,
    transparent: np.ndarray | None = None,
) -> None:
    """Apply transparent alpha to colorized nodata and land pixels."""
    if transparent is None:
        transparent = _transparent_pixel_mask(input_path)
    with rasterio.open(output_path, "r+") as ds:
        if ds.count < 4:
            raise RuntimeError(
                f"Colorized raster must have an alpha band: {output_path}"
            )

        alpha = ds.read(4)
        opaque_value = (
            DEFAULT_OPAQUE_ALPHA if np.issubdtype(alpha.dtype, np.integer) else 1.0
        )
        alpha[transparent] = DEFAULT_TRANSPARENT_ALPHA
        alpha[~transparent] = opaque_value
        ds.write(alpha, 4)

        # Also clear RGB behind transparent pixels to avoid colored tile edges.
        for band_index in range(1, min(3, ds.count) + 1):
            band = ds.read(band_index)
            band[transparent] = 0
            ds.write(band, band_index)


def _validate_raster_transparency_contract(path: Path) -> None:
    """Require rasters to use their nodata value for land-masked pixels."""
    with rasterio.open(path) as ds:
        tags = ds.tags()
    land_zeroed = str(tags.get("land_zeroed", "")).strip().lower() == "true"
    land_masked_to_nodata = (
        str(tags.get("land_masked_to_nodata", "")).strip().lower() == "true"
    )
    if land_zeroed and not land_masked_to_nodata:
        raise RuntimeError(
            "This GeoTIFF uses the old land-mask convention with land pixels set "
            f"to 0.0: {path}. Re-run inference/export or repair the TIFF so land "
            "pixels use the GeoTIFF nodata value before packaging Cesium assets."
        )


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
    resampling: str = "near",
    tile_driver: str = DEFAULT_TILE_DRIVER,
    webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> list[str]:
    gdal2tiles_exe = shutil.which("gdal2tiles.py")
    if gdal2tiles_exe is None:
        raise RuntimeError("gdal2tiles.py was not found on PATH.")

    max_zoom = _estimate_native_zoom_level(input_path) + max(0, int(extra_zoom_levels))
    command = [
        gdal2tiles_exe,
        "-p",
        "mercator",
        "-r",
        str(resampling),
        "-z",
        f"0-{max_zoom}",
        "-w",
        "none",
    ]
    if tile_driver:
        command.append(f"--tiledriver={tile_driver}")
        if str(tile_driver).upper() == "WEBP":
            command.append(f"--webp-quality={int(webp_quality)}")
    command.extend([str(input_path), str(output_dir)])
    return command


def _remove_gdal_auxiliary_files(output_dir: Path) -> int:
    """Delete GDAL sidecar metadata files that are not needed for hosted tiles."""
    removed = 0
    for path in Path(output_dir).rglob("*.aux.xml"):
        path.unlink()
        removed += 1
    return removed


def _run_gdal2tiles(
    input_path: Path,
    output_dir: Path,
    *,
    extra_zoom_levels: int,
    resampling: str = "near",
    tile_driver: str = DEFAULT_TILE_DRIVER,
    webp_quality: int = DEFAULT_WEBP_QUALITY,
) -> None:
    _ensure_clean_directory(output_dir)
    command = _build_gdal2tiles_command(
        input_path,
        output_dir,
        extra_zoom_levels=extra_zoom_levels,
        resampling=resampling,
        tile_driver=tile_driver,
        webp_quality=webp_quality,
    )
    subprocess.run(command, check=True)
    _remove_gdal_auxiliary_files(output_dir)


def _export_base_map_tiles(
    globe_dir: Path,
    *,
    public_base_url: str | None,
    base_map_raster_path: Path = DEFAULT_BASE_MAP_RASTER_PATH,
) -> tuple[Path | None, str | None, str | None]:
    """Tile the optional hosted basemap and return its local path and config URL."""
    base_map_raster_path = Path(base_map_raster_path)
    if not base_map_raster_path.exists():
        return None, None, None

    output_dir = Path(globe_dir) / DEFAULT_BASE_MAP_TILES_PATH
    _run_gdal2tiles(
        base_map_raster_path,
        output_dir,
        extra_zoom_levels=0,
        resampling="bilinear",
    )
    return (
        output_dir,
        _resolve_layer_url(
            DEFAULT_BASE_MAP_TILES_PATH.as_posix(),
            public_base_url=public_base_url,
        ),
        DEFAULT_BASE_MAP_CREDIT,
    )


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
        return [_round_geojson_coordinates(item, decimals=decimals) for item in value]
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


def _feature_identity(
    feature: dict[str, Any], *, coordinate_precision: int
) -> tuple[Any, ...]:
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


def _raster_transparency_config(
    raster_path: Path,
    *,
    land_mask_path: Path | None,
    color_scale_min: float = DEFAULT_COLOR_SCALE_MIN_C,
    color_scale_max: float = DEFAULT_COLOR_SCALE_MAX_C,
    value_unit_label: str = "°C",
) -> dict[str, Any]:
    """Describe how raster pixels are converted to transparent globe tiles."""
    with rasterio.open(raster_path) as ds:
        nodata_value = None if ds.nodata is None else float(ds.nodata)
        tags = ds.tags()
    land_masked_to_nodata = (
        str(tags.get("land_masked_to_nodata", "")).strip().lower() == "true"
    )
    land_mask_applied_value = None
    land_mask_mode = "none"
    if land_masked_to_nodata:
        land_mask_applied_value = nodata_value
        land_mask_mode = "nodata"
    return {
        "nodata_value": nodata_value,
        "nodata_alpha": DEFAULT_TRANSPARENT_ALPHA,
        "land_mask_path": None if land_mask_path is None else str(land_mask_path),
        "land_mask_applied_value": land_mask_applied_value,
        "land_mask_mode": land_mask_mode,
        "land_mask_alpha": DEFAULT_TRANSPARENT_ALPHA,
        "valid_alpha": DEFAULT_OPAQUE_ALPHA,
        "color_scale_min": float(color_scale_min),
        "color_scale_max": float(color_scale_max),
        "color_scale_min_c": float(color_scale_min),
        "color_scale_max_c": float(color_scale_max),
        "note": (
            "Land-masked source pixels use the GeoTIFF nodata value before "
            f"color relief so valid 0 {value_unit_label} ocean values remain visible "
            "when present."
        ),
    }


def build_globe_config(
    *,
    selected_date: int | None,
    target_date: int | None,
    iso_year: int | None,
    iso_week: int | None,
    prediction_tiles_url: str | None,
    ground_truth_tiles_url: str | None,
    absolute_error_tiles_url: str | None,
    depth_levels: list[dict[str, Any]],
    argo_sample_locations_url: str | None,
    argo_points_url: str | None,
    patch_splits_url: str | None,
    full_sample_points_url: str | None,
    bounds: dict[str, Any],
    prediction_credit: str,
    ground_truth_credit: str | None,
    absolute_error_credit: str | None,
    points_credit: str | None,
    patch_splits_credit: str | None,
    full_sample_points_credit: str | None,
    color_scale_min_c: float,
    color_scale_max_c: float,
    color_palette: str,
    raster_transparency: dict[str, Any],
    template: dict[str, Any],
    variable: str = "temperature",
    variable_label: str = "Temperature",
    value_units: str = "degree_Celsius",
    value_unit_label: str = "°C",
    variables: dict[str, Any] | None = None,
    default_variable: str | None = None,
    uncertainty_tiles_url: str | None = None,
    uncertainty_credit: str | None = None,
    uncertainty_color_palette: str = DEFAULT_ABSOLUTE_ERROR_COLOR_PALETTE,
    uncertainty_color_scale_min: float | None = None,
    uncertainty_color_scale_max: float | None = None,
    uncertainty_legend_min: float = DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
    uncertainty_legend_max: int | None = None,
    uncertainty_value_units: str | None = None,
    uncertainty_value_unit_label: str | None = None,
    base_map_tiles_url: str | None = None,
    base_map_credit: str | None = None,
    error_analysis_data_url: str | None = None,
    analysis_grid_geojson_url: str | None = None,
) -> dict[str, Any]:
    config = dict(template)
    uncertainty_units = (
        value_units if uncertainty_value_units is None else uncertainty_value_units
    )
    uncertainty_unit_label = (
        value_unit_label
        if uncertainty_value_unit_label is None
        else uncertainty_value_unit_label
    )
    config.update(
        {
            "selected_date": selected_date,
            "target_date": target_date,
            "iso_year": iso_year,
            "iso_week": iso_week,
            "variable": str(variable),
            "variable_label": str(variable_label),
            "value_units": str(value_units),
            "value_unit_label": str(value_unit_label),
            "prediction_tiles_url": prediction_tiles_url,
            "ground_truth_tiles_url": ground_truth_tiles_url,
            "absolute_error_tiles_url": absolute_error_tiles_url,
            "uncertainty_tiles_url": uncertainty_tiles_url,
            "uncertainty_color_palette": str(uncertainty_color_palette),
            "uncertainty_value_units": str(uncertainty_units),
            "uncertainty_value_unit_label": str(uncertainty_unit_label),
            "uncertainty_color_scale_min": (
                None
                if uncertainty_color_scale_min is None
                else float(uncertainty_color_scale_min)
            ),
            "uncertainty_color_scale_max": (
                None
                if uncertainty_color_scale_max is None
                else float(uncertainty_color_scale_max)
            ),
            "uncertainty_legend_min": float(uncertainty_legend_min),
            "uncertainty_legend_max": (
                None if uncertainty_legend_max is None else int(uncertainty_legend_max)
            ),
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
            "color_scale_min": float(color_scale_min_c),
            "color_scale_max": float(color_scale_max_c),
            "color_scale_min_c": float(color_scale_min_c),
            "color_scale_max_c": float(color_scale_max_c),
            "color_palette": str(color_palette),
            "raster_transparency": dict(raster_transparency),
        }
    )
    if base_map_tiles_url is not None:
        config["base_map_tiles_url"] = str(base_map_tiles_url)
    else:
        config.pop("base_map_tiles_url", None)
    if base_map_credit is not None:
        config["base_map_credit"] = str(base_map_credit)
    else:
        config.pop("base_map_credit", None)
    if variables is not None:
        config["variables"] = dict(variables)
        config["default_variable"] = (
            str(default_variable) if default_variable is not None else str(variable)
        )
    config.pop("error_analysis_url", None)
    if error_analysis_data_url is not None:
        config["error_analysis_data_url"] = str(error_analysis_data_url)
    else:
        config.pop("error_analysis_data_url", None)
    if analysis_grid_geojson_url is not None:
        config["analysis_grid_geojson_url"] = str(analysis_grid_geojson_url)
    else:
        config.pop("analysis_grid_geojson_url", None)
    credits = dict(config.get("credits", {}))
    credits["prediction"] = prediction_credit
    if ground_truth_credit is not None:
        credits["ground_truth"] = ground_truth_credit
    if absolute_error_credit is not None:
        credits["absolute_error"] = absolute_error_credit
    if uncertainty_credit is not None:
        credits["uncertainty"] = uncertainty_credit
    if points_credit is not None:
        credits["points"] = points_credit
    if patch_splits_credit is not None:
        credits["patch_splits"] = patch_splits_credit
    if full_sample_points_credit is not None:
        credits["full_sample_points"] = full_sample_points_credit
    if base_map_credit is not None:
        credits["base_map"] = str(base_map_credit)
    else:
        credits.pop("base_map", None)
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
    parser.add_argument(
        "--error-analysis",
        "--include-error-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="include_error_analysis",
        help=(
            "Copy or generate error-analysis.json into the globe directory and add "
            "the dashboard data URL to globe-config.json."
        ),
    )
    return parser


def export_cesium_globe_assets(
    *,
    run_dir: Path,
    public_base_url: str | None = None,
    globe_dir_name: str = "globe",
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    color_ramp_path: Path = DEFAULT_COLOR_RAMP_PATH,
    rclone_remote: str | None = None,
    rclone_sync_scope: str = DEFAULT_RCLONE_SYNC_SCOPE,
    extra_zoom_levels: int = DEFAULT_EXTRA_ZOOM_LEVELS,
    include_base_map: bool = True,
    include_error_analysis: bool = True,
) -> dict[str, Any]:
    """Build Cesium globe assets for one global inference run and optionally upload."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    (
        prediction_path,
        ground_truth_path,
        absolute_error_path,
        points_path,
        patch_splits_path,
        full_sample_points_path,
        graphs_dir_path,
        uncertainty_path,
        run_summary,
    ) = _resolve_run_artifacts(run_dir)

    variable_metadata = _run_variable_metadata(run_summary)
    globe_dir = run_dir / str(globe_dir_name)
    globe_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = globe_dir / ".tmp_colorized_rasters"
    _ensure_clean_directory(temp_dir)
    color_ramp_path = Path(color_ramp_path)
    if (
        variable_metadata["name"] != "temperature"
        and color_ramp_path == DEFAULT_COLOR_RAMP_PATH
    ):
        color_ramp_path = Path(variable_metadata["color_ramp_path"])
    if not color_ramp_path.exists():
        raise FileNotFoundError(f"Color ramp not found: {color_ramp_path}")
    land_mask_path = _resolve_land_mask_path(run_summary, run_dir=run_dir)

    depth_exports = _resolve_depth_export_artifacts(
        run_dir=run_dir,
        run_summary=run_summary,
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
    )
    config_depth_levels: list[dict[str, Any]] = []
    prediction_tiles_dir: Path | None = None
    ground_truth_tiles_dir: Path | None = None
    absolute_error_tiles_dir: Path | None = None
    uncertainty_tiles_dir: Path | None = None
    uncertainty_scale: dict[str, float | int] | None = None
    for depth_export in depth_exports:
        suffix = str(depth_export["suffix"])
        prediction_export_path = Path(depth_export["prediction_path"])
        _validate_raster_transparency_contract(prediction_export_path)
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
            extra_zoom_levels=extra_zoom_levels,
        )
        if prediction_tiles_dir is None:
            prediction_tiles_dir = prediction_tiles_dir_for_depth

        ground_truth_tiles_dir_for_depth: Path | None = None
        ground_truth_export_path = depth_export.get("ground_truth_path")
        if ground_truth_export_path is not None:
            ground_truth_export_path = Path(ground_truth_export_path)
            _validate_raster_transparency_contract(ground_truth_export_path)
            ground_truth_colorized_path = (
                temp_dir / f"{ground_truth_export_path.stem}_colorized.tif"
            )
            _colorize_raster(
                ground_truth_export_path,
                ground_truth_colorized_path,
                color_ramp_path=color_ramp_path,
            )
            ground_truth_tiles_dir_for_depth = (
                globe_dir / f"ground_truth_tiles_{suffix}"
            )
            _run_gdal2tiles(
                ground_truth_colorized_path,
                ground_truth_tiles_dir_for_depth,
                extra_zoom_levels=extra_zoom_levels,
            )
            if ground_truth_tiles_dir is None:
                ground_truth_tiles_dir = ground_truth_tiles_dir_for_depth

        absolute_error_tiles_dir_for_depth: Path | None = None
        absolute_error_scale: dict[str, float | int] | None = None
        absolute_error_export_path = depth_export.get("absolute_error_path")
        if absolute_error_export_path is not None:
            absolute_error_export_path = Path(absolute_error_export_path)
            _validate_raster_transparency_contract(absolute_error_export_path)
            absolute_error_scale = _absolute_error_color_scale(
                absolute_error_export_path
            )
            absolute_error_ramp_path = (
                temp_dir / f"{absolute_error_export_path.stem}_green_red_ramp.txt"
            )
            _write_absolute_error_color_ramp(
                absolute_error_ramp_path,
                color_scale_min_c=float(absolute_error_scale["color_scale_min_c"]),
                color_scale_max_c=float(absolute_error_scale["color_scale_max_c"]),
                valid_max_c=float(absolute_error_scale["valid_max_c"]),
            )
            absolute_error_colorized_path = (
                temp_dir / f"{absolute_error_export_path.stem}_colorized.tif"
            )
            _colorize_raster(
                absolute_error_export_path,
                absolute_error_colorized_path,
                color_ramp_path=absolute_error_ramp_path,
            )
            absolute_error_tiles_dir_for_depth = (
                globe_dir / f"absolute_error_tiles_{suffix}"
            )
            _run_gdal2tiles(
                absolute_error_colorized_path,
                absolute_error_tiles_dir_for_depth,
                extra_zoom_levels=extra_zoom_levels,
            )
            if absolute_error_tiles_dir is None:
                absolute_error_tiles_dir = absolute_error_tiles_dir_for_depth

        config_depth_levels.append(
            {
                "suffix": suffix,
                "label": str(depth_export["label"]),
                "requested_depth_m": float(depth_export["requested_depth_m"]),
                "actual_depth_m": float(depth_export["actual_depth_m"]),
                "channel_index": int(depth_export["channel_index"]),
                "variable": str(variable_metadata["name"]),
                "variable_label": str(variable_metadata["label"]),
                "value_units": str(variable_metadata["value_units"]),
                "value_unit_label": str(variable_metadata["value_unit_label"]),
                "color_scale_min": float(variable_metadata["color_scale_min"]),
                "color_scale_max": float(variable_metadata["color_scale_max"]),
                "color_scale_min_c": float(variable_metadata["color_scale_min"]),
                "color_scale_max_c": float(variable_metadata["color_scale_max"]),
                "prediction_tiles_url": _resolve_layer_url(
                    prediction_tiles_dir_for_depth.name,
                    public_base_url=public_base_url,
                ),
                "ground_truth_tiles_url": (
                    None
                    if ground_truth_tiles_dir_for_depth is None
                    else _resolve_layer_url(
                        ground_truth_tiles_dir_for_depth.name,
                        public_base_url=public_base_url,
                    )
                ),
                "absolute_error_tiles_url": (
                    None
                    if absolute_error_tiles_dir_for_depth is None
                    else _resolve_layer_url(
                        absolute_error_tiles_dir_for_depth.name,
                        public_base_url=public_base_url,
                    )
                ),
                "absolute_error_color_palette": DEFAULT_ABSOLUTE_ERROR_COLOR_PALETTE,
                "absolute_error_value_units": str(variable_metadata["value_units"]),
                "absolute_error_value_unit_label": str(
                    variable_metadata["value_unit_label"]
                ),
                "absolute_error_color_scale_min": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["color_scale_min_c"])
                ),
                "absolute_error_color_scale_max": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["color_scale_max_c"])
                ),
                "absolute_error_color_scale_min_c": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["color_scale_min_c"])
                ),
                "absolute_error_color_scale_max_c": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["color_scale_max_c"])
                ),
                "absolute_error_legend_min": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
                "absolute_error_legend_max": (
                    None
                    if absolute_error_scale is None
                    else int(absolute_error_scale["legend_max_c"])
                ),
                "absolute_error_legend_min_c": DEFAULT_ABSOLUTE_ERROR_LEGEND_MIN_C,
                "absolute_error_legend_max_c": (
                    None
                    if absolute_error_scale is None
                    else int(absolute_error_scale["legend_max_c"])
                ),
                "absolute_error_valid_max": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["valid_max_c"])
                ),
                "absolute_error_valid_max_c": (
                    None
                    if absolute_error_scale is None
                    else float(absolute_error_scale["valid_max_c"])
                ),
                "absolute_error_scale_min_percentile": (
                    DEFAULT_ABSOLUTE_ERROR_SCALE_MIN_PERCENTILE
                ),
                "absolute_error_scale_max_percentile": (
                    DEFAULT_ABSOLUTE_ERROR_SCALE_MAX_PERCENTILE
                ),
            }
        )

    if uncertainty_path is not None:
        _validate_raster_transparency_contract(uncertainty_path)
        uncertainty_scale = _absolute_error_color_scale(uncertainty_path)
        uncertainty_ramp_path = temp_dir / f"{uncertainty_path.stem}_green_red_ramp.txt"
        _write_absolute_error_color_ramp(
            uncertainty_ramp_path,
            color_scale_min_c=float(uncertainty_scale["color_scale_min_c"]),
            color_scale_max_c=float(uncertainty_scale["color_scale_max_c"]),
            valid_max_c=float(uncertainty_scale["valid_max_c"]),
        )
        uncertainty_colorized_path = temp_dir / f"{uncertainty_path.stem}_colorized.tif"
        _colorize_raster(
            uncertainty_path,
            uncertainty_colorized_path,
            color_ramp_path=uncertainty_ramp_path,
        )
        uncertainty_tiles_dir = globe_dir / "uncertainty_tiles"
        _run_gdal2tiles(
            uncertainty_colorized_path,
            uncertainty_tiles_dir,
            extra_zoom_levels=extra_zoom_levels,
        )

    if prediction_tiles_dir is None and uncertainty_tiles_dir is None:
        raise RuntimeError("No prediction or uncertainty tiles were generated.")

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
        if graphs_dir_path.resolve() != copied_graphs_dir_path.resolve():
            if copied_graphs_dir_path.exists():
                shutil.rmtree(copied_graphs_dir_path)
            shutil.copytree(graphs_dir_path, copied_graphs_dir_path)

    base_map_tiles_dir: Path | None = None
    base_map_tiles_url: str | None = None
    base_map_credit: str | None = None
    if include_base_map:
        base_map_tiles_dir, base_map_tiles_url, base_map_credit = (
            _export_base_map_tiles(
                globe_dir,
                public_base_url=public_base_url,
            )
        )

    copied_error_analysis_json_path: Path | None = None
    copied_analysis_grid_geojson_path: Path | None = None
    if include_error_analysis:
        copied_error_analysis_json_path = _copy_precomputed_error_analysis_json(
            run_dir=run_dir,
            globe_dir=globe_dir,
            run_summary=run_summary,
        )
        copied_analysis_grid_geojson_path = _copy_precomputed_analysis_grid_geojson(
            run_dir=run_dir,
            globe_dir=globe_dir,
            run_summary=run_summary,
        )
    if include_error_analysis and copied_error_analysis_json_path is None:
        from depth_recon.inference.export_error_analysis_dashboard import (
            export_error_analysis_dashboard,
        )

        error_analysis_result = export_error_analysis_dashboard(
            run_dir=run_dir,
            output_dir=globe_dir,
            public_base_url=public_base_url,
        )
        copied_error_analysis_json_path = Path(error_analysis_result["json_path"])
        if error_analysis_result.get("grid_geojson_path") is not None:
            copied_analysis_grid_geojson_path = Path(
                error_analysis_result["grid_geojson_path"]
            )
    if include_error_analysis and copied_analysis_grid_geojson_path is None:
        copied_analysis_grid_geojson_path = _write_analysis_grid_geojson_from_mask(
            globe_dir=globe_dir,
            land_mask_path=land_mask_path,
            error_analysis_json_path=copied_error_analysis_json_path,
        )

    bounds_source_path = (
        prediction_path if prediction_path is not None else uncertainty_path
    )
    if bounds_source_path is None:
        raise RuntimeError("No raster was available for globe bounds metadata.")
    prediction_meta = _read_raster_metadata(bounds_source_path)
    ground_truth_meta = (
        _read_raster_metadata(ground_truth_path)
        if ground_truth_path is not None
        else None
    )
    absolute_error_meta = (
        _read_raster_metadata(absolute_error_path)
        if absolute_error_path is not None
        else None
    )
    uncertainty_meta = (
        _read_raster_metadata(uncertainty_path)
        if uncertainty_path is not None
        else None
    )
    raster_transparency = _raster_transparency_config(
        bounds_source_path,
        land_mask_path=land_mask_path,
        color_scale_min=float(variable_metadata["color_scale_min"]),
        color_scale_max=float(variable_metadata["color_scale_max"]),
        value_unit_label=str(variable_metadata["value_unit_label"]),
    )
    template = _load_template(Path(template_path))
    config = build_globe_config(
        selected_date=run_summary.get("selected_date"),
        target_date=run_summary.get("target_date", run_summary.get("selected_date")),
        iso_year=run_summary.get("iso_year"),
        iso_week=run_summary.get("iso_week"),
        prediction_tiles_url=(
            None
            if not config_depth_levels
            else str(config_depth_levels[0]["prediction_tiles_url"])
        ),
        ground_truth_tiles_url=(
            None
            if ground_truth_tiles_dir is None or not config_depth_levels
            else config_depth_levels[0]["ground_truth_tiles_url"]
        ),
        absolute_error_tiles_url=(
            None
            if absolute_error_tiles_dir is None or not config_depth_levels
            else config_depth_levels[0]["absolute_error_tiles_url"]
        ),
        depth_levels=config_depth_levels,
        argo_sample_locations_url=(
            None
            if copied_argo_sample_locations_path is None
            else _resolve_layer_url(
                copied_argo_sample_locations_path.name,
                public_base_url=public_base_url,
            )
        ),
        argo_points_url=(
            None
            if copied_points_path is None
            else _resolve_layer_url(
                copied_points_path.name,
                public_base_url=public_base_url,
            )
        ),
        patch_splits_url=(
            None
            if copied_patch_splits_path is None
            else _resolve_layer_url(
                copied_patch_splits_path.name,
                public_base_url=public_base_url,
            )
        ),
        full_sample_points_url=(
            None
            if copied_full_sample_points_path is None
            else _resolve_layer_url(
                copied_full_sample_points_path.name,
                public_base_url=public_base_url,
            )
        ),
        bounds=prediction_meta,
        prediction_credit=prediction_meta["credit"],
        ground_truth_credit=(
            None if ground_truth_meta is None else ground_truth_meta["credit"]
        ),
        absolute_error_credit=(
            None if absolute_error_meta is None else absolute_error_meta["credit"]
        ),
        uncertainty_credit=(
            None if uncertainty_meta is None else uncertainty_meta["credit"]
        ),
        points_credit=None if copied_points_path is None else "Observed Argo points",
        patch_splits_credit=(
            None if copied_patch_splits_path is None else "Inference patch grid"
        ),
        full_sample_points_credit=(
            None
            if copied_full_sample_points_path is None
            else "Random full-depth profile locations"
        ),
        color_scale_min_c=float(variable_metadata["color_scale_min"]),
        color_scale_max_c=float(variable_metadata["color_scale_max"]),
        color_palette=str(variable_metadata["color_palette"]),
        raster_transparency=raster_transparency,
        template=template,
        variable=str(variable_metadata["name"]),
        variable_label=str(variable_metadata["label"]),
        value_units=str(variable_metadata["value_units"]),
        value_unit_label=str(variable_metadata["value_unit_label"]),
        uncertainty_tiles_url=(
            None
            if uncertainty_tiles_dir is None
            else _resolve_layer_url(
                uncertainty_tiles_dir.name,
                public_base_url=public_base_url,
            )
        ),
        uncertainty_color_scale_min=(
            None
            if uncertainty_scale is None
            else float(uncertainty_scale["color_scale_min_c"])
        ),
        uncertainty_color_scale_max=(
            None
            if uncertainty_scale is None
            else float(uncertainty_scale["color_scale_max_c"])
        ),
        uncertainty_legend_max=(
            None
            if uncertainty_scale is None
            else int(uncertainty_scale["legend_max_c"])
        ),
        base_map_tiles_url=base_map_tiles_url,
        base_map_credit=base_map_credit,
        error_analysis_data_url=(
            None
            if copied_error_analysis_json_path is None
            else _resolve_layer_url(
                copied_error_analysis_json_path.name,
                public_base_url=public_base_url,
            )
        ),
        analysis_grid_geojson_url=(
            None
            if copied_analysis_grid_geojson_path is None
            else _resolve_layer_url(
                copied_analysis_grid_geojson_path.name,
                public_base_url=public_base_url,
            )
        ),
    )
    config_path = globe_dir / "globe-config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Wrote globe assets to: {globe_dir}")
    print(f"- prediction depth tile sets: {len(config_depth_levels)}")
    if prediction_tiles_dir is not None:
        print(f"- first prediction tiles: {prediction_tiles_dir}")
    if ground_truth_tiles_dir is not None:
        print(f"- first ground-truth tiles: {ground_truth_tiles_dir}")
    if absolute_error_tiles_dir is not None:
        print(f"- first absolute-error tiles: {absolute_error_tiles_dir}")
    if uncertainty_tiles_dir is not None:
        print(f"- uncertainty tiles: {uncertainty_tiles_dir}")
    if base_map_tiles_dir is not None:
        print(f"- hosted base map tiles: {base_map_tiles_dir}")
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
    if copied_error_analysis_json_path is not None:
        print(f"- error analysis data: {copied_error_analysis_json_path}")
    if copied_analysis_grid_geojson_path is not None:
        print(f"- analysis ocean grid: {copied_analysis_grid_geojson_path}")
    print(f"- config: {config_path}")
    print(
        f"- fixed {variable_metadata['label']} color scale: "
        f"[{float(variable_metadata['color_scale_min']):.1f}, "
        f"{float(variable_metadata['color_scale_max']):.1f}] "
        f"{variable_metadata['value_unit_label']}"
    )
    print(f"- color ramp: {color_ramp_path}")
    print(f"- transparent land mask: {land_mask_path}")
    print(f"- extra zoom levels: {max(0, int(extra_zoom_levels))}")
    print(f"- tile format: {DEFAULT_TILE_DRIVER} q{DEFAULT_WEBP_QUALITY}")
    print("- tile resampling: nearest-neighbor")
    upload_ok: bool | None = None
    upload_message: str | None = None
    upload_source: Path | None = None
    if rclone_remote is not None:
        sync_source, sync_scope_label = _resolve_rclone_sync_source(
            run_dir=run_dir,
            globe_dir=globe_dir,
            sync_scope=str(rclone_sync_scope),
        )
        upload_source = sync_source
        ok, message = _sync_with_rclone(sync_source, rclone_remote)
        upload_ok = bool(ok)
        upload_message = str(message)
        if ok:
            print(f"- rclone upload ({sync_scope_label}): {rclone_remote}")
            print(f"  source: {sync_source}")
            print(f"  {message}")
        else:
            print(f"WARNING: {message}")
            print(
                "WARNING: Globe assets were still created locally and can be uploaded later with:"
            )
            print(f"  rclone sync {sync_source} {rclone_remote}")

    return {
        "globe_dir": str(globe_dir),
        "config_path": str(config_path),
        "variable": str(variable_metadata["name"]),
        "depth_tile_set_count": int(len(config_depth_levels)),
        "absolute_error_tile_set_count": int(
            sum(
                1
                for depth_level in config_depth_levels
                if depth_level.get("absolute_error_tiles_url") is not None
            )
        ),
        "uncertainty_tile_set_count": int(uncertainty_tiles_dir is not None),
        "base_map_tile_set_count": int(base_map_tiles_dir is not None),
        "upload_requested": rclone_remote is not None,
        "upload_ok": upload_ok,
        "upload_message": upload_message,
        "upload_remote": rclone_remote,
        "upload_source": None if upload_source is None else str(upload_source),
    }


ASSET_URL_KEYS = (
    "prediction_tiles_url",
    "ground_truth_tiles_url",
    "absolute_error_tiles_url",
    "uncertainty_tiles_url",
    "argo_sample_locations_url",
    "argo_points_url",
    "patch_splits_url",
    "full_sample_points_url",
    "analysis_grid_geojson_url",
)


def _is_absolute_asset_url(asset_url: str) -> bool:
    """Return True when an asset URL should not be rewritten as relative."""
    lowered = str(asset_url).strip().lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("data:")
    )


def _prefix_variable_asset_url(
    asset_url: str | None,
    *,
    variable: str,
    public_base_url: str | None,
) -> str | None:
    """Prefix a single-variable asset URL for the combined globe config."""
    if asset_url is None:
        return None
    raw = str(asset_url).strip()
    if raw == "" or _is_absolute_asset_url(raw):
        return raw
    clean = raw[2:] if raw.startswith("./") else raw
    clean = clean.lstrip("/")
    if public_base_url is not None:
        return f"{public_base_url.rstrip('/')}/{variable}/{clean}"
    return f"{variable}/{clean}"


def _prefix_variable_config_asset_urls(
    variable_config: dict[str, Any],
    *,
    variable: str,
    public_base_url: str | None,
) -> dict[str, Any]:
    """Rewrite one variable config so URLs resolve from the combined config."""
    rewritten = dict(variable_config)
    for key in ASSET_URL_KEYS + ("error_analysis_data_url",):
        rewritten[key] = _prefix_variable_asset_url(
            rewritten.get(key),
            variable=variable,
            public_base_url=public_base_url,
        )
    depth_levels = []
    for depth_level in rewritten.get("depth_levels", []):
        if not isinstance(depth_level, dict):
            continue
        rewritten_depth = dict(depth_level)
        for key in ASSET_URL_KEYS[:3]:
            rewritten_depth[key] = _prefix_variable_asset_url(
                rewritten_depth.get(key),
                variable=variable,
                public_base_url=public_base_url,
            )
        depth_levels.append(rewritten_depth)
    rewritten["depth_levels"] = depth_levels
    return rewritten


def _prefix_geojson_graph_paths(
    geojson_path: Path,
    *,
    variable: str,
    public_base_url: str | None,
) -> None:
    """Rewrite graph PNG paths so popups work from the combined config URL."""
    if not geojson_path.exists():
        return
    with geojson_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    changed = False
    for feature in payload.get("features", []):
        properties = feature.get("properties")
        if not isinstance(properties, dict) or not properties.get("graph_png_path"):
            continue
        properties["graph_png_path"] = _prefix_variable_asset_url(
            str(properties["graph_png_path"]),
            variable=variable,
            public_base_url=public_base_url,
        )
        changed = True
    if changed:
        with geojson_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.write("\n")


def _ordered_variable_items(
    variable_run_dirs: dict[str, Path],
) -> list[tuple[str, Path]]:
    """Return variable run directories in a stable viewer order."""
    items = {
        str(key).strip().lower(): Path(value)
        for key, value in variable_run_dirs.items()
    }
    ordered: list[tuple[str, Path]] = []
    for key in ("temperature", "salinity"):
        if key in items:
            ordered.append((key, items.pop(key)))
    ordered.extend(sorted(items.items(), key=lambda item: item[0]))
    return ordered


def export_cesium_globe_variable_assets(
    *,
    variable_run_dirs: dict[str, Path],
    globe_dir: Path,
    public_base_url: str | None = None,
    rclone_remote: str | None = None,
    rclone_sync_scope: str = DEFAULT_RCLONE_SYNC_SCOPE,
    extra_zoom_levels: int = DEFAULT_EXTRA_ZOOM_LEVELS,
) -> dict[str, Any]:
    """Build one combined Cesium globe bundle from per-variable export runs."""
    ordered_variables = _ordered_variable_items(variable_run_dirs)
    if not ordered_variables:
        raise ValueError("At least one variable run directory is required.")

    globe_dir = Path(globe_dir).resolve()
    _ensure_clean_directory(globe_dir)
    variables: dict[str, Any] = {}
    single_results: dict[str, Any] = {}
    for variable, run_dir in ordered_variables:
        run_dir = Path(run_dir).resolve()
        single_result = export_cesium_globe_assets(
            run_dir=run_dir,
            public_base_url=None,
            globe_dir_name="globe",
            rclone_remote=None,
            rclone_sync_scope=DEFAULT_RCLONE_SYNC_SCOPE,
            extra_zoom_levels=extra_zoom_levels,
            include_base_map=False,
            include_error_analysis=True,
        )
        source_globe_dir = Path(str(single_result["globe_dir"])).resolve()
        variable_globe_dir = globe_dir / variable
        if variable_globe_dir.exists():
            shutil.rmtree(variable_globe_dir)
        if source_globe_dir != variable_globe_dir:
            shutil.move(str(source_globe_dir), str(variable_globe_dir))
        for geojson_name in (
            "argo_sample_locations.geojson",
            "full_sample_locations.geojson",
        ):
            _prefix_geojson_graph_paths(
                variable_globe_dir / geojson_name,
                variable=variable,
                public_base_url=public_base_url,
            )
        with (variable_globe_dir / "globe-config.json").open(
            "r", encoding="utf-8"
        ) as f:
            variable_config = json.load(f)
        variables[variable] = _prefix_variable_config_asset_urls(
            variable_config,
            variable=variable,
            public_base_url=public_base_url,
        )
        single_results[variable] = single_result

    default_variable = (
        "temperature" if "temperature" in variables else next(iter(variables))
    )
    base_map_tiles_dir, base_map_tiles_url, base_map_credit = _export_base_map_tiles(
        globe_dir,
        public_base_url=public_base_url,
    )
    combined_config = dict(variables[default_variable])
    if base_map_tiles_url is not None:
        combined_config["base_map_tiles_url"] = base_map_tiles_url
    if base_map_credit is not None:
        combined_config["base_map_credit"] = base_map_credit
        combined_credits = dict(combined_config.get("credits", {}))
        combined_credits["base_map"] = base_map_credit
        combined_config["credits"] = combined_credits
    combined_config["variables"] = variables
    combined_config["default_variable"] = default_variable
    combined_config["available_variables"] = list(variables.keys())
    default_analysis_source = globe_dir / default_variable / "error-analysis.json"
    if default_analysis_source.exists():
        default_analysis_path = globe_dir / "error-analysis.json"
        shutil.copy2(default_analysis_source, default_analysis_path)
        combined_config["error_analysis_data_url"] = _resolve_layer_url(
            default_analysis_path.name,
            public_base_url=public_base_url,
        )
    default_analysis_grid_source = (
        globe_dir / default_variable / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
    )
    if default_analysis_grid_source.exists():
        default_analysis_grid_path = globe_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
        shutil.copy2(default_analysis_grid_source, default_analysis_grid_path)
        combined_config["analysis_grid_geojson_url"] = _resolve_layer_url(
            default_analysis_grid_path.name,
            public_base_url=public_base_url,
        )
    combined_config_path = globe_dir / "globe-config.json"
    with combined_config_path.open("w", encoding="utf-8") as f:
        json.dump(combined_config, f, indent=2)
        f.write("\n")

    upload_ok: bool | None = None
    upload_message: str | None = None
    upload_source: Path | None = None
    if rclone_remote is not None:
        if str(rclone_sync_scope) == "run":
            upload_source = globe_dir.parent
            sync_scope_label = "combined run directory"
        else:
            upload_source = globe_dir
            sync_scope_label = "combined globe assets"
        ok, message = _sync_with_rclone(upload_source, rclone_remote)
        upload_ok = bool(ok)
        upload_message = str(message)
        if ok:
            print(f"- rclone upload ({sync_scope_label}): {rclone_remote}")
            print(f"  source: {upload_source}")
            print(f"  {message}")
        else:
            print(f"WARNING: {message}")
            print("WARNING: Combined globe assets were still created locally.")

    print(f"Wrote combined globe config: {combined_config_path}")
    print(f"- variables: {', '.join(variables.keys())}")
    if base_map_tiles_dir is not None:
        print(f"- hosted base map tiles: {base_map_tiles_dir}")
    return {
        "globe_dir": str(globe_dir),
        "config_path": str(combined_config_path),
        "variables": list(variables.keys()),
        "default_variable": default_variable,
        "single_results": single_results,
        "base_map_tile_set_count": int(base_map_tiles_dir is not None),
        "upload_requested": rclone_remote is not None,
        "upload_ok": upload_ok,
        "upload_message": upload_message,
        "upload_remote": rclone_remote,
        "upload_source": None if upload_source is None else str(upload_source),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    export_cesium_globe_assets(
        run_dir=args.run_dir,
        public_base_url=args.public_base_url,
        globe_dir_name=args.globe_dir_name,
        template_path=args.template_path,
        color_ramp_path=args.color_ramp_path,
        rclone_remote=args.rclone_remote,
        rclone_sync_scope=args.rclone_sync_scope,
        extra_zoom_levels=args.extra_zoom_levels,
        include_error_analysis=bool(args.include_error_analysis),
    )


if __name__ == "__main__":
    main()
