# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_wavenumber_spectra \
#   --run-dir inference/outputs/global_variables_2018_W25_v2 \
#   --include-temporal-runs \
#   --variables temperature salinity \
#   --output-dir inference/outputs/global_variables_2018_W25_v2/wavenumber_spectra \
#   --min-wavelength-km 30 \
#   --max-wavelength-km 1000 \
#   --wavelength-bin-count 32 \
#   --basin-overlap-threshold 0.30 \
#   --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe/wavenumber_spectra \
#   --rclone-remote r2:depth-data/inference_production/globe/wavenumber_spectra
# Optional switches for non-default behavior: --require-complete-patches --no-plots --no-dashboard
"""Export 2D wavenumber spectra from existing inference GeoTIFF runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from functools import lru_cache
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import yaml
from shapely.geometry import box, shape

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import _sync_with_rclone
from depth_recon.inference.export_error_analysis_dashboard import (
    world_ocean_region_geojson_features,
)
from depth_recon.utils.normalizations import CELSIUS_TO_KELVIN_OFFSET

DEFAULT_OUTPUT_DIR_NAME = "wavenumber_spectra"
DEFAULT_MIN_WAVELENGTH_KM = 30.0
DEFAULT_MAX_WAVELENGTH_KM = 1000.0
DEFAULT_WAVELENGTH_BIN_COUNT = 32
DEFAULT_BASIN_OVERLAP_THRESHOLD = 0.30
SPECTRAL_DASHBOARD_CONFIG_NAME = "spectral-config.json"
SPECTRAL_DASHBOARD_BASIN_MAP_NAME = "basin-map.geojson"
SPECTRAL_DASHBOARD_BASIN_DIR_NAME = "basins"
ALL_OCEANS_BASIN = "All Oceans"
SOURCE_LAYER_BY_VARIABLE = {
    "temperature": ("ostia", "analysed_sst", "analysed_sst"),
    "salinity": ("sss", "sos", "sos"),
}
GLORYS_SOURCE_BY_VARIABLE = {
    "temperature": ("thetao", "thetao"),
    "salinity": ("so", "so"),
}
LAYER_LABELS = {
    "prediction": "Prediction",
    "glorys": "GLORYS",
    "surface_observation": "OSTIA",
}
LAYER_LINESTYLES = {
    "prediction": "-",
    "glorys": ":",
    "surface_observation": "--",
}
EARTH_KM_PER_DEGREE = 111.32


@dataclass(frozen=True)
class VariableRun:
    """Resolved single-variable inference run metadata."""

    variable: str
    run_dir: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class DepthLayerSpec:
    """Raster paths and metadata for one exported depth layer."""

    variable: str
    layer: str
    suffix: str
    label: str
    requested_depth_m: float
    actual_depth_m: float
    channel_index: int
    path: Path
    band_index: int = 1
    decode_stretched_uint8: bool = False
    source_kind: str = "exported"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}.")
    return payload


def _resolve_run_artifact_path(run_dir: Path, raw_path: Any) -> Path | None:
    """Resolve a run-summary artifact path relative to ``run_dir``."""
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text)
    return path if path.is_absolute() else Path(run_dir) / path


def _resolve_referenced_path(root: Path, raw_path: Any) -> Path:
    """Resolve a summary-referenced path from absolute, cwd, or root-relative text."""
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    return Path(root) / path


def _date_parts(date_value: int) -> tuple[int, int, str]:
    """Return calendar year, month, and meteorological season for YYYYMMDD."""
    text = str(int(date_value))
    parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    season = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }[int(parsed.month)]
    return int(parsed.year), int(parsed.month), season


def wavelength_bin_edges_km(
    *,
    min_wavelength_km: float = DEFAULT_MIN_WAVELENGTH_KM,
    max_wavelength_km: float = DEFAULT_MAX_WAVELENGTH_KM,
    bin_count: int = DEFAULT_WAVELENGTH_BIN_COUNT,
) -> np.ndarray:
    """Return ascending logarithmic wavelength-bin edges in kilometers."""
    min_value = float(min_wavelength_km)
    max_value = float(max_wavelength_km)
    count = int(bin_count)
    if min_value <= 0.0 or max_value <= min_value:
        raise ValueError("Wavelength range must be positive and increasing.")
    if count < 1:
        raise ValueError("bin_count must be >= 1.")
    return np.geomspace(min_value, max_value, count + 1).astype(np.float64)


def wavelength_bin_centers_km(edges: np.ndarray) -> np.ndarray:
    """Return geometric centers for wavelength bins."""
    edge_values = np.asarray(edges, dtype=np.float64)
    if edge_values.ndim != 1 or edge_values.size < 2:
        raise ValueError("edges must be a 1D array with at least two values.")
    return np.sqrt(edge_values[:-1] * edge_values[1:])


def detrend_plane_2d(values: np.ndarray) -> np.ndarray:
    """Remove a least-squares fitted plane from a 2D array."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {tuple(data.shape)}.")
    rows, cols = np.indices(data.shape, dtype=np.float64)
    finite = np.isfinite(data)
    if int(np.count_nonzero(finite)) < 3:
        return np.full(data.shape, np.nan, dtype=np.float64)
    design = np.column_stack(
        [
            cols[finite].reshape(-1),
            rows[finite].reshape(-1),
            np.ones(np.count_nonzero(finite)),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, data[finite].reshape(-1), rcond=None)
    trend = coeffs[0] * cols + coeffs[1] * rows + coeffs[2]
    out = data - trend
    out[~finite] = np.nan
    return out


def hann_window_2d(shape_2d: tuple[int, int]) -> np.ndarray:
    """Return a separable 2D Hann window for ``shape_2d``."""
    height, width = int(shape_2d[0]), int(shape_2d[1])
    if height < 2 or width < 2:
        raise ValueError("Hann-window dimensions must both be >= 2.")
    return (np.hanning(height)[:, None] * np.hanning(width)[None, :]).astype(
        np.float64,
        copy=False,
    )


def radial_wavenumber_spectrum(
    values: np.ndarray,
    *,
    pixel_size_x_km: float,
    pixel_size_y_km: float,
    wavelength_edges_km: np.ndarray,
    require_complete: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return wavelength-binned 2D FFT power for one patch.

    The input is detrended, Hann-windowed, transformed with ``fft2``, and binned
    by isotropic wavelength. Power is normalized by Hann-window energy so
    spectra from same-sized patches remain comparable.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D patch, got {tuple(data.shape)}.")
    if require_complete and not np.all(np.isfinite(data)):
        return None
    if not np.any(np.isfinite(data)):
        return None
    if float(pixel_size_x_km) <= 0.0 or float(pixel_size_y_km) <= 0.0:
        return None

    detrended = detrend_plane_2d(data)
    finite_detrended = np.isfinite(detrended)
    if require_complete and not np.all(finite_detrended):
        return None
    if not np.any(finite_detrended):
        return None
    if not np.all(finite_detrended):
        # Missing pixels have no fitted residual; zero-fill them after detrending
        # so prediction and GLORYS spectra use the same land/coast support.
        detrended = np.where(finite_detrended, detrended, 0.0)
    window = hann_window_2d(detrended.shape)
    windowed = detrended * window
    transform = np.fft.fft2(windowed)
    power = (np.abs(transform) ** 2) / max(
        float(np.sum(window**2)), np.finfo(float).eps
    )

    height, width = detrended.shape
    kx = np.fft.fftfreq(width, d=float(pixel_size_x_km))
    ky = np.fft.fftfreq(height, d=float(pixel_size_y_km))
    radial_k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    valid_k = radial_k > 0.0
    wavelengths = np.full(radial_k.shape, np.nan, dtype=np.float64)
    wavelengths[valid_k] = 1.0 / radial_k[valid_k]

    edges = np.asarray(wavelength_edges_km, dtype=np.float64)
    spectrum = np.full(edges.size - 1, np.nan, dtype=np.float64)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    for bin_idx in range(edges.size - 1):
        in_bin = (wavelengths >= edges[bin_idx]) & (wavelengths < edges[bin_idx + 1])
        if not np.any(in_bin):
            continue
        counts[bin_idx] = int(np.count_nonzero(in_bin))
        spectrum[bin_idx] = float(np.nanmean(power[in_bin]))
    return spectrum, counts


def decode_stretched_uint8(
    values: np.ndarray,
    tags: dict[str, Any],
    *,
    variable: str,
    nodata: float | int | None = None,
) -> np.ndarray:
    """Decode a stretched uint8 raster block to physical units."""
    arr = np.asarray(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    tag_nodata = tags.get("nodata")
    nodata_value = (
        float(tag_nodata)
        if tag_nodata is not None and str(tag_nodata).strip() != ""
        else (None if nodata is None else float(nodata))
    )
    valid = np.ones(arr.shape, dtype=bool)
    if nodata_value is not None and np.isfinite(nodata_value):
        valid &= ~np.isclose(arr.astype(np.float64), nodata_value, atol=0.0, rtol=0.0)
    valid &= np.isfinite(arr.astype(np.float64))
    if not np.any(valid):
        return out

    stretch_min = float(tags["stretch_min"])
    stretch_max = float(tags["stretch_max"])
    valid_code_max = float(tags.get("valid_code_max", 254.0))
    decoded = stretch_min + (
        arr.astype(np.float64) / valid_code_max * (stretch_max - stretch_min)
    )
    if (
        str(tags.get("stretch_units", "")).strip().upper() == "K"
        or str(tags.get("stretch_name", "")).strip() == "temperature_kelvin"
        or str(variable).strip().lower() == "temperature"
        and str(tags.get("source_product", "")).strip().lower() in {"ostia", "glorys"}
        and stretch_max > 200.0
    ):
        decoded = decoded - CELSIUS_TO_KELVIN_OFFSET
    out[valid] = decoded[valid]
    return out


def _row_bounds(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return left, bottom, right, top bounds for a selected-patch row."""
    left = min(float(row["lon0"]), float(row["lon1"]))
    right = max(float(row["lon0"]), float(row["lon1"]))
    bottom = min(float(row["lat0"]), float(row["lat1"]))
    top = max(float(row["lat0"]), float(row["lat1"]))
    return left, bottom, right, top


def _window_for_row(
    dataset: rasterio.DatasetReader,
    row: dict[str, Any],
) -> rasterio.windows.Window | None:
    """Return a rasterio window for one selected patch row."""
    left, bottom, right, top = _row_bounds(row)
    raw_window = rasterio.windows.from_bounds(
        left,
        bottom,
        right,
        top,
        transform=dataset.transform,
    )
    row_off = int(round(float(raw_window.row_off)))
    col_off = int(round(float(raw_window.col_off)))
    height = int(round(float(raw_window.height)))
    width = int(round(float(raw_window.width)))
    if height <= 0 or width <= 0:
        return None
    if (
        row_off < 0
        or col_off < 0
        or row_off + height > int(dataset.height)
        or col_off + width > int(dataset.width)
    ):
        return None
    return rasterio.windows.Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height,
    )


def _read_patch_window_from_dataset(
    dataset: rasterio.DatasetReader,
    row: dict[str, Any],
    *,
    variable: str,
    band_index: int = 1,
    decode_uint8: bool = False,
) -> np.ndarray | None:
    """Read one selected-patch window from an already opened raster."""
    window = _window_for_row(dataset, row)
    if window is None:
        return None
    if int(band_index) < 1 or int(band_index) > int(dataset.count):
        return None
    data = dataset.read(int(band_index), window=window, masked=False)
    tags = dataset.tags()
    nodata = dataset.nodata
    if decode_uint8 or (
        np.issubdtype(data.dtype, np.integer) and "stretch_min" in tags
    ):
        return decode_stretched_uint8(
            data,
            tags,
            variable=variable,
            nodata=nodata,
        )
    out = data.astype(np.float64, copy=False)
    if nodata is not None and np.isfinite(float(nodata)):
        out = out.copy()
        out[np.isclose(out, float(nodata), atol=0.0, rtol=0.0)] = np.nan
    return out


def read_patch_window(
    path: Path,
    row: dict[str, Any],
    *,
    variable: str,
    band_index: int = 1,
    decode_uint8: bool = False,
) -> np.ndarray | None:
    """Read one selected-patch window from a raster path."""
    with rasterio.open(path) as dataset:
        return _read_patch_window_from_dataset(
            dataset,
            row,
            variable=variable,
            band_index=band_index,
            decode_uint8=decode_uint8,
        )


@lru_cache(maxsize=1)
def _ocean_region_shapes() -> tuple[tuple[str, Any], ...]:
    """Return shapely ocean-region geometries in dashboard order."""
    regions: list[tuple[str, Any]] = []
    for feature in world_ocean_region_geojson_features():
        properties = feature.get("properties", {})
        name = str(properties.get("name") or properties.get("region") or "").strip()
        geometry = feature.get("geometry")
        if name and geometry is not None:
            regions.append((name, shape(geometry)))
    if not regions:
        raise RuntimeError("No ocean-region polygons were available.")
    return tuple(regions)


def assign_patch_basin_by_overlap(
    row: dict[str, Any],
    *,
    threshold: float = DEFAULT_BASIN_OVERLAP_THRESHOLD,
) -> str | None:
    """Return the ocean basin covering at least ``threshold`` of a patch."""
    patch_polygon = box(*_row_bounds(row))
    patch_area = float(patch_polygon.area)
    if patch_area <= 0.0:
        return None
    best_name: str | None = None
    best_fraction = 0.0
    for name, geometry in _ocean_region_shapes():
        if not geometry.intersects(patch_polygon):
            continue
        fraction = float(geometry.intersection(patch_polygon).area) / patch_area
        if fraction > best_fraction:
            best_name = str(name)
            best_fraction = fraction
    if best_name is None or best_fraction < float(threshold):
        return None
    return best_name


def _dataset_root_from_summary(summary: dict[str, Any]) -> Path | None:
    """Resolve the packaged GeoTIFF dataset root referenced by a run summary."""
    data_config = summary.get("data_config")
    if data_config is not None:
        data_config_path = Path(str(data_config))
        if data_config_path.exists():
            data_cfg = _load_yaml(data_config_path)
            root = data_cfg.get("dataset", {}).get("core", {}).get("geotiff_root_dir")
            if root:
                return Path(root)

    land_mask_path = summary.get("land_mask_path") or summary.get(
        "inference_grid", {}
    ).get("land_mask_path")
    if land_mask_path is not None:
        path = Path(str(land_mask_path))
        if path.parent.name == "masks":
            return path.parent.parent
    return None


def _source_raster_path(
    *,
    dataset_root: Path | None,
    variable: str,
    date_value: int,
    surface: bool,
) -> Path | None:
    """Return a packaged source raster path for one variable/date."""
    if dataset_root is None:
        return None
    variable = str(variable)
    if surface:
        source = SOURCE_LAYER_BY_VARIABLE.get(variable)
        if source is None:
            return None
        group, name, stem = source
        path = dataset_root / "rasters" / group / name / f"{stem}_{int(date_value)}.tif"
    else:
        source = GLORYS_SOURCE_BY_VARIABLE.get(variable)
        if source is None:
            return None
        name, stem = source
        path = (
            dataset_root / "rasters" / "glorys" / name / f"{stem}_{int(date_value)}.tif"
        )
    return path if path.exists() else None


def _requested_variables(raw_variables: Sequence[str] | None) -> set[str]:
    """Normalize requested variable names."""
    if not raw_variables:
        return {"temperature", "salinity"}
    return {str(variable).strip().lower() for variable in raw_variables}


def _load_variable_run(path: Path, requested: set[str]) -> VariableRun | None:
    """Load a single-variable run summary when ``path`` points to one."""
    summary_path = Path(path) / "run_summary.yaml"
    if not summary_path.exists():
        return None
    summary = _load_yaml(summary_path)
    variable = str(summary.get("variable", "")).strip().lower()
    if variable not in requested:
        return None
    return VariableRun(
        variable=variable,
        run_dir=Path(path),
        summary_path=summary_path,
        summary=summary,
    )


def _paired_variable_run_paths(root: Path, summary: dict[str, Any]) -> list[Path]:
    """Return variable run paths from a paired run root summary."""
    paths: list[Path] = []
    variables = summary.get("variables", {})
    if isinstance(variables, dict):
        for metadata in variables.values():
            if not isinstance(metadata, dict):
                continue
            raw_dir = metadata.get("run_dir")
            raw_summary = metadata.get("summary_path")
            if raw_dir:
                paths.append(_resolve_referenced_path(root, raw_dir))
            elif raw_summary:
                paths.append(_resolve_referenced_path(root, raw_summary).parent)
    for variable in ("temperature", "salinity"):
        candidate = root / variable
        if candidate.joinpath("run_summary.yaml").exists():
            paths.append(candidate)
    return paths


def _temporal_run_paths(root: Path, summary: dict[str, Any]) -> list[Path]:
    """Return temporal run paths from paired or filesystem metadata."""
    paths: list[Path] = []
    temporal = summary.get("temporal_consistency", {})
    if isinstance(temporal, dict):
        variable_run_dirs = temporal.get("variable_run_dirs", {})
        if isinstance(variable_run_dirs, dict):
            for run_dirs in variable_run_dirs.values():
                if not isinstance(run_dirs, list):
                    continue
                for raw_path in run_dirs:
                    paths.append(_resolve_referenced_path(root, raw_path))
    for candidate in root.glob("temporal_runs/*/*"):
        if candidate.joinpath("run_summary.yaml").exists():
            paths.append(candidate)
    return paths


def discover_variable_runs(
    run_dirs: Sequence[Path],
    *,
    variables: Sequence[str] | None = None,
    include_temporal_runs: bool = False,
) -> list[VariableRun]:
    """Discover single-variable run summaries under one or more roots."""
    requested = _requested_variables(variables)
    found: list[VariableRun] = []
    seen: set[Path] = set()

    def _add_path(path: Path) -> None:
        run = _load_variable_run(path, requested)
        if run is None:
            return
        key = run.summary_path.resolve()
        if key in seen:
            return
        seen.add(key)
        found.append(run)

    for raw_root in run_dirs:
        root = Path(raw_root)
        direct = _load_variable_run(root, requested)
        if direct is not None:
            _add_path(root)
            if include_temporal_runs:
                paired_root = (
                    root.parent if (root.parent / "temporal_runs").exists() else root
                )
                for candidate in _temporal_run_paths(paired_root, direct.summary):
                    _add_path(candidate)
            continue

        summary_path = root / "run_summary.yaml"
        summary = _load_yaml(summary_path) if summary_path.exists() else {}
        for candidate in _paired_variable_run_paths(root, summary):
            _add_path(candidate)
        if include_temporal_runs:
            for candidate in _temporal_run_paths(root, summary):
                _add_path(candidate)

    return found


def _depth_layer_specs(run: VariableRun) -> list[DepthLayerSpec]:
    """Resolve prediction and GLORYS layer specs for one variable run."""
    dataset_root = _dataset_root_from_summary(run.summary)
    date_value = int(run.summary.get("selected_date") or run.summary.get("target_date"))
    layers: list[DepthLayerSpec] = []
    for raw_export in run.summary.get("depth_exports", []):
        if not isinstance(raw_export, dict):
            continue
        prediction_path = _resolve_run_artifact_path(
            run.run_dir, raw_export.get("prediction_tif_path")
        )
        if prediction_path is None or not prediction_path.exists():
            continue
        suffix = str(raw_export.get("suffix", "depth"))
        label = str(raw_export.get("label", suffix))
        requested_depth_m = float(raw_export.get("requested_depth_m", np.nan))
        actual_depth_m = float(raw_export.get("actual_depth_m", requested_depth_m))
        channel_index = int(raw_export.get("channel_index", 0))
        layers.append(
            DepthLayerSpec(
                variable=run.variable,
                layer="prediction",
                suffix=suffix,
                label=label,
                requested_depth_m=requested_depth_m,
                actual_depth_m=actual_depth_m,
                channel_index=channel_index,
                path=prediction_path,
            )
        )

        glorys_path = _resolve_run_artifact_path(
            run.run_dir, raw_export.get("ground_truth_tif_path")
        )
        if glorys_path is not None and glorys_path.exists():
            layers.append(
                DepthLayerSpec(
                    variable=run.variable,
                    layer="glorys",
                    suffix=suffix,
                    label=label,
                    requested_depth_m=requested_depth_m,
                    actual_depth_m=actual_depth_m,
                    channel_index=channel_index,
                    path=glorys_path,
                )
            )
            continue

        source_glorys_path = _source_raster_path(
            dataset_root=dataset_root,
            variable=run.variable,
            date_value=date_value,
            surface=False,
        )
        if source_glorys_path is not None:
            layers.append(
                DepthLayerSpec(
                    variable=run.variable,
                    layer="glorys",
                    suffix=suffix,
                    label=label,
                    requested_depth_m=requested_depth_m,
                    actual_depth_m=actual_depth_m,
                    channel_index=channel_index,
                    path=source_glorys_path,
                    band_index=channel_index + 1,
                    decode_stretched_uint8=True,
                    source_kind="packaged_source",
                )
            )

    surface_path = _source_raster_path(
        dataset_root=dataset_root,
        variable=run.variable,
        date_value=date_value,
        surface=True,
    )
    if surface_path is not None:
        layers.append(
            DepthLayerSpec(
                variable=run.variable,
                layer="surface_observation",
                suffix="surface",
                label="Surface",
                requested_depth_m=0.0,
                actual_depth_m=0.0,
                channel_index=0,
                path=surface_path,
                decode_stretched_uint8=True,
                source_kind="packaged_source",
            )
        )
    return layers


def _pixel_sizes_km(
    row: dict[str, Any], shape_2d: tuple[int, int]
) -> tuple[float, float]:
    """Return approximate zonal and meridional pixel sizes for one patch."""
    left, bottom, right, top = _row_bounds(row)
    height, width = int(shape_2d[0]), int(shape_2d[1])
    lat_center = 0.5 * (bottom + top)
    pixel_width_deg = abs(right - left) / float(width)
    pixel_height_deg = abs(top - bottom) / float(height)
    pixel_x = (
        pixel_width_deg
        * EARTH_KM_PER_DEGREE
        * max(
            abs(float(np.cos(np.deg2rad(lat_center)))),
            1.0e-6,
        )
    )
    pixel_y = pixel_height_deg * EARTH_KM_PER_DEGREE
    return float(pixel_x), float(pixel_y)


def _spectra_for_run(
    run: VariableRun,
    *,
    wavelength_edges: np.ndarray,
    basin_overlap_threshold: float,
    require_complete_patches: bool,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, int]]:
    """Compute all patch spectra for one variable run."""
    selected_patches_path = run.run_dir / "selected_patches.csv"
    if not selected_patches_path.exists():
        return [], [], {"missing_selected_patches": 1}
    patches = pd.read_csv(selected_patches_path).to_dict(orient="records")
    date_value = int(run.summary.get("selected_date") or run.summary.get("target_date"))
    year, month, season = _date_parts(date_value)
    iso_year = int(run.summary.get("iso_year", year))
    iso_week = int(run.summary.get("iso_week", 0))
    layer_specs = _depth_layer_specs(run)
    records: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    skip_counts = {
        "missing_selected_patches": 0,
        "missing_layer_specs": 0,
        "missing_windows": 0,
        "incomplete_patches": 0,
        "empty_spectra": 0,
    }
    if not layer_specs:
        skip_counts["missing_layer_specs"] += 1
        return records, spectra, skip_counts

    basin_cache: dict[str, str | None] = {}
    for layer_spec in layer_specs:
        with rasterio.open(layer_spec.path) as dataset:
            for row in patches:
                patch_key = str(row.get("patch_id", len(basin_cache)))
                if patch_key not in basin_cache:
                    basin_cache[patch_key] = assign_patch_basin_by_overlap(
                        row,
                        threshold=basin_overlap_threshold,
                    )
                patch_basin = basin_cache[patch_key]
                patch = _read_patch_window_from_dataset(
                    dataset,
                    row,
                    variable=run.variable,
                    band_index=layer_spec.band_index,
                    decode_uint8=layer_spec.decode_stretched_uint8,
                )
                if patch is None:
                    skip_counts["missing_windows"] += 1
                    continue
                if require_complete_patches and not np.all(np.isfinite(patch)):
                    skip_counts["incomplete_patches"] += 1
                    continue
                pixel_x_km, pixel_y_km = _pixel_sizes_km(row, patch.shape)
                spectrum_result = radial_wavenumber_spectrum(
                    patch,
                    pixel_size_x_km=pixel_x_km,
                    pixel_size_y_km=pixel_y_km,
                    wavelength_edges_km=wavelength_edges,
                    require_complete=require_complete_patches,
                )
                if spectrum_result is None:
                    skip_counts["empty_spectra"] += 1
                    continue
                spectrum, bin_counts = spectrum_result
                if not np.any(np.isfinite(spectrum)):
                    skip_counts["empty_spectra"] += 1
                    continue
                record = {
                    "spectrum_index": len(spectra),
                    "variable": run.variable,
                    "layer": layer_spec.layer,
                    "layer_label": LAYER_LABELS.get(layer_spec.layer, layer_spec.layer),
                    "source_kind": layer_spec.source_kind,
                    "run_dir": str(run.run_dir),
                    "selected_date": date_value,
                    "iso_year": iso_year,
                    "iso_week": iso_week,
                    "year": year,
                    "month": month,
                    "season": season,
                    "patch_id": str(row.get("patch_id", "")),
                    "grid_y0": int(row.get("grid_y0", -1)),
                    "grid_x0": int(row.get("grid_x0", -1)),
                    "lon0": float(row.get("lon0", np.nan)),
                    "lon1": float(row.get("lon1", np.nan)),
                    "lat0": float(row.get("lat0", np.nan)),
                    "lat1": float(row.get("lat1", np.nan)),
                    "basin": "" if patch_basin is None else patch_basin,
                    "included_in_basin": bool(patch_basin is not None),
                    "depth_suffix": layer_spec.suffix,
                    "depth_label": layer_spec.label,
                    "requested_depth_m": layer_spec.requested_depth_m,
                    "actual_depth_m": layer_spec.actual_depth_m,
                    "channel_index": layer_spec.channel_index,
                    "raster_path": str(layer_spec.path),
                    "raster_band_index": int(layer_spec.band_index),
                    "finite_pixel_count": int(np.count_nonzero(np.isfinite(patch))),
                    "pixel_size_x_km": float(pixel_x_km),
                    "pixel_size_y_km": float(pixel_y_km),
                    "fft_bin_count_total": int(np.sum(bin_counts)),
                }
                spectra.append(spectrum.astype(np.float32, copy=False))
                records.append(record)
    return records, spectra, skip_counts


def aggregate_spectra(
    records: pd.DataFrame,
    spectra: np.ndarray,
    wavelength_centers: np.ndarray,
    wavelength_edges: np.ndarray,
) -> pd.DataFrame:
    """Aggregate patch spectra by basin, time period, layer, and depth."""
    if records.empty or spectra.size == 0:
        return pd.DataFrame()
    wavelength_edges = np.asarray(wavelength_edges, dtype=np.float64)
    wavenumber_edges = 1.0 / wavelength_edges[::-1]
    wavenumber_centers = 1.0 / np.asarray(wavelength_centers, dtype=np.float64)
    wavenumber_widths = np.diff(wavenumber_edges)[::-1]
    rows: list[dict[str, Any]] = []
    base_group_cols = [
        "variable",
        "layer",
        "layer_label",
        "depth_suffix",
        "depth_label",
        "requested_depth_m",
        "actual_depth_m",
        "channel_index",
    ]
    period_specs = (
        ("year", ["year"], lambda values: str(int(values["year"]))),
        (
            "season",
            ["year", "season"],
            lambda values: f"{int(values['year'])}_{values['season']}",
        ),
        (
            "month",
            ["year", "month"],
            lambda values: f"{int(values['year'])}-{int(values['month']):02d}",
        ),
    )
    basin_scopes = [(ALL_OCEANS_BASIN, records)]
    basin_records = records[records["included_in_basin"].astype(bool)]
    for basin_name, basin_df in basin_records.groupby("basin", sort=True):
        basin_scopes.append((str(basin_name), basin_df))

    for basin_name, basin_df in basin_scopes:
        if basin_df.empty:
            continue
        for period_type, period_cols, label_fn in period_specs:
            group_cols = base_group_cols + period_cols
            for keys, group in basin_df.groupby(group_cols, sort=True):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                values = dict(zip(group_cols, keys, strict=True))
                indices = group["spectrum_index"].to_numpy(dtype=np.int64)
                group_spectra = spectra[indices]
                mean_power = np.full(group_spectra.shape[1], np.nan, dtype=np.float64)
                finite_bins = np.any(np.isfinite(group_spectra), axis=0)
                if np.any(finite_bins):
                    # Avoid nanmean warnings for bins with no FFT samples.
                    mean_power[finite_bins] = np.nanmean(
                        group_spectra[:, finite_bins],
                        axis=0,
                    )
                finite_rows = np.any(np.isfinite(group_spectra), axis=1)
                for bin_idx, wavelength in enumerate(wavelength_centers.tolist()):
                    power_value = mean_power[bin_idx]
                    width_value = wavenumber_widths[bin_idx]
                    psd_value = (
                        power_value / width_value
                        if np.isfinite(power_value) and width_value > 0.0
                        else np.nan
                    )
                    rows.append(
                        {
                            "variable": values["variable"],
                            "layer": values["layer"],
                            "layer_label": values["layer_label"],
                            "basin": basin_name,
                            "period_type": period_type,
                            "period_label": label_fn(values),
                            "year": int(values["year"]),
                            "month": (
                                int(values["month"]) if "month" in values else None
                            ),
                            "season": values.get("season"),
                            "depth_suffix": values["depth_suffix"],
                            "depth_label": values["depth_label"],
                            "requested_depth_m": float(values["requested_depth_m"]),
                            "actual_depth_m": float(values["actual_depth_m"]),
                            "channel_index": int(values["channel_index"]),
                            "wavelength_km": float(wavelength),
                            "horizontal_wavenumber_cpkm": float(
                                wavenumber_centers[bin_idx]
                            ),
                            "wavenumber_bin_width_cpkm": float(width_value),
                            "wavelength_bin_index": int(bin_idx),
                            "power_mean": (
                                None
                                if not np.isfinite(power_value)
                                else float(power_value)
                            ),
                            "psd_mean": (
                                None if not np.isfinite(psd_value) else float(psd_value)
                            ),
                            "spectrum_count": int(np.count_nonzero(finite_rows)),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _sanitize_filename(value: str) -> str:
    """Return a filesystem-safe lowercase filename component."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_").lower() or "item"


def _json_ready_value(value: Any) -> Any:
    """Return a JSON-serializable scalar with missing values normalized."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return dataframe records safe for compact dashboard JSON files."""
    records: list[dict[str, Any]] = []
    for raw_record in frame.to_dict(orient="records"):
        records.append(
            {str(key): _json_ready_value(value) for key, value in raw_record.items()}
        )
    return records


def _dashboard_available_depths(aggregated: pd.DataFrame) -> list[dict[str, Any]]:
    """Return depth selector metadata for dashboard configs."""
    required = {
        "variable",
        "layer",
        "depth_suffix",
        "depth_label",
        "requested_depth_m",
        "actual_depth_m",
        "channel_index",
    }
    if aggregated.empty or not required.issubset(set(aggregated.columns)):
        return []

    depth_rows = aggregated[aggregated["layer"] != "surface_observation"]
    depth_rows = depth_rows[
        [
            "variable",
            "depth_suffix",
            "depth_label",
            "requested_depth_m",
            "actual_depth_m",
            "channel_index",
        ]
    ].drop_duplicates()
    depth_rows = depth_rows.sort_values(
        ["variable", "actual_depth_m", "channel_index", "depth_suffix"],
        kind="mergesort",
    )
    depths: list[dict[str, Any]] = []
    for record in depth_rows.to_dict(orient="records"):
        depths.append(
            {
                "variable": str(record["variable"]),
                "suffix": str(record["depth_suffix"]),
                "label": str(record["depth_label"]),
                "requested_depth_m": float(record["requested_depth_m"]),
                "actual_depth_m": float(record["actual_depth_m"]),
                "channel_index": int(record["channel_index"]),
            }
        )
    return depths


def _spectral_basin_map_payload() -> dict[str, Any]:
    """Return authoritative ocean-region polygons for spectral basin selection."""
    return {
        "type": "FeatureCollection",
        "name": "DepthDif world_oceans.geojson spectral basins",
        "features": world_ocean_region_geojson_features(),
    }


def _copy_spectral_dashboard_static_files(output_dir: Path) -> list[str]:
    """Copy standalone spectral dashboard files beside generated JSON data."""
    repo_root = Path(__file__).resolve().parents[3]
    source_files = {
        repo_root
        / "docs"
        / "spectral-dashboard"
        / "index.html": Path(output_dir)
        / "index.html",
        repo_root
        / "docs"
        / "javascripts"
        / "spectral-dashboard.js": Path(output_dir)
        / "javascripts"
        / "spectral-dashboard.js",
        repo_root
        / "docs"
        / "stylesheets"
        / "spectral-dashboard.css": Path(output_dir)
        / "stylesheets"
        / "spectral-dashboard.css",
    }
    copied: list[str] = []
    for source_path, destination_path in source_files.items():
        if not source_path.exists():
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.name == "index.html":
            html = source_path.read_text(encoding="utf-8")
            # Exported dashboards live at output_dir/index.html, unlike docs pages.
            html = html.replace("../stylesheets/", "stylesheets/")
            html = html.replace("../javascripts/", "javascripts/")
            destination_path.write_text(html, encoding="utf-8")
        else:
            shutil.copy2(source_path, destination_path)
        copied.append(str(destination_path.relative_to(output_dir)))
    return copied


def write_spectral_dashboard_assets(
    aggregated: pd.DataFrame,
    *,
    output_dir: Path,
    run_paths: Sequence[Path],
    wavelength_edges: np.ndarray,
    wavelength_centers: np.ndarray,
    summary: dict[str, Any],
    layer_labels: dict[str, str] | None = None,
    layer_order: Sequence[str] | None = None,
    line_styles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write compact JSON and static files for the spectral dashboard."""
    output_path = Path(output_dir)
    resolved_layer_labels = dict(LAYER_LABELS if layer_labels is None else layer_labels)
    resolved_layer_order = (
        [key for key in ("prediction", "glorys", "surface_observation")]
        if layer_order is None
        else [str(layer) for layer in layer_order]
    )
    resolved_line_styles = dict(
        LAYER_LINESTYLES if line_styles is None else line_styles
    )
    basin_dir = output_path / SPECTRAL_DASHBOARD_BASIN_DIR_NAME
    basin_dir.mkdir(parents=True, exist_ok=True)

    map_payload = _spectral_basin_map_payload()
    with (output_path / SPECTRAL_DASHBOARD_BASIN_MAP_NAME).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(map_payload, f)
        f.write("\n")

    map_basin_names = [
        str(feature.get("properties", {}).get("name"))
        for feature in map_payload.get("features", [])
        if feature.get("properties", {}).get("name")
    ]
    basin_names = [ALL_OCEANS_BASIN] + [
        name for name in map_basin_names if name != ALL_OCEANS_BASIN
    ]

    basin_entries: list[dict[str, Any]] = []
    basin_data_urls: dict[str, str] = {}
    for basin_name in basin_names:
        file_name = f"{_sanitize_filename(basin_name)}.json"
        relative_url = f"{SPECTRAL_DASHBOARD_BASIN_DIR_NAME}/{file_name}"
        basin_rows = (
            aggregated[aggregated["basin"] == basin_name]
            if not aggregated.empty and "basin" in aggregated.columns
            else aggregated
        )
        payload = {
            "schema_version": 1,
            "kind": "wavenumber_spectral_basin",
            "basin": basin_name,
            "is_global": basin_name == ALL_OCEANS_BASIN,
            "rows": _json_records(basin_rows),
        }
        with (basin_dir / file_name).open("w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.write("\n")
        basin_entries.append(
            {
                "name": basin_name,
                "label": basin_name,
                "is_global": basin_name == ALL_OCEANS_BASIN,
                "data_url": relative_url,
            }
        )
        basin_data_urls[basin_name] = relative_url

    variables = []
    if not aggregated.empty and "variable" in aggregated.columns:
        variables = sorted(
            str(value) for value in aggregated["variable"].dropna().unique()
        )
    default_variable = (
        "temperature"
        if "temperature" in variables
        else (variables[0] if variables else None)
    )
    period_types = []
    if not aggregated.empty and "period_type" in aggregated.columns:
        period_types = sorted(
            str(value) for value in aggregated["period_type"].dropna().unique()
        )
    config = {
        "schema_version": 1,
        "kind": "wavenumber_spectral_dashboard",
        "available_variables": variables,
        "default_variable": default_variable,
        "period_types": period_types,
        "available_depths": _dashboard_available_depths(aggregated),
        "layers": resolved_layer_labels,
        "layer_order": resolved_layer_order,
        "line_styles": resolved_line_styles,
        "wavelength_bin_edges_km": [
            float(value) for value in wavelength_edges.tolist()
        ],
        "wavelength_bin_centers_km": [
            float(value) for value in wavelength_centers.tolist()
        ],
        "horizontal_wavenumber_bin_edges_cpkm": [
            float(value) for value in (1.0 / wavelength_edges[::-1]).tolist()
        ],
        "horizontal_wavenumber_bin_centers_cpkm": [
            float(value) for value in (1.0 / wavelength_centers).tolist()
        ],
        "basins": basin_entries,
        "basin_data_urls": basin_data_urls,
        "basin_map_geojson_url": SPECTRAL_DASHBOARD_BASIN_MAP_NAME,
        "run_dirs": [str(path) for path in run_paths],
        "source_summary": summary,
    }
    with (output_path / SPECTRAL_DASHBOARD_CONFIG_NAME).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    copied_files = _copy_spectral_dashboard_static_files(output_path)
    return {
        "dashboard_config_json": SPECTRAL_DASHBOARD_CONFIG_NAME,
        "dashboard_basin_map_geojson": SPECTRAL_DASHBOARD_BASIN_MAP_NAME,
        "dashboard_basin_dir": SPECTRAL_DASHBOARD_BASIN_DIR_NAME,
        "dashboard_basin_count": len(basin_entries),
        "dashboard_static_files": copied_files,
    }


def _spectrum_line_label(
    layer: str,
    depth_label: str,
    *,
    layer_labels: dict[str, str] | None = None,
) -> str:
    """Return a compact legend label for one plotted spectrum."""
    labels = LAYER_LABELS if layer_labels is None else layer_labels
    layer_label = labels.get(str(layer), str(layer))
    depth_text = str(depth_label)
    if str(layer) == "surface_observation" or depth_text.lower() == layer_label.lower():
        return layer_label
    return f"{layer_label} {depth_text}"


def _spectral_power_unit_label(variable: str) -> str:
    """Return the dashboard display unit for spectral density."""
    if str(variable).strip().lower() == "temperature":
        return "degC^2/cpkm"
    if str(variable).strip().lower() == "salinity":
        return "salinity^2/cpkm"
    return "field^2/cpkm"


def write_spectrum_plots(
    aggregated: pd.DataFrame,
    *,
    output_dir: Path,
    layer_labels: dict[str, str] | None = None,
    line_styles: dict[str, str] | None = None,
) -> list[Path]:
    """Write default log-log spectra plots from aggregated spectra."""
    if aggregated.empty:
        return []
    labels = LAYER_LABELS if layer_labels is None else layer_labels
    styles = LAYER_LINESTYLES if line_styles is None else line_styles
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plot_rows = aggregated[aggregated["period_type"].isin(["season", "year"])]
    group_cols = ["variable", "basin", "period_type", "period_label"]
    for keys, group in plot_rows.groupby(group_cols, sort=True):
        variable, basin, period_type, period_label = [str(value) for value in keys]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        depth_values = np.asarray(
            group["actual_depth_m"].dropna().unique(), dtype=float
        )
        if depth_values.size == 0:
            depth_values = np.asarray([0.0], dtype=float)
        depth_min = float(np.nanmin(depth_values))
        depth_max = float(np.nanmax(depth_values))
        denom = max(depth_max - depth_min, 1.0)
        cmap = plt.get_cmap("viridis")

        line_cols = ["layer", "depth_label", "actual_depth_m", "channel_index"]
        for line_keys, line_group in group.groupby(line_cols, sort=True):
            layer, depth_label, actual_depth_m, _channel_index = line_keys
            line_group = line_group.sort_values("wavelength_km")
            y = line_group["psd_mean"].to_numpy(dtype=np.float64)
            wavelength = line_group["wavelength_km"].to_numpy(dtype=np.float64)
            x = np.divide(
                1.0,
                wavelength,
                out=np.full_like(wavelength, np.nan),
                where=wavelength > 0.0,
            )
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            if not np.any(valid):
                continue
            color_value = (float(actual_depth_m) - depth_min) / denom
            ax.loglog(
                x[valid],
                y[valid],
                linestyle=styles.get(str(layer), "-"),
                color=cmap(float(np.clip(color_value, 0.0, 1.0))),
                linewidth=1.6,
                label=_spectrum_line_label(
                    str(layer),
                    str(depth_label),
                    layer_labels=labels,
                ),
            )

        if not ax.lines:
            plt.close(fig)
            continue
        ax.set_xlabel("Horizontal wavenumber [cpkm]")
        ax.set_ylabel(f"PSD [{_spectral_power_unit_label(variable)}]")
        ax.set_title(f"{variable.title()} spectra - {basin} - {period_label}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        plot_path = (
            plot_dir / f"{_sanitize_filename(variable)}_{_sanitize_filename(basin)}_"
            f"{_sanitize_filename(period_type)}_{_sanitize_filename(period_label)}.png"
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        written.append(plot_path)
    return written


def export_wavenumber_spectra(
    *,
    run_dirs: Sequence[Path],
    output_dir: Path | None = None,
    variables: Sequence[str] | None = None,
    include_temporal_runs: bool = False,
    min_wavelength_km: float = DEFAULT_MIN_WAVELENGTH_KM,
    max_wavelength_km: float = DEFAULT_MAX_WAVELENGTH_KM,
    wavelength_bin_count: int = DEFAULT_WAVELENGTH_BIN_COUNT,
    basin_overlap_threshold: float = DEFAULT_BASIN_OVERLAP_THRESHOLD,
    require_complete_patches: bool = False,
    write_plots: bool = True,
    write_dashboard: bool = True,
    public_base_url: str | None = None,
    rclone_remote: str | None = None,
) -> dict[str, Any]:
    """Compute and write wavenumber spectra for existing inference runs."""
    if not run_dirs:
        raise ValueError("At least one run directory is required.")
    run_paths = [Path(path) for path in run_dirs]
    if output_dir is None:
        output_dir = run_paths[0] / DEFAULT_OUTPUT_DIR_NAME
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered_runs = discover_variable_runs(
        run_paths,
        variables=variables,
        include_temporal_runs=include_temporal_runs,
    )
    if not discovered_runs:
        raise RuntimeError("No matching single-variable inference run summaries found.")

    edges = wavelength_bin_edges_km(
        min_wavelength_km=min_wavelength_km,
        max_wavelength_km=max_wavelength_km,
        bin_count=wavelength_bin_count,
    )
    centers = wavelength_bin_centers_km(edges)
    all_records: list[dict[str, Any]] = []
    all_spectra: list[np.ndarray] = []
    skipped_by_run: list[dict[str, Any]] = []
    for run in discovered_runs:
        records, spectra, skip_counts = _spectra_for_run(
            run,
            wavelength_edges=edges,
            basin_overlap_threshold=basin_overlap_threshold,
            require_complete_patches=require_complete_patches,
        )
        offset = len(all_spectra)
        for record in records:
            record["spectrum_index"] = int(record["spectrum_index"]) + offset
        all_records.extend(records)
        all_spectra.extend(spectra)
        skipped_by_run.append(
            {
                "variable": run.variable,
                "run_dir": str(run.run_dir),
                **{key: int(value) for key, value in skip_counts.items()},
            }
        )

    spectra_array = (
        np.stack(all_spectra, axis=0).astype(np.float32, copy=False)
        if all_spectra
        else np.zeros((0, centers.size), dtype=np.float32)
    )
    records_df = pd.DataFrame.from_records(all_records)
    aggregated_df = aggregate_spectra(records_df, spectra_array, centers, edges)

    np.savez_compressed(
        output_dir / "patch_spectra.npz",
        spectra=spectra_array,
        wavelength_bin_edges_km=edges.astype(np.float32),
        wavelength_bin_centers_km=centers.astype(np.float32),
    )
    records_df.to_csv(output_dir / "patch_spectra_records.csv", index=False)
    aggregated_df.to_csv(output_dir / "aggregated_spectra.csv", index=False)
    plot_paths = (
        write_spectrum_plots(aggregated_df, output_dir=output_dir)
        if write_plots
        else []
    )

    summary = {
        "schema_version": 1,
        "kind": "wavenumber_spectra",
        "run_dirs": [str(path) for path in run_paths],
        "output_dir": str(output_dir),
        "include_temporal_runs": bool(include_temporal_runs),
        "variables": sorted(_requested_variables(variables)),
        "run_count": int(len(discovered_runs)),
        "spectrum_count": int(spectra_array.shape[0]),
        "wavelength_bin_count": int(centers.size),
        "wavelength_min_km": float(min_wavelength_km),
        "wavelength_max_km": float(max_wavelength_km),
        "basin_overlap_threshold": float(basin_overlap_threshold),
        "require_complete_patches": bool(require_complete_patches),
        "dashboard_enabled": bool(write_dashboard),
        "public_base_url": public_base_url,
        "upload_requested": rclone_remote is not None,
        "upload_remote": rclone_remote,
        "upload_ok": None,
        "upload_message": None,
        "artifacts": {
            "patch_spectra_npz": "patch_spectra.npz",
            "patch_spectra_records_csv": "patch_spectra_records.csv",
            "aggregated_spectra_csv": "aggregated_spectra.csv",
            "summary_json": "summary.json",
            "plot_count": int(len(plot_paths)),
            "plots_dir": "plots",
        },
        "skipped_by_run": skipped_by_run,
    }
    if write_dashboard:
        dashboard_artifacts = write_spectral_dashboard_assets(
            aggregated_df,
            output_dir=output_dir,
            run_paths=run_paths,
            wavelength_edges=edges,
            wavelength_centers=centers,
            summary={
                key: value for key, value in summary.items() if key != "artifacts"
            },
        )
        summary["artifacts"].update(dashboard_artifacts)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    if rclone_remote is not None:
        ok, message = _sync_with_rclone(output_dir, rclone_remote)
        summary["upload_ok"] = bool(ok)
        summary["upload_message"] = str(message)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the wavenumber-spectra CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute 2D wavenumber spectra from existing inference GeoTIFF runs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Paired root or single-variable run directory. Repeat to combine multiple roots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <first-run-dir>/{DEFAULT_OUTPUT_DIR_NAME}.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=("temperature", "salinity"),
        default=None,
        help="Variables to include. Defaults to both when present.",
    )
    parser.add_argument(
        "--include-temporal-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include temporal_runs discovered from a paired production root.",
    )
    parser.add_argument(
        "--min-wavelength-km",
        type=float,
        default=DEFAULT_MIN_WAVELENGTH_KM,
    )
    parser.add_argument(
        "--max-wavelength-km",
        type=float,
        default=DEFAULT_MAX_WAVELENGTH_KM,
    )
    parser.add_argument(
        "--wavelength-bin-count",
        type=int,
        default=DEFAULT_WAVELENGTH_BIN_COUNT,
    )
    parser.add_argument(
        "--basin-overlap-threshold",
        type=float,
        default=DEFAULT_BASIN_OVERLAP_THRESHOLD,
    )
    parser.add_argument(
        "--require-complete-patches",
        action="store_true",
        help="Use only patches with complete finite raster windows.",
    )
    parser.add_argument(
        "--allow-incomplete-patches",
        action="store_false",
        dest="require_complete_patches",
        help="Allow patches with nodata/NaN. This is the default.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation and only write tabular/intermediate artifacts.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip spectral dashboard JSON/static asset generation.",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="Optional hosted base URL recorded in summary metadata.",
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help="Optional rclone destination for uploading generated spectra assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the wavenumber-spectra CLI."""
    args = _build_parser().parse_args(argv)
    summary = export_wavenumber_spectra(
        run_dirs=args.run_dir,
        output_dir=args.output_dir,
        variables=args.variables,
        include_temporal_runs=bool(args.include_temporal_runs),
        min_wavelength_km=float(args.min_wavelength_km),
        max_wavelength_km=float(args.max_wavelength_km),
        wavelength_bin_count=int(args.wavelength_bin_count),
        basin_overlap_threshold=float(args.basin_overlap_threshold),
        require_complete_patches=bool(args.require_complete_patches),
        write_plots=not bool(args.no_plots),
        write_dashboard=not bool(args.no_dashboard),
        public_base_url=args.public_base_url,
        rclone_remote=args.rclone_remote,
    )
    print(
        "Wrote wavenumber spectra: "
        f"{summary['spectrum_count']} spectra across {summary['run_count']} runs "
        f"to {summary['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
