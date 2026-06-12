# Example bundle mode:
# /work/envs/depth/bin/python -m depth_recon.inference.export_paper_metrics \
#   --paper-run-dir inference/outputs/paper_2018_W25 \
#   --output-dir inference/outputs/paper_2018_W25/metrics \
#   --climatology-path inference/outputs/paper_2018_W25/methods/climatology \
#   --en4-holdout-fraction 0.2 --seed 7 --validation-year 2018 \
#   --climatology-idw-power 2.0 --climatology-idw-eps 1.0e-6 \
#   --climatology-idw-neighbors 16 --climatology-idw-chunk-size 250000 \
#   --profile-chunk-size 100000 --overwrite-climatology
# Example legacy mode:
# /work/envs/depth/bin/python -m depth_recon.inference.export_paper_metrics \
#   --year 2018 --iso-week 25 \
#   --idw-run-dir inference/outputs/paper_2018_W25_idw \
#   --lstm-run-dir inference/outputs/paper_2018_W25_lstm \
#   --unet-run-dir inference/outputs/paper_2018_W25_unet \
#   --output-dir inference/outputs/paper_metrics_2018_W25
"""Export paper-ready weekly reconstruction metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy
from scipy.spatial import cKDTree
from tqdm import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.data.dataset_argo_geotiff_gridded import (  # noqa: E402
    ArgoGeoTIFFProfileStore,
)
from depth_recon.inference.export_wavenumber_spectra import (  # noqa: E402
    DepthLayerSpec,
    VariableRun,
    _depth_layer_specs,
    discover_variable_runs,
)
from depth_recon.utils.normalizations import CELSIUS_TO_KELVIN_OFFSET  # noqa: E402

VARIABLES = ("temperature", "salinity")
METHOD_ORDER = ("climatology", "idw", "lstm", "unet")
DEFAULT_PAPER_METHOD_ORDER = (
    "climatology",
    "idw",
    "lstm",
    "cnn",
    "unet",
    "depthdif",
)
METHOD_LABELS = {
    "climatology": "Climatology",
    "idw": "IDW",
    "lstm": "LSTM",
    "cnn": "CNN",
    "unet": "U-Net",
    "depthdif": "DepthDif",
}
TARGET_ORDER = ("en4", "glorys")
TARGET_LABELS = {"en4": "EN4 Validation Set", "glorys": "GLORYS12"}
METRIC_ORDER = ("rmse", "mae", "r2")
GLORYS_MANIFEST_VARIABLES = {"temperature": "thetao", "salinity": "so"}
ARGO_VALUE_VARIABLES = {
    "temperature": ("argo_temp_kelvin_uint8", "argo_temp_valid"),
    "salinity": ("argo_psal_uint8", "argo_psal_valid"),
}
ARGO_STRETCH_NAMES = {"temperature": "temperature", "salinity": "salinity"}
DEFAULT_NODATA = -9999.0
DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS = 16
DEFAULT_CLIMATOLOGY_IDW_POWER = 2.0
DEFAULT_CLIMATOLOGY_IDW_EPS = 1.0e-6
DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE = 250_000
DEFAULT_PROFILE_CHUNK_SIZE = 100_000
DEFAULT_VALIDATION_YEAR = 2018


@dataclass(frozen=True)
class MetricStats:
    """Scalar metrics and support count for one evaluation group."""

    count: int
    rmse: float | None
    mae: float | None
    r2: float | None


@dataclass(frozen=True)
class DatasetContext:
    """Packaged dataset paths and metadata used by paper metrics."""

    root: Path
    manifest: dict[str, Any]
    depth_axis_m: np.ndarray
    land_mask_path: Path
    transform: rasterio.Affine
    crs: Any
    height: int
    width: int
    ocean_mask: np.ndarray


@dataclass(frozen=True)
class ClimatologyArtifacts:
    """Resolved climatology GeoTIFF artifact paths."""

    temperature_path: Path
    salinity_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _method_label(method: str, method_labels: dict[str, str] | None = None) -> str:
    """Return the display label for one metric method."""
    labels = method_labels or METHOD_LABELS
    return str(labels.get(str(method), METHOD_LABELS.get(str(method), str(method))))


def _ordered_methods(
    methods: Iterable[str],
    *,
    preferred_order: Sequence[str] | None = None,
) -> list[str]:
    """Return methods in preferred order with unknown methods appended."""
    method_list = [str(method) for method in methods]
    order = list(preferred_order or DEFAULT_PAPER_METHOD_ORDER)
    ordered = [method for method in order if method in method_list]
    ordered.extend(method for method in method_list if method not in ordered)
    return ordered


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON mapping in {path}.")
    return payload


def _resolve_path(root: Path, raw_path: Any) -> Path:
    """Resolve an absolute, cwd-relative, or root-relative path."""
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    return Path(root) / path


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


def _dataset_root_from_runs(runs_by_method: dict[str, dict[str, VariableRun]]) -> Path:
    """Return the first resolvable packaged dataset root from method runs."""
    for runs_by_variable in runs_by_method.values():
        for run in runs_by_variable.values():
            root = _dataset_root_from_summary(run.summary)
            if root is not None:
                return root
    raise RuntimeError(
        "Could not resolve packaged GeoTIFF dataset root from run summaries."
    )


def load_dataset_context(dataset_root: Path) -> DatasetContext:
    """Load manifest, land mask, grid geometry, and depth axis for a dataset root."""
    root = Path(dataset_root)
    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = _load_yaml(manifest_path)
    depth_axis = np.asarray(manifest.get("depth_axis_m", ()), dtype=np.float64)
    if depth_axis.size < 1:
        raise RuntimeError(f"Manifest lacks depth_axis_m: {manifest_path}")
    raw_land_path = manifest.get("grid", {}).get("source") or (
        "masks/world_land_mask_glorys_0p1.tif"
    )
    land_mask_path = _resolve_path(root, raw_land_path)
    with rasterio.open(land_mask_path) as src:
        land = src.read(1)
        transform = src.transform
        crs = src.crs
        height = int(src.height)
        width = int(src.width)
    ocean_mask = np.asarray(land, dtype=np.float32) <= 0.5
    return DatasetContext(
        root=root,
        manifest=manifest,
        depth_axis_m=depth_axis,
        land_mask_path=land_mask_path,
        transform=transform,
        crs=crs,
        height=height,
        width=width,
        ocean_mask=ocean_mask,
    )


def _manifest_argo_path(context: DatasetContext) -> Path:
    """Return the compact EN4/ARGO profile zarr path from a dataset manifest."""
    raw_path = context.manifest.get("argo", {}).get("path")
    if raw_path is None:
        raise RuntimeError("Dataset manifest does not reference an ARGO profile zarr.")
    return _resolve_path(context.root, raw_path)


def _manifest_source_raster_path(
    context: DatasetContext, *, variable: str, date_value: int
) -> Path:
    """Return a packaged GLORYS source raster for one variable/date."""
    manifest_name = GLORYS_MANIFEST_VARIABLES[str(variable)]
    entries = (
        context.manifest.get("rasters", {}).get("glorys", {}).get(manifest_name, [])
    )
    for entry in entries:
        if int(entry.get("date")) == int(date_value):
            return _resolve_path(context.root, entry["path"])
    raise FileNotFoundError(
        f"No GLORYS source raster for {variable} date {int(date_value)}."
    )


def _stretch_from_tags_or_manifest(
    tags: dict[str, Any], manifest_stretch: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Normalize stretch metadata from GeoTIFF tags or manifest entries."""
    if "stretch_min" in tags and "stretch_max" in tags:
        return {
            "minimum": float(tags["stretch_min"]),
            "maximum": float(tags["stretch_max"]),
            "nodata": int(float(tags.get("nodata", 255))),
            "valid_code_max": float(tags.get("valid_code_max", 254.0)),
            "units": str(tags.get("stretch_units", "")),
            "name": str(tags.get("stretch_name", "")),
        }
    if manifest_stretch is not None and "minimum" in manifest_stretch:
        return dict(manifest_stretch)
    return None


def _decode_stretched_array(
    values: np.ndarray,
    stretch: dict[str, Any],
    *,
    variable: str,
) -> np.ndarray:
    """Decode a stretched uint8 array to physical units."""
    arr = np.asarray(values)
    nodata = int(stretch.get("nodata", 255))
    valid_code_max = float(stretch.get("valid_code_max", 254.0))
    minimum = float(stretch["minimum"])
    maximum = float(stretch["maximum"])
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.asarray(arr != nodata) & np.isfinite(arr.astype(np.float64))
    decoded = minimum + arr.astype(np.float32) / np.float32(
        valid_code_max
    ) * np.float32(maximum - minimum)
    units = str(stretch.get("units", "")).strip().upper()
    name = str(stretch.get("name", "")).strip().lower()
    if str(variable) == "temperature" and (units == "K" or "kelvin" in name):
        decoded = decoded - np.float32(CELSIUS_TO_KELVIN_OFFSET)
    out[valid] = decoded[valid]
    return out


def read_raster_band_physical(
    path: Path,
    *,
    variable: str,
    band_index: int = 1,
    manifest_stretch: dict[str, Any] | None = None,
) -> np.ndarray:
    """Read one raster band as physical float values with nodata set to NaN."""
    with rasterio.open(path) as src:
        data = src.read(int(band_index), masked=False)
        tags = src.tags()
        nodata = src.nodata
    stretch = _stretch_from_tags_or_manifest(tags, manifest_stretch)
    if np.issubdtype(data.dtype, np.integer) and stretch is not None:
        return _decode_stretched_array(data, stretch, variable=variable)
    out = data.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(float(nodata)):
        out = out.copy()
        out[np.isclose(out, float(nodata), atol=0.0, rtol=0.0)] = np.nan
    return out.astype(np.float32, copy=False)


def _write_float_multiband_tif(
    path: Path,
    *,
    context: DatasetContext,
    band_count: int,
) -> rasterio.io.DatasetWriter:
    """Open a float32 multiband GeoTIFF writer matching the dataset grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=context.height,
        width=context.width,
        count=int(band_count),
        dtype="float32",
        crs=context.crs,
        transform=context.transform,
        nodata=DEFAULT_NODATA,
        compress="deflate",
        predictor=3,
    )


def _eligible_profile_indices(
    store: ArgoGeoTIFFProfileStore, *, excluded_year: int
) -> np.ndarray:
    """Return profile indices outside the excluded validation year."""
    years = np.asarray(store.target_date, dtype=np.int32) // 10000
    return np.flatnonzero(years != int(excluded_year)).astype(np.int64)


def _decode_profile_values(
    encoded: np.ndarray,
    stretch: dict[str, Any],
    *,
    variable: str,
) -> np.ndarray:
    """Decode one chunk of profile values to physical units."""
    return _decode_stretched_array(encoded, stretch, variable=variable)


def _average_profile_band_to_grid(
    *,
    store: ArgoGeoTIFFProfileStore,
    variable: str,
    profile_indices: np.ndarray,
    depth_index: int,
    height: int,
    width: int,
    profile_chunk_size: int,
) -> tuple[np.ndarray, int]:
    """Average one EN4/ARGO profile depth band into a sparse grid."""
    value_name, valid_name = ARGO_VALUE_VARIABLES[variable]
    stretch = (
        store.temperature_stretch
        if variable == "temperature"
        else store.salinity_stretch
    )
    if stretch is None:
        raise RuntimeError(f"ARGO {variable} stretch metadata is unavailable.")
    flat_size = int(height) * int(width)
    sums = np.zeros((flat_size,), dtype=np.float64)
    counts = np.zeros((flat_size,), dtype=np.float64)
    group = store._ensure_zarr_group()
    profile_indices = np.asarray(profile_indices, dtype=np.int64).reshape(-1)
    for start in range(0, int(profile_indices.size), int(profile_chunk_size)):
        chunk = profile_indices[start : start + int(profile_chunk_size)]
        encoded = np.asarray(
            group[value_name].get_orthogonal_selection(
                (chunk, slice(int(depth_index), int(depth_index) + 1))
            )
        ).reshape(-1)
        valid = np.asarray(
            group[valid_name].get_orthogonal_selection(
                (chunk, slice(int(depth_index), int(depth_index) + 1))
            ),
            dtype=bool,
        ).reshape(-1)
        values = _decode_profile_values(encoded, stretch, variable=variable).reshape(-1)
        rows = np.asarray(store.grid_row[chunk], dtype=np.int64)
        cols = np.asarray(store.grid_col[chunk], dtype=np.int64)
        keep = (
            valid
            & np.isfinite(values)
            & (rows >= 0)
            & (rows < int(height))
            & (cols >= 0)
            & (cols < int(width))
        )
        if not np.any(keep):
            continue
        flat = rows[keep] * int(width) + cols[keep]
        sums += np.bincount(flat, weights=values[keep], minlength=flat_size)
        counts += np.bincount(flat, minlength=flat_size)
    averaged = np.full((flat_size,), np.nan, dtype=np.float32)
    valid_cells = counts > 0.0
    averaged[valid_cells] = (sums[valid_cells] / counts[valid_cells]).astype(
        np.float32, copy=False
    )
    return averaged.reshape(int(height), int(width)), int(np.sum(counts))


def idw_fill_2d(
    values: np.ndarray,
    *,
    target_mask: np.ndarray,
    power: float = DEFAULT_CLIMATOLOGY_IDW_POWER,
    eps: float = DEFAULT_CLIMATOLOGY_IDW_EPS,
    neighbors: int = DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    chunk_size: int = DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    periodic_x: bool = True,
) -> np.ndarray:
    """Fill a 2D field by k-nearest-neighbor inverse-distance weighting."""
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(target_mask, dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("values and target_mask must have the same 2D shape.")
    if float(power) <= 0.0:
        raise ValueError("IDW power must be > 0.")
    if float(eps) <= 0.0:
        raise ValueError("IDW eps must be > 0.")
    if int(neighbors) < 1:
        raise ValueError("IDW neighbors must be >= 1.")
    observed_mask = np.isfinite(data) & mask
    output = np.full(data.shape, np.nan, dtype=np.float32)
    if not np.any(observed_mask):
        return output
    observed_rows, observed_cols = np.nonzero(observed_mask)
    observed_values = data[observed_rows, observed_cols].astype(np.float64)
    coords = np.column_stack(
        [observed_rows.astype(np.float64), observed_cols.astype(np.float64)]
    )
    values_for_tree = observed_values
    if periodic_x:
        width = int(data.shape[1])
        coords = np.concatenate(
            [
                coords,
                coords + np.asarray([0.0, -float(width)]),
                coords + np.asarray([0.0, float(width)]),
            ],
            axis=0,
        )
        values_for_tree = np.tile(observed_values, 3)
    tree = cKDTree(coords)
    target_rows, target_cols = np.nonzero(mask)
    target_coords = np.column_stack(
        [target_rows.astype(np.float64), target_cols.astype(np.float64)]
    )
    k = min(int(neighbors), int(coords.shape[0]))
    for start in range(0, int(target_coords.shape[0]), int(chunk_size)):
        chunk = target_coords[start : start + int(chunk_size)]
        dist, idx = tree.query(chunk, k=k)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
        if dist.ndim == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        exact = dist[:, 0] <= float(eps)
        chunk_values = np.empty((int(chunk.shape[0]),), dtype=np.float32)
        if np.any(exact):
            chunk_values[exact] = values_for_tree[idx[exact, 0]].astype(np.float32)
        if np.any(~exact):
            safe_dist = np.maximum(dist[~exact], float(eps))
            weights = 1.0 / np.power(safe_dist, float(power))
            weighted = np.sum(weights * values_for_tree[idx[~exact]], axis=1)
            denom = np.sum(weights, axis=1)
            chunk_values[~exact] = (weighted / denom).astype(np.float32)
        rows = target_rows[start : start + int(chunk.shape[0])]
        cols = target_cols[start : start + int(chunk.shape[0])]
        output[rows, cols] = chunk_values
    return output


def build_climatology_artifacts(
    *,
    context: DatasetContext,
    output_dir: Path,
    excluded_year: int = DEFAULT_VALIDATION_YEAR,
    idw_power: float = DEFAULT_CLIMATOLOGY_IDW_POWER,
    idw_eps: float = DEFAULT_CLIMATOLOGY_IDW_EPS,
    idw_neighbors: int = DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    idw_chunk_size: int = DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    profile_chunk_size: int = DEFAULT_PROFILE_CHUNK_SIZE,
    overwrite: bool = False,
) -> ClimatologyArtifacts:
    """Build or load climatology baseline GeoTIFFs from non-validation EN4 data."""
    out_dir = Path(output_dir)
    temperature_path = out_dir / "climatology_temperature.tif"
    salinity_path = out_dir / "climatology_salinity.tif"
    summary_path = out_dir / "climatology_summary.json"
    if (
        not bool(overwrite)
        and temperature_path.exists()
        and salinity_path.exists()
        and summary_path.exists()
    ):
        return ClimatologyArtifacts(
            temperature_path=temperature_path,
            salinity_path=salinity_path,
            summary_path=summary_path,
            summary=_load_json(summary_path),
        )

    store = ArgoGeoTIFFProfileStore(_manifest_argo_path(context), include_salinity=True)
    try:
        profile_indices = _eligible_profile_indices(store, excluded_year=excluded_year)
        if profile_indices.size == 0:
            raise RuntimeError(
                f"No EN4/ARGO profiles remain after excluding {excluded_year}."
            )
        depth_axis = np.asarray(context.depth_axis_m, dtype=np.float64)
        valid_counts: dict[str, list[int]] = {"temperature": [], "salinity": []}
        paths = {"temperature": temperature_path, "salinity": salinity_path}
        for variable in VARIABLES:
            with _write_float_multiband_tif(
                paths[variable], context=context, band_count=int(depth_axis.size)
            ) as dst:
                dst.update_tags(
                    baseline="en4_climatology_idw",
                    excluded_year=int(excluded_year),
                    idw_power=float(idw_power),
                    idw_eps=float(idw_eps),
                    idw_neighbors=int(idw_neighbors),
                )
                for depth_idx, depth_m in enumerate(
                    tqdm(
                        depth_axis.tolist(),
                        desc=f"Building {variable} climatology",
                        unit="depth",
                    )
                ):
                    sparse, count = _average_profile_band_to_grid(
                        store=store,
                        variable=variable,
                        profile_indices=profile_indices,
                        depth_index=int(depth_idx),
                        height=context.height,
                        width=context.width,
                        profile_chunk_size=int(profile_chunk_size),
                    )
                    valid_counts[variable].append(int(count))
                    filled = idw_fill_2d(
                        sparse,
                        target_mask=context.ocean_mask,
                        power=float(idw_power),
                        eps=float(idw_eps),
                        neighbors=int(idw_neighbors),
                        chunk_size=int(idw_chunk_size),
                    )
                    write_array = np.where(
                        np.isfinite(filled), filled, np.float32(DEFAULT_NODATA)
                    )
                    dst.write(write_array.astype(np.float32, copy=False), depth_idx + 1)
                    dst.set_band_description(
                        depth_idx + 1,
                        f"{variable}_climatology_{float(depth_m):.3f}m",
                    )
                    dst.update_tags(
                        depth_idx + 1,
                        actual_depth_m=f"{float(depth_m):.6f}",
                        channel_index=int(depth_idx),
                    )
        summary = {
            "schema_version": 1,
            "kind": "en4_climatology_idw",
            "dataset_root": str(context.root),
            "excluded_year": int(excluded_year),
            "eligible_profile_count": int(profile_indices.size),
            "depth_axis_m": [float(value) for value in depth_axis.tolist()],
            "idw": {
                "power": float(idw_power),
                "eps": float(idw_eps),
                "neighbors": int(idw_neighbors),
                "chunk_size": int(idw_chunk_size),
            },
            "valid_observation_counts_by_depth": valid_counts,
            "artifacts": {
                "temperature": temperature_path.name,
                "salinity": salinity_path.name,
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    finally:
        store.close()
    return ClimatologyArtifacts(
        temperature_path=temperature_path,
        salinity_path=salinity_path,
        summary_path=summary_path,
        summary=summary,
    )


def build_week_climatology_artifacts(
    *,
    context: DatasetContext,
    output_dir: Path,
    date_value: int,
    holdout_df: pd.DataFrame | None = None,
    idw_power: float = DEFAULT_CLIMATOLOGY_IDW_POWER,
    idw_eps: float = DEFAULT_CLIMATOLOGY_IDW_EPS,
    idw_neighbors: int = DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    idw_chunk_size: int = DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    profile_chunk_size: int = DEFAULT_PROFILE_CHUNK_SIZE,
    overwrite: bool = False,
) -> ClimatologyArtifacts:
    """Build or load a same-week EN4/ARGO IDW baseline excluding held-out locations."""
    out_dir = Path(output_dir)
    temperature_path = out_dir / "climatology_temperature.tif"
    salinity_path = out_dir / "climatology_salinity.tif"
    summary_path = out_dir / "climatology_summary.json"
    if (
        not bool(overwrite)
        and temperature_path.exists()
        and salinity_path.exists()
        and summary_path.exists()
    ):
        summary = _load_json(summary_path)
        if summary.get("kind") == "en4_same_week_visible_argo_idw" and int(
            summary.get("date_value", -1)
        ) == int(date_value):
            return ClimatologyArtifacts(
                temperature_path=temperature_path,
                salinity_path=salinity_path,
                summary_path=summary_path,
                summary=summary,
            )

    holdout_keys: set[tuple[int, int, int]] = set()
    if holdout_df is not None and not holdout_df.empty:
        for row in holdout_df.itertuples(index=False):
            holdout_keys.add(
                (
                    int(getattr(row, "date")),
                    int(getattr(row, "grid_row")),
                    int(getattr(row, "grid_col")),
                )
            )

    store = ArgoGeoTIFFProfileStore(_manifest_argo_path(context), include_salinity=True)
    try:
        same_date_indices = np.flatnonzero(
            np.asarray(store.target_date, dtype=np.int32) == int(date_value)
        ).astype(np.int64)
        if same_date_indices.size == 0:
            raise RuntimeError(
                f"No EN4/ARGO profiles found for date {int(date_value)}."
            )
        if holdout_keys:
            rows = np.asarray(store.grid_row[same_date_indices], dtype=np.int64)
            cols = np.asarray(store.grid_col[same_date_indices], dtype=np.int64)
            keep = np.asarray(
                [
                    (int(date_value), int(row), int(col)) not in holdout_keys
                    for row, col in zip(rows.tolist(), cols.tolist(), strict=True)
                ],
                dtype=bool,
            )
            profile_indices = same_date_indices[keep]
        else:
            profile_indices = same_date_indices
        if profile_indices.size == 0:
            raise RuntimeError(
                "No same-week EN4/ARGO profiles remain after excluding held-out locations."
            )

        depth_axis = np.asarray(context.depth_axis_m, dtype=np.float64)
        valid_counts: dict[str, list[int]] = {"temperature": [], "salinity": []}
        paths = {"temperature": temperature_path, "salinity": salinity_path}
        for variable in VARIABLES:
            with _write_float_multiband_tif(
                paths[variable], context=context, band_count=int(depth_axis.size)
            ) as dst:
                dst.update_tags(
                    baseline="en4_same_week_visible_argo_idw",
                    date_value=int(date_value),
                    heldout_location_count=int(len(holdout_keys)),
                    idw_power=float(idw_power),
                    idw_eps=float(idw_eps),
                    idw_neighbors=int(idw_neighbors),
                )
                for depth_idx, depth_m in enumerate(
                    tqdm(
                        depth_axis.tolist(),
                        desc=f"Building {variable} same-week baseline",
                        unit="depth",
                    )
                ):
                    sparse, count = _average_profile_band_to_grid(
                        store=store,
                        variable=variable,
                        profile_indices=profile_indices,
                        depth_index=int(depth_idx),
                        height=context.height,
                        width=context.width,
                        profile_chunk_size=int(profile_chunk_size),
                    )
                    valid_counts[variable].append(int(count))
                    filled = idw_fill_2d(
                        sparse,
                        target_mask=context.ocean_mask,
                        power=float(idw_power),
                        eps=float(idw_eps),
                        neighbors=int(idw_neighbors),
                        chunk_size=int(idw_chunk_size),
                    )
                    write_array = np.where(
                        np.isfinite(filled), filled, np.float32(DEFAULT_NODATA)
                    )
                    dst.write(write_array.astype(np.float32, copy=False), depth_idx + 1)
                    dst.set_band_description(
                        depth_idx + 1,
                        f"{variable}_same_week_baseline_{float(depth_m):.3f}m",
                    )
                    dst.update_tags(
                        depth_idx + 1,
                        actual_depth_m=f"{float(depth_m):.6f}",
                        channel_index=int(depth_idx),
                    )
        summary = {
            "schema_version": 1,
            "kind": "en4_same_week_visible_argo_idw",
            "dataset_root": str(context.root),
            "date_value": int(date_value),
            "same_date_profile_count": int(same_date_indices.size),
            "used_profile_count": int(profile_indices.size),
            "heldout_location_count": int(len(holdout_keys)),
            "depth_axis_m": [float(value) for value in depth_axis.tolist()],
            "idw": {
                "power": float(idw_power),
                "eps": float(idw_eps),
                "neighbors": int(idw_neighbors),
                "chunk_size": int(idw_chunk_size),
            },
            "valid_observation_counts_by_depth": valid_counts,
            "artifacts": {
                "temperature": temperature_path.name,
                "salinity": salinity_path.name,
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    finally:
        store.close()
    return ClimatologyArtifacts(
        temperature_path=temperature_path,
        salinity_path=salinity_path,
        summary_path=summary_path,
        summary=summary,
    )


def resolve_climatology_artifacts(path: Path) -> ClimatologyArtifacts:
    """Resolve climatology artifact paths from a directory or summary JSON file."""
    raw_path = Path(path)
    summary_path = (
        raw_path / "climatology_summary.json" if raw_path.is_dir() else raw_path
    )
    summary = _load_json(summary_path)
    artifacts = summary.get("artifacts", {})
    temperature_path = _resolve_path(summary_path.parent, artifacts.get("temperature"))
    salinity_path = _resolve_path(summary_path.parent, artifacts.get("salinity"))
    for artifact_path in (temperature_path, salinity_path):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Climatology artifact not found: {artifact_path}")
    return ClimatologyArtifacts(
        temperature_path=temperature_path,
        salinity_path=salinity_path,
        summary_path=summary_path,
        summary=summary,
    )


def load_method_runs(
    method_run_dirs: dict[str, Path],
    *,
    method_labels: dict[str, str] | None = None,
) -> dict[str, dict[str, VariableRun]]:
    """Discover single-variable run summaries for each requested method."""
    runs_by_method: dict[str, dict[str, VariableRun]] = {}
    for method, run_dir in method_run_dirs.items():
        discovered = discover_variable_runs([Path(run_dir)], variables=VARIABLES)
        by_variable = {run.variable: run for run in discovered}
        missing = sorted(set(VARIABLES) - set(by_variable))
        if missing:
            raise RuntimeError(
                f"{_method_label(method, method_labels)} run is missing variable runs: {missing}."
            )
        runs_by_method[method] = by_variable
    return runs_by_method


def selected_date_from_runs(runs_by_method: dict[str, dict[str, VariableRun]]) -> int:
    """Return the shared selected date across all method runs."""
    dates: set[int] = set()
    for method_runs in runs_by_method.values():
        for run in method_runs.values():
            value = run.summary.get("selected_date") or run.summary.get("target_date")
            if value is not None:
                dates.add(int(value))
    if len(dates) != 1:
        raise RuntimeError(f"Expected one shared selected date, found {sorted(dates)}.")
    return int(next(iter(dates)))


def _validate_requested_week(date_value: int, *, year: int, iso_week: int) -> None:
    """Raise when a selected date does not belong to the requested ISO week."""
    text = str(int(date_value))
    parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    iso = parsed.isocalendar()
    if int(iso.year) != int(year) or int(iso.week) != int(iso_week):
        raise RuntimeError(
            f"Run selected date {date_value} is ISO {iso.year}-W{iso.week:02d}, "
            f"not requested {int(year)}-W{int(iso_week):02d}."
        )


def prediction_specs_by_method(
    runs_by_method: dict[str, dict[str, VariableRun]],
    *,
    depth_count: int,
    method_labels: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[int, DepthLayerSpec]]]:
    """Return prediction raster specs keyed by method, variable, and depth channel."""
    out: dict[str, dict[str, dict[int, DepthLayerSpec]]] = {}
    required = set(range(int(depth_count)))
    for method, runs_by_variable in runs_by_method.items():
        out[method] = {}
        for variable, run in runs_by_variable.items():
            specs = {
                int(spec.channel_index): spec
                for spec in _depth_layer_specs(run)
                if spec.layer == "prediction"
            }
            missing = sorted(required - set(specs))
            if missing:
                raise RuntimeError(
                    f"{_method_label(method, method_labels)} {variable} run must contain all "
                    f"{int(depth_count)} native depth prediction rasters; "
                    f"missing channel indices {missing[:10]}."
                )
            out[method][variable] = specs
    return out


def _metric_stats(prediction: np.ndarray, reference: np.ndarray) -> MetricStats:
    """Compute RMSE, MAE, and R² over finite paired values."""
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(ref)
    count = int(np.count_nonzero(valid))
    if count < 1:
        return MetricStats(count=0, rmse=None, mae=None, r2=None)
    diff = pred[valid] - ref[valid]
    sse = float(np.sum(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(sse / float(count)))
    ref_values = ref[valid]
    denom = float(np.sum((ref_values - float(np.mean(ref_values))) ** 2))
    r2 = None if denom <= 0.0 else float(1.0 - sse / denom)
    return MetricStats(count=count, rmse=rmse, mae=mae, r2=r2)


def _stats_record(
    *,
    method: str,
    target: str,
    variable: str,
    depth_index: int,
    depth_m: float,
    stats: MetricStats,
    method_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return one serializable metric row."""
    return {
        "method": method,
        "method_label": _method_label(method, method_labels),
        "target": target,
        "target_label": TARGET_LABELS[target],
        "variable": variable,
        "channel_index": int(depth_index),
        "depth_m": float(depth_m),
        "count": int(stats.count),
        "rmse": np.nan if stats.rmse is None else float(stats.rmse),
        "mae": np.nan if stats.mae is None else float(stats.mae),
        "r2": np.nan if stats.r2 is None else float(stats.r2),
    }


def select_en4_holdout_locations(
    *,
    context: DatasetContext,
    date_value: int,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Select deterministic held-out EN4/ARGO date-location profiles."""
    frac = float(fraction)
    if frac <= 0.0 or frac >= 1.0:
        raise ValueError("EN4 holdout fraction must be in (0, 1).")
    store = ArgoGeoTIFFProfileStore(_manifest_argo_path(context), include_salinity=True)
    try:
        all_indices = np.flatnonzero(
            np.asarray(store.target_date, dtype=np.int32) == int(date_value)
        ).astype(np.int64)
        if all_indices.size == 0:
            raise RuntimeError(f"No EN4/ARGO profiles found for date {date_value}.")
        group = store._ensure_zarr_group()
        temp_valid = np.asarray(
            group["argo_temp_valid"].get_orthogonal_selection(
                (all_indices, slice(None))
            ),
            dtype=bool,
        )
        sal_valid = np.asarray(
            group["argo_psal_valid"].get_orthogonal_selection(
                (all_indices, slice(None))
            ),
            dtype=bool,
        )
        # Keep EN4 holdout selection aligned with dataloader QC filtering.
        if store.filter_bad_quality:
            temp_valid &= store._quality_mask_for_variable("temp", indices=all_indices)
            sal_valid &= store._quality_mask_for_variable("psal", indices=all_indices)
        temp_counts = temp_valid.sum(axis=1).astype(np.int64)
        sal_counts = sal_valid.sum(axis=1).astype(np.int64)
        usable = (temp_counts > 0) | (sal_counts > 0)
        profile_indices = all_indices[usable]
        if profile_indices.size == 0:
            raise RuntimeError(
                f"No valid EN4/ARGO profiles found for date {date_value}."
            )
        rows = np.asarray(store.grid_row[profile_indices], dtype=np.int64)
        cols = np.asarray(store.grid_col[profile_indices], dtype=np.int64)
        loc_df = pd.DataFrame(
            {"date": int(date_value), "grid_row": rows, "grid_col": cols}
        ).drop_duplicates(ignore_index=True)
        holdout_count = int(round(float(len(loc_df)) * frac))
        holdout_count = min(max(holdout_count, 1), int(len(loc_df)))
        rng = np.random.default_rng(int(seed))
        selected_loc_indices = set(
            int(value)
            for value in rng.choice(
                np.arange(int(len(loc_df))), size=holdout_count, replace=False
            ).tolist()
        )
        selected_keys = {
            (
                int(loc_df.iloc[idx].date),
                int(loc_df.iloc[idx].grid_row),
                int(loc_df.iloc[idx].grid_col),
            )
            for idx in selected_loc_indices
        }
        records: list[dict[str, Any]] = []
        temp_counts = temp_counts[usable]
        sal_counts = sal_counts[usable]
        for local_idx, profile_idx in enumerate(profile_indices.tolist()):
            grid_row = int(rows[int(local_idx)])
            grid_col = int(cols[int(local_idx)])
            key = (int(date_value), grid_row, grid_col)
            if key not in selected_keys:
                continue
            lon, lat = transform_xy(
                context.transform, grid_row, grid_col, offset="center"
            )
            records.append(
                {
                    "date": int(date_value),
                    "grid_row": grid_row,
                    "grid_col": grid_col,
                    "lon": float(lon),
                    "lat": float(lat),
                    "profile_index": int(profile_idx),
                    "temperature_valid_depth_count": int(temp_counts[int(local_idx)]),
                    "salinity_valid_depth_count": int(sal_counts[int(local_idx)]),
                    "holdout_fraction": frac,
                    "split_seed": int(seed),
                }
            )
    finally:
        store.close()
    if not records:
        raise RuntimeError("EN4 holdout selection produced no profile records.")
    return pd.DataFrame.from_records(records).sort_values(
        ["date", "grid_row", "grid_col", "profile_index"]
    )


def _load_holdout_profiles(
    *,
    context: DatasetContext,
    holdout_df: pd.DataFrame,
    variable: str,
) -> np.ndarray:
    """Load held-out EN4/ARGO profile values for one variable."""
    indices = holdout_df["profile_index"].to_numpy(dtype=np.int64)
    store = ArgoGeoTIFFProfileStore(_manifest_argo_path(context), include_salinity=True)
    try:
        if variable == "temperature":
            return store.load_temperature_profiles(indices)
        if variable == "salinity":
            return store.load_salinity_profiles(indices)
    finally:
        store.close()
    raise ValueError(f"Unsupported variable: {variable}")


def write_en4_holdout_profiles_csv(
    *,
    context: DatasetContext,
    holdout_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write held-out EN4/ARGO profile values as a tidy metrics CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_axis = np.asarray(context.depth_axis_m, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for variable in VARIABLES:
        profiles = _load_holdout_profiles(
            context=context,
            holdout_df=holdout_df,
            variable=variable,
        )
        for profile_row, (_, holdout_row) in enumerate(holdout_df.iterrows()):
            for channel_index, depth_m in enumerate(depth_axis.tolist()):
                rows.append(
                    {
                        "date": int(holdout_row["date"]),
                        "grid_row": int(holdout_row["grid_row"]),
                        "grid_col": int(holdout_row["grid_col"]),
                        "profile_index": int(holdout_row["profile_index"]),
                        "variable": variable,
                        "channel_index": int(channel_index),
                        "depth_m": float(depth_m),
                        "value": float(profiles[int(profile_row), int(channel_index)]),
                    }
                )
    pd.DataFrame.from_records(rows).to_csv(output_path, index=False)
    return output_path


def _profiles_from_holdout_csv(
    *,
    holdout_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    variable: str,
    depth_count: int,
) -> np.ndarray:
    """Return a profile matrix from a tidy held-out EN4 profile CSV."""
    subset = profiles_df[profiles_df["variable"] == str(variable)]
    if subset.empty:
        raise RuntimeError(f"Holdout profile CSV has no rows for {variable}.")
    matrix = np.full((len(holdout_df), int(depth_count)), np.nan, dtype=np.float32)
    key_to_row = {
        int(profile_index): row_idx
        for row_idx, profile_index in enumerate(
            holdout_df["profile_index"].to_numpy(dtype=np.int64).tolist()
        )
    }
    for row in subset.itertuples(index=False):
        profile_index = int(getattr(row, "profile_index"))
        channel_index = int(getattr(row, "channel_index"))
        if (
            profile_index not in key_to_row
            or channel_index < 0
            or channel_index >= depth_count
        ):
            continue
        matrix[key_to_row[profile_index], channel_index] = np.float32(
            getattr(row, "value")
        )
    return matrix


def _prediction_array_for_method(
    *,
    method: str,
    variable: str,
    depth_index: int,
    specs: dict[str, dict[str, dict[int, DepthLayerSpec]]],
    climatology: ClimatologyArtifacts,
) -> np.ndarray:
    """Read one prediction field for a method/variable/depth."""
    if method == "climatology":
        path = (
            climatology.temperature_path
            if variable == "temperature"
            else climatology.salinity_path
        )
        return read_raster_band_physical(
            path, variable=variable, band_index=depth_index + 1
        )
    spec = specs[method][variable][int(depth_index)]
    return read_raster_band_physical(
        spec.path,
        variable=variable,
        band_index=int(spec.band_index),
    )


def _glorys_reference_array(
    *,
    context: DatasetContext,
    selected_date: int,
    variable: str,
    depth_index: int,
    glorys_reference_specs: dict[str, dict[int, DepthLayerSpec]] | None = None,
) -> np.ndarray:
    """Read one dense GLORYS reference field from bundle refs or dataset rasters."""
    if glorys_reference_specs is not None:
        variable_specs = glorys_reference_specs.get(variable, {})
        reference_spec = variable_specs.get(int(depth_index))
        if reference_spec is None:
            raise RuntimeError(
                f"Paper bundle is missing persisted GLORYS {variable} "
                f"reference raster for native depth channel {int(depth_index)}."
            )
        return read_raster_band_physical(
            reference_spec.path,
            variable=variable,
            band_index=int(reference_spec.band_index),
        )

    source_path = _manifest_source_raster_path(
        context, variable=variable, date_value=int(selected_date)
    )
    stretch_name = "temperature_kelvin" if variable == "temperature" else "salinity"
    stretch = context.manifest.get("stretch", {}).get(stretch_name)
    return read_raster_band_physical(
        source_path,
        variable=variable,
        band_index=int(depth_index) + 1,
        manifest_stretch=stretch,
    )


def compute_glorys_metrics(
    *,
    context: DatasetContext,
    selected_date: int,
    specs: dict[str, dict[str, dict[int, DepthLayerSpec]]],
    climatology: ClimatologyArtifacts,
    method_order: Sequence[str] = METHOD_ORDER,
    method_labels: dict[str, str] | None = None,
    glorys_reference_specs: dict[str, dict[int, DepthLayerSpec]] | None = None,
) -> pd.DataFrame:
    """Compute dense-field metrics against GLORYS12 source rasters."""
    rows: list[dict[str, Any]] = []
    depth_axis = np.asarray(context.depth_axis_m, dtype=np.float64)
    for variable in VARIABLES:
        for depth_index, depth_m in enumerate(depth_axis.tolist()):
            reference = _glorys_reference_array(
                context=context,
                selected_date=int(selected_date),
                variable=variable,
                depth_index=int(depth_index),
                glorys_reference_specs=glorys_reference_specs,
            )
            reference = np.where(context.ocean_mask, reference, np.nan)
            for method in method_order:
                prediction = _prediction_array_for_method(
                    method=method,
                    variable=variable,
                    depth_index=int(depth_index),
                    specs=specs,
                    climatology=climatology,
                )
                prediction = np.where(context.ocean_mask, prediction, np.nan)
                rows.append(
                    _stats_record(
                        method=method,
                        target="glorys",
                        variable=variable,
                        depth_index=int(depth_index),
                        depth_m=float(depth_m),
                        stats=_metric_stats(prediction, reference),
                        method_labels=method_labels,
                    )
                )
    return pd.DataFrame.from_records(rows)


def compute_en4_holdout_metrics(
    *,
    context: DatasetContext,
    holdout_df: pd.DataFrame,
    specs: dict[str, dict[str, dict[int, DepthLayerSpec]]],
    climatology: ClimatologyArtifacts,
    method_order: Sequence[str] = METHOD_ORDER,
    method_labels: dict[str, str] | None = None,
    holdout_profiles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute held-out EN4 profile metrics by method, variable, and depth."""
    rows: list[dict[str, Any]] = []
    grid_rows = holdout_df["grid_row"].to_numpy(dtype=np.int64)
    grid_cols = holdout_df["grid_col"].to_numpy(dtype=np.int64)
    depth_axis = np.asarray(context.depth_axis_m, dtype=np.float64)
    for variable in VARIABLES:
        if holdout_profiles_df is None:
            profiles = _load_holdout_profiles(
                context=context,
                holdout_df=holdout_df,
                variable=variable,
            )
        else:
            profiles = _profiles_from_holdout_csv(
                holdout_df=holdout_df,
                profiles_df=holdout_profiles_df,
                variable=variable,
                depth_count=int(depth_axis.size),
            )
        for depth_index, depth_m in enumerate(depth_axis.tolist()):
            reference = profiles[:, int(depth_index)]
            for method in method_order:
                prediction_raster = _prediction_array_for_method(
                    method=method,
                    variable=variable,
                    depth_index=int(depth_index),
                    specs=specs,
                    climatology=climatology,
                )
                prediction = np.full(reference.shape, np.nan, dtype=np.float32)
                in_bounds = (
                    (grid_rows >= 0)
                    & (grid_rows < int(prediction_raster.shape[0]))
                    & (grid_cols >= 0)
                    & (grid_cols < int(prediction_raster.shape[1]))
                )
                prediction[in_bounds] = prediction_raster[
                    grid_rows[in_bounds], grid_cols[in_bounds]
                ]
                rows.append(
                    _stats_record(
                        method=method,
                        target="en4",
                        variable=variable,
                        depth_index=int(depth_index),
                        depth_m=float(depth_m),
                        stats=_metric_stats(prediction, reference),
                        method_labels=method_labels,
                    )
                )
    return pd.DataFrame.from_records(rows)


def summarize_equal_depth_metrics(by_depth: pd.DataFrame) -> pd.DataFrame:
    """Average per-depth metrics equally for paper table cells."""
    rows: list[dict[str, Any]] = []
    group_cols = ["method", "method_label", "target", "target_label", "variable"]
    for keys, group in by_depth.groupby(group_cols, sort=False):
        values = dict(zip(group_cols, keys, strict=True))
        row: dict[str, Any] = {**values, "depth_count": int(len(group))}
        row["count"] = int(group["count"].sum())
        for metric in METRIC_ORDER:
            metric_values = group[metric].to_numpy(dtype=np.float64)
            finite = np.isfinite(metric_values)
            row[metric] = (
                float(np.mean(metric_values[finite])) if np.any(finite) else None
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _format_metric(value: Any) -> str:
    """Format one table metric value."""
    if value is None:
        return "--"
    try:
        if not np.isfinite(float(value)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    return f"{float(value):.3f}"


def write_latex_table(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    method_order: Sequence[str] | None = None,
) -> Path:
    """Write a LaTeX table matching the paper reconstruction layout."""
    ordered_methods = _ordered_methods(
        summary["method"].dropna().astype(str).unique().tolist(),
        preferred_order=method_order,
    )
    best: dict[tuple[str, str, str], str] = {}
    for target in TARGET_ORDER:
        for variable in VARIABLES:
            subset = summary[
                (summary["target"] == target) & (summary["variable"] == variable)
            ]
            for metric in METRIC_ORDER:
                values = subset[["method", metric]].dropna()
                values = values[np.isfinite(values[metric].astype(float))]
                if values.empty:
                    continue
                idx = (
                    values[metric].astype(float).idxmax()
                    if metric == "r2"
                    else values[metric].astype(float).idxmin()
                )
                best[(target, variable, metric)] = str(values.loc[idx, "method"])

    def cell(method: str, target: str, variable: str, metric: str) -> str:
        row = summary[
            (summary["method"] == method)
            & (summary["target"] == target)
            & (summary["variable"] == variable)
        ]
        value = None if row.empty else row.iloc[0][metric]
        text = _format_metric(value)
        if text != "--" and best.get((target, variable, metric)) == method:
            return rf"\textbf{{{text}}}"
        return text

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Subsurface reconstruction results} (evaluation year 2018). Performance is reported for temperature and salinity against two reference targets: dense GLORYS12 fields and held-out EN4 profiles, averaged across depth levels. Best values highlighted in bold.}",
        r"\label{tab:recon_results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lcccccccccccc@{}}",
        r"\toprule",
        r"& \multicolumn{6}{c}{\textbf{EN4 Validation Set}} & \multicolumn{6}{c}{\textbf{GLORYS12}} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}",
        r"& \multicolumn{3}{c}{\textbf{Temperature}} & \multicolumn{3}{c}{\textbf{Salinity}} & \multicolumn{3}{c}{\textbf{Temperature}} & \multicolumn{3}{c}{\textbf{Salinity}} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        r"\textbf{Method} & RMSE & MAE & $R^2$ & RMSE & MAE & $R^2$ & RMSE & MAE & $R^2$ & RMSE & MAE & $R^2$ \\",
        r"\midrule",
    ]
    for method in ordered_methods:
        label_rows = summary[summary["method"] == method]["method_label"].dropna()
        method_label = (
            str(label_rows.iloc[0]) if not label_rows.empty else _method_label(method)
        )
        row_values = []
        for target in TARGET_ORDER:
            for variable in VARIABLES:
                for metric in METRIC_ORDER:
                    row_values.append(cell(method, target, variable, metric))
        lines.append(f"{method_label} & " + " & ".join(row_values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _resolve_manifest_artifact(root: Path, raw_path: Any) -> Path | None:
    """Resolve a paper-week manifest artifact path."""
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if text == "" or text.lower() == "none":
        return None
    path = Path(text)
    return path if path.is_absolute() else Path(root) / path


def _load_paper_week_bundle(paper_run_dir: Path) -> dict[str, Any]:
    """Load a paper-week inference bundle manifest."""
    root = Path(paper_run_dir)
    manifest_path = root / "paper_week_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paper-week manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    methods = manifest.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise RuntimeError(f"Paper-week manifest has no methods: {manifest_path}")
    method_order = [str(method) for method in manifest.get("method_order", [])]
    if not method_order:
        method_order = list(methods)
    method_labels = {
        str(method): str(meta.get("label", _method_label(str(method))))
        for method, meta in methods.items()
        if isinstance(meta, dict)
    }
    method_run_dirs: dict[str, Path] = {}
    for method in method_order:
        meta = methods.get(method, {})
        if not isinstance(meta, dict):
            continue
        if str(meta.get("kind", "model")) == "climatology":
            continue
        run_dir = _resolve_manifest_artifact(root, meta.get("run_dir"))
        if run_dir is not None:
            method_run_dirs[method] = run_dir

    references = manifest.get("references", {})
    if not isinstance(references, dict):
        references = {}
    climatology_path = None
    climatology_meta = methods.get("climatology", {})
    if isinstance(climatology_meta, dict):
        climatology_path = _resolve_manifest_artifact(
            root,
            climatology_meta.get("climatology_summary_json")
            or climatology_meta.get("summary_path"),
        )
    if climatology_path is None:
        climatology_path = _resolve_manifest_artifact(
            root, references.get("climatology_summary_json")
        )

    holdout_locations_path = _resolve_manifest_artifact(
        root, references.get("en4_holdout_locations_csv")
    )
    holdout_profiles_path = _resolve_manifest_artifact(
        root, references.get("en4_holdout_profiles_csv")
    )

    glorys_reference_specs: dict[str, dict[int, DepthLayerSpec]] = {}
    glorys_refs = references.get("glorys", {})
    if isinstance(glorys_refs, dict):
        for variable, variable_refs in glorys_refs.items():
            exports = (
                variable_refs.get("depth_exports", [])
                if isinstance(variable_refs, dict)
                else []
            )
            variable_specs: dict[int, DepthLayerSpec] = {}
            for raw_export in exports:
                if not isinstance(raw_export, dict):
                    continue
                path = _resolve_manifest_artifact(root, raw_export.get("path"))
                if path is None:
                    continue
                channel_index = int(raw_export.get("channel_index", 0))
                variable_specs[channel_index] = DepthLayerSpec(
                    variable=str(variable),
                    layer="glorys",
                    suffix=str(raw_export.get("suffix", f"depth_{channel_index:03d}")),
                    label=str(raw_export.get("label", f"Depth {channel_index}")),
                    requested_depth_m=float(
                        raw_export.get("requested_depth_m", np.nan)
                    ),
                    actual_depth_m=float(raw_export.get("actual_depth_m", np.nan)),
                    channel_index=channel_index,
                    path=path,
                    band_index=int(raw_export.get("band_index", 1)),
                )
            if variable_specs:
                glorys_reference_specs[str(variable)] = variable_specs

    return {
        "root": root,
        "manifest": manifest,
        "method_order": method_order,
        "method_labels": method_labels,
        "method_run_dirs": method_run_dirs,
        "climatology_path": climatology_path,
        "holdout_locations_path": holdout_locations_path,
        "holdout_profiles_path": holdout_profiles_path,
        "glorys_reference_specs": glorys_reference_specs,
    }


def _context_from_bundle_or_runs(
    *,
    bundle: dict[str, Any] | None,
    runs_by_method: dict[str, dict[str, VariableRun]],
) -> DatasetContext:
    """Resolve dataset context from a paper bundle or method run summaries."""
    if bundle is not None:
        dataset_root = bundle["manifest"].get("dataset_root")
        if dataset_root is not None:
            root = _resolve_manifest_artifact(bundle["root"], dataset_root)
            if root is not None:
                return load_dataset_context(root)
    return load_dataset_context(_dataset_root_from_runs(runs_by_method))


def export_paper_metrics(
    *,
    output_dir: Path,
    year: int | None = None,
    iso_week: int | None = None,
    paper_run_dir: Path | None = None,
    idw_run_dir: Path | None = None,
    lstm_run_dir: Path | None = None,
    unet_run_dir: Path | None = None,
    climatology_path: Path | None = None,
    en4_holdout_fraction: float = 0.2,
    seed: int = 7,
    validation_year: int = DEFAULT_VALIDATION_YEAR,
    climatology_idw_power: float = DEFAULT_CLIMATOLOGY_IDW_POWER,
    climatology_idw_eps: float = DEFAULT_CLIMATOLOGY_IDW_EPS,
    climatology_idw_neighbors: int = DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    climatology_idw_chunk_size: int = DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    profile_chunk_size: int = DEFAULT_PROFILE_CHUNK_SIZE,
    overwrite_climatology: bool = False,
) -> dict[str, Any]:
    """Export paper metrics and table artifacts for one standard week."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_paper_week_bundle(Path(paper_run_dir)) if paper_run_dir else None

    if bundle is None:
        missing = [
            name
            for name, value in {
                "year": year,
                "iso_week": iso_week,
                "idw_run_dir": idw_run_dir,
                "lstm_run_dir": lstm_run_dir,
                "unet_run_dir": unet_run_dir,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Legacy paper metrics mode requires: " + ", ".join(missing)
            )
        method_order = list(METHOD_ORDER)
        method_labels = dict(METHOD_LABELS)
        method_run_dirs = {
            "idw": Path(idw_run_dir),
            "lstm": Path(lstm_run_dir),
            "unet": Path(unet_run_dir),
        }
        holdout_locations_path = None
        holdout_profiles_df = None
        glorys_reference_specs = None
    else:
        manifest = bundle["manifest"]
        year = int(manifest.get("year", year)) if year is None else int(year)
        iso_week = (
            int(manifest.get("iso_week", iso_week))
            if iso_week is None
            else int(iso_week)
        )
        validation_year = int(manifest.get("validation_year", validation_year))
        en4_holdout_fraction = float(
            manifest.get("en4_holdout_fraction", en4_holdout_fraction)
        )
        seed = int(manifest.get("seed", seed))
        method_labels = dict(bundle["method_labels"])
        method_order = _ordered_methods(
            bundle["method_order"], preferred_order=bundle["method_order"]
        )
        method_run_dirs = dict(bundle["method_run_dirs"])
        if climatology_path is None:
            climatology_path = bundle["climatology_path"]
        holdout_locations_path = bundle["holdout_locations_path"]
        holdout_profiles_path = bundle["holdout_profiles_path"]
        holdout_profiles_df = (
            None
            if holdout_profiles_path is None
            else pd.read_csv(holdout_profiles_path)
        )
        glorys_reference_specs = bundle["glorys_reference_specs"] or None

    runs_by_method = load_method_runs(
        method_run_dirs,
        method_labels=method_labels,
    )
    selected_date = selected_date_from_runs(runs_by_method)
    if bundle is not None and bundle["manifest"].get("selected_date") is not None:
        selected_date = int(bundle["manifest"]["selected_date"])
    _validate_requested_week(selected_date, year=int(year), iso_week=int(iso_week))
    context = _context_from_bundle_or_runs(bundle=bundle, runs_by_method=runs_by_method)

    if climatology_path is None:
        climatology = build_climatology_artifacts(
            context=context,
            output_dir=output_dir,
            excluded_year=int(validation_year),
            idw_power=float(climatology_idw_power),
            idw_eps=float(climatology_idw_eps),
            idw_neighbors=int(climatology_idw_neighbors),
            idw_chunk_size=int(climatology_idw_chunk_size),
            profile_chunk_size=int(profile_chunk_size),
            overwrite=bool(overwrite_climatology),
        )
    else:
        climatology = resolve_climatology_artifacts(Path(climatology_path))

    specs = prediction_specs_by_method(
        runs_by_method,
        depth_count=int(context.depth_axis_m.size),
        method_labels=method_labels,
    )
    if "climatology" in method_order:
        specs["climatology"] = {}
    method_order = [
        method for method in method_order if method == "climatology" or method in specs
    ]

    if holdout_locations_path is not None:
        holdout_df = pd.read_csv(holdout_locations_path)
    else:
        holdout_df = select_en4_holdout_locations(
            context=context,
            date_value=int(selected_date),
            fraction=float(en4_holdout_fraction),
            seed=int(seed),
        )
    holdout_path = output_dir / "en4_holdout_locations.csv"
    holdout_df.to_csv(holdout_path, index=False)

    glorys_metrics = compute_glorys_metrics(
        context=context,
        selected_date=int(selected_date),
        specs=specs,
        climatology=climatology,
        method_order=method_order,
        method_labels=method_labels,
        glorys_reference_specs=glorys_reference_specs,
    )
    en4_metrics = compute_en4_holdout_metrics(
        context=context,
        holdout_df=holdout_df,
        specs=specs,
        climatology=climatology,
        method_order=method_order,
        method_labels=method_labels,
        holdout_profiles_df=holdout_profiles_df,
    )
    glorys_path = output_dir / "glorys_field_metrics.csv"
    en4_path = output_dir / "en4_holdout_metrics.csv"
    by_depth_path = output_dir / "paper_metrics_by_depth.csv"
    summary_path = output_dir / "paper_metrics_summary.csv"
    glorys_metrics.to_csv(glorys_path, index=False)
    en4_metrics.to_csv(en4_path, index=False)
    by_depth = pd.concat([en4_metrics, glorys_metrics], ignore_index=True)
    by_depth.to_csv(by_depth_path, index=False)
    summary = summarize_equal_depth_metrics(by_depth)
    summary.to_csv(summary_path, index=False)
    table_path = write_latex_table(
        summary,
        output_dir / "recon_results_table.tex",
        method_order=method_order,
    )
    manifest = {
        "schema_version": 1,
        "kind": "paper_reconstruction_metrics",
        "paper_run_dir": None if paper_run_dir is None else str(paper_run_dir),
        "year": int(year),
        "iso_week": int(iso_week),
        "selected_date": int(selected_date),
        "validation_year": int(validation_year),
        "en4_holdout_fraction": float(en4_holdout_fraction),
        "seed": int(seed),
        "depth_averaging": "equal_depth_mean",
        "method_order": list(method_order),
        "method_labels": {
            method: _method_label(method, method_labels) for method in method_order
        },
        "run_dirs": {method: str(path) for method, path in method_run_dirs.items()},
        "artifacts": {
            "paper_metrics_summary_csv": summary_path.name,
            "paper_metrics_by_depth_csv": by_depth_path.name,
            "en4_holdout_metrics_csv": en4_path.name,
            "glorys_field_metrics_csv": glorys_path.name,
            "en4_holdout_locations_csv": holdout_path.name,
            "recon_results_table_tex": table_path.name,
            "climatology_summary_json": str(climatology.summary_path),
            "climatology_temperature_tif": str(climatology.temperature_path),
            "climatology_salinity_tif": str(climatology.salinity_path),
        },
    }
    manifest_path = output_dir / "paper_metrics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    manifest["artifacts"]["manifest_json"] = manifest_path.name
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the paper metrics CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export paper-ready reconstruction metrics for one standard week."
    )
    parser.add_argument(
        "--paper-run-dir",
        type=Path,
        default=None,
        help="Paper-week inference bundle directory with paper_week_manifest.json.",
    )
    parser.add_argument("--year", type=int, default=None, help="ISO year to evaluate.")
    parser.add_argument(
        "--iso-week", type=int, default=None, help="ISO week to evaluate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idw-run-dir", type=Path, default=None)
    parser.add_argument("--lstm-run-dir", type=Path, default=None)
    parser.add_argument("--unet-run-dir", type=Path, default=None)
    parser.add_argument(
        "--climatology-path",
        type=Path,
        default=None,
        help="Optional climatology directory or climatology_summary.json path.",
    )
    parser.add_argument(
        "--en4-holdout-fraction",
        type=float,
        default=0.2,
        help="Date-location EN4 holdout fraction used for profile validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument(
        "--climatology-idw-power",
        type=float,
        default=DEFAULT_CLIMATOLOGY_IDW_POWER,
    )
    parser.add_argument(
        "--climatology-idw-eps", type=float, default=DEFAULT_CLIMATOLOGY_IDW_EPS
    )
    parser.add_argument(
        "--climatology-idw-neighbors",
        type=int,
        default=DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    )
    parser.add_argument(
        "--climatology-idw-chunk-size",
        type=int,
        default=DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    )
    parser.add_argument(
        "--profile-chunk-size", type=int, default=DEFAULT_PROFILE_CHUNK_SIZE
    )
    parser.add_argument(
        "--overwrite-climatology",
        action="store_true",
        help="Rebuild climatology GeoTIFFs even when artifacts already exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the paper metrics CLI."""
    args = _build_parser().parse_args(argv)
    manifest = export_paper_metrics(
        paper_run_dir=args.paper_run_dir,
        year=args.year,
        iso_week=args.iso_week,
        output_dir=args.output_dir,
        idw_run_dir=args.idw_run_dir,
        lstm_run_dir=args.lstm_run_dir,
        unet_run_dir=args.unet_run_dir,
        climatology_path=args.climatology_path,
        en4_holdout_fraction=args.en4_holdout_fraction,
        seed=args.seed,
        validation_year=args.validation_year,
        climatology_idw_power=args.climatology_idw_power,
        climatology_idw_eps=args.climatology_idw_eps,
        climatology_idw_neighbors=args.climatology_idw_neighbors,
        climatology_idw_chunk_size=args.climatology_idw_chunk_size,
        profile_chunk_size=args.profile_chunk_size,
        overwrite_climatology=bool(args.overwrite_climatology),
    )
    print(
        "Wrote paper metrics: "
        f"{Path(args.output_dir) / manifest['artifacts']['paper_metrics_summary_csv']}"
    )


if __name__ == "__main__":
    main()
