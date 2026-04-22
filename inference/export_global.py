"""Run one global inference export for one selected daily snapshot.

This exporter selects every spatial patch for one exact date. When users
provide an ISO week/year, the exporter picks the earliest available date in
that week so the output stays one spatially complete raster rather than seven.

Typical CLI:
/work/envs/depth/bin/python inference/export_global.py \
  --data-config configs/px_space/data_ostia_argo_disk_actual.yaml \
  --year 2015 \
  --iso-week 25 \
  --checkpoint logs/selection/argo_in_glorys_target/last.ckpt \
  --device cuda \
  --export-ground-truth
  
  
Run full including push:
/work/envs/depth/bin/python inference/export_global.py \
  --data-config configs/px_space/data_ostia_argo_disk_actual.yaml \
  --year 2015 \
  --iso-week 25 \
  --checkpoint logs/selection/argo_in_glorys_target/last.ckpt \
  --device cuda \
  --export-ground-truth;
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_20150615 \
  --public-base-url https://pub-a0d604187e144d18a52f7c9e679577dc.r2.dev/inference_production/globe \
  --rclone-remote r2:depth-data/inference_production/globe \
  --rclone-sync-scope globe


"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from scipy import ndimage as ndi
import torch
from torch import nn
from tqdm import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datamodule import DepthTileDataModule
from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset
from inference.core import (
    build_model,
    choose_device,
    load_yaml,
    resolve_checkpoint_path,
)
from utils.normalizations import temperature_normalize
from utils.validation_denoise import save_glorys_profile_comparison_plot

DEFAULT_MODEL_CONFIG = "configs/px_space/model_config.yaml"
DEFAULT_DATA_CONFIG = "configs/px_space/data_ostia_argo_disk_actual.yaml"
DEFAULT_TRAIN_CONFIG = "configs/px_space/training_config.yaml"
DEFAULT_OUTPUT_ROOT = Path("inference/outputs")
DEFAULT_PRODUCTION_RUN_DIR_NAME = "inference_production"
DEFAULT_PRODUCTION_RUN_STEM = "global_top_band"
DEFAULT_FULL_SAMPLE_COUNT = 100


@dataclass(frozen=True)
class ExportSelection:
    selected_date: int
    iso_year: int
    iso_week: int
    indices: list[int]


@dataclass(frozen=True)
class MosaicLayout:
    left: float
    bottom: float
    right: float
    top: float
    pixel_width: float
    pixel_height: float
    width: int
    height: int
    patch_width: int
    patch_height: int
    transform: Affine


@dataclass
class RasterAccumulator:
    sum_path: Path
    count_path: Path
    sum_array: np.memmap
    count_array: np.memmap


@dataclass
class FullProfileSample:
    dataset_index: int
    row: dict[str, Any]
    point_row: int
    point_col: int
    patch_height: int
    patch_width: int
    lon: float
    lat: float
    x_profile_c: np.ndarray
    y_hat_profile_c: np.ndarray
    y_target_profile_c: np.ndarray
    observed_profile: np.ndarray
    target_valid_profile: np.ndarray


class ExportInferenceWrapper(nn.Module):
    """Thin wrapper so DataParallel can fan out batch-level inference."""

    def __init__(
        self,
        model: nn.Module,
        *,
        export_ground_truth: bool,
        export_full_prediction_stack: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.export_ground_truth = bool(export_ground_truth)
        self.export_full_prediction_stack = bool(export_full_prediction_stack)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = self.model.predict_step(batch, batch_idx=0)
        out: dict[str, torch.Tensor] = {
            "prediction_top_band": pred["y_hat_denorm"][:, 0].contiguous(),
        }
        if self.export_full_prediction_stack:
            out["prediction_full_stack"] = pred["y_hat_denorm"].contiguous()
        if self.export_ground_truth:
            target_denorm = temperature_normalize(mode="denorm", tensor=batch["y"])
            y_valid_mask = batch["y_valid_mask"]
            gt_top_band = target_denorm[:, 0]
            # Match the inference export mask semantics so both rasters share support.
            gt_top_band = torch.where(
                y_valid_mask[:, 0],
                gt_top_band,
                torch.full_like(gt_top_band, float("nan")),
            )
            out["ground_truth_top_band"] = gt_top_band.contiguous()
        return out


class GeoJSONPointWriter:
    """Stream GeoJSON point features to disk without keeping all features in memory."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._fh = None
        self._feature_count = 0

    @property
    def feature_count(self) -> int:
        return int(self._feature_count)

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")
        self._fh.write('{"type":"FeatureCollection","features":[\n')

    def write_feature(self, feature: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError(
                "GeoJSONPointWriter must be opened before writing features."
            )
        if self._feature_count > 0:
            self._fh.write(",\n")
        self._fh.write(json.dumps(feature, separators=(",", ":")))
        self._feature_count += 1

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.write("\n]}\n")
        self._fh.close()
        self._fh = None


def _split_label_for_row(row: dict[str, Any]) -> str | None:
    for key in ("split", "phase"):
        value = row.get(key)
        if value is None:
            continue
        label = str(value).strip().lower()
        if label in {"train", "val"}:
            return label
    return None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_run_stem(selected_date: int) -> str:
    return f"global_top_band_{int(selected_date)}"


def _production_run_stem() -> str:
    return DEFAULT_PRODUCTION_RUN_STEM


def _production_artifact_name(name: str, *, run_stem: str) -> str:
    prefix = f"{run_stem}_"
    if name.startswith(prefix):
        # Strip the date-stamped run stem so production files share one stable name.
        return f"{_production_run_stem()}_{name[len(prefix):]}"
    return name


def _rewrite_summary_for_production(
    summary_path: Path, *, production_dir: Path, run_stem: str
) -> None:
    if not summary_path.exists():
        return

    with summary_path.open("r", encoding="utf-8") as f:
        summary = yaml.safe_load(f) or {}
    if not isinstance(summary, dict):
        return

    for key in (
        "prediction_tif_path",
        "ground_truth_tif_path",
        "argo_points_geojson_path",
        "full_sample_locations_geojson_path",
        "patch_splits_geojson_path",
    ):
        value = summary.get(key)
        if isinstance(value, str):
            summary[key] = _production_artifact_name(value, run_stem=run_stem)

    summary["run_dir"] = str(production_dir)
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)


def _prepare_run_directory(
    output_root: Path,
    *,
    run_stem: str,
    output_name: str | None,
) -> tuple[Path, Path | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    # Default to a date-stamped run directory so repeated exports stay on disk.
    run_dir = output_root / (output_name if output_name is not None else run_stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, None


def _promote_production_run(staging_dir: Path, production_dir: Path) -> None:
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging run directory not found: {staging_dir}")

    backup_dir = production_dir.parent / f".{production_dir.name}.previous"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if production_dir.exists():
        # Keep the previous production export available until the staged replacement is ready.
        production_dir.replace(backup_dir)

    try:
        staging_dir.replace(production_dir)
    except Exception:
        if backup_dir.exists() and not production_dir.exists():
            backup_dir.replace(production_dir)
        raise

    summary_path = production_dir / "run_summary.yaml"
    selected_date = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = yaml.safe_load(f) or {}
        if isinstance(summary, dict) and summary.get("selected_date") is not None:
            selected_date = int(summary["selected_date"])

    if selected_date is not None:
        run_stem = _default_run_stem(selected_date)
        for child in sorted(production_dir.iterdir()):
            if not child.is_file():
                continue
            renamed_name = _production_artifact_name(child.name, run_stem=run_stem)
            if renamed_name == child.name:
                continue
            child.replace(production_dir / renamed_name)
        _rewrite_summary_for_production(
            summary_path,
            production_dir=production_dir,
            run_stem=run_stem,
        )

    shutil.rmtree(backup_dir, ignore_errors=True)


def _summary_artifact_path(path: Path) -> str:
    # Store run-local artifact references so staged production exports remain valid after promotion.
    return str(path.name)


def _parse_yyyymmdd(value: Any) -> date:
    return datetime.strptime(str(int(value)), "%Y%m%d").date()


def _date_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    # Keep the export order deterministic and geospatially easy to inspect.
    lat_top = max(float(row["lat0"]), float(row["lat1"]))
    lon_left = min(float(row["lon0"]), float(row["lon1"]))
    return (-lat_top, lon_left)


def select_export_indices(
    rows: Sequence[dict[str, Any]],
    *,
    exact_date: int | None = None,
    iso_year: int | None = None,
    iso_week: int | None = None,
) -> ExportSelection:
    if exact_date is not None and (iso_year is not None or iso_week is not None):
        raise ValueError("Use either --date or --year/--iso-week, not both.")
    if (iso_year is None) ^ (iso_week is None):
        raise ValueError("--year and --iso-week must be provided together.")

    parsed_dates = [(_parse_yyyymmdd(row["date"]), idx) for idx, row in enumerate(rows)]
    if not parsed_dates:
        raise RuntimeError("The manifest dataset is empty.")

    if exact_date is not None:
        selected = _parse_yyyymmdd(exact_date)
        matching_indices = [
            idx for sample_date, idx in parsed_dates if sample_date == selected
        ]
        if not matching_indices:
            raise RuntimeError(
                f"No manifest rows matched exact date {int(exact_date)}."
            )
    elif iso_year is not None and iso_week is not None:
        matching_dates = [
            sample_date
            for sample_date, _ in parsed_dates
            if sample_date.isocalendar()[:2] == (int(iso_year), int(iso_week))
        ]
        if not matching_dates:
            raise RuntimeError(
                f"No manifest rows matched ISO week {int(iso_year)}-W{int(iso_week):02d}."
            )
        # One ISO week can contain several daily snapshots in this manifest. Keep the
        # export single-raster by choosing the earliest date that week.
        selected = min(matching_dates)
        matching_indices = [
            idx for sample_date, idx in parsed_dates if sample_date == selected
        ]
    else:
        selected = min(sample_date for sample_date, _ in parsed_dates)
        matching_indices = [
            idx for sample_date, idx in parsed_dates if sample_date == selected
        ]

    sorted_indices = sorted(matching_indices, key=lambda idx: _date_sort_key(rows[idx]))
    iso_info = selected.isocalendar()
    return ExportSelection(
        selected_date=int(selected.strftime("%Y%m%d")),
        iso_year=int(iso_info[0]),
        iso_week=int(iso_info[1]),
        indices=sorted_indices,
    )


def _collate_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in samples[0]:
        first_value = samples[0][key]
        if torch.is_tensor(first_value):
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
            continue
        # Date values arrive as Python ints from the dataset and need explicit batching.
        batch[key] = torch.as_tensor([sample[key] for sample in samples])
    return batch


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _row_bounds(row: dict[str, Any]) -> tuple[float, float, float, float]:
    left = min(float(row["lon0"]), float(row["lon1"]))
    right = max(float(row["lon0"]), float(row["lon1"]))
    bottom = min(float(row["lat0"]), float(row["lat1"]))
    top = max(float(row["lat0"]), float(row["lat1"]))
    return left, bottom, right, top


def _point_lon_lat_for_pixel(
    *,
    row: dict[str, Any],
    patch_shape: tuple[int, int],
    point_row: int,
    point_col: int,
) -> tuple[float, float]:
    left, bottom, right, top = _row_bounds(row)
    patch_height, patch_width = patch_shape
    pixel_width = (right - left) / float(patch_width)
    pixel_height = (top - bottom) / float(patch_height)
    if pixel_width <= 0.0 or pixel_height <= 0.0:
        raise RuntimeError(
            "Encountered non-positive pixel resolution while building Argo points."
        )
    lon = left + (float(point_col) + 0.5) * pixel_width
    lat = top - (float(point_row) + 0.5) * pixel_height
    return lon, lat


def _point_feature_for_pixel(
    *,
    row: dict[str, Any],
    patch_shape: tuple[int, int],
    point_row: int,
    point_col: int,
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lon, lat = _point_lon_lat_for_pixel(
        row=row,
        patch_shape=patch_shape,
        point_row=point_row,
        point_col=point_col,
    )
    properties = {
        # Keep enough metadata to trace each observed Argo point back to
        # the timestep and source patch without bloating the GeoJSON.
        "date": int(row["date"]),
        "patch_id": str(row.get("patch_id", "")),
        "export_index": int(row.get("export_index", -1)),
        "pixel_row": int(point_row),
        "pixel_col": int(point_col),
    }
    if extra_properties is not None:
        properties.update(extra_properties)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


def _argo_point_features_for_patch(
    *,
    row: dict[str, Any],
    observed_mask_2d: np.ndarray,
) -> list[dict[str, Any]]:
    mask = np.asarray(observed_mask_2d, dtype=bool)
    if mask.ndim != 2:
        raise RuntimeError(f"Expected a 2D observed-mask patch, got {tuple(mask.shape)}.")

    features: list[dict[str, Any]] = []
    point_rows, point_cols = np.nonzero(mask)
    for point_row, point_col in zip(point_rows.tolist(), point_cols.tolist()):
        features.append(
            _point_feature_for_pixel(
                row=row,
                patch_shape=mask.shape,
                point_row=int(point_row),
                point_col=int(point_col),
            )
        )
    return features


def _patch_split_feature_for_row(row: dict[str, Any]) -> dict[str, Any] | None:
    split_label = _split_label_for_row(row)
    if split_label is None:
        return None

    left, bottom, right, top = _row_bounds(row)
    # GeoJSON polygon rings are lon/lat pairs and must repeat the first vertex at the end.
    ring = [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
        [left, top],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "date": int(row["date"]),
            "patch_id": str(row.get("patch_id", "")),
            "export_index": int(row.get("export_index", -1)),
            "split": split_label,
        },
    }


def _parse_depth_values_text(
    depth_text: str,
    *,
    expected_size: int,
) -> np.ndarray:
    depth_values: list[float] = []
    for token in str(depth_text).split("|"):
        token_stripped = token.strip()
        if token_stripped == "":
            continue
        depth_values.append(float(token_stripped))
    if int(len(depth_values)) != int(expected_size):
        return np.arange(int(expected_size), dtype=np.float64)
    return np.asarray(depth_values, dtype=np.float64)


def _load_glorys_depth_axis_m(
    dataset: OstiaArgoTiffDataset,
    row: dict[str, Any],
    *,
    expected_size: int,
) -> np.ndarray:
    glorys_path = dataset._resolve_index_path(str(row[dataset._glorys_path_col]))
    with rasterio.open(glorys_path) as ds:
        tags = ds.tags()
    depth_text = str(tags.get("glorys_depth_m", "")).strip()
    if depth_text == "":
        depth_text = str(tags.get("argo_glorys_depth_m", "")).strip()
    return _parse_depth_values_text(
        depth_text,
        expected_size=int(expected_size),
    )


def _profile_to_json_list(
    profile_values: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> list[float | None]:
    values = np.asarray(profile_values, dtype=np.float64).reshape(-1)
    keep_mask = np.isfinite(values)
    if valid_mask is not None:
        keep_mask &= np.asarray(valid_mask, dtype=bool).reshape(-1)
    return [
        None if not bool(is_valid) else float(value)
        for value, is_valid in zip(values.tolist(), keep_mask.tolist(), strict=False)
    ]


def _full_profile_feature_for_sample(
    *,
    sample: FullProfileSample,
    location_id: str,
    graph_png_path: str,
    depth_axis_m: np.ndarray,
) -> dict[str, Any]:
    return _point_feature_for_pixel(
        row=sample.row,
        patch_shape=(int(sample.patch_height), int(sample.patch_width)),
        point_row=sample.point_row,
        point_col=sample.point_col,
        extra_properties={
            "location_id": str(location_id),
            "graph_png_path": str(graph_png_path),
            "depth_m": _profile_to_json_list(depth_axis_m),
            "argo_profile_c": _profile_to_json_list(
                sample.x_profile_c,
                valid_mask=sample.observed_profile,
            ),
            "prediction_profile_c": _profile_to_json_list(
                sample.y_hat_profile_c,
                valid_mask=sample.target_valid_profile,
            ),
            "glorys_profile_c": _profile_to_json_list(
                sample.y_target_profile_c,
                valid_mask=sample.target_valid_profile,
            ),
        },
    )


def _maybe_store_full_profile_sample(
    *,
    samples: list[FullProfileSample],
    seen_count: int,
    limit: int,
    rng: np.random.Generator,
    candidate: FullProfileSample,
) -> None:
    if limit <= 0:
        return
    if len(samples) < limit:
        samples.append(candidate)
        return

    chosen_idx = int(rng.integers(0, int(seen_count)))
    if chosen_idx < limit:
        samples[chosen_idx] = candidate


def build_mosaic_layout(
    rows: Sequence[dict[str, Any]],
    *,
    patch_shape: tuple[int, int],
) -> MosaicLayout:
    if not rows:
        raise RuntimeError("No rows were selected for mosaicking.")

    patch_height, patch_width = patch_shape
    if patch_height <= 0 or patch_width <= 0:
        raise RuntimeError(f"Invalid patch shape {patch_shape}.")

    left0, bottom0, right0, top0 = _row_bounds(rows[0])
    pixel_width = (right0 - left0) / float(patch_width)
    pixel_height = (top0 - bottom0) / float(patch_height)
    if pixel_width <= 0.0 or pixel_height <= 0.0:
        raise RuntimeError(
            "Encountered non-positive pixel resolution while building the mosaic."
        )

    bounds = np.asarray([_row_bounds(row) for row in rows], dtype=np.float64)
    mosaic_left = float(bounds[:, 0].min())
    mosaic_bottom = float(bounds[:, 1].min())
    mosaic_right = float(bounds[:, 2].max())
    mosaic_top = float(bounds[:, 3].max())
    mosaic_width = int(round((mosaic_right - mosaic_left) / pixel_width))
    mosaic_height = int(round((mosaic_top - mosaic_bottom) / pixel_height))
    if mosaic_width <= 0 or mosaic_height <= 0:
        raise RuntimeError("Computed an invalid mosaic shape.")

    return MosaicLayout(
        left=mosaic_left,
        bottom=mosaic_bottom,
        right=mosaic_right,
        top=mosaic_top,
        pixel_width=float(pixel_width),
        pixel_height=float(pixel_height),
        width=int(mosaic_width),
        height=int(mosaic_height),
        patch_width=int(patch_width),
        patch_height=int(patch_height),
        transform=from_origin(mosaic_left, mosaic_top, pixel_width, pixel_height),
    )


def _window_for_row(row: dict[str, Any], layout: MosaicLayout) -> tuple[slice, slice]:
    patch_left, patch_bottom, patch_right, patch_top = _row_bounds(row)
    current_pixel_width = (patch_right - patch_left) / float(layout.patch_width)
    current_pixel_height = (patch_top - patch_bottom) / float(layout.patch_height)
    if not np.isclose(current_pixel_width, layout.pixel_width) or not np.isclose(
        current_pixel_height, layout.pixel_height
    ):
        raise RuntimeError("Selected patches do not share one spatial resolution.")

    row_off = int(round((layout.top - patch_top) / layout.pixel_height))
    col_off = int(round((patch_left - layout.left) / layout.pixel_width))
    return (
        slice(row_off, row_off + layout.patch_height),
        slice(col_off, col_off + layout.patch_width),
    )


def _accumulate_patch_into_arrays(
    sum_array: np.ndarray,
    count_array: np.ndarray,
    *,
    row: dict[str, Any],
    patch_values: np.ndarray,
    layout: MosaicLayout,
) -> None:
    patch = np.asarray(patch_values, dtype=np.float32)
    if patch.shape != (layout.patch_height, layout.patch_width):
        raise RuntimeError(
            "All selected patches must share one raster shape. "
            f"Expected {(layout.patch_height, layout.patch_width)}, got {tuple(patch.shape)}."
        )

    row_slice, col_slice = _window_for_row(row, layout)
    valid_mask = np.isfinite(patch)
    if not np.any(valid_mask):
        return

    # The accumulation buffers may live on disk via memmap. Update only the valid
    # pixels inside each patch window so we can average overlaps later without
    # storing every patch prediction in memory.
    sum_window = sum_array[row_slice, col_slice]
    count_window = count_array[row_slice, col_slice]
    sum_window[valid_mask] += patch[valid_mask]
    count_window[valid_mask] += 1


def build_global_mosaic(
    *,
    rows: Sequence[dict[str, Any]],
    top_band_predictions: Sequence[np.ndarray],
    nodata: float,
) -> tuple[np.ndarray, Affine]:
    if len(rows) != len(top_band_predictions):
        raise ValueError("rows and top_band_predictions must have the same length.")
    if not rows:
        raise RuntimeError("No rows were selected for mosaicking.")

    first_patch = np.asarray(top_band_predictions[0], dtype=np.float32)
    if first_patch.ndim != 2:
        raise RuntimeError(
            f"Expected each top-band prediction to be 2D, got {tuple(first_patch.shape)}."
        )

    layout = build_mosaic_layout(rows, patch_shape=first_patch.shape)
    mosaic_sum = np.zeros((layout.height, layout.width), dtype=np.float64)
    mosaic_count = np.zeros((layout.height, layout.width), dtype=np.uint16)
    for row, patch_pred in zip(rows, top_band_predictions):
        _accumulate_patch_into_arrays(
            mosaic_sum,
            mosaic_count,
            row=row,
            patch_values=patch_pred,
            layout=layout,
        )

    mosaic = np.full((layout.height, layout.width), float(nodata), dtype=np.float32)
    valid_output = mosaic_count > 0
    mosaic[valid_output] = (
        mosaic_sum[valid_output] / mosaic_count[valid_output].astype(np.float64)
    ).astype(np.float32, copy=False)
    return mosaic, layout.transform


def _repair_small_nodata_gaps_2d(
    image: np.ndarray,
    *,
    nodata: float,
    max_gap_width: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill tiny nodata seams in a single-band raster without touching valid pixels."""
    if max_gap_width < 1:
        raise ValueError("max_gap_width must be >= 1.")

    patch = np.asarray(image, dtype=np.float32)
    if patch.ndim != 2:
        raise RuntimeError(
            "Expected a 2D raster for nodata-gap repair, "
            f"got shape {tuple(patch.shape)}."
        )

    valid_mask = np.isfinite(patch)
    if np.isfinite(nodata):
        valid_mask &= ~np.isclose(patch, float(nodata), atol=0.0, rtol=0.0)

    if not np.any(valid_mask):
        return patch.copy(), np.zeros(patch.shape, dtype=bool)

    # Close the support mask separately along rows and columns so we can catch
    # narrow seams that run fully across the raster without touching the valid
    # pixels on either side.
    row_structure = np.ones((1, 2 * max_gap_width + 1), dtype=bool)
    col_structure = np.ones((2 * max_gap_width + 1, 1), dtype=bool)
    closed_mask = (
        ndi.binary_closing(valid_mask, structure=row_structure)
        | ndi.binary_closing(valid_mask, structure=col_structure)
    )
    repair_mask = closed_mask & ~valid_mask
    if not np.any(repair_mask):
        return patch.copy(), np.zeros(patch.shape, dtype=bool)

    _, nearest_indices = ndi.distance_transform_edt(~valid_mask, return_indices=True)
    repaired = patch.copy()
    repaired[repair_mask] = patch[
        nearest_indices[0][repair_mask],
        nearest_indices[1][repair_mask],
    ]
    return repaired, repair_mask


def _geotiff_block_size(dimension: int) -> int:
    """Pick a tile size that satisfies GDAL's 16-pixel block constraint."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1.")

    block_size = 16
    while block_size * 2 <= min(512, int(dimension)):
        block_size *= 2
    return block_size


def _overview_factors(width: int, height: int) -> list[int]:
    factors: list[int] = []
    factor = 2
    while min(width, height) // factor >= 256:
        factors.append(factor)
        factor *= 2
    return factors


def create_raster_accumulator(
    *,
    root_dir: Path,
    stem: str,
    layout: MosaicLayout,
) -> RasterAccumulator:
    root_dir.mkdir(parents=True, exist_ok=True)
    sum_path = root_dir / f"{stem}_sum.dat"
    count_path = root_dir / f"{stem}_count.dat"
    sum_array = np.memmap(
        sum_path,
        dtype=np.float64,
        mode="w+",
        shape=(layout.height, layout.width),
    )
    sum_array[:] = 0.0
    count_array = np.memmap(
        count_path,
        dtype=np.uint16,
        mode="w+",
        shape=(layout.height, layout.width),
    )
    count_array[:] = 0
    return RasterAccumulator(
        sum_path=sum_path,
        count_path=count_path,
        sum_array=sum_array,
        count_array=count_array,
    )


def _flush_accumulator(accumulator: RasterAccumulator) -> None:
    accumulator.sum_array.flush()
    accumulator.count_array.flush()


def _cleanup_accumulator(accumulator: RasterAccumulator) -> None:
    del accumulator.sum_array
    del accumulator.count_array
    accumulator.sum_path.unlink(missing_ok=True)
    accumulator.count_path.unlink(missing_ok=True)


def write_global_top_band_geotiff(
    *,
    output_path: Path,
    accumulator: RasterAccumulator,
    layout: MosaicLayout,
    nodata: float,
    band_description: str,
    tags: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(layout.height),
        "width": int(layout.width),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": layout.transform,
        "nodata": float(nodata),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": _geotiff_block_size(int(layout.width)),
        "blockysize": _geotiff_block_size(int(layout.height)),
        "BIGTIFF": "IF_SAFER",
    }

    block_height = min(1024, int(layout.height))
    with rasterio.open(output_path, "w", **profile) as ds:
        ds.set_band_description(1, band_description)
        ds.update_tags(**tags)
        for row_off in tqdm(
            range(0, layout.height, block_height),
            total=(layout.height + block_height - 1) // block_height,
            desc=f"Finalizing {output_path.name}",
            unit="chunk",
        ):
            row_stop = min(row_off + block_height, layout.height)
            sum_block = np.asarray(
                accumulator.sum_array[row_off:row_stop, :], dtype=np.float64
            )
            count_block = np.asarray(
                accumulator.count_array[row_off:row_stop, :],
                dtype=np.uint16,
            )
            out_block = np.full(sum_block.shape, float(nodata), dtype=np.float32)
            valid_mask = count_block > 0
            out_block[valid_mask] = (
                sum_block[valid_mask] / count_block[valid_mask].astype(np.float64)
            ).astype(np.float32, copy=False)
            ds.write(
                out_block,
                1,
                window=Window(
                    col_off=0,
                    row_off=row_off,
                    width=layout.width,
                    height=row_stop - row_off,
                ),
            )

    with rasterio.open(output_path, "r+") as ds:
        repaired_band, repaired_mask = _repair_small_nodata_gaps_2d(
            ds.read(1),
            nodata=nodata,
        )
        if np.any(repaired_mask):
            ds.write(repaired_band, 1)

    overview_factors = _overview_factors(layout.width, layout.height)
    if overview_factors:
        with rasterio.open(output_path, "r+") as ds:
            ds.build_overviews(overview_factors, Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DepthDif inference for one globally complete daily snapshot selected "
            "from the OSTIA/ARGO disk manifest, then export band 0 as one large GeoTIFF."
        )
    )
    parser.add_argument("--model-config", type=str, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--data-config", type=str, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--checkpoint-path",
        "--model-path-checkpoint",
        "--checkpoint",
        "--ckpt-path",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint override. Accepts the trained model .ckpt path "
            "and defaults to model config resolution."
        ),
    )
    parser.add_argument(
        "--date",
        type=int,
        default=None,
        help="Exact daily snapshot to export as YYYYMMDD.",
    )
    parser.add_argument("--year", type=int, default=None, help="ISO year filter.")
    parser.add_argument("--iso-week", type=int, default=None, help="ISO week filter.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "val"),
        help="Manifest split filter. Global raster export requires 'all'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to dataloader.val_batch_size then 4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device selection.",
    )
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.nn.DataParallel across all visible CUDA devices when available.",
    )
    parser.add_argument(
        "--export-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export the GLORYS top-band ground-truth raster for the selected timestep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Torch RNG seed so stochastic samplers remain reproducible.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory that receives the run folder and GeoTIFF export.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional run directory and filename stem. "
            "When omitted, the export uses the selected date as "
            "'global_top_band_<YYYYMMDD>'."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Nodata value stored in the exported GeoTIFF.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Load checkpoint weights with strict=True instead of the repo default strict=False.",
    )
    parser.add_argument(
        "--full-sample-count",
        type=int,
        default=DEFAULT_FULL_SAMPLE_COUNT,
        help=(
            "Number of random observed Argo locations whose full depth profiles are "
            "saved with graph PNGs for the globe viewer."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    if args.split != "all":
        raise ValueError(
            "Global raster export requires --split all so the selected timestep "
            "contains every spatial patch, not just train or val."
        )

    training_cfg = _load_yaml(args.train_config)
    dataset = OstiaArgoTiffDataset.from_config(args.data_config, split=args.split)
    dataset.return_info = False
    dataset.return_coords = True
    selection = select_export_indices(
        dataset._rows,
        exact_date=args.date,
        iso_year=args.year,
        iso_week=args.iso_week,
    )

    run_stem = (
        args.output_name
        if args.output_name is not None
        else _default_run_stem(selection.selected_date)
    )
    run_dir, production_dir = _prepare_run_directory(
        args.output_root,
        run_stem=run_stem,
        output_name=args.output_name,
    )

    selected_rows = [dataset._rows[idx] for idx in selection.indices]
    selected_manifest = pd.DataFrame.from_records(selected_rows)
    selected_manifest.to_csv(run_dir / "selected_patches.csv", index=False)

    model_cfg = load_yaml(args.model_config)
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training_cfg.get("dataloader", {}).get("val_batch_size", 4)
    )
    full_sample_count = int(max(0, int(args.full_sample_count)))
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    sample_for_shape = dataset[selection.indices[0]]
    patch_shape = tuple(int(v) for v in sample_for_shape["y"].shape[-2:])
    layout = build_mosaic_layout(selected_rows, patch_shape=patch_shape)
    scratch_dir = run_dir / ".scratch"
    pred_accumulator = create_raster_accumulator(
        root_dir=scratch_dir,
        stem="prediction_top_band",
        layout=layout,
    )
    gt_accumulator = (
        create_raster_accumulator(
            root_dir=scratch_dir,
            stem="ground_truth_top_band",
            layout=layout,
        )
        if bool(args.export_ground_truth)
        else None
    )
    argo_points_geojson_path = (
        run_dir / f"{run_stem}_argo_points.geojson"
        if bool(args.export_ground_truth)
        else None
    )
    full_sample_locations_geojson_path = (
        run_dir / f"{run_stem}_full_sample_locations.geojson"
        if full_sample_count > 0
        else None
    )
    patch_splits_geojson_path = run_dir / f"{run_stem}_patch_splits.geojson"
    argo_points_writer = (
        GeoJSONPointWriter(argo_points_geojson_path)
        if argo_points_geojson_path is not None
        else None
    )
    patch_splits_writer = GeoJSONPointWriter(patch_splits_geojson_path)
    sampled_full_profiles: list[FullProfileSample] = []
    full_sample_rng = np.random.default_rng(int(args.seed))
    observed_point_total = 0
    if argo_points_writer is not None:
        argo_points_writer.open()
    patch_splits_writer.open()

    print(
        "Preparing global export: "
        f"split={args.split}, "
        f"selected_date={selection.selected_date}, "
        f"iso_week={selection.iso_year}-W{selection.iso_week:02d}, "
        f"selected_patches={len(selection.indices)}, "
        f"batch_size={batch_size}, "
        f"export_ground_truth={bool(args.export_ground_truth)}, "
        f"full_sample_count={full_sample_count}"
    )

    device = choose_device(args.device)
    visible_gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = bool(
        args.multi_gpu and device.type == "cuda" and visible_gpu_count > 1
    )
    print(
        "Using device setup: "
        f"device={device}, "
        f"visible_gpus={visible_gpu_count}, "
        f"multi_gpu_enabled={use_multi_gpu}"
    )

    datamodule = DepthTileDataModule(
        dataset=dataset,
        val_dataset=dataset,
        dataloader_cfg={"batch_size": batch_size, "val_batch_size": batch_size},
    )
    model = build_model(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        training_config_path=args.train_config,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )

    ckpt_path = resolve_checkpoint_path(args.checkpoint_path, model_cfg)
    if ckpt_path is None:
        raise RuntimeError(
            "No checkpoint was resolved. Set --checkpoint-path or configure "
            "model.load_checkpoint/model.resume_checkpoint."
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=bool(args.strict_load))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    inference_module = ExportInferenceWrapper(
        model,
        export_ground_truth=bool(args.export_ground_truth),
        export_full_prediction_stack=(full_sample_count > 0),
    )
    if use_multi_gpu:
        inference_runner: nn.Module = nn.DataParallel(inference_module)
    else:
        inference_runner = inference_module

    total_batches = (len(selection.indices) + batch_size - 1) // batch_size
    progress = tqdm(
        range(0, len(selection.indices), batch_size),
        total=total_batches,
        desc="Running inference and streaming patches",
        unit="batch",
    )
    for start in progress:
        batch_indices = selection.indices[start : start + batch_size]
        batch_samples = [dataset[idx] for idx in batch_indices]
        batch = _collate_samples(batch_samples)
        model_batch = batch if use_multi_gpu else _to_device(batch, device)
        with torch.no_grad():
            outputs = inference_runner(model_batch)

        prediction_batch = outputs["prediction_top_band"].detach().float().cpu().numpy()
        prediction_full_stack_batch = (
            outputs["prediction_full_stack"].detach().float().cpu().numpy()
            if full_sample_count > 0
            else None
        )
        ground_truth_batch = (
            outputs["ground_truth_top_band"].detach().float().cpu().numpy()
            if bool(args.export_ground_truth)
            else None
        )
        x_denorm_batch = (
            temperature_normalize(mode="denorm", tensor=batch["x"])
            .detach()
            .float()
            .cpu()
            .numpy()
            if full_sample_count > 0
            else None
        )
        y_denorm_batch = (
            temperature_normalize(mode="denorm", tensor=batch["y"])
            .detach()
            .float()
            .cpu()
            .numpy()
            if full_sample_count > 0
            else None
        )
        selected_row_batch = selected_rows[start : start + len(batch_indices)]
        for local_idx, (dataset_idx, row) in enumerate(
            zip(batch_indices, selected_row_batch, strict=False)
        ):
            patch_split_feature = _patch_split_feature_for_row(row)
            if patch_split_feature is not None:
                patch_splits_writer.write_feature(patch_split_feature)
            _accumulate_patch_into_arrays(
                pred_accumulator.sum_array,
                pred_accumulator.count_array,
                row=row,
                patch_values=prediction_batch[local_idx],
                layout=layout,
            )
            if gt_accumulator is not None and ground_truth_batch is not None:
                _accumulate_patch_into_arrays(
                    gt_accumulator.sum_array,
                    gt_accumulator.count_array,
                    row=row,
                    patch_values=ground_truth_batch[local_idx],
                    layout=layout,
                )
            if argo_points_writer is not None:
                # Use the dataset's horizontal support mask so the globe shows one
                # marker per observed location instead of one marker per depth level.
                observed_mask_2d = (
                    batch["x_valid_mask_1d"][local_idx, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool, copy=False)
                )
                for feature in _argo_point_features_for_patch(
                    row=row,
                    observed_mask_2d=observed_mask_2d,
                ):
                    argo_points_writer.write_feature(feature)
            else:
                observed_mask_2d = (
                    batch["x_valid_mask_1d"][local_idx, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool, copy=False)
                )

            if (
                full_sample_count > 0
                and prediction_full_stack_batch is not None
                and x_denorm_batch is not None
                and y_denorm_batch is not None
            ):
                point_rows, point_cols = np.nonzero(observed_mask_2d)
                patch_shape = tuple(int(v) for v in observed_mask_2d.shape)
                for point_row, point_col in zip(
                    point_rows.tolist(), point_cols.tolist(), strict=False
                ):
                    observed_point_total += 1
                    lon, lat = _point_lon_lat_for_pixel(
                        row=row,
                        patch_shape=patch_shape,
                        point_row=int(point_row),
                        point_col=int(point_col),
                    )
                    candidate = FullProfileSample(
                        dataset_index=int(dataset_idx),
                        row=row,
                        point_row=int(point_row),
                        point_col=int(point_col),
                        patch_height=int(patch_shape[0]),
                        patch_width=int(patch_shape[1]),
                        lon=float(lon),
                        lat=float(lat),
                        x_profile_c=x_denorm_batch[local_idx, :, point_row, point_col].copy(),
                        y_hat_profile_c=prediction_full_stack_batch[
                            local_idx, :, point_row, point_col
                        ].copy(),
                        y_target_profile_c=y_denorm_batch[
                            local_idx, :, point_row, point_col
                        ].copy(),
                        observed_profile=batch["x_valid_mask"][
                            local_idx, :, point_row, point_col
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool, copy=False),
                        target_valid_profile=batch["y_valid_mask"][
                            local_idx, :, point_row, point_col
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool, copy=False),
                    )
                    _maybe_store_full_profile_sample(
                        samples=sampled_full_profiles,
                        seen_count=int(observed_point_total),
                        limit=full_sample_count,
                        rng=full_sample_rng,
                        candidate=candidate,
                    )

    _flush_accumulator(pred_accumulator)
    if gt_accumulator is not None:
        _flush_accumulator(gt_accumulator)
    if argo_points_writer is not None:
        argo_points_writer.close()
    patch_splits_writer.close()
    print(f"Finished inference for {len(selection.indices)} patches.")

    graphs_dir_path: Path | None = None
    full_sample_locations_writer: GeoJSONPointWriter | None = None
    if (
        full_sample_locations_geojson_path is not None
        and len(sampled_full_profiles) > 0
    ):
        graphs_dir_path = run_dir / "graphs"
        full_sample_locations_writer = GeoJSONPointWriter(
            full_sample_locations_geojson_path
        )
        full_sample_locations_writer.open()
        sampled_full_profiles_sorted = sorted(
            sampled_full_profiles,
            key=lambda sample: (
                int(sample.dataset_index),
                int(sample.point_row),
                int(sample.point_col),
            ),
        )
        for sample_idx, sample in enumerate(sampled_full_profiles_sorted, start=1):
            location_id = f"full_sample_{sample_idx:03d}"
            graph_rel_path = Path("graphs") / f"{location_id}.png"
            depth_axis_m = _load_glorys_depth_axis_m(
                dataset,
                sample.row,
                expected_size=int(sample.y_target_profile_c.shape[0]),
            )
            x_profile_plot = np.asarray(sample.x_profile_c, dtype=np.float64).copy()
            y_hat_profile_plot = np.asarray(sample.y_hat_profile_c, dtype=np.float64).copy()
            y_target_profile_plot = np.asarray(sample.y_target_profile_c, dtype=np.float64).copy()
            x_profile_plot[~sample.observed_profile] = np.nan
            y_hat_profile_plot[~sample.target_valid_profile] = np.nan
            y_target_profile_plot[~sample.target_valid_profile] = np.nan
            save_glorys_profile_comparison_plot(
                output_path=run_dir / graph_rel_path,
                x_profile=x_profile_plot,
                y_hat_profile=y_hat_profile_plot,
                y_target_profile=y_target_profile_plot,
                observed_profile=sample.observed_profile,
                depth_axis=depth_axis_m,
                title=f"Pixel ({sample.point_row}, {sample.point_col})",
                figure_title=(
                    f"{location_id}, patch={sample.row.get('patch_id', '')}, "
                    f"date={int(sample.row['date'])}"
                ),
            )
            full_sample_locations_writer.write_feature(
                _full_profile_feature_for_sample(
                    sample=sample,
                    location_id=location_id,
                    graph_png_path=str(graph_rel_path).replace("\\", "/"),
                    depth_axis_m=depth_axis_m,
                )
            )
        full_sample_locations_writer.close()
    else:
        full_sample_locations_geojson_path = None

    prediction_tif_path = run_dir / f"{run_stem}_prediction.tif"
    write_global_top_band_geotiff(
        output_path=prediction_tif_path,
        accumulator=pred_accumulator,
        layout=layout,
        nodata=float(args.nodata),
        band_description="predicted_top_band_celsius",
        tags={
            "source": "DepthDif global weekly inference export",
            "selected_date": str(int(selection.selected_date)),
            "checkpoint_path": str(ckpt_path),
            "selected_patch_count": str(int(len(selection.indices))),
            "kind": "prediction",
        },
    )
    print(f"Wrote prediction GeoTIFF: {prediction_tif_path}")

    ground_truth_tif_path: Path | None = None
    if gt_accumulator is not None:
        ground_truth_tif_path = run_dir / f"{run_stem}_glorys_top_band.tif"
        write_global_top_band_geotiff(
            output_path=ground_truth_tif_path,
            accumulator=gt_accumulator,
            layout=layout,
            nodata=float(args.nodata),
            band_description="glorys_top_band_celsius",
            tags={
                "source": "DepthDif global weekly ground-truth export",
                "selected_date": str(int(selection.selected_date)),
                "selected_patch_count": str(int(len(selection.indices))),
                "kind": "ground_truth",
            },
        )
        print(f"Wrote GLORYS GeoTIFF: {ground_truth_tif_path}")
        print(f"Wrote Argo points GeoJSON: {argo_points_geojson_path}")
    if full_sample_locations_geojson_path is not None and graphs_dir_path is not None:
        print(f"Wrote full-sample locations GeoJSON: {full_sample_locations_geojson_path}")
        print(f"Wrote full-sample graphs directory: {graphs_dir_path}")
    print(f"Wrote patch split GeoJSON: {patch_splits_geojson_path}")

    run_summary = {
        "selected_date": int(selection.selected_date),
        "iso_year": int(selection.iso_year),
        "iso_week": int(selection.iso_week),
        "selected_patch_count": int(len(selection.indices)),
        "checkpoint_path": str(ckpt_path),
        "model_config": str(args.model_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "device": str(device),
        "visible_gpus": int(visible_gpu_count),
        "multi_gpu_enabled": bool(use_multi_gpu),
        "batch_size": int(batch_size),
        "split": str(args.split),
        "run_dir": (
            str(production_dir) if production_dir is not None else str(run_dir)
        ),
        "prediction_tif_path": _summary_artifact_path(prediction_tif_path),
        "ground_truth_tif_path": (
            None
            if ground_truth_tif_path is None
            else _summary_artifact_path(ground_truth_tif_path)
        ),
        "argo_points_geojson_path": (
            None
            if argo_points_geojson_path is None
            else _summary_artifact_path(argo_points_geojson_path)
        ),
        "argo_point_count": (
            0 if argo_points_writer is None else int(argo_points_writer.feature_count)
        ),
        "full_sample_count_requested": int(full_sample_count),
        "full_sample_locations_geojson_path": (
            None
            if full_sample_locations_geojson_path is None
            else _summary_artifact_path(full_sample_locations_geojson_path)
        ),
        "full_sample_location_count": (
            0
            if full_sample_locations_writer is None
            else int(full_sample_locations_writer.feature_count)
        ),
        "graphs_dir_path": (
            None if graphs_dir_path is None else str(graphs_dir_path.name)
        ),
        "patch_splits_geojson_path": _summary_artifact_path(patch_splits_geojson_path),
        "patch_split_count": int(patch_splits_writer.feature_count),
    }
    with (run_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    _cleanup_accumulator(pred_accumulator)
    if gt_accumulator is not None:
        _cleanup_accumulator(gt_accumulator)
    scratch_dir.rmdir()

    if production_dir is not None:
        _promote_production_run(run_dir, production_dir)
        run_dir = production_dir
        prediction_tif_path = run_dir / prediction_tif_path.name
        if ground_truth_tif_path is not None:
            ground_truth_tif_path = run_dir / ground_truth_tif_path.name
        if argo_points_geojson_path is not None:
            argo_points_geojson_path = run_dir / argo_points_geojson_path.name
        if full_sample_locations_geojson_path is not None:
            full_sample_locations_geojson_path = (
                run_dir / full_sample_locations_geojson_path.name
            )
        if graphs_dir_path is not None:
            graphs_dir_path = run_dir / graphs_dir_path.name
        patch_splits_geojson_path = run_dir / patch_splits_geojson_path.name

    print(
        "Export complete: "
        f"date={selection.selected_date}, "
        f"iso_week={selection.iso_year}-W{selection.iso_week:02d}, "
        f"patches={len(selection.indices)}, "
        f"prediction={prediction_tif_path}, "
        f"ground_truth={ground_truth_tif_path}, "
        f"argo_points={argo_points_geojson_path}, "
        f"full_sample_locations={full_sample_locations_geojson_path}, "
        f"graphs={graphs_dir_path}, "
        f"patch_splits={patch_splits_geojson_path}"
    )


if __name__ == "__main__":
    main()
