"""Run one global ISO-week inference export.

This exporter treats an ISO week as one globally complete Wednesday snapshot,
forces 75% overlapping land-mask-grid patches at inference time, stitches
overlaps with spatial weights, masks land pixels to nodata, and can package/upload
Cesium assets.

Typical CLI:
/work/envs/depth/bin/python -m depth_recon.inference.export_global \
  --scenario temperature \
  --year 2018 \
  --iso-week 25 \
  --split all \
  --checkpoint logs/selection/argo_in_glorys_target/last.ckpt \
  --device cuda \
  --sampler ddim \
  --ddim-steps 100 \
  --export-ground-truth \
  --sigma 0 \
  --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe \
  --rclone-remote r2:depth-data/inference_production/globe \
  --rclone-sync-scope globe
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.data.datamodule import DepthTileDataModule
from depth_recon.data.dataset_argo_geotiff_gridded import (
    DEFAULT_GEOTIFF_ROOT_DIR,
    DEFAULT_LAND_MASK_RELATIVE_PATH,
)
from depth_recon.inference.export_cesium_globe_assets import (
    DEFAULT_ABSOLUTE_ERROR_SCALE_MAX_PERCENTILE,
    DEFAULT_ABSOLUTE_ERROR_SCALE_MIN_PERCENTILE,
    DEFAULT_COLOR_SCALE_MAX_C,
    DEFAULT_COLOR_SCALE_MIN_C,
    DEFAULT_EXTRA_ZOOM_LEVELS,
    DEFAULT_RCLONE_SYNC_SCOPE,
    export_cesium_globe_assets,
)
from depth_recon.inference.export_error_analysis_dashboard import (
    DEFAULT_ANALYSIS_GRID_GEOJSON_NAME,
    DEFAULT_ANALYSIS_JSON_NAME,
    build_basin_depth_error_summary_payload_from_depth_arrays,
    build_error_analysis_payload_from_depth_arrays,
    write_analysis_grid_geojson,
)
from depth_recon.configs.config_resolver_pixel import (
    DEFAULT_PIXEL_INFERENCE_CONFIG_PATH,
    PIXEL_SCENARIOS,
    load_pixel_inference_config,
)
from depth_recon.inference.core import (
    INFERENCE_SAMPLERS,
    apply_inference_sampling_config,
    build_dataset,
    build_model,
    choose_device,
    load_checkpoint_weights,
    load_yaml,
    resolve_checkpoint_path,
)
from depth_recon.models.diffusion.DenoisingDiffusionProcess.samplers import (
    DDIM_Sampler,
)
from depth_recon.paths import resolve_config_path
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.validation_denoise import save_glorys_profile_comparison_plot

DEFAULT_INFERENCE_CONFIG = DEFAULT_PIXEL_INFERENCE_CONFIG_PATH
DEFAULT_OUTPUT_ROOT = Path("inference/outputs")
DEFAULT_PRODUCTION_RUN_DIR_NAME = "inference_production"
DEFAULT_PRODUCTION_RUN_STEM = "global_top_band"
# Keep hosted graph bundles bounded by default; negative values still request all
# observed ARGO locations for explicit full-profile exports.
DEFAULT_FULL_SAMPLE_COUNT = 1000
DEFAULT_PROFILE_GRAPH_WEBP_QUALITY = 95
DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA = 0.0
DEFAULT_EXPORT_GAUSSIAN_BLUR_KERNEL_SIZE = 3
PREDICTION_ZERO_ARTIFACT_EPSILON = 1.0e-6
DEFAULT_INFERENCE_NUM_WORKERS = 8
DEFAULT_INFERENCE_PREFETCH_FACTOR = 2
DEFAULT_UNCERTAINTY_NUM_SAMPLES = 20
DEFAULT_TEMPORAL_BASIN_DEPTH_ERRORS_JSON_NAME = "temporal-basin-depth-errors.json"
DEFAULT_DEPTH_EXPORT_REQUESTS = (
    ("surface", "Surface", 0.0),
    ("10m", "10m", 10.0),
    ("50m", "50m", 50.0),
    ("100m", "100m", 100.0),
    ("250m", "250m", 250.0),
    ("500m", "500m", 500.0),
    ("1000m", "1000m", 1000.0),
    ("2000m", "2000m", 2000.0),
    ("2500m", "2500m", 2500.0),
    ("5000m", "5000m", 5000.0),
)
MONTH_ABBREVIATIONS = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
DEFAULT_SALINITY_COLOR_SCALE_MIN = 30.0
DEFAULT_SALINITY_COLOR_SCALE_MAX = 40.0


@dataclass(frozen=True)
class ExportVariableSpec:
    """Export-time metadata for one predicted physical variable."""

    name: str
    label: str
    prediction_denorm_key: str
    x_key: str
    y_key: str
    x_valid_mask_key: str
    y_valid_mask_key: str
    x_valid_mask_1d_key: str
    ground_truth_loader_name: str
    normalize_fn: Callable[..., torch.Tensor]
    value_units: str
    value_unit_label: str
    value_space: str
    absolute_error_value_space: str
    band_description_unit: str
    color_scale_min: float
    color_scale_max: float
    color_palette: str
    profile_x_label: str
    error_x_label: str
    include_surface_context_marker: bool
    prediction_source_value_transform: str
    ground_truth_source_value_transform: str
    absolute_error_source_value_transform: str


EXPORT_VARIABLE_SPECS: dict[str, ExportVariableSpec] = {
    "temperature": ExportVariableSpec(
        name="temperature",
        label="Temperature",
        prediction_denorm_key="y_hat_temperature_denorm",
        x_key="x",
        y_key="y",
        x_valid_mask_key="x_valid_mask",
        y_valid_mask_key="y_valid_mask",
        x_valid_mask_1d_key="x_valid_mask_1d",
        ground_truth_loader_name="_load_y_patch",
        normalize_fn=temperature_normalize,
        value_units="degree_Celsius",
        value_unit_label="deg C",
        value_space="denormalized_dequantized_celsius",
        absolute_error_value_space="absolute_error_celsius",
        band_description_unit="celsius",
        color_scale_min=DEFAULT_COLOR_SCALE_MIN_C,
        color_scale_max=DEFAULT_COLOR_SCALE_MAX_C,
        color_palette="temperature_blue_red",
        profile_x_label="Temperature (deg C)",
        error_x_label="Absolute error (deg C)",
        include_surface_context_marker=True,
        prediction_source_value_transform="model_prediction_denormalized_to_celsius",
        ground_truth_source_value_transform="source_glorys_decoded_dequantized_to_celsius",
        absolute_error_source_value_transform="abs(prediction_celsius_minus_glorys_celsius)",
    ),
    "salinity": ExportVariableSpec(
        name="salinity",
        label="Salinity",
        prediction_denorm_key="y_hat_salinity_denorm",
        x_key="x_salinity",
        y_key="y_salinity",
        x_valid_mask_key="x_salinity_valid_mask",
        y_valid_mask_key="y_salinity_valid_mask",
        x_valid_mask_1d_key="x_salinity_valid_mask_1d",
        ground_truth_loader_name="_load_y_salinity_patch",
        normalize_fn=salinity_normalize,
        value_units="PSU",
        value_unit_label="PSU",
        value_space="denormalized_dequantized_psu",
        absolute_error_value_space="absolute_error_psu",
        band_description_unit="psu",
        color_scale_min=DEFAULT_SALINITY_COLOR_SCALE_MIN,
        color_scale_max=DEFAULT_SALINITY_COLOR_SCALE_MAX,
        color_palette="salinity_blue_green",
        profile_x_label="Salinity (PSU)",
        error_x_label="Absolute error (PSU)",
        include_surface_context_marker=False,
        prediction_source_value_transform="model_prediction_denormalized_to_psu",
        ground_truth_source_value_transform="source_glorys_decoded_dequantized_to_psu",
        absolute_error_source_value_transform="abs(prediction_psu_minus_glorys_psu)",
    ),
}


def resolve_export_variable_spec(model_cfg: dict[str, Any]) -> ExportVariableSpec:
    """Resolve the single physical variable exported by the configured model."""
    model_section = model_cfg.get("model", {})
    output_fields = model_section.get("output_fields")
    if output_fields is None:
        output_fields = [model_section.get("scenario", "temperature")]
    if isinstance(output_fields, str):
        output_field_list = [output_fields]
    else:
        output_field_list = list(output_fields)
    output_field_list = [str(field).strip().lower() for field in output_field_list]
    if len(output_field_list) != 1:
        raise ValueError(
            "Global single-variable export expects exactly one model.output_fields "
            f"entry, got {output_field_list}. Run separate temperature/salinity "
            "exports or use export_global_variables for production packaging."
        )
    field = output_field_list[0]
    if field not in EXPORT_VARIABLE_SPECS:
        supported = ", ".join(sorted(EXPORT_VARIABLE_SPECS))
        raise ValueError(
            f"Unsupported global export variable {field!r}. Supported variables: {supported}."
        )
    return EXPORT_VARIABLE_SPECS[field]


@dataclass(frozen=True)
class ExportSelection:
    selected_date: int
    iso_year: int
    iso_week: int
    indices: list[int]


@dataclass(frozen=True)
class ExportRunResult:
    """Paths and metadata returned by the callable global inference exporter."""

    run_dir: Path
    summary_path: Path
    prediction_tif_path: Path | None
    ground_truth_tif_path: Path | None
    selected_date: int
    iso_year: int
    iso_week: int
    selected_patch_count: int
    absolute_error_tif_path: Path | None = None
    uncertainty_tif_path: Path | None = None
    variable: str = "temperature"


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


@dataclass(frozen=True)
class DepthExportLevel:
    suffix: str
    label: str
    requested_depth_m: float
    actual_depth_m: float
    channel_index: int


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
    ostia_sst_c: float
    observed_profile: np.ndarray
    target_valid_profile: np.ndarray


class SelectedInferencePatchDataset(Dataset):
    """Dataset view over the exact patch indices selected for one export run."""

    def __init__(
        self,
        *,
        dataset: Any,
        indices: Sequence[int],
        rows: Sequence[dict[str, Any]],
    ) -> None:
        self.dataset = dataset
        self.indices = [int(idx) for idx in indices]
        self.rows = [dict(row) for row in rows]
        if len(self.indices) != len(self.rows):
            raise ValueError("indices and rows must have the same length.")

    def __len__(self) -> int:
        """Return the number of selected inference patches."""
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one selected patch and keep metadata needed for stitching."""
        local_idx = int(idx)
        dataset_idx = int(self.indices[local_idx])
        # Keep the original dataset index and row beside the tensor sample so
        # multi-worker loading can still feed the mosaic writer in grid order.
        return {
            "dataset_idx": dataset_idx,
            "row": self.rows[local_idx],
            "sample": self.dataset[dataset_idx],
        }


def _sampling_metadata_from_overrides(
    training_cfg: dict[str, Any],
    *,
    sampler: str | None,
    ddim_num_timesteps: int | None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve sampler metadata without mutating the caller's training config."""
    if sampler is None and ddim_num_timesteps is None and fallback is not None:
        return dict(fallback)
    training_copy = json.loads(json.dumps(training_cfg))
    return apply_inference_sampling_config(
        training_copy,
        sampler=sampler,
        ddim_num_timesteps=ddim_num_timesteps,
    )


def _build_sampler_for_metadata(
    model: nn.Module, metadata: dict[str, Any]
) -> nn.Module | None:
    """Build a sampler module matching resolved inference metadata."""
    sampler_name = str(metadata["sampler"]).strip().lower()
    if sampler_name == "ddpm":
        # The diffusion process already owns the full-chain DDPM sampler. Reuse it so
        # a DDPM override still works when the model default sampler is DDIM.
        return getattr(getattr(model, "model", None), "sampler", None)
    if sampler_name != "ddim":
        raise ValueError(f"Unsupported sampler metadata: {sampler_name!r}.")

    diffusion_process = getattr(model, "model", None)
    forward_process = getattr(diffusion_process, "forward_process", None)
    if forward_process is None or not hasattr(forward_process, "betas"):
        raise RuntimeError("Cannot build DDIM sampler without model diffusion betas.")
    train_betas = forward_process.betas.detach().clone()
    return DDIM_Sampler(
        num_timesteps=min(
            int(metadata["ddim_num_timesteps"]),
            int(train_betas.numel()),
        ),
        train_timesteps=int(train_betas.numel()),
        betas=train_betas,
        parameterization=str(getattr(diffusion_process, "parameterization", "epsilon")),
        clip_sample=False,
        eta=float(getattr(model, "val_ddim_eta", 0.0)),
        temperature=float(getattr(model, "val_ddim_temperature", 1.0)),
    )


class ExportInferenceWrapper(nn.Module):
    """Thin wrapper so DataParallel can fan out batch-level inference."""

    def __init__(
        self,
        model: nn.Module,
        *,
        variable_spec: ExportVariableSpec,
        export_ground_truth: bool,
        export_full_prediction_stack: bool,
        export_prediction: bool = True,
        export_uncertainty: bool = False,
        uncertainty_num_samples: int = DEFAULT_UNCERTAINTY_NUM_SAMPLES,
        uncertainty_sampler: nn.Module | None = None,
        collapse_uncertainty_channels: bool = True,
        depth_channel_indices: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.variable_spec = variable_spec
        self.export_ground_truth = bool(export_ground_truth)
        self.export_full_prediction_stack = bool(export_full_prediction_stack)
        self.export_prediction = bool(export_prediction)
        self.export_uncertainty = bool(export_uncertainty)
        self.uncertainty_num_samples = int(uncertainty_num_samples)
        self.uncertainty_sampler = uncertainty_sampler
        self.collapse_uncertainty_channels = bool(collapse_uncertainty_channels)
        self.depth_channel_indices = tuple(int(idx) for idx in depth_channel_indices)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.export_prediction or self.export_full_prediction_stack:
            pred = self.model.predict_step(batch, batch_idx=0)
            prediction_denorm = pred.get(self.variable_spec.prediction_denorm_key)
            if prediction_denorm is None:
                prediction_denorm = pred["y_hat_denorm"]
            if self.export_prediction:
                out["prediction_depth_stack"] = prediction_denorm[
                    :, self.depth_channel_indices
                ].contiguous()
            if self.export_full_prediction_stack:
                out["prediction_full_stack"] = prediction_denorm.contiguous()
        if self.export_ground_truth:
            target_denorm = self.variable_spec.normalize_fn(
                mode="denorm", tensor=batch[self.variable_spec.y_key]
            )
            y_valid_mask = batch[self.variable_spec.y_valid_mask_key]
            gt_depth_stack = target_denorm[:, self.depth_channel_indices]
            gt_depth_mask = y_valid_mask[:, self.depth_channel_indices]
            # Match the inference export mask semantics so both rasters share support.
            gt_depth_stack = torch.where(
                gt_depth_mask,
                gt_depth_stack,
                torch.full_like(gt_depth_stack, float("nan")),
            )
            out["ground_truth_depth_stack"] = gt_depth_stack.contiguous()
        if self.export_uncertainty:
            try:
                uncertainty = self.model.uncertainty_step(
                    batch,
                    batch_idx=0,
                    num_samples=int(self.uncertainty_num_samples),
                    sampler=self.uncertainty_sampler,
                    collapse_channels=bool(self.collapse_uncertainty_channels),
                )
            except TypeError:
                if not self.collapse_uncertainty_channels:
                    raise
                uncertainty = self.model.uncertainty_step(
                    batch,
                    batch_idx=0,
                    num_samples=int(self.uncertainty_num_samples),
                    sampler=self.uncertainty_sampler,
                )
            uncertainty_map = uncertainty.get(
                f"uncertainty_{self.variable_spec.name}",
                uncertainty.get("uncertainty"),
            )
            if uncertainty_map is None:
                raise RuntimeError(
                    "Model uncertainty_step did not return an uncertainty map "
                    f"for {self.variable_spec.name}."
                )
            if not self.collapse_uncertainty_channels:
                uncertainty_map = uncertainty_map[:, self.depth_channel_indices]
            out["uncertainty_map"] = uncertainty_map.contiguous()
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
    with resolve_config_path(path).open("r", encoding="utf-8") as f:
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
        "absolute_error_tif_path",
        "uncertainty_tif_path",
        "error_analysis_json_path",
        "argo_points_geojson_path",
        "full_sample_locations_geojson_path",
        "patch_splits_geojson_path",
    ):
        value = summary.get(key)
        if isinstance(value, str):
            summary[key] = _production_artifact_name(value, run_stem=run_stem)

    for depth_export in summary.get("depth_exports", []):
        if not isinstance(depth_export, dict):
            continue
        for key in (
            "prediction_tif_path",
            "ground_truth_tif_path",
            "absolute_error_tif_path",
        ):
            value = depth_export.get(key)
            if isinstance(value, str):
                depth_export[key] = _production_artifact_name(value, run_stem=run_stem)

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


def _iso_week_wednesday(iso_year: int, iso_week: int) -> date:
    """Return the Wednesday date for an ISO year/week pair."""
    return date.fromisocalendar(int(iso_year), int(iso_week), 3)


def _closest_available_iso_week_date(
    parsed_dates: Sequence[tuple[date, int]],
    *,
    iso_year: int,
    iso_week: int,
) -> date:
    """Return the available dataset date closest to the ISO-week Wednesday."""
    target = _iso_week_wednesday(int(iso_year), int(iso_week))
    candidate_dates = sorted(
        {
            sample_date
            for sample_date, _idx in parsed_dates
            if sample_date.isocalendar()[:2] == (int(iso_year), int(iso_week))
        }
    )
    if not candidate_dates:
        raise RuntimeError(
            "No dataset rows matched ISO week " f"{int(iso_year)}-W{int(iso_week):02d}."
        )
    return min(
        candidate_dates,
        key=lambda sample_date: (abs(sample_date - target), sample_date),
    )


def _iso_week_label_for_date(parsed_date: date) -> str:
    iso_year, iso_week, _iso_day = parsed_date.isocalendar()
    week_start = parsed_date - timedelta(days=parsed_date.isoweekday() - 1)
    month_counts: dict[int, int] = {}
    for offset in range(7):
        month = int((week_start + timedelta(days=offset)).month)
        month_counts[month] = month_counts.get(month, 0) + 1
    # ISO weeks span at most two months; the month with four or more days is the label.
    dominant_month = max(month_counts.items(), key=lambda item: item[1])[0]
    return f"ISO week {int(iso_year)}-W{int(iso_week):02d} ({MONTH_ABBREVIATIONS[dominant_month]})"


def _profile_graph_figure_title(*, sample_date: Any, lat: float, lon: float) -> str:
    parsed_date = _parse_yyyymmdd(sample_date)

    def _format_coord(value: float, *, positive_label: str, negative_label: str) -> str:
        direction = positive_label if float(value) >= 0.0 else negative_label
        return f"{abs(float(value)):.4f} deg {direction}"

    lat_text = _format_coord(lat, positive_label="N", negative_label="S")
    lon_text = _format_coord(lon, positive_label="E", negative_label="W")
    return f"Week: {_iso_week_label_for_date(parsed_date)}\nLocation: {lat_text}, {lon_text}"


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
        raise RuntimeError("The configured dataset has no row metadata.")

    if exact_date is not None:
        selected = _parse_yyyymmdd(exact_date)
        matching_indices = [
            idx for sample_date, idx in parsed_dates if sample_date == selected
        ]
        if not matching_indices:
            raise RuntimeError(f"No dataset rows matched exact date {int(exact_date)}.")
    elif iso_year is not None and iso_week is not None:
        selected = _closest_available_iso_week_date(
            parsed_dates,
            iso_year=int(iso_year),
            iso_week=int(iso_week),
        )
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


def _lon_intervals(lon0: float, lon1: float) -> list[tuple[float, float]]:
    """Return normalized lon intervals, splitting antimeridian crossings."""
    raw_width = abs(float(lon1) - float(lon0))
    if raw_width >= 360.0:
        return [(-180.0, 180.0)]
    left = float(((float(lon0) + 180.0) % 360.0) - 180.0)
    right = float(((float(lon1) + 180.0) % 360.0) - 180.0)
    if left <= right:
        return [(left, right)]
    return [(left, 180.0), (-180.0, right)]


def _lon_intervals_intersect(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> bool:
    """Return True when two normalized longitude interval groups overlap."""
    for first_left, first_right in first:
        for second_left, second_right in second:
            if max(float(first_left), float(second_left)) <= min(
                float(first_right), float(second_right)
            ):
                return True
    return False


def _row_intersects_rectangle(
    row: dict[str, Any],
    rectangle: Sequence[float],
) -> bool:
    """Return True when a patch row intersects a lon/lat rectangle."""
    if len(rectangle) != 4:
        raise ValueError("rectangle must contain lon_min, lat_min, lon_max, lat_max.")
    lon_min, lat_min, lon_max, lat_max = [float(value) for value in rectangle]
    row_left, row_bottom, row_right, row_top = _row_bounds(row)
    rect_bottom = min(lat_min, lat_max)
    rect_top = max(lat_min, lat_max)
    if max(float(row_bottom), rect_bottom) > min(float(row_top), rect_top):
        return False
    return _lon_intervals_intersect(
        _lon_intervals(row_left, row_right),
        _lon_intervals(lon_min, lon_max),
    )


def filter_selection_by_rectangle(
    rows: Sequence[dict[str, Any]],
    selection: ExportSelection,
    rectangle: Sequence[float] | None,
) -> ExportSelection:
    """Filter selected patch indices to those intersecting a rectangle."""
    if rectangle is None:
        return selection
    indices = [
        int(idx)
        for idx in selection.indices
        if _row_intersects_rectangle(rows[int(idx)], rectangle)
    ]
    if not indices:
        raise RuntimeError(
            "No selected inference patches intersected rectangle "
            f"{tuple(float(value) for value in rectangle)}."
        )
    return ExportSelection(
        selected_date=selection.selected_date,
        iso_year=selection.iso_year,
        iso_week=selection.iso_week,
        indices=indices,
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


def _collate_inference_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate selected inference items while preserving row metadata."""
    return {
        "dataset_indices": [int(item["dataset_idx"]) for item in items],
        "rows": [dict(item["row"]) for item in items],
        "batch": _collate_samples([item["sample"] for item in items]),
    }


def _build_inference_loader(
    *,
    dataset: Any,
    indices: Sequence[int],
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    """Build the optimized DataLoader used by global export inference."""
    selected_dataset = SelectedInferencePatchDataset(
        dataset=dataset,
        indices=indices,
        rows=rows,
    )
    worker_count = int(num_workers)
    loader_kwargs: dict[str, Any] = {
        "dataset": selected_dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "persistent_workers": worker_count > 0,
        "collate_fn": _collate_inference_items,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def _print_global_inference_settings(
    *,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    effective_land_mask_path: Path,
    effective_min_ocean_fraction: float,
    effective_patch_stride: int | None,
    batch_size: int,
    inference_num_workers: int,
    inference_prefetch_factor: int,
    requested_full_sample_count: int,
    uncertainty_only: bool,
    export_uncertainty: bool,
    uncertainty_num_samples: int,
) -> None:
    """Print a concise summary of the resolved global inference settings."""
    tile_size = int(_nested_cfg_value(data_cfg, "dataset.grid.tile_size", default=128))
    resolved_patch_stride = (
        max(1, tile_size // 4)
        if effective_patch_stride is None
        else int(effective_patch_stride)
    )
    training_section = training_cfg.get("training", {})
    noise_cfg = training_section.get("noise", {})
    validation_sampling_cfg = training_section.get("validation_sampling", {})
    sampler_name = str(validation_sampling_cfg.get("sampler", "ddpm"))
    diffusion_steps = int(noise_cfg.get("num_timesteps", 1000))
    ddim_steps = int(validation_sampling_cfg.get("ddim_num_timesteps", 200))
    overlap_fraction = 1.0 - (float(resolved_patch_stride) / float(tile_size))
    print(
        "\nGlobal inference settings:"
        f"\n  model_config: {args.model_config}"
        f"\n  data_config: {args.data_config}"
        f"\n  train_config: {args.train_config}"
        f"\n  inference_config: {args.inference_config}"
        f"\n  checkpoint: {args.checkpoint_path}"
        f"\n  iso_week: {int(args.year)}-W{int(args.iso_week):02d}"
        f"\n  requested_split: {args.split}"
        f"\n  tile_size: {tile_size}"
        f"\n  patch_stride: {resolved_patch_stride}"
        f"\n  patch_overlap_fraction: {overlap_fraction:.3f}"
        f"\n  min_ocean_fraction: {float(effective_min_ocean_fraction):.3f}"
        f"\n  land_mask_path: {effective_land_mask_path}"
        f"\n  batch_size: {int(batch_size)}"
        f"\n  inference_num_workers: {int(inference_num_workers)}"
        f"\n  inference_prefetch_factor: {int(inference_prefetch_factor)}"
        f"\n  sampler: {sampler_name}"
        f"\n  diffusion_num_timesteps: {diffusion_steps}"
        f"\n  ddim_num_timesteps: {ddim_steps}"
        f"\n  export_ground_truth: {bool(args.export_ground_truth) and not bool(uncertainty_only)}"
        f"\n  uncertainty_only: {bool(uncertainty_only)}"
        f"\n  export_uncertainty: {bool(export_uncertainty)}"
        f"\n  uncertainty_num_samples: {int(uncertainty_num_samples)}"
        f"\n  full_sample_count: {requested_full_sample_count}"
        f"\n  sigma: {float(args.sigma)}"
        f"\n  rectangle: {args.rectangle}\n",
        flush=True,
    )


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
        raise RuntimeError(
            f"Expected a 2D observed-mask patch, got {tuple(mask.shape)}."
        )

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


def _dataset_rows(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "rows"):
        return list(getattr(dataset, "rows"))
    if hasattr(dataset, "_rows"):
        return list(getattr(dataset, "_rows"))
    raise RuntimeError("Dataset does not expose rows metadata for global export.")


def _nested_cfg_value(mapping: dict[str, Any], path: str, *, default: Any) -> Any:
    """Read a nested config value from a mapping."""
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _inference_section(inference_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the root inference config section."""
    section = inference_cfg.get("inference", inference_cfg)
    return section if isinstance(section, dict) else {}


def _resolve_dataset_root_relative_path(
    data_cfg: dict[str, Any], raw_path: str | Path
) -> Path:
    """Resolve relative inference paths against the packaged dataset root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    root_dir = Path(
        _nested_cfg_value(
            data_cfg,
            "dataset.core.geotiff_root_dir",
            default=DEFAULT_GEOTIFF_ROOT_DIR,
        )
    )
    return root_dir / path


def global_inference_dataset_overrides(
    data_cfg: dict[str, Any],
    *,
    land_mask_path: str | Path,
    min_ocean_fraction: float = 0.05,
    patch_stride: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build forced dataset overrides and metadata for global inference."""
    tile_size = int(_nested_cfg_value(data_cfg, "dataset.grid.tile_size", default=128))
    effective_patch_stride = (
        max(1, tile_size // 4) if patch_stride is None else int(patch_stride)
    )
    if effective_patch_stride < 1:
        raise ValueError("inference.grid.patch_stride must be >= 1.")
    min_ocean_fraction = float(min_ocean_fraction)
    if not (0.0 <= min_ocean_fraction <= 1.0):
        raise ValueError("--min-ocean-fraction must be in [0, 1].")
    max_land_fraction = 1.0 - min_ocean_fraction
    overrides = {
        "grid": {
            "patch_grid_source": "land_mask",
            "land_mask_path": str(land_mask_path),
            "patch_stride": int(effective_patch_stride),
            "max_land_fraction": float(max_land_fraction),
        },
        "selection": {"require_argo_for_all": False},
    }
    metadata = {
        "tile_size": int(tile_size),
        "patch_stride": int(effective_patch_stride),
        "patch_overlap_fraction": 1.0
        - (float(effective_patch_stride) / float(tile_size)),
        "min_ocean_fraction": float(min_ocean_fraction),
        "max_land_fraction": float(max_land_fraction),
        "patch_grid_source": "land_mask",
        "land_mask_path": str(land_mask_path),
        "require_argo_for_all": False,
        "overlap_stitching": "weighted",
    }
    return overrides, metadata


def resolve_global_inference_dataset(
    source: Any,
    *,
    data_config_path: str,
    data_cfg: dict[str, Any],
    split: str,
    land_mask_path: str | Path,
    min_ocean_fraction: float = 0.05,
    patch_stride: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve an existing dataset/dataloader or build the raw-product dataset."""
    overrides, metadata = global_inference_dataset_overrides(
        data_cfg,
        land_mask_path=land_mask_path,
        min_ocean_fraction=min_ocean_fraction,
        patch_stride=patch_stride,
    )
    if source is None:
        dataset = build_dataset(
            data_config_path,
            data_cfg.get("dataset", {}),
            split=split,
            dataset_overrides=overrides,
        )
        return dataset, metadata
    if hasattr(source, "dataset"):
        return source.dataset, metadata
    return source, metadata


def _load_glorys_depth_axis_m(
    dataset: Any,
    row: dict[str, Any],
    *,
    expected_size: int,
) -> np.ndarray:
    if hasattr(dataset, "depth_axis_m"):
        depth_axis = np.asarray(getattr(dataset, "depth_axis_m"), dtype=np.float64)
        if int(depth_axis.size) != int(expected_size):
            raise RuntimeError(
                "Dataset depth_axis_m does not match sample channel count: "
                f"{int(depth_axis.size)} != {int(expected_size)}"
            )
        return depth_axis

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


def resolve_depth_export_levels(depth_axis_m: np.ndarray) -> list[DepthExportLevel]:
    depth_values = np.asarray(depth_axis_m, dtype=np.float64).reshape(-1)
    if depth_values.size < 1:
        raise RuntimeError("Cannot resolve export depths from an empty depth axis.")
    if not np.all(np.isfinite(depth_values)):
        raise RuntimeError("GLORYS depth axis must contain only finite values.")

    levels: list[DepthExportLevel] = []
    for suffix, label, requested_depth_m in DEFAULT_DEPTH_EXPORT_REQUESTS:
        # Preserve model/GLORYS channels exactly; only choose the nearest physical depth.
        channel_index = int(np.argmin(np.abs(depth_values - float(requested_depth_m))))
        levels.append(
            DepthExportLevel(
                suffix=str(suffix),
                label=str(label),
                requested_depth_m=float(requested_depth_m),
                actual_depth_m=float(depth_values[channel_index]),
                channel_index=channel_index,
            )
        )
    return levels


def _filter_depth_export_levels(
    levels: Sequence[DepthExportLevel],
    allowed_suffixes: Sequence[str] | None,
) -> list[DepthExportLevel]:
    """Return depth raster export levels restricted to optional suffixes."""
    if not allowed_suffixes:
        return list(levels)

    suffixes = [
        str(suffix).strip() for suffix in allowed_suffixes if str(suffix).strip()
    ]
    suffix_set = set(suffixes)
    filtered = [level for level in levels if level.suffix in suffix_set]
    found_suffixes = {level.suffix for level in filtered}
    missing = [suffix for suffix in suffixes if suffix not in found_suffixes]
    if missing:
        available = ", ".join(level.suffix for level in levels)
        raise ValueError(
            "Requested depth export suffixes are unavailable: "
            f"{', '.join(missing)}. Available suffixes: {available}."
        )
    return filtered


def _format_native_depth_label(depth_m: float, *, channel_index: int) -> str:
    """Return a compact dashboard label for a native depth channel."""
    depth_value = float(depth_m)
    if int(channel_index) == 0 or np.isclose(depth_value, 0.0):
        return "Surface"
    label = f"{depth_value:.1f}".rstrip("0").rstrip(".")
    return f"{label}m"


def resolve_full_depth_analysis_levels(
    depth_axis_m: np.ndarray,
) -> list[DepthExportLevel]:
    """Return one analysis depth level for every native model target channel."""
    depth_values = np.asarray(depth_axis_m, dtype=np.float64).reshape(-1)
    if depth_values.size < 1:
        raise RuntimeError("Cannot resolve analysis depths from an empty depth axis.")
    if not np.all(np.isfinite(depth_values)):
        raise RuntimeError("GLORYS depth axis must contain only finite values.")

    return [
        DepthExportLevel(
            suffix=f"depth_{channel_index:03d}",
            label=_format_native_depth_label(
                float(depth_m),
                channel_index=channel_index,
            ),
            requested_depth_m=float(depth_m),
            actual_depth_m=float(depth_m),
            channel_index=int(channel_index),
        )
        for channel_index, depth_m in enumerate(depth_values.tolist())
    ]


def _absolute_error_array_from_signed_accumulator(
    accumulator: RasterAccumulator,
) -> np.ndarray:
    """Finalize one signed-error accumulator to absolute stitched error."""
    counts = np.asarray(accumulator.count_array, dtype=np.float64)
    values = np.full(counts.shape, np.nan, dtype=np.float32)
    valid = counts > 0.0
    if np.any(valid):
        signed_mean = (
            np.asarray(accumulator.sum_array, dtype=np.float64)[valid] / counts[valid]
        )
        values[valid] = np.abs(signed_mean).astype(np.float32, copy=False)
    return values


def _write_depth_error_analysis_json(
    *,
    output_path: Path,
    run_summary: dict[str, Any],
    variable_spec: ExportVariableSpec,
    analysis_depth_levels: Sequence[DepthExportLevel],
    signed_error_accumulators: dict[str, RasterAccumulator],
    layout: MosaicLayout,
    land_mask: np.ndarray | None,
) -> Path:
    """Write selected-depth error analysis JSON from signed-error mosaics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_metadata = [
        {
            "suffix": level.suffix,
            "label": level.label,
            "requested_depth_m": float(level.requested_depth_m),
            "actual_depth_m": float(level.actual_depth_m),
            "channel_index": int(level.channel_index),
        }
        for level in analysis_depth_levels
    ]

    def _absolute_error_arrays() -> Iterable[np.ndarray]:
        for level in analysis_depth_levels:
            # Errors are accumulated as signed prediction-minus-GLORYS values so
            # overlapping patches are stitched before the absolute value is taken.
            yield _absolute_error_array_from_signed_accumulator(
                signed_error_accumulators[level.suffix]
            )

    payload = build_error_analysis_payload_from_depth_arrays(
        run_summary=run_summary,
        variable_metadata={
            "name": variable_spec.name,
            "label": variable_spec.label,
            "value_units": variable_spec.value_units,
            "value_unit_label": variable_spec.value_unit_label,
        },
        depth_levels_metadata=depth_metadata,
        absolute_error_arrays=_absolute_error_arrays(),
        transform=layout.transform,
        land_mask=land_mask,
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    return output_path


def _write_compact_basin_depth_error_json(
    *,
    output_path: Path,
    run_summary: dict[str, Any],
    variable_spec: ExportVariableSpec,
    analysis_depth_levels: Sequence[DepthExportLevel],
    signed_error_accumulators: dict[str, RasterAccumulator],
    layout: MosaicLayout,
    land_mask: np.ndarray | None,
) -> Path:
    """Write compact basin-by-depth absolute-error sums for temporal exports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_metadata = [
        {
            "suffix": level.suffix,
            "label": level.label,
            "requested_depth_m": float(level.requested_depth_m),
            "actual_depth_m": float(level.actual_depth_m),
            "channel_index": int(level.channel_index),
        }
        for level in analysis_depth_levels
    ]

    def _absolute_error_arrays() -> Iterable[np.ndarray]:
        for level in analysis_depth_levels:
            # Keep the temporal summary exact by taking the absolute value only
            # after overlap-weighted signed errors have been stitched.
            yield _absolute_error_array_from_signed_accumulator(
                signed_error_accumulators[level.suffix]
            )

    payload = build_basin_depth_error_summary_payload_from_depth_arrays(
        run_summary=run_summary,
        variable_metadata={
            "name": variable_spec.name,
            "label": variable_spec.label,
            "value_units": variable_spec.value_units,
            "value_unit_label": variable_spec.value_unit_label,
        },
        depth_levels_metadata=depth_metadata,
        absolute_error_arrays=_absolute_error_arrays(),
        transform=layout.transform,
        land_mask=land_mask,
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    return output_path


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
    variable_spec: ExportVariableSpec = EXPORT_VARIABLE_SPECS["temperature"],
) -> dict[str, Any]:
    argo_profile = _profile_to_json_list(
        sample.x_profile_c,
        valid_mask=sample.observed_profile,
    )
    prediction_profile = _profile_to_json_list(
        sample.y_hat_profile_c,
        valid_mask=sample.target_valid_profile,
    )
    glorys_profile = _profile_to_json_list(
        sample.y_target_profile_c,
        valid_mask=sample.target_valid_profile,
    )
    properties = {
        "location_id": str(location_id),
        "graph_png_path": str(graph_png_path),
        "variable": variable_spec.name,
        "value_units": variable_spec.value_units,
        "depth_m": _profile_to_json_list(depth_axis_m),
        "ostia_sst_c": (
            None
            if not np.isfinite(float(sample.ostia_sst_c))
            else float(sample.ostia_sst_c)
        ),
        "argo_profile": argo_profile,
        "prediction_profile": prediction_profile,
        "glorys_profile": glorys_profile,
        # Retain legacy property names for downstream readers of existing exports.
        "argo_profile_c": argo_profile,
        "prediction_profile_c": prediction_profile,
        "glorys_profile_c": glorys_profile,
    }
    return _point_feature_for_pixel(
        row=sample.row,
        patch_shape=(int(sample.patch_height), int(sample.patch_width)),
        point_row=sample.point_row,
        point_col=sample.point_col,
        extra_properties=properties,
    )


def _write_full_profile_sample_artifacts(
    *,
    run_dir: Path,
    dataset: Any,
    writer: GeoJSONPointWriter,
    sample: FullProfileSample,
    location_id: str,
    variable_spec: ExportVariableSpec = EXPORT_VARIABLE_SPECS["temperature"],
) -> None:
    graph_rel_path = Path("graphs") / f"{location_id}.webp"
    depth_axis_m = _load_glorys_depth_axis_m(
        dataset,
        sample.row,
        expected_size=int(sample.y_target_profile_c.shape[0]),
    )
    x_profile_plot = np.asarray(sample.x_profile_c, dtype=np.float64).copy()
    y_hat_profile_plot = np.asarray(sample.y_hat_profile_c, dtype=np.float64).copy()
    y_target_profile_plot = np.asarray(
        sample.y_target_profile_c, dtype=np.float64
    ).copy()
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
        ostia_sst_c=(
            sample.ostia_sst_c if variable_spec.include_surface_context_marker else None
        ),
        profile_x_label=variable_spec.profile_x_label,
        error_x_label=variable_spec.error_x_label,
        figure_title=_profile_graph_figure_title(
            sample_date=sample.row["date"],
            lat=sample.lat,
            lon=sample.lon,
        ),
        webp_quality=DEFAULT_PROFILE_GRAPH_WEBP_QUALITY,
    )
    writer.write_feature(
        _full_profile_feature_for_sample(
            sample=sample,
            location_id=location_id,
            graph_png_path=str(graph_rel_path).replace("\\", "/"),
            depth_axis_m=depth_axis_m,
            variable_spec=variable_spec,
        )
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


def _patch_weight_window(patch_shape: tuple[int, int]) -> np.ndarray:
    """Build a deterministic overlap window with lower patch-edge weights."""
    patch_height, patch_width = (int(patch_shape[0]), int(patch_shape[1]))
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch_shape dimensions must be >= 1.")

    def _axis_weights(size: int) -> np.ndarray:
        if size == 1:
            return np.ones((1,), dtype=np.float32)
        indices = np.arange(size, dtype=np.float32)
        distance_to_edge = np.minimum(indices + 1.0, float(size) - indices)
        center_distance = float((size + 1) // 2)
        return (distance_to_edge / center_distance).astype(np.float32, copy=False)

    y_weights = _axis_weights(patch_height)[:, None]
    x_weights = _axis_weights(patch_width)[None, :]
    return (y_weights * x_weights).astype(np.float32, copy=False)


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
    # pixels inside each patch window so weighted overlaps can be finalized later
    # without storing every patch prediction in memory.
    patch_weights = _patch_weight_window(patch.shape)
    sum_window = sum_array[row_slice, col_slice]
    count_window = count_array[row_slice, col_slice]
    valid_weights = patch_weights[valid_mask].astype(np.float64, copy=False)
    sum_window[valid_mask] += patch[valid_mask].astype(np.float64) * valid_weights
    count_window[valid_mask] += valid_weights


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
    mosaic_count = np.zeros((layout.height, layout.width), dtype=np.float64)
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
    closed_mask = ndi.binary_closing(
        valid_mask, structure=row_structure
    ) | ndi.binary_closing(valid_mask, structure=col_structure)
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


def _apply_valid_gaussian_blur_2d(
    image: np.ndarray,
    *,
    nodata: float,
    sigma: float,
    kernel_size: int = DEFAULT_EXPORT_GAUSSIAN_BLUR_KERNEL_SIZE,
) -> np.ndarray:
    """Apply Gaussian blur to valid raster pixels while preserving nodata support."""
    blur_sigma = float(sigma)
    if blur_sigma <= 0.0:
        return np.asarray(image, dtype=np.float32).copy()

    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")
    if kernel_size % 2 == 0:
        kernel_size += 1

    patch = np.asarray(image, dtype=np.float32)
    if patch.ndim != 2:
        raise RuntimeError(
            "Expected a 2D raster for Gaussian blur, "
            f"got shape {tuple(patch.shape)}."
        )

    valid_mask = np.isfinite(patch)
    if np.isfinite(nodata):
        valid_mask &= ~np.isclose(patch, float(nodata), atol=0.0, rtol=0.0)

    if not np.any(valid_mask):
        return patch.copy()

    radius = int(kernel_size // 2)
    truncate = float(radius) / blur_sigma
    valid_weights = valid_mask.astype(np.float32)
    valid_values = np.where(valid_mask, patch, 0.0).astype(np.float32, copy=False)
    # Blur values and support separately so nodata pixels do not leak into valid water cells.
    blurred_values = ndi.gaussian_filter(
        valid_values,
        sigma=blur_sigma,
        mode="nearest",
        truncate=truncate,
    )
    blurred_weights = ndi.gaussian_filter(
        valid_weights,
        sigma=blur_sigma,
        mode="nearest",
        truncate=truncate,
    )

    blurred = patch.copy()
    weighted_valid_mask = valid_mask & (blurred_weights > 0.0)
    blurred[weighted_valid_mask] = (
        blurred_values[weighted_valid_mask] / blurred_weights[weighted_valid_mask]
    ).astype(np.float32, copy=False)
    return blurred


def _layout_spans_periodic_longitude(layout: MosaicLayout) -> bool:
    """Return True when a mosaic covers the full -180..180 longitude span."""
    lon_span = float(layout.right) - float(layout.left)
    tolerance = max(float(layout.pixel_width) * 1.0e-6, 1.0e-6)
    return (
        float(layout.left) <= -180.0 + tolerance
        and float(layout.right) >= 180.0 - tolerance
        and abs(lon_span - 360.0) <= tolerance
    )


def _apply_periodic_longitude_edge_blend_2d(
    image: np.ndarray,
    *,
    nodata: float,
    blend_width: int,
    layout: MosaicLayout,
) -> tuple[np.ndarray, bool, int]:
    """Blend valid west/east edge pixels for periodic longitude mosaics."""
    patch = np.asarray(image, dtype=np.float32)
    if patch.ndim != 2:
        raise RuntimeError(
            "Expected a 2D raster for longitude wrap stitching, "
            f"got shape {tuple(patch.shape)}."
        )

    requested_width = int(blend_width)
    if requested_width <= 0 or not _layout_spans_periodic_longitude(layout):
        return patch.copy(), False, 0

    edge_width = min(requested_width, int(patch.shape[1]) // 2)
    if edge_width <= 0:
        return patch.copy(), False, 0

    left_edge = patch[:, :edge_width]
    # Reverse the eastern edge so column 0 pairs with the pixel nearest +180.
    right_edge = patch[:, -edge_width:][:, ::-1]
    valid_pair_mask = _valid_raster_mask(left_edge, nodata) & _valid_raster_mask(
        right_edge,
        nodata,
    )
    if not np.any(valid_pair_mask):
        return patch.copy(), False, edge_width

    offsets = np.arange(edge_width, dtype=np.float32)
    other_weight = (0.5 * (float(edge_width) - offsets) / float(edge_width))[None, :]
    left_blended = (
        (left_edge * (np.float32(1.0) - other_weight)) + (right_edge * other_weight)
    ).astype(np.float32, copy=False)
    right_blended = (
        (right_edge * (np.float32(1.0) - other_weight)) + (left_edge * other_weight)
    ).astype(np.float32, copy=False)

    out = patch.copy()
    out_left = out[:, :edge_width]
    out_right = out[:, -edge_width:][:, ::-1]
    out_left[valid_pair_mask] = left_blended[valid_pair_mask]
    out_right[valid_pair_mask] = right_blended[valid_pair_mask]
    return out, True, edge_width


def _periodic_longitude_blend_width_for_layout(
    layout: MosaicLayout,
    *,
    patch_stride: int | None,
) -> int:
    """Return the missing horizontal overlap width for a periodic mosaic."""
    if not _layout_spans_periodic_longitude(layout):
        return 0
    if patch_stride is None:
        return 0
    return max(0, int(layout.patch_width) - int(patch_stride))


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
        dtype=np.float64,
        mode="w+",
        shape=(layout.height, layout.width),
    )
    count_array[:] = 0.0
    return RasterAccumulator(
        sum_path=sum_path,
        count_path=count_path,
        sum_array=sum_array,
        count_array=count_array,
    )


def load_land_mask_for_layout(
    *, land_mask_path: Path, layout: MosaicLayout
) -> np.ndarray:
    """Load and resample a land-mask GeoTIFF to a mosaic layout."""
    if not land_mask_path.exists():
        raise FileNotFoundError(f"Land-mask GeoTIFF does not exist: {land_mask_path}")
    land_mask = np.zeros((int(layout.height), int(layout.width)), dtype=np.uint8)
    with rasterio.open(land_mask_path) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=land_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=layout.transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
    return land_mask > 0


def _flush_accumulator(accumulator: RasterAccumulator) -> None:
    accumulator.sum_array.flush()
    accumulator.count_array.flush()


def _cleanup_accumulator(accumulator: RasterAccumulator) -> None:
    del accumulator.sum_array
    del accumulator.count_array
    accumulator.sum_path.unlink(missing_ok=True)
    accumulator.count_path.unlink(missing_ok=True)


def _prediction_zero_artifact_mask(patch: np.ndarray) -> np.ndarray:
    """Return pixels that are indistinguishable from prediction land-mask zeros."""
    patch_array = np.asarray(patch, dtype=np.float32)
    return np.isfinite(patch_array) & (
        np.abs(patch_array) <= PREDICTION_ZERO_ARTIFACT_EPSILON
    )


def _prediction_zeros_to_nan(patch: np.ndarray) -> np.ndarray:
    """Return a prediction patch with zero-artifact values masked as NaN."""
    patch_array = np.asarray(patch, dtype=np.float32).copy()
    # Prediction zeros come from model land-mask post-processing, not plausible ocean values.
    patch_array[_prediction_zero_artifact_mask(patch_array)] = np.nan
    return patch_array


def _load_ground_truth_patch_for_variable(
    dataset: Any, row: dict[str, Any], variable_spec: ExportVariableSpec
) -> np.ndarray | None:
    """Load one decoded GLORYS patch for the requested export variable."""
    loader = getattr(dataset, variable_spec.ground_truth_loader_name, None)
    if loader is None:
        return None

    # NetCDF-backed temperature datasets need explicit patch axes; GeoTIFF-backed
    # datasets decode stretched uint8 rasters internally and only need row metadata.
    if variable_spec.name == "temperature" and hasattr(dataset, "_patch_axes"):
        patch = loader(row, dataset._patch_axes(row))
    else:
        patch = loader(row)
    patch_array = np.asarray(patch, dtype=np.float32)
    if patch_array.ndim != 3:
        raise RuntimeError(
            f"Expected decoded GLORYS {variable_spec.name} patch shape (D,H,W), "
            f"got {tuple(patch_array.shape)}."
        )
    return patch_array


def _load_ground_truth_patch_celsius(
    dataset: Any, row: dict[str, Any]
) -> np.ndarray | None:
    """Load one decoded GLORYS temperature patch in degrees Celsius."""
    return _load_ground_truth_patch_for_variable(
        dataset, row, EXPORT_VARIABLE_SPECS["temperature"]
    )


def write_global_top_band_geotiff(
    *,
    output_path: Path,
    accumulator: RasterAccumulator,
    layout: MosaicLayout,
    nodata: float,
    band_description: str,
    tags: dict[str, str],
    extra_gaussian_blur_sigma: float = 0.0,
    land_mask: np.ndarray | None = None,
    prediction_zero_masked_to_nodata: bool = True,
    periodic_longitude_blend_width: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    land_mask_bool: np.ndarray | None = None
    if land_mask is not None:
        land_mask_bool = np.asarray(land_mask, dtype=bool)
        if land_mask_bool.shape != (int(layout.height), int(layout.width)):
            raise RuntimeError(
                "Land mask shape does not match mosaic layout: "
                f"{tuple(land_mask_bool.shape)} != {(int(layout.height), int(layout.width))}."
            )
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
    output_tags = dict(tags)
    output_tags["prediction_zero_masked_to_nodata"] = str(
        bool(prediction_zero_masked_to_nodata)
    ).lower()
    output_tags["longitude_wrap_stitching"] = "false"
    output_tags["longitude_wrap_blend_width_pixels"] = "0"
    if prediction_zero_masked_to_nodata:
        output_tags["prediction_zero_mask_epsilon_c"] = (
            f"{PREDICTION_ZERO_ARTIFACT_EPSILON:.1e}"
        )

    with rasterio.open(output_path, "w", **profile) as ds:
        ds.set_band_description(1, band_description)
        ds.update_tags(**output_tags)
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
                dtype=np.float64,
            )
            out_block = np.full(sum_block.shape, float(nodata), dtype=np.float32)
            valid_mask = count_block > 0.0
            out_block[valid_mask] = (
                sum_block[valid_mask] / count_block[valid_mask]
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
        final_band = (
            _apply_valid_gaussian_blur_2d(
                repaired_band,
                nodata=nodata,
                sigma=float(extra_gaussian_blur_sigma),
                kernel_size=DEFAULT_EXPORT_GAUSSIAN_BLUR_KERNEL_SIZE,
            )
            if float(extra_gaussian_blur_sigma) > 0.0
            else repaired_band
        )
        if land_mask_bool is not None:
            # Land is masked after all averaging/repair/blur steps so it does
            # not affect neighboring ocean pixels during post-processing.
            final_band[land_mask_bool] = float(nodata)
        zero_mask = np.zeros(final_band.shape, dtype=bool)
        if prediction_zero_masked_to_nodata:
            # Zero-valued prediction artifacts should not survive into GeoTIFFs
            # or Cesium color relief.
            zero_mask = _prediction_zero_artifact_mask(final_band)
            final_band[zero_mask] = float(nodata)
        final_band, wrap_blended, effective_blend_width = (
            _apply_periodic_longitude_edge_blend_2d(
                final_band,
                nodata=nodata,
                blend_width=int(periodic_longitude_blend_width),
                layout=layout,
            )
        )
        if effective_blend_width > 0:
            ds.update_tags(
                longitude_wrap_stitching="true",
                longitude_wrap_blend_width_pixels=str(int(effective_blend_width)),
            )
        if (
            np.any(repaired_mask)
            or land_mask_bool is not None
            or np.any(zero_mask)
            or wrap_blended
        ):
            ds.write(final_band, 1)
        elif float(extra_gaussian_blur_sigma) > 0.0:
            ds.write(final_band, 1)

    overview_factors = _overview_factors(layout.width, layout.height)
    if overview_factors:
        with rasterio.open(output_path, "r+") as ds:
            ds.build_overviews(overview_factors, Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")


def _valid_raster_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return finite raster pixels that are not equal to the dataset nodata value."""
    mask = np.isfinite(data)
    if nodata is not None and np.isfinite(float(nodata)):
        mask &= ~np.isclose(data, float(nodata), atol=0.0, rtol=0.0)
    return mask


def write_absolute_error_geotiff(
    *,
    prediction_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    nodata: float,
    band_description: str,
    tags: dict[str, str],
) -> None:
    """Write a GeoTIFF of per-pixel absolute prediction-vs-GLORYS error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(prediction_path) as prediction_ds,
        rasterio.open(ground_truth_path) as ground_truth_ds,
    ):
        if (
            prediction_ds.width != ground_truth_ds.width
            or prediction_ds.height != ground_truth_ds.height
            or prediction_ds.transform != ground_truth_ds.transform
            or prediction_ds.crs != ground_truth_ds.crs
        ):
            raise RuntimeError(
                "Prediction and GLORYS rasters must share shape, transform, and CRS "
                f"before absolute-error export: {prediction_path}, {ground_truth_path}"
            )

        profile = prediction_ds.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=float(nodata),
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=_geotiff_block_size(int(prediction_ds.width)),
            blockysize=_geotiff_block_size(int(prediction_ds.height)),
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as output_ds:
            output_ds.set_band_description(1, band_description)
            output_ds.update_tags(**tags)
            for _block_index, window in prediction_ds.block_windows(1):
                prediction_block = prediction_ds.read(1, window=window, masked=False)
                ground_truth_block = ground_truth_ds.read(
                    1, window=window, masked=False
                )
                valid_mask = _valid_raster_mask(
                    prediction_block,
                    (
                        None
                        if prediction_ds.nodata is None
                        else float(prediction_ds.nodata)
                    ),
                ) & _valid_raster_mask(
                    ground_truth_block,
                    (
                        None
                        if ground_truth_ds.nodata is None
                        else float(ground_truth_ds.nodata)
                    ),
                )
                out_block = np.full(
                    prediction_block.shape,
                    float(nodata),
                    dtype=np.float32,
                )
                # Source rasters are already denormalized physical values, so
                # the absolute difference remains in the active variable's unit.
                out_block[valid_mask] = np.abs(
                    prediction_block[valid_mask].astype(np.float32, copy=False)
                    - ground_truth_block[valid_mask].astype(np.float32, copy=False)
                )
                output_ds.write(out_block, 1, window=window)

    overview_factors = _overview_factors(profile["width"], profile["height"])
    if overview_factors:
        with rasterio.open(output_path, "r+") as ds:
            ds.build_overviews(overview_factors, Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DepthDif inference for one globally complete daily snapshot selected "
            "from the GeoTIFF patch dataset, then export configured depth rasters."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_INFERENCE_CONFIG,
        dest="config_path",
        help="Path to the pixel inference super-config yaml.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(PIXEL_SCENARIOS),
        default=None,
        help="High-level pixel inference scenario; derives data/model channel settings.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
        help=(
            "Override config values. Format: "
            "<data|training|model|inference>.<nested.path>=<yaml_value>. "
            "Repeat --set for multiple overrides."
        ),
    )
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--year", type=int, required=True, help="ISO year filter.")
    parser.add_argument("--iso-week", type=int, required=True, help="ISO week filter.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "val"),
        help="Dataset split filter. Global raster export requires 'all'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to dataloader.batch_size then 4.",
    )
    parser.add_argument(
        "--inference-num-workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes used by the optimized inference DataLoader. "
            "Defaults to inference.dataloader.num_workers."
        ),
    )
    parser.add_argument(
        "--inference-prefetch-factor",
        type=int,
        default=None,
        help=(
            "Number of batches prefetched by each inference DataLoader worker. "
            "Defaults to inference.dataloader.prefetch_factor and is only used "
            "when worker processes are enabled."
        ),
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
        help="Also export GLORYS ground-truth rasters for the selected depth levels.",
    )
    parser.add_argument(
        "--persist-ground-truth-rasters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist GLORYS depth GeoTIFFs. Disable this for lightweight temporal "
            "runs that still need GLORYS internally for absolute-error outputs."
        ),
    )
    parser.add_argument(
        "--depth-export-suffix",
        action="append",
        default=[],
        help=(
            "Restrict persisted prediction/absolute-error depth rasters to a suffix "
            "such as 10m. Repeat for multiple suffixes; omit to export the default depths."
        ),
    )
    parser.add_argument(
        "--compact-basin-depth-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write compact basin-by-depth absolute-error JSON instead of the full "
            "grid-cell analysis payload."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Torch RNG seed so stochastic samplers remain reproducible.",
    )
    parser.add_argument(
        "--sampler",
        "--sampling-method",
        choices=INFERENCE_SAMPLERS,
        default=None,
        help=(
            "Inference sampler override. Defaults to inference.sampling.sampler "
            "then training.validation_sampling.sampler."
        ),
    )
    parser.add_argument(
        "--ddim-steps",
        "--ddim-num-timesteps",
        dest="ddim_num_timesteps",
        type=int,
        default=None,
        help=(
            "DDIM inference step count. Passing this without --sampler also "
            "selects DDIM."
        ),
    )
    parser.add_argument(
        "--uncertainty-sampler",
        choices=INFERENCE_SAMPLERS,
        default=None,
        help=(
            "Sampler override used only by --export-uncertainty. Defaults to "
            "inference.uncertainty_sampling.sampler, then the reconstruction sampler."
        ),
    )
    parser.add_argument(
        "--uncertainty-ddim-steps",
        "--uncertainty-ddim-num-timesteps",
        dest="uncertainty_ddim_num_timesteps",
        type=int,
        default=None,
        help="DDIM step count used only by --export-uncertainty.",
    )
    parser.add_argument(
        "--export-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the model uncertainty step and export one stitched 1-channel "
            "uncertainty raster for the active variable."
        ),
    )
    parser.add_argument(
        "--uncertainty-num-samples",
        type=int,
        default=DEFAULT_UNCERTAINTY_NUM_SAMPLES,
        help="Number of stochastic generations used by --export-uncertainty.",
    )
    parser.add_argument(
        "--uncertainty-collapse-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collapse depth/channel uncertainty to one surface-independent raster. "
            "By default uncertainty is exported for each selected depth level."
        ),
    )
    parser.add_argument(
        "--uncertainty-only",
        action="store_true",
        help=(
            "Only run the uncertainty step and export its stitched 1-channel "
            "raster; implies --export-uncertainty and skips prediction, "
            "GLORYS, absolute-error, and full-profile exports."
        ),
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
        "--sigma",
        type=float,
        default=DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
        help=(
            "Extra export-time Gaussian blur sigma for prediction GeoTIFFs. "
            "Defaults to 0.0; pass a positive value to enable the kernel size 3 blur."
        ),
    )
    parser.add_argument(
        "--land-mask-path",
        type=Path,
        default=None,
        help=(
            "GLORYS-aligned GeoTIFF with 1=land and 0=water for final zeroing. "
            "Defaults to inference.grid.land_mask_path."
        ),
    )
    parser.add_argument(
        "--min-ocean-fraction",
        type=float,
        default=None,
        help=(
            "Minimum ocean fraction required for an inference patch. "
            "Defaults to inference.grid.min_ocean_fraction."
        ),
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=None,
        help=(
            "Patch-grid stride in pixels for inference. Defaults to "
            "inference.grid.patch_stride."
        ),
    )
    parser.add_argument(
        "--rectangle",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=None,
        help=(
            "Optional lon/lat rectangle. Runs only selected ISO-week patches "
            "that intersect these bounds."
        ),
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="Optional public base URL for generated Cesium globe assets.",
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help="Optional rclone destination for uploading generated globe assets.",
    )
    parser.add_argument(
        "--rclone-sync-scope",
        type=str,
        choices=("globe", "run"),
        default=DEFAULT_RCLONE_SYNC_SCOPE,
        help="Choose whether rclone sync uploads generated globe assets or the full run.",
    )
    parser.add_argument(
        "--extra-zoom-levels",
        type=int,
        default=DEFAULT_EXTRA_ZOOM_LEVELS,
        help=(
            "Cesium tile zoom-level adjustment relative to the raster native estimate. "
            "Negative values reduce max zoom."
        ),
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
            "Number of observed Argo locations whose full depth profiles are saved "
            "with graph PNGs for the globe viewer. The default is 1000; use "
            "0 to disable and any negative value to export all observed locations."
        ),
    )
    return parser


def _normalize_cli_args(argv: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for arg in argv:
        if arg.startswith("sigma:"):
            normalized.extend(["--sigma", arg.split(":", 1)[1]])
            continue
        normalized.append(arg)
    return normalized


def run_global_inference(args: argparse.Namespace) -> ExportRunResult:
    """Run global inference from parsed/exporter-style arguments."""
    torch.manual_seed(int(args.seed))
    if args.split != "all":
        raise ValueError(
            "Global raster export requires --split all so the selected timestep "
            "contains every spatial patch, not just train or val."
        )

    config_bundle = load_pixel_inference_config(
        config_path_value=args.config_path,
        scenario_override=args.scenario,
        overrides=list(args.config_overrides or []),
        runtime_config_dir=Path("/tmp/depthdif_inference_configs")
        / f"global_{os.getpid()}",
        write_snapshots=False,
    )
    training_cfg = config_bundle.training_cfg
    data_cfg = config_bundle.data_cfg
    inference_cfg = config_bundle.inference_cfg
    args.model_config = config_bundle.effective_model_config_path
    args.data_config = config_bundle.effective_data_config_path
    args.train_config = config_bundle.effective_training_config_path
    args.inference_config = config_bundle.effective_inference_config_path
    inference_section = _inference_section(inference_cfg)
    inference_grid_cfg = inference_section.get("grid", {})
    inference_dataloader_cfg = inference_section.get("dataloader", {})
    inference_sampling_cfg = inference_section.get("sampling", {})
    inference_uncertainty_sampling_cfg = inference_section.get(
        "uncertainty_sampling", {}
    )
    if not isinstance(inference_grid_cfg, dict):
        inference_grid_cfg = {}
    if not isinstance(inference_dataloader_cfg, dict):
        inference_dataloader_cfg = {}
    if not isinstance(inference_sampling_cfg, dict):
        inference_sampling_cfg = {}
    if not isinstance(inference_uncertainty_sampling_cfg, dict):
        inference_uncertainty_sampling_cfg = {}
    sampler_override = getattr(args, "sampler", None)
    if sampler_override is None:
        sampler_override = inference_sampling_cfg.get("sampler")
    ddim_steps_override = getattr(args, "ddim_num_timesteps", None)
    if ddim_steps_override is None:
        ddim_steps_override = inference_sampling_cfg.get("ddim_num_timesteps")
    sampling_metadata = apply_inference_sampling_config(
        training_cfg,
        sampler=sampler_override,
        ddim_num_timesteps=ddim_steps_override,
    )
    uncertainty_sampler_override = getattr(args, "uncertainty_sampler", None)
    if uncertainty_sampler_override is None:
        uncertainty_sampler_override = inference_uncertainty_sampling_cfg.get("sampler")
    uncertainty_ddim_steps_override = getattr(
        args, "uncertainty_ddim_num_timesteps", None
    )
    if uncertainty_ddim_steps_override is None:
        uncertainty_ddim_steps_override = inference_uncertainty_sampling_cfg.get(
            "ddim_num_timesteps"
        )
    uncertainty_sampling_metadata = _sampling_metadata_from_overrides(
        training_cfg,
        sampler=uncertainty_sampler_override,
        ddim_num_timesteps=uncertainty_ddim_steps_override,
        fallback=sampling_metadata,
    )
    # The model is constructed from the materialized split config, so persist any
    # inference-only sampler overrides before build_model reads the file.
    with Path(args.train_config).open("w", encoding="utf-8") as f:
        yaml.safe_dump(training_cfg, f, sort_keys=False)
    raw_land_mask_path = (
        args.land_mask_path
        if args.land_mask_path is not None
        else inference_grid_cfg.get(
            "land_mask_path",
            DEFAULT_LAND_MASK_RELATIVE_PATH,
        )
    )
    effective_land_mask_path = _resolve_dataset_root_relative_path(
        data_cfg,
        raw_land_mask_path,
    )
    effective_min_ocean_fraction = float(
        args.min_ocean_fraction
        if args.min_ocean_fraction is not None
        else inference_grid_cfg.get("min_ocean_fraction", 0.05)
    )
    effective_patch_stride = (
        int(args.patch_stride)
        if args.patch_stride is not None
        else (
            None
            if inference_grid_cfg.get("patch_stride") is None
            else int(inference_grid_cfg["patch_stride"])
        )
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else inference_dataloader_cfg.get(
            "batch_size",
            training_cfg.get("dataloader", {}).get("batch_size", 4),
        )
    )
    inference_num_workers = int(
        args.inference_num_workers
        if args.inference_num_workers is not None
        else inference_dataloader_cfg.get(
            "num_workers",
            DEFAULT_INFERENCE_NUM_WORKERS,
        )
    )
    inference_prefetch_factor = int(
        args.inference_prefetch_factor
        if args.inference_prefetch_factor is not None
        else inference_dataloader_cfg.get(
            "prefetch_factor",
            DEFAULT_INFERENCE_PREFETCH_FACTOR,
        )
    )
    requested_full_sample_count = int(args.full_sample_count)
    uncertainty_only = bool(getattr(args, "uncertainty_only", False))
    export_uncertainty = (
        bool(getattr(args, "export_uncertainty", False)) or uncertainty_only
    )
    export_prediction = not uncertainty_only
    export_ground_truth = bool(args.export_ground_truth) and export_prediction
    uncertainty_num_samples = int(
        getattr(args, "uncertainty_num_samples", DEFAULT_UNCERTAINTY_NUM_SAMPLES)
    )
    uncertainty_collapse_depth = bool(
        getattr(args, "uncertainty_collapse_depth", False)
    )
    if export_uncertainty and uncertainty_num_samples < 2:
        raise ValueError("--uncertainty-num-samples must be at least 2.")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if inference_num_workers < 0:
        raise ValueError("--inference-num-workers must be >= 0.")
    if inference_prefetch_factor < 1:
        raise ValueError("--inference-prefetch-factor must be >= 1.")
    _print_global_inference_settings(
        args=args,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        effective_land_mask_path=effective_land_mask_path,
        effective_min_ocean_fraction=effective_min_ocean_fraction,
        effective_patch_stride=effective_patch_stride,
        batch_size=batch_size,
        inference_num_workers=inference_num_workers,
        inference_prefetch_factor=inference_prefetch_factor,
        requested_full_sample_count=requested_full_sample_count,
        uncertainty_only=uncertainty_only,
        export_uncertainty=export_uncertainty,
        uncertainty_num_samples=uncertainty_num_samples,
    )

    configured_val_year = data_cfg.get("split", {}).get("val_year")
    dataset, inference_grid_metadata = resolve_global_inference_dataset(
        None,
        data_config_path=args.data_config,
        data_cfg=data_cfg,
        split=args.split,
        land_mask_path=effective_land_mask_path,
        min_ocean_fraction=effective_min_ocean_fraction,
        patch_stride=effective_patch_stride,
    )
    if hasattr(dataset, "return_info"):
        dataset.return_info = False
    if hasattr(dataset, "return_coords"):
        dataset.return_coords = True
    rows = _dataset_rows(dataset)
    selection = select_export_indices(
        rows,
        exact_date=args.date,
        iso_year=args.year,
        iso_week=args.iso_week,
    )
    selection = filter_selection_by_rectangle(rows, selection, args.rectangle)

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

    selected_rows = [rows[idx] for idx in selection.indices]
    selected_patches = pd.DataFrame.from_records(selected_rows)
    selected_patches.to_csv(run_dir / "selected_patches.csv", index=False)

    model_cfg = load_yaml(args.model_config)
    variable_spec = resolve_export_variable_spec(model_cfg)
    export_all_full_samples = requested_full_sample_count < 0 and export_prediction
    full_sample_count = int(max(0, requested_full_sample_count))
    export_full_profiles = bool(
        export_prediction and (export_all_full_samples or full_sample_count > 0)
    )

    sample_for_shape = dataset[selection.indices[0]]
    sample_target = sample_for_shape[variable_spec.y_key]
    patch_shape = tuple(int(v) for v in sample_target.shape[-2:])
    depth_axis_m = _load_glorys_depth_axis_m(
        dataset,
        selected_rows[0],
        expected_size=int(sample_target.shape[0]),
    )
    depth_export_levels = _filter_depth_export_levels(
        resolve_depth_export_levels(depth_axis_m),
        getattr(args, "depth_export_suffix", None),
    )
    analysis_depth_levels = list(depth_export_levels)
    collect_full_depth_error_analysis = bool(export_prediction and export_ground_truth)
    export_compact_basin_depth_error = bool(
        collect_full_depth_error_analysis and args.compact_basin_depth_error
    )
    export_full_depth_error_analysis = bool(
        collect_full_depth_error_analysis and not export_compact_basin_depth_error
    )
    depth_channel_indices = tuple(
        int(level.channel_index) for level in depth_export_levels
    )
    layout = build_mosaic_layout(selected_rows, patch_shape=patch_shape)
    longitude_wrap_blend_width = _periodic_longitude_blend_width_for_layout(
        layout,
        patch_stride=int(inference_grid_metadata["patch_stride"]),
    )
    inference_grid_metadata = dict(inference_grid_metadata)
    inference_grid_metadata["longitude_wrap_stitching"] = bool(
        longitude_wrap_blend_width > 0
    )
    inference_grid_metadata["longitude_wrap_blend_width_pixels"] = int(
        longitude_wrap_blend_width
    )
    land_mask = load_land_mask_for_layout(
        land_mask_path=effective_land_mask_path,
        layout=layout,
    )
    scratch_dir = run_dir / ".scratch"
    pred_accumulators = (
        {
            level.suffix: create_raster_accumulator(
                root_dir=scratch_dir,
                stem=f"prediction_{level.suffix}",
                layout=layout,
            )
            for level in depth_export_levels
        }
        if export_prediction
        else {}
    )
    gt_accumulators = (
        {
            level.suffix: create_raster_accumulator(
                root_dir=scratch_dir,
                stem=f"ground_truth_{level.suffix}",
                layout=layout,
            )
            for level in depth_export_levels
        }
        if export_ground_truth
        else {}
    )
    signed_error_accumulators = (
        {
            level.suffix: create_raster_accumulator(
                root_dir=scratch_dir,
                stem=f"signed_error_{level.suffix}",
                layout=layout,
            )
            for level in analysis_depth_levels
        }
        if collect_full_depth_error_analysis
        else {}
    )
    uncertainty_accumulators = (
        {
            (
                "collapsed" if uncertainty_collapse_depth else level.suffix
            ): create_raster_accumulator(
                root_dir=scratch_dir,
                stem=(
                    "uncertainty"
                    if uncertainty_collapse_depth
                    else f"uncertainty_{level.suffix}"
                ),
                layout=layout,
            )
            for level in (
                depth_export_levels[:1]
                if uncertainty_collapse_depth
                else depth_export_levels
            )
        }
        if export_uncertainty
        else {}
    )
    argo_points_geojson_path = (
        run_dir / f"{run_stem}_argo_points.geojson" if export_ground_truth else None
    )
    full_sample_locations_geojson_path = (
        run_dir / f"{run_stem}_full_sample_locations.geojson"
        if export_full_profiles
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

    graphs_dir_path: Path | None = run_dir / "graphs" if export_full_profiles else None
    full_sample_locations_writer: GeoJSONPointWriter | None = None
    if full_sample_locations_geojson_path is not None and export_all_full_samples:
        full_sample_locations_writer = GeoJSONPointWriter(
            full_sample_locations_geojson_path
        )
        full_sample_locations_writer.open()

    print(
        "Preparing global export: "
        f"split={args.split}, "
        f"selected_date={selection.selected_date}, "
        f"iso_week={selection.iso_year}-W{selection.iso_week:02d}, "
        f"selected_patches={len(selection.indices)}, "
        f"batch_size={batch_size}, "
        f"inference_num_workers={inference_num_workers}, "
        f"inference_prefetch_factor={inference_prefetch_factor}, "
        f"export_ground_truth={export_ground_truth}, "
        f"full_depth_error_analysis={export_full_depth_error_analysis}, "
        f"compact_basin_depth_error={export_compact_basin_depth_error}, "
        f"uncertainty_only={uncertainty_only}, "
        f"full_sample_count="
        f"{'all' if export_all_full_samples else full_sample_count}, "
        f"prediction_runs_per_patch={1 if export_prediction else 0}, "
        f"extra_gaussian_blur_sigma={float(args.sigma)}, "
        f"export_uncertainty={export_uncertainty}, "
        f"uncertainty_num_samples={uncertainty_num_samples}, "
        f"uncertainty_collapse_depth={uncertainty_collapse_depth}, "
        f"uncertainty_sampler={uncertainty_sampling_metadata['sampler']}, "
        f"uncertainty_ddim_num_timesteps="
        f"{int(uncertainty_sampling_metadata['ddim_num_timesteps'])}, "
        f"patch_stride={inference_grid_metadata['patch_stride']}, "
        f"patch_overlap_fraction={inference_grid_metadata['patch_overlap_fraction']:.2f}, "
        f"longitude_wrap_blend_width={longitude_wrap_blend_width}, "
        f"min_ocean_fraction={inference_grid_metadata['min_ocean_fraction']:.2f}, "
        f"land_mask_path={effective_land_mask_path}, "
        f"rectangle={args.rectangle}, "
        f"variable={variable_spec.name}, "
        f"depth_exports={','.join(level.label for level in depth_export_levels)}"
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
            "model.resume_checkpoint."
        )
    weight_source = load_checkpoint_weights(
        model,
        ckpt_path,
        strict=bool(args.strict_load),
    )
    model = model.to(device)
    model.eval()
    uncertainty_sampler = None
    if export_uncertainty:
        uncertainty_sampler = _build_sampler_for_metadata(
            model, uncertainty_sampling_metadata
        )
        if uncertainty_sampler is not None:
            uncertainty_sampler = uncertainty_sampler.to(device)
    print(f"Loaded checkpoint: {ckpt_path} ({weight_source} weights)")

    inference_module = ExportInferenceWrapper(
        model,
        variable_spec=variable_spec,
        export_ground_truth=export_ground_truth,
        export_full_prediction_stack=(
            export_full_profiles or collect_full_depth_error_analysis
        ),
        export_prediction=export_prediction,
        export_uncertainty=export_uncertainty,
        uncertainty_num_samples=uncertainty_num_samples,
        uncertainty_sampler=uncertainty_sampler,
        collapse_uncertainty_channels=uncertainty_collapse_depth,
        depth_channel_indices=depth_channel_indices,
    )
    if use_multi_gpu:
        inference_runner: nn.Module = nn.DataParallel(inference_module)
    else:
        inference_runner = inference_module

    inference_loader = _build_inference_loader(
        dataset=dataset,
        indices=selection.indices,
        rows=selected_rows,
        batch_size=batch_size,
        num_workers=inference_num_workers,
        prefetch_factor=inference_prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    progress = tqdm(
        inference_loader,
        total=len(inference_loader),
        desc="Running inference and streaming patches",
        unit="batch",
    )
    for inference_items in progress:
        batch_indices = inference_items["dataset_indices"]
        batch = inference_items["batch"]
        model_batch = batch if use_multi_gpu else _to_device(batch, device)
        with torch.no_grad():
            outputs = inference_runner(model_batch)

        prediction_depth_batch = (
            outputs["prediction_depth_stack"].detach().float().cpu().numpy()
            if export_prediction
            else None
        )
        prediction_full_stack_batch = (
            outputs["prediction_full_stack"].detach().float().cpu().numpy()
            if export_full_profiles or collect_full_depth_error_analysis
            else None
        )
        ground_truth_batch = (
            outputs["ground_truth_depth_stack"].detach().float().cpu().numpy()
            if export_ground_truth
            else None
        )
        uncertainty_batch = (
            outputs["uncertainty_map"].detach().float().cpu().numpy()
            if uncertainty_accumulators
            else None
        )
        eo_denorm_batch = (
            temperature_normalize(mode="denorm", tensor=batch["eo"])
            .detach()
            .float()
            .cpu()
            .numpy()
            if export_full_profiles
            else None
        )
        x_denorm_batch = (
            variable_spec.normalize_fn(mode="denorm", tensor=batch[variable_spec.x_key])
            .detach()
            .float()
            .cpu()
            .numpy()
            if export_full_profiles
            else None
        )
        y_denorm_batch = (
            variable_spec.normalize_fn(mode="denorm", tensor=batch[variable_spec.y_key])
            .detach()
            .float()
            .cpu()
            .numpy()
            if export_full_profiles or collect_full_depth_error_analysis
            else None
        )
        target_valid_full_mask_batch = (
            batch[variable_spec.y_valid_mask_key].detach().cpu().numpy().astype(bool)
            if collect_full_depth_error_analysis
            else None
        )
        selected_row_batch = inference_items["rows"]
        for local_idx, (dataset_idx, row) in enumerate(
            zip(batch_indices, selected_row_batch, strict=False)
        ):
            patch_split_feature = _patch_split_feature_for_row(row)
            if patch_split_feature is not None:
                patch_splits_writer.write_feature(patch_split_feature)
            ground_truth_patch = (
                _load_ground_truth_patch_for_variable(dataset, row, variable_spec)
                if ground_truth_batch is not None
                else None
            )
            if prediction_depth_batch is not None:
                for depth_idx, level in enumerate(depth_export_levels):
                    pred_accumulator = pred_accumulators[level.suffix]
                    _accumulate_patch_into_arrays(
                        pred_accumulator.sum_array,
                        pred_accumulator.count_array,
                        row=row,
                        patch_values=_prediction_zeros_to_nan(
                            prediction_depth_batch[local_idx, depth_idx]
                        ),
                        layout=layout,
                    )
                    if (
                        ground_truth_batch is not None
                        and level.suffix in gt_accumulators
                    ):
                        gt_accumulator = gt_accumulators[level.suffix]
                        gt_patch_values = (
                            ground_truth_patch[int(level.channel_index)]
                            if ground_truth_patch is not None
                            else ground_truth_batch[local_idx, depth_idx]
                        )
                        _accumulate_patch_into_arrays(
                            gt_accumulator.sum_array,
                            gt_accumulator.count_array,
                            row=row,
                            patch_values=gt_patch_values,
                            layout=layout,
                        )
            if (
                collect_full_depth_error_analysis
                and prediction_full_stack_batch is not None
                and y_denorm_batch is not None
                and target_valid_full_mask_batch is not None
            ):
                for level in analysis_depth_levels:
                    channel_index = int(level.channel_index)
                    pred_patch_values = _prediction_zeros_to_nan(
                        prediction_full_stack_batch[local_idx, channel_index]
                    )
                    gt_patch_values = (
                        ground_truth_patch[channel_index]
                        if ground_truth_patch is not None
                        else y_denorm_batch[local_idx, channel_index]
                    )
                    target_valid_mask = target_valid_full_mask_batch[
                        local_idx, channel_index
                    ]
                    gt_patch_values = np.where(
                        target_valid_mask,
                        gt_patch_values,
                        np.nan,
                    )
                    signed_error_accumulator = signed_error_accumulators[level.suffix]
                    _accumulate_patch_into_arrays(
                        signed_error_accumulator.sum_array,
                        signed_error_accumulator.count_array,
                        row=row,
                        patch_values=pred_patch_values - gt_patch_values,
                        layout=layout,
                    )
            if uncertainty_batch is not None and uncertainty_accumulators:
                if uncertainty_collapse_depth:
                    uncertainty_items = [("collapsed", 0)]
                else:
                    uncertainty_items = [
                        (level.suffix, depth_idx)
                        for depth_idx, level in enumerate(depth_export_levels)
                    ]
                for uncertainty_suffix, uncertainty_depth_idx in uncertainty_items:
                    uncertainty_accumulator = uncertainty_accumulators[
                        uncertainty_suffix
                    ]
                    _accumulate_patch_into_arrays(
                        uncertainty_accumulator.sum_array,
                        uncertainty_accumulator.count_array,
                        row=row,
                        patch_values=uncertainty_batch[
                            local_idx, uncertainty_depth_idx
                        ],
                        layout=layout,
                    )
            if argo_points_writer is not None:
                # Use the dataset's horizontal support mask so the globe shows one
                # marker per observed location instead of one marker per depth level.
                observed_mask_2d = (
                    batch[variable_spec.x_valid_mask_1d_key][local_idx, 0]
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
                    batch[variable_spec.x_valid_mask_1d_key][local_idx, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool, copy=False)
                )

            if (
                export_full_profiles
                and prediction_full_stack_batch is not None
                and x_denorm_batch is not None
                and y_denorm_batch is not None
                and eo_denorm_batch is not None
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
                        x_profile_c=x_denorm_batch[
                            local_idx, :, point_row, point_col
                        ].copy(),
                        y_hat_profile_c=prediction_full_stack_batch[
                            local_idx, :, point_row, point_col
                        ].copy(),
                        y_target_profile_c=y_denorm_batch[
                            local_idx, :, point_row, point_col
                        ].copy(),
                        ostia_sst_c=float(
                            eo_denorm_batch[local_idx, 0, point_row, point_col]
                        ),
                        observed_profile=batch[variable_spec.x_valid_mask_key][
                            local_idx, :, point_row, point_col
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool, copy=False),
                        target_valid_profile=batch[variable_spec.y_valid_mask_key][
                            local_idx, :, point_row, point_col
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool, copy=False),
                    )
                    if export_all_full_samples:
                        if full_sample_locations_writer is None:
                            raise RuntimeError(
                                "Full-sample writer was not initialized for all-location export."
                            )
                        _write_full_profile_sample_artifacts(
                            run_dir=run_dir,
                            dataset=dataset,
                            writer=full_sample_locations_writer,
                            sample=candidate,
                            location_id=f"full_sample_{observed_point_total:03d}",
                            variable_spec=variable_spec,
                        )
                    else:
                        _maybe_store_full_profile_sample(
                            samples=sampled_full_profiles,
                            seen_count=int(observed_point_total),
                            limit=full_sample_count,
                            rng=full_sample_rng,
                            candidate=candidate,
                        )

    for accumulator in pred_accumulators.values():
        _flush_accumulator(accumulator)
    for accumulator in gt_accumulators.values():
        _flush_accumulator(accumulator)
    for accumulator in signed_error_accumulators.values():
        _flush_accumulator(accumulator)
    for accumulator in uncertainty_accumulators.values():
        _flush_accumulator(accumulator)
    if argo_points_writer is not None:
        argo_points_writer.close()
    patch_splits_writer.close()
    if full_sample_locations_writer is not None:
        full_sample_locations_writer.close()
        if full_sample_locations_writer.feature_count < 1:
            if full_sample_locations_geojson_path is not None:
                full_sample_locations_geojson_path.unlink(missing_ok=True)
            full_sample_locations_geojson_path = None
            if graphs_dir_path is not None and graphs_dir_path.exists():
                graphs_dir_path.rmdir()
            graphs_dir_path = None
    print(f"Finished inference for {len(selection.indices)} patches.")

    if (
        not export_all_full_samples
        and full_sample_locations_geojson_path is not None
        and len(sampled_full_profiles) > 0
    ):
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
            _write_full_profile_sample_artifacts(
                run_dir=run_dir,
                dataset=dataset,
                writer=full_sample_locations_writer,
                sample=sample,
                location_id=f"full_sample_{sample_idx:03d}",
                variable_spec=variable_spec,
            )
        full_sample_locations_writer.close()
    elif not export_all_full_samples:
        full_sample_locations_geojson_path = None
        graphs_dir_path = None

    depth_export_records: list[dict[str, Any]] = []
    persist_ground_truth_rasters = bool(args.persist_ground_truth_rasters)
    for level in (depth_export_levels if export_prediction else ()):
        prediction_tif_path_for_level = (
            run_dir / f"{run_stem}_prediction_{level.suffix}.tif"
        )
        common_depth_tags = {
            "selected_date": str(int(selection.selected_date)),
            "selected_patch_count": str(int(len(selection.indices))),
            "variable": variable_spec.name,
            "variable_label": variable_spec.label,
            "depth_label": level.label,
            "requested_depth_m": f"{float(level.requested_depth_m):.3f}",
            "actual_depth_m": f"{float(level.actual_depth_m):.3f}",
            "channel_index": str(int(level.channel_index)),
            "land_mask_path": str(effective_land_mask_path),
            "land_zeroed": "false",
            "land_masked_to_nodata": "true",
            "value_units": variable_spec.value_units,
            "value_unit_label": variable_spec.value_unit_label,
            "value_space": variable_spec.value_space,
            "globe_color_palette": variable_spec.color_palette,
            "globe_color_scale_min": f"{float(variable_spec.color_scale_min):.3f}",
            "globe_color_scale_max": f"{float(variable_spec.color_scale_max):.3f}",
        }
        if variable_spec.name == "temperature":
            common_depth_tags.update(
                {
                    "globe_color_scale_min_c": f"{float(variable_spec.color_scale_min):.3f}",
                    "globe_color_scale_max_c": f"{float(variable_spec.color_scale_max):.3f}",
                }
            )

        write_global_top_band_geotiff(
            output_path=prediction_tif_path_for_level,
            accumulator=pred_accumulators[level.suffix],
            layout=layout,
            nodata=float(args.nodata),
            band_description=(
                f"predicted_{level.suffix}_{variable_spec.band_description_unit}"
            ),
            tags={
                **common_depth_tags,
                "source": (
                    f"DepthDif global weekly {variable_spec.name} inference export"
                ),
                "checkpoint_path": str(ckpt_path),
                "kind": "prediction",
                "prediction_runs_per_patch": "1",
                "sampler": str(sampling_metadata["sampler"]),
                "diffusion_num_timesteps": str(
                    int(sampling_metadata["diffusion_num_timesteps"])
                ),
                "ddim_num_timesteps": str(int(sampling_metadata["ddim_num_timesteps"])),
                "extra_gaussian_blur_sigma": f"{float(args.sigma):.3f}",
                "extra_gaussian_blur_kernel_size": str(
                    int(DEFAULT_EXPORT_GAUSSIAN_BLUR_KERNEL_SIZE)
                ),
                "prediction_zero_masked_to_nodata": "true",
                "source_value_transform": (
                    variable_spec.prediction_source_value_transform
                ),
            },
            extra_gaussian_blur_sigma=float(args.sigma),
            land_mask=land_mask,
            prediction_zero_masked_to_nodata=True,
            periodic_longitude_blend_width=longitude_wrap_blend_width,
        )
        print(f"Wrote prediction GeoTIFF: {prediction_tif_path_for_level}")

        ground_truth_tif_path_for_level: Path | None = None
        ground_truth_summary_path_for_level: Path | None = None
        if level.suffix in gt_accumulators:
            ground_truth_tif_path_for_level = (
                run_dir / f"{run_stem}_glorys_{level.suffix}.tif"
                if persist_ground_truth_rasters
                else scratch_dir / f"{run_stem}_glorys_{level.suffix}.tif"
            )
            if persist_ground_truth_rasters:
                ground_truth_summary_path_for_level = ground_truth_tif_path_for_level
            write_global_top_band_geotiff(
                output_path=ground_truth_tif_path_for_level,
                accumulator=gt_accumulators[level.suffix],
                layout=layout,
                nodata=float(args.nodata),
                band_description=(
                    f"glorys_{level.suffix}_{variable_spec.band_description_unit}"
                ),
                tags={
                    **common_depth_tags,
                    "source": (
                        f"DepthDif global weekly {variable_spec.name} ground-truth export"
                    ),
                    "kind": "ground_truth",
                    "source_value_transform": (
                        variable_spec.ground_truth_source_value_transform
                    ),
                },
                land_mask=land_mask,
                prediction_zero_masked_to_nodata=False,
            )
            if persist_ground_truth_rasters:
                print(f"Wrote GLORYS GeoTIFF: {ground_truth_tif_path_for_level}")

        absolute_error_tif_path_for_level: Path | None = None
        if ground_truth_tif_path_for_level is not None:
            absolute_error_tif_path_for_level = (
                run_dir / f"{run_stem}_absolute_error_{level.suffix}.tif"
            )
            write_absolute_error_geotiff(
                prediction_path=prediction_tif_path_for_level,
                ground_truth_path=ground_truth_tif_path_for_level,
                output_path=absolute_error_tif_path_for_level,
                nodata=float(args.nodata),
                band_description=(
                    f"absolute_error_{level.suffix}_{variable_spec.band_description_unit}"
                ),
                tags={
                    **common_depth_tags,
                    "source": (
                        f"DepthDif global weekly {variable_spec.name} absolute-error export"
                    ),
                    "kind": "absolute_error",
                    "source_prediction_tif_path": prediction_tif_path_for_level.name,
                    "source_ground_truth_tif_path": ground_truth_tif_path_for_level.name,
                    "source_ground_truth_persisted": str(
                        bool(persist_ground_truth_rasters)
                    ).lower(),
                    "source_value_transform": (
                        variable_spec.absolute_error_source_value_transform
                    ),
                    "value_space": variable_spec.absolute_error_value_space,
                    "globe_color_palette": "absolute_error_green_red",
                    "globe_color_scale_min_percentile": (
                        f"{float(DEFAULT_ABSOLUTE_ERROR_SCALE_MIN_PERCENTILE):.3f}"
                    ),
                    "globe_color_scale_max_percentile": (
                        f"{float(DEFAULT_ABSOLUTE_ERROR_SCALE_MAX_PERCENTILE):.3f}"
                    ),
                },
            )
            if not persist_ground_truth_rasters:
                ground_truth_tif_path_for_level.unlink(missing_ok=True)
            print(
                "Wrote absolute-error GeoTIFF: " f"{absolute_error_tif_path_for_level}"
            )

        depth_export_records.append(
            {
                "suffix": level.suffix,
                "label": level.label,
                "variable": variable_spec.name,
                "variable_label": variable_spec.label,
                "value_units": variable_spec.value_units,
                "value_unit_label": variable_spec.value_unit_label,
                "color_scale_min": float(variable_spec.color_scale_min),
                "color_scale_max": float(variable_spec.color_scale_max),
                "color_palette": variable_spec.color_palette,
                "requested_depth_m": float(level.requested_depth_m),
                "actual_depth_m": float(level.actual_depth_m),
                "channel_index": int(level.channel_index),
                "prediction_tif_path": _summary_artifact_path(
                    prediction_tif_path_for_level
                ),
                "ground_truth_tif_path": (
                    None
                    if ground_truth_summary_path_for_level is None
                    else _summary_artifact_path(ground_truth_summary_path_for_level)
                ),
                "absolute_error_tif_path": (
                    None
                    if absolute_error_tif_path_for_level is None
                    else _summary_artifact_path(absolute_error_tif_path_for_level)
                ),
            }
        )

    uncertainty_tif_path: Path | None = None
    if uncertainty_accumulators:
        uncertainty_levels = (
            [depth_export_levels[0]]
            if uncertainty_collapse_depth
            else depth_export_levels
        )
        uncertainty_record_paths: dict[str, Path] = {}
        for level in uncertainty_levels:
            uncertainty_key = (
                "collapsed" if uncertainty_collapse_depth else level.suffix
            )
            uncertainty_tif_path_for_level = (
                run_dir / f"{run_stem}_uncertainty.tif"
                if uncertainty_collapse_depth
                else run_dir / f"{run_stem}_uncertainty_{level.suffix}.tif"
            )
            if uncertainty_tif_path is None:
                uncertainty_tif_path = uncertainty_tif_path_for_level
            uncertainty_record_paths[level.suffix] = uncertainty_tif_path_for_level
            write_global_top_band_geotiff(
                output_path=uncertainty_tif_path_for_level,
                accumulator=uncertainty_accumulators[uncertainty_key],
                layout=layout,
                nodata=float(args.nodata),
                band_description=(
                    f"uncertainty_std_{level.suffix}_{variable_spec.band_description_unit}"
                ),
                tags={
                    "selected_date": str(int(selection.selected_date)),
                    "selected_patch_count": str(int(len(selection.indices))),
                    "variable": variable_spec.name,
                    "variable_label": variable_spec.label,
                    "depth_label": level.label,
                    "requested_depth_m": f"{float(level.requested_depth_m):.3f}",
                    "actual_depth_m": f"{float(level.actual_depth_m):.3f}",
                    "channel_index": str(int(level.channel_index)),
                    "land_mask_path": str(effective_land_mask_path),
                    "land_zeroed": "false",
                    "land_masked_to_nodata": "true",
                    "value_units": variable_spec.value_units,
                    "value_unit_label": variable_spec.value_unit_label,
                    "value_space": f"generation_uncertainty_std_{variable_spec.band_description_unit}",
                    "source": (
                        f"DepthDif global weekly {variable_spec.name} uncertainty export"
                    ),
                    "checkpoint_path": str(ckpt_path),
                    "kind": "uncertainty",
                    "uncertainty_stat": "std",
                    "uncertainty_num_samples": str(int(uncertainty_num_samples)),
                    "uncertainty_collapse_depth": str(
                        bool(uncertainty_collapse_depth)
                    ).lower(),
                    "sampler": str(uncertainty_sampling_metadata["sampler"]),
                    "diffusion_num_timesteps": str(
                        int(uncertainty_sampling_metadata["diffusion_num_timesteps"])
                    ),
                    "ddim_num_timesteps": str(
                        int(uncertainty_sampling_metadata["ddim_num_timesteps"])
                    ),
                    "source_value_transform": (
                        "std(model_prediction_denormalized_repeated_samples)"
                    ),
                    "prediction_zero_masked_to_nodata": "false",
                },
                land_mask=land_mask,
                prediction_zero_masked_to_nodata=False,
                periodic_longitude_blend_width=longitude_wrap_blend_width,
            )
            print(f"Wrote uncertainty GeoTIFF: {uncertainty_tif_path_for_level}")
        for record in depth_export_records:
            record_path = uncertainty_record_paths.get(str(record["suffix"]))
            record["uncertainty_tif_path"] = (
                None if record_path is None else _summary_artifact_path(record_path)
            )

    prediction_tif_path = (
        None
        if not depth_export_records
        else run_dir / str(depth_export_records[0]["prediction_tif_path"])
    )
    ground_truth_tif_path = (
        None
        if not depth_export_records
        or depth_export_records[0]["ground_truth_tif_path"] is None
        else run_dir / str(depth_export_records[0]["ground_truth_tif_path"])
    )
    absolute_error_tif_path = (
        None
        if not depth_export_records
        or depth_export_records[0]["absolute_error_tif_path"] is None
        else run_dir / str(depth_export_records[0]["absolute_error_tif_path"])
    )
    if ground_truth_tif_path is not None:
        print(f"Wrote Argo points GeoJSON: {argo_points_geojson_path}")
    if full_sample_locations_geojson_path is not None and graphs_dir_path is not None:
        print(
            f"Wrote full-sample locations GeoJSON: {full_sample_locations_geojson_path}"
        )
        print(f"Wrote full-sample graphs directory: {graphs_dir_path}")
    print(f"Wrote patch split GeoJSON: {patch_splits_geojson_path}")

    run_summary = {
        "selected_date": int(selection.selected_date),
        "target_date": int(selection.selected_date),
        "iso_year": int(selection.iso_year),
        "iso_week": int(selection.iso_week),
        "selected_patch_count": int(len(selection.indices)),
        "variable": variable_spec.name,
        "variable_label": variable_spec.label,
        "value_units": variable_spec.value_units,
        "value_unit_label": variable_spec.value_unit_label,
        "color_scale_min": float(variable_spec.color_scale_min),
        "color_scale_max": float(variable_spec.color_scale_max),
        "color_palette": variable_spec.color_palette,
        "rectangle": (
            None
            if args.rectangle is None
            else [float(value) for value in args.rectangle]
        ),
        "inference_grid": inference_grid_metadata,
        "validation_year": (
            None if configured_val_year is None else int(configured_val_year)
        ),
        "global_export_uses_all_patches": True,
        "global_export_split_note": (
            "Global raster export uses split=all for complete world coverage; "
            "the configured validation year identifies validation-year rows."
        ),
        "land_mask_path": str(effective_land_mask_path),
        "land_zeroed": False,
        "land_masked_to_nodata": True,
        "longitude_wrap_stitching": bool(longitude_wrap_blend_width > 0),
        "longitude_wrap_blend_width_pixels": int(longitude_wrap_blend_width),
        "prediction_zero_masked_to_nodata": bool(export_prediction),
        "uncertainty_only": bool(uncertainty_only),
        "prediction_zero_mask_epsilon_c": float(PREDICTION_ZERO_ARTIFACT_EPSILON),
        "checkpoint_path": str(ckpt_path),
        "model_config": str(args.model_config),
        "data_config": str(args.data_config),
        "train_config": str(args.train_config),
        "inference_config": str(args.inference_config),
        "device": str(device),
        "visible_gpus": int(visible_gpu_count),
        "multi_gpu_enabled": bool(use_multi_gpu),
        "batch_size": int(batch_size),
        "sampler": str(sampling_metadata["sampler"]),
        "diffusion_num_timesteps": int(sampling_metadata["diffusion_num_timesteps"]),
        "ddim_num_timesteps": int(sampling_metadata["ddim_num_timesteps"]),
        "uncertainty_sampler": str(uncertainty_sampling_metadata["sampler"]),
        "uncertainty_diffusion_num_timesteps": int(
            uncertainty_sampling_metadata["diffusion_num_timesteps"]
        ),
        "uncertainty_ddim_num_timesteps": int(
            uncertainty_sampling_metadata["ddim_num_timesteps"]
        ),
        "prediction_runs_per_patch": 1 if export_prediction else 0,
        "export_uncertainty": bool(export_uncertainty),
        "uncertainty_collapse_depth": bool(uncertainty_collapse_depth),
        "uncertainty_num_samples": (
            int(uncertainty_num_samples) if export_uncertainty else None
        ),
        "extra_gaussian_blur_sigma": float(args.sigma),
        "extra_gaussian_blur_kernel_size": int(
            DEFAULT_EXPORT_GAUSSIAN_BLUR_KERNEL_SIZE
        ),
        "split": str(args.split),
        "run_dir": (
            str(production_dir) if production_dir is not None else str(run_dir)
        ),
        "prediction_tif_path": (
            None
            if prediction_tif_path is None
            else _summary_artifact_path(prediction_tif_path)
        ),
        "ground_truth_tif_path": (
            None
            if ground_truth_tif_path is None
            else _summary_artifact_path(ground_truth_tif_path)
        ),
        "absolute_error_tif_path": (
            None
            if absolute_error_tif_path is None
            else _summary_artifact_path(absolute_error_tif_path)
        ),
        "uncertainty_tif_path": (
            None
            if uncertainty_tif_path is None
            else _summary_artifact_path(uncertainty_tif_path)
        ),
        "error_analysis_json_path": None,
        "error_analysis_grid_geojson_path": None,
        "temporal_basin_depth_error_json_path": None,
        "depth_exports": depth_export_records,
        "argo_points_geojson_path": (
            None
            if argo_points_geojson_path is None
            else _summary_artifact_path(argo_points_geojson_path)
        ),
        "argo_point_count": (
            0 if argo_points_writer is None else int(argo_points_writer.feature_count)
        ),
        "full_sample_count_requested": int(requested_full_sample_count),
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
        "globe_packaging": None,
    }
    if export_full_depth_error_analysis:
        error_analysis_json_path = run_dir / DEFAULT_ANALYSIS_JSON_NAME
        error_analysis_grid_geojson_path = run_dir / DEFAULT_ANALYSIS_GRID_GEOJSON_NAME
        run_summary["error_analysis_json_path"] = _summary_artifact_path(
            error_analysis_json_path
        )
        run_summary["error_analysis_grid_geojson_path"] = _summary_artifact_path(
            error_analysis_grid_geojson_path
        )
        _write_depth_error_analysis_json(
            output_path=error_analysis_json_path,
            run_summary=run_summary,
            variable_spec=variable_spec,
            analysis_depth_levels=analysis_depth_levels,
            signed_error_accumulators=signed_error_accumulators,
            layout=layout,
            land_mask=land_mask,
        )
        write_analysis_grid_geojson(
            output_path=error_analysis_grid_geojson_path,
            land_mask_path=effective_land_mask_path,
        )
        print(f"Wrote depth error analysis JSON: {error_analysis_json_path}")
        print(f"Wrote analysis ocean grid GeoJSON: {error_analysis_grid_geojson_path}")
    if export_compact_basin_depth_error:
        compact_error_path = run_dir / DEFAULT_TEMPORAL_BASIN_DEPTH_ERRORS_JSON_NAME
        run_summary["temporal_basin_depth_error_json_path"] = _summary_artifact_path(
            compact_error_path
        )
        _write_compact_basin_depth_error_json(
            output_path=compact_error_path,
            run_summary=run_summary,
            variable_spec=variable_spec,
            analysis_depth_levels=analysis_depth_levels,
            signed_error_accumulators=signed_error_accumulators,
            layout=layout,
            land_mask=land_mask,
        )
        print(f"Wrote compact basin-depth error JSON: {compact_error_path}")
    with (run_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    for accumulator in pred_accumulators.values():
        _cleanup_accumulator(accumulator)
    for accumulator in gt_accumulators.values():
        _cleanup_accumulator(accumulator)
    for accumulator in signed_error_accumulators.values():
        _cleanup_accumulator(accumulator)
    for accumulator in uncertainty_accumulators.values():
        _cleanup_accumulator(accumulator)
    scratch_dir.rmdir()

    if production_dir is not None:
        _promote_production_run(run_dir, production_dir)
        run_dir = production_dir
        if prediction_tif_path is not None:
            prediction_tif_path = run_dir / _production_artifact_name(
                prediction_tif_path.name,
                run_stem=_default_run_stem(selection.selected_date),
            )
        if ground_truth_tif_path is not None:
            ground_truth_tif_path = run_dir / _production_artifact_name(
                ground_truth_tif_path.name,
                run_stem=_default_run_stem(selection.selected_date),
            )
        if absolute_error_tif_path is not None:
            absolute_error_tif_path = run_dir / _production_artifact_name(
                absolute_error_tif_path.name,
                run_stem=_default_run_stem(selection.selected_date),
            )
        if uncertainty_tif_path is not None:
            uncertainty_tif_path = run_dir / _production_artifact_name(
                uncertainty_tif_path.name,
                run_stem=_default_run_stem(selection.selected_date),
            )
        if argo_points_geojson_path is not None:
            argo_points_geojson_path = run_dir / argo_points_geojson_path.name
        if full_sample_locations_geojson_path is not None:
            full_sample_locations_geojson_path = (
                run_dir / full_sample_locations_geojson_path.name
            )
        if graphs_dir_path is not None:
            graphs_dir_path = run_dir / graphs_dir_path.name
        patch_splits_geojson_path = run_dir / patch_splits_geojson_path.name

    packaging_result: dict[str, Any] | None = None
    if args.public_base_url is not None or args.rclone_remote is not None:
        packaging_result = export_cesium_globe_assets(
            run_dir=run_dir,
            public_base_url=args.public_base_url,
            rclone_remote=args.rclone_remote,
            rclone_sync_scope=args.rclone_sync_scope,
            extra_zoom_levels=args.extra_zoom_levels,
        )
        summary_path = run_dir / "run_summary.yaml"
        with summary_path.open("r", encoding="utf-8") as f:
            packaged_summary = yaml.safe_load(f) or {}
        packaged_summary["globe_packaging"] = packaging_result
        with summary_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(packaged_summary, f, sort_keys=False)

    print(
        "Export complete: "
        f"date={selection.selected_date}, "
        f"iso_week={selection.iso_year}-W{selection.iso_week:02d}, "
        f"variable={variable_spec.name}, "
        f"patches={len(selection.indices)}, "
        f"prediction={prediction_tif_path}, "
        f"ground_truth={ground_truth_tif_path}, "
        f"absolute_error={absolute_error_tif_path}, "
        f"uncertainty={uncertainty_tif_path}, "
        f"argo_points={argo_points_geojson_path}, "
        f"full_sample_locations={full_sample_locations_geojson_path}, "
        f"graphs={graphs_dir_path}, "
        f"patch_splits={patch_splits_geojson_path}, "
        f"globe_packaging={packaging_result}"
    )
    return ExportRunResult(
        run_dir=run_dir,
        summary_path=run_dir / "run_summary.yaml",
        prediction_tif_path=prediction_tif_path,
        ground_truth_tif_path=ground_truth_tif_path,
        selected_date=int(selection.selected_date),
        iso_year=int(selection.iso_year),
        iso_week=int(selection.iso_week),
        selected_patch_count=int(len(selection.indices)),
        absolute_error_tif_path=absolute_error_tif_path,
        uncertainty_tif_path=uncertainty_tif_path,
        variable=variable_spec.name,
    )


def main() -> None:
    """Run the CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(_normalize_cli_args(sys.argv[1:]))
    run_global_inference(args)


if __name__ == "__main__":
    main()
