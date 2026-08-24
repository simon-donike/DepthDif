from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from shapely import intersects_xy
from shapely.geometry import box, shape
import torch
from torch.utils.data import default_collate
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import _prior_patch_coordinates
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize


@dataclass(frozen=True)
class HardRegionDefinition:
    """One named hard-region polygon used during Lightning validation."""

    region_id: str
    label: str
    geometry: Any


@dataclass(frozen=True)
class HardRegionMetricResult:
    """Pooled prediction-vs-GLORYS metrics for one region and variable."""

    count: int
    rmse: float
    mae: float


@dataclass(frozen=True)
class HardRegionImageData:
    """Physical prediction/reference arrays used for one regional figure."""

    variable: str
    region_id: str
    region_label: str
    date: int
    prediction: np.ndarray
    reference: np.ndarray
    support: np.ndarray


def load_hard_region_definitions(
    path: Path,
) -> tuple[list[HardRegionDefinition], int | None]:
    """Load named Polygon or MultiPolygon regions from YAML or GeoJSON."""
    region_path = Path(path)
    with region_path.open("r", encoding="utf-8") as f:
        payload = (
            yaml.safe_load(f)
            if region_path.suffix.lower() in {".yaml", ".yml"}
            else json.load(f)
        )
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError(
            f"Hard-region file must contain a GeoJSON FeatureCollection: {region_path}"
        )

    regions: list[HardRegionDefinition] = []
    seen_ids: set[str] = set()
    for feature_index, feature in enumerate(payload.get("features", [])):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(
                f"Hard-region feature {feature_index} must be a GeoJSON Feature."
            )
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            raise ValueError(
                f"Hard-region feature {feature_index} properties must be a mapping."
            )
        raw_id = properties.get("id") or feature.get("id") or properties.get("name")
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(
                f"Hard-region feature {feature_index} requires an id or name."
            )
        region_id = str(raw_id).strip()
        if region_id in seen_ids:
            raise ValueError(f"Duplicate hard-region id: {region_id}")
        geometry = shape(feature.get("geometry"))
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"Hard-region feature {feature_index} must use Polygon or "
                "MultiPolygon geometry."
            )
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Hard-region geometry is empty or invalid: {region_id}")
        seen_ids.add(region_id)
        regions.append(
            HardRegionDefinition(
                region_id=region_id,
                label=str(
                    properties.get("label") or properties.get("name") or region_id
                ),
                geometry=geometry,
            )
        )
    if not regions:
        raise ValueError(f"Hard-region file contains no region features: {region_path}")
    collection_properties = payload.get("properties") or {}
    if not isinstance(collection_properties, dict):
        raise ValueError("Hard-region FeatureCollection properties must be a mapping.")
    raw_year = collection_properties.get("evaluation_year")
    return regions, None if raw_year is None else int(raw_year)


def _patch_geometry(row: pd.Series) -> Any:
    """Return one patch footprint, including dateline-wrapped footprints."""
    lon0 = float(row["lon0"])
    lon1 = float(row["lon1"])
    lat0 = min(float(row["lat0"]), float(row["lat1"]))
    lat1 = max(float(row["lat0"]), float(row["lat1"]))
    if lon0 <= lon1:
        return box(lon0, lat0, lon1, lat1)
    # A wrapped patch is represented by its two WGS84 edge pieces.
    return box(lon0, lat0, 180.0, lat1).union(box(-180.0, lat0, lon1, lat1))


class HardRegionValidationCallback(pl.Callback):
    """Evaluate fixed hard-region patches against GLORYS during validation."""

    def __init__(
        self,
        *,
        dataset: Any,
        regions_path: Path,
        evaluation_year: int = 2016,
        max_patches_per_region: int = 1,
        random_seed: int = 7,
        image_depths_m: Sequence[float] = (0.0, 100.0, 500.0),
    ) -> None:
        """Select deterministic validation-year patches for each hard region."""
        super().__init__()
        if not hasattr(dataset, "_rows") or not hasattr(dataset, "tile_size"):
            raise TypeError(
                "Hard-region validation requires the active GeoTIFF patch dataset."
            )
        self.dataset = dataset
        self.evaluation_year = int(evaluation_year)
        self.max_patches_per_region = max(1, int(max_patches_per_region))
        self.random_seed = int(random_seed)
        self.image_depths_m = tuple(float(value) for value in image_depths_m)
        if not self.image_depths_m:
            raise ValueError("hard_region_eval.image_depths_m cannot be empty.")
        self.depth_axis_m = np.asarray(dataset.depth_axis_m, dtype=np.float64)
        self.regions, geometry_year = load_hard_region_definitions(regions_path)
        if geometry_year is not None and geometry_year != self.evaluation_year:
            raise ValueError(
                f"Hard-region geometry year {geometry_year} does not match "
                f"evaluation year {self.evaluation_year}."
            )
        self.patch_indices, self.region_masks, self.selection_summary = (
            self._select_eval_patches()
        )
        self._last_logged_global_step: int | None = None
        self._latest_image_data: dict[tuple[str, str], HardRegionImageData] = {}

    def _select_eval_patches(
        self,
    ) -> tuple[list[int], dict[str, np.ndarray], dict[str, Any]]:
        """Select fixed validation-year patches and build their regional masks."""
        rows = self.dataset._rows.reset_index(drop=True)
        row_years = pd.to_numeric(
            rows["date"].astype(str).str.slice(0, 4), errors="coerce"
        ).to_numpy()
        year_indices = np.flatnonzero(row_years == self.evaluation_year).astype(int)
        if year_indices.size == 0:
            raise RuntimeError(
                f"Hard-region validation found no {self.evaluation_year} dataset rows."
            )

        selected_by_region: dict[str, list[int]] = {}
        candidate_counts: dict[str, int] = {}
        for region in self.regions:
            candidates = [
                int(row_index)
                for row_index in year_indices.tolist()
                if region.geometry.intersects(_patch_geometry(rows.iloc[row_index]))
            ]
            candidates = sorted(
                candidates,
                key=lambda index: (
                    int(rows.iloc[index]["date"]),
                    int(rows.iloc[index]["grid_y0"]),
                    int(rows.iloc[index]["grid_x0"]),
                ),
            )
            candidate_counts[region.region_id] = len(candidates)
            if not candidates:
                raise RuntimeError(
                    f"Hard region {region.region_id!r} matched no "
                    f"{self.evaluation_year} validation patches."
                )
            selection_count = min(self.max_patches_per_region, len(candidates))
            positions = (
                np.asarray([len(candidates) // 2], dtype=np.int64)
                if selection_count == 1
                else np.linspace(
                    0, len(candidates) - 1, num=selection_count, dtype=np.int64
                )
            )
            selected_by_region[region.region_id] = [
                candidates[int(position)] for position in positions.tolist()
            ]

        patch_indices = sorted(
            {
                patch_index
                for selected in selected_by_region.values()
                for patch_index in selected
            }
        )
        batch_index_by_patch = {
            patch_index: batch_index
            for batch_index, patch_index in enumerate(patch_indices)
        }
        region_masks: dict[str, np.ndarray] = {}
        for region in self.regions:
            masks = np.zeros(
                (
                    len(patch_indices),
                    int(self.dataset.tile_size),
                    int(self.dataset.tile_size),
                ),
                dtype=bool,
            )
            for patch_index in selected_by_region[region.region_id]:
                row = rows.iloc[patch_index]
                latitude, longitude = _prior_patch_coordinates(
                    row, int(self.dataset.tile_size)
                )
                masks[batch_index_by_patch[patch_index]] = intersects_xy(
                    region.geometry, longitude, latitude
                )
            if not bool(np.any(masks)):
                raise RuntimeError(
                    f"Hard region {region.region_id!r} contains no selected pixel centers."
                )
            region_masks[region.region_id] = masks

        summary = {
            "evaluation_year": self.evaluation_year,
            "candidate_patch_counts": candidate_counts,
            "selected_patch_count": len(patch_indices),
            "selected_dates": sorted(
                {int(rows.iloc[index]["date"]) for index in patch_indices}
            ),
        }
        return patch_indices, region_masks, summary

    def _build_batch(self) -> dict[str, Any]:
        """Load fixed hard-region samples through the validation dataset."""
        return default_collate([self.dataset[index] for index in self.patch_indices])

    @staticmethod
    def _denormalize(variable: str, tensor: torch.Tensor) -> torch.Tensor:
        """Convert one normalized model field to physical units."""
        if variable == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

    @staticmethod
    def _align_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Broadcast a one-channel validation mask across depth when needed."""
        aligned = mask.to(device=reference.device, dtype=torch.bool)
        if aligned.ndim == reference.ndim - 1:
            aligned = aligned.unsqueeze(1)
        if int(aligned.size(1)) == 1 and int(reference.size(1)) > 1:
            aligned = aligned.expand(-1, int(reference.size(1)), -1, -1)
        if tuple(aligned.shape) != tuple(reference.shape):
            raise RuntimeError(
                f"Hard-region mask shape {tuple(aligned.shape)} does not match "
                f"field shape {tuple(reference.shape)}."
            )
        return aligned

    @torch.no_grad()
    def evaluate(
        self, pl_module: pl.LightningModule
    ) -> dict[str, dict[str, HardRegionMetricResult]]:
        """Run fixed regional predictions and compare them with dense GLORYS."""
        self._latest_image_data = {}
        batch = self._build_batch()
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
        device_index = (
            pl_module.device.index if pl_module.device.type == "cuda" else None
        )
        fork_context = (
            torch.random.fork_rng(devices=[device_index])
            if device_index is not None
            else torch.random.fork_rng(devices=[])
        )
        with fork_context:
            torch.manual_seed(self.random_seed)
            if device_index is not None:
                torch.cuda.manual_seed_all(self.random_seed)
            prediction = pl_module.predict_step(batch, batch_idx=0)

        results: dict[str, dict[str, HardRegionMetricResult]] = {}
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        for variable in fields:
            prediction_key = f"y_hat_{variable}_denorm"
            predicted = prediction.get(prediction_key)
            if predicted is None and len(fields) == 1:
                predicted = prediction.get("y_hat_denorm")
            if not torch.is_tensor(predicted):
                continue
            target_key = "y_salinity" if variable == "salinity" else "y"
            glorys_key = "y_salinity_glorys" if variable == "salinity" else "y_glorys"
            valid_key = (
                "y_salinity_valid_mask" if variable == "salinity" else "y_valid_mask"
            )
            glorys_valid_key = (
                "y_salinity_glorys_valid_mask"
                if variable == "salinity"
                else "y_glorys_valid_mask"
            )
            glorys_target = batch.get(glorys_key, batch.get(target_key))
            if not torch.is_tensor(glorys_target):
                continue
            reference = self._denormalize(variable, glorys_target)
            valid_mask = batch.get(glorys_valid_key, batch.get(valid_key))
            if torch.is_tensor(valid_mask):
                support = self._align_mask(valid_mask, reference)
            else:
                support = torch.ones_like(reference, dtype=torch.bool)
            land_mask = batch.get("land_mask")
            if torch.is_tensor(land_mask):
                support = support & self._align_mask(land_mask, reference)
            support = support & torch.isfinite(predicted) & torch.isfinite(reference)

            variable_results: dict[str, HardRegionMetricResult] = {}
            for region in self.regions:
                region_mask = torch.from_numpy(self.region_masks[region.region_id]).to(
                    device=reference.device
                )
                region_support = support & self._align_mask(region_mask, reference)
                count = int(torch.count_nonzero(region_support).item())
                if count < 1:
                    variable_results[region.region_id] = HardRegionMetricResult(
                        count=0, rmse=float("nan"), mae=float("nan")
                    )
                    continue
                error = predicted[region_support] - reference[region_support]
                variable_results[region.region_id] = HardRegionMetricResult(
                    count=count,
                    rmse=float(torch.sqrt(torch.mean(torch.square(error))).item()),
                    mae=float(torch.mean(torch.abs(error)).item()),
                )
                selected_batches = np.flatnonzero(
                    np.any(self.region_masks[region.region_id], axis=(1, 2))
                )
                image_batch_index = int(
                    selected_batches[int(selected_batches.size) // 2]
                )
                self._latest_image_data[(region.region_id, str(variable))] = (
                    HardRegionImageData(
                        variable=str(variable),
                        region_id=region.region_id,
                        region_label=region.label,
                        date=int(
                            self.dataset._rows.iloc[
                                self.patch_indices[image_batch_index]
                            ]["date"]
                        ),
                        prediction=(
                            predicted[image_batch_index].detach().float().cpu().numpy()
                        ),
                        reference=(
                            reference[image_batch_index].detach().float().cpu().numpy()
                        ),
                        support=(
                            region_support[image_batch_index]
                            .detach()
                            .bool()
                            .cpu()
                            .numpy()
                        ),
                    )
                )
            results[str(variable)] = variable_results
        return results

    def _image_depth_indices(self, depth_count: int) -> list[int]:
        """Map requested image depths to unique nearest model depth channels."""
        available_depths = self.depth_axis_m[: int(depth_count)]
        indices: list[int] = []
        for requested_depth_m in self.image_depths_m:
            index = int(np.argmin(np.abs(available_depths - requested_depth_m)))
            if index not in indices:
                indices.append(index)
        return indices

    @staticmethod
    def _shared_value_limits(
        reference: np.ndarray, prediction: np.ndarray
    ) -> tuple[float, float]:
        """Return robust shared color limits for prediction/reference panels."""
        values = np.concatenate(
            [reference[np.isfinite(reference)], prediction[np.isfinite(prediction)]]
        )
        if values.size == 0:
            return 0.0, 1.0
        vmin, vmax = np.percentile(values.astype(np.float64), [2.0, 98.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if vmax <= vmin:
            padding = max(abs(float(vmin)) * 0.01, 1.0e-6)
            return float(vmin - padding), float(vmax + padding)
        return float(vmin), float(vmax)

    def _region_figure(self, image_data: HardRegionImageData) -> plt.Figure:
        """Build GLORYS, prediction, and absolute-error regional image panels."""
        depth_indices = self._image_depth_indices(image_data.prediction.shape[0])
        figure, axes = plt.subplots(
            len(depth_indices),
            3,
            figsize=(12.0, max(3.5, 3.4 * len(depth_indices))),
            squeeze=False,
        )
        value_cmap = "viridis" if image_data.variable == "salinity" else "coolwarm"
        units = "PSU" if image_data.variable == "salinity" else "deg C"
        for row_index, depth_index in enumerate(depth_indices):
            support = image_data.support[depth_index]
            reference = np.where(support, image_data.reference[depth_index], np.nan)
            prediction = np.where(support, image_data.prediction[depth_index], np.nan)
            absolute_error = np.abs(prediction - reference)
            vmin, vmax = self._shared_value_limits(reference, prediction)
            finite_error = absolute_error[np.isfinite(absolute_error)]
            error_max = (
                float(np.percentile(finite_error, 98.0))
                if finite_error.size > 0
                else 1.0
            )
            error_max = max(error_max, 1.0e-6)
            panels = (
                (reference, "GLORYS12", value_cmap, vmin, vmax, units),
                (prediction, "Prediction", value_cmap, vmin, vmax, units),
                (
                    absolute_error,
                    "Absolute error",
                    "magma",
                    0.0,
                    error_max,
                    units,
                ),
            )
            for column_index, (
                values,
                title,
                cmap,
                panel_min,
                panel_max,
                label,
            ) in enumerate(panels):
                axis = axes[row_index, column_index]
                image = axis.imshow(
                    values,
                    cmap=cmap,
                    vmin=panel_min,
                    vmax=panel_max,
                    interpolation="nearest",
                )
                actual_depth_m = float(self.depth_axis_m[depth_index])
                axis.set_title(f"{title} | {actual_depth_m:g} m")
                axis.set_xticks([])
                axis.set_yticks([])
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=label)
        figure.suptitle(
            f"Hard-region validation: {image_data.region_label} | "
            f"{image_data.variable} | {image_data.date}",
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return figure

    def _log_region_figures(self, trainer: pl.Trainer) -> None:
        """Log the latest regional comparison figures to a W&B experiment."""
        if not trainer.is_global_zero:
            return
        experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb

            figures: list[plt.Figure] = []
            payload: dict[str, Any] = {}
            try:
                for (
                    region_id,
                    variable,
                ), image_data in self._latest_image_data.items():
                    figure = self._region_figure(image_data)
                    figures.append(figure)
                    payload[f"hard_region_eval/{region_id}/{variable}_comparison"] = (
                        wandb.Image(figure)
                    )
                if payload:
                    experiment.log(payload)
            finally:
                for figure in figures:
                    plt.close(figure)
        except Exception as exc:
            warnings.warn(
                f"Hard-region validation image logging failed: {exc}", stacklevel=2
            )

    @staticmethod
    def _logging_payload(
        results: dict[str, dict[str, HardRegionMetricResult]],
        *,
        selected_patch_count: int,
    ) -> dict[str, float]:
        """Build Lightning scalar keys for regional and pooled validation metrics."""
        payload: dict[str, float] = {
            "hard_region_eval/monitored_patch_count": float(selected_patch_count)
        }
        for variable, region_results in results.items():
            total_count = sum(result.count for result in region_results.values())
            squared_error_sum = 0.0
            absolute_error_sum = 0.0
            for region_id, result in region_results.items():
                prefix = f"hard_region_eval/{region_id}/{variable}"
                payload[f"{prefix}_rmse"] = result.rmse
                payload[f"{prefix}_mae"] = result.mae
                payload[f"{prefix}_valid_value_count"] = float(result.count)
                if result.count > 0 and np.isfinite(result.rmse):
                    squared_error_sum += result.rmse**2 * result.count
                    absolute_error_sum += result.mae * result.count
            payload[f"hard_region_eval/{variable}_rmse"] = (
                float(np.sqrt(squared_error_sum / total_count))
                if total_count > 0
                else float("nan")
            )
            payload[f"hard_region_eval/{variable}_mae"] = (
                float(absolute_error_sum / total_count)
                if total_count > 0
                else float("nan")
            )
        return payload

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run and log hard-region metrics during each real validation epoch."""
        if trainer.sanity_checking:
            return
        global_step = int(trainer.global_step)
        if self._last_logged_global_step == global_step:
            return
        try:
            results = self.evaluate(pl_module)
            payload = self._logging_payload(
                results,
                selected_patch_count=int(
                    self.selection_summary["selected_patch_count"]
                ),
            )
            pl_module.log_dict(
                payload,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=1,
            )
            self._log_region_figures(trainer)
            self._last_logged_global_step = global_step
        except Exception as exc:
            warnings.warn(f"Hard-region validation failed: {exc}", stacklevel=2)
