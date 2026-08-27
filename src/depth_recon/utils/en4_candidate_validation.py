from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import default_collate

from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.validation_denoise import (
    log_wandb_average_depth_errors,
    log_wandb_average_depth_profiles,
)


@dataclass(frozen=True)
class CandidateProfileResult:
    """One prediction/GLORYS/EN4 profile comparison used for W&B plots."""

    variable: str
    date: int
    latitude: float
    longitude: float
    profile_source_file: str
    source_profile_idx: int
    prediction: np.ndarray
    glorys: np.ndarray
    en4: np.ndarray


@dataclass(frozen=True)
class CandidatePatchImageData:
    """Physical full-patch fields used for one candidate reconstruction figure."""

    variable: str
    patch_number: int
    date: int
    input_values: np.ndarray
    prediction: np.ndarray
    glorys: np.ndarray
    heldout_rows: np.ndarray
    heldout_cols: np.ndarray


def _metric_summary(results: list[CandidateProfileResult]) -> dict[str, float]:
    """Compute pooled prediction and GLORYS errors against EN4 profiles."""
    prediction_errors: list[np.ndarray] = []
    glorys_errors: list[np.ndarray] = []
    for result in results:
        valid = (
            np.isfinite(result.en4)
            & np.isfinite(result.prediction)
            & np.isfinite(result.glorys)
        )
        if bool(np.any(valid)):
            prediction_errors.append(result.prediction[valid] - result.en4[valid])
            glorys_errors.append(result.glorys[valid] - result.en4[valid])
    if not prediction_errors:
        return {
            "prediction_rmse": float("nan"),
            "glorys_rmse": float("nan"),
            "prediction_mae": float("nan"),
            "glorys_mae": float("nan"),
            "skill_vs_glorys": float("nan"),
            "valid_value_count": 0.0,
        }
    prediction_error = np.concatenate(prediction_errors).astype(np.float64)
    glorys_error = np.concatenate(glorys_errors).astype(np.float64)
    prediction_rmse = float(np.sqrt(np.mean(np.square(prediction_error))))
    glorys_rmse = float(np.sqrt(np.mean(np.square(glorys_error))))
    return {
        "prediction_rmse": prediction_rmse,
        "glorys_rmse": glorys_rmse,
        "prediction_mae": float(np.mean(np.abs(prediction_error))),
        "glorys_mae": float(np.mean(np.abs(glorys_error))),
        "skill_vs_glorys": (
            float(1.0 - prediction_rmse / glorys_rmse)
            if glorys_rmse > 0.0
            else float("nan")
        ),
        "valid_value_count": float(prediction_error.size),
    }


def _profile_figure(
    results: list[CandidateProfileResult],
    *,
    depth_axis_m: np.ndarray,
    max_profiles: int,
) -> plt.Figure:
    """Build value and absolute-error panels for deterministic EN4 profiles."""
    plotted = results[: max(1, int(max_profiles))]
    figure, axes = plt.subplots(
        len(plotted), 2, figsize=(12.0, max(4.0, 3.5 * len(plotted))), squeeze=False
    )
    units = "PSU" if plotted[0].variable == "salinity" else "deg C"
    for row_idx, result in enumerate(plotted):
        value_ax, error_ax = axes[row_idx]
        valid_en4 = np.isfinite(result.en4)
        value_ax.plot(result.glorys, depth_axis_m, color="black", label="GLORYS12")
        value_ax.plot(
            result.prediction,
            depth_axis_m,
            color="tab:orange",
            label="Prediction",
        )
        value_ax.scatter(
            result.en4[valid_en4],
            depth_axis_m[valid_en4],
            color="tab:blue",
            marker="o",
            s=20,
            label="EN4 profile",
            zorder=5,
        )
        error_ax.plot(
            np.abs(result.prediction - result.en4),
            depth_axis_m,
            color="tab:orange",
            label="|Prediction - EN4|",
        )
        error_ax.plot(
            np.abs(result.glorys - result.en4),
            depth_axis_m,
            color="black",
            label="|GLORYS12 - EN4|",
        )
        title = (
            f"{result.date} | {result.latitude:.2f}, {result.longitude:.2f}\n"
            f"{result.profile_source_file} #{result.source_profile_idx}"
        )
        value_ax.set_title(title, fontsize=9)
        value_ax.set_xlabel(f"Profile value ({units})")
        error_ax.set_xlabel(f"Absolute error ({units})")
        for axis in (value_ax, error_ax):
            axis.set_ylabel("Depth (m)")
            if bool(np.any(valid_en4)):
                # Keep both panels focused on the depths observed by this EN4 profile.
                deepest_en4_depth = float(np.max(depth_axis_m[valid_en4]))
                axis.set_ylim(max(deepest_en4_depth, 1.0), 0.0)
            else:
                axis.invert_yaxis()
            axis.grid(True, alpha=0.25)
        if row_idx == 0:
            value_ax.legend(loc="best")
            error_ax.legend(loc="best")
    figure.suptitle(f"EN4 candidate evaluation: {plotted[0].variable}", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return figure


class EN4CandidateValidationCallback(pl.Callback):
    """Evaluate a fixed candidate-profile patch subset during validation epochs."""

    def __init__(
        self,
        *,
        dataset: Any,
        candidate_df: pd.DataFrame,
        holdout_fraction: float = 0.2,
        min_input_profiles: int = 8,
        max_patches: int = 1,
        max_profiles_to_plot: int = 6,
        random_seed: int = 7,
        image_depths_m: tuple[float, ...] = (0.0, 100.0, 500.0),
    ) -> None:
        """Prepare deterministic patches and exact EN4 profiles for epoch evaluation."""
        super().__init__()
        if not hasattr(dataset, "_rows") or not hasattr(dataset, "argo_store"):
            raise TypeError(
                "EN4 candidate validation requires the active GeoTIFF EN4 dataset."
            )
        if dataset.argo_store is None:
            raise RuntimeError(
                "EN4 candidate validation requires a compact profile store."
            )
        fraction = float(holdout_fraction)
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("EN4 holdout fraction must be in (0, 1).")
        if int(min_input_profiles) < 0:
            raise ValueError("min_input_profiles cannot be negative.")
        if candidate_df.empty:
            raise ValueError("candidate_df must contain at least one EN4 profile.")
        self.dataset = dataset
        self.candidate_df = candidate_df.reset_index(drop=True).copy()
        self.candidate_metadata = dict(candidate_df.attrs)
        self.holdout_fraction = fraction
        self.min_input_profiles = int(min_input_profiles)
        self.max_profiles_to_plot = max(1, int(max_profiles_to_plot))
        self.random_seed = int(random_seed)
        self.image_depths_m = tuple(float(value) for value in image_depths_m)
        if not self.image_depths_m:
            raise ValueError("image_depths_m cannot be empty.")
        (
            self.patch_indices,
            self.holdout_df,
            self.profile_assignments,
            self.selection_metadata,
        ) = self._select_eval_patches(max_patches=max_patches)
        dataset.set_heldout_argo_locations(
            [
                (int(row.date), int(row.grid_row), int(row.grid_col))
                for row in self.holdout_df.itertuples(index=False)
            ]
        )
        profile_indices = self.profile_assignments["profile_index"].to_numpy(
            dtype=np.int64
        )
        self.temperature_profiles = dataset.argo_store.load_temperature_profiles(
            profile_indices
        )
        self.salinity_profiles = (
            dataset.argo_store.load_salinity_profiles(profile_indices)
            if bool(dataset.argo_store.include_salinity)
            else None
        )
        self.depth_axis_m = np.asarray(
            dataset.argo_store.depth_axis_m, dtype=np.float64
        )
        self._latest_patch_images: dict[str, list[CandidatePatchImageData]] = {}
        self._last_logged_global_step: int | None = None

    def _select_eval_patches(
        self, *, max_patches: int
    ) -> tuple[list[int], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Uniformly select patches that retain enough profiles after local holdout."""
        limit = max(1, int(max_patches))
        rows = self.dataset._rows.reset_index(drop=True)
        selected_dates = self.candidate_df["date"].astype(np.int64).unique()
        if int(selected_dates.size) != 1:
            raise ValueError("candidate_df must contain exactly one target date.")
        selected_date = int(selected_dates[0])
        candidate_rows = rows.index[rows["date"].astype(np.int64) == selected_date]
        locations = {
            (int(row.grid_row), int(row.grid_col))
            for row in self.candidate_df.itertuples(index=False)
        }
        coverage_by_patch: dict[int, set[tuple[int, int]]] = {}
        profile_indices_by_patch: dict[int, np.ndarray] = {}
        local_holdouts_by_patch: dict[int, set[tuple[int, int]]] = {}
        tile_size = int(self.dataset.tile_size)
        for patch_idx in candidate_rows.tolist():
            patch = rows.iloc[int(patch_idx)]
            y0, x0 = int(patch.grid_y0), int(patch.grid_x0)
            covered = {
                location
                for location in locations
                if y0 <= location[0] < y0 + tile_size
                and x0 <= location[1] < x0 + tile_size
            }
            if covered:
                coverage_by_patch[int(patch_idx)] = covered
                profile_indices = self.dataset.argo_store.query_indices(
                    target_date=selected_date,
                    grid_y0=y0,
                    grid_x0=x0,
                    tile_size=tile_size,
                )
                profile_indices_by_patch[int(patch_idx)] = profile_indices
                holdout_count = int(round(len(covered) * self.holdout_fraction))
                holdout_count = min(max(holdout_count, 1), len(covered))
                ordered_locations = sorted(covered)
                # A patch-derived seed makes each local holdout stable independently
                # of row ordering and of how many other patches qualify.
                patch_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [self.random_seed, selected_date, max(y0, 0), max(x0, 0)]
                    )
                )
                selected_positions = patch_rng.choice(
                    np.arange(len(ordered_locations)),
                    size=holdout_count,
                    replace=False,
                )
                local_holdouts_by_patch[int(patch_idx)] = {
                    ordered_locations[int(position)]
                    for position in selected_positions.tolist()
                }

        def retained_profile_count(
            patch_idx: int, heldout_locations: set[tuple[int, int]]
        ) -> int:
            """Count source profiles left in a patch after removing locations."""
            indices = profile_indices_by_patch[patch_idx]
            store = self.dataset.argo_store
            return sum(
                (
                    int(store.grid_row[int(profile_idx)]),
                    int(store.grid_col[int(profile_idx)]),
                )
                not in heldout_locations
                for profile_idx in indices.tolist()
            )

        qualifying = [
            patch_idx
            for patch_idx in sorted(coverage_by_patch)
            if retained_profile_count(patch_idx, local_holdouts_by_patch[patch_idx])
            >= self.min_input_profiles
        ]
        if not qualifying:
            raise RuntimeError(
                "No EN4 candidate validation patch retains at least "
                f"{self.min_input_profiles} QC-valid input profiles after holdout."
            )

        rng = np.random.default_rng(self.random_seed)
        randomized_candidates = rng.permutation(np.asarray(qualifying, dtype=np.int64))
        chosen: list[int] = []
        heldout_locations: set[tuple[int, int]] = set()
        retained_counts: dict[int, int] = {}
        for raw_patch_idx in randomized_candidates.tolist():
            patch_idx = int(raw_patch_idx)
            proposed_holdouts = heldout_locations | local_holdouts_by_patch[patch_idx]
            proposed_patches = chosen + [patch_idx]
            proposed_counts = {
                selected_patch: retained_profile_count(
                    selected_patch, proposed_holdouts
                )
                for selected_patch in proposed_patches
            }
            if min(proposed_counts.values()) < self.min_input_profiles:
                continue
            chosen.append(patch_idx)
            heldout_locations = proposed_holdouts
            retained_counts = proposed_counts
            if len(chosen) >= limit:
                break
        if not chosen:
            raise RuntimeError(
                "No EN4 candidate validation patch satisfies the retained-profile "
                "minimum after combining held-out locations."
            )

        holdout_df = self.candidate_df.loc[
            [
                (int(row.grid_row), int(row.grid_col)) in heldout_locations
                for row in self.candidate_df.itertuples(index=False)
            ]
        ].copy()
        assignments: list[dict[str, Any]] = []
        for row in holdout_df.to_dict(orient="records"):
            location = (int(row["grid_row"]), int(row["grid_col"]))
            for batch_idx, patch_idx in enumerate(chosen):
                if location not in coverage_by_patch[patch_idx]:
                    continue
                patch = rows.iloc[patch_idx]
                assignments.append(
                    {
                        **row,
                        "eval_batch_index": int(batch_idx),
                        "local_grid_row": int(location[0] - int(patch.grid_y0)),
                        "local_grid_col": int(location[1] - int(patch.grid_x0)),
                    }
                )
                break
        profile_assignments = pd.DataFrame.from_records(assignments)
        selection_metadata = {
            **self.candidate_metadata,
            "candidate_patch_count": int(len(coverage_by_patch)),
            "qualifying_patch_count": int(len(qualifying)),
            "selected_patch_count": int(len(chosen)),
            "selected_location_count": int(len(heldout_locations)),
            "selected_profile_count": int(len(holdout_df)),
            "min_input_profiles": int(self.min_input_profiles),
            "retained_input_profile_count_min": int(min(retained_counts.values())),
            "retained_input_profile_count_max": int(max(retained_counts.values())),
            "retained_input_profile_count_total": int(sum(retained_counts.values())),
            "holdout_fraction": float(self.holdout_fraction),
            "split_seed": int(self.random_seed),
        }
        holdout_df.attrs.update(selection_metadata)
        return chosen, holdout_df, profile_assignments, selection_metadata

    def _build_batch(self) -> dict[str, Any]:
        """Load the fixed evaluation patches through the normal validation dataset."""
        return default_collate([self.dataset[index] for index in self.patch_indices])

    @staticmethod
    def _denormalize_target(variable: str, tensor: torch.Tensor) -> torch.Tensor:
        """Convert one dense GLORYS validation target to physical units."""
        if variable == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

    def _image_depth_indices(self, depth_count: int) -> list[int]:
        """Map requested image depths to unique nearest output channels."""
        available_depths = self.depth_axis_m[: int(depth_count)]
        indices: list[int] = []
        for requested_depth_m in self.image_depths_m:
            index = int(np.argmin(np.abs(available_depths - requested_depth_m)))
            if index not in indices:
                indices.append(index)
        return indices

    @staticmethod
    def _shared_value_limits(*arrays: np.ndarray) -> tuple[float, float]:
        """Return robust shared limits for physical input/reference/prediction fields."""
        finite_parts = [values[np.isfinite(values)] for values in arrays]
        finite_parts = [values for values in finite_parts if values.size > 0]
        if not finite_parts:
            return 0.0, 1.0
        vmin, vmax = np.percentile(np.concatenate(finite_parts), [2.0, 98.0])
        if vmax <= vmin:
            padding = max(abs(float(vmin)) * 0.01, 1.0e-6)
            return float(vmin - padding), float(vmax + padding)
        return float(vmin), float(vmax)

    def _reconstruction_figure(self, image_data: CandidatePatchImageData) -> plt.Figure:
        """Build sparse-input, GLORYS, prediction, and error full-patch panels."""
        depth_indices = self._image_depth_indices(image_data.prediction.shape[0])
        figure, axes = plt.subplots(
            len(depth_indices),
            4,
            figsize=(16.0, max(3.5, 3.4 * len(depth_indices))),
            squeeze=False,
        )
        value_cmap = "viridis" if image_data.variable == "salinity" else "coolwarm"
        units = "PSU" if image_data.variable == "salinity" else "deg C"
        for row_index, depth_index in enumerate(depth_indices):
            input_values = image_data.input_values[depth_index]
            glorys = image_data.glorys[depth_index]
            prediction = image_data.prediction[depth_index]
            absolute_error = np.abs(prediction - glorys)
            vmin, vmax = self._shared_value_limits(input_values, glorys, prediction)
            finite_error = absolute_error[np.isfinite(absolute_error)]
            error_max = (
                float(np.percentile(finite_error, 98.0))
                if finite_error.size > 0
                else 1.0
            )
            panels = (
                (input_values, "Sparse EN4 input", value_cmap, vmin, vmax),
                (glorys, "GLORYS12", value_cmap, vmin, vmax),
                (prediction, "Reconstruction", value_cmap, vmin, vmax),
                (absolute_error, "Absolute error", "magma", 0.0, max(error_max, 1e-6)),
            )
            for column_index, (values, title, cmap, panel_min, panel_max) in enumerate(
                panels
            ):
                axis = axes[row_index, column_index]
                image = axis.imshow(
                    values,
                    cmap=cmap,
                    vmin=panel_min,
                    vmax=panel_max,
                    interpolation="nearest",
                )
                axis.scatter(
                    image_data.heldout_cols,
                    image_data.heldout_rows,
                    marker="x",
                    s=36,
                    linewidths=1.5,
                    color="red",
                    label="Held-out EN4",
                )
                actual_depth_m = float(self.depth_axis_m[depth_index])
                axis.set_title(f"{title} | {actual_depth_m:g} m")
                axis.set_xticks([])
                axis.set_yticks([])
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=units)
                if row_index == 0 and column_index == 0:
                    axis.legend(loc="best", fontsize=8)
        figure.suptitle(
            f"EN4 candidate full reconstruction: {image_data.variable} | "
            f"patch {image_data.patch_number} | {image_data.date}"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return figure

    @torch.no_grad()
    def evaluate(
        self, pl_module: pl.LightningModule
    ) -> dict[str, list[CandidateProfileResult]]:
        """Run one deterministic candidate reconstruction and return profile results."""
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

        results: dict[str, list[CandidateProfileResult]] = {}
        self._latest_patch_images = {}
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        for variable in fields:
            profile_values = (
                self.salinity_profiles
                if variable == "salinity"
                else self.temperature_profiles
            )
            if profile_values is None:
                continue
            prediction_key = f"y_hat_{variable}_denorm"
            predicted = prediction.get(prediction_key)
            if predicted is None and len(fields) == 1:
                predicted = prediction.get("y_hat_denorm")
            target_key = "y_salinity" if variable == "salinity" else "y"
            glorys_key = "y_salinity_glorys" if variable == "salinity" else "y_glorys"
            target_valid_key = (
                "y_salinity_valid_mask" if variable == "salinity" else "y_valid_mask"
            )
            glorys_valid_key = (
                "y_salinity_glorys_valid_mask"
                if variable == "salinity"
                else "y_glorys_valid_mask"
            )
            input_key = "x_salinity" if variable == "salinity" else "x"
            input_valid_key = (
                "x_salinity_valid_mask" if variable == "salinity" else "x_valid_mask"
            )
            if not torch.is_tensor(predicted) or target_key not in batch:
                continue
            glorys_target = batch.get(glorys_key)
            if glorys_target is None:
                # Direct-GLORYS and custom datasets retain the legacy y fallback.
                glorys_target = batch[target_key]
                glorys_valid_mask = batch.get(target_valid_key)
            else:
                glorys_valid_mask = batch.get(glorys_valid_key)
            glorys = self._denormalize_target(variable, glorys_target)
            if torch.is_tensor(glorys_valid_mask):
                # Invalid normalized targets are stored as zero, so restore NaNs
                # before plotting or computing profile metrics.
                glorys = torch.where(
                    glorys_valid_mask.to(device=glorys.device, dtype=torch.bool),
                    glorys,
                    torch.full_like(glorys, float("nan")),
                )
            input_values = batch.get(input_key)
            input_valid_mask = batch.get(input_valid_key)
            variable_images: list[CandidatePatchImageData] = []
            if torch.is_tensor(input_values):
                input_physical = self._denormalize_target(variable, input_values)
                if torch.is_tensor(input_valid_mask):
                    input_physical = torch.where(
                        input_valid_mask.to(
                            device=input_physical.device, dtype=torch.bool
                        ),
                        input_physical,
                        torch.full_like(input_physical, float("nan")),
                    )
                rows = self.dataset._rows.reset_index(drop=True)
                for batch_idx, patch_idx in enumerate(self.patch_indices):
                    patch = rows.iloc[int(patch_idx)]
                    y0, x0 = int(patch.grid_y0), int(patch.grid_x0)
                    glorys_image = glorys[batch_idx]
                    # Restrict the reconstruction panel to the same valid-ocean
                    # support as GLORYS so land/nodata predictions are not visualized.
                    prediction_image = torch.where(
                        torch.isfinite(glorys_image),
                        predicted[batch_idx],
                        torch.full_like(predicted[batch_idx], float("nan")),
                    )
                    patch_holdouts = self.holdout_df.loc[
                        self.holdout_df["grid_row"].between(
                            y0, y0 + int(self.dataset.tile_size) - 1
                        )
                        & self.holdout_df["grid_col"].between(
                            x0, x0 + int(self.dataset.tile_size) - 1
                        )
                    ][["grid_row", "grid_col"]].drop_duplicates()
                    variable_images.append(
                        CandidatePatchImageData(
                            variable=str(variable),
                            patch_number=int(batch_idx),
                            date=int(patch.date),
                            input_values=(
                                input_physical[batch_idx].detach().float().cpu().numpy()
                            ),
                            prediction=(
                                prediction_image.detach().float().cpu().numpy()
                            ),
                            glorys=(glorys_image.detach().float().cpu().numpy()),
                            heldout_rows=(
                                patch_holdouts["grid_row"].to_numpy(dtype=np.int64) - y0
                            ),
                            heldout_cols=(
                                patch_holdouts["grid_col"].to_numpy(dtype=np.int64) - x0
                            ),
                        )
                    )
            self._latest_patch_images[str(variable)] = variable_images
            variable_results: list[CandidateProfileResult] = []
            for profile_row, assignment in self.profile_assignments.iterrows():
                batch_idx = int(assignment["eval_batch_index"])
                row_idx = int(assignment["local_grid_row"])
                col_idx = int(assignment["local_grid_col"])
                variable_results.append(
                    CandidateProfileResult(
                        variable=str(variable),
                        date=int(assignment["date"]),
                        latitude=float(assignment["lat"]),
                        longitude=float(assignment["lon"]),
                        profile_source_file=str(assignment["profile_source_file"]),
                        source_profile_idx=int(assignment["source_profile_idx"]),
                        prediction=(
                            predicted[batch_idx, :, row_idx, col_idx]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        ),
                        glorys=(
                            glorys[batch_idx, :, row_idx, col_idx]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        ),
                        en4=np.asarray(
                            profile_values[int(profile_row)], dtype=np.float32
                        ),
                    )
                )
            results[str(variable)] = variable_results
        return results

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log candidate metrics and profile figures during each validation run."""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        global_step = int(trainer.global_step)
        if self._last_logged_global_step == global_step:
            return
        logger = trainer.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb

            results = self.evaluate(pl_module)
            payload: dict[str, Any] = {
                "en4_candidate_eval/monitored_profile_count": int(
                    len(self.profile_assignments)
                ),
                "en4_candidate_eval/monitored_location_count": int(
                    self.profile_assignments[["date", "grid_row", "grid_col"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
            for count_name in (
                "eligible_profile_count",
                "eligible_location_count",
                "selected_profile_count",
                "selected_location_count",
                "candidate_patch_count",
                "qualifying_patch_count",
                "selected_patch_count",
                "min_input_profiles",
                "retained_input_profile_count_min",
                "retained_input_profile_count_max",
                "retained_input_profile_count_total",
            ):
                if count_name in self.selection_metadata:
                    payload[f"en4_candidate_eval/{count_name}"] = int(
                        self.selection_metadata[count_name]
                    )
            figures: list[plt.Figure] = []
            try:
                for variable, variable_results in results.items():
                    summary = _metric_summary(variable_results)
                    for metric, value in summary.items():
                        payload[f"en4_candidate_eval/{variable}_{metric}"] = value
                    for image_data in self._latest_patch_images.get(variable, []):
                        reconstruction_figure = self._reconstruction_figure(image_data)
                        figures.append(reconstruction_figure)
                        payload[
                            "en4_candidate_eval/"
                            f"{variable}_patch_{image_data.patch_number}_"
                            "full_reconstruction"
                        ] = wandb.Image(reconstruction_figure)
                    if variable_results:
                        figure = _profile_figure(
                            variable_results,
                            depth_axis_m=self.depth_axis_m,
                            max_profiles=self.max_profiles_to_plot,
                        )
                        figures.append(figure)
                        payload[f"en4_candidate_eval/{variable}_profiles"] = (
                            wandb.Image(figure)
                        )
                        log_wandb_average_depth_profiles(
                            logger=logger,
                            profiles={
                                "Prediction": np.stack(
                                    [result.prediction for result in variable_results]
                                ),
                                "GLORYS": np.stack(
                                    [result.glorys for result in variable_results]
                                ),
                                "EN4": np.stack(
                                    [result.en4 for result in variable_results]
                                ),
                            },
                            depth_axis_m=self.depth_axis_m,
                            depth_dimension=1,
                            prefix="en4_candidate_eval",
                            image_key=f"{variable}_average_profile_by_depth",
                            value_label=(
                                "Salinity (PSU)"
                                if variable == "salinity"
                                else "Temperature (deg C)"
                            ),
                            title=f"Average EN4 candidate profile: {variable}",
                        )
                        if variable == "temperature":
                            log_wandb_average_depth_errors(
                                logger=logger,
                                predictions={
                                    "Prediction": np.stack(
                                        [
                                            result.prediction
                                            for result in variable_results
                                        ]
                                    ),
                                    "GLORYS": np.stack(
                                        [result.glorys for result in variable_results]
                                    ),
                                },
                                reference=np.stack(
                                    [result.en4 for result in variable_results]
                                ),
                                depth_axis_m=self.depth_axis_m,
                                depth_dimension=1,
                                prefix="en4_candidate_eval",
                                image_key="temperature_average_absolute_error_by_depth",
                                error_label="Mean absolute error vs EN4 (deg C)",
                                title="Average temperature absolute error vs EN4",
                            )
                experiment.log(payload)
            finally:
                for figure in figures:
                    plt.close(figure)
            self._last_logged_global_step = global_step
        except Exception as exc:
            warnings.warn(
                f"EN4 candidate validation logging failed: {exc}", stacklevel=2
            )
