from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, default_collate

from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.validation_denoise import (
    _finite_mean_profile,
    _finite_std_profile,
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
    max_profiles: int | None,
) -> plt.Figure:
    """Build value and absolute-error panels for deterministic EN4 profiles."""
    if not results:
        raise ValueError("At least one candidate profile result is required.")
    if max_profiles is None:
        # Overlay the complete evaluation set so plotting all profiles remains
        # bounded in figure dimensions while preserving every evaluated trace.
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 8.0), squeeze=False)
        value_ax, error_ax = axes[0]
        units = "PSU" if results[0].variable == "salinity" else "deg C"
        for result_idx, result in enumerate(results):
            valid_en4 = np.isfinite(result.en4)
            label = result_idx == 0
            value_ax.plot(
                result.glorys,
                depth_axis_m,
                color="black",
                alpha=0.12,
                label="GLORYS12" if label else "_nolegend_",
            )
            value_ax.plot(
                result.prediction,
                depth_axis_m,
                color="tab:orange",
                alpha=0.12,
                label="Prediction" if label else "_nolegend_",
            )
            value_ax.scatter(
                result.en4[valid_en4],
                depth_axis_m[valid_en4],
                color="tab:blue",
                marker=".",
                s=10,
                alpha=0.12,
                label="EN4 profile" if label else "_nolegend_",
                zorder=5,
            )
            error_ax.plot(
                np.abs(result.prediction - result.en4),
                depth_axis_m,
                color="tab:orange",
                alpha=0.12,
                label="|Prediction - EN4|" if label else "_nolegend_",
            )
            error_ax.plot(
                np.abs(result.glorys - result.en4),
                depth_axis_m,
                color="black",
                alpha=0.12,
                label="|GLORYS12 - EN4|" if label else "_nolegend_",
            )
        deepest_en4_depth = max(
            (
                float(np.max(depth_axis_m[np.isfinite(result.en4)]))
                for result in results
                if bool(np.any(np.isfinite(result.en4)))
            ),
            default=float(np.max(depth_axis_m)),
        )
        for axis in (value_ax, error_ax):
            axis.set_ylim(max(deepest_en4_depth, 1.0), 0.0)
            axis.set_ylabel("Depth (m)")
            axis.grid(True, alpha=0.25)
        value_ax.set_xlabel(f"Profile value ({units})")
        error_ax.set_xlabel(f"Absolute error ({units})")
        value_ax.set_title(f"All evaluated profiles ({len(results)})")
        error_ax.set_title(f"All evaluated profiles ({len(results)})")
        value_ax.legend(loc="best")
        error_ax.legend(loc="best")
        figure.suptitle(f"EN4 candidate evaluation: {results[0].variable}", fontsize=14)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return figure

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


def _aggregate_profile_figure(
    results: list[CandidateProfileResult], *, depth_axis_m: np.ndarray
) -> plt.Figure:
    """Build aggregate mean and standard-deviation profile traces."""
    if not results:
        raise ValueError("At least one candidate profile result is required.")
    profiles = {
        "Prediction": np.stack([result.prediction for result in results]),
        "GLORYS": np.stack([result.glorys for result in results]),
        "EN4": np.stack([result.en4 for result in results]),
    }
    colors = {"Prediction": "tab:orange", "GLORYS": "black", "EN4": "tab:blue"}
    figure, axis = plt.subplots(1, 1, figsize=(7.0, 8.0))
    for label, values in profiles.items():
        mean_profile = _finite_mean_profile(values, depth_dimension=1)
        std_profile = _finite_std_profile(values, depth_dimension=1)
        color = colors[label]
        axis.plot(mean_profile, depth_axis_m, label=label, color=color, linewidth=1.8)
        axis.fill_betweenx(
            depth_axis_m,
            mean_profile - std_profile,
            mean_profile + std_profile,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
    units = "PSU" if results[0].variable == "salinity" else "deg C"
    axis.set_xlabel(f"Profile value ({units})")
    axis.set_ylabel("Depth (m)")
    axis.set_title(f"Aggregate validated held-out EN4 profiles (n={len(results)})")
    axis.invert_yaxis()
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


class EN4CandidateValidationCallback(pl.Callback):
    """Evaluate a deterministic candidate-profile patch set during validation."""

    def __init__(
        self,
        *,
        dataset: Any,
        candidate_df: pd.DataFrame,
        holdout_fraction: float = 0.2,
        min_input_profiles: int = 8,
        max_patches: int | None = None,
        max_profiles_to_plot: int | None = None,
        max_patch_images_to_log: int = 3,
        patch_batch_size: int = 8,
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
        self.max_profiles_to_plot = (
            None
            if max_profiles_to_plot is None
            else max(1, int(max_profiles_to_plot))
        )
        self.max_patch_images_to_log = max(1, int(max_patch_images_to_log))
        self.patch_batch_size = max(1, int(patch_batch_size))
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
        self._epoch_results: dict[str, list[CandidateProfileResult]] = {}
        self._candidate_loader_batches_seen = 0
        self._last_logged_global_step: int | None = None

    @property
    def validation_dataset(self) -> Subset[Any]:
        """Return selected candidate patches for the second validation loader."""
        return Subset(self.dataset, self.patch_indices)

    def _select_eval_patches(
        self, *, max_patches: int | None
    ) -> tuple[list[int], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Select candidate patches after one global location holdout."""
        limit = None if max_patches is None else max(1, int(max_patches))
        rows = self.dataset._rows.reset_index(drop=True)
        selected_dates = set(self.candidate_df["date"].astype(np.int64).tolist())
        candidate_rows = rows.index[rows["date"].astype(np.int64).isin(selected_dates)]
        candidate_locations = {
            (int(row.date), int(row.grid_row), int(row.grid_col))
            for row in self.candidate_df.itertuples(index=False)
        }
        coverage_by_patch: dict[int, set[tuple[int, int, int]]] = {}
        profile_indices_by_patch: dict[int, np.ndarray] = {}
        tile_size = int(self.dataset.tile_size)
        patch_lookup_by_date: dict[int, dict[tuple[int, int], int]] = {}
        y_origins_by_date: dict[int, np.ndarray] = {}
        x_origins_by_date: dict[int, np.ndarray] = {}
        for patch_idx in candidate_rows.tolist():
            patch = rows.iloc[int(patch_idx)]
            patch_date = int(patch.date)
            y0, x0 = int(patch.grid_y0), int(patch.grid_x0)
            patch_lookup_by_date.setdefault(patch_date, {})[(y0, x0)] = int(patch_idx)
        for patch_date, patch_lookup in patch_lookup_by_date.items():
            y_origins_by_date[patch_date] = np.asarray(
                sorted({origin[0] for origin in patch_lookup}), dtype=np.int64
            )
            x_origins_by_date[patch_date] = np.asarray(
                sorted({origin[1] for origin in patch_lookup}), dtype=np.int64
            )

        for date_value, grid_row, grid_col in sorted(candidate_locations):
            y_origins = y_origins_by_date.get(date_value)
            x_origins = x_origins_by_date.get(date_value)
            if y_origins is None or x_origins is None:
                continue
            y_start = int(np.searchsorted(y_origins, grid_row - tile_size + 1))
            y_stop = int(np.searchsorted(y_origins, grid_row, side="right"))
            x_start = int(np.searchsorted(x_origins, grid_col - tile_size + 1))
            x_stop = int(np.searchsorted(x_origins, grid_col, side="right"))
            location = (date_value, grid_row, grid_col)
            for y0 in y_origins[y_start:y_stop].tolist():
                for x0 in x_origins[x_start:x_stop].tolist():
                    patch_idx = patch_lookup_by_date[date_value].get((int(y0), int(x0)))
                    if patch_idx is not None:
                        coverage_by_patch.setdefault(patch_idx, set()).add(location)

        for patch_idx in coverage_by_patch:
            patch = rows.iloc[int(patch_idx)]
            profile_indices_by_patch[patch_idx] = self.dataset.argo_store.query_indices(
                target_date=int(patch.date),
                grid_y0=int(patch.grid_y0),
                grid_x0=int(patch.grid_x0),
                tile_size=tile_size,
            )

        locations = (
            sorted(set().union(*coverage_by_patch.values()))
            if coverage_by_patch
            else []
        )
        holdout_count = int(round(len(locations) * self.holdout_fraction))
        holdout_count = min(max(holdout_count, 1), len(locations))
        rng = np.random.default_rng(self.random_seed)
        selected_positions = rng.choice(
            np.arange(len(locations)), size=holdout_count, replace=False
        )
        heldout_locations = {
            locations[int(position)] for position in selected_positions.tolist()
        }

        def retained_profile_count(
            patch_idx: int, heldout_locations: set[tuple[int, int, int]]
        ) -> int:
            """Count source profiles left in a patch after removing locations."""
            indices = profile_indices_by_patch[patch_idx]
            store = self.dataset.argo_store
            patch_date = int(rows.iloc[patch_idx].date)
            return sum(
                (
                    patch_date,
                    int(store.grid_row[int(profile_idx)]),
                    int(store.grid_col[int(profile_idx)]),
                )
                not in heldout_locations
                for profile_idx in indices.tolist()
            )

        qualifying = [
            patch_idx
            for patch_idx in sorted(coverage_by_patch)
            if retained_profile_count(patch_idx, heldout_locations)
            >= self.min_input_profiles
        ]
        if not coverage_by_patch:
            raise RuntimeError(
                "No EN4 candidate validation patch covers the candidate locations."
            )

        # Select patches to cover every held-out location exactly once. In the
        # uncapped production mode, qualifying patches are preferred, with a
        # non-qualifying patch used only when it is needed for coverage.
        if limit is None:
            selection_pool = qualifying + [
                patch_idx for patch_idx in sorted(coverage_by_patch) if patch_idx not in qualifying
            ]
        else:
            selection_pool = qualifying
        if not selection_pool:
            raise RuntimeError(
                "No EN4 candidate validation patch retains at least "
                f"{self.min_input_profiles} QC-valid input profiles after holdout."
            )
        randomized_candidates = rng.permutation(np.asarray(selection_pool, dtype=np.int64))
        assigned_patch_by_location: dict[tuple[int, int, int], int] = {}
        chosen: list[int] = []
        for raw_patch_idx in randomized_candidates.tolist():
            patch_idx = int(raw_patch_idx)
            newly_covered = {
                location
                for location in heldout_locations
                if location in coverage_by_patch[patch_idx]
                and location not in assigned_patch_by_location
            }
            if not newly_covered:
                continue
            chosen.append(patch_idx)
            assigned_patch_by_location.update(
                {location: patch_idx for location in newly_covered}
            )
            if limit is not None and len(chosen) >= limit:
                break
            if len(assigned_patch_by_location) == len(heldout_locations):
                break
        if len(assigned_patch_by_location) != len(heldout_locations):
            raise RuntimeError(
                "EN4 candidate holdout locations are not covered by the selected "
                "candidate patches."
            )
        retained_counts = {
            patch_idx: retained_profile_count(patch_idx, heldout_locations)
            for patch_idx in chosen
        }
        if not chosen:
            raise RuntimeError(
                "No EN4 candidate validation patch satisfies the retained-profile "
                "minimum after combining held-out locations."
            )

        holdout_df = self.candidate_df.loc[
            [
                (int(row.date), int(row.grid_row), int(row.grid_col))
                in heldout_locations
                for row in self.candidate_df.itertuples(index=False)
            ]
        ].copy()
        batch_index_by_patch = {
            patch_idx: batch_idx for batch_idx, patch_idx in enumerate(chosen)
        }
        assignments: list[dict[str, Any]] = []
        for row in holdout_df.to_dict(orient="records"):
            location = (
                int(row["date"]),
                int(row["grid_row"]),
                int(row["grid_col"]),
            )
            patch_idx = assigned_patch_by_location[location]
            patch = rows.iloc[patch_idx]
            assignments.append(
                {
                    **row,
                    "eval_batch_index": int(batch_index_by_patch[patch_idx]),
                    "local_grid_row": int(location[1] - int(patch.grid_y0)),
                    "local_grid_col": int(location[2] - int(patch.grid_x0)),
                }
            )
        profile_assignments = pd.DataFrame.from_records(assignments)
        if len(profile_assignments) != len(holdout_df):
            raise RuntimeError(
                "EN4 candidate holdout profiles are not covered by the selected "
                "candidate patches."
            )
        covered_location_set = set(locations)
        selection_metadata = {
            **self.candidate_metadata,
            "candidate_patch_count": int(len(coverage_by_patch)),
            "candidate_location_count": int(len(candidate_locations)),
            "candidate_covered_location_count": int(len(locations)),
            "candidate_uncovered_location_count": int(
                len(candidate_locations - covered_location_set)
            ),
            "candidate_covered_profile_count": int(
                sum(
                    (
                        int(row.date),
                        int(row.grid_row),
                        int(row.grid_col),
                    )
                    in covered_location_set
                    for row in self.candidate_df.itertuples(index=False)
                )
            ),
            "candidate_uncovered_profile_count": int(
                sum(
                    (
                        int(row.date),
                        int(row.grid_row),
                        int(row.grid_col),
                    )
                    not in covered_location_set
                    for row in self.candidate_df.itertuples(index=False)
                )
            ),
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

    def _build_batch(self, patch_indices: list[int] | None = None) -> dict[str, Any]:
        """Load the fixed evaluation patches through the normal validation dataset."""
        indices = self.patch_indices if patch_indices is None else patch_indices
        return default_collate([self.dataset[index] for index in indices])

    @staticmethod
    def _denormalize_target(variable: str, tensor: torch.Tensor) -> torch.Tensor:
        """Convert one dense GLORYS validation target to physical units."""
        if variable == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

    @torch.no_grad()
    def _consume_validation_batch(
        self,
        batch: dict[str, Any],
        prediction: dict[str, Any],
        *,
        patch_start: int,
    ) -> None:
        """Collect profile results and patch images from one Lightning val batch."""
        fields = tuple(
            variable
            for variable in ("temperature", "salinity")
            if f"y_hat_{variable}_denorm" in prediction
        )
        if not fields and "y_hat_denorm" in prediction:
            fields = ("temperature",)
        rows = self.dataset._rows.reset_index(drop=True)
        batch_tensor = batch.get("x")
        if batch_tensor is None:
            batch_tensor = batch.get("x_salinity")
        if not torch.is_tensor(batch_tensor):
            return
        patch_indices = self.patch_indices[
            patch_start : patch_start + int(batch_tensor.size(0))
        ]
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
                glorys_target = batch[target_key]
                glorys_valid_mask = batch.get(target_valid_key)
            else:
                glorys_valid_mask = batch.get(glorys_valid_key)
            glorys = self._denormalize_target(variable, glorys_target)
            if torch.is_tensor(glorys_valid_mask):
                glorys = torch.where(
                    glorys_valid_mask.to(device=glorys.device, dtype=torch.bool),
                    glorys,
                    torch.full_like(glorys, float("nan")),
                )
            input_values = batch.get(input_key)
            input_valid_mask = batch.get(input_valid_key)
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
                for local_idx, patch_idx in enumerate(patch_indices):
                    patch = rows.iloc[int(patch_idx)]
                    y0, x0 = int(patch.grid_y0), int(patch.grid_x0)
                    glorys_image = glorys[local_idx]
                    prediction_image = torch.where(
                        torch.isfinite(glorys_image),
                        predicted[local_idx],
                        torch.full_like(predicted[local_idx], float("nan")),
                    )
                    patch_holdouts = self.holdout_df.loc[
                        self.holdout_df["date"].eq(int(patch.date))
                        & self.holdout_df["grid_row"].between(
                            y0, y0 + int(self.dataset.tile_size) - 1
                        )
                        & self.holdout_df["grid_col"].between(
                            x0, x0 + int(self.dataset.tile_size) - 1
                        )
                    ][["grid_row", "grid_col"]].drop_duplicates()
                    if (
                        len(self._latest_patch_images[str(variable)])
                        < self.max_patch_images_to_log
                    ):
                        self._latest_patch_images[str(variable)].append(
                            CandidatePatchImageData(
                                variable=str(variable),
                                patch_number=int(patch_start + local_idx),
                                date=int(patch.date),
                                input_values=input_physical[local_idx]
                                .detach()
                                .float()
                                .cpu()
                                .numpy(),
                                prediction=prediction_image.detach()
                                .float()
                                .cpu()
                                .numpy(),
                                glorys=glorys_image.detach().float().cpu().numpy(),
                                heldout_rows=patch_holdouts["grid_row"].to_numpy(
                                    dtype=np.int64
                                )
                                - y0,
                                heldout_cols=patch_holdouts["grid_col"].to_numpy(
                                    dtype=np.int64
                                )
                                - x0,
                            )
                        )
            assignments = self.profile_assignments.loc[
                self.profile_assignments["eval_batch_index"].between(
                    patch_start, patch_start + len(patch_indices) - 1
                )
            ]
            for profile_row, assignment in assignments.iterrows():
                local_idx = int(assignment["eval_batch_index"]) - patch_start
                row_idx = int(assignment["local_grid_row"])
                col_idx = int(assignment["local_grid_col"])
                self._epoch_results.setdefault(str(variable), []).append(
                    CandidateProfileResult(
                        variable=str(variable),
                        date=int(assignment["date"]),
                        latitude=float(assignment["lat"]),
                        longitude=float(assignment["lon"]),
                        profile_source_file=str(assignment["profile_source_file"]),
                        source_profile_idx=int(assignment["source_profile_idx"]),
                        prediction=predicted[local_idx, :, row_idx, col_idx]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        glorys=glorys[local_idx, :, row_idx, col_idx]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        en4=np.asarray(profile_values[int(profile_row)], dtype=np.float32),
                    )
                )

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
        """Run all selected candidate patches and return all profile results."""
        device_index = (
            pl_module.device.index if pl_module.device.type == "cuda" else None
        )
        fork_context = (
            torch.random.fork_rng(devices=[device_index])
            if device_index is not None
            else torch.random.fork_rng(devices=[])
        )
        results: dict[str, list[CandidateProfileResult]] = {
            str(variable): [] for variable in getattr(pl_module, "output_fields", ("temperature",))
        }
        self._latest_patch_images = {}
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        self._latest_patch_images = {str(variable): [] for variable in fields}
        rows = self.dataset._rows.reset_index(drop=True)
        with fork_context:
            torch.manual_seed(self.random_seed)
            if device_index is not None:
                torch.cuda.manual_seed_all(self.random_seed)
            for chunk_start in range(0, len(self.patch_indices), self.patch_batch_size):
                chunk_patch_indices = self.patch_indices[
                    chunk_start : chunk_start + self.patch_batch_size
                ]
                batch = (
                    self._build_batch()
                    if len(chunk_patch_indices) == len(self.patch_indices) == 1
                    else self._build_batch(chunk_patch_indices)
                )
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                prediction = pl_module.predict_step(batch, batch_idx=chunk_start)
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
                    glorys_key = (
                        "y_salinity_glorys" if variable == "salinity" else "y_glorys"
                    )
                    target_valid_key = (
                        "y_salinity_valid_mask"
                        if variable == "salinity"
                        else "y_valid_mask"
                    )
                    glorys_valid_key = (
                        "y_salinity_glorys_valid_mask"
                        if variable == "salinity"
                        else "y_glorys_valid_mask"
                    )
                    input_key = "x_salinity" if variable == "salinity" else "x"
                    input_valid_key = (
                        "x_salinity_valid_mask"
                        if variable == "salinity"
                        else "x_valid_mask"
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
                            glorys_valid_mask.to(
                                device=glorys.device, dtype=torch.bool
                            ),
                            glorys,
                            torch.full_like(glorys, float("nan")),
                        )
                    input_values = batch.get(input_key)
                    input_valid_mask = batch.get(input_valid_key)
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
                        for local_batch_idx, patch_idx in enumerate(chunk_patch_indices):
                            patch = rows.iloc[int(patch_idx)]
                            y0, x0 = int(patch.grid_y0), int(patch.grid_x0)
                            glorys_image = glorys[local_batch_idx]
                            # Restrict the reconstruction panel to the same valid-ocean
                            # support as GLORYS so land/nodata predictions are not visualized.
                            prediction_image = torch.where(
                                torch.isfinite(glorys_image),
                                predicted[local_batch_idx],
                                torch.full_like(
                                    predicted[local_batch_idx], float("nan")
                                ),
                            )
                            patch_holdouts = self.holdout_df.loc[
                                self.holdout_df["date"].eq(int(patch.date))
                                & self.holdout_df["grid_row"].between(
                                    y0, y0 + int(self.dataset.tile_size) - 1
                                )
                                & self.holdout_df["grid_col"].between(
                                    x0, x0 + int(self.dataset.tile_size) - 1
                                )
                            ][["grid_row", "grid_col"]].drop_duplicates()
                            self._latest_patch_images[str(variable)].append(
                                CandidatePatchImageData(
                                    variable=str(variable),
                                    patch_number=int(chunk_start + local_batch_idx),
                                    date=int(patch.date),
                                    input_values=(
                                        input_physical[local_batch_idx]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy()
                                    ),
                                    prediction=(
                                        prediction_image.detach().float().cpu().numpy()
                                    ),
                                    glorys=(
                                        glorys_image.detach().float().cpu().numpy()
                                    ),
                                    heldout_rows=(
                                        patch_holdouts["grid_row"].to_numpy(
                                            dtype=np.int64
                                        )
                                        - y0
                                    ),
                                    heldout_cols=(
                                        patch_holdouts["grid_col"].to_numpy(
                                            dtype=np.int64
                                        )
                                        - x0
                                    ),
                                )
                            )
                    assignments = self.profile_assignments.loc[
                        self.profile_assignments["eval_batch_index"].between(
                            chunk_start,
                            chunk_start + len(chunk_patch_indices) - 1,
                        )
                    ]
                    for profile_row, assignment in assignments.iterrows():
                        batch_idx = int(assignment["eval_batch_index"]) - chunk_start
                        row_idx = int(assignment["local_grid_row"])
                        col_idx = int(assignment["local_grid_col"])
                        results[str(variable)].append(
                            CandidateProfileResult(
                                variable=str(variable),
                                date=int(assignment["date"]),
                                latitude=float(assignment["lat"]),
                                longitude=float(assignment["lon"]),
                                profile_source_file=str(
                                    assignment["profile_source_file"]
                                ),
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
        return results

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset candidate results before Lightning starts the second val loader."""
        del trainer
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        self._epoch_results = {str(variable): [] for variable in fields}
        self._latest_patch_images = {str(variable): [] for variable in fields}
        self._candidate_loader_batches_seen = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions produced by the candidate validation dataloader."""
        del trainer, pl_module
        if int(dataloader_idx) != 1:
            return
        self._candidate_loader_batches_seen += 1
        if not isinstance(outputs, dict):
            return
        self._consume_validation_batch(
            batch,
            outputs,
            patch_start=int(batch_idx) * self.patch_batch_size,
        )

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

            results = self._epoch_results
            if self._candidate_loader_batches_seen == 0:
                # Preserve direct callback use in tests/tools that do not attach the
                # optional candidate dataloader to a LightningDataModule.
                results = self.evaluate(pl_module)
            payload: dict[str, Any] = {
                "en4_candidate_eval/selected_profile_count": int(
                    len(self.profile_assignments)
                ),
                "en4_candidate_eval/selected_location_count": int(
                    self.profile_assignments[["date", "grid_row", "grid_col"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "en4_candidate_eval/validated_batch_count": int(
                    self._candidate_loader_batches_seen
                ),
            }
            configured_val_limit = getattr(trainer, "limit_val_batches", None)
            if isinstance(configured_val_limit, (int, float)) and not isinstance(
                configured_val_limit, bool
            ):
                payload["en4_candidate_eval/limit_val_batches"] = configured_val_limit
            for count_name in (
                "eligible_profile_count",
                "eligible_location_count",
                "candidate_location_count",
                "candidate_covered_location_count",
                "candidate_uncovered_location_count",
                "candidate_covered_profile_count",
                "candidate_uncovered_profile_count",
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
                    payload[
                        f"en4_candidate_eval/{variable}_validated_profile_count"
                    ] = int(len(variable_results))
                    payload[
                        f"en4_candidate_eval/{variable}_validated_location_count"
                    ] = int(
                        len(
                            {
                                (result.date, result.latitude, result.longitude)
                                for result in variable_results
                            }
                        )
                    )
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
                        figure = _aggregate_profile_figure(
                            variable_results, depth_axis_m=self.depth_axis_m
                        )
                        figures.append(figure)
                        payload[f"en4_candidate_eval/{variable}_profiles"] = (
                            wandb.Image(figure)
                        )
                        example_figure = _profile_figure(
                            variable_results,
                            depth_axis_m=self.depth_axis_m,
                            max_profiles=(
                                5
                                if self.max_profiles_to_plot is None
                                else self.max_profiles_to_plot
                            ),
                        )
                        figures.append(example_figure)
                        payload[f"en4_candidate_eval/{variable}_profile_examples"] = (
                            wandb.Image(example_figure)
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
                            title=(
                                "Average temperature absolute error vs EN4 "
                                f"(n={len(variable_results)} held-out profiles)"
                            ),
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
