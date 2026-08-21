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
        holdout_df: pd.DataFrame,
        max_patches: int = 1,
        max_profiles_to_plot: int = 6,
        random_seed: int = 7,
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
        self.dataset = dataset
        self.selection_metadata = dict(holdout_df.attrs)
        self.holdout_df = holdout_df.reset_index(drop=True).copy()
        self.max_profiles_to_plot = max(1, int(max_profiles_to_plot))
        self.random_seed = int(random_seed)
        self.patch_indices, self.profile_assignments = self._select_eval_patches(
            max_patches=max_patches
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
        self._last_logged_global_step: int | None = None

    def _select_eval_patches(
        self, *, max_patches: int
    ) -> tuple[list[int], pd.DataFrame]:
        """Greedily choose deterministic patches covering the most holdout locations."""
        limit = max(1, int(max_patches))
        rows = self.dataset._rows.reset_index(drop=True)
        selected_date = int(self.holdout_df["date"].iloc[0])
        candidate_rows = rows.index[rows["date"].astype(np.int64) == selected_date]
        locations = {
            (int(row.grid_row), int(row.grid_col))
            for row in self.holdout_df.itertuples(index=False)
        }
        uncovered = set(locations)
        chosen: list[int] = []
        coverage_by_patch: dict[int, set[tuple[int, int]]] = {}
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
        while uncovered and len(chosen) < limit:
            ranked = sorted(
                (
                    (len(covered & uncovered), patch_idx)
                    for patch_idx, covered in coverage_by_patch.items()
                    if patch_idx not in chosen
                ),
                key=lambda item: (-item[0], item[1]),
            )
            if not ranked or ranked[0][0] <= 0:
                break
            patch_idx = int(ranked[0][1])
            chosen.append(patch_idx)
            uncovered -= coverage_by_patch[patch_idx]
        if not chosen:
            raise RuntimeError(
                "No validation patch covers the selected EN4 candidate locations."
            )

        assignments: list[dict[str, Any]] = []
        for row in self.holdout_df.to_dict(orient="records"):
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
        return chosen, pd.DataFrame.from_records(assignments)

    def _build_batch(self) -> dict[str, Any]:
        """Load the fixed evaluation patches through the normal validation dataset."""
        return default_collate([self.dataset[index] for index in self.patch_indices])

    @staticmethod
    def _denormalize_target(variable: str, tensor: torch.Tensor) -> torch.Tensor:
        """Convert one dense GLORYS validation target to physical units."""
        if variable == "salinity":
            return salinity_normalize(mode="denorm", tensor=tensor)
        return temperature_normalize(mode="denorm", tensor=tensor)

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
            if not torch.is_tensor(predicted) or target_key not in batch:
                continue
            glorys_target = batch.get(glorys_key)
            if glorys_target is None:
                # Direct-GLORYS and custom datasets retain the legacy y fallback.
                glorys_target = batch[target_key]
            glorys = self._denormalize_target(variable, glorys_target)
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
                experiment.log(payload)
            finally:
                for figure in figures:
                    plt.close(figure)
            self._last_logged_global_step = global_step
        except Exception as exc:
            warnings.warn(
                f"EN4 candidate validation logging failed: {exc}", stacklevel=2
            )
