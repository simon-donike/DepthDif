"""Bounded, isolated validation against archive-screened EN4 profile holdouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import default_collate

from depth_recon.utils.en4_candidate_validation import (
    CandidateProfileResult,
    _profile_figure,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.reconstruction_validation import _evaluation_rng
from depth_recon.utils.validation_denoise import log_wandb_depth_errors

AUDIT_STATUS = "no_spatiotemporal_candidate"
PREFIX = "val/en4_archive_holdout"


class EN4ArchiveHoldoutValidation(pl.Callback):
    """Evaluate up to 32 fixed patches at the first real validation check each epoch."""

    def __init__(
        self,
        *,
        dataset: Any,
        candidate_df: pd.DataFrame,
        audit_path: str | Path,
        output_dir: str | Path,
        sample_count: int = 32,
        batch_size: int = 4,
        seed: int = 7,
        holdout_fraction: float = 0.2,
        min_input_profiles: int = 8,
        max_profiles_to_plot: int = 4,
    ) -> None:
        """Keep audit provenance and defer patch loading until real validation."""
        super().__init__()
        if min(sample_count, batch_size, min_input_profiles) < 1 or seed < 0:
            raise ValueError(
                "Holdout sample_count, batch_size and min_input_profiles must be positive; seed must be nonnegative."
            )
        if not 0 < holdout_fraction < 1 or max_profiles_to_plot < 0:
            raise ValueError(
                "Holdout fraction must be in (0, 1), and plot count nonnegative."
            )
        if candidate_df.attrs.get("audit_status_filter") != AUDIT_STATUS:
            raise ValueError(
                "Archive holdouts require the historical no-candidate audit filter."
            )
        if not candidate_df.attrs.get("quality_filter_enabled"):
            raise ValueError("Archive holdouts require QC-filtered profiles.")
        if candidate_df.empty:
            raise ValueError("No historical archive candidates survive the selection.")
        if getattr(dataset, "heldout_argo_location_keys", set()):
            raise ValueError(
                "Archive holdouts require an unmodified validation dataset."
            )
        self.dataset = dataset
        self.candidates = candidate_df.drop_duplicates(
            ["profile_source_file", "source_profile_idx"]
        ).reset_index(drop=True)
        self.audit_path = Path(audit_path)
        with self.audit_path.open("rb") as stream:
            self.audit_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.output_dir = Path(output_dir)
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.holdout_fraction = float(holdout_fraction)
        self.min_input_profiles = int(min_input_profiles)
        self.max_profiles_to_plot = int(max_profiles_to_plot)
        self.patch_indices: list[int] = []
        self.holdouts: set[tuple[int, int, int]] = set()
        self.assignments = pd.DataFrame()
        self.references: dict[str, np.ndarray] = {}
        self.depth_axis_m = np.asarray(dataset.argo_store.depth_axis_m)
        self._last_epoch: int | None = None
        self.selection: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        """Preserve once-per-epoch cadence when resuming mid-epoch."""
        return {"last_epoch": self._last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the last successfully evaluated epoch."""
        self._last_epoch = state_dict.get("last_epoch")

    def _select_patches(self, fields: tuple[str, ...]) -> None:
        """Draw patches across dates, preserving input support under global holdouts."""
        rng = np.random.default_rng(self.seed)
        rows = self.dataset._rows.reset_index(drop=True)
        usable = np.zeros(len(self.candidates), dtype=bool)
        for field in fields:
            usable |= self.candidates[f"{field}_valid_depth_count"].to_numpy() > 0
        candidates = self.candidates.loc[usable]
        by_date = {int(date): group for date, group in candidates.groupby("date")}
        dates = rng.permutation(sorted(set(rows["date"]) & set(by_date))).tolist()
        pools = {
            date: iter(rng.permutation(rows.index[rows["date"].eq(date)]).tolist())
            for date in dates
        }
        observed_by_patch: dict[int, set[tuple[int, int, int]]] = {}
        assignments = []
        tile = int(self.dataset.tile_size)
        while dates and len(self.patch_indices) < self.sample_count:
            for date in list(dates):
                # Select at most one eligible patch per date per round, so a
                # dense cruise cannot consume the entire annual patch budget.
                for index in pools[date]:
                    row = rows.iloc[index]
                    y0, x0 = int(row.grid_y0), int(row.grid_x0)
                    covered = by_date[date].loc[
                        by_date[date]["grid_row"].between(y0, y0 + tile - 1)
                        & by_date[date]["grid_col"].between(x0, x0 + tile - 1)
                    ]
                    if covered.empty:
                        continue
                    sample = self.dataset[index]
                    observed_mask = torch.zeros((tile, tile), dtype=torch.bool)
                    for field in fields:
                        key = (
                            "x_salinity_valid_mask"
                            if field == "salinity"
                            else "x_valid_mask"
                        )
                        observed_mask |= sample[key].bool().any(dim=0)
                    observed = {
                        (date, y0 + int(y), x0 + int(x))
                        for y, x in torch.nonzero(observed_mask).tolist()
                    }
                    locations = sorted(
                        {
                            (date, int(item.grid_row), int(item.grid_col))
                            for item in covered.itertuples(index=False)
                        }
                        & observed - self.holdouts
                    )
                    available = len(observed - self.holdouts) - self.min_input_profiles
                    count = min(
                        max(1, round(len(locations) * self.holdout_fraction)), available
                    )
                    if not locations or count < 1:
                        continue
                    new_holdouts = {
                        locations[position]
                        for position in rng.choice(
                            len(locations), size=count, replace=False
                        )
                    }
                    proposed = self.holdouts | new_holdouts
                    if any(
                        len(support - proposed) < self.min_input_profiles
                        for support in observed_by_patch.values()
                    ):
                        continue
                    self.patch_indices.append(int(index))
                    observed_by_patch[index] = observed
                    self.holdouts = proposed
                    for item in covered.to_dict(orient="records"):
                        if (
                            date,
                            int(item["grid_row"]),
                            int(item["grid_col"]),
                        ) in new_holdouts:
                            assignments.append(
                                {
                                    **item,
                                    "patch_index": int(index),
                                    "local_row": int(item["grid_row"]) - y0,
                                    "local_col": int(item["grid_col"]) - x0,
                                }
                            )
                    break
                else:
                    dates.remove(date)
                if len(self.patch_indices) == self.sample_count:
                    break
        if not assignments:
            raise ValueError(
                "No archive-holdout patches retain the configured minimum input locations."
            )
        self.assignments = pd.DataFrame.from_records(assignments)
        indices = self.assignments["profile_index"].to_numpy(dtype=np.int64)
        store = self.dataset.argo_store
        for field in fields:
            values = getattr(store, f"load_{field}_profiles")(indices)
            # Enforce the same QC policy even when the training store is permissive.
            quality = store._quality_mask_for_variable(
                "psal" if field == "salinity" else "temp", indices=indices
            )
            self.references[field] = np.where(quality, values, np.nan)
        self.selection = {
            "audit_path": str(self.audit_path.resolve()),
            "audit_sha256": self.audit_sha256,
            "audit_status": AUDIT_STATUS,
            "evidence": "No candidate in the supplied historical archive within 24 hours / 25 km; not confirmed non-assimilation.",
            "validation_year": self.candidates.attrs.get("validation_year"),
            "quality_filter_enabled": True,
            "accepted_qc_flags": list(store.accepted_qc_flags),
            "output_fields": list(fields),
            "seed": self.seed,
            "holdout_fraction": self.holdout_fraction,
            "requested_patch_count": self.sample_count,
            "batch_size": self.batch_size,
            "min_input_locations": self.min_input_profiles,
            "retained_input_locations_min": min(
                len(v - self.holdouts) for v in observed_by_patch.values()
            ),
            "depth_axis_m": self.depth_axis_m.tolist(),
            "patches": [
                {
                    "index": i,
                    **rows.loc[i, ["date", "grid_y0", "grid_x0"]].astype(int).to_dict(),
                }
                for i in self.patch_indices
            ],
            "heldout_locations": [list(key) for key in sorted(self.holdouts)],
            "profiles": self.assignments.to_dict(orient="records"),
        }

    def _masked_sample(self, index: int) -> dict[str, Any]:
        """Mask all fields and depths on tensor copies, including overlapping patches."""
        sample = dict(self.dataset[index])
        row = self.dataset._rows.iloc[index]
        for key in ("x", "x_salinity"):
            if key not in sample:
                continue
            values = sample[key].clone()
            mask = sample[f"{key}_valid_mask"].clone()
            for date, y, x in self.holdouts:
                local_y, local_x = y - int(row.grid_y0), x - int(row.grid_x0)
                if (
                    date == int(row.date)
                    and 0 <= local_y < values.shape[-2]
                    and 0 <= local_x < values.shape[-1]
                ):
                    values[:, local_y, local_x] = 0
                    mask[:, local_y, local_x] = False
            sample[key] = values
            sample[f"{key}_valid_mask"] = mask
            if f"{key}_valid_mask_1d" in sample:
                sample[f"{key}_valid_mask_1d"] = mask.bool().any(dim=0, keepdim=True)
        return sample

    @staticmethod
    def _statistics(
        predicted: np.ndarray, glorys: np.ndarray, observed: np.ndarray
    ) -> np.ndarray:
        """Pool paired errors without dropping failed predictions from scored support."""
        support = np.isfinite(observed) & np.isfinite(glorys)
        output = []
        for values in (predicted, glorys):
            with np.errstate(invalid="ignore"):
                error = np.where(
                    np.isfinite(values), values.astype(np.float64) - observed, np.inf
                )
            error = np.where(support, error, 0.0)
            output.append(
                np.stack(
                    (np.square(error).sum(0), np.abs(error).sum(0), support.sum(0))
                )
            )
        return np.stack(output)

    def _consume_batch(
        self, batch, prediction, indices, fields, statistics, counts, examples
    ):
        """Match predictions to immutable patch indices and accumulate raw-profile errors."""
        for local_index, index in enumerate(indices):
            assigned = self.assignments.loc[self.assignments["patch_index"].eq(index)]
            for field in fields:
                normalize = (
                    salinity_normalize if field == "salinity" else temperature_normalize
                )
                key = "y_salinity" if field == "salinity" else "y"
                glorys_key = f"{key}_glorys" if f"{key}_glorys" in batch else key
                glorys = normalize(
                    mode="denorm", tensor=batch[glorys_key][local_index].float()
                )
                domain = batch[f"{glorys_key}_valid_mask"][local_index].bool()
                if "land_mask" in batch:
                    domain = domain & batch["land_mask"][local_index].bool()
                glorys = glorys.masked_fill(~domain, float("nan"))
                predicted = prediction.get(f"y_hat_{field}_denorm")
                if predicted is None and len(fields) == 1:
                    predicted = prediction.get("y_hat_denorm")
                if (
                    not torch.is_tensor(predicted)
                    or predicted[local_index].shape != glorys.shape
                ):
                    raise ValueError(
                        f"Missing or mismatched {field} holdout prediction."
                    )
                yy = assigned["local_row"].to_numpy()
                xx = assigned["local_col"].to_numpy()
                pred_values = (
                    predicted[local_index, :, yy, xx].T.detach().float().cpu().numpy()
                )
                glorys_values = glorys[:, yy, xx].T.detach().float().cpu().numpy()
                observed = self.references[field][assigned.index.to_numpy()]
                stats = self._statistics(pred_values, glorys_values, observed)
                statistics[field] += torch.as_tensor(
                    stats, device=statistics[field].device
                )
                supported = (np.isfinite(observed) & np.isfinite(glorys_values)).any(
                    axis=1
                )
                counts[field][0] += int(supported.sum())
                counts[field][1] += len(
                    assigned.loc[
                        supported, ["date", "grid_row", "grid_col"]
                    ].drop_duplicates()
                )
                for position, (profile_index, item) in enumerate(assigned.iterrows()):
                    if (
                        not supported[position]
                        or len(examples[field]) >= self.max_profiles_to_plot
                    ):
                        continue
                    result = CandidateProfileResult(
                        variable=field,
                        date=int(item.date),
                        latitude=float(item.lat),
                        longitude=float(item.lon),
                        profile_source_file=str(item.profile_source_file),
                        source_profile_idx=int(item.source_profile_idx),
                        prediction=pred_values[position],
                        glorys=glorys_values[position],
                        en4=observed[position],
                    )
                    examples[field].append((int(profile_index), result))

    def _log_figures(self, trainer, statistics, examples):
        """Log error curves, bounded profile examples and selection provenance on rank zero."""
        logger = trainer.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        import wandb

        for field, stats in statistics.items():
            unit = "PSU" if field == "salinity" else "deg C"
            log_wandb_depth_errors(
                logger=logger,
                statistics=dict(zip(("Prediction", "GLORYS"), stats)),
                depth_axis_m=self.depth_axis_m,
                prefix=PREFIX,
                image_key=f"{field}_absolute_error_by_depth",
                error_label=f"Mean absolute error vs held-out EN4 ({unit})",
                title=f"Held-out EN4 — no historical archive candidate: {field}",
            )
            selected = sorted(examples[field], key=lambda item: item[0])[
                : self.max_profiles_to_plot
            ]
            if selected:
                figure = _profile_figure(
                    [item[1] for item in selected],
                    depth_axis_m=self.depth_axis_m,
                    max_profiles=self.max_profiles_to_plot,
                )
                try:
                    figure.suptitle(
                        f"Held-out EN4 — no historical archive candidate: {field}"
                    )
                    experiment.log({f"{PREFIX}/{field}_profiles": wandb.Image(figure)})
                finally:
                    plt.close(figure)
        columns = [
            "date",
            "lat",
            "lon",
            "profile_source_file",
            "source_profile_idx",
            "patch_index",
        ]
        experiment.log(
            {
                f"{PREFIX}/selection": wandb.Table(
                    columns=columns, data=self.assignments[columns].values.tolist()
                ),
                f"{PREFIX}/audit_sha256": self.audit_sha256,
                f"{PREFIX}/audit_status": AUDIT_STATUS,
                f"{PREFIX}/seed": self.seed,
            }
        )

    def _record_selection(self, trainer) -> None:
        """Check identical rank selections and preserve the original run manifest."""
        content = json.dumps(self.selection, indent=2, sort_keys=True) + "\n"
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed:
            digests = [None] * trainer.world_size
            torch.distributed.all_gather_object(
                digests, hashlib.sha256(content.encode()).hexdigest()
            )
            if len(set(digests)) != 1:
                raise ValueError("EN4 holdout selection differs between ranks.")
        error = [None]
        if trainer.is_global_zero:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / "en4_archive_holdout_selection.json"
                if path.exists() and path.read_text() != content:
                    raise ValueError(
                        "Existing EN4 holdout selection differs; use a new run directory."
                    )
                path.write_text(content)
            except (OSError, ValueError) as exc:
                error[0] = str(exc)
        # A rank-zero provenance failure must reach peers before they enter metric reductions.
        if distributed:
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise ValueError(error[0])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """Reconstruct the bounded subset once per epoch, using active raw/EMA weights."""
        epoch = int(trainer.current_epoch)
        if trainer.sanity_checking or self._last_epoch == epoch:
            return
        fields = tuple(getattr(pl_module, "output_fields", ("temperature",)))
        if not self.patch_indices:
            with _evaluation_rng(pl_module.device, self.seed):
                self._select_patches(fields)
            self._record_selection(trainer)
        statistics = {
            field: torch.zeros(
                (2, 3, len(self.depth_axis_m)),
                dtype=torch.float64,
                device=pl_module.device,
            )
            for field in fields
        }
        counts = {
            field: torch.zeros(2, dtype=torch.float64, device=pl_module.device)
            for field in fields
        }
        examples = {field: [] for field in fields}
        for batch_index, offset in enumerate(
            range(0, len(self.patch_indices), self.batch_size)
        ):
            # Explicit global batch indices prevent DDP sampler padding or mismatched profiles.
            if batch_index % trainer.world_size != trainer.global_rank:
                continue
            indices = self.patch_indices[offset : offset + self.batch_size]
            with _evaluation_rng(pl_module.device, self.seed + batch_index):
                batch = default_collate([self._masked_sample(i) for i in indices])
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                batch["return_intermediates"] = False
                with trainer.precision_plugin.forward_context():
                    prediction = pl_module.predict_step(batch, batch_idx=batch_index)
                self._consume_batch(
                    batch, prediction, indices, fields, statistics, counts, examples
                )
                del prediction, batch
        metrics = {
            f"{PREFIX}/patch_count": float(len(self.patch_indices)),
            f"{PREFIX}/selected_profile_count": float(len(self.assignments)),
            f"{PREFIX}/selected_location_count": float(len(self.holdouts)),
            f"{PREFIX}/date_count": float(self.assignments["date"].nunique()),
            f"{PREFIX}/ema_weights": float(
                any(
                    bool(getattr(callback, "weights_are_applied", False))
                    for callback in getattr(trainer, "callbacks", [])
                )
            ),
        }
        for field in fields:
            statistics[field] = trainer.strategy.reduce(
                statistics[field], reduce_op="sum"
            )
            counts[field] = trainer.strategy.reduce(counts[field], reduce_op="sum")
            for label, stats in zip(("prediction", "glorys"), statistics[field]):
                squared, absolute, count = stats
                supported = count > 0
                prefix = f"{PREFIX}/{field}_{label}"
                metrics[f"{prefix}_mae"] = (
                    absolute.sum() / count.sum().clamp_min(1)
                    if supported.any()
                    else torch.tensor(float("nan"), device=stats.device)
                )
                metrics[f"{prefix}_rmse"] = (
                    torch.sqrt(squared.sum() / count.sum().clamp_min(1))
                    if supported.any()
                    else torch.tensor(float("nan"), device=stats.device)
                )
                metrics[f"{prefix}_equal_depth_mae"] = (
                    absolute[supported] / count[supported]
                ).mean()
                metrics[f"{prefix}_equal_depth_rmse"] = torch.sqrt(
                    squared[supported] / count[supported]
                ).mean()
            reference_rmse = metrics[f"{PREFIX}/{field}_glorys_rmse"]
            metrics[f"{PREFIX}/{field}_skill_vs_glorys"] = (
                1 - metrics[f"{PREFIX}/{field}_prediction_rmse"] / reference_rmse
                if reference_rmse > 0
                else torch.tensor(float("nan"), device=pl_module.device)
            )
            metrics[f"{PREFIX}/{field}_valid_values"] = statistics[field][0, 2].sum()
            metrics[f"{PREFIX}/{field}_validated_profile_count"] = counts[field][0]
            metrics[f"{PREFIX}/{field}_validated_location_count"] = counts[field][1]
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, sync_dist=False, batch_size=1
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Only bounded examples cross ranks; dense reconstructions are discarded per batch.
            gathered = [None] * trainer.world_size
            torch.distributed.all_gather_object(gathered, examples)
            examples = {
                field: [
                    item for rank_examples in gathered for item in rank_examples[field]
                ]
                for field in fields
            }
        if trainer.is_global_zero:
            self._log_figures(trainer, statistics, examples)
        self._last_epoch = epoch
