"""Small, provenance-preserving inference adapter for validation cohorts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from depth_recon.configs.config_resolver_pixel import load_pixel_training_config
from depth_recon.inference.core import (
    build_datamodule,
    build_model,
    load_checkpoint_weights,
    model_requires_checkpoint,
    to_device,
)
from depth_recon.utils.reconstruction_validation import _evaluation_rng
from val.conditioning import (
    local_oi_predictions,
    nearest_context_distance_km,
    nested_selections,
    selection_table,
)


def _sha256(path: Path) -> str:
    """Return a file digest without loading the complete file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_file(path: str | Path, label: str) -> Path:
    """Resolve one required input before constructing a GPU model."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic metadata with JSON-safe scalar conversion."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def _prediction_points(
    prediction: torch.Tensor, points: pd.DataFrame, *, batch_offset: int
) -> pd.DataFrame:
    """Extract physical predictions at point rows from a model output tensor."""
    required = {"point_id", "batch_index", "row", "col"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("Point assignments are missing: " + ", ".join(missing))
    values = prediction.detach().float().cpu().numpy()
    rows = []
    for item in points.to_dict(orient="records"):
        batch_index = int(item["batch_index"]) - int(batch_offset)
        if {"depth_index_lower", "depth_index_upper", "depth_weight"}.issubset(item):
            lower = int(item["depth_index_lower"])
            upper = int(item["depth_index_upper"])
            weight = float(item["depth_weight"])
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Depth interpolation weight is outside [0, 1]: {item}"
                )
        elif "depth_index" in item:
            lower = upper = int(item["depth_index"])
            weight = 0.0
        else:
            raise ValueError(
                "Point assignments require depth_index or depth_index_lower/depth_index_upper/depth_weight"
            )
        row = int(item["row"])
        col = int(item["col"])
        if not (0 <= batch_index < values.shape[0]):
            raise ValueError(f"Point batch_index is outside the batch: {item}")
        if not (
            0 <= lower < values.shape[1]
            and 0 <= upper < values.shape[1]
            and 0 <= row < values.shape[2]
            and 0 <= col < values.shape[3]
        ):
            raise ValueError(f"Point tensor location is outside the prediction: {item}")
        result = dict(item)
        result["prediction"] = float(
            (1.0 - weight) * values[batch_index, lower, row, col]
            + weight * values[batch_index, upper, row, col]
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _physical_output(result: Mapping[str, Any], scenario: str) -> torch.Tensor:
    """Select the requested physical field from a prediction result."""
    field = str(scenario)
    if field not in {"temperature", "salinity"}:
        raise ValueError(
            f"Point variable must be temperature or salinity, got {scenario!r}"
        )
    prediction = result.get(f"y_hat_{field}_denorm")
    if prediction is None and scenario not in {"temperature", "salinity"}:
        prediction = result.get("y_hat_denorm")
    if prediction is None:
        raise ValueError(f"Model prediction has no physical {field} output")
    return prediction


def run_model(
    config: dict[str, Any],
    output_dir: Path,
    batches: Iterable[tuple[dict[str, Any], pd.DataFrame]],
    *,
    model_factory: Any | None = None,
    bundle: Any | None = None,
    dataset: Any | None = None,
) -> dict[str, Any]:
    """Run repeated physical predictions for already-selected cohort batches.

    ``batches`` yields a model batch and a point-assignment table.  Assignment tables
    use zero-based ``batch_index``, ``row`` and ``col`` columns plus either
    ``depth_index`` or ``depth_index_lower``, ``depth_index_upper`` and
    ``depth_weight``.  The latter interpolates model levels to raw observation
    depths. Tables must contain a globally unique ``point_id``. Cohort construction
    and leakage decisions therefore remain outside this model-only adapter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing validation output: {output_dir}"
        )
    model_cfg = config.get("model", {})
    config_path = _require_file(model_cfg.get("config", ""), "model config")
    checkpoint = _require_file(model_cfg.get("checkpoint", ""), "checkpoint")
    expected_digest = model_cfg.get("checkpoint_sha256")
    actual_digest = _sha256(checkpoint)
    if expected_digest and actual_digest != expected_digest:
        raise ValueError("Checkpoint SHA256 does not match config")
    weight_variant = str(model_cfg.get("weights", "ema")).lower()
    if weight_variant not in {"ema", "standard"}:
        raise ValueError("model.weights must be 'ema' or 'standard'.")
    sampling = config.get("sampling", {})
    members = int(sampling.get("members", 20))
    if members < 2:
        raise ValueError("sampling.members must be at least 2 for an ensemble")
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    runtime = output_dir / "effective"
    overrides = [
        f"training.training.validation_sampling.sampler={sampling.get('sampler', 'ddpm')}",
    ]
    if sampling.get("steps") is not None:
        overrides.append(
            f"training.training.validation_sampling.ddim_num_timesteps={int(sampling['steps'])}"
        )
    if bundle is None:
        bundle = load_pixel_training_config(
            config_path_value=config_path,
            runtime_config_dir=runtime,
            overrides=overrides,
            write_snapshots=True,
        )
    datamodule = build_datamodule(
        dataset or torch.utils.data.TensorDataset(torch.zeros(1)),
        bundle.data_cfg,
        bundle.training_cfg,
    )
    model = (model_factory or build_model)(
        bundle.effective_model_config_path,
        bundle.effective_data_config_path,
        bundle.effective_training_config_path,
        bundle.model_cfg,
        datamodule,
    )
    loaded = load_checkpoint_weights(
        model, checkpoint, strict=True, prefer_ema=weight_variant == "ema"
    )
    if loaded != weight_variant:
        raise ValueError(f"Requested {weight_variant} weights but loaded {loaded}")
    model.to(device).eval()
    configured_variables = sampling.get("variables", [bundle.scenario])
    if configured_variables == "joint":
        configured_variables = bundle.model_cfg.get("model", {}).get(
            "output_fields", []
        )
    variables = [str(value) for value in configured_variables]
    if not variables or any(
        value not in {"temperature", "salinity"} for value in variables
    ):
        raise ValueError("sampling.variables must contain temperature and/or salinity")
    model_fields = set(bundle.model_cfg.get("model", {}).get("output_fields", []))
    if not model_fields:
        raise ValueError(
            "Resolved model config has no model.output_fields capability declaration"
        )
    unavailable = sorted(set(variables) - model_fields)
    if unavailable:
        raise ValueError(
            f"Checkpoint/model does not provide requested fields: {', '.join(unavailable)}"
        )
    collected: list[pd.DataFrame] = []
    all_assignments: list[pd.DataFrame] = []
    point_seen: set[str] = set()
    conditioning_count = 0
    batch_records: list[dict[str, Any]] = []
    baseline_models: dict[str, tuple[Any, Any, str]] = {}
    baseline_status: dict[str, Any] = {}
    baseline_values: dict[str, dict[str, float]] = {}
    conditioning_cfg = (
        config.get("paper", {}).get("conditioning", {})
        if isinstance(config.get("paper", {}), Mapping)
        else {}
    )
    conditioning_enabled = bool(conditioning_cfg.get("enabled", False))
    conditioning_budgets = sorted(
        {int(value) for value in conditioning_cfg.get("budgets", [0, 1, 2, 4, 8])}
    )
    if conditioning_enabled and any(value < 0 for value in conditioning_budgets):
        raise ValueError("paper.conditioning.budgets must be non-negative")
    conditioning_values: dict[int, list[np.ndarray]] = {
        budget: [] for budget in conditioning_budgets
    }
    conditioning_records: list[dict[str, Any]] = []
    member_timing_records: list[dict[str, Any]] = []
    conditioning_covariates: list[dict[str, Any]] = []
    conditioning_oi_values: dict[str, list[np.ndarray]] = {}
    oi_errors: list[str] = []
    oi_cfg = conditioning_cfg.get("local_oi", {})
    oi_noise = oi_cfg.get("noise", {"temperature": 0.1, "salinity": 0.02})
    oi_depth_axis = oi_cfg.get("depth_axis_m")
    if oi_depth_axis is None and dataset is not None:
        oi_depth_axis = getattr(dataset, "depth_axis_m", None)

    def _oi_or_none(mask: torch.Tensor, assignments: pd.DataFrame, background: str):
        """Capture OI only when the batch exposes all physical fields."""
        try:
            return local_oi_predictions(
                batch,
                assignments,
                mask,
                background=background,
                variables=variables,
                depth_axis_m=oi_depth_axis,
                horizontal_scale_km=float(oi_cfg.get("horizontal_scale_km", 150.0)),
                vertical_scale_m=float(oi_cfg.get("vertical_scale_m", 50.0)),
                grid_spacing_deg=float(oi_cfg.get("grid_spacing_deg", 0.1)),
                noise_by_variable=oi_noise,
                max_profiles=int(oi_cfg.get("max_profiles", 32)),
            )
        except ValueError as error:
            oi_errors.append(str(error))
            return None

    for baseline in config.get("baselines", []):
        name = str(baseline.get("name", "baseline"))
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"Invalid baseline name: {name!r}")
        if name in baseline_models:
            raise ValueError(f"Duplicate baseline name: {name!r}")
        baseline_config_path = _require_file(
            baseline.get("config", ""), f"baseline {name} config"
        )
        baseline_bundle = load_pixel_training_config(
            config_path_value=baseline_config_path,
            runtime_config_dir=output_dir / "effective" / f"baseline_{name}",
            write_snapshots=True,
        )
        baseline_checkpoint = baseline.get("checkpoint")
        baseline_digest = None
        if baseline_checkpoint:
            baseline_checkpoint_path = _require_file(
                baseline_checkpoint, f"baseline {name} checkpoint"
            )
            baseline_digest = _sha256(baseline_checkpoint_path)
            if (
                baseline.get("checkpoint_sha256")
                and baseline_digest != baseline["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"Baseline {name} checkpoint SHA256 does not match config"
                )
        baseline_datamodule = build_datamodule(
            dataset or torch.utils.data.TensorDataset(torch.zeros(1)),
            baseline_bundle.data_cfg,
            baseline_bundle.training_cfg,
        )
        baseline_model = (model_factory or build_model)(
            baseline_bundle.effective_model_config_path,
            baseline_bundle.effective_data_config_path,
            baseline_bundle.effective_training_config_path,
            baseline_bundle.model_cfg,
            baseline_datamodule,
        )
        requested_weights = str(baseline.get("weights", "standard"))
        if (
            model_requires_checkpoint(baseline_bundle.model_cfg)
            and not baseline_checkpoint
        ):
            raise ValueError(f"Learned baseline {name} requires a checkpoint")
        if baseline_checkpoint:
            loaded_baseline = load_checkpoint_weights(
                baseline_model,
                baseline_checkpoint_path,
                strict=True,
                prefer_ema=requested_weights == "ema",
            )
            if loaded_baseline != requested_weights:
                raise ValueError(
                    f"Baseline {name} loaded {loaded_baseline}, not requested weights"
                )
        else:
            loaded_baseline = "checkpoint_free"
        baseline_model.to(device).eval()
        baseline_models[name] = (baseline_model, baseline_bundle, loaded_baseline)
        baseline_values[name] = {}
        baseline_status[name] = {
            "config": str(baseline_config_path),
            "config_sha256": _sha256(baseline_config_path),
            "checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
            "checkpoint_sha256": baseline_digest,
            "weights": loaded_baseline,
        }
    seed = int(sampling.get("seed", 20261103))
    for batch_number, (batch, assignments) in enumerate(batches):
        assignments = assignments.copy()
        if assignments.empty:
            continue
        if assignments["point_id"].duplicated().any():
            raise ValueError("A point is assigned more than once within a batch")
        overlap = point_seen.intersection(assignments["point_id"].astype(str))
        if overlap:
            raise ValueError(
                "A raw profile point is assigned across overlapping patches"
            )
        point_seen.update(assignments["point_id"].astype(str))
        if len(variables) > 1 and "variable" not in assignments:
            raise ValueError("Joint validation assignments require a variable column")
        if "variable" not in assignments:
            assignments["variable"] = variables[0]
        if set(assignments["variable"].astype(str)) - set(variables):
            raise ValueError(
                "Point assignment contains a variable outside sampling.variables"
            )
        location_support = None
        for key in ("x_valid_mask", "x_salinity_valid_mask"):
            if key in batch and torch.is_tensor(batch[key]):
                support = batch[key].bool().any(dim=1)
                location_support = (
                    support if location_support is None else location_support | support
                )
        if location_support is not None:
            conditioning_count += int(location_support.sum().item())
        all_assignments.append(assignments)
        batch_hash = _batch_fingerprint(batch)
        batch_seeds = []
        member_values = []
        member_timings = []
        for member in range(members):
            member_seed = seed + batch_number * 100_000 + member
            batch_seeds.append(member_seed)
            batch_fingerprint = _batch_fingerprint(batch)
            member_started = time.monotonic()
            with _evaluation_rng(device, member_seed), torch.inference_mode():
                result = model.predict_step(
                    to_device(_clone_batch(batch), device), batch_idx=batch_number
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            member_timings.append(time.monotonic() - member_started)
            member_timing_records.append(
                {
                    "budget": "full",
                    "batch": batch_number,
                    "member": member,
                    "elapsed_seconds": member_timings[-1],
                }
            )
            if _batch_fingerprint(batch) != batch_fingerprint:
                raise RuntimeError("Model mutated the cohort batch during prediction")
            values = np.full(len(assignments), np.nan, dtype=np.float32)
            for variable in variables:
                selected = assignments["variable"].astype(str).eq(variable)
                if selected.any():
                    extracted = _prediction_points(
                        _physical_output(result, variable),
                        assignments.loc[selected],
                        batch_offset=0,
                    )
                    values[np.flatnonzero(selected.to_numpy())] = extracted[
                        "prediction"
                    ].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite physical prediction at a selected point")
            member_values.append(values)
        batch_records.append(
            {
                "batch": batch_number,
                "input_sha256": batch_hash,
                "member_seeds": batch_seeds,
                "member_elapsed_seconds": member_timings,
            }
        )
        selections = (
            nested_selections(
                batch,
                assignments,
                conditioning_budgets,
                seed=seed + batch_number * 100_000,
            )
            if conditioning_enabled
            else []
        )
        if conditioning_enabled and selections:
            available_mask = selections[0].available_mask
            for background in ("glorys", "climatology"):
                key = f"baseline_oi_{background}_full"
                oi = _oi_or_none(available_mask, assignments, background)
                if oi is not None:
                    conditioning_oi_values.setdefault(key, []).append(oi[:, None])
            for point in assignments.to_dict(orient="records"):
                latitude = pd.to_numeric(point.get("latitude"), errors="coerce")
                grid_distance = (
                    nearest_context_distance_km(
                        available_mask,
                        int(point["batch_index"]),
                        int(point["row"]),
                        int(point["col"]),
                        float(latitude),
                    )
                    if pd.notna(latitude)
                    else float("nan")
                )
                conditioning_covariates.append(
                    {
                        "point_id": str(point["point_id"]),
                        "batch": batch_number,
                        "budget": "full",
                        "available_profile_locations": int(
                            available_mask[int(point["batch_index"])].sum()
                        ),
                        "selected_profile_locations": int(
                            available_mask[int(point["batch_index"])].sum()
                        ),
                        "nearest_input_distance_km": grid_distance,
                        "glorys_context": point.get("glorys", np.nan),
                        "climatology_context": point.get("climatology", np.nan),
                        "distance_status": (
                            "available_grid_estimate"
                            if pd.notna(latitude)
                            else "blocked_missing_point_latitude"
                        ),
                    }
                )
        for name, (baseline_model, baseline_bundle, _) in baseline_models.items():
            with _evaluation_rng(device, seed + batch_number), torch.inference_mode():
                baseline_result = baseline_model.predict_step(
                    to_device(_clone_batch(batch), device), batch_idx=batch_number
                )
            baseline_points = np.full(len(assignments), np.nan, dtype=np.float32)
            for variable in variables:
                selected = assignments["variable"].eq(variable)
                if selected.any():
                    baseline_points[np.flatnonzero(selected.to_numpy())] = (
                        _prediction_points(
                            _physical_output(baseline_result, variable),
                            assignments.loc[selected],
                            batch_offset=0,
                        )["prediction"].to_numpy(float)
                    )
            if not np.isfinite(baseline_points).all():
                raise ValueError(
                    f"Baseline {name} has invalid predictions on common support"
                )
            baseline_values[name].update(
                zip(assignments["point_id"].astype(str), baseline_points)
            )
        points_result = assignments.copy()
        points_result["prediction_mean"] = np.mean(member_values, axis=0)
        points_result["member_offset"] = len(collected)
        collected.append(points_result)
        if conditioning_enabled:
            for selection in selections:
                started_budget = time.monotonic()
                budget_values = []
                budget_member_timings = []
                # The member seed is intentionally identical to the full run;
                # only sparse profile conditioning changes between budgets.
                for member in range(members):
                    member_seed = seed + batch_number * 100_000 + member
                    member_started = time.monotonic()
                    with _evaluation_rng(device, member_seed), torch.inference_mode():
                        budget_result = model.predict_step(
                            to_device(_clone_batch(selection.batch), device),
                            batch_idx=batch_number,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    budget_member_timings.append(time.monotonic() - member_started)
                    member_timing_records.append(
                        {
                            "budget": selection.budget,
                            "batch": batch_number,
                            "member": member,
                            "elapsed_seconds": budget_member_timings[-1],
                        }
                    )
                    values = np.full(len(assignments), np.nan, dtype=np.float32)
                    for variable in variables:
                        selected_points = (
                            assignments["variable"].astype(str).eq(variable)
                        )
                        if selected_points.any():
                            extracted = _prediction_points(
                                _physical_output(budget_result, variable),
                                assignments.loc[selected_points],
                                batch_offset=0,
                            )
                            values[np.flatnonzero(selected_points.to_numpy())] = (
                                extracted["prediction"].to_numpy()
                            )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"Nonfinite conditioning-ablation prediction at budget {selection.budget}"
                        )
                    budget_values.append(values)
                conditioning_values[selection.budget].append(
                    np.asarray(budget_values, dtype=np.float32).T
                )
                for background in ("glorys", "climatology"):
                    key = f"baseline_oi_{background}_budget_{selection.budget}"
                    oi = _oi_or_none(selection.selected_mask, assignments, background)
                    if oi is not None:
                        conditioning_oi_values.setdefault(key, []).append(oi[:, None])
                conditioning_records.append(
                    {
                        "batch": batch_number,
                        "budget": selection.budget,
                        "selected_profiles": selection.selected,
                        "available_profiles": selection.available,
                        "protected_target_locations": selection.protected,
                        "selection_seed": selection.seed,
                        "nearest_input_distance_km": selection.nearest_distance_km,
                        "elapsed_seconds": time.monotonic() - started_budget,
                        "member_elapsed_seconds": budget_member_timings,
                    }
                )
                for point in assignments.to_dict(orient="records"):
                    # The current compact batch contract has no profile
                    # coordinates. Preserve the per-point row and make the
                    # missing nearest-distance covariate explicit.
                    distance = point.get("nearest_input_distance_km", np.nan)
                    latitude = pd.to_numeric(point.get("latitude"), errors="coerce")
                    grid_distance = (
                        nearest_context_distance_km(
                            selection.selected_mask,
                            int(point["batch_index"]),
                            int(point["row"]),
                            int(point["col"]),
                            float(latitude),
                        )
                        if pd.isna(distance) and pd.notna(latitude)
                        else float("nan")
                    )
                    conditioning_covariates.append(
                        {
                            "point_id": str(point["point_id"]),
                            "batch": batch_number,
                            "budget": selection.budget,
                            "available_profile_locations": int(
                                selection.available_mask[
                                    int(point["batch_index"])
                                ].sum()
                            ),
                            "selected_profile_locations": int(
                                selection.selected_mask[int(point["batch_index"])].sum()
                            ),
                            "nearest_input_distance_km": (
                                grid_distance
                                if pd.isna(distance)
                                else pd.to_numeric(distance, errors="coerce")
                            ),
                            "glorys_context": point.get("glorys", np.nan),
                            "climatology_context": point.get("climatology", np.nan),
                            "distance_status": (
                                "available"
                                if pd.notna(distance)
                                else (
                                    "available_grid_estimate"
                                    if pd.notna(latitude)
                                    else "blocked_missing_point_latitude"
                                )
                            ),
                        }
                    )
        if batch_number == 0:
            first_members = np.asarray(member_values, dtype=np.float32).T
        else:
            first_members = np.concatenate(
                (first_members, np.asarray(member_values, dtype=np.float32).T), axis=0
            )
    if not collected:
        raise ValueError("No selected validation points were supplied")
    points = pd.concat(collected, ignore_index=True)
    if conditioning_enabled and conditioning_covariates:
        full_covariates = pd.DataFrame(conditioning_covariates)
        full_covariates = full_covariates[full_covariates.budget.eq("full")]
        full_covariates = full_covariates.drop_duplicates("point_id").set_index(
            "point_id"
        )
        points["context_density"] = points.point_id.astype(str).map(
            full_covariates.available_profile_locations
        )
        points["nearest_context_distance_km"] = points.point_id.astype(str).map(
            full_covariates.nearest_input_distance_km
        )
    for name, values in baseline_values.items():
        points[f"baseline_{name}"] = points["point_id"].astype(str).map(values)
        if points[f"baseline_{name}"].isna().any():
            raise ValueError(f"Baseline {name} did not produce every selected point")
    members_array = first_members.astype(np.float32, copy=False)
    points_path = output_dir / "points.csv"
    points.to_csv(points_path, index=False)
    saved_members = {"residual": members_array}
    if conditioning_enabled:
        for budget, values in conditioning_values.items():
            if values:
                saved_members[f"conditioning_{budget}"] = np.concatenate(values, axis=0)
        for name, values in conditioning_oi_values.items():
            if values:
                saved_members[name] = np.concatenate(values, axis=0)
    np.savez_compressed(output_dir / "members.npz", **saved_members)
    selection = pd.concat(all_assignments, ignore_index=True)
    selection.to_csv(output_dir / "selection.csv", index=False)
    exclusions = pd.DataFrame(columns=["point_id", "reason"])
    exclusions.to_csv(output_dir / "exclusions.csv", index=False)
    if conditioning_enabled:
        selection_table(conditioning_records).to_csv(
            output_dir / "conditioning_selection.csv", index=False
        )
        pd.DataFrame(conditioning_covariates).to_csv(
            output_dir / "conditioning_covariates.csv", index=False
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata = {
        "scenario": bundle.scenario,
        "members": members,
        "seed": seed,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "weight_variant": loaded,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_digest,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "point_count": int(len(points)),
        "conditioning_count": int(conditioning_count),
        "batch_records": batch_records,
        "baselines": baseline_status,
        "effective_config_sha256": {
            name: _sha256(Path(path))
            for name, path in (
                ("data", bundle.effective_data_config_path),
                ("model", bundle.effective_model_config_path),
                ("training", bundle.effective_training_config_path),
            )
        },
        "sampling_effective": bundle.training_cfg.get("training", {}).get(
            "validation_sampling", {}
        ),
        "elapsed_seconds": time.monotonic() - started,
        "member_timings": member_timing_records,
        "timing_measured_with_cuda_synchronization": device.type == "cuda",
    }
    if conditioning_enabled:
        metadata["conditioning_ablations"] = {
            "enabled": True,
            "budgets": conditioning_budgets,
            "same_member_seeds": True,
            "selection": "nested_per_batch_seeded_profile_locations",
            "surface_inputs_preserved": True,
            "record_count": len(conditioning_records),
            "local_oi": {
                "enabled": True,
                "backgrounds": ["glorys", "climatology"],
                "horizontal_scale_km": float(oi_cfg.get("horizontal_scale_km", 150.0)),
                "vertical_scale_m": float(oi_cfg.get("vertical_scale_m", 50.0)),
                "grid_spacing_deg": float(oi_cfg.get("grid_spacing_deg", 0.1)),
                "depth_axis_m": (
                    list(np.asarray(oi_depth_axis, dtype=float))
                    if oi_depth_axis is not None
                    else None
                ),
                "noise": oi_noise,
                "max_profiles": int(oi_cfg.get("max_profiles", 32)),
                "fit_scope": "fixed_config_or_calibration_only; no evaluation-label fitting",
                "unavailable_reasons": sorted(set(oi_errors)),
            },
        }
    if device.type == "cuda":
        metadata["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    _write_json(output_dir / "metadata.json", metadata)
    return metadata


def run_inference(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Reject unresolved orchestration; the validation CLI supplies cohort batches."""
    raise ValueError("run_inference requires resolved cohort batches; call run_model")


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Clone tensor inputs before prediction to protect cached cohort batches."""
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _batch_fingerprint(batch: Mapping[str, Any]) -> str:
    """Fingerprint tensor conditioning inputs to detect model mutation."""
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
