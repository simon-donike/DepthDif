"""Utilities for reproducible profile-conditioning ablations.

The ablation removes complete horizontal profile locations from a cached model
batch.  Surface observations and all other non-profile inputs are preserved so
the resulting curves isolate the contribution of the sparse profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


def _denorm(variable: str, values: torch.Tensor) -> np.ndarray:
    """Convert one normalized model field to physical units."""
    from depth_recon.utils.normalizations import (
        salinity_normalize,
        temperature_normalize,
    )

    normalize = salinity_normalize if variable == "salinity" else temperature_normalize
    return normalize("denorm", values.detach().cpu()).numpy()


def local_oi_predictions(
    batch: Mapping[str, Any],
    assignments: pd.DataFrame,
    retained_mask: torch.Tensor,
    *,
    background: str,
    variables: Sequence[str],
    depth_axis_m: Sequence[float] | None = None,
    horizontal_scale_km: float = 150.0,
    vertical_scale_m: float = 50.0,
    grid_spacing_deg: float = 0.1,
    noise_by_variable: Mapping[str, float] | None = None,
    max_profiles: int = 32,
) -> np.ndarray:
    """Compute deterministic local OI predictions from retained sparse profiles.

    The horizontal and vertical Gaussian scales and observation noise are fixed
    inputs to this operator.  They are deliberately not fit to evaluation labels.
    One nearest valid depth sample per retained profile is used for each target,
    keeping the small linear solve bounded while retaining vertical locality.
    """
    if background not in {"glorys", "climatology"}:
        raise ValueError("OI background must be glorys or climatology")
    # Support filtering retains original DataFrame labels; returned values use
    # row position, exactly like every other point-aligned prediction matrix.
    assignments = assignments.reset_index(drop=True)
    if depth_axis_m is None:
        raise ValueError("OI requires the actual model depth_axis_m")
    depth_axis = np.asarray(depth_axis_m, dtype=float)
    if (
        depth_axis.ndim != 1
        or not np.isfinite(depth_axis).all()
        or np.any(np.diff(depth_axis) <= 0)
    ):
        raise ValueError("OI depth_axis_m must be finite and strictly increasing")
    noise_by_variable = dict(
        noise_by_variable or {"temperature": 0.1, "salinity": 0.02}
    )
    result = np.full(len(assignments), np.nan, dtype=np.float32)
    retained = retained_mask.detach().cpu().bool()
    for variable in variables:
        selected = assignments["variable"].astype(str).eq(variable)
        if not selected.any():
            continue
        x_key = "x_salinity" if variable == "salinity" else "x"
        mask_key = "x_salinity_valid_mask" if variable == "salinity" else "x_valid_mask"
        background_key = (
            "y_salinity_glorys"
            if variable == "salinity" and background == "glorys"
            else (
                "y_glorys"
                if variable == "temperature" and background == "glorys"
                else "climatology_salinity" if variable == "salinity" else "climatology"
            )
        )
        if background_key not in batch and background == "glorys":
            fallback = "y_salinity" if variable == "salinity" else "y"
            if fallback in batch:
                background_key = fallback
        x_values = batch.get(x_key)
        x_valid = batch.get(mask_key)
        bg_values = batch.get(background_key)
        if not all(torch.is_tensor(value) for value in (x_values, x_valid, bg_values)):
            raise ValueError(
                f"Missing physical OI fields for {variable}: {x_key}, {mask_key}, {background_key}"
            )
        x_physical = _denorm(variable, x_values)
        bg_physical = _denorm(variable, bg_values)
        valid = x_valid.detach().cpu().bool().numpy()
        if valid.ndim == 4:
            valid = valid
        for point_index, point in assignments.loc[selected].iterrows():
            batch_index, row, col = (
                int(point.batch_index),
                int(point.row),
                int(point.col),
            )
            lower = int(
                point.depth_index_lower
                if "depth_index_lower" in point
                else point.depth_index
            )
            upper = int(
                point.depth_index_upper if "depth_index_upper" in point else lower
            )
            weight = float(point.depth_weight) if "depth_weight" in point else 0.0
            depth_index = lower + weight * (upper - lower)
            target_depth_m = (
                float(point.depth_m)
                if "depth_m" in point
                else float(
                    np.interp(depth_index, np.arange(len(depth_axis)), depth_axis)
                )
            )
            bg = float(
                (1.0 - weight) * bg_physical[batch_index, lower, row, col]
                + weight * bg_physical[batch_index, upper, row, col]
            )
            if not np.isfinite(bg):
                continue
            candidate_locations = np.argwhere(retained[batch_index].numpy())
            if candidate_locations.size == 0:
                result[point_index] = bg
                continue
            latitude = float(point.get("latitude", 0.0))
            lat_scale = horizontal_scale_km / (111.32 * grid_spacing_deg)
            lon_scale = horizontal_scale_km / (
                111.32 * grid_spacing_deg * max(np.cos(np.deg2rad(latitude)), 1e-3)
            )
            horizontal = ((candidate_locations[:, 0] - row) / lat_scale) ** 2 + (
                (candidate_locations[:, 1] - col) / lon_scale
            ) ** 2
            nearest = np.argsort(horizontal)[:max_profiles]
            obs, distances = [], []
            for candidate_index in nearest:
                profile_row, profile_col = candidate_locations[candidate_index]
                depths = np.flatnonzero(valid[batch_index, :, profile_row, profile_col])
                if not len(depths):
                    continue
                nearest_depth = depths[
                    np.argmin(np.abs(depth_axis[depths] - target_depth_m))
                ]
                observation = float(
                    x_physical[batch_index, nearest_depth, profile_row, profile_col]
                )
                background_value = float(
                    bg_physical[batch_index, nearest_depth, profile_row, profile_col]
                )
                if not np.isfinite(observation) or not np.isfinite(background_value):
                    continue
                obs.append(observation - background_value)
                distances.append(
                    (
                        (float(profile_row) - row) / lat_scale,
                        (float(profile_col) - col) / lon_scale,
                        (float(depth_axis[nearest_depth] - target_depth_m))
                        / vertical_scale_m,
                    )
                )
            if not obs:
                result[point_index] = bg
                continue
            distances = np.asarray(distances, dtype=float)
            obs = np.asarray(obs, dtype=float)
            pairwise = distances[:, None, :] - distances[None, :, :]
            covariance = np.exp(-0.5 * np.sum(pairwise**2, axis=2))
            # The nugget keeps this small local solve well conditioned.
            covariance += np.eye(len(obs)) * float(noise_by_variable[variable]) ** 2
            target_covariance = np.exp(-0.5 * np.sum(distances**2, axis=1))
            result[point_index] = bg + float(
                target_covariance @ np.linalg.solve(covariance, obs)
            )
    return result


@dataclass(frozen=True)
class ConditioningSelection:
    """One nested selection of profile locations for a batch and budget."""

    batch: dict[str, Any]
    budget: int
    selected: int
    available: int
    protected: int
    seed: int
    nearest_distance_km: float
    selected_mask: torch.Tensor
    available_mask: torch.Tensor


def _support(batch: Mapping[str, Any]) -> torch.Tensor:
    """Return the union of valid temperature and salinity input locations."""
    supports = []
    for key in ("x_valid_mask", "x_salinity_valid_mask"):
        value = batch.get(key)
        if torch.is_tensor(value):
            mask = value.bool()
            # Input masks are [B, depth, y, x]; tolerate a singleton channel.
            while mask.ndim > 3:
                mask = mask.any(dim=1)
            supports.append(mask)
    if not supports:
        raise ValueError(
            "Conditioning ablation requires x_valid_mask or x_salinity_valid_mask"
        )
    result = supports[0]
    for value in supports[1:]:
        result = result | value
    return result


def _protected(assignments: pd.DataFrame, shape: tuple[int, int]) -> torch.Tensor:
    """Mark target cells so an imperfectly masked batch cannot leak them."""
    height, width = shape
    result = torch.zeros(shape, dtype=torch.bool)
    for item in assignments.to_dict(orient="records"):
        row, col = int(item.get("row", -1)), int(item.get("col", -1))
        if 0 <= row < height and 0 <= col < width:
            result[row, col] = True
    return result


def _mask_batch(batch: Mapping[str, Any], selected: torch.Tensor) -> dict[str, Any]:
    """Clone a batch and remove every depth sample at selected locations."""
    result = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    for value_key, mask_key in (
        ("x", "x_valid_mask"),
        ("x_salinity", "x_salinity_valid_mask"),
    ):
        values = result.get(value_key)
        valid = result.get(mask_key)
        if not torch.is_tensor(values) or not torch.is_tensor(valid):
            continue
        location_mask = selected.to(device=values.device)
        # Inputs normally have [B,D,H,W]; the branch also supports compact
        # [B,H,W] fixtures used by lightweight validation tests.
        broadcast = (
            location_mask
            if values.ndim == location_mask.ndim
            else location_mask.unsqueeze(1)
        )
        values.masked_fill_(broadcast, 0.0)
        valid_broadcast = (
            location_mask
            if valid.ndim == location_mask.ndim
            else location_mask.unsqueeze(1)
        )
        valid.masked_fill_(valid_broadcast, False)
        one_d = result.get(mask_key + "_1d")
        if torch.is_tensor(one_d):
            one_d.masked_fill_(
                (
                    location_mask
                    if one_d.ndim == location_mask.ndim
                    else location_mask.unsqueeze(1)
                ),
                False,
            )
    return result


def nested_selections(
    batch: Mapping[str, Any],
    assignments: pd.DataFrame,
    budgets: Sequence[int],
    *,
    seed: int,
) -> list[ConditioningSelection]:
    """Build deterministic nested retained-profile budgets shared by every draw."""
    support = _support(batch).cpu()
    if support.ndim != 3:
        raise ValueError(
            f"Expected batch location support [B,H,W], got {support.shape}"
        )
    budgets = sorted({int(value) for value in budgets})
    if any(value < 0 for value in budgets):
        raise ValueError("Conditioning budgets must be non-negative")
    protected = torch.stack(
        [
            _protected(
                assignments.loc[assignments.batch_index.eq(index)],
                tuple(support.shape[1:]),
            )
            for index in range(support.shape[0])
        ]
    )
    candidates = support & ~protected
    generator = np.random.default_rng(int(seed))
    order = np.full(candidates.shape, -1, dtype=np.int64)
    for batch_index in range(candidates.shape[0]):
        locations = np.flatnonzero(candidates[batch_index].numpy().ravel())
        generator.shuffle(locations)
        for rank, location in enumerate(locations):
            order[batch_index].ravel()[location] = rank
    selections = []
    for budget in budgets:
        retained = torch.zeros_like(candidates)
        for batch_index in range(candidates.shape[0]):
            retained[batch_index] = torch.from_numpy(
                (order[batch_index] >= 0) & (order[batch_index] < budget)
            )
        # Protected target cells are absent from candidates and are therefore
        # always removed from the conditioning input, even at a large budget.
        removed = support & ~retained
        masked = _mask_batch(batch, removed)
        distances = (
            pd.to_numeric(
                assignments.loc[
                    assignments.apply(
                        lambda row: bool(
                            retained[int(row.batch_index), int(row.row), int(row.col)]
                        ),
                        axis=1,
                    ),
                    "nearest_input_distance_km",
                ],
                errors="coerce",
            )
            if "nearest_input_distance_km" in assignments
            else pd.Series(dtype=float)
        )
        selections.append(
            ConditioningSelection(
                batch=masked,
                budget=budget,
                selected=int(retained.sum()),
                available=int(candidates.sum()),
                protected=int(protected.sum()),
                seed=int(seed),
                nearest_distance_km=(
                    float(distances.mean())
                    if not distances.empty and distances.notna().any()
                    else float("nan")
                ),
                selected_mask=retained,
                available_mask=candidates,
            )
        )
    return selections


def nearest_context_distance_km(
    selected_mask: torch.Tensor,
    batch_index: int,
    row: int,
    col: int,
    latitude: float,
    *,
    spacing_deg: float = 0.1,
) -> float:
    """Estimate nearest retained context distance on the fixed 0.1-degree grid."""
    locations = torch.nonzero(selected_mask[int(batch_index)], as_tuple=False)
    if locations.numel() == 0:
        return float("nan")
    delta_row = locations[:, 0].numpy().astype(float) - float(row)
    delta_col = locations[:, 1].numpy().astype(float) - float(col)
    lat_km = delta_row * float(spacing_deg) * 111.32
    lon_km = (
        delta_col * float(spacing_deg) * 111.32 * np.cos(np.deg2rad(float(latitude)))
    )
    return float(np.sqrt(lat_km**2 + lon_km**2).min())


def selection_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a stable tabular representation of captured ablation metadata."""
    columns = [
        "batch",
        "budget",
        "selected_profiles",
        "available_profiles",
        "protected_target_locations",
        "selection_seed",
        "nearest_input_distance_km",
        "elapsed_seconds",
        "member_elapsed_seconds",
    ]
    return pd.DataFrame.from_records(records, columns=columns)
