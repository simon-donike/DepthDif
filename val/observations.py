"""Build leakage controlled cohorts from external profile observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import torch

CANONICAL_COLUMNS = (
    "profile_id",
    "cluster_id",
    "source",
    "observation_time",
    "longitude",
    "latitude",
    "depth_m",
    "variable",
    "observation",
    "independence",
)
_CLUSTER_VARIABLES = ("cluster_id", "platform_number", "platform", "cruise_id")


def _as_date(value: Any) -> int:
    """Convert a timestamp to the dataset's YYYYMMDD integer representation."""
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return int(timestamp.strftime("%Y%m%d"))


def _region_label(latitude: float, longitude: float, size: float) -> str:
    """Return a stable geographic tile label for one WGS84 point."""
    lat0 = int(np.floor(float(latitude) / size) * size)
    lon0 = int(np.floor(((float(longitude) + 180.0) % 360.0 - 180.0) / size) * size)
    return f"lat{lat0:+03d}_lon{lon0:+04d}"


def _validate_points(
    points: pd.DataFrame, accepted_qc: tuple[int, ...]
) -> pd.DataFrame:
    """Validate and normalize raw external observation rows without inventing data."""
    if not isinstance(points, pd.DataFrame):
        raise TypeError("points must be a pandas DataFrame")
    missing = sorted(set(CANONICAL_COLUMNS) - set(points.columns))
    if missing:
        raise ValueError("points missing canonical columns: " + ", ".join(missing))
    out = points.copy()
    if (
        out["profile_id"].isna().any()
        or out["profile_id"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("profile_id must be populated")
    for column in ("longitude", "latitude", "depth_m", "observation"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or not np.isfinite(out[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if ((out["longitude"] < -180) | (out["longitude"] > 180)).any():
        raise ValueError("longitude must be in [-180, 180]")
    if ((out["latitude"] < -90) | (out["latitude"] > 90)).any():
        raise ValueError("latitude must be in [-90, 90]")
    if (out["depth_m"] < 0).any():
        raise ValueError("depth_m cannot be negative")
    out["observation_time"] = pd.to_datetime(
        out["observation_time"], utc=True, errors="coerce", format="mixed"
    )
    if out["observation_time"].isna().any():
        raise ValueError("observation_time must contain valid timestamps")
    out["variable"] = out["variable"].astype(str).str.strip().str.lower()
    if (~out["variable"].isin({"temperature", "salinity"})).any():
        raise ValueError("variable must be temperature or salinity")
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    if out["source"].eq("").any():
        raise ValueError("source must be populated")
    source_prefix = out["source"].str.replace("_", "-", regex=False) + ":"
    profile_ids = out["profile_id"].astype(str).str.strip()
    out["profile_id"] = [
        (
            profile_id
            if profile_id.lower().startswith(prefix.lower())
            else prefix + profile_id
        )
        for profile_id, prefix in zip(profile_ids, source_prefix)
    ]
    if out.duplicated(["source", "profile_id", "variable", "depth_m"]).any():
        raise ValueError("duplicate profile_id/variable/depth_m rows are ambiguous")
    if "qc_flag" in out:
        qc = pd.to_numeric(out["qc_flag"], errors="coerce")
        if qc.isna().any() or (qc % 1 != 0).any() or ((qc < 0) | (qc > 9)).any():
            raise ValueError("qc_flag must contain integer codes in [0, 9]")
        out["qc_flag"] = qc.astype(int)
        out["_qc_valid"] = out["qc_flag"].isin(accepted_qc)
    else:
        out["_qc_valid"] = True
    cluster_ids = out["cluster_id"].fillna(out["profile_id"]).astype(str).str.strip()
    if cluster_ids.eq("").any():
        raise ValueError("cluster_id must be populated")
    out["cluster_id"] = cluster_ids
    out["independence"] = out["independence"].fillna("unknown").astype(str)
    return out.reset_index(drop=True)


def _model_depth_assignment(
    observation_depth: float, model_depths: np.ndarray, max_gap: float
) -> dict[str, Any] | None:
    """Map one raw observation depth to exact or adjacent model levels."""
    right = int(np.searchsorted(model_depths, observation_depth, side="left"))
    if right < len(model_depths) and np.isclose(
        model_depths[right], observation_depth, rtol=0.0, atol=1e-8
    ):
        return {
            "depth_index_lower": right,
            "depth_index_upper": right,
            "depth_weight": 0.0,
            "matched_depth_m": float(model_depths[right]),
        }
    left = right - 1
    if left < 0 or right >= len(model_depths):
        return None
    gap = float(model_depths[right] - model_depths[left])
    if gap <= 0 or gap > max_gap:
        return None
    return {
        "depth_index_lower": left,
        "depth_index_upper": right,
        "depth_weight": float((observation_depth - model_depths[left]) / gap),
        "matched_depth_m": float(observation_depth),
    }


def _temperature_kind(points: pd.DataFrame, config: Mapping[str, Any]) -> str | None:
    """Resolve an explicit temperature representation from config or input rows."""
    temperature = points.loc[points["variable"].eq("temperature")]
    if temperature.empty:
        return None
    kinds: set[str] = set()
    if "temperature_kind" in temperature:
        kinds = set(
            temperature["temperature_kind"].dropna().astype(str).str.strip().str.lower()
        )
        kinds.discard("")
    configured = config.get("temperature_kind")
    if configured is not None:
        configured = str(configured).strip().lower()
        if kinds and kinds != {configured}:
            raise ValueError(
                "temperature_kind conflicts between config and observation rows"
            )
        kinds = {configured}
    if len(kinds) != 1 or next(iter(kinds), None) not in {"potential", "in_situ"}:
        raise ValueError(
            "temperature_kind must be explicitly set to potential or in_situ"
        )
    return next(iter(kinds))


def _convert_temperature(
    points: pd.DataFrame, kind: str | None, config: Mapping[str, Any]
):
    """Convert in-situ observations to potential temperature with explicit salinity."""
    out = points.copy()
    out["observation_raw"] = out["observation"]
    if kind != "in_situ":
        return out, None
    operator = config.get("temperature_observation_operator")
    if operator != "in_situ_to_potential_teos10":
        raise ValueError(
            "in_situ temperature requires temperature_observation_operator="
            "in_situ_to_potential_teos10 and per-row reference_salinity"
        )
    salinity_kind = config.get("reference_salinity_kind")
    if salinity_kind not in {"practical", "absolute"}:
        raise ValueError(
            "reference_salinity_kind must be practical or absolute for TEOS-10 conversion"
        )
    if "reference_salinity" not in out:
        raise ValueError("TEOS-10 conversion requires a reference_salinity column")
    provenance = config.get("reference_salinity_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "TEOS-10 conversion requires reference_salinity_provenance with path and sha256"
        )
    provenance_path = Path(str(provenance.get("path", "")))
    expected_sha256 = str(provenance.get("sha256", "")).lower()
    if not provenance_path.is_file() or len(expected_sha256) != 64:
        raise ValueError(
            "reference_salinity_provenance must identify an existing file and SHA256"
        )
    with provenance_path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("reference salinity provenance SHA256 does not match")
    selected = out["variable"].eq("temperature")
    reference = pd.to_numeric(out.loc[selected, "reference_salinity"], errors="coerce")
    if reference.isna().any() or not np.isfinite(reference).all():
        raise ValueError("temperature rows require finite reference_salinity values")
    try:
        import gsw
    except ImportError as error:
        raise ValueError(
            "TEOS-10 conversion requires the optional gsw package"
        ) from error
    depth = out.loc[selected, "depth_m"].to_numpy(float)
    latitude = out.loc[selected, "latitude"].to_numpy(float)
    longitude = out.loc[selected, "longitude"].to_numpy(float)
    pressure = gsw.p_from_z(-depth, latitude)
    absolute = reference.to_numpy(float)
    if salinity_kind == "practical":
        absolute = gsw.SA_from_SP(absolute, pressure, longitude, latitude)
    converted = gsw.pt0_from_t(
        absolute,
        out.loc[selected, "observation"].to_numpy(float),
        pressure,
    )
    if not np.isfinite(converted).all():
        raise ValueError("TEOS-10 temperature conversion produced non-finite values")
    out.loc[selected, "observation"] = converted
    return out, "in_situ_to_potential_teos10"


def _grid_transform(dataset: Any):
    """Resolve the authoritative raster transform from the first GLORYS GeoTIFF."""
    store = dataset.glorys_store
    path = next(iter(store.paths_by_date.values()))
    with rasterio.open(path) as source:
        return source.transform, source.height, source.width


@dataclass
class ExternalObservationCohort:
    """Leakage controlled external profile cohort and masked patch adapter."""

    dataset: Any
    assignments: pd.DataFrame
    patch_indices: list[int]
    holdouts: set[tuple[int, int, int]]
    metadata: dict[str, Any]
    exclusions: pd.DataFrame

    def __getitem__(self, key: str) -> Any:
        """Expose cohort fields through the dictionary-like API."""
        return getattr(self, key)

    def keys(self):
        """Return dictionary-style cohort field names."""
        return (
            "assignments",
            "patch_indices",
            "holdouts",
            "metadata",
            "exclusions",
            "masked_sample",
        )

    def masked_sample(self, index: int) -> dict[str, Any]:
        """Return a cloned dataset sample with every holdout cell removed."""
        sample = dict(self.dataset[int(index)])
        row = self.dataset._rows.iloc[int(index)]
        for key in ("x", "x_salinity"):
            mask_key = f"{key}_valid_mask"
            if key not in sample or mask_key not in sample:
                continue
            values = sample[key].clone()
            valid = sample[mask_key].clone()
            for date, grid_row, grid_col in self.holdouts:
                yy = int(grid_row) - int(row.grid_y0)
                xx = int(grid_col) - int(row.grid_x0)
                if (
                    int(row.date) == int(date)
                    and 0 <= yy < values.shape[-2]
                    and 0 <= xx < values.shape[-1]
                ):
                    values[..., yy, xx] = 0
                    valid[..., yy, xx] = False
            sample[key] = values
            sample[mask_key] = valid
            one_d = f"{mask_key}_1d"
            if one_d in sample:
                sample[one_d] = valid.bool().any(dim=0, keepdim=True)
        return sample


def prepare_external(
    dataset: Any, points: pd.DataFrame, config: Mapping[str, Any] | None = None
) -> ExternalObservationCohort:
    """Prepare raw external observations for interpolation from model depths."""
    cfg = dict(config or {})
    accepted_qc = tuple(int(value) for value in cfg.get("accepted_qc_flags", (1,)))
    if not accepted_qc or any(value < 0 or value > 9 for value in accepted_qc):
        raise ValueError("accepted_qc_flags must contain codes in [0, 9]")
    data = _validate_points(points, accepted_qc)
    temp_kind = _temperature_kind(data, cfg)
    data, temperature_operator = _convert_temperature(data, temp_kind, cfg)
    max_gap = float(cfg.get("max_vertical_gap_m", 100.0))
    if max_gap <= 0:
        raise ValueError("max_vertical_gap_m must be positive")
    tolerance_days = float(cfg.get("max_time_delta_days", 3.5))
    transform, height, width = _grid_transform(dataset)
    target_dates = np.unique(np.asarray(dataset._rows["date"], dtype=np.int64))
    if target_dates.size == 0:
        raise ValueError("dataset has no target dates")
    target_timestamps = {
        int(value): pd.Timestamp(str(int(value)), tz="UTC")
        for value in target_dates.tolist()
    }
    native_depths = np.asarray(dataset.depth_axis_m, dtype=float)
    if (
        native_depths.ndim != 1
        or native_depths.size == 0
        or not np.isfinite(native_depths).all()
        or np.any(np.diff(native_depths) <= 0)
    ):
        raise ValueError("dataset.depth_axis_m must be finite and strictly increasing")
    assignments: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    holdout_seed_rows: list[dict[str, Any]] = []
    patch_indices: set[int] = set()
    for profile_id, group in data.groupby("profile_id", sort=False):
        for column in (
            "source",
            "cluster_id",
            "observation_time",
            "longitude",
            "latitude",
        ):
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(
                    f"profile_id {profile_id!r} has inconsistent {column} metadata"
                )
        timestamp = group["observation_time"].iloc[0]
        nearest_date = min(
            target_dates.tolist(),
            key=lambda value: abs(target_timestamps[int(value)] - timestamp),
        )
        # Keep the sign so audits can detect systematic observation/week offsets.
        delta = (
            timestamp - target_timestamps[int(nearest_date)]
        ).total_seconds() / 86400.0
        if abs(delta) > tolerance_days:
            exclusions.append(
                {"profile_id": profile_id, "reason": "no_target_date_within_tolerance"}
            )
            continue
        longitude = float(group["longitude"].iloc[0])
        latitude = float(group["latitude"].iloc[0])
        grid_row, grid_col = rowcol(transform, longitude, latitude)
        grid_row, grid_col = int(grid_row), int(grid_col)
        if not (0 <= grid_row < height and 0 <= grid_col < width):
            exclusions.append(
                {"profile_id": profile_id, "reason": "coordinate_outside_grid"}
            )
            continue
        rows = dataset._rows
        candidate_mask = (
            (rows.date == nearest_date)
            & (rows.grid_y0 <= grid_row)
            & (rows.grid_y0 + dataset.tile_size > grid_row)
            & (rows.grid_x0 <= grid_col)
            & (rows.grid_x0 + dataset.tile_size > grid_col)
        )
        candidates = np.flatnonzero(candidate_mask.to_numpy()).tolist()
        if not candidates:
            exclusions.append(
                {
                    "profile_id": profile_id,
                    "reason": "no_validation_patch_covers_coordinate",
                }
            )
            continue
        patch_index = int(candidates[0])
        assignment_start = len(assignments)
        for variable, variable_group in group.groupby("variable", sort=False):
            for row in variable_group.to_dict(orient="records"):
                if not bool(row.pop("_qc_valid")):
                    continue
                depth_assignment = _model_depth_assignment(
                    float(row["depth_m"]), native_depths, max_gap
                )
                if depth_assignment is None:
                    continue
                row.update(depth_assignment)
                supplied_region = row.get("region")
                region = (
                    str(supplied_region).strip()
                    if supplied_region is not None
                    and not pd.isna(supplied_region)
                    and str(supplied_region).strip()
                    else _region_label(
                        latitude, longitude, float(cfg.get("region_size_deg", 20.0))
                    )
                )
                depth_token = f"{float(row['depth_m']):.8g}"
                row.update(
                    {
                        "variable": variable,
                        "target_date": int(nearest_date),
                        "time_delta_days": float(delta),
                        "patch_index": patch_index,
                        "local_row": grid_row - int(rows.iloc[patch_index].grid_y0),
                        "local_col": grid_col - int(rows.iloc[patch_index].grid_x0),
                        "point_id": f"{profile_id}:{variable}:{depth_token}",
                        "region": region,
                    }
                )
                assignments.append(row)
        if len(assignments) == assignment_start:
            exclusions.append(
                {"profile_id": profile_id, "reason": "no_qc_supported_model_depth"}
            )
            continue
        patch_indices.add(patch_index)
        holdout_seed_rows.append(
            {
                "cluster_id": str(group["cluster_id"].iloc[0]),
                "date": int(nearest_date),
                "grid_row": grid_row,
                "grid_col": grid_col,
            }
        )
    if not assignments:
        raise ValueError(
            "no external observations matched dataset dates, grid, and native depths"
        )
    holdouts = {
        (item["date"], item["grid_row"], item["grid_col"]) for item in holdout_seed_rows
    }
    buffer_cells = int(cfg.get("buffer_cells", 0))
    if buffer_cells < 0:
        raise ValueError("buffer_cells cannot be negative")
    argo_store = getattr(dataset, "argo_store", None)
    argo_dataset = getattr(argo_store, "ds", None)
    cluster_variables = set(getattr(argo_dataset, "variables", ()))
    available_cluster = next(
        (name for name in _CLUSTER_VARIABLES if name in cluster_variables), None
    )
    cluster_match_count = 0
    if available_cluster is None:
        if cfg.get("split_role") == "confirmation":
            raise ValueError(
                "confirmatory external validation requires cluster metadata for complete conditioning exclusion"
            )
        warnings.warn(
            "compact store has no cluster metadata; excluding all evaluated weekly coordinate cells only",
            UserWarning,
            stacklevel=2,
        )
    else:
        cluster_values = np.asarray(argo_dataset[available_cluster].values).reshape(-1)
        cluster_values = np.asarray(
            [
                (
                    value.decode(errors="ignore")
                    if isinstance(value, (bytes, np.bytes_))
                    else str(value)
                )
                for value in cluster_values
            ]
        )
        for item in holdout_seed_rows:
            matches = np.flatnonzero(cluster_values == str(item["cluster_id"]))
            matches = matches[
                np.asarray(argo_store.target_date)[matches] == item["date"]
            ]
            cluster_match_count += len(matches)
            holdouts.update(
                (
                    int(item["date"]),
                    int(argo_store.grid_row[index]),
                    int(argo_store.grid_col[index]),
                )
                for index in matches.tolist()
            )
        if cfg.get("split_role") == "confirmation" and cluster_match_count == 0:
            raise ValueError(
                "confirmatory external validation found no conditioning cluster identity matches"
            )
    if buffer_cells:
        holdouts = {
            (date, y + dy, x + dx)
            for date, y, x in holdouts
            for dy in range(-buffer_cells, buffer_cells + 1)
            for dx in range(-buffer_cells, buffer_cells + 1)
        }
    metadata = {
        "temperature_kind_input": temp_kind,
        "temperature_kind_compared": "potential" if temp_kind else None,
        "temperature_observation_operator": temperature_operator,
        "accepted_qc_flags": list(accepted_qc),
        "max_vertical_gap_m": max_gap,
        "max_time_delta_days": tolerance_days,
        "cluster_metadata_variable": available_cluster,
        "cluster_match_count": cluster_match_count,
        "confirmation_eligible": available_cluster is not None
        and cluster_match_count > 0,
        "conditioning_exclusion_verified": available_cluster is not None
        and cluster_match_count > 0,
        "conditioning_exclusion_scope": (
            "complete matching cluster and evaluated coordinate cells"
            if available_cluster is not None
            else "evaluated coordinate cells only"
        ),
        "independence_values": sorted(data["independence"].unique().tolist()),
    }
    if temperature_operator is not None:
        metadata["reference_salinity_provenance"] = dict(
            cfg["reference_salinity_provenance"]
        )
    result = pd.DataFrame.from_records(assignments)
    result["batch_index"] = result["patch_index"].map(
        {value: index for index, value in enumerate(sorted(patch_indices))}
    )
    result["row"] = result["local_row"]
    result["col"] = result["local_col"]
    return ExternalObservationCohort(
        dataset,
        result.reset_index(drop=True),
        sorted(patch_indices),
        holdouts,
        metadata,
        pd.DataFrame.from_records(exclusions, columns=("profile_id", "reason")),
    )
