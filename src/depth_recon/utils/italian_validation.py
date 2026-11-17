"""Leakage controlled Italian profile validation for training runs.

The callback deliberately keeps this validation dataset separate from the normal
Lightning validation loader.  The model specific reconstruction function is
injected by the training worker, which keeps the utility independent of a
particular diffusion model implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from rasterio.transform import rowcol
from torch.utils.data import default_collate

from depth_recon.data.dataset_argo_geotiff_gridded import (
    ArgoGeoTIFFGriddedPatchDataset,
)
from depth_recon.utils.normalizations import temperature_normalize

try:  # Lightning is optional when this module is used by offline tooling.
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - exercised only in minimal environments

    class _Callback:
        """Small fallback allowing metric helpers to be imported without Lightning."""

    class pl:  # type: ignore[no-redef]
        Callback = _Callback


CANONICAL_BANDS_M = (0.0, 10.0, 50.0, 100.0, 200.0, 500.0, 700.0, 2000.0, float("inf"))
_CLUSTER_VARIABLES = ("cluster_id", "platform_number", "platform", "cruise_id")


def prepare_italian_assignments(
    profiles: pd.DataFrame,
    model_depths_m: Sequence[float],
    target_dates: Sequence[int],
    *,
    max_time_delta_days: float = 3.5,
    max_vertical_gap_m: float = 100.0,
) -> pd.DataFrame:
    """Assign screened profile rows to nearest target dates and model depths.

    Rows outside the temporal/depth support are retained with ``eligible=False``
    so the caller can report exclusions without changing the scored denominator.
    """
    depths = np.asarray(model_depths_m, dtype=float)
    if depths.ndim != 1 or len(depths) == 0 or np.any(np.diff(depths) <= 0):
        raise ValueError("model_depths_m must be strictly increasing")
    dates = np.asarray(target_dates, dtype=np.int64)
    if len(dates) == 0:
        raise ValueError("target_dates cannot be empty")
    out = profiles.copy()
    timestamps = pd.to_datetime(out["observation_time"], utc=True)
    target_ts = pd.to_datetime(dates.astype(str), format="%Y%m%d", utc=True)
    nearest = np.asarray(
        [int(dates[np.argmin(np.abs(target_ts - value))]) for value in timestamps]
    )
    delta = np.asarray(
        [
            (value - pd.Timestamp(str(date), tz="UTC")).total_seconds() / 86400.0
            for value, date in zip(timestamps, nearest)
        ]
    )
    depth_values = out.depth_m.to_numpy(float)
    right = np.searchsorted(depths, depth_values, side="left")
    valid_depth = right < len(depths)
    exact = valid_depth & np.isclose(
        depths[np.minimum(right, len(depths) - 1)], depth_values, atol=1e-8
    )
    lower = np.clip(right - 1, 0, len(depths) - 1)
    right_safe = np.minimum(right, len(depths) - 1)
    valid_depth &= exact | (
        (right > 0)
        & (right < len(depths))
        & ((depths[right_safe] - depths[lower]) <= max_vertical_gap_m)
    )
    out["target_date"] = nearest
    out["time_delta_days"] = delta
    out["depth_index"] = np.where(exact, right, lower)
    out["matched_depth_m"] = depths[out.depth_index.to_numpy(int)]
    out["eligible"] = (np.abs(delta) <= max_time_delta_days) & valid_depth
    return out


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact(path: Path, expected: str | None, label: str) -> str:
    """Verify a configured bundle artifact and return its digest."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if expected and actual.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA256 does not match: {path}")
    return actual


@dataclass(frozen=True)
class ItalianValidationConfig:
    """Immutable settings for one fixed Italian validation cohort."""

    profiles_path: Path
    data_config_path: Path | None = None
    profiles_sha256: str | None = None
    reference_manifest: Path | None = None
    reference_manifest_sha256: str | None = None
    screening_registry: Path | None = None
    screening_registry_sha256: str | None = None
    temperature_kind: str = "potential"
    temperature_observation_operator: str | None = None
    reference_salinity_kind: str = "practical"
    year: int = 2016
    batch_size: int = 2
    min_input_profiles: int = 1
    max_time_delta_days: float = 3.5
    buffer_cells: int = 1
    max_vertical_gap_m: float = 100.0
    ddpm_steps: int = 1000
    seed: int = 20261103
    invalid_prediction_penalty: float = 1.0e6
    depth_bands_m: tuple[float, ...] = CANONICAL_BANDS_M

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ItalianValidationConfig":
        """Build settings from ``training.training.italian_validation``."""
        raw = dict(config)
        dataset_root = raw.get("dataset_root")
        if not dataset_root:
            dataset = raw.get("dataset", {})
            dataset_root = dataset.get("core", {}).get("geotiff_root_dir")
        dataset_root = Path(str(dataset_root)) if dataset_root else None

        def resolve(name: str, *aliases: str) -> Path | None:
            value = next(
                (raw.get(key) for key in (name, *aliases) if raw.get(key)), None
            )
            if value is None:
                return None
            path = Path(str(value))
            return (
                path
                if path.is_absolute() or dataset_root is None
                else dataset_root / path
            )

        profiles_path = resolve("profiles_path", "profiles")
        if profiles_path is None:
            raise ValueError("Italian validation requires profiles_path")
        return cls(
            profiles_path=profiles_path,
            data_config_path=(
                Path(str(raw["data_config_path"]))
                if raw.get("data_config_path")
                else None
            ),
            profiles_sha256=raw.get("profiles_sha256"),
            reference_manifest=resolve("reference_manifest_path", "reference_manifest"),
            reference_manifest_sha256=raw.get("reference_manifest_sha256"),
            screening_registry=resolve(
                "screened_profiles_path", "screening_registry", "screened_profiles"
            ),
            screening_registry_sha256=raw.get(
                "screening_registry_sha256", raw.get("screened_profiles_sha256")
            ),
            temperature_kind=str(raw.get("temperature_kind", "potential")),
            temperature_observation_operator=raw.get(
                "temperature_observation_operator"
            ),
            reference_salinity_kind=str(
                raw.get("reference_salinity_kind", "practical")
            ),
            year=int(raw.get("year", 2016)),
            batch_size=int(raw.get("batch_size", 2)),
            min_input_profiles=int(raw.get("min_input_profiles", 1)),
            max_time_delta_days=float(raw.get("max_time_delta_days", 3.5)),
            buffer_cells=int(raw.get("buffer_cells", 1)),
            max_vertical_gap_m=float(raw.get("max_vertical_gap_m", 100.0)),
            ddpm_steps=int(raw.get("ddpm_steps", 1000)),
            seed=int(raw.get("seed", 20261103)),
            invalid_prediction_penalty=float(
                raw.get("invalid_prediction_penalty", 1.0e6)
            ),
            depth_bands_m=tuple(
                float(x) for x in raw.get("depth_bands_m", CANONICAL_BANDS_M)
            ),
        )

    def load_profiles(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load and verify the prepared profile table and its manifest."""
        if self.year != 2016:
            raise ValueError(
                "Italian validation is fixed to the 2016 temperature cohort"
            )
        profiles_digest = _verify_artifact(
            self.profiles_path, self.profiles_sha256, "Italian profiles"
        )
        profiles = pd.read_csv(self.profiles_path)
        required = {"profile_id", "observation_time", "depth_m", "observation"}
        missing = sorted(required - set(profiles.columns))
        if missing:
            raise ValueError("Italian profiles missing columns: " + ", ".join(missing))
        profiles["observation_time"] = pd.to_datetime(
            profiles["observation_time"], utc=True, errors="coerce"
        )
        profiles["depth_m"] = pd.to_numeric(profiles["depth_m"], errors="coerce")
        profiles["observation"] = pd.to_numeric(
            profiles["observation"], errors="coerce"
        )
        if profiles[["observation_time", "depth_m", "observation"]].isna().any().any():
            raise ValueError("Italian profiles contain invalid timestamps or values")
        if profiles["observation_time"].dt.year.ne(self.year).any():
            raise ValueError("Italian profile table contains observations outside 2016")
        if (
            "variable" in profiles
            and profiles["variable"].astype(str).str.lower().ne("temperature").any()
        ):
            raise ValueError("Italian callback accepts temperature profiles only")
        manifest: dict[str, Any] | None = None
        if self.reference_manifest is not None:
            _verify_artifact(
                self.reference_manifest,
                self.reference_manifest_sha256,
                "Italian reference manifest",
            )
            manifest = json.loads(self.reference_manifest.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("reference_manifest must contain a JSON object")
            provenance = manifest.get("config_snippet", {}).get(
                "reference_salinity_provenance", {}
            )
            if self.temperature_kind == "in_situ":
                if str(provenance.get("sha256", "")).lower() != profiles_digest:
                    raise ValueError(
                        "Italian reference manifest does not pin the prepared salinity profiles"
                    )
        if self.temperature_kind == "in_situ":
            if self.temperature_observation_operator != "in_situ_to_potential_teos10":
                raise ValueError(
                    "in-situ Italian validation requires temperature_observation_operator="
                    "in_situ_to_potential_teos10"
                )
            if self.reference_salinity_kind not in {"practical", "absolute"}:
                raise ValueError(
                    "reference_salinity_kind must be practical or absolute"
                )
            if "reference_salinity" not in profiles:
                raise ValueError("in-situ Italian profiles require reference_salinity")
            try:
                import gsw
            except ImportError as error:
                raise ValueError("in-situ Italian validation requires gsw") from error
            salinity = profiles.reference_salinity.to_numpy(float)
            pressure = gsw.p_from_z(
                -profiles.depth_m.to_numpy(float), profiles.latitude.to_numpy(float)
            )
            if self.reference_salinity_kind == "practical":
                salinity = gsw.SA_from_SP(
                    salinity,
                    pressure,
                    profiles.longitude.to_numpy(float),
                    profiles.latitude.to_numpy(float),
                )
            profiles["observation"] = gsw.pt0_from_t(
                salinity, profiles.observation.to_numpy(float), pressure
            )
            if not np.isfinite(profiles.observation.to_numpy(float)).all():
                raise ValueError("TEOS-10 conversion produced non-finite values")
        metadata: dict[str, Any] = {
            "profiles": str(self.profiles_path.resolve()),
            "profiles_sha256": profiles_digest,
            "profile_count": int(profiles.profile_id.nunique()),
        }
        if self.reference_manifest is not None:
            metadata["reference_manifest"] = str(self.reference_manifest.resolve())
            metadata["reference_manifest_sha256"] = _sha256(self.reference_manifest)
        if self.screening_registry is not None:
            metadata["screening_registry_sha256"] = _verify_artifact(
                self.screening_registry,
                self.screening_registry_sha256,
                "screening registry",
            )
        return profiles.reset_index(drop=True), metadata


def _metric(
    errors: np.ndarray, signed: np.ndarray, count: int
) -> dict[str, float | int]:
    """Calculate pooled metrics from already common-support errors."""
    if count == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "count": 0,
        }
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(signed)),
        "count": int(count),
    }


def score_italian_predictions(
    rows: pd.DataFrame,
    *,
    prediction_columns: Sequence[str] = (
        "model",
        "raw_glorys",
        "corrected_glorys",
        "climatology",
    ),
    invalid_penalty: float = 1.0e6,
    bands_m: Sequence[float] = CANONICAL_BANDS_M,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Score methods on identical observation support, including invalid outputs.

    ``rows`` must contain profile_id, depth_m, observation and one column per
    method.  Invalid predictions remain in denominators and receive a finite
    configurable penalty, preventing a failed reconstruction from disappearing.
    """
    required = {"profile_id", "depth_m", "observation"} | set(prediction_columns)
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError("prediction rows missing columns: " + ", ".join(missing))
    frame = rows.copy()
    frame["depth_m"] = pd.to_numeric(frame["depth_m"], errors="coerce")
    frame["observation"] = pd.to_numeric(frame["observation"], errors="coerce")
    if frame[["depth_m", "observation"]].isna().any().any():
        raise ValueError("prediction rows contain invalid observations")
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    edges = tuple(float(x) for x in bands_m)
    for method in prediction_columns:
        values = pd.to_numeric(frame[method], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        error = np.where(
            valid, values - frame.observation.to_numpy(float), invalid_penalty
        )
        signed = np.where(
            valid, values - frame.observation.to_numpy(float), invalid_penalty
        )
        methods: dict[str, dict[str, float | int]] = {
            "overall": _metric(error, signed, len(frame))
        }
        profile_errors = pd.DataFrame(
            {
                "profile_id": frame.profile_id.astype(str),
                "error": error,
                "signed": signed,
            }
        )
        profile_groups = profile_errors.groupby("profile_id", sort=True)
        profile_mse = profile_groups.error.apply(
            lambda values: float(np.mean(values**2))
        ).to_numpy()
        profile_mae = profile_groups.error.apply(
            lambda values: float(np.mean(np.abs(values)))
        ).to_numpy()
        profile_bias = profile_groups.signed.mean().to_numpy()
        methods["equal_profile"] = {
            "mae": float(np.mean(profile_mae)) if len(profile_groups) else float("nan"),
            "rmse": (
                float(np.sqrt(np.mean(profile_mse)))
                if len(profile_groups)
                else float("nan")
            ),
            "bias": (
                float(np.mean(profile_bias)) if len(profile_groups) else float("nan")
            ),
            "count": int(len(profile_groups)),
        }
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (frame.depth_m.to_numpy(float) >= lower) & (
                frame.depth_m.to_numpy(float) < upper
            )
            band = f"{lower:g}_{upper:g}"
            band_rows = frame.loc[selected]
            band_values = error[selected]
            band_signed = signed[selected]
            band_profile_groups = profile_errors.loc[selected].groupby(
                "profile_id", sort=True
            )
            band_profile_mse = band_profile_groups.error.apply(
                lambda values: float(np.mean(values**2))
            ).to_numpy()
            band_profile_mae = band_profile_groups.error.apply(
                lambda values: float(np.mean(np.abs(values)))
            ).to_numpy()
            band_profile_bias = band_profile_groups.signed.mean().to_numpy()
            methods[band] = _metric(band_values, band_signed, len(band_rows))
            methods[f"{band}_equal_profile"] = {
                "mae": (
                    float(np.mean(band_profile_mae))
                    if len(band_profile_groups)
                    else float("nan")
                ),
                "rmse": (
                    float(np.sqrt(np.mean(band_profile_mse)))
                    if len(band_profile_groups)
                    else float("nan")
                ),
                "bias": (
                    float(np.mean(band_profile_bias))
                    if len(band_profile_groups)
                    else float("nan")
                ),
                "count": int(len(band_profile_groups)),
            }
        result[method] = methods
    for method, metrics in result.items():
        for group, values in metrics.items():
            reference = result.get("raw_glorys", {}).get(group, {}).get("rmse", np.nan)
            values["skill_vs_raw_glorys"] = (
                float(1.0 - values["rmse"] / reference)
                if np.isfinite(reference) and reference > 0
                else float("nan")
            )
    return result


@dataclass
class _ItalianCohort:
    """Prepared independent observations and their masked dataset patches."""

    dataset: ArgoGeoTIFFGriddedPatchDataset
    assignments: pd.DataFrame
    patch_indices: list[int]
    holdouts: set[tuple[int, int, int]]
    exclusions: pd.DataFrame

    def masked_sample(self, index: int) -> dict[str, Any]:
        """Return a patch with Italian observation cells removed from conditioning."""
        sample = dict(self.dataset[int(index)])
        patch = self.dataset._rows.iloc[int(index)]
        for key in ("x", "x_salinity"):
            mask_key = f"{key}_valid_mask"
            if key not in sample or mask_key not in sample:
                continue
            values, valid = sample[key].clone(), sample[mask_key].clone()
            for date, grid_row, grid_col in self.holdouts:
                yy, xx = int(grid_row) - int(patch.grid_y0), int(grid_col) - int(
                    patch.grid_x0
                )
                if (
                    int(patch.date) == int(date)
                    and 0 <= yy < values.shape[-2]
                    and 0 <= xx < values.shape[-1]
                ):
                    values[..., yy, xx] = 0
                    valid[..., yy, xx] = False
            sample[key], sample[mask_key] = values, valid
            one_d_key = f"{mask_key}_1d"
            if one_d_key in sample:
                sample[one_d_key] = valid.bool().any(dim=0, keepdim=True)
        return sample


def _interpolate_depth(
    depths: np.ndarray, value: float, maximum_gap: float
) -> tuple[int, int, float] | None:
    """Map a raw observation depth to one exact or interpolated model level."""
    right = int(np.searchsorted(depths, value, side="left"))
    if right < len(depths) and np.isclose(depths[right], value, rtol=0.0, atol=1e-8):
        return right, right, 0.0
    left = right - 1
    if left < 0 or right >= len(depths) or depths[right] - depths[left] > maximum_gap:
        return None
    return left, right, float((value - depths[left]) / (depths[right] - depths[left]))


def _verify_screening(profiles: pd.DataFrame, config: ItalianValidationConfig) -> None:
    """Require every profile to occur in the frozen Italian screening registry."""
    if config.screening_registry is None:
        return
    registry = pd.read_parquet(config.screening_registry)
    if "datetime_utc" in registry:
        timestamp = pd.to_datetime(registry["datetime_utc"], utc=True, errors="coerce")
        registry = registry.loc[timestamp.dt.year.eq(config.year)].copy()
    if "profile_id" not in registry or registry.profile_id.duplicated().any():
        raise ValueError(
            "Italian screening registry has missing or duplicate profile_id values"
        )
    keys = [
        "profile_id",
        "observation_time",
        "source_file",
        "cruise",
        "station",
        "latitude",
        "longitude",
        "qc_flag",
    ]
    missing = sorted(set(keys) - set(profiles.columns))
    if missing:
        raise ValueError(
            "Italian profiles missing screening identity columns: " + ", ".join(missing)
        )
    observed = profiles[keys].drop_duplicates().copy()
    if observed.empty or observed.profile_id.duplicated().any():
        raise ValueError("Italian profiles have empty or inconsistent identities")
    audit = observed.merge(
        registry,
        on="profile_id",
        how="left",
        suffixes=("", "_registry"),
        validate="one_to_one",
        indicator=True,
    )
    valid = audit["_merge"].eq("both") & pd.to_datetime(
        audit.observation_time, utc=True, format="mixed"
    ).eq(pd.to_datetime(audit.datetime_utc, utc=True, format="mixed"))
    for column in ("source_file", "cruise", "station"):
        valid &= audit[column].astype(str).eq(audit[f"{column}_registry"].astype(str))
    valid &= np.isclose(audit.latitude, audit.latitude_registry, atol=1e-6, rtol=0)
    valid &= np.isclose(audit.longitude, audit.longitude_raw, atol=1e-6, rtol=0)
    valid &= pd.to_numeric(audit.qc_flag).eq(1)
    if not valid.all():
        raise ValueError("Italian profiles do not match the frozen screening registry")


def _prepare_cohort(
    config: ItalianValidationConfig,
) -> tuple[_ItalianCohort, dict[str, Any]]:
    """Build the independent 2016 cohort from packaged profiles and GeoTIFF data."""
    profiles, metadata = config.load_profiles()
    _verify_screening(profiles, config)
    if config.data_config_path is None:
        raise ValueError("Italian validation requires the resolved data_config_path")
    dataset = ArgoGeoTIFFGriddedPatchDataset.from_config(
        config.data_config_path, split="val"
    )
    import rasterio

    with rasterio.open(
        next(iter(dataset.glorys_store.paths_by_date.values()))
    ) as source:
        transform, height, width = source.transform, source.height, source.width
    dates = np.unique(np.asarray(dataset._rows.date, dtype=np.int64))
    timestamps = {int(date): pd.Timestamp(str(int(date)), tz="UTC") for date in dates}
    depths = np.asarray(dataset.depth_axis_m, dtype=float)
    assignments: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    holdout_seed_rows: list[dict[str, Any]] = []
    patches: set[int] = set()
    for profile_id, group in profiles.groupby("profile_id", sort=False):
        point = group.iloc[0]
        for column in (
            "source",
            "cluster_id",
            "observation_time",
            "longitude",
            "latitude",
        ):
            if column not in group:
                raise ValueError(f"Italian profile {profile_id!r} is missing {column}")
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Italian profile {profile_id!r} has inconsistent {column}"
                )
        target_date = min(
            dates.tolist(),
            key=lambda item: abs(timestamps[int(item)] - point.observation_time),
        )
        delta = (
            point.observation_time - timestamps[int(target_date)]
        ).total_seconds() / 86400.0
        if abs(delta) > config.max_time_delta_days:
            exclusions.append(
                {
                    "profile_id": str(profile_id),
                    "reason": "no_target_date_within_tolerance",
                }
            )
            continue
        grid_row, grid_col = rowcol(
            transform, float(point.longitude), float(point.latitude)
        )
        grid_row, grid_col = int(grid_row), int(grid_col)
        if not (0 <= grid_row < height and 0 <= grid_col < width):
            exclusions.append(
                {"profile_id": str(profile_id), "reason": "coordinate_outside_grid"}
            )
            continue
        rows = dataset._rows
        matches = np.flatnonzero(
            (
                (rows.date == target_date)
                & (rows.grid_y0 <= grid_row)
                & (rows.grid_y0 + dataset.tile_size > grid_row)
                & (rows.grid_x0 <= grid_col)
                & (rows.grid_x0 + dataset.tile_size > grid_col)
            ).to_numpy()
        )
        if not len(matches):
            exclusions.append(
                {
                    "profile_id": str(profile_id),
                    "reason": "no_validation_patch_covers_coordinate",
                }
            )
            continue
        patch_index = int(matches[0])
        before = len(assignments)
        for source_row in group.itertuples(index=False):
            if hasattr(source_row, "qc_flag") and int(source_row.qc_flag) != 1:
                continue
            interpolation = _interpolate_depth(
                depths, float(source_row.depth_m), config.max_vertical_gap_m
            )
            if interpolation is None:
                continue
            lower, upper, weight = interpolation
            assignments.append(
                {
                    "profile_id": str(profile_id),
                    "observation": float(source_row.observation),
                    "depth_m": float(source_row.depth_m),
                    "target_date": int(target_date),
                    "time_delta_days": float(delta),
                    "patch_index": patch_index,
                    "row": grid_row - int(rows.iloc[patch_index].grid_y0),
                    "col": grid_col - int(rows.iloc[patch_index].grid_x0),
                    "depth_index_lower": lower,
                    "depth_index_upper": upper,
                    "depth_weight": weight,
                }
            )
        if len(assignments) == before:
            exclusions.append(
                {"profile_id": str(profile_id), "reason": "no_qc_supported_model_depth"}
            )
            continue
        patches.add(patch_index)
        holdout_seed_rows.append(
            {
                "cluster_id": str(point.get("cluster_id", profile_id)),
                "date": int(target_date),
                "grid_row": grid_row,
                "grid_col": grid_col,
            }
        )
    if not assignments:
        raise ValueError("No Italian observations match the 2016 validation patches")
    holdouts = {
        (item["date"], item["grid_row"], item["grid_col"]) for item in holdout_seed_rows
    }
    argo_store = getattr(dataset, "argo_store", None)
    argo_dataset = getattr(argo_store, "ds", None)
    available_cluster = next(
        (
            name
            for name in _CLUSTER_VARIABLES
            if name in set(getattr(argo_dataset, "variables", ()))
        ),
        None,
    )
    cluster_match_count = 0
    if available_cluster is not None:
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
        target_dates = np.asarray(argo_store.target_date)
        for item in holdout_seed_rows:
            matches = np.flatnonzero(cluster_values == item["cluster_id"])
            matches = matches[target_dates[matches] == item["date"]]
            cluster_match_count += len(matches)
            holdouts.update(
                (
                    int(item["date"]),
                    int(argo_store.grid_row[index]),
                    int(argo_store.grid_col[index]),
                )
                for index in matches.tolist()
            )
    if config.buffer_cells < 0:
        raise ValueError("buffer_cells cannot be negative")
    if config.buffer_cells:
        holdouts = {
            (date, row + dy, col + dx)
            for date, row, col in holdouts
            for dy in range(-config.buffer_cells, config.buffer_cells + 1)
            for dx in range(-config.buffer_cells, config.buffer_cells + 1)
        }
    metadata.update(
        eligible_profiles=len({row["profile_id"] for row in assignments}),
        patch_count=len(patches),
        cluster_metadata_variable=available_cluster,
        cluster_match_count=int(cluster_match_count),
        conditioning_exclusion_verified=available_cluster is not None
        and cluster_match_count > 0,
    )
    return (
        _ItalianCohort(
            dataset,
            pd.DataFrame(assignments),
            sorted(patches),
            holdouts,
            pd.DataFrame(exclusions),
        ),
        metadata,
    )


class ItalianValidationCallback(pl.Callback):
    """Run fixed DDPM Italian validation once per real validation epoch."""

    def __init__(
        self,
        config: ItalianValidationConfig,
        evaluator: Callable[[Any, pd.DataFrame, int, int], pd.DataFrame] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize a fixed external cohort evaluator for a training run."""
        super().__init__()
        self.config = config
        self.evaluator = evaluator
        if evaluator is None:
            self.cohort, self.metadata = _prepare_cohort(config)
            self.profiles = self.cohort.assignments
        else:
            # The compatibility seam is deliberately limited to offline tests/tools;
            # production construction always prepares real masked patch batches.
            self.cohort = None
            self.profiles, self.metadata = config.load_profiles()
        self.last_evaluated_epoch: int | None = None
        self.patch_count = 0 if self.cohort is None else len(self.cohort.patch_indices)
        self.ema_evaluation_active = False
        self.output_dir = None if output_dir is None else Path(output_dir)
        # Object collectives must use CPU transport: an early rank entering an
        # NCCL object gather can occupy its GPU while another rank is still sampling.
        self._object_collective_group: Any | None = None

    def state_dict(self) -> dict[str, Any]:
        """Persist callback progress for checkpoint resume."""
        return {"last_evaluated_epoch": self.last_evaluated_epoch}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore callback progress without restoring model state."""
        value = state_dict.get("last_evaluated_epoch")
        self.last_evaluated_epoch = None if value is None else int(value)

    def _ensure_object_collective_group(self) -> Any | None:
        """Create one cached Gloo group collectively before uneven GPU work."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return None
        if self._object_collective_group is None:
            # new_group is collective over the default group, so every rank calls
            # this before explicit batch sharding lets ranks finish at different times.
            self._object_collective_group = torch.distributed.new_group(backend="gloo")
        return self._object_collective_group

    @staticmethod
    def _rank_indices(length: int) -> list[int]:
        """Shard explicit batches across ranks without sampler padding."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return list(range(length))
        return list(
            range(
                torch.distributed.get_rank(), length, torch.distributed.get_world_size()
            )
        )

    @staticmethod
    def _point_values(tensor: torch.Tensor, points: pd.DataFrame) -> np.ndarray:
        """Interpolate a physical patch tensor to raw profile depths."""
        values = tensor.detach().float().cpu().numpy()
        result = np.full(len(points), np.nan, dtype=np.float64)
        for offset, point in enumerate(points.itertuples(index=False)):
            result[offset] = (1.0 - float(point.depth_weight)) * values[
                int(point.batch_index),
                int(point.depth_index_lower),
                int(point.row),
                int(point.col),
            ] + float(point.depth_weight) * values[
                int(point.batch_index),
                int(point.depth_index_upper),
                int(point.row),
                int(point.col),
            ]
        return result

    @staticmethod
    def _point_support(mask: torch.Tensor, points: pd.DataFrame) -> np.ndarray:
        """Require both bracketing model levels on an interpolated observation."""
        values = mask.detach().bool().cpu().numpy()
        supported = np.zeros(len(points), dtype=bool)
        for offset, point in enumerate(points.itertuples(index=False)):
            batch_index, row, col = (
                int(point.batch_index),
                int(point.row),
                int(point.col),
            )
            lower, upper = int(point.depth_index_lower), int(point.depth_index_upper)
            supported[offset] = bool(
                values[batch_index, lower, row, col]
                and values[batch_index, upper, row, col]
            )
        return supported

    def _batches(self):
        """Yield fixed-size masked batches with common-support reference columns."""
        if self.cohort is None:
            raise RuntimeError("No Italian cohort was prepared for tensor evaluation")
        for start in range(0, len(self.cohort.patch_indices), self.config.batch_size):
            global_batch_index = start // self.config.batch_size
            indices = self.cohort.patch_indices[start : start + self.config.batch_size]
            points = self.cohort.assignments.loc[
                self.cohort.assignments.patch_index.isin(indices)
            ].copy()
            points["batch_index"] = points.patch_index.map(
                {value: index for index, value in enumerate(indices)}
            )
            batch = default_collate(
                [self.cohort.masked_sample(index) for index in indices]
            )
            support = batch["x_valid_mask"].bool().any(dim=1).sum(dim=(1, 2))
            points = points.loc[
                points.batch_index.map(
                    lambda index: int(support[int(index)])
                    >= self.config.min_input_profiles
                )
            ].copy()
            if points.empty:
                continue
            reference_mask = batch.get("y_glorys_valid_mask", batch.get("y_valid_mask"))
            if reference_mask is None:
                raise ValueError(
                    "Italian validation batch lacks a GLORYS validity mask"
                )
            common_support = self._point_support(reference_mask, points)
            if "land_mask" in batch:
                land = batch["land_mask"].bool()
                common_support &= np.asarray(
                    [
                        bool(
                            land[
                                int(point.batch_index),
                                0,
                                int(point.row),
                                int(point.col),
                            ]
                        )
                        for point in points.itertuples(index=False)
                    ]
                )
            points = points.loc[common_support].copy()
            if points.empty:
                continue
            for name, key in (
                ("corrected_glorys", "y"),
                ("raw_glorys", "y_glorys"),
                ("climatology", "climatology"),
            ):
                if key not in batch:
                    raise ValueError(f"Italian validation batch lacks {key!r}")
                points[name] = self._point_values(
                    temperature_normalize("denorm", batch[key]), points
                )
            finite_reference = np.isfinite(
                points[["raw_glorys", "corrected_glorys", "climatology"]].to_numpy(
                    float
                )
            ).all(axis=1)
            if finite_reference.any():
                yield global_batch_index, batch, points.loc[
                    finite_reference
                ].reset_index(drop=True)

    @staticmethod
    def _model_device(model: Any) -> torch.device:
        """Resolve the device used by the Lightning module without assumptions."""
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _evaluate_real_model(
        self, model: Any, trainer: Any, object_collective_group: Any | None
    ) -> pd.DataFrame:
        """Run the regular tensor prediction contract over the independent cohort."""
        batches = list(self._batches())
        records: list[pd.DataFrame] = []
        device = self._model_device(model)
        for index in self._rank_indices(len(batches)):
            global_batch_index, batch, points = batches[index]
            # Explicitly pin the full DDPM sampler; ordinary validation may use DDIM.
            batch["sampler"] = self._italian_sampler
            # This seed is a function of the unsharded cohort index, so a patch
            # receives identical DDPM noise regardless of the DDP world size.
            seed = self.config.seed + int(global_batch_index) * 100_000
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            precision_plugin = getattr(trainer, "precision_plugin", None)
            forward_context = getattr(precision_plugin, "forward_context", None)
            # Callback-invoked predict_step is outside Lightning's automatic loop
            # context, so explicitly preserve the configured mixed precision.
            with forward_context() if callable(forward_context) else nullcontext():
                result = model.predict_step(device_batch, batch_idx=index)
            if (
                not isinstance(result, Mapping)
                or "y_hat_temperature_denorm" not in result
            ):
                raise TypeError(
                    "Italian validation requires predict_step() to return y_hat_temperature_denorm"
                )
            points = points.copy()
            points["model"] = self._point_values(
                result["y_hat_temperature_denorm"], points
            )
            records.append(points)
        local = (
            pd.concat(records, ignore_index=True)
            if records
            else pd.DataFrame(
                columns=[
                    "profile_id",
                    "depth_m",
                    "observation",
                    "model",
                    "raw_glorys",
                    "corrected_glorys",
                    "climatology",
                ]
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[pd.DataFrame | None] = [
                None
            ] * torch.distributed.get_world_size(group=object_collective_group)
            torch.distributed.all_gather_object(
                gathered, local, group=object_collective_group
            )
            return pd.concat(
                [item for item in gathered if item is not None], ignore_index=True
            )
        return local

    def _log(self, trainer: Any, metrics: Mapping[str, Any]) -> None:
        """Log scalar metrics once through the trainer's rank-zero logger."""
        payload: dict[str, float] = {}
        failures = 0
        for name, value in metrics.items():
            if not np.isscalar(value):
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                payload[f"italian_validation/{name}"] = numeric
            else:
                failures += 1
        if failures:
            payload["italian_validation/metric_failure"] = 1.0
            payload["italian_validation/nonfinite_metrics"] = float(failures)
        rank_zero = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if (
            rank_zero
            and hasattr(trainer, "logger")
            and trainer.logger is not None
            and payload
        ):
            trainer.logger.log_metrics(
                payload, step=getattr(trainer, "global_step", None)
            )

    def _write_artifact(
        self, trainer: Any, *, epoch: int, rows: pd.DataFrame, scores: Mapping[str, Any]
    ) -> None:
        """Persist auditable cohort coverage and metrics alongside the training run."""
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        # Trainer.log_dir broadcasts through the distributed strategy and cannot
        # be read after nonzero ranks return above. Use the non-collective root.
        root = self.output_dir or getattr(trainer, "default_root_dir", None)
        if not root:
            return

        def json_value(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): json_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            if isinstance(value, (np.floating, float)):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, np.integer):
                return int(value)
            return value

        destination = Path(root) / "italian_validation"
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "ddpm_steps": int(self.config.ddpm_steps),
            "profile_count_prepared": int(self.metadata.get("profile_count", 0)),
            "eligible_profiles": int(self.profiles.profile_id.nunique()),
            "scored_profiles": int(rows.profile_id.nunique()),
            "scored_points": int(len(rows)),
            "patch_count": int(self.patch_count),
            "ema_evaluation_active": bool(self.ema_evaluation_active),
            "excluded_profiles": (
                int(len(self.cohort.exclusions)) if self.cohort is not None else 0
            ),
            "provenance": self.metadata,
            "scores": scores,
        }
        (destination / f"epoch_{epoch:04d}.json").write_text(
            json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _assert_ema_is_active(self, trainer: Any) -> None:
        """Verify that configured EMA validation weights have been installed."""
        callbacks = getattr(trainer, "callbacks", ())
        for callback in callbacks:
            if callback.__class__.__name__ != "EMA":
                continue
            if not bool(getattr(callback, "evaluate_ema_weights_instead", False)):
                continue
            if not bool(getattr(callback, "ema_initialized", False)):
                raise RuntimeError(
                    "Italian validation requires initialized EMA weights"
                )
            if not bool(getattr(callback, "weights_are_applied", False)):
                raise RuntimeError("Italian validation requires active EMA weights")
            self.ema_evaluation_active = True
            return

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Evaluate after sanity checks, at most once for each epoch."""
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(getattr(trainer, "current_epoch", -1))
        if self.last_evaluated_epoch == epoch:
            return
        self._assert_ema_is_active(trainer)
        # Group creation itself is collective. Complete it on every rank before
        # explicit sharding produces different amounts of DDPM sampling work.
        object_collective_group = self._ensure_object_collective_group()
        rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        device = self._model_device(pl_module)
        cuda_rng_state = (
            torch.cuda.get_rng_state(device=device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        previous_sampler = getattr(pl_module, "val_sampler", None)
        sampler_swapped = False
        try:
            torch.random.default_generator.manual_seed(self.config.seed)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(self.config.seed)
            if self.evaluator is not None:
                rows = self.evaluator(
                    pl_module,
                    self.profiles.copy(),
                    self.config.ddpm_steps,
                    self.config.seed,
                )
            else:
                self._italian_sampler = getattr(
                    getattr(pl_module, "model", None), "sampler", None
                )
                if self._italian_sampler is None:
                    self._italian_sampler = previous_sampler
                if (
                    int(getattr(self._italian_sampler, "num_timesteps", -1))
                    != self.config.ddpm_steps
                ):
                    raise ValueError(
                        "Italian validation requires a DDPM sampler with 1000 timesteps"
                    )
                # predict_step defaults to val_sampler, so swap only for this isolated pass.
                pl_module.val_sampler = self._italian_sampler
                sampler_swapped = True
                rows = self._evaluate_real_model(
                    pl_module, trainer, object_collective_group
                )
        finally:
            torch.random.set_rng_state(rng_state)
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=device)
            if sampler_swapped:
                pl_module.val_sampler = previous_sampler
        scored = score_italian_predictions(
            rows,
            invalid_penalty=self.config.invalid_prediction_penalty,
            bands_m=self.config.depth_bands_m,
        )
        flat: dict[str, Any] = {
            "prepared_profiles": int(self.metadata.get("profile_count", 0)),
            "eligible_profiles": int(self.profiles.profile_id.nunique()),
            "scored_profiles": int(rows.profile_id.nunique()),
            "scored_points": len(rows),
            "patch_count": self.patch_count,
        }
        for method, groups in scored.items():
            for group, values in groups.items():
                for metric in ("mae", "rmse", "bias", "count", "skill_vs_raw_glorys"):
                    flat[f"{method}/{group}/{metric}"] = values[metric]
        self._log(trainer, flat)
        self._write_artifact(trainer, epoch=epoch, rows=rows, scores=scored)
        self.last_evaluated_epoch = epoch


def build_italian_validation_callback(
    config: Mapping[str, Any],
    evaluator: Callable[[Any, pd.DataFrame, int, int], pd.DataFrame] | None = None,
    *,
    data_config: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    data_config_path: str | Path | None = None,
) -> ItalianValidationCallback | None:
    """Build the callback from training config, returning ``None`` when disabled."""
    settings = training_config or config.get("training", config)
    if isinstance(settings, Mapping) and isinstance(settings.get("training"), Mapping):
        settings = settings["training"]
    raw = (
        settings.get("italian_validation", {}) if isinstance(settings, Mapping) else {}
    )
    if not raw or not bool(raw.get("enabled", False)):
        return None
    if data_config:
        raw = {**data_config, **raw}
    if data_config_path is not None:
        raw = {**raw, "data_config_path": str(data_config_path)}
    return ItalianValidationCallback(
        ItalianValidationConfig.from_mapping(raw), evaluator
    )
