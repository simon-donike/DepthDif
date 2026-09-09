from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from profile_audit.common import haversine_km, normalize_identifier, read_table, write_table


@dataclass(frozen=True)
class MatchThresholds:
    primary_minutes: float = 10
    primary_km: float = 5
    recovery_minutes: float = 60
    recovery_km: float = 10
    broad_hours: float = 24
    broad_km: float = 25
    near_exact_min_levels: int = 30
    near_exact_mae_c: float = 0.15
    probable_min_levels: int = 15
    probable_correlation: float = 0.95
    probable_mae_c: float = 0.5
    ambiguity_score_delta: float = 0.05


IDENTIFIER_COLUMNS = (
    "dc_reference", "platform_code", "ncei_accession", "replacement_accession",
    "wod_id", "argo_id", "instrument_reference", "normalized_cruise_station",
)


def _times(frame: pd.DataFrame) -> pd.Series:
    if "datetime_utc" in frame:
        return pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
    if "profile_juld" in frame:
        return pd.to_datetime(frame["profile_juld"], unit="D", origin="1950-01-01", utc=True, errors="coerce")
    if "profile_date" in frame:
        return pd.to_datetime(frame["profile_date"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    raise ValueError("Profile table needs datetime_utc, profile_juld, or profile_date.")


def _longitude_column(frame: pd.DataFrame) -> str:
    for name in ("longitude", "longitude_180", "longitude_raw"):
        if name in frame:
            return name
    raise ValueError("Profile table has no longitude column.")


def _observation_groups(observations: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if observations is None or observations.empty:
        return {}
    return {str(key): group for key, group in observations.groupby("profile_id", sort=False)}


def _clean_curve(group: pd.DataFrame, depth_column: str, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    if depth_column not in group or value_column not in group:
        return np.array([]), np.array([])
    depth = pd.to_numeric(group[depth_column], errors="coerce").to_numpy(float)
    value = pd.to_numeric(group[value_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(depth) & np.isfinite(value) & (depth >= 0)
    if not np.any(valid):
        return np.array([]), np.array([])
    frame = pd.DataFrame({"depth": depth[valid], "value": value[valid]}).groupby("depth", as_index=False).mean().sort_values("depth")
    return frame["depth"].to_numpy(), frame["value"].to_numpy()


def profile_fingerprint(target: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    candidate_depth, candidate_temp = _clean_curve(candidate, "depth_m", "temperature_c")
    variants = (("corrected", "depth_3", "temperature_2"), ("hanawa", "depth_2", "temperature_1"), ("raw", "depth_1", "temperature_1"), ("standard", "depth_m", "temperature_c"))
    best: dict[str, Any] | None = None
    for variant, depth_name, temp_name in variants:
        target_depth, target_temp = _clean_curve(target, depth_name, temp_name)
        if target_depth.size == 0 or candidate_depth.size == 0:
            continue
        inside = (target_depth >= candidate_depth.min()) & (target_depth <= candidate_depth.max())
        if not np.any(inside):
            continue
        depths = target_depth[inside]
        observed = target_temp[inside]
        interpolated = np.interp(depths, candidate_depth, candidate_temp)
        residual = observed - interpolated
        nearest = np.min(np.abs(depths[:, None] - candidate_depth[None, :]), axis=1)
        count = len(depths)
        correlation = float(np.corrcoef(observed, interpolated)[0, 1]) if count >= 2 and np.std(observed) > 0 and np.std(interpolated) > 0 else np.nan
        gradient_correlation = float(np.corrcoef(np.diff(observed), np.diff(interpolated))[0, 1]) if count >= 3 and np.std(np.diff(observed)) > 0 and np.std(np.diff(interpolated)) > 0 else np.nan
        result = {
            "profile_variant": variant,
            "overlap_levels": count,
            "overlap_fraction": count / max(len(target_depth), 1),
            "median_depth_difference_m": float(np.median(nearest)),
            "maximum_depth_difference_m": float(np.max(nearest)),
            "temperature_mae_c": float(np.mean(np.abs(residual))),
            "temperature_rmse_c": float(np.sqrt(np.mean(residual**2))),
            "temperature_correlation": correlation,
            "temperature_mean_offset_c": float(np.mean(residual)),
            "temperature_gradient_correlation": gradient_correlation,
        }
        if best is None or result["temperature_mae_c"] < best["temperature_mae_c"]:
            best = result
    return best or {
        "profile_variant": None, "overlap_levels": 0, "overlap_fraction": 0.0,
        "median_depth_difference_m": np.nan, "maximum_depth_difference_m": np.nan,
        "temperature_mae_c": np.nan, "temperature_rmse_c": np.nan,
        "temperature_correlation": np.nan, "temperature_mean_offset_c": np.nan,
        "temperature_gradient_correlation": np.nan,
    }


def _exact_pairs(targets: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    pairs: dict[str, list[tuple[str, str]]] = {}
    if {"profile_source_file", "source_profile_idx"}.issubset(targets) and {"source_file", "source_profile_index"}.issubset(candidates):
        right = candidates[["profile_id", "source_file", "source_profile_index"]].copy()
        merged = targets[["profile_id", "profile_source_file", "source_profile_idx"]].merge(
            right, left_on=["profile_source_file", "source_profile_idx"], right_on=["source_file", "source_profile_index"]
        )
        for row in merged.itertuples(index=False):
            pairs.setdefault(str(row.profile_id_x), []).append((str(row.profile_id_y), "source_file_index"))
    for column in IDENTIFIER_COLUMNS:
        if column not in targets or column not in candidates:
            continue
        left = targets[["profile_id", column]].copy()
        right = candidates[["profile_id", column]].copy()
        left["key"] = left[column].map(normalize_identifier)
        right["key"] = right[column].map(normalize_identifier)
        merged = left[left["key"] != ""].merge(right[right["key"] != ""], on="key")
        for row in merged.itertuples(index=False):
            pairs.setdefault(str(row.profile_id_x), []).append((str(row.profile_id_y), column))
    return pairs


def _classification(row: dict[str, Any], thresholds: MatchThresholds, exact: bool) -> str:
    if exact:
        return "Exact"
    if row["time_difference_minutes"] <= thresholds.primary_minutes and row["distance_km"] <= thresholds.primary_km and row["overlap_levels"] >= thresholds.near_exact_min_levels and row["temperature_mae_c"] <= thresholds.near_exact_mae_c:
        return "Near-exact"
    if row["overlap_levels"] >= thresholds.probable_min_levels and row["temperature_mae_c"] <= thresholds.probable_mae_c and row["temperature_correlation"] >= thresholds.probable_correlation:
        return "Probable duplicate"
    return "Candidate"


def _score(row: dict[str, Any], thresholds: MatchThresholds, exact: bool) -> float:
    if exact:
        return 0.0
    mae = row["temperature_mae_c"] if np.isfinite(row["temperature_mae_c"]) else 2.0
    correlation = row["temperature_correlation"] if np.isfinite(row["temperature_correlation"]) else 0.0
    return float(0.25 * min(row["time_difference_minutes"] / (thresholds.broad_hours * 60), 1) + 0.25 * min(row["distance_km"] / thresholds.broad_km, 1) + 0.35 * min(mae / 2, 1) + 0.15 * (1 - np.clip(correlation, -1, 1)) / 2)


def calibrate_profile_thresholds(positive_metrics: pd.DataFrame, negative_metrics: pd.DataFrame, base: MatchThresholds = MatchThresholds()) -> MatchThresholds:
    """Choose conservative fingerprint cutoffs between control distributions."""
    positive_mae = pd.to_numeric(positive_metrics["temperature_mae_c"], errors="coerce").dropna()
    negative_mae = pd.to_numeric(negative_metrics["temperature_mae_c"], errors="coerce").dropna()
    positive_corr = pd.to_numeric(positive_metrics["temperature_correlation"], errors="coerce").dropna()
    negative_corr = pd.to_numeric(negative_metrics["temperature_correlation"], errors="coerce").dropna()
    if min(len(positive_mae), len(negative_mae), len(positive_corr), len(negative_corr)) == 0:
        raise ValueError("Positive and negative controls need finite MAE and correlation values.")
    mae_cutoff = float((positive_mae.quantile(0.95) + negative_mae.quantile(0.05)) / 2)
    correlation_cutoff = float((positive_corr.quantile(0.05) + negative_corr.quantile(0.95)) / 2)
    return MatchThresholds(
        **{
            **base.__dict__,
            "near_exact_mae_c": min(base.near_exact_mae_c, mae_cutoff),
            "probable_mae_c": mae_cutoff,
            "probable_correlation": correlation_cutoff,
        }
    )


def match_profiles(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    target_observations: pd.DataFrame | None = None,
    candidate_observations: pd.DataFrame | None = None,
    thresholds: MatchThresholds = MatchThresholds(),
) -> pd.DataFrame:
    targets = targets.copy()
    candidates = candidates.copy()
    targets["_time"] = _times(targets)
    candidates["_time"] = _times(candidates)
    target_lon, candidate_lon = _longitude_column(targets), _longitude_column(candidates)
    target_groups = _observation_groups(target_observations)
    candidate_groups = _observation_groups(candidate_observations)
    exact = _exact_pairs(targets, candidates)
    candidates_by_id = candidates.set_index("profile_id", drop=False)
    candidate_days = {day: group for day, group in candidates.groupby(candidates["_time"].dt.floor("D"), sort=False)}
    results: list[dict[str, Any]] = []

    for _, target in targets.iterrows():
        target_dict = target.to_dict()
        target_id = str(target_dict["profile_id"])
        pair_reasons = exact.get(target_id, [])
        candidate_map = {candidate_id: reason for candidate_id, reason in pair_reasons}
        if not candidate_map and pd.notna(target_dict["_time"]):
            day = target_dict["_time"].floor("D")
            nearby = pd.concat([candidate_days.get(day + pd.Timedelta(days=offset), candidates.iloc[0:0]) for offset in (-1, 0, 1)], ignore_index=True)
            if not nearby.empty:
                minutes = (nearby["_time"] - target_dict["_time"]).abs().dt.total_seconds() / 60
                distance = haversine_km(target_dict["latitude"], target_dict[target_lon], nearby["latitude"], nearby[candidate_lon])
                mask = (minutes <= thresholds.broad_hours * 60) & (distance <= thresholds.broad_km)
                candidate_map.update({str(profile_id): "spatiotemporal" for profile_id in nearby.loc[mask, "profile_id"]})
        if not candidate_map:
            results.append({"target_profile_id": target_id, "candidate_profile_id": None, "classification": "No archive match", "rank": 1, "match_score": np.nan, "exact_identifier": None})
            continue
        ranked = []
        for candidate_id, reason in candidate_map.items():
            candidate = candidates_by_id.loc[candidate_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            minutes = abs((candidate["_time"] - target_dict["_time"]).total_seconds()) / 60 if pd.notna(candidate["_time"]) and pd.notna(target_dict["_time"]) else np.nan
            distance = float(haversine_km(target_dict["latitude"], target_dict[target_lon], candidate["latitude"], candidate[candidate_lon]))
            fingerprint = profile_fingerprint(target_groups.get(target_id, pd.DataFrame()), candidate_groups.get(candidate_id, pd.DataFrame()))
            record = {
                "target_profile_id": target_id, "candidate_profile_id": candidate_id,
                "archive_version": candidate.get("archive_version"), "candidate_source_file": candidate.get("source_file"),
                "candidate_source_profile_index": candidate.get("source_profile_index"),
                "candidate_dc_reference": candidate.get("dc_reference"), "exact_identifier": reason if reason != "spatiotemporal" else None,
                "time_difference_minutes": float(minutes), "distance_km": distance, **fingerprint,
            }
            if reason != "spatiotemporal":
                record["candidate_window"] = "exact_identifier"
            elif minutes <= thresholds.primary_minutes and distance <= thresholds.primary_km:
                record["candidate_window"] = "primary"
            elif minutes <= thresholds.recovery_minutes and distance <= thresholds.recovery_km:
                record["candidate_window"] = "recovery"
            else:
                record["candidate_window"] = "broad_duplicate"
            record["classification"] = _classification(record, thresholds, exact=reason != "spatiotemporal")
            record["match_score"] = _score(record, thresholds, exact=reason != "spatiotemporal")
            ranked.append(record)
        ranked.sort(key=lambda item: item["match_score"])
        if len(ranked) > 1 and ranked[1]["match_score"] - ranked[0]["match_score"] <= thresholds.ambiguity_score_delta:
            ranked[0]["classification"] = "Ambiguous"
        for rank, record in enumerate(ranked, 1):
            record["rank"] = rank
            results.append(record)
    return pd.DataFrame.from_records(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match target profiles to frozen assimilation inputs.")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--target-observations", type=Path)
    parser.add_argument("--candidate-observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = match_profiles(read_table(args.targets), read_table(args.candidates), target_observations=read_table(args.target_observations) if args.target_observations else None, candidate_observations=read_table(args.candidate_observations) if args.candidate_observations else None)
    print(write_table(result, args.output))


if __name__ == "__main__":
    main()
