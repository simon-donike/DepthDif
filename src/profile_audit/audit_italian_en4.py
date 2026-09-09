from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from profile_audit.common import haversine_km, read_table, valid_coordinates, write_table
from profile_audit.match_profiles import match_profiles


EN4_START = pd.Timestamp("2000-01-01", tz="UTC")
BROAD_HOURS = 24.0
BROAD_KM = 25.0
STRONG_MATCHES = ("Exact", "Near-exact", "Probable duplicate")
REVIEW_MATCHES = ("Ambiguous", "Candidate")


def _times(frame: pd.DataFrame) -> pd.Series:
    if "datetime_utc" in frame:
        return pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
    if "profile_juld" in frame:
        return pd.to_datetime(frame["profile_juld"], unit="D", origin="1950-01-01", utc=True, errors="coerce")
    raise ValueError("EN4 index needs datetime_utc or profile_juld.")


def _target_candidates(targets: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Return all EN4 metadata candidates in the broad audit window."""
    required = {"profile_source_file", "profile_idx", "latitude", "longitude"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"EN4 index is missing columns: {sorted(missing)}")
    targets = targets.copy()
    if "longitude" not in targets:
        for longitude_name in ("longitude_180", "longitude_raw"):
            if longitude_name in targets:
                targets["longitude"] = targets[longitude_name]
                break
    if "longitude" not in targets:
        raise ValueError("Targets need longitude, longitude_180, or longitude_raw.")
    candidates = index.copy()
    candidates["datetime_utc"] = _times(candidates)
    candidates = candidates.loc[
        candidates["datetime_utc"].notna()
        & valid_coordinates(candidates["latitude"], candidates["longitude"])
    ].copy()
    candidates["profile_id"] = (
        candidates["profile_source_file"].astype(str)
        + ":"
        + candidates["profile_idx"].astype(str)
    )
    candidates["source_file"] = candidates["profile_source_file"].astype(str)
    candidates["source_profile_index"] = candidates["profile_idx"].astype("int64")
    candidates["archive_version"] = "EN4.2.2"
    candidates = candidates.sort_values("datetime_utc", kind="stable").reset_index(drop=True)
    candidate_ns = candidates["datetime_utc"].astype("int64").to_numpy()

    target_times = _times(targets)
    pairs: list[pd.DataFrame] = []
    for target, target_time in zip(targets.itertuples(index=False), target_times):
        if pd.isna(target_time):
            continue
        target_ns = target_time.value
        left = int(np.searchsorted(candidate_ns, target_ns - int(BROAD_HOURS * 3600 * 1e9), side="left"))
        right = int(np.searchsorted(candidate_ns, target_ns + int(BROAD_HOURS * 3600 * 1e9), side="right"))
        window = candidates.iloc[left:right].copy()
        if window.empty:
            continue
        minutes = (window["datetime_utc"] - target_time).abs().dt.total_seconds() / 60.0
        distance = haversine_km(
            float(target.latitude),
            float(target.longitude),
            window["latitude"].to_numpy(),
            window["longitude"].to_numpy(),
        )
        selected = window.loc[(minutes <= BROAD_HOURS * 60.0) & (distance <= BROAD_KM)].copy()
        if not selected.empty:
            selected["target_profile_id"] = str(target.profile_id)
            selected["candidate_time_difference_minutes"] = minutes.loc[selected.index].to_numpy()
            selected["candidate_distance_km"] = pd.Series(distance, index=window.index).loc[selected.index].to_numpy()
            pairs.append(selected)
    if not pairs:
        return pd.DataFrame(columns=["target_profile_id", "profile_id"])
    return pd.concat(pairs, ignore_index=True).drop_duplicates(["target_profile_id", "profile_id"])


def _extract_oceandepths_candidates(store: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), pd.DataFrame(columns=["profile_id", "depth_m", "temperature_c"])
    with xr.open_zarr(store, consolidated=None) as ds:
        available = set(ds.variables)
        required = {"profile_source_file", "profile_idx", "latitude", "longitude", "profile_juld"}
        missing = required - available
        if missing:
            raise ValueError(f"OceanDepths store is missing variables: {sorted(missing)}")
        source_files = np.asarray(ds["profile_source_file"].values).astype(str)
        source_indices = np.asarray(ds["profile_idx"].values, dtype=np.int64)
        keys = pd.MultiIndex.from_arrays([source_files, source_indices])
        requested_keys = pd.MultiIndex.from_frame(candidates[["profile_source_file", "profile_idx"]])
        positions = keys.get_indexer(requested_keys)
        if np.any(positions < 0):
            missing_count = int(np.count_nonzero(positions < 0))
            raise ValueError(f"{missing_count} EN4 candidates are absent from the OceanDepths store.")
        selected = ds.isel(profile=positions)
        records = candidates.copy().reset_index(drop=True)
        records["profile_id"] = records["profile_id"].astype(str)
        observations: list[dict[str, object]] = []
        if "argo_temp_on_glorys_depth" in selected and "glorys_depth" in selected:
            depth_values = np.asarray(selected["glorys_depth"].values, dtype=float)
            if depth_values.ndim == 1:
                depth = np.broadcast_to(depth_values, (len(records), depth_values.size))
            else:
                depth = depth_values.reshape(len(records), -1)
            temperature = np.asarray(selected["argo_temp_on_glorys_depth"].values, dtype=float).reshape(len(records), -1)
            for row_index, profile_id in enumerate(records["profile_id"]):
                valid = np.isfinite(depth[row_index]) & np.isfinite(temperature[row_index]) & (depth[row_index] >= 0)
                observations.extend(
                    {
                        "profile_id": profile_id,
                        "level_index": int(level),
                        "depth_m": float(depth[row_index, level]),
                        "temperature_c": float(temperature[row_index, level]),
                    }
                    for level in np.flatnonzero(valid)
                )
        return records, pd.DataFrame.from_records(observations)


def audit_italian_en4(
    targets_path: Path,
    observations_path: Path,
    en4_index_path: Path,
    oceandepths_root: Path,
    matches_output: Path,
    summary_output: Path,
) -> tuple[Path, Path]:
    targets = read_table(targets_path).copy()
    observations = read_table(observations_path)
    targets["datetime_utc"] = _times(targets)
    targets["en4_coverage_status"] = np.where(
        targets["datetime_utc"] >= EN4_START,
        "within_oceandepths_en4_snapshot",
        "outside_oceandepths_en4_snapshot",
    )
    covered = targets.loc[targets["en4_coverage_status"] == "within_oceandepths_en4_snapshot"].copy()
    available_index_columns = set(pq.ParquetFile(en4_index_path).schema_arrow.names)
    index_columns = [
        column for column in
        ("profile_source_file", "profile_idx", "latitude", "longitude", "datetime_utc", "profile_juld")
        if column in available_index_columns
    ]
    index = pd.read_parquet(en4_index_path, columns=index_columns)
    candidates = _target_candidates(covered, index)
    candidate_profiles, candidate_observations = _extract_oceandepths_candidates(
        Path(oceandepths_root) / "data/argo_glors_ostia_ssh.zarr" if Path(oceandepths_root).suffix != ".zarr" else Path(oceandepths_root),
        candidates,
    )
    covered_observations = observations[observations["profile_id"].isin(set(covered["profile_id"]))]
    if candidate_profiles.empty:
        matches = pd.DataFrame({
            "target_profile_id": covered["profile_id"].astype(str),
            "candidate_profile_id": None,
            "classification": "No archive match",
            "rank": 1,
            "match_score": np.nan,
            "exact_identifier": None,
        })
    else:
        matches = match_profiles(
            covered,
            candidate_profiles,
            target_observations=covered_observations,
            candidate_observations=candidate_observations,
        )
    matches["evidence_scope"] = "EN4.2.2_OceanDepths_snapshot"
    matches["en4_coverage_status"] = "within_oceandepths_en4_snapshot"
    outside = targets.loc[targets["en4_coverage_status"] != "within_oceandepths_en4_snapshot", ["profile_id"]].copy()
    outside["target_profile_id"] = outside["profile_id"]
    outside["classification"] = "Outside EN4 snapshot coverage"
    outside["evidence_scope"] = "EN4.2.2_OceanDepths_snapshot"
    outside["en4_coverage_status"] = "outside_oceandepths_en4_snapshot"
    outside["candidate_profile_id"] = None
    outside["rank"] = 1
    matches = pd.concat([matches, outside.drop(columns=["profile_id"])], ignore_index=True, sort=False)
    write_table(matches, matches_output)

    best = matches.sort_values(["target_profile_id", "rank"]).drop_duplicates("target_profile_id")
    candidate_counts = matches.groupby("target_profile_id")["candidate_profile_id"].count().rename("candidate_count")
    summary = targets.merge(best, left_on="profile_id", right_on="target_profile_id", how="left")
    for column in ("en4_coverage_status",):
        suffixed = f"{column}_x"
        if suffixed in summary:
            summary = summary.rename(columns={suffixed: column})
            summary = summary.drop(columns=[f"{column}_y"], errors="ignore")
    summary = summary.merge(candidate_counts, left_on="profile_id", right_index=True, how="left")
    summary["candidate_count"] = summary["candidate_count"].fillna(0).astype(int)
    write_table(summary, summary_output)
    return Path(matches_output), Path(summary_output)


def format_en4_audit_summary(summary: pd.DataFrame) -> str:
    """Format the best-match table as a concise training-provenance result."""
    classifications = summary["classification"].value_counts()
    coverage = summary["en4_coverage_status"].value_counts()
    strong = int(classifications.reindex(STRONG_MATCHES, fill_value=0).sum())
    review = int(classifications.reindex(REVIEW_MATCHES, fill_value=0).sum())
    no_match = int(classifications.get("No archive match", 0))
    outside = int(coverage.get("outside_oceandepths_en4_snapshot", 0))
    candidate_pairs = int(summary["candidate_count"].sum())
    lines = [
        "EN4.2.2 training-provenance result (best match per target):",
        f"  Targets: {len(summary):,}; within local snapshot: {len(summary) - outside:,}; outside snapshot: {outside:,}",
    ]
    for classification in (*STRONG_MATCHES, *REVIEW_MATCHES, "No archive match", "Outside EN4 snapshot coverage"):
        count = int(classifications.get(classification, 0))
        if count:
            lines.append(f"  {classification}: {count:,}")
    lines.extend([
        f"  Strong EN4 duplicate evidence: {strong:,}",
        f"  Candidate evidence requiring review: {review:,}",
        f"  No EN4 candidate within {BROAD_HOURS:g} hours/{BROAD_KM:g} km: {no_match:,}",
        f"  Spatiotemporal EN4 candidate pairs inspected: {candidate_pairs:,}",
        "  Interpretation: an EN4 match indicates possible training-snapshot overlap, not GLORYS assimilation or influence.",
    ])
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit normalized external profiles against EN4.2.2 OceanDepths profiles.")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--en4-index", type=Path, required=True)
    parser.add_argument("--oceandepths-root", type=Path, required=True)
    parser.add_argument("--matches-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    outputs = audit_italian_en4(
        args.targets,
        args.observations,
        args.en4_index,
        args.oceandepths_root,
        args.matches_output,
        args.summary_output,
    )
    for path in outputs:
        print(path)
    print(format_en4_audit_summary(read_table(outputs[1])))


if __name__ == "__main__":
    main()
