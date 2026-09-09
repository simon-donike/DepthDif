from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from profile_audit.common import read_table, write_table


SOURCE_TIMELINE = (
    (pd.Timestamp("1993-01-01", tz="UTC"), pd.Timestamp("2013-12-31 23:59:59", tz="UTC"), "CORA4.1", "documented_with_2003_2013_typo_pending_confirmation"),
    (pd.Timestamp("2014-01-01", tz="UTC"), pd.Timestamp("2015-12-31 23:59:59", tz="UTC"), "CORA5.0", "documented"),
    (pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2016-12-31 23:59:59", tz="UTC"), "CORA5.1", "documented"),
    (pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2021-06-30 23:59:59", tz="UTC"), "unresolved", "unresolved"),
    (pd.Timestamp("2021-07-01", tz="UTC"), pd.Timestamp("2100-12-31", tz="UTC"), "INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047", "documented_internal_snapshot_unavailable"),
)


def source_for_time(timestamp: pd.Timestamp) -> tuple[str, str]:
    for start, end, source, status in SOURCE_TIMELINE:
        if start <= timestamp <= end:
            return source, status
    return "outside_glorys_period", "not_applicable"


def _same_archive(observed: object, expected: str) -> bool | None:
    if expected == "unresolved":
        return None
    left = "".join(character for character in str(observed).upper() if character.isalnum())
    right = "".join(character for character in expected.upper() if character.isalnum())
    return bool(left and (left in right or right in left))


def classify_matches(
    profiles: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    feedback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    best = matches.sort_values(["target_profile_id", "rank"]).drop_duplicates("target_profile_id")
    result = profiles.merge(best, left_on="profile_id", right_on="target_profile_id", how="left")
    if "datetime_utc" in result:
        times = pd.to_datetime(result["datetime_utc"], utc=True, errors="coerce")
    elif "profile_date" in result:
        times = pd.to_datetime(result["profile_date"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    else:
        raise ValueError("Profiles need datetime_utc or profile_date for source-timeline classification.")
    timeline = [source_for_time(timestamp) if pd.notna(timestamp) else ("unknown", "invalid_time") for timestamp in times]
    result["expected_glorys_input_archive"] = [item[0] for item in timeline]
    result["source_timeline_status"] = [item[1] for item in timeline]
    result["input_archive_match"] = result["classification"].isin(("Exact", "Near-exact", "Probable duplicate"))
    observed_archives = result["archive_version"] if "archive_version" in result else pd.Series("", index=result.index)
    result["correct_archive_for_period"] = pd.array([
        _same_archive(observed, expected) if matched else False
        for observed, expected, matched in zip(observed_archives, result["expected_glorys_input_archive"], result["input_archive_match"])
    ], dtype="boolean")
    profile_qc_source = result["input_profile_qc"] if "input_profile_qc" in result else result["profile_qc"] if "profile_qc" in result else pd.Series(-1, index=result.index)
    profile_qc = pd.to_numeric(profile_qc_source, errors="coerce").fillna(-1)
    result["input_profile_qc"] = profile_qc.astype("int16")
    temperature_levels = result["usable_temperature_levels"] if "usable_temperature_levels" in result else result["overlap_levels"] if "overlap_levels" in result else pd.Series(0, index=result.index)
    salinity_levels = result["usable_salinity_levels"] if "usable_salinity_levels" in result else pd.Series(0, index=result.index)
    result["usable_temperature_levels"] = pd.to_numeric(temperature_levels, errors="coerce").fillna(0).astype(int)
    result["usable_salinity_levels"] = pd.to_numeric(salinity_levels, errors="coerce").fillna(0).astype(int)
    eligible_qc = (result["input_profile_qc"] < 0) | result["input_profile_qc"].isin((0, 1, 2))
    result["potentially_assimilated"] = result["input_archive_match"] & result["correct_archive_for_period"].fillna(False) & eligible_qc & ((result["usable_temperature_levels"] > 0) | (result["usable_salinity_levels"] > 0))
    result["confirmed_accepted"] = pd.array([pd.NA] * len(result), dtype="boolean")
    result["evidence_level"] = np.where(result["input_archive_match"], "A", "none")
    result.loc[result["potentially_assimilated"], "evidence_level"] = "B"
    result["reason"] = np.where(result["input_archive_match"], "Matched in a candidate input archive; GLORYS acceptance is not proven.", "No qualifying frozen input-archive match.")
    unresolved = result["source_timeline_status"] == "unresolved"
    result.loc[unresolved, "reason"] = "GLORYS input source for 2017 through June 2021 is unresolved."
    internal_unavailable = result["source_timeline_status"] == "documented_internal_snapshot_unavailable"
    result.loc[internal_unavailable & ~result["input_archive_match"], "reason"] = "The documented internal 013_047 input snapshot is unavailable publicly; archive membership cannot be classified."
    if feedback is not None and not feedback.empty:
        key = "profile_id" if "profile_id" in feedback else "target_profile_id"
        feedback_columns = [key] + [column for column in ("accepted", "rejection_reason", "assimilation_cycle", "innovation", "analysis_residual") if column in feedback]
        result = result.merge(feedback[feedback_columns], left_on="profile_id", right_on=key, how="left", suffixes=("", "_feedback"))
        available = result["accepted"].notna()
        result.loc[available, "confirmed_accepted"] = result.loc[available, "accepted"].astype(bool)
        result.loc[available, "evidence_level"] = "C"
        influenced = available & result.get("innovation", pd.Series(np.nan, index=result.index)).notna() & result.get("analysis_residual", pd.Series(np.nan, index=result.index)).notna()
        result.loc[influenced, "evidence_level"] = "D"
        result.loc[available & result["confirmed_accepted"].fillna(False), "reason"] = "Confirmed accepted by profile-level GLORYS feedback."
        result.loc[available & ~result["confirmed_accepted"].fillna(False), "reason"] = "Confirmed rejected by profile-level GLORYS feedback."
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify archive matches without overstating assimilation evidence.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = classify_matches(read_table(args.profiles), read_table(args.matches), feedback=read_table(args.feedback) if args.feedback else None)
    print(write_table(result, args.output))


if __name__ == "__main__":
    main()
