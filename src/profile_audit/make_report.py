from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from profile_audit.common import read_table, utc_now, write_table


def _summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame:
        return pd.DataFrame(columns=[column, "profile_count"])
    return frame.groupby(column, dropna=False).agg(
        profile_count=("profile_id", "size"),
        archive_matches=("input_archive_match", "sum"),
        potentially_assimilated=("potentially_assimilated", "sum"),
        confirmed_accepted=("confirmed_accepted", lambda values: int(values.fillna(False).sum())),
    ).reset_index()


def _diagnostic_plots(audit: pd.DataFrame, output_dir: Path) -> int:
    probable = audit[audit["classification"].isin(("Near-exact", "Probable duplicate"))]
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in probable.itertuples(index=False):
        identifier = str(row.profile_id).replace("/", "_").replace(":", "_")
        labels = ["time min", "distance km", "MAE C", "correlation"]
        values = [getattr(row, "time_difference_minutes", float("nan")), getattr(row, "distance_km", float("nan")), getattr(row, "temperature_mae_c", float("nan")), getattr(row, "temperature_correlation", float("nan"))]
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.bar(labels, values)
        axis.set_title(f"{row.profile_id}: {row.classification}")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{identifier}.png", dpi=140)
        plt.close(figure)
    return len(probable)


def _true_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _format_counts(counts: dict[object, int]) -> str:
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def _coverage_counts(audit: pd.DataFrame) -> tuple[int, int, int]:
    if "source_timeline_status" not in audit:
        return 0, 0, 0
    unresolved = int((audit["source_timeline_status"] == "unresolved").sum())
    proxy_era = int((audit["source_timeline_status"] == "documented_internal_snapshot_unavailable").sum())
    proxy_candidates = int(
        audit.loc[
            audit["source_timeline_status"] == "documented_internal_snapshot_unavailable",
            "candidate_profile_id",
        ].notna().sum()
    ) if "candidate_profile_id" in audit else 0
    return unresolved, proxy_era, proxy_candidates


def _print_summary(audit: pd.DataFrame, plot_count: int) -> None:
    italian = audit[audit["profile_id"].astype(str).str.startswith("italian-xbt:")]
    classifications = audit["classification"].value_counts(dropna=False).to_dict() if "classification" in audit else {}
    if "classification" in audit:
        classifications = audit["classification"].fillna("Unclassified (source unavailable)").value_counts().to_dict()
    evidence = audit["evidence_level"].value_counts(dropna=False).to_dict() if "evidence_level" in audit else {}
    ambiguous = int((audit["classification"] == "Ambiguous").sum()) if "classification" in audit else 0
    unresolved, proxy_era, proxy_candidates = _coverage_counts(audit)

    print("\nReport summary")
    print(f"  Profiles audited: {len(audit):,}")
    print(f"  Italian XBT profiles: {len(italian):,}")
    print(f"  Archive matches: {_true_count(audit, 'input_archive_match'):,}")
    print(f"  Potentially assimilated: {_true_count(audit, 'potentially_assimilated'):,}")
    print(f"  Confirmed accepted: {_true_count(audit, 'confirmed_accepted'):,}")
    print(f"  Ambiguous matches: {ambiguous:,}")
    print(f"  Diagnostic plots: {plot_count:,}")
    print(f"  Historical source unresolved (2017-June 2021): {unresolved:,}")
    print(f"  Checked against current 013_030 proxy (July 2021 onward): {proxy_era:,}")
    print(f"    Proxy candidates: {proxy_candidates:,}")
    print(f"    No proxy candidate by ID or 24-hour/25-km search: {proxy_era - proxy_candidates:,}")
    print(f"  Classifications: {_format_counts(classifications)}")
    print(f"  Evidence levels: {_format_counts(evidence)}")


def make_report(audit: pd.DataFrame, output_dir: Path, *, source_inventory: Path | None = None, make_plots: bool = True) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit.copy()
    if "datetime_utc" in audit:
        audit["year"] = pd.to_datetime(audit["datetime_utc"], utc=True, errors="coerce").dt.year
    profile_path = write_table(audit, output_dir / "profile_audit.parquet")
    italian = audit[audit["profile_id"].astype(str).str.startswith("italian-xbt:")]
    italian_path = write_table(italian, output_dir / "italian_xbt_audit.csv")
    cruise_path = write_table(_summary(italian, "cruise"), output_dir / "summary_by_cruise.csv")
    year_path = write_table(_summary(audit, "year"), output_dir / "summary_by_year.csv")
    ambiguous_path = write_table(audit[audit["classification"] == "Ambiguous"], output_dir / "ambiguous_matches.csv")
    feedback_columns = [column for column in ("profile_id", "datetime_utc", "latitude", "longitude", "longitude_180", "cruise", "station", "candidate_dc_reference", "candidate_source_file") if column in audit]
    feedback_path = write_table(audit[feedback_columns], output_dir / "mercator_feedback_request.csv")
    plot_count = _diagnostic_plots(audit, output_dir / "diagnostic_plots") if make_plots else 0
    inventory = json.loads(Path(source_inventory).read_text(encoding="utf-8")) if source_inventory else None
    counts = audit["evidence_level"].value_counts(dropna=False).to_dict()
    methods = [
        "# GLORYS12 Profile Assimilation Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Evidence",
        "",
        "Archive membership is level A evidence and eligibility is level B. Only profile-level GLORYS feedback establishes level C acceptance; innovation or increment diagnostics establish level D influence.",
        "",
        f"Evidence counts: `{json.dumps(counts, default=str, sort_keys=True)}`.",
        "",
        "## Coverage Interpretation",
        "",
        f"{_coverage_counts(audit)[0]} profiles from 2017 through June 2021 have an unresolved historical GLORYS input source.",
        f"{_coverage_counts(audit)[1]} profiles from July 2021 onward were checked against the current 013_030 proxy; {_coverage_counts(audit)[2]} proxy candidates were found.",
        "A missing current-proxy candidate is not proof that a profile was absent from the historical GLORYS input snapshot.",
        "",
        "## Dataset Provenance",
        "",
        "OceanDepths uses ARGO names, but these profiles were exported from the UK Met Office EN4.2.2 profile archive. Source filenames and source-profile row indices are retained.",
        "",
        "## Caveats",
        "",
        "- A current CORA release cannot substitute for the exact historical snapshot supplied to GLORYS.",
        "- The GLORYS source for 2017 through June 2021 remains unresolved.",
        "- Public daily analyses provide only weak influence evidence, not accepted-observation flags.",
        f"- Generated {plot_count} probable-match diagnostic plots.",
    ]
    if inventory:
        methods.extend(["", "## Frozen Sources", "", f"```json\n{json.dumps(inventory, indent=2, sort_keys=True)}\n```"])
    report_path = output_dir / "assimilation_audit.md"
    report_path.write_text("\n".join(methods) + "\n", encoding="utf-8")
    return {
        "profile_audit": profile_path, "italian_xbt_audit": italian_path,
        "summary_by_cruise": cruise_path, "summary_by_year": year_path,
        "ambiguous_matches": ambiguous_path, "mercator_feedback_request": feedback_path,
        "report": report_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate audit tables, plots, and methods report.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    audit = read_table(args.audit)
    outputs = make_report(audit, args.output_dir, source_inventory=args.source_inventory, make_plots=not args.no_plots)
    probable_count = int(audit["classification"].isin(("Near-exact", "Probable duplicate")).sum()) if "classification" in audit and not args.no_plots else 0
    _print_summary(audit, probable_count)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
