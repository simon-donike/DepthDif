"""Verify selected observation identities against the documented screened lists."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from val.baselines import check_artifact


def screening_source(cohort: dict) -> tuple[str | None, str | None]:
    """Resolve the cohort's original exclusion registry and optional frozen hash."""
    if cohort["kind"] == "en4_archive":
        return cohort.get("audit_path"), cohort.get("audit_sha256")
    return cohort.get("screened_profiles"), cohort.get("screened_profiles_sha256")


def verify_screened_profiles(points: pd.DataFrame, cohort: dict, output: Path) -> dict:
    """Abort on unlisted or misidentified profiles before any model inference."""
    path, expected = screening_source(cohort)
    if not path:
        return {"status": "not_configured", "verified_not_assimilated": False}
    digest = check_artifact(path, expected)
    year = int(cohort["year"])
    registry = pd.read_parquet(
        path,
        filters=[
            ("datetime_utc", ">=", pd.Timestamp(f"{year}-01-01", tz="UTC")),
            ("datetime_utc", "<", pd.Timestamp(f"{year + 1}-01-01", tz="UTC")),
        ],
    )
    en4 = cohort["kind"] == "en4_archive"
    if en4:
        registry = registry.loc[
            registry.audit_status.eq("no_spatiotemporal_candidate")
        ].copy()
        registry["profile_id"] = (
            "en4:"
            + registry.profile_source_file.astype(str)
            + ":"
            + registry.source_profile_idx.astype(int).astype(str)
        )
    if registry.profile_id.duplicated().any():
        raise ValueError("Screened registry contains duplicate profile identities")
    keys = ["profile_id", "observation_time"]
    if not en4:
        keys += ["source_file", "cruise", "station", "latitude", "longitude", "qc_flag"]
    selected = points[keys].drop_duplicates().copy()
    if selected.empty or selected.profile_id.duplicated().any():
        raise ValueError("Selected profiles have empty or inconsistent identities")
    audit = selected.merge(
        registry,
        on="profile_id",
        how="left",
        suffixes=("", "_registry"),
        validate="one_to_one",
        indicator=True,
    )
    audit["listed"] = audit["_merge"].eq("both")
    audit["identity_matches"] = pd.to_datetime(
        audit.observation_time, utc=True, format="mixed"
    ).eq(pd.to_datetime(audit.datetime_utc, utc=True, format="mixed"))
    if not en4:
        for key in ("source_file", "cruise", "station"):
            audit["identity_matches"] &= (
                audit[key].astype(str).eq(audit[f"{key}_registry"].astype(str))
            )
        # XBT retains native coordinates; EN4 positions are already grid centres.
        audit["identity_matches"] &= np.isclose(
            audit.latitude, audit.latitude_registry, atol=1e-6, rtol=0
        ) & np.isclose(audit.longitude, audit.longitude_raw, atol=1e-6, rtol=0)
        audit["identity_matches"] &= pd.to_numeric(audit.qc_flag).eq(1)
    audit["membership_verified"] = audit.listed & audit.identity_matches
    result = {
        "status": "passed" if audit.membership_verified.all() else "failed",
        "registry": str(Path(path).resolve()),
        "registry_sha256": digest,
        "year": year,
        "registry_profiles": len(registry),
        "selected_profiles": len(audit),
        "verified_profiles": int(audit.membership_verified.sum()),
        "unlisted_profiles": int((~audit.listed).sum()),
        "identity_mismatches": int((audit.listed & ~audit.identity_matches).sum()),
        "verified_not_assimilated": False,
        "evidence": "Membership in the documented archive-screened set; not GLORYS assimilation-log verification",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit.drop(columns="_merge").to_csv(output / "profile_membership.csv", index=False)
    (output / "profile_membership.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise ValueError(f"Screened profile membership failed; see {output}")
    return result
