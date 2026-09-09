from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from profile_audit.common import normalize_identifier, read_table, write_table


def resolve_external_ids(profiles: pd.DataFrame, alias_sources: Sequence[pd.DataFrame] = ()) -> pd.DataFrame:
    """Create identifier aliases and merge curated accession/WOD mappings.

    Alias sources must contain ``profile_id`` and may contain any additional
    identifier columns. Curated mappings deliberately take precedence over
    fuzzy matching later in the pipeline.
    """
    base_columns = [column for column in ("profile_id", "cruise", "station", "ship_name", "project_name") if column in profiles]
    aliases = profiles[base_columns].copy()
    cruise = aliases["cruise"].astype(str) if "cruise" in aliases else pd.Series("", index=aliases.index)
    station = aliases["station"].astype(str) if "station" in aliases else pd.Series("", index=aliases.index)
    aliases["cruise_station"] = cruise + ":" + station
    aliases["normalized_cruise_station"] = aliases["cruise_station"].map(normalize_identifier)
    aliases["ncei_accession"] = pd.NA
    aliases["replacement_accession"] = pd.NA
    aliases["wod_id"] = pd.NA
    aliases["platform_code"] = pd.NA
    for source in alias_sources:
        if "profile_id" not in source:
            raise ValueError("Every alias source must contain profile_id.")
        aliases = aliases.merge(source, on="profile_id", how="left", suffixes=("", "_curated"), validate="one_to_one")
        for column in list(aliases):
            if column.endswith("_curated"):
                original = column.removesuffix("_curated")
                aliases[original] = aliases[column].combine_first(aliases.get(original))
                aliases = aliases.drop(columns=column)
    return aliases


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact identifier aliases for Italian profiles.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--alias-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = resolve_external_ids(read_table(args.profiles), [read_table(path) for path in args.alias_source])
    print(write_table(result, args.output))


if __name__ == "__main__":
    main()
