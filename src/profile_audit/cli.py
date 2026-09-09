from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

COMMANDS = {
    "acquire-013030-proxy": "profile_audit.acquire_013030_proxy",
    "acquire-inputs": "profile_audit.acquire_assimilation_inputs",
    "audit-en4": "profile_audit.audit_oceandepths",
    "audit-e4-e5-inputs": "profile_audit.parse_e4_e5_profiles",
    "audit-italian-en4": "profile_audit.audit_italian_en4",
    "audit-external-en4": "profile_audit.audit_italian_en4",
    "inventory": "profile_audit.inventory_sources",
    "parse-italian-xbt": "profile_audit.parse_italian_xbt",
    "resolve-ids": "profile_audit.resolve_external_ids",
    "export-oceandepths": "profile_audit.export_oceandepths_profiles",
    "index-inputs": "profile_audit.index_assimilation_inputs",
    "match": "profile_audit.match_profiles",
    "classify": "profile_audit.classify_matches",
    "influence": "profile_audit.analyze_glorys_influence",
    "report": "profile_audit.make_report",
}


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in COMMANDS:
        importlib.import_module(COMMANDS[arguments[0]]).main(arguments[1:])
        return
    parser = argparse.ArgumentParser(prog="profile-audit", description="GLORYS12 profile assimilation audit pipeline.")
    parser.add_argument("command", choices=COMMANDS)
    args, remainder = parser.parse_known_args(arguments)
    importlib.import_module(COMMANDS[args.command]).main(remainder)


if __name__ == "__main__":
    main()
