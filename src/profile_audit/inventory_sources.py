from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Sequence

import requests

from profile_audit.common import sha256_file, utc_now, write_json

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
HF_API = "https://huggingface.co/api/datasets/{repository}/revision/{revision}"


def _remote_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inventory_sources(
    *,
    output: Path,
    oceandepths_revision: str,
    oceandepths_repository: str = "ESA-philab/OceanDepths",
    italian_xbt_record: int = 14848849,
    local_files: Sequence[Path] = (),
    oceandepths_root: Path | None = None,
    glorys_doi: str = "10.48670/moi-00021",
    glorys_version: str = "GLORYS12V1",
) -> Path:
    zenodo = _remote_json(ZENODO_API.format(record_id=italian_xbt_record))
    try:
        hf = _remote_json(HF_API.format(repository=oceandepths_repository, revision=oceandepths_revision))
        hf_commit = hf.get("sha", oceandepths_revision)
    except requests.RequestException:
        hf = {}
        hf_commit = oceandepths_revision

    files = []
    for path in map(Path, local_files):
        stat = path.stat()
        files.append(
            {
                "path": str(path.resolve()),
                "filename": path.name,
                "size": stat.st_size,
                "sha256": sha256_file(path),
                "acquired_utc": utc_now(),
                "download_url": None,
            }
        )
    zenodo_files = [
        {
            "filename": item["key"],
            "size": item["size"],
            "md5": item["checksum"].removeprefix("md5:"),
            "download_url": item["links"]["self"],
        }
        for item in zenodo.get("files", [])
    ]
    payload = {
        "created_utc": utc_now(),
        "oceandepths": {
            "repository": oceandepths_repository,
            "requested_revision": oceandepths_revision,
            "resolved_commit": hf_commit,
            "local_revision": _git_revision(oceandepths_root) if oceandepths_root else None,
            "source_product": "UK Met Office EN4.2.2 profile archive",
            "note": "The dataset uses ARGO names, but its profile source is EN4.2.2.",
        },
        "italian_xbt": {
            "record_id": italian_xbt_record,
            "doi": zenodo.get("doi"),
            "record_revision": zenodo.get("revision"),
            "files": zenodo_files,
        },
        "glorys": {"doi": glorys_doi, "version": glorys_version},
        "historical_inputs": {
            "status": "user_supplied_snapshots_required",
            "warning": "Current CORA/NRT releases cannot establish historical GLORYS input membership.",
        },
        "local_files": files,
    }
    return write_json(payload, output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze source revisions and checksums for an audit.")
    parser.add_argument("--oceandepths-revision", required=True)
    parser.add_argument("--oceandepths-repository", default="ESA-philab/OceanDepths")
    parser.add_argument("--oceandepths-root", type=Path)
    parser.add_argument("--italian-xbt-record", type=int, default=14848849)
    parser.add_argument("--local-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    print(inventory_sources(
        output=args.output,
        oceandepths_revision=args.oceandepths_revision,
        oceandepths_repository=args.oceandepths_repository,
        italian_xbt_record=args.italian_xbt_record,
        local_files=args.local_file,
        oceandepths_root=args.oceandepths_root,
    ))


if __name__ == "__main__":
    main()
