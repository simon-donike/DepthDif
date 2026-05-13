# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
#   --output-dir /work/data/depthdif/aligned_argo \
#   --archive-name aligned_argo_zarr.zip \
#   --timeout-seconds 120 \
#   --chunk-size-mb 8 \
#   --force-download \
#   --overwrite
"""Download and unpack the packaged aligned ARGO zarr archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

from depth_recon.data.dataset_creation.data_download_packaged._dataset_links import (
    load_dataset_url,
)

DATASET_LINK_KEY = "argo_aligned"
DEFAULT_ARCHIVE_NAME = "aligned_argo_zarr.zip"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CHUNK_SIZE_MB = 8
REQUEST_HEADERS = {"User-Agent": "DepthDif packaged dataset downloader"}


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
) -> Path:
    """Download ``url`` to ``output_path`` unless an existing file can be reused."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    chunk_size_bytes = int(chunk_size_mb) * 1024 * 1024
    try:
        request = Request(url, headers=REQUEST_HEADERS)
        with urlopen(request, timeout=int(timeout_seconds)) as response:
            with tmp_path.open("wb") as dst:
                while True:
                    chunk = response.read(chunk_size_bytes)
                    if not chunk:
                        break
                    dst.write(chunk)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return output_path


def _safe_member_path(output_dir: Path, member: zipfile.ZipInfo) -> Path:
    """Return the target path for a zip member after path traversal validation."""
    output_root = output_dir.resolve()
    target_path = (output_root / member.filename).resolve()
    try:
        target_path.relative_to(output_root)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe zip member path: {member.filename}") from exc
    return target_path


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract ``archive_path`` into ``output_dir`` and return written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            target_path = _safe_member_path(output_dir, member)
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {target_path}. "
                    "Pass --overwrite to replace extracted files."
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Avoid ZipFile.extractall so archive paths are validated before writes.
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written_paths.append(target_path)

    if not written_paths:
        raise RuntimeError(f"No files were extracted from: {archive_path}")
    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for aligned ARGO zarr downloads."""
    parser = argparse.ArgumentParser(
        description="Download the packaged aligned ARGO zarr zip and extract it."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the archive is downloaded and extracted.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Local archive filename inside --output-dir.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MB,
        help="Download streaming chunk size in MiB.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the zip even if the local archive already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist during extraction.",
    )
    return parser


def main() -> None:
    """Download and extract the packaged aligned ARGO zarr archive."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.timeout_seconds) <= 0:
        parser.error("--timeout-seconds must be positive.")
    if int(args.chunk_size_mb) <= 0:
        parser.error("--chunk-size-mb must be positive.")

    output_dir = Path(args.output_dir)
    archive_path = output_dir / str(args.archive_name)
    source_url = load_dataset_url(DATASET_LINK_KEY)
    downloaded_path = download_file(
        source_url,
        archive_path,
        force=bool(args.force_download),
        timeout_seconds=int(args.timeout_seconds),
        chunk_size_mb=int(args.chunk_size_mb),
    )
    written_paths = extract_archive(
        downloaded_path,
        output_dir,
        overwrite=bool(args.overwrite),
    )

    print(f"Downloaded archive: {downloaded_path}")
    print(f"Extracted files: {len(written_paths)}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
