# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
#   --output-dir ./data/aligned_argo \
#   --revision main \
#   --zarr-path data/aligned_argo_profiles.zarr \
#   --timeout-seconds 120 \
#   --chunk-size-mb 8 \
#   --force-download \
#   --overwrite
"""Download the packaged Hugging Face aligned ARGO dataset folder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile

from depth_recon.data.dataset_creation.data_download_packaged._dataset_links import (
    load_dataset_url,
)

DATASET_LINK_KEY = "argo_aligned"
DEFAULT_ARCHIVE_NAME = "aligned_argo_zarr.zip"
DEFAULT_OUTPUT_DIR = Path("./data/aligned_argo")
DEFAULT_REVISION = "main"
DEFAULT_ZARR_PATH = Path("data/aligned_argo_profiles.zarr")
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CHUNK_SIZE_MB = 8
REQUEST_HEADERS = {"User-Agent": "ocean-depth-reconstruction dataset downloader"}
HF_PACKAGE_PREFIXES = (
    "data/",
    "indices/",
    "examples/",
    "metadata/",
    "README.md",
    "LICENSE",
    "LICENSE.md",
)


def _headers(token: str | None = None) -> dict[str, str]:
    """Return HTTP headers for Hugging Face Hub requests."""
    headers = dict(REQUEST_HEADERS)
    resolved_token = (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"
    return headers


def _open_url(url: str, *, timeout_seconds: int, token: str | None = None):
    """Open a URL with the repository-standard headers."""
    return urlopen(Request(url, headers=_headers(token)), timeout=int(timeout_seconds))


def _write_response(response, output_path: Path, *, chunk_size_bytes: int) -> None:
    """Stream an HTTP response body to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as dst:
            while True:
                chunk = response.read(int(chunk_size_bytes))
                if not chunk:
                    break
                dst.write(chunk)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
    token: str | None = None,
) -> Path:
    """Download ``url`` to ``output_path`` unless an existing file can be reused."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    chunk_size_bytes = int(chunk_size_mb) * 1024 * 1024
    with _open_url(url, timeout_seconds=timeout_seconds, token=token) as response:
        _write_response(response, output_path, chunk_size_bytes=chunk_size_bytes)
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


def _looks_like_zip_url(url: str) -> bool:
    """Return whether the configured URL should use the legacy zip downloader."""
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".zip") or (
        "/resolve/" in parsed.path and parsed.path.lower().endswith(".zip")
    )


def _parse_hf_dataset_url(
    url: str, *, default_revision: str
) -> tuple[str, str, str, str]:
    """Parse a Hugging Face dataset repo URL into endpoint, repo id, revision, and subdir."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        parts = [part for part in url.strip("/").split("/") if part]
        if len(parts) != 2:
            raise ValueError(f"Could not parse Hugging Face dataset URL: {url}")
        return "https://huggingface.co", "/".join(parts), default_revision, ""

    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        raise ValueError(f"Could not parse Hugging Face dataset URL: {url}")
    if path_parts[0] == "datasets":
        path_parts = path_parts[1:]
    if len(path_parts) < 2:
        raise ValueError(f"Expected a dataset repo URL with owner/name: {url}")

    repo_id = "/".join(path_parts[:2])
    revision = default_revision
    subdir_parts: list[str] = []
    remainder = path_parts[2:]
    if remainder and remainder[0] in {"tree", "resolve", "blob"}:
        if len(remainder) >= 2:
            revision = remainder[1]
            subdir_parts = remainder[2:]
    elif remainder:
        subdir_parts = remainder
    endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.rstrip("/"), repo_id, revision, "/".join(subdir_parts)


def _hf_tree_url(endpoint: str, repo_id: str, revision: str, path_in_repo: str) -> str:
    """Build the Hugging Face Hub tree API URL for a dataset repo."""
    quoted_revision = quote(revision, safe="")
    quoted_path = quote(path_in_repo.strip("/"), safe="/")
    base = f"{endpoint}/api/datasets/{repo_id}/tree/{quoted_revision}"
    if quoted_path:
        base = f"{base}/{quoted_path}"
    return f"{base}?recursive=true"


def _hf_resolve_url(
    endpoint: str, repo_id: str, revision: str, path_in_repo: str
) -> str:
    """Build a raw file download URL for one Hugging Face dataset file."""
    quoted_revision = quote(revision, safe="")
    quoted_path = quote(path_in_repo.strip("/"), safe="/")
    return f"{endpoint}/datasets/{repo_id}/resolve/{quoted_revision}/{quoted_path}?download=true"


def _link_next(headers) -> str | None:
    """Extract a RFC 5988 next-page URL from response headers."""
    link_header = headers.get("Link")
    if not link_header:
        return None
    for item in str(link_header).split(","):
        parts = item.split(";")
        if len(parts) < 2:
            continue
        url_part = parts[0].strip()
        rel_parts = [part.strip() for part in parts[1:]]
        if (
            'rel="next"' in rel_parts
            and url_part.startswith("<")
            and url_part.endswith(">")
        ):
            return url_part[1:-1]
    return None


def list_hf_dataset_files(
    source_url: str,
    *,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, str, str, list[str]]:
    """List files in a Hugging Face dataset repo using the Hub tree API."""
    endpoint, repo_id, parsed_revision, subdir = _parse_hf_dataset_url(
        source_url, default_revision=revision
    )
    active_revision = parsed_revision or revision
    url = _hf_tree_url(endpoint, repo_id, active_revision, subdir)
    files: list[str] = []
    while url:
        with _open_url(url, timeout_seconds=timeout_seconds, token=token) as response:
            payload = json.loads(response.read().decode("utf-8"))
            next_url = _link_next(response.headers)
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Hugging Face tree API payload from {url}")
        for item in payload:
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            path = item.get("path")
            if isinstance(path, str):
                files.append(path)
        url = next_url
    return endpoint, repo_id, active_revision, files


def _is_package_file(path: str) -> bool:
    """Return whether a repo path belongs to the aligned ARGO HF package layout."""
    return any(
        path == prefix or path.startswith(prefix) for prefix in HF_PACKAGE_PREFIXES
    )


def _safe_repo_target(output_dir: Path, repo_path: str) -> Path:
    """Resolve a repo file path under the output dir after traversal checks."""
    pure_path = PurePosixPath(repo_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise RuntimeError(f"Unsafe repository path: {repo_path}")
    output_root = output_dir.resolve()
    target = (output_root / Path(*pure_path.parts)).resolve()
    try:
        target.relative_to(output_root)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe repository path: {repo_path}") from exc
    return target


def download_hf_package(
    source_url: str,
    output_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    zarr_path: Path = DEFAULT_ZARR_PATH,
    force: bool = False,
    overwrite: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
    token: str | None = None,
) -> list[Path]:
    """Download the Hugging Face aligned ARGO package folder into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint, repo_id, active_revision, files = list_hf_dataset_files(
        source_url,
        revision=revision,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    package_files = [path for path in files if _is_package_file(path)]
    if not package_files:
        raise RuntimeError(
            f"No aligned ARGO package files found in Hugging Face dataset repo: {repo_id}"
        )

    written_paths: list[Path] = []
    for repo_path in package_files:
        target_path = _safe_repo_target(output_dir, repo_path)
        if target_path.exists() and not (overwrite or force):
            continue
        file_url = _hf_resolve_url(endpoint, repo_id, active_revision, repo_path)
        download_file(
            file_url,
            target_path,
            force=True,
            timeout_seconds=timeout_seconds,
            chunk_size_mb=chunk_size_mb,
            token=token,
        )
        written_paths.append(target_path)

    expected_zarr = output_dir / zarr_path
    if not expected_zarr.exists():
        raise RuntimeError(
            f"Downloaded package is missing expected Zarr path: {expected_zarr}"
        )
    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for aligned ARGO package downloads."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the Hugging Face packaged aligned ARGO dataset folder. "
            "Legacy .zip links are still supported."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the Hugging Face package files are downloaded.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face repo revision to download when the configured URL does not include one.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help="Expected package-relative Zarr path used for post-download validation.",
    )
    parser.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE_NAME,
        help="Local archive filename for legacy .zip links only.",
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
        help="Download files even if local package files already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite package files that already exist during download or zip extraction.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_TOKEN.",
    )
    return parser


def main() -> None:
    """Download the packaged aligned ARGO dataset."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.timeout_seconds) <= 0:
        parser.error("--timeout-seconds must be positive.")
    if int(args.chunk_size_mb) <= 0:
        parser.error("--chunk-size-mb must be positive.")

    output_dir = Path(args.output_dir)
    source_url = load_dataset_url(DATASET_LINK_KEY)
    if _looks_like_zip_url(source_url):
        archive_path = output_dir / str(args.archive_name)
        downloaded_path = download_file(
            source_url,
            archive_path,
            force=bool(args.force_download),
            timeout_seconds=int(args.timeout_seconds),
            chunk_size_mb=int(args.chunk_size_mb),
            token=args.hf_token,
        )
        written_paths = extract_archive(
            downloaded_path,
            output_dir,
            overwrite=bool(args.overwrite),
        )
        print(f"Downloaded archive: {downloaded_path}")
        print(f"Extracted files: {len(written_paths)}")
    else:
        written_paths = download_hf_package(
            source_url,
            output_dir,
            revision=str(args.revision),
            zarr_path=Path(args.zarr_path),
            force=bool(args.force_download),
            overwrite=bool(args.overwrite),
            timeout_seconds=int(args.timeout_seconds),
            chunk_size_mb=int(args.chunk_size_mb),
            token=args.hf_token,
        )
        print(f"Downloaded Hugging Face package files: {len(written_paths)}")
    print(f"Output folder: {output_dir}")
    print(f"Aligned ARGO Zarr: {output_dir / Path(args.zarr_path)}")


if __name__ == "__main__":
    main()
