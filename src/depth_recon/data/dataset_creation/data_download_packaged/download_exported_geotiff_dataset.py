# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
#   --output-dir ./data/ocean_depth_reconstruction/geotiff_export \
#   --archive-name exported_geotiff_dataset.zip \
#   --timeout-seconds 120 \
#   --chunk-size-mb 8 \
#   --force-download \
#   --overwrite
"""Download and unpack the packaged exported GeoTIFF dataset archive."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
import shutil
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener
import zipfile

from depth_recon.data.dataset_creation.data_download_packaged._dataset_links import (
    load_dataset_url,
)

DATASET_LINK_KEY = "geotiff_training"
DEFAULT_ARCHIVE_NAME = "exported_geotiff_dataset.zip"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CHUNK_SIZE_MB = 8
REQUEST_HEADERS = {"User-Agent": "ocean-depth-reconstruction dataset downloader"}


class GoogleDriveDownloadParser(HTMLParser):
    """Collect likely download URLs from a Google Drive confirmation page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.forms: list[tuple[str, list[tuple[str, str]]]] = []
        self._form_action: str | None = None
        self._form_inputs: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record anchors and form fields that can contain confirmation links."""
        attrs_dict = {name: value or "" for name, value in attrs}
        if tag == "a" and attrs_dict.get("href"):
            self.hrefs.append(attrs_dict["href"])
            return

        if tag == "form":
            self._form_action = attrs_dict.get("action", "")
            self._form_inputs = []
            return

        if tag == "input" and self._form_action is not None:
            name = attrs_dict.get("name")
            if name:
                self._form_inputs.append((name, attrs_dict.get("value", "")))

    def handle_endtag(self, tag: str) -> None:
        """Store each completed form for later URL reconstruction."""
        if tag != "form" or self._form_action is None:
            return

        self.forms.append((self._form_action, list(self._form_inputs)))
        self._form_action = None
        self._form_inputs = []

    def download_urls(self, page_url: str) -> list[str]:
        """Return candidate Google Drive download URLs found on the page."""
        urls: list[str] = []
        for action, inputs in self.forms:
            urls.append(_with_query_params(urljoin(page_url, action), inputs))
        urls.extend(urljoin(page_url, href) for href in self.hrefs)

        return [url for url in urls if _looks_like_google_drive_download_url(url)]


def _with_query_params(url: str, params: list[tuple[str, str]]) -> str:
    """Append query parameters to ``url`` while preserving existing values."""
    parsed = urlparse(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query_items.extend(params)
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _looks_like_google_drive_download_url(url: str) -> bool:
    """Return whether ``url`` looks like a Google Drive file download URL."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    is_drive_url = netloc.endswith("drive.google.com") or netloc.endswith(
        "drive.usercontent.google.com"
    )
    return (
        is_drive_url
        and ("download" in parsed.path or "export=download" in parsed.query)
        and bool(parsed.query)
    )


def _google_drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file id from common public share URLs."""
    parsed = urlparse(url)
    query_ids = parse_qs(parsed.query).get("id")
    if query_ids:
        return query_ids[0]

    path_parts = [part for part in parsed.path.split("/") if part]
    if "d" not in path_parts:
        return None

    file_id_index = path_parts.index("d") + 1
    if file_id_index >= len(path_parts):
        return None
    return path_parts[file_id_index]


def _google_drive_download_url(url: str) -> str:
    """Convert a public Google Drive share URL into a download URL."""
    file_id = _google_drive_file_id(url)
    if not file_id:
        return url

    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _is_placeholder_url(url: str) -> bool:
    """Return whether ``url`` still contains a known placeholder token."""
    return "PLACEHOLDER" in url or "GOOGLE_DRIVE_FILE_ID" in url


def _response_is_download(response) -> bool:
    """Return whether an HTTP response appears to be file content."""
    content_type = str(response.headers.get("Content-Type", "")).lower()
    content_disposition = str(response.headers.get("Content-Disposition", ""))
    return bool(content_disposition) or "text/html" not in content_type


def _open_url(opener, url: str, timeout_seconds: int):
    """Open ``url`` with repository-standard headers."""
    request = Request(url, headers=REQUEST_HEADERS)
    return opener.open(request, timeout=int(timeout_seconds))


def _write_response(response, output_path: Path, *, chunk_size_bytes: int) -> None:
    """Stream an HTTP response body to ``output_path``."""
    with output_path.open("wb") as dst:
        while True:
            chunk = response.read(int(chunk_size_bytes))
            if not chunk:
                break
            dst.write(chunk)


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB,
) -> Path:
    """Download a public Google Drive file to ``output_path``."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path

    if _is_placeholder_url(url):
        raise ValueError(
            f"Dataset URL is still a placeholder: {url}. Update "
            "dataset_links.yaml with the hosted zip URL once it is available."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    chunk_size_bytes = int(chunk_size_mb) * 1024 * 1024
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    try:
        with _open_url(
            opener, _google_drive_download_url(url), timeout_seconds
        ) as response:
            if _response_is_download(response):
                _write_response(response, tmp_path, chunk_size_bytes=chunk_size_bytes)
                tmp_path.replace(output_path)
                return output_path

            # Large public Drive files can return a small HTML confirmation page first.
            warning_page_url = response.url
            warning_html = response.read().decode("utf-8", errors="replace")

        parser = GoogleDriveDownloadParser()
        parser.feed(warning_html)
        for candidate_url in parser.download_urls(warning_page_url):
            with _open_url(opener, candidate_url, timeout_seconds) as response:
                if not _response_is_download(response):
                    continue
                _write_response(response, tmp_path, chunk_size_bytes=chunk_size_bytes)
                tmp_path.replace(output_path)
                return output_path
    finally:
        tmp_path.unlink(missing_ok=True)

    raise RuntimeError(
        "Google Drive did not return a downloadable zip. Check that the file is "
        "publicly shared and update dataset_links.yaml with a direct public file link."
    )


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
    """Create the command-line parser for exported GeoTIFF dataset downloads."""
    parser = argparse.ArgumentParser(
        description="Download the exported GeoTIFF dataset zip and extract it."
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
    """Download and extract the exported GeoTIFF dataset archive."""
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
