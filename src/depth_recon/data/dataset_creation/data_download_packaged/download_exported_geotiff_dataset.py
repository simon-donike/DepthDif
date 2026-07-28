# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
#   --output-dir /work/data/OceanVariableReconstruction \
#   --revision main \
#   --zarr-path data/argo_glors_ostia_ssh.zarr \
#   --timeout-seconds 120 \
#   --chunk-size-mb 8 \
#   --force-download \
#   --overwrite
"""Download the packaged exported GeoTIFF dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from depth_recon.data.dataset_creation.data_download_packaged._dataset_links import (
    load_dataset_url,
)
from depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr import (
    DEFAULT_CHUNK_SIZE_MB,
    DEFAULT_REVISION,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_ZARR_PATH,
    HF_FULL_PACKAGE_PREFIXES,
    download_hf_package,
)

DATASET_LINK_KEY = "depthdif_training"
DEFAULT_OUTPUT_DIR = Path("/work/data/OceanVariableReconstruction")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for exported GeoTIFF dataset downloads."""
    parser = argparse.ArgumentParser(
        description="Download the exported GeoTIFF dataset from Hugging Face."
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
        help="Overwrite package files that already exist during download.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_TOKEN.",
    )
    return parser


def main() -> None:
    """Download the packaged exported GeoTIFF dataset."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.timeout_seconds) <= 0:
        parser.error("--timeout-seconds must be positive.")
    if int(args.chunk_size_mb) <= 0:
        parser.error("--chunk-size-mb must be positive.")

    output_dir = Path(args.output_dir)
    source_url = load_dataset_url(DATASET_LINK_KEY)
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
        package_prefixes=HF_FULL_PACKAGE_PREFIXES,
    )

    print(f"Downloaded Hugging Face package files: {len(written_paths)}")
    print(f"Output folder: {output_dir}")
    print(f"Aligned ARGO Zarr: {output_dir / Path(args.zarr_path)}")


if __name__ == "__main__":
    main()
