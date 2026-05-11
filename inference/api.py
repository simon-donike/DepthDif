# Example:
# /work/envs/depth/bin/python -m inference.api infer-week --year 2015 --iso-week 25 --rectangle -20 30 10 50 --device cuda --config-repo donike/depthdif
"""Public inference API for PyPI and notebook usage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys
import urllib.request
from typing import Callable, Sequence
from zipfile import ZipFile

import yaml

from data.dataset_argo_netcdf_gridded import DEFAULT_LAND_MASK_PATH
from inference.export_global import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAIN_CONFIG,
    ExportRunResult,
    _build_parser as _build_export_parser,
    run_global_inference,
)

DEFAULT_HF_REPO_ID = "donike/depthdif"
DEFAULT_HF_REVISION = "main"
DEFAULT_HF_MODEL_CONFIG = DEFAULT_MODEL_CONFIG
DEFAULT_HF_DATA_CONFIG = DEFAULT_DATA_CONFIG
DEFAULT_HF_TRAIN_CONFIG = DEFAULT_TRAIN_CONFIG
DEFAULT_HF_CHECKPOINT = "checkpoints/depthdif.ckpt"
DEFAULT_EN4_BASE_URL = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1"

Downloader = Callable[[str, Path], Path]


@dataclass(frozen=True)
class InferenceAssets:
    """Local paths to model/config artifacts used by public inference."""

    model_config: Path
    data_config: Path
    train_config: Path
    checkpoint: Path


def _default_cache_dir() -> Path:
    """Return the default DepthDif artifact cache directory."""
    return Path.home() / ".cache" / "depthdif"


def _download_url(url: str, output_path: Path) -> Path:
    """Download one URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def _hf_url(repo_id: str, revision: str, repo_path: str) -> str:
    """Build a Hugging Face resolve URL for one repository artifact."""
    clean_path = str(repo_path).lstrip("/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{clean_path}"


def _resolve_hf_file(
    *,
    repo_id: str,
    revision: str,
    repo_path: str,
    cache_dir: Path,
    downloader: Downloader,
    force_download: bool,
) -> Path:
    """Resolve one Hugging Face artifact into the local cache."""
    local_path = cache_dir / repo_id.replace("/", "--") / revision / repo_path
    if local_path.exists() and not force_download:
        return local_path
    return downloader(_hf_url(repo_id, revision, repo_path), local_path)


def resolve_hf_assets(
    *,
    config_repo: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    model_config_path: str = DEFAULT_HF_MODEL_CONFIG,
    data_config_path: str = DEFAULT_HF_DATA_CONFIG,
    train_config_path: str = DEFAULT_HF_TRAIN_CONFIG,
    checkpoint_path: str = DEFAULT_HF_CHECKPOINT,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> InferenceAssets:
    """Download or reuse configs and checkpoint from Hugging Face."""
    target_cache = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    fetch = _download_url if downloader is None else downloader
    return InferenceAssets(
        model_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=model_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        data_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=data_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        train_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=train_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        checkpoint=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=checkpoint_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
    )


def _week_months(
    year: int, iso_week: int, radius_days: int = 3
) -> set[tuple[int, int]]:
    """Return calendar year/month pairs touched by an ISO-week window."""
    center = date.fromisocalendar(int(year), int(iso_week), 3)
    return {
        (
            (center + timedelta(days=offset)).year,
            (center + timedelta(days=offset)).month,
        )
        for offset in range(-int(radius_days), int(radius_days) + 1)
    }


def _en4_zip_url(base_url: str, year: int) -> str:
    """Return the EN4 yearly profile ZIP URL for one calendar year."""
    return f"{base_url.rstrip('/')}/EN.4.2.2.profiles.g10.{int(year)}.zip"


def _extract_matching_en4_months(
    *,
    zip_path: Path,
    output_dir: Path,
    months: set[tuple[int, int]],
) -> list[Path]:
    """Extract EN4 monthly NetCDF files matching requested months."""
    extracted: list[Path] = []
    month_tokens = {f"{year:04d}{month:02d}" for year, month in months}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not name.endswith(".nc"):
                continue
            if not any(f".{token}." in name for token in month_tokens):
                continue
            target_path = output_dir / name
            if not target_path.exists():
                # Extract through ZipFile.open to avoid trusting archive paths.
                with archive.open(member) as src, target_path.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(target_path)
    return extracted


def download_argo_for_week(
    year: int,
    iso_week: int,
    output_dir: str | Path,
    *,
    base_url: str = DEFAULT_EN4_BASE_URL,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Download and extract EN4/ARGO profile files needed for one ISO week."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    months = _week_months(int(year), int(iso_week))
    missing_months = {
        (month_year, month)
        for month_year, month in months
        if not list(output_path.glob(f"*{month_year:04d}{month:02d}*.nc"))
    }
    if not missing_months:
        return output_path

    archive_cache = (
        _default_cache_dir() / "en4" if cache_dir is None else Path(cache_dir)
    )
    archive_cache.mkdir(parents=True, exist_ok=True)
    fetch = _download_url if downloader is None else downloader

    for archive_year in sorted({year_value for year_value, _ in missing_months}):
        zip_path = archive_cache / f"EN.4.2.2.profiles.g10.{archive_year}.zip"
        if force_download or not zip_path.exists():
            fetch(_en4_zip_url(base_url, archive_year), zip_path)
        extracted = _extract_matching_en4_months(
            zip_path=zip_path,
            output_dir=output_path,
            months={item for item in missing_months if item[0] == archive_year},
        )
        if not extracted:
            raise RuntimeError(
                "Downloaded EN4 archive did not contain the requested month(s): "
                f"{sorted(missing_months)}"
            )
    return output_path


def _write_public_data_config(
    *,
    source_config: Path,
    output_dir: Path,
    argo_dir: str | Path | None,
    glorys_dir: str | Path | None,
    ostia_dir: str | Path | None,
    sealevel_dir: str | Path | None,
    metadata_cache_dir: str | Path | None,
    land_mask_path: str | Path | None,
) -> Path:
    """Write a data config copy with user-provided source directories."""
    with Path(source_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg.setdefault("dataset", {})
    core_cfg = dataset_cfg.setdefault("core", {})
    grid_cfg = dataset_cfg.setdefault("grid", {})
    for key, value in (
        ("argo_dir", argo_dir),
        ("glorys_dir", glorys_dir),
        ("ostia_dir", ostia_dir),
        ("sealevel_dir", sealevel_dir),
        ("metadata_cache_dir", metadata_cache_dir),
    ):
        if value is not None:
            core_cfg[key] = str(value)
    if land_mask_path is not None:
        grid_cfg["land_mask_path"] = str(land_mask_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = output_dir / "depthdif_public_data_config.yaml"
    with public_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return public_config


def _export_args_from_public_api(
    *,
    assets: InferenceAssets,
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None,
    output_root: str | Path,
    device: str,
    batch_size: int | None,
    export_ground_truth: bool,
    full_sample_count: int,
    land_mask_path: str | Path,
    min_ocean_fraction: float,
    sigma: float,
    strict_load: bool,
) -> argparse.Namespace:
    """Build exporter args using the same defaults as the global CLI parser."""
    parser = _build_export_parser()
    args = parser.parse_args(
        ["--year", str(int(year)), "--iso-week", str(int(iso_week))]
    )
    args.model_config = str(assets.model_config)
    args.data_config = str(assets.data_config)
    args.train_config = str(assets.train_config)
    args.checkpoint_path = str(assets.checkpoint)
    args.rectangle = (
        None if rectangle is None else [float(value) for value in rectangle]
    )
    args.output_root = Path(output_root)
    args.device = str(device)
    args.batch_size = batch_size
    args.export_ground_truth = bool(export_ground_truth)
    args.full_sample_count = int(full_sample_count)
    args.land_mask_path = Path(land_mask_path)
    args.min_ocean_fraction = float(min_ocean_fraction)
    args.sigma = float(sigma)
    args.strict_load = bool(strict_load)
    return args


def run_week_inference(
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    checkpoint: str | Path | None = None,
    config_repo: str = DEFAULT_HF_REPO_ID,
    *,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    argo_dir: str | Path | None = None,
    glorys_dir: str | Path | None = None,
    ostia_dir: str | Path | None = None,
    sealevel_dir: str | Path | None = None,
    metadata_cache_dir: str | Path | None = None,
    auto_download_argo: bool = False,
    export_ground_truth: bool = True,
    full_sample_count: int = 0,
    batch_size: int | None = None,
    land_mask_path: str | Path = DEFAULT_LAND_MASK_PATH,
    min_ocean_fraction: float = 0.05,
    sigma: float = DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    strict_load: bool = False,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Run DepthDif inference for one ISO week and return the run directory."""
    assets = resolve_hf_assets(
        config_repo=config_repo,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        downloader=downloader,
    )
    if checkpoint is not None:
        assets = InferenceAssets(
            model_config=assets.model_config,
            data_config=assets.data_config,
            train_config=assets.train_config,
            checkpoint=Path(checkpoint),
        )
    if auto_download_argo:
        argo_target = (
            Path(argo_dir)
            if argo_dir is not None
            else (_default_cache_dir() / "en4_profiles")
        )
        argo_dir = download_argo_for_week(
            year,
            iso_week,
            argo_target,
            cache_dir=None if cache_dir is None else Path(cache_dir) / "en4",
            force_download=force_download,
            downloader=downloader,
        )

    output_root_path = Path(output_root)
    data_config = _write_public_data_config(
        source_config=assets.data_config,
        output_dir=output_root_path,
        argo_dir=argo_dir,
        glorys_dir=glorys_dir,
        ostia_dir=ostia_dir,
        sealevel_dir=sealevel_dir,
        metadata_cache_dir=metadata_cache_dir,
        land_mask_path=land_mask_path,
    )
    public_assets = InferenceAssets(
        model_config=assets.model_config,
        data_config=data_config,
        train_config=assets.train_config,
        checkpoint=assets.checkpoint,
    )
    args = _export_args_from_public_api(
        assets=public_assets,
        year=year,
        iso_week=iso_week,
        rectangle=rectangle,
        output_root=output_root_path,
        device=device,
        batch_size=batch_size,
        export_ground_truth=export_ground_truth,
        full_sample_count=full_sample_count,
        land_mask_path=land_mask_path,
        min_ocean_fraction=min_ocean_fraction,
        sigma=sigma,
        strict_load=strict_load,
    )
    result: ExportRunResult = run_global_inference(args)
    return result.run_dir


def _build_public_parser() -> argparse.ArgumentParser:
    """Build CLI parser for public inference commands."""
    parser = argparse.ArgumentParser(description="DepthDif public inference helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer-week", help="Run one ISO-week inference.")
    infer.add_argument("--year", type=int, required=True)
    infer.add_argument("--iso-week", type=int, required=True)
    infer.add_argument("--rectangle", type=float, nargs=4, default=None)
    infer.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    infer.add_argument("--checkpoint", type=Path, default=None)
    infer.add_argument("--config-repo", default=DEFAULT_HF_REPO_ID)
    infer.add_argument("--revision", default=DEFAULT_HF_REVISION)
    infer.add_argument("--cache-dir", type=Path, default=None)
    infer.add_argument("--argo-dir", type=Path, default=None)
    infer.add_argument("--glorys-dir", type=Path, default=None)
    infer.add_argument("--ostia-dir", type=Path, default=None)
    infer.add_argument("--sealevel-dir", type=Path, default=None)
    infer.add_argument("--metadata-cache-dir", type=Path, default=None)
    infer.add_argument("--auto-download-argo", action="store_true")
    infer.add_argument("--no-export-ground-truth", action="store_true")
    infer.add_argument("--batch-size", type=int, default=None)
    infer.add_argument("--full-sample-count", type=int, default=0)
    infer.add_argument(
        "--land-mask-path", type=Path, default=Path(DEFAULT_LAND_MASK_PATH)
    )
    infer.add_argument("--min-ocean-fraction", type=float, default=0.05)
    infer.add_argument(
        "--sigma", type=float, default=DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA
    )
    infer.add_argument("--strict-load", action="store_true")
    infer.add_argument("--force-download", action="store_true")

    argo = subparsers.add_parser("download-argo", help="Download EN4/ARGO for a week.")
    argo.add_argument("--year", type=int, required=True)
    argo.add_argument("--iso-week", type=int, required=True)
    argo.add_argument("--output-dir", type=Path, required=True)
    argo.add_argument("--base-url", default=DEFAULT_EN4_BASE_URL)
    argo.add_argument("--cache-dir", type=Path, default=None)
    argo.add_argument("--force-download", action="store_true")
    return parser


def infer_week_cli(argv: Sequence[str] | None = None) -> None:
    """Run the public week-inference console command."""
    parser = _build_public_parser()
    args = parser.parse_args(["infer-week", *(sys.argv[1:] if argv is None else argv)])
    run_dir = run_week_inference(
        year=args.year,
        iso_week=args.iso_week,
        rectangle=args.rectangle,
        output_root=args.output_root,
        device=args.device,
        checkpoint=args.checkpoint,
        config_repo=args.config_repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
        argo_dir=args.argo_dir,
        glorys_dir=args.glorys_dir,
        ostia_dir=args.ostia_dir,
        sealevel_dir=args.sealevel_dir,
        metadata_cache_dir=args.metadata_cache_dir,
        auto_download_argo=args.auto_download_argo,
        export_ground_truth=not bool(args.no_export_ground_truth),
        full_sample_count=args.full_sample_count,
        batch_size=args.batch_size,
        land_mask_path=args.land_mask_path,
        min_ocean_fraction=args.min_ocean_fraction,
        sigma=args.sigma,
        strict_load=args.strict_load,
        force_download=args.force_download,
    )
    print(run_dir)


def download_argo_cli(argv: Sequence[str] | None = None) -> None:
    """Run the public ARGO download console command."""
    parser = _build_public_parser()
    args = parser.parse_args(
        ["download-argo", *(sys.argv[1:] if argv is None else argv)]
    )
    output_dir = download_argo_for_week(
        args.year,
        args.iso_week,
        args.output_dir,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
    print(output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Run public API subcommands when invoked as a module."""
    parser = _build_public_parser()
    args = parser.parse_args(argv)
    if args.command == "infer-week":
        run_dir = run_week_inference(
            year=args.year,
            iso_week=args.iso_week,
            rectangle=args.rectangle,
            output_root=args.output_root,
            device=args.device,
            checkpoint=args.checkpoint,
            config_repo=args.config_repo,
            revision=args.revision,
            cache_dir=args.cache_dir,
            argo_dir=args.argo_dir,
            glorys_dir=args.glorys_dir,
            ostia_dir=args.ostia_dir,
            sealevel_dir=args.sealevel_dir,
            metadata_cache_dir=args.metadata_cache_dir,
            auto_download_argo=args.auto_download_argo,
            export_ground_truth=not bool(args.no_export_ground_truth),
            full_sample_count=args.full_sample_count,
            batch_size=args.batch_size,
            land_mask_path=args.land_mask_path,
            min_ocean_fraction=args.min_ocean_fraction,
            sigma=args.sigma,
            strict_load=args.strict_load,
            force_download=args.force_download,
        )
        print(run_dir)
    elif args.command == "download-argo":
        output_dir = download_argo_for_week(
            args.year,
            args.iso_week,
            args.output_dir,
            base_url=args.base_url,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
        print(output_dir)


if __name__ == "__main__":
    main()
