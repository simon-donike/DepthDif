# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_temporal_global_variables --start-year 2018 --start-iso-week 1 --week-count 52 --temperature-checkpoint logs/temperature/best.ckpt --salinity-checkpoint logs/salinity/best.ckpt --device cuda --temporal-sampler ddim --temporal-ddim-steps 50 --output-root inference/outputs --output-name temporal_variables_2018 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/temporal --rclone-remote r2:depth-data/inference_production/temporal
"""Run weekly temperature/salinity exports and package temporal diagnostics."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import yaml

from depth_recon.configs.config_resolver_pixel import PIXEL_SCENARIOS
from depth_recon.inference.core import INFERENCE_SAMPLERS
from depth_recon.inference.export_cesium_globe_assets import DEFAULT_EXTRA_ZOOM_LEVELS
from depth_recon.inference.export_global import (
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_INFERENCE_NUM_WORKERS,
    DEFAULT_INFERENCE_PREFETCH_FACTOR,
    DEFAULT_OUTPUT_ROOT,
    _build_parser as _build_single_export_parser,
    run_global_inference,
)
from depth_recon.inference.export_temporal_consistency_dashboard import (
    DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME,
    DEFAULT_TEMPORAL_VALIDATION_YEAR,
    DEFAULT_TEMPORAL_YEAR_WEEK_COUNT,
    export_temporal_dashboard_assets,
)
from depth_recon.inference.export_temporal_cesium_globe_assets import (
    DEFAULT_TEMPORAL_GLOBE_DIR_NAME,
    DEFAULT_TEMPORAL_GLOBE_MAX_ZOOM_LEVEL,
    DEFAULT_TEMPORAL_GLOBE_WEBP_QUALITY,
    export_temporal_cesium_globe_assets,
)


def _iso_week_sequence(
    *,
    start_year: int,
    start_iso_week: int,
    week_count: int,
) -> list[tuple[int, int]]:
    """Return consecutive ISO `(year, week)` pairs."""
    if int(week_count) < 2:
        raise ValueError("--week-count must be at least 2.")
    start = date.fromisocalendar(int(start_year), int(start_iso_week), 3)
    weeks: list[tuple[int, int]] = []
    for offset in range(int(week_count)):
        iso = (start + timedelta(weeks=offset)).isocalendar()
        weeks.append((int(iso.year), int(iso.week)))
    return weeks


def _default_output_name(start_year: int, start_iso_week: int, week_count: int) -> str:
    """Return the default temporal paired-run directory name."""
    weeks = _iso_week_sequence(
        start_year=start_year,
        start_iso_week=start_iso_week,
        week_count=week_count,
    )
    end_year, end_week = weeks[-1]
    return (
        f"temporal_variables_{int(start_year)}_W{int(start_iso_week):02d}_"
        f"{int(end_year)}_W{int(end_week):02d}"
    )


def _sibling_asset_location(value: str | None, sibling_name: str) -> str | None:
    """Return a sibling URL/remote path with the final path component replaced."""
    if value is None:
        return None
    raw = str(value).rstrip("/")
    prefix, separator, leaf = raw.rpartition("/")
    if leaf == sibling_name:
        return raw
    if separator and leaf in {"globe", "temporal", "temporal-globe"}:
        return f"{prefix}/{sibling_name}"
    return f"{raw}/{sibling_name}"


def _temporal_globe_public_base_url(args: argparse.Namespace) -> str | None:
    """Resolve the hosted URL base for temporal globe assets."""
    if args.temporal_globe_public_base_url is not None:
        return str(args.temporal_globe_public_base_url)
    return _sibling_asset_location(args.public_base_url, "temporal-globe")


def _temporal_globe_rclone_remote(args: argparse.Namespace) -> str | None:
    """Resolve the rclone destination for temporal globe assets."""
    if args.temporal_globe_rclone_remote is not None:
        return str(args.temporal_globe_rclone_remote)
    return _sibling_asset_location(args.rclone_remote, "temporal-globe")


def _build_parser() -> argparse.ArgumentParser:
    """Build the temporal multi-week export CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run consecutive weekly temperature/salinity global exports and "
            "package temporal dashboard assets."
        )
    )
    parser.add_argument("--config", type=str, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
        help="Shared config override passed to all weekly variable exports.",
    )
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_TEMPORAL_VALIDATION_YEAR
    )
    parser.add_argument("--start-iso-week", type=int, default=1)
    parser.add_argument(
        "--week-count", type=int, default=DEFAULT_TEMPORAL_YEAR_WEEK_COUNT
    )
    parser.add_argument(
        "--temperature-checkpoint",
        type=str,
        required=True,
        help="Temperature checkpoint path used with --scenario temperature.",
    )
    parser.add_argument(
        "--salinity-checkpoint",
        type=str,
        required=True,
        help="Salinity checkpoint path used with --scenario salinity.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DataParallel for each weekly export when multiple CUDA devices are visible.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--inference-num-workers",
        type=int,
        default=DEFAULT_INFERENCE_NUM_WORKERS,
    )
    parser.add_argument(
        "--inference-prefetch-factor",
        type=int,
        default=DEFAULT_INFERENCE_PREFETCH_FACTOR,
    )
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--min-ocean-fraction", type=float, default=None)
    parser.add_argument(
        "--rectangle",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--sampler",
        "--sampling-method",
        "--temporal-sampler",
        choices=INFERENCE_SAMPLERS,
        dest="sampler",
        default=None,
        help="Sampler override passed to every weekly variable export.",
    )
    parser.add_argument(
        "--ddim-steps",
        "--ddim-num-timesteps",
        "--temporal-ddim-steps",
        "--temporal-ddim-num-timesteps",
        dest="ddim_num_timesteps",
        type=int,
        default=None,
        help="DDIM step count passed to every weekly variable export.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    )
    parser.add_argument(
        "--full-sample-count",
        type=int,
        default=0,
        help="Profile graph count passed to each weekly export. Defaults to 0 for temporal runs.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--public-base-url", type=str, default=None)
    parser.add_argument("--rclone-remote", type=str, default=None)
    parser.add_argument(
        "--export-temporal-globe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also tile weekly 10m rasters into a lightweight temporal Cesium globe.",
    )
    parser.add_argument("--temporal-globe-public-base-url", type=str, default=None)
    parser.add_argument("--temporal-globe-rclone-remote", type=str, default=None)
    parser.add_argument(
        "--temporal-globe-extra-zoom-levels",
        type=int,
        default=DEFAULT_EXTRA_ZOOM_LEVELS,
    )
    parser.add_argument(
        "--temporal-globe-max-zoom-level",
        type=int,
        default=DEFAULT_TEMPORAL_GLOBE_MAX_ZOOM_LEVEL,
    )
    parser.add_argument(
        "--temporal-globe-webp-quality",
        type=int,
        default=DEFAULT_TEMPORAL_GLOBE_WEBP_QUALITY,
    )
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--reuse-existing-runs",
        action="store_true",
        help="Skip a weekly export when its expected run_summary.yaml already exists.",
    )
    parser.add_argument(
        "--grid-size-degrees",
        type=float,
        default=None,
        help="Temporal dashboard grid size. Defaults to the standard dashboard grid.",
    )
    parser.add_argument("--top-cell-count", type=int, default=24)
    return parser


def _append_optional_arg(argv: list[str], flag: str, value: Any) -> None:
    """Append an optional CLI argument when its value is not None."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend([flag, str(value)])


def _weekly_output_name(year: int, iso_week: int) -> str:
    """Return a stable directory name for one weekly run."""
    return f"{int(year)}_W{int(iso_week):02d}"


def _single_export_args(
    args: argparse.Namespace,
    *,
    scenario: str,
    checkpoint_path: str,
    year: int,
    iso_week: int,
    variable_run_root: Path,
) -> argparse.Namespace:
    """Materialize argparse args for one variable/week global export."""
    if scenario not in PIXEL_SCENARIOS:
        raise ValueError(f"Unsupported temporal scenario: {scenario}")
    argv = [
        "--config",
        str(args.config),
        "--scenario",
        scenario,
        "--checkpoint",
        str(checkpoint_path),
        "--year",
        str(int(year)),
        "--iso-week",
        str(int(iso_week)),
        "--split",
        "all",
        "--device",
        str(args.device),
        "--output-root",
        str(variable_run_root),
        "--output-name",
        _weekly_output_name(year, iso_week),
        "--sigma",
        str(float(args.sigma)),
        "--full-sample-count",
        str(int(args.full_sample_count)),
        "--seed",
        str(int(args.seed)),
        "--export-ground-truth",
        "--no-export-uncertainty",
        "--depth-export-suffix",
        "10m",
        "--compact-basin-depth-error",
        "--no-persist-ground-truth-rasters",
    ]
    if not bool(args.multi_gpu):
        argv.append("--no-multi-gpu")
    if bool(args.strict_load):
        argv.append("--strict-load")
    for override in args.config_overrides or []:
        argv.extend(["--set", str(override)])
    _append_optional_arg(argv, "--sampler", args.sampler)
    _append_optional_arg(argv, "--ddim-steps", args.ddim_num_timesteps)
    _append_optional_arg(argv, "--batch-size", args.batch_size)
    _append_optional_arg(argv, "--inference-num-workers", args.inference_num_workers)
    _append_optional_arg(
        argv,
        "--inference-prefetch-factor",
        args.inference_prefetch_factor,
    )
    _append_optional_arg(argv, "--patch-stride", args.patch_stride)
    _append_optional_arg(argv, "--min-ocean-fraction", args.min_ocean_fraction)
    _append_optional_arg(argv, "--rectangle", args.rectangle)
    return _build_single_export_parser().parse_args(argv)


def _run_or_reuse_weekly_export(
    args: argparse.Namespace,
    *,
    scenario: str,
    checkpoint_path: str,
    year: int,
    iso_week: int,
    variable_run_root: Path,
) -> Path:
    """Run one weekly export or reuse an existing output directory."""
    run_dir = variable_run_root / _weekly_output_name(year, iso_week)
    if bool(args.reuse_existing_runs) and (run_dir / "run_summary.yaml").exists():
        print(f"Reusing existing {scenario} run: {run_dir}")
        return run_dir

    result = run_global_inference(
        _single_export_args(
            args,
            scenario=scenario,
            checkpoint_path=checkpoint_path,
            year=year,
            iso_week=iso_week,
            variable_run_root=variable_run_root,
        )
    )
    return Path(result.run_dir)


def run_temporal_global_variable_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run weekly paired variable exports and package temporal dashboard assets."""
    output_name = args.output_name or _default_output_name(
        args.start_year,
        args.start_iso_week,
        args.week_count,
    )
    paired_run_dir = Path(args.output_root) / output_name
    runs_root = paired_run_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    weeks = _iso_week_sequence(
        start_year=args.start_year,
        start_iso_week=args.start_iso_week,
        week_count=args.week_count,
    )

    variable_run_dirs: dict[str, list[Path]] = {"temperature": [], "salinity": []}
    for variable, checkpoint_path in (
        ("temperature", args.temperature_checkpoint),
        ("salinity", args.salinity_checkpoint),
    ):
        variable_run_root = runs_root / variable
        variable_run_root.mkdir(parents=True, exist_ok=True)
        for year, iso_week in weeks:
            variable_run_dirs[variable].append(
                _run_or_reuse_weekly_export(
                    args,
                    scenario=variable,
                    checkpoint_path=checkpoint_path,
                    year=year,
                    iso_week=iso_week,
                    variable_run_root=variable_run_root,
                )
            )

    temporal_result = export_temporal_dashboard_assets(
        variable_run_dirs=variable_run_dirs,
        output_dir=paired_run_dir / DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME,
        public_base_url=args.public_base_url,
        grid_size_degrees=args.grid_size_degrees,
        top_cell_count=int(args.top_cell_count),
        rclone_remote=args.rclone_remote,
        copy_dashboard=True,
        validation_year=int(args.start_year),
    )
    temporal_globe_result = None
    if bool(args.export_temporal_globe):
        temporal_globe_result = export_temporal_cesium_globe_assets(
            variable_run_dirs=variable_run_dirs,
            output_dir=paired_run_dir / DEFAULT_TEMPORAL_GLOBE_DIR_NAME,
            public_base_url=_temporal_globe_public_base_url(args),
            rclone_remote=_temporal_globe_rclone_remote(args),
            copy_viewer=True,
            validation_year=int(args.start_year),
            extra_zoom_levels=int(args.temporal_globe_extra_zoom_levels),
            max_zoom_level=args.temporal_globe_max_zoom_level,
            webp_quality=int(args.temporal_globe_webp_quality),
        )
    summary = {
        "start_year": int(args.start_year),
        "start_iso_week": int(args.start_iso_week),
        "week_count": int(args.week_count),
        "weeks": [
            {"iso_year": int(year), "iso_week": int(iso_week)}
            for year, iso_week in weeks
        ],
        "run_dir": str(paired_run_dir),
        "variables": {
            variable: [str(run_dir) for run_dir in run_dirs]
            for variable, run_dirs in variable_run_dirs.items()
        },
        "temporal_dashboard": temporal_result,
        "temporal_globe": temporal_globe_result,
    }
    summary_path = paired_run_dir / "run_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the temporal paired-variable CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_temporal_global_variable_inference(args)


if __name__ == "__main__":
    main()
