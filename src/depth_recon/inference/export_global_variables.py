# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_global_variables --year 2018 --iso-week 25 --temperature-checkpoint logs/temperature/best.ckpt --salinity-checkpoint logs/salinity/best.ckpt --device cuda --sampler ddim --ddim-steps 200 --uncertainty-sampler ddim --uncertainty-ddim-steps 50 --temporal-sampler ddim --temporal-ddim-steps 50 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe --rclone-remote r2:depth-data/inference_production/globe --rclone-sync-scope globe --output-root inference/outputs --output-name global_variables_2018_W25 --sigma 0 --extra-zoom-levels 0 --full-sample-count 1000
"""Run and package paired temperature/salinity global inference exports."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Any, Sequence

import yaml

from depth_recon.configs.config_resolver_pixel import (
    PIXEL_SCENARIOS,
    load_pixel_inference_config,
)
from depth_recon.inference.core import INFERENCE_SAMPLERS
from depth_recon.inference.export_cesium_globe_assets import (
    DEFAULT_EXTRA_ZOOM_LEVELS,
    DEFAULT_RCLONE_SYNC_SCOPE,
    export_cesium_globe_variable_assets,
)
from depth_recon.inference.export_global import (
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_FULL_SAMPLE_COUNT,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_UNCERTAINTY_NUM_SAMPLES,
    _build_parser as _build_single_export_parser,
    run_global_inference,
)
from depth_recon.inference.export_temporal_consistency_dashboard import (
    DEFAULT_GRID_SIZE_DEGREES,
    DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME,
    export_temporal_dashboard_assets,
)


def _default_output_name(year: int, iso_week: int) -> str:
    """Return the default paired-run directory name."""
    return f"global_variables_{int(year)}_W{int(iso_week):02d}"


def _build_parser() -> argparse.ArgumentParser:
    """Build the dual-variable global export CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run separate temperature and salinity global inference exports, then "
            "package them into one Cesium globe bundle with a variable selector."
        )
    )
    parser.add_argument("--config", type=str, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
        help="Shared config override passed to both variable exports.",
    )
    parser.add_argument("--year", type=int, required=True, help="ISO year filter.")
    parser.add_argument("--iso-week", type=int, required=True, help="ISO week filter.")
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
        help="Use DataParallel for each variable export when multiple CUDA devices are visible.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-num-workers", type=int, default=None)
    parser.add_argument("--inference-prefetch-factor", type=int, default=None)
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
        choices=INFERENCE_SAMPLERS,
        default=None,
        help="Sampler override passed to both variable exports.",
    )
    parser.add_argument(
        "--ddim-steps",
        "--ddim-num-timesteps",
        dest="ddim_num_timesteps",
        type=int,
        default=None,
        help="DDIM step count passed to both variable exports.",
    )
    parser.add_argument(
        "--uncertainty-sampler",
        choices=INFERENCE_SAMPLERS,
        default=None,
        help=(
            "Sampler override used only by ensemble uncertainty exports. "
            "Defaults to the reconstruction sampler."
        ),
    )
    parser.add_argument(
        "--uncertainty-ddim-steps",
        "--uncertainty-ddim-num-timesteps",
        dest="uncertainty_ddim_num_timesteps",
        type=int,
        default=None,
        help="DDIM step count used only by ensemble uncertainty exports.",
    )
    parser.add_argument(
        "--temporal-sampler",
        choices=INFERENCE_SAMPLERS,
        default=None,
        help=(
            "Sampler override used for temporal consistency runs. Defaults to "
            "the uncertainty sampler, then the reconstruction sampler."
        ),
    )
    parser.add_argument(
        "--temporal-ddim-steps",
        "--temporal-ddim-num-timesteps",
        dest="temporal_ddim_num_timesteps",
        type=int,
        default=None,
        help=(
            "DDIM step count used for temporal consistency runs. Defaults to "
            "the uncertainty DDIM steps, then the reconstruction DDIM steps."
        ),
    )
    parser.add_argument(
        "--export-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run and export one stitched uncertainty raster for both "
            "temperature and salinity."
        ),
    )
    parser.add_argument(
        "--uncertainty-num-samples",
        type=int,
        default=DEFAULT_UNCERTAINTY_NUM_SAMPLES,
        help="Repeated generations per variable uncertainty map.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
        help="Extra export-time Gaussian blur sigma passed to both variable exports.",
    )
    parser.add_argument(
        "--full-sample-count",
        type=int,
        default=DEFAULT_FULL_SAMPLE_COUNT,
        help="Full-depth profile sample count passed to each variable export.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Paired run directory name under --output-root.",
    )
    parser.add_argument("--public-base-url", type=str, default=None)
    parser.add_argument("--rclone-remote", type=str, default=None)
    parser.add_argument(
        "--rclone-sync-scope",
        choices=("globe", "run"),
        default=DEFAULT_RCLONE_SYNC_SCOPE,
    )
    parser.add_argument(
        "--extra-zoom-levels", type=int, default=DEFAULT_EXTRA_ZOOM_LEVELS
    )
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--export-temporal-consistency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also run consecutive weekly exports and package temporal consistency "
            "dashboard assets under <output-name>/temporal/."
        ),
    )
    parser.add_argument(
        "--temporal-week-count",
        type=int,
        default=3,
        help="Number of consecutive ISO weeks included when temporal consistency is enabled.",
    )
    parser.add_argument(
        "--temporal-start-year",
        type=int,
        default=None,
        help="Optional temporal window start ISO year. Defaults to --year.",
    )
    parser.add_argument(
        "--temporal-start-iso-week",
        type=int,
        default=None,
        help="Optional temporal window start ISO week. Defaults to --iso-week.",
    )
    parser.add_argument(
        "--temporal-public-base-url",
        type=str,
        default=None,
        help=(
            "Optional hosted base URL for temporal dashboard assets. Defaults to a "
            "sibling 'temporal' URL next to --public-base-url when provided."
        ),
    )
    parser.add_argument(
        "--temporal-rclone-remote",
        type=str,
        default=None,
        help=(
            "Optional rclone destination for temporal dashboard assets. Defaults to "
            "a sibling 'temporal' remote next to --rclone-remote when provided."
        ),
    )
    parser.add_argument(
        "--temporal-grid-size-degrees",
        type=float,
        default=DEFAULT_GRID_SIZE_DEGREES,
    )
    parser.add_argument("--temporal-top-cell-count", type=int, default=24)
    parser.add_argument(
        "--reuse-existing-temporal-runs",
        action="store_true",
        help=(
            "Reuse already exported extra temporal week folders when their "
            "run_summary.yaml files exist."
        ),
    )
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


def _iso_week_sequence(
    *,
    start_year: int,
    start_iso_week: int,
    week_count: int,
) -> list[tuple[int, int]]:
    """Return consecutive ISO `(year, week)` pairs for temporal exports."""
    if int(week_count) < 2:
        raise ValueError("--temporal-week-count must be at least 2.")
    start = date.fromisocalendar(int(start_year), int(start_iso_week), 3)
    weeks: list[tuple[int, int]] = []
    for offset in range(int(week_count)):
        iso = (start + timedelta(weeks=offset)).isocalendar()
        weeks.append((int(iso.year), int(iso.week)))
    return weeks


def _weekly_output_name(year: int, iso_week: int) -> str:
    """Return a stable output folder name for one temporal week."""
    return f"{int(year)}_W{int(iso_week):02d}"


def _sibling_asset_location(value: str | None, sibling_name: str) -> str | None:
    """Return a sibling URL/remote path with the final path component replaced."""
    if value is None:
        return None
    raw = str(value).rstrip("/")
    prefix, separator, leaf = raw.rpartition("/")
    if leaf == sibling_name:
        return raw
    if separator and leaf in {"globe", "temporal"}:
        return f"{prefix}/{sibling_name}"
    return f"{raw}/{sibling_name}"


def _auxiliary_sampler(args: argparse.Namespace) -> str | None:
    """Return sampler used by auxiliary uncertainty/temporal runs by default."""
    return (
        args.uncertainty_sampler
        if args.uncertainty_sampler is not None
        else args.sampler
    )


def _auxiliary_ddim_num_timesteps(args: argparse.Namespace) -> int | None:
    """Return DDIM steps used by auxiliary uncertainty/temporal runs by default."""
    if args.uncertainty_ddim_num_timesteps is not None:
        return args.uncertainty_ddim_num_timesteps
    return args.ddim_num_timesteps


def _temporal_sampler(args: argparse.Namespace) -> str | None:
    """Return sampler used by extra temporal reconstruction runs."""
    if args.temporal_sampler is not None:
        return args.temporal_sampler
    return _auxiliary_sampler(args)


def _temporal_ddim_num_timesteps(args: argparse.Namespace) -> int | None:
    """Return DDIM steps used by extra temporal reconstruction runs."""
    if args.temporal_ddim_num_timesteps is not None:
        return args.temporal_ddim_num_timesteps
    return _auxiliary_ddim_num_timesteps(args)


def _apply_sampling_defaults_from_config(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Populate omitted paired-run sampling options from the inference config."""
    config_bundle = load_pixel_inference_config(
        config_path_value=args.config,
        scenario_override="temperature",
        overrides=list(args.config_overrides or []),
        runtime_config_dir=Path("/tmp/depthdif_inference_configs")
        / f"paired_sampling_{os.getpid()}",
        write_snapshots=False,
    )
    inference_section = config_bundle.inference_cfg.get("inference", {})
    if not isinstance(inference_section, dict):
        return args
    sampling_cfg = inference_section.get("sampling", {})
    uncertainty_sampling_cfg = inference_section.get("uncertainty_sampling", {})
    if not isinstance(sampling_cfg, dict):
        sampling_cfg = {}
    if not isinstance(uncertainty_sampling_cfg, dict):
        uncertainty_sampling_cfg = {}

    # The paired wrapper must materialize YAML defaults before it decides which
    # sampler settings should be forwarded to lower-cost temporal runs.
    if args.sampler is None:
        args.sampler = sampling_cfg.get("sampler")
    if args.ddim_num_timesteps is None:
        args.ddim_num_timesteps = sampling_cfg.get("ddim_num_timesteps")
    if args.uncertainty_sampler is None:
        args.uncertainty_sampler = uncertainty_sampling_cfg.get("sampler")
    if args.uncertainty_ddim_num_timesteps is None:
        args.uncertainty_ddim_num_timesteps = uncertainty_sampling_cfg.get(
            "ddim_num_timesteps"
        )
    return args


def _temporal_sampling_matches_reconstruction(args: argparse.Namespace) -> bool:
    """Return whether current-week reconstruction runs can serve temporal metrics."""
    return (
        _temporal_sampler(args) == args.sampler
        and _temporal_ddim_num_timesteps(args) == args.ddim_num_timesteps
    )


def _temporal_public_base_url(args: argparse.Namespace) -> str | None:
    """Resolve the hosted URL base for temporal dashboard assets."""
    if args.temporal_public_base_url is not None:
        return str(args.temporal_public_base_url)
    return _sibling_asset_location(args.public_base_url, "temporal")


def _temporal_rclone_remote(args: argparse.Namespace) -> str | None:
    """Resolve the rclone destination for temporal dashboard assets."""
    if args.temporal_rclone_remote is not None:
        return str(args.temporal_rclone_remote)
    return _sibling_asset_location(args.rclone_remote, "temporal")


def _single_export_args(
    args: argparse.Namespace,
    *,
    scenario: str,
    checkpoint_path: str,
    paired_run_dir: Path,
) -> argparse.Namespace:
    """Materialize argparse args for one variable-specific global export."""
    if scenario not in PIXEL_SCENARIOS:
        raise ValueError(f"Unsupported scenario for paired export: {scenario}")
    argv = [
        "--config",
        str(args.config),
        "--scenario",
        scenario,
        "--checkpoint",
        str(checkpoint_path),
        "--year",
        str(int(args.year)),
        "--iso-week",
        str(int(args.iso_week)),
        "--split",
        "all",
        "--device",
        str(args.device),
        "--output-root",
        str(paired_run_dir),
        "--output-name",
        scenario,
        "--sigma",
        str(float(args.sigma)),
        "--full-sample-count",
        str(int(args.full_sample_count)),
        "--seed",
        str(int(args.seed)),
        "--export-ground-truth",
        "--uncertainty-num-samples",
        str(int(args.uncertainty_num_samples)),
    ]
    argv.append(
        "--export-uncertainty"
        if bool(args.export_uncertainty)
        else "--no-export-uncertainty"
    )
    if not bool(args.multi_gpu):
        argv.append("--no-multi-gpu")
    if bool(args.strict_load):
        argv.append("--strict-load")
    for override in args.config_overrides or []:
        argv.extend(["--set", str(override)])
    _append_optional_arg(argv, "--sampler", args.sampler)
    _append_optional_arg(argv, "--ddim-steps", args.ddim_num_timesteps)
    _append_optional_arg(argv, "--uncertainty-sampler", _auxiliary_sampler(args))
    _append_optional_arg(
        argv, "--uncertainty-ddim-steps", _auxiliary_ddim_num_timesteps(args)
    )
    _append_optional_arg(argv, "--batch-size", args.batch_size)
    _append_optional_arg(argv, "--inference-num-workers", args.inference_num_workers)
    _append_optional_arg(
        argv, "--inference-prefetch-factor", args.inference_prefetch_factor
    )
    _append_optional_arg(argv, "--patch-stride", args.patch_stride)
    _append_optional_arg(argv, "--min-ocean-fraction", args.min_ocean_fraction)
    _append_optional_arg(argv, "--rectangle", args.rectangle)
    return _build_single_export_parser().parse_args(argv)


def _temporal_single_export_args(
    args: argparse.Namespace,
    *,
    scenario: str,
    checkpoint_path: str,
    year: int,
    iso_week: int,
    output_root: Path,
) -> argparse.Namespace:
    """Materialize argparse args for one extra temporal week export."""
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
        str(output_root),
        "--output-name",
        _weekly_output_name(year, iso_week),
        "--sigma",
        str(float(args.sigma)),
        "--full-sample-count",
        "0",
        "--seed",
        str(int(args.seed)),
        "--export-ground-truth",
        "--no-export-uncertainty",
    ]
    if not bool(args.multi_gpu):
        argv.append("--no-multi-gpu")
    if bool(args.strict_load):
        argv.append("--strict-load")
    for override in args.config_overrides or []:
        argv.extend(["--set", str(override)])
    _append_optional_arg(argv, "--sampler", _temporal_sampler(args))
    _append_optional_arg(argv, "--ddim-steps", _temporal_ddim_num_timesteps(args))
    _append_optional_arg(argv, "--batch-size", args.batch_size)
    _append_optional_arg(argv, "--inference-num-workers", args.inference_num_workers)
    _append_optional_arg(
        argv, "--inference-prefetch-factor", args.inference_prefetch_factor
    )
    _append_optional_arg(argv, "--patch-stride", args.patch_stride)
    _append_optional_arg(argv, "--min-ocean-fraction", args.min_ocean_fraction)
    _append_optional_arg(argv, "--rectangle", args.rectangle)
    return _build_single_export_parser().parse_args(argv)


def _run_or_reuse_temporal_week(
    args: argparse.Namespace,
    *,
    scenario: str,
    checkpoint_path: str,
    year: int,
    iso_week: int,
    output_root: Path,
) -> Path:
    """Run or reuse one extra week needed for temporal aggregation."""
    run_dir = output_root / _weekly_output_name(year, iso_week)
    if (
        bool(args.reuse_existing_temporal_runs)
        and (run_dir / "run_summary.yaml").exists()
    ):
        print(f"Reusing existing temporal {scenario} run: {run_dir}")
        return run_dir
    result = run_global_inference(
        _temporal_single_export_args(
            args,
            scenario=scenario,
            checkpoint_path=checkpoint_path,
            year=year,
            iso_week=iso_week,
            output_root=output_root,
        )
    )
    return Path(result.run_dir)


def _export_temporal_consistency_for_standard_run(
    args: argparse.Namespace,
    *,
    paired_run_dir: Path,
    temperature_result: Any,
    salinity_result: Any,
) -> dict[str, Any]:
    """Run extra temporal weeks and package temporal dashboard assets."""
    start_year = int(args.temporal_start_year or args.year)
    start_iso_week = int(args.temporal_start_iso_week or args.iso_week)
    weeks = _iso_week_sequence(
        start_year=start_year,
        start_iso_week=start_iso_week,
        week_count=int(args.temporal_week_count),
    )
    main_week = (int(args.year), int(args.iso_week))
    existing_main_runs = {
        "temperature": Path(temperature_result.run_dir),
        "salinity": Path(salinity_result.run_dir),
    }
    checkpoints = {
        "temperature": str(args.temperature_checkpoint),
        "salinity": str(args.salinity_checkpoint),
    }
    reuse_main_week = _temporal_sampling_matches_reconstruction(args)
    variable_run_dirs: dict[str, list[Path]] = {"temperature": [], "salinity": []}
    for variable in ("temperature", "salinity"):
        temporal_root = paired_run_dir / "temporal_runs" / variable
        temporal_root.mkdir(parents=True, exist_ok=True)
        for year, iso_week in weeks:
            if (int(year), int(iso_week)) == main_week and reuse_main_week:
                # Reuse the website reconstruction only when it was sampled with
                # the same settings requested for temporal diagnostics.
                variable_run_dirs[variable].append(existing_main_runs[variable])
                continue
            variable_run_dirs[variable].append(
                _run_or_reuse_temporal_week(
                    args,
                    scenario=variable,
                    checkpoint_path=checkpoints[variable],
                    year=year,
                    iso_week=iso_week,
                    output_root=temporal_root,
                )
            )

    temporal_result = export_temporal_dashboard_assets(
        variable_run_dirs=variable_run_dirs,
        output_dir=paired_run_dir / DEFAULT_TEMPORAL_DASHBOARD_DIR_NAME,
        public_base_url=_temporal_public_base_url(args),
        grid_size_degrees=float(args.temporal_grid_size_degrees),
        top_cell_count=int(args.temporal_top_cell_count),
        rclone_remote=_temporal_rclone_remote(args),
        copy_dashboard=True,
    )
    return {
        "enabled": True,
        "start_iso_year": start_year,
        "start_iso_week": start_iso_week,
        "week_count": int(args.temporal_week_count),
        "weeks": [
            {"iso_year": int(year), "iso_week": int(iso_week)}
            for year, iso_week in weeks
        ],
        "variable_run_dirs": {
            variable: [str(run_dir) for run_dir in run_dirs]
            for variable, run_dirs in variable_run_dirs.items()
        },
        "dashboard": temporal_result,
    }


def run_global_variable_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run paired variable exports and package one combined globe directory."""
    args = _apply_sampling_defaults_from_config(args)
    output_name = args.output_name or _default_output_name(args.year, args.iso_week)
    paired_run_dir = Path(args.output_root) / output_name
    paired_run_dir.mkdir(parents=True, exist_ok=True)

    temperature_result = run_global_inference(
        _single_export_args(
            args,
            scenario="temperature",
            checkpoint_path=args.temperature_checkpoint,
            paired_run_dir=paired_run_dir,
        )
    )
    salinity_result = run_global_inference(
        _single_export_args(
            args,
            scenario="salinity",
            checkpoint_path=args.salinity_checkpoint,
            paired_run_dir=paired_run_dir,
        )
    )
    if temperature_result.selected_date != salinity_result.selected_date:
        raise RuntimeError(
            "Temperature and salinity exports selected different dates: "
            f"{temperature_result.selected_date} != {salinity_result.selected_date}."
        )

    packaging_result = export_cesium_globe_variable_assets(
        variable_run_dirs={
            "temperature": temperature_result.run_dir,
            "salinity": salinity_result.run_dir,
        },
        globe_dir=paired_run_dir / "globe",
        public_base_url=args.public_base_url,
        rclone_remote=args.rclone_remote,
        rclone_sync_scope=args.rclone_sync_scope,
        extra_zoom_levels=args.extra_zoom_levels,
    )
    temporal_consistency_result = None
    if bool(args.export_temporal_consistency):
        temporal_consistency_result = _export_temporal_consistency_for_standard_run(
            args,
            paired_run_dir=paired_run_dir,
            temperature_result=temperature_result,
            salinity_result=salinity_result,
        )

    summary = {
        "selected_date": int(temperature_result.selected_date),
        "iso_year": int(temperature_result.iso_year),
        "iso_week": int(temperature_result.iso_week),
        "run_dir": str(paired_run_dir),
        "variables": {
            "temperature": {
                "run_dir": str(temperature_result.run_dir),
                "summary_path": str(temperature_result.summary_path),
                "uncertainty_tif_path": (
                    None
                    if temperature_result.uncertainty_tif_path is None
                    else str(temperature_result.uncertainty_tif_path)
                ),
            },
            "salinity": {
                "run_dir": str(salinity_result.run_dir),
                "summary_path": str(salinity_result.summary_path),
                "uncertainty_tif_path": (
                    None
                    if salinity_result.uncertainty_tif_path is None
                    else str(salinity_result.uncertainty_tif_path)
                ),
            },
        },
        "export_uncertainty": bool(args.export_uncertainty),
        "uncertainty_num_samples": (
            int(args.uncertainty_num_samples) if bool(args.export_uncertainty) else None
        ),
        "globe_packaging": packaging_result,
        "temporal_consistency": temporal_consistency_result,
    }
    summary_path = paired_run_dir / "run_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the paired variable CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_global_variable_inference(args)


if __name__ == "__main__":
    main()
