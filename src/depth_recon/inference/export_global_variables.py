# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_global_variables --year 2018 --iso-week 25 --temperature-checkpoint logs/temperature/best.ckpt --salinity-checkpoint logs/salinity/best.ckpt --device cuda --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe --rclone-remote r2:depth-data/inference_production/globe --rclone-sync-scope globe --output-root inference/outputs --output-name global_variables_2018_W25 --sigma 0 --extra-zoom-levels 0 --full-sample-count -1
"""Run and package paired temperature/salinity global inference exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from depth_recon.configs.config_resolver_pixel import PIXEL_SCENARIOS
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
        help="Full-depth profile sample count passed to both variable exports.",
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
    _append_optional_arg(argv, "--batch-size", args.batch_size)
    _append_optional_arg(argv, "--inference-num-workers", args.inference_num_workers)
    _append_optional_arg(
        argv, "--inference-prefetch-factor", args.inference_prefetch_factor
    )
    _append_optional_arg(argv, "--patch-stride", args.patch_stride)
    _append_optional_arg(argv, "--min-ocean-fraction", args.min_ocean_fraction)
    _append_optional_arg(argv, "--rectangle", args.rectangle)
    return _build_single_export_parser().parse_args(argv)


def run_global_variable_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run paired variable exports and package one combined globe directory."""
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
