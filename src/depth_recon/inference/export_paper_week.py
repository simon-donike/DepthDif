# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_paper_week \
#   --config src/depth_recon/configs/px_space/inference_super_config.yaml \
#   --set inference.grid.patch_stride=128 \
#   --year 2018 --iso-week 25 \
#   --output-dir inference/outputs/paper_2018_W25 \
#   --models-config configs/paper_week_models.yaml \
#   --device cuda --multi-gpu --strict-load \
#   --batch-size 8 --inference-num-workers 8 --inference-prefetch-factor 2 \
#   --patch-stride 128 --min-ocean-fraction 0.05 \
#   --land-mask-path masks/world_land_mask_glorys_0p1.tif \
#   --rectangle -80 -60 20 70 \
#   --sampler ddim --ddim-steps 100 \
#   --seed 7 --sigma 0.0 --full-sample-count 0 \
#   --validation-year 2018 --en4-holdout-fraction 0.2 \
#   --climatology-idw-power 2.0 --climatology-idw-eps 1.0e-6 \
#   --climatology-idw-neighbors 16 --climatology-idw-chunk-size 250000 \
#   --profile-chunk-size 100000 --overwrite-climatology
"""Export reusable full-native-depth inference artifacts for paper metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.configs.config_resolver_pixel import load_pixel_inference_config
from depth_recon.inference.core import CHECKPOINT_FREE_MODEL_TYPES, INFERENCE_SAMPLERS
from depth_recon.inference.export_global import (
    DEFAULT_INFERENCE_CONFIG,
    _build_parser as _build_single_export_parser,
    _dataset_rows,
    _inference_section,
    _nested_cfg_value,
    _resolve_dataset_root_relative_path,
    resolve_global_inference_dataset,
    run_global_inference,
    select_export_indices,
)
from depth_recon.inference.export_paper_metrics import (
    DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    DEFAULT_CLIMATOLOGY_IDW_EPS,
    DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    DEFAULT_CLIMATOLOGY_IDW_POWER,
    DEFAULT_PROFILE_CHUNK_SIZE,
    DEFAULT_VALIDATION_YEAR,
    VARIABLES,
    build_week_climatology_artifacts,
    load_dataset_context,
    select_en4_holdout_locations,
    write_en4_holdout_profiles_csv,
)


@dataclass(frozen=True)
class MethodSpec:
    """Paper-week model method configuration."""

    name: str
    label: str
    model_type: str
    temperature_checkpoint: str | None = None
    salinity_checkpoint: str | None = None


RunGlobalInference = Callable[[argparse.Namespace], Any]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}.")
    return payload


def _load_method_specs(models_config: Path) -> list[MethodSpec]:
    """Load and validate paper-week method specs from YAML."""
    payload = _load_yaml_mapping(Path(models_config))
    methods = payload.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise RuntimeError("models-config must contain a non-empty 'methods' mapping.")
    specs: list[MethodSpec] = []
    for name, raw_spec in methods.items():
        if not isinstance(raw_spec, dict):
            raise RuntimeError(f"Method {name!r} must be a mapping.")
        method_name = str(name).strip()
        model_type = str(raw_spec.get("model_type", "")).strip()
        if not model_type:
            raise RuntimeError(f"Method {method_name!r} is missing model_type.")
        label = str(raw_spec.get("label", method_name))
        temp_checkpoint = raw_spec.get("temperature_checkpoint")
        sal_checkpoint = raw_spec.get("salinity_checkpoint")
        if model_type not in CHECKPOINT_FREE_MODEL_TYPES:
            missing = []
            if not temp_checkpoint:
                missing.append("temperature_checkpoint")
            if not sal_checkpoint:
                missing.append("salinity_checkpoint")
            if missing:
                raise RuntimeError(
                    f"Method {method_name!r} ({model_type}) is missing: "
                    + ", ".join(missing)
                )
        specs.append(
            MethodSpec(
                name=method_name,
                label=label,
                model_type=model_type,
                temperature_checkpoint=(
                    None if temp_checkpoint is None else str(temp_checkpoint)
                ),
                salinity_checkpoint=(
                    None if sal_checkpoint is None else str(sal_checkpoint)
                ),
            )
        )
    return specs


def _optional_arg(argv: list[str], flag: str, value: Any) -> None:
    """Append an optional CLI argument when value is not None."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend([flag, str(value)])


def _checkpoint_for_variable(method: MethodSpec, variable: str) -> str | None:
    """Return the checkpoint configured for one variable."""
    if variable == "temperature":
        return method.temperature_checkpoint
    if variable == "salinity":
        return method.salinity_checkpoint
    raise ValueError(f"Unsupported variable: {variable}")


def _relative_artifact(root: Path, path: Path | str | None) -> str | None:
    """Return a manifest path relative to root when possible."""
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}.")
    return payload


def _glorys_reference_records(paper_root: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Extract persisted GLORYS depth-export references from a run summary."""
    summary = _read_yaml(Path(run_dir) / "run_summary.yaml")
    records: list[dict[str, Any]] = []
    for raw_export in summary.get("depth_exports", []):
        if not isinstance(raw_export, dict):
            continue
        raw_path = raw_export.get("ground_truth_tif_path")
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(run_dir) / path
        if not path.exists():
            continue
        records.append(
            {
                "suffix": str(raw_export.get("suffix", "depth")),
                "label": str(
                    raw_export.get("label", raw_export.get("suffix", "depth"))
                ),
                "requested_depth_m": float(raw_export.get("requested_depth_m", 0.0)),
                "actual_depth_m": float(raw_export.get("actual_depth_m", 0.0)),
                "channel_index": int(raw_export.get("channel_index", 0)),
                "path": _relative_artifact(paper_root, path),
                "band_index": 1,
            }
        )
    return records


def _resolve_context_bundle(
    *,
    config_path: str,
    config_overrides: Sequence[str],
    year: int,
    iso_week: int,
    output_dir: Path,
    land_mask_path: Path | None,
    min_ocean_fraction: float | None,
    patch_stride: int | None,
) -> tuple[Any, dict[str, Any], int, Path]:
    """Build the dataset context needed before model inference starts."""
    config_bundle = load_pixel_inference_config(
        config_path_value=config_path,
        scenario_override="temperature",
        overrides=list(config_overrides),
        runtime_config_dir=output_dir / ".paper_week_configs" / "context",
        write_snapshots=False,
    )
    data_cfg = config_bundle.data_cfg
    inference_section = _inference_section(config_bundle.inference_cfg)
    inference_grid_cfg = inference_section.get("grid", {})
    if not isinstance(inference_grid_cfg, dict):
        inference_grid_cfg = {}
    raw_land_mask_path = (
        land_mask_path
        if land_mask_path is not None
        else inference_grid_cfg.get(
            "land_mask_path", "masks/world_land_mask_glorys_0p1.tif"
        )
    )
    effective_land_mask_path = _resolve_dataset_root_relative_path(
        data_cfg,
        raw_land_mask_path,
    )
    dataset, _metadata = resolve_global_inference_dataset(
        None,
        data_config_path=config_bundle.effective_data_config_path,
        data_cfg=data_cfg,
        split="all",
        land_mask_path=effective_land_mask_path,
        min_ocean_fraction=(
            0.05 if min_ocean_fraction is None else float(min_ocean_fraction)
        ),
        patch_stride=patch_stride,
    )
    rows = _dataset_rows(dataset)
    selection = select_export_indices(rows, iso_year=int(year), iso_week=int(iso_week))
    dataset_root = Path(
        _nested_cfg_value(
            data_cfg,
            "dataset.core.geotiff_root_dir",
            default="/work/data/OceanVariableReconstruction",
        )
    )
    return config_bundle, data_cfg, int(selection.selected_date), dataset_root


def _single_export_args(
    args: argparse.Namespace,
    *,
    method: MethodSpec,
    variable: str,
    method_root: Path,
    holdout_locations_csv: Path,
    persist_ground_truth_rasters: bool,
) -> argparse.Namespace:
    """Materialize single-variable global-export args for one paper method."""
    argv = [
        "--config",
        str(args.config),
        "--scenario",
        variable,
        "--year",
        str(int(args.year)),
        "--iso-week",
        str(int(args.iso_week)),
        "--split",
        "all",
        "--device",
        str(args.device),
        "--output-root",
        str(method_root),
        "--output-name",
        variable,
        "--seed",
        str(int(args.seed)),
        "--sigma",
        str(float(args.sigma)),
        "--full-sample-count",
        str(int(args.full_sample_count)),
        "--depth-export-mode",
        "native",
        "--export-ground-truth",
        "--no-export-uncertainty",
        "--en4-holdout-locations-csv",
        str(holdout_locations_csv),
    ]
    if not persist_ground_truth_rasters:
        argv.append("--no-persist-ground-truth-rasters")
    checkpoint = _checkpoint_for_variable(method, variable)
    if checkpoint is not None:
        argv.extend(["--checkpoint", str(checkpoint)])
    if not bool(args.multi_gpu):
        argv.append("--no-multi-gpu")
    if bool(args.strict_load):
        argv.append("--strict-load")
    for override in args.config_overrides or []:
        argv.extend(["--set", str(override)])
    argv.extend(["--set", f"model.model_type={method.model_type}"])
    _optional_arg(argv, "--sampler", args.sampler)
    _optional_arg(argv, "--ddim-steps", args.ddim_num_timesteps)
    _optional_arg(argv, "--batch-size", args.batch_size)
    _optional_arg(argv, "--inference-num-workers", args.inference_num_workers)
    _optional_arg(argv, "--inference-prefetch-factor", args.inference_prefetch_factor)
    _optional_arg(argv, "--min-ocean-fraction", args.min_ocean_fraction)
    _optional_arg(argv, "--patch-stride", args.patch_stride)
    _optional_arg(argv, "--land-mask-path", args.land_mask_path)
    _optional_arg(argv, "--rectangle", args.rectangle)
    return _build_single_export_parser().parse_args(argv)


def export_paper_week(
    args: argparse.Namespace,
    *,
    run_inference: RunGlobalInference = run_global_inference,
) -> dict[str, Any]:
    """Run paper-week inference exports and write a bundle manifest."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references_dir = output_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    method_specs = _load_method_specs(Path(args.models_config))

    _config_bundle, _data_cfg, selected_date, dataset_root = _resolve_context_bundle(
        config_path=str(args.config),
        config_overrides=list(args.config_overrides or []),
        year=int(args.year),
        iso_week=int(args.iso_week),
        output_dir=output_dir,
        land_mask_path=args.land_mask_path,
        min_ocean_fraction=args.min_ocean_fraction,
        patch_stride=args.patch_stride,
    )
    context = load_dataset_context(dataset_root)
    holdout_df = select_en4_holdout_locations(
        context=context,
        date_value=int(selected_date),
        fraction=float(args.en4_holdout_fraction),
        seed=int(args.seed),
    )
    holdout_locations_path = references_dir / "en4_holdout_locations.csv"
    holdout_df.to_csv(holdout_locations_path, index=False)
    holdout_profiles_path = write_en4_holdout_profiles_csv(
        context=context,
        holdout_df=holdout_df,
        output_path=references_dir / "en4_holdout_profiles.csv",
    )

    climatology = build_week_climatology_artifacts(
        context=context,
        output_dir=output_dir / "methods" / "climatology",
        date_value=int(selected_date),
        holdout_df=holdout_df,
        idw_power=float(args.climatology_idw_power),
        idw_eps=float(args.climatology_idw_eps),
        idw_neighbors=int(args.climatology_idw_neighbors),
        idw_chunk_size=int(args.climatology_idw_chunk_size),
        profile_chunk_size=int(args.profile_chunk_size),
        overwrite=bool(args.overwrite_climatology),
    )

    methods_manifest: dict[str, Any] = {
        "climatology": {
            "kind": "climatology",
            "label": "Climatology",
            "climatology_summary_json": _relative_artifact(
                output_dir, climatology.summary_path
            ),
        }
    }
    glorys_refs: dict[str, Any] = {}
    method_order = ["climatology"]

    for method in method_specs:
        method_order.append(method.name)
        method_root = output_dir / "methods" / method.name
        method_root.mkdir(parents=True, exist_ok=True)
        variables_manifest: dict[str, Any] = {}
        for variable in VARIABLES:
            persist_gt = variable not in glorys_refs
            result = run_inference(
                _single_export_args(
                    args,
                    method=method,
                    variable=variable,
                    method_root=method_root,
                    holdout_locations_csv=holdout_locations_path,
                    persist_ground_truth_rasters=persist_gt,
                )
            )
            if int(result.selected_date) != int(selected_date):
                raise RuntimeError(
                    f"{method.name} {variable} selected date {int(result.selected_date)} "
                    f"does not match bundle date {int(selected_date)}."
                )
            run_dir = Path(result.run_dir)
            variables_manifest[variable] = {
                "run_dir": _relative_artifact(output_dir, run_dir),
                "summary_path": _relative_artifact(output_dir, result.summary_path),
            }
            if persist_gt:
                records = _glorys_reference_records(output_dir, run_dir)
                if records:
                    glorys_refs[variable] = {"depth_exports": records}
        with (method_root / "run_summary.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump({"variables": variables_manifest}, f, sort_keys=False)
        methods_manifest[method.name] = {
            "kind": "model",
            "label": method.label,
            "model_type": method.model_type,
            "run_dir": _relative_artifact(output_dir, method_root),
            "variables": variables_manifest,
        }

    manifest = {
        "schema_version": 1,
        "kind": "paper_week_inference_bundle",
        "year": int(args.year),
        "iso_week": int(args.iso_week),
        "selected_date": int(selected_date),
        "validation_year": int(args.validation_year),
        "en4_holdout_fraction": float(args.en4_holdout_fraction),
        "seed": int(args.seed),
        "dataset_root": str(dataset_root),
        "variables": list(VARIABLES),
        "depth_export_mode": "native",
        "method_order": method_order,
        "methods": methods_manifest,
        "references": {
            "en4_holdout_locations_csv": _relative_artifact(
                output_dir, holdout_locations_path
            ),
            "en4_holdout_profiles_csv": _relative_artifact(
                output_dir, holdout_profiles_path
            ),
            "climatology_summary_json": _relative_artifact(
                output_dir, climatology.summary_path
            ),
            "glorys": glorys_refs,
        },
    }
    manifest_path = output_dir / "paper_week_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the paper-week export CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run full-native-depth paper-week inference exports."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--iso-week", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--multi-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--inference-num-workers", type=int, default=None)
    parser.add_argument("--inference-prefetch-factor", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--min-ocean-fraction", type=float, default=None)
    parser.add_argument("--land-mask-path", type=Path, default=None)
    parser.add_argument(
        "--rectangle",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=None,
    )
    parser.add_argument("--sampler", choices=INFERENCE_SAMPLERS, default=None)
    parser.add_argument(
        "--ddim-steps",
        "--ddim-num-timesteps",
        dest="ddim_num_timesteps",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--full-sample-count", type=int, default=0)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument("--en4-holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--climatology-idw-power",
        type=float,
        default=DEFAULT_CLIMATOLOGY_IDW_POWER,
    )
    parser.add_argument(
        "--climatology-idw-eps", type=float, default=DEFAULT_CLIMATOLOGY_IDW_EPS
    )
    parser.add_argument(
        "--climatology-idw-neighbors",
        type=int,
        default=DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    )
    parser.add_argument(
        "--climatology-idw-chunk-size",
        type=int,
        default=DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    )
    parser.add_argument(
        "--profile-chunk-size", type=int, default=DEFAULT_PROFILE_CHUNK_SIZE
    )
    parser.add_argument("--overwrite-climatology", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the paper-week export CLI."""
    args = _build_parser().parse_args(argv)
    manifest = export_paper_week(args)
    print(
        f"Wrote paper-week manifest: {Path(args.output_dir) / 'paper_week_manifest.json'}"
    )
    print(f"Selected date: {manifest['selected_date']}")


if __name__ == "__main__":
    main()
