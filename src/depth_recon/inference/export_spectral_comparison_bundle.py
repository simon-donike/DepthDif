# Example:
# /work/envs/depth/bin/python -m depth_recon.inference.export_spectral_comparison_bundle --config src/depth_recon/inference/inference_config.yaml --models-config inference/outputs/paper_2018_W25_selection/models_config.yaml --year 2018 --all-weeks --output-dir inference/outputs/spectral_comparison_2018 --device cuda --sampler ddpm --batch-size 1 --inference-num-workers 4 --patch-stride 128 --min-ocean-fraction 0.05 --en4-holdout-fraction 0.2 --seed 7 --wavenumber-output-name wavenumber_spectra --min-wavelength-km 30 --max-wavelength-km 1000 --wavelength-bin-count 32 --basin-overlap-threshold 0.30 --public-base-url https://globe-assets.hyperalislabs.com/inference_production/globe/wavenumber_spectra --rclone-remote r2:depth-data/inference_production/globe/wavenumber_spectra --upload-scope spectral
"""Run paper-week inference bundles and export baseline spectral comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import rasterio
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.export_cesium_globe_assets import _sync_with_rclone
from depth_recon.inference.export_global import INFERENCE_SAMPLERS
from depth_recon.inference.export_paper_metrics import METHOD_LABELS
from depth_recon.inference.export_paper_week import (
    DEFAULT_CLIMATOLOGY_IDW_CHUNK_SIZE,
    DEFAULT_CLIMATOLOGY_IDW_EPS,
    DEFAULT_CLIMATOLOGY_IDW_NEIGHBORS,
    DEFAULT_CLIMATOLOGY_IDW_POWER,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_PROFILE_CHUNK_SIZE,
    DEFAULT_VALIDATION_YEAR,
    _build_parser as _build_paper_week_parser,
    export_paper_week,
)
from depth_recon.inference.export_wavenumber_spectra import (
    ALL_OCEANS_BASIN,
    DEFAULT_BASIN_OVERLAP_THRESHOLD,
    DEFAULT_MAX_WAVELENGTH_KM,
    DEFAULT_MIN_WAVELENGTH_KM,
    DEFAULT_WAVELENGTH_BIN_COUNT,
    DepthLayerSpec,
    VariableRun,
    _json_records,
    _load_yaml,
    _read_patch_window_from_dataset,
    _resolve_run_artifact_path,
    _sanitize_filename,
    _spectral_power_unit_label,
    aggregate_spectra,
    assign_patch_basin_by_overlap,
    radial_wavenumber_spectrum,
    wavelength_bin_centers_km,
    wavelength_bin_edges_km,
    write_spectral_dashboard_assets,
    write_spectrum_plots,
)

DEFAULT_PREDICTION_METHOD = "depthdif"
DEFAULT_WAVENUMBER_OUTPUT_NAME = "wavenumber_spectra"
PAPER_MANIFEST_NAME = "paper_week_manifest.json"
DEFAULT_LAYER_ORDER = (
    "glorys",
    "prediction",
    "climatology",
    "idw",
    "lstm",
    "cnn",
    "unet",
)
DEFAULT_LINE_STYLES = {
    "glorys": ":",
    "prediction": "-",
    "climatology": "--",
    "idw": "-.",
    "lstm": "--",
    "cnn": "-.",
    "unet": "--",
}


@dataclass(frozen=True)
class PaperSpectralRun:
    """Resolved spectral layer set for one paper week and variable."""

    variable: str
    run_dir: Path
    selected_patches_path: Path
    selected_date: int
    iso_year: int
    iso_week: int
    layer_specs: tuple[DepthLayerSpec, ...]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping from ``path``."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON mapping in {path}.")
    return payload


def _resolve_referenced_path(root: Path, raw_path: Any) -> Path:
    """Resolve a manifest path from absolute, cwd, or bundle-root-relative text."""
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    return Path(root) / path


def _load_variable_run_from_manifest(
    root: Path, metadata: dict[str, Any]
) -> VariableRun | None:
    """Load a manifest-referenced single-variable run."""
    raw_summary = metadata.get("summary_path")
    raw_run_dir = metadata.get("run_dir")
    if raw_summary is None and raw_run_dir is None:
        return None
    summary_path = (
        _resolve_referenced_path(root, raw_summary)
        if raw_summary is not None
        else _resolve_referenced_path(root, raw_run_dir) / "run_summary.yaml"
    )
    if not summary_path.exists():
        return None
    summary = _load_yaml(summary_path)
    run_dir = (
        _resolve_referenced_path(root, raw_run_dir)
        if raw_run_dir is not None
        else summary_path.parent
    )
    variable = str(summary.get("variable", "")).strip().lower()
    if not variable:
        return None
    return VariableRun(
        variable=variable,
        run_dir=run_dir,
        summary_path=summary_path,
        summary=summary,
    )


def _method_layer_key(method: str, *, prediction_method: str) -> str:
    """Return the dashboard layer key for one paper method."""
    method_key = str(method).strip().lower()
    return (
        "prediction"
        if method_key == str(prediction_method).strip().lower()
        else method_key
    )


def _method_label(method: str, metadata: dict[str, Any] | None = None) -> str:
    """Return the display label for one paper method."""
    if metadata and metadata.get("label"):
        return str(metadata["label"])
    return METHOD_LABELS.get(str(method), str(method))


def _depth_allowed(
    raw_export: dict[str, Any],
    *,
    depth_suffixes: set[str] | None,
    depth_indices: set[int] | None,
) -> bool:
    """Return whether one depth export matches optional depth filters."""
    suffix = str(raw_export.get("suffix", ""))
    channel_index = int(raw_export.get("channel_index", -1))
    if depth_suffixes is not None and suffix not in depth_suffixes:
        return False
    if depth_indices is not None and channel_index not in depth_indices:
        return False
    return True


def _prediction_layer_specs(
    run: VariableRun,
    *,
    layer: str,
    depth_suffixes: set[str] | None,
    depth_indices: set[int] | None,
) -> list[DepthLayerSpec]:
    """Resolve method prediction rasters as spectral layer specs."""
    specs: list[DepthLayerSpec] = []
    for raw_export in run.summary.get("depth_exports", []):
        if not isinstance(raw_export, dict) or not _depth_allowed(
            raw_export,
            depth_suffixes=depth_suffixes,
            depth_indices=depth_indices,
        ):
            continue
        prediction_path = _resolve_run_artifact_path(
            run.run_dir, raw_export.get("prediction_tif_path")
        )
        if prediction_path is None or not prediction_path.exists():
            continue
        suffix = str(raw_export.get("suffix", "depth"))
        label = str(raw_export.get("label", suffix))
        requested_depth_m = float(raw_export.get("requested_depth_m", np.nan))
        actual_depth_m = float(raw_export.get("actual_depth_m", requested_depth_m))
        channel_index = int(raw_export.get("channel_index", 0))
        specs.append(
            DepthLayerSpec(
                variable=run.variable,
                layer=layer,
                suffix=suffix,
                label=label,
                requested_depth_m=requested_depth_m,
                actual_depth_m=actual_depth_m,
                channel_index=channel_index,
                path=prediction_path,
            )
        )
    return specs


def _glorys_layer_specs(
    root: Path,
    manifest: dict[str, Any],
    *,
    variable: str,
    depth_suffixes: set[str] | None,
    depth_indices: set[int] | None,
) -> list[DepthLayerSpec]:
    """Resolve persisted GLORYS reference rasters from a paper manifest."""
    refs = manifest.get("references", {}).get("glorys", {}).get(variable, {})
    specs: list[DepthLayerSpec] = []
    for raw_export in refs.get("depth_exports", []):
        if not isinstance(raw_export, dict) or not _depth_allowed(
            raw_export,
            depth_suffixes=depth_suffixes,
            depth_indices=depth_indices,
        ):
            continue
        raw_path = raw_export.get("path")
        if raw_path is None:
            continue
        path = _resolve_referenced_path(root, raw_path)
        if not path.exists():
            continue
        suffix = str(raw_export.get("suffix", "depth"))
        label = str(raw_export.get("label", suffix))
        requested_depth_m = float(raw_export.get("requested_depth_m", np.nan))
        actual_depth_m = float(raw_export.get("actual_depth_m", requested_depth_m))
        channel_index = int(raw_export.get("channel_index", 0))
        specs.append(
            DepthLayerSpec(
                variable=variable,
                layer="glorys",
                suffix=suffix,
                label=label,
                requested_depth_m=requested_depth_m,
                actual_depth_m=actual_depth_m,
                channel_index=channel_index,
                path=path,
                band_index=int(raw_export.get("band_index", 1)),
                source_kind="paper_reference",
            )
        )
    return specs


def _depth_template(
    manifest: dict[str, Any],
    *,
    variable: str,
    reference_run: VariableRun,
) -> list[dict[str, Any]]:
    """Return depth metadata used to index full-depth climatology rasters."""
    refs = manifest.get("references", {}).get("glorys", {}).get(variable, {})
    depth_exports = refs.get("depth_exports", [])
    if isinstance(depth_exports, list) and depth_exports:
        return [export for export in depth_exports if isinstance(export, dict)]
    return [
        export
        for export in reference_run.summary.get("depth_exports", [])
        if isinstance(export, dict)
    ]


def _climatology_layer_specs(
    root: Path,
    manifest: dict[str, Any],
    *,
    variable: str,
    reference_run: VariableRun,
    depth_suffixes: set[str] | None,
    depth_indices: set[int] | None,
) -> list[DepthLayerSpec]:
    """Resolve multi-band climatology rasters from a paper manifest."""
    raw_summary = manifest.get("methods", {}).get("climatology", {}).get(
        "climatology_summary_json"
    ) or manifest.get("references", {}).get("climatology_summary_json")
    if raw_summary is None:
        return []
    summary_path = _resolve_referenced_path(root, raw_summary)
    if not summary_path.exists():
        return []
    summary = _load_json(summary_path)
    raw_artifact = summary.get("artifacts", {}).get(variable)
    if raw_artifact is None:
        return []
    artifact_path = _resolve_referenced_path(summary_path.parent, raw_artifact)
    if not artifact_path.exists():
        return []

    specs: list[DepthLayerSpec] = []
    for idx, raw_export in enumerate(
        _depth_template(manifest, variable=variable, reference_run=reference_run)
    ):
        if not _depth_allowed(
            raw_export,
            depth_suffixes=depth_suffixes,
            depth_indices=depth_indices,
        ):
            continue
        suffix = str(raw_export.get("suffix", f"depth_{idx:03d}"))
        label = str(raw_export.get("label", suffix))
        requested_depth_m = float(raw_export.get("requested_depth_m", np.nan))
        actual_depth_m = float(raw_export.get("actual_depth_m", requested_depth_m))
        channel_index = int(raw_export.get("channel_index", idx))
        specs.append(
            DepthLayerSpec(
                variable=variable,
                layer="climatology",
                suffix=suffix,
                label=label,
                requested_depth_m=requested_depth_m,
                actual_depth_m=actual_depth_m,
                channel_index=channel_index,
                path=artifact_path,
                band_index=channel_index + 1,
                source_kind="paper_climatology",
            )
        )
    return specs


def _method_order(
    manifest: dict[str, Any], requested: Sequence[str] | None
) -> list[str]:
    """Return selected paper methods in manifest order."""
    order = [str(method) for method in manifest.get("method_order", [])]
    if not order:
        order = [str(method) for method in manifest.get("methods", {}).keys()]
    if requested:
        requested_set = {str(method).strip().lower() for method in requested}
        order = [method for method in order if method.lower() in requested_set]
    return order


def _reference_run_for_variable(
    root: Path,
    manifest: dict[str, Any],
    *,
    variable: str,
    methods: Sequence[str],
    prediction_method: str,
) -> VariableRun | None:
    """Return a representative method run for patch/date metadata."""
    preferred = [prediction_method] + [
        method for method in methods if method != prediction_method
    ]
    method_metadata = manifest.get("methods", {})
    for method in preferred:
        metadata = method_metadata.get(method, {})
        variable_meta = metadata.get("variables", {}).get(variable)
        if not isinstance(variable_meta, dict):
            continue
        run = _load_variable_run_from_manifest(root, variable_meta)
        if run is not None and (run.run_dir / "selected_patches.csv").exists():
            return run
    return None


def discover_paper_spectral_runs(
    paper_run_dirs: Sequence[Path],
    *,
    variables: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    prediction_method: str = DEFAULT_PREDICTION_METHOD,
    depth_suffixes: set[str] | None = None,
    depth_indices: set[int] | None = None,
) -> tuple[
    list[PaperSpectralRun],
    dict[str, str],
    list[str],
    dict[str, str],
    list[dict[str, Any]],
]:
    """Discover spectral layer runs from paper-week manifests."""
    requested_variables = (
        {"temperature", "salinity"}
        if not variables
        else {str(variable).strip().lower() for variable in variables}
    )
    runs: list[PaperSpectralRun] = []
    skipped: list[dict[str, Any]] = []
    layer_labels = {"glorys": "GLORYS"}
    layer_order = ["glorys"]
    line_styles = dict(DEFAULT_LINE_STYLES)

    for paper_run_dir in paper_run_dirs:
        root = Path(paper_run_dir)
        manifest_path = root / PAPER_MANIFEST_NAME
        if not manifest_path.exists():
            skipped.append({"paper_run_dir": str(root), "reason": "missing_manifest"})
            continue
        manifest = _load_json(manifest_path)
        selected_methods = _method_order(manifest, methods)
        method_metadata = manifest.get("methods", {})
        for method in selected_methods:
            if method == "glorys":
                continue
            layer = _method_layer_key(method, prediction_method=prediction_method)
            if layer not in layer_order:
                layer_order.append(layer)
            layer_labels[layer] = _method_label(method, method_metadata.get(method))

        for variable in sorted(requested_variables):
            reference_run = _reference_run_for_variable(
                root,
                manifest,
                variable=variable,
                methods=selected_methods,
                prediction_method=prediction_method,
            )
            if reference_run is None:
                skipped.append(
                    {
                        "paper_run_dir": str(root),
                        "variable": variable,
                        "reason": "missing_reference_run",
                    }
                )
                continue
            layer_specs: list[DepthLayerSpec] = []
            layer_specs.extend(
                _glorys_layer_specs(
                    root,
                    manifest,
                    variable=variable,
                    depth_suffixes=depth_suffixes,
                    depth_indices=depth_indices,
                )
            )
            if "climatology" in selected_methods:
                layer_specs.extend(
                    _climatology_layer_specs(
                        root,
                        manifest,
                        variable=variable,
                        reference_run=reference_run,
                        depth_suffixes=depth_suffixes,
                        depth_indices=depth_indices,
                    )
                )
            for method in selected_methods:
                if method == "climatology":
                    continue
                metadata = method_metadata.get(method, {})
                variable_meta = metadata.get("variables", {}).get(variable)
                if not isinstance(variable_meta, dict):
                    continue
                variable_run = _load_variable_run_from_manifest(root, variable_meta)
                if variable_run is None:
                    skipped.append(
                        {
                            "paper_run_dir": str(root),
                            "variable": variable,
                            "method": method,
                            "reason": "missing_method_run",
                        }
                    )
                    continue
                layer_specs.extend(
                    _prediction_layer_specs(
                        variable_run,
                        layer=_method_layer_key(
                            method, prediction_method=prediction_method
                        ),
                        depth_suffixes=depth_suffixes,
                        depth_indices=depth_indices,
                    )
                )
            if not layer_specs:
                skipped.append(
                    {
                        "paper_run_dir": str(root),
                        "variable": variable,
                        "reason": "missing_layer_specs",
                    }
                )
                continue
            date_value = int(
                reference_run.summary.get("selected_date")
                or reference_run.summary.get("target_date")
            )
            year = int(reference_run.summary.get("iso_year", manifest.get("year", 0)))
            week = int(
                reference_run.summary.get("iso_week", manifest.get("iso_week", 0))
            )
            runs.append(
                PaperSpectralRun(
                    variable=variable,
                    run_dir=root,
                    selected_patches_path=reference_run.run_dir
                    / "selected_patches.csv",
                    selected_date=date_value,
                    iso_year=year,
                    iso_week=week,
                    layer_specs=tuple(layer_specs),
                )
            )
    ordered_layers = [layer for layer in DEFAULT_LAYER_ORDER if layer in layer_order]
    ordered_layers.extend(layer for layer in layer_order if layer not in ordered_layers)
    return runs, layer_labels, ordered_layers, line_styles, skipped


def _pixel_sizes_km(
    row: dict[str, Any], shape_2d: tuple[int, int]
) -> tuple[float, float]:
    """Return approximate zonal and meridional pixel sizes for one patch."""
    left = min(float(row["lon0"]), float(row["lon1"]))
    right = max(float(row["lon0"]), float(row["lon1"]))
    bottom = min(float(row["lat0"]), float(row["lat1"]))
    top = max(float(row["lat0"]), float(row["lat1"]))
    height, width = int(shape_2d[0]), int(shape_2d[1])
    lat_center = 0.5 * (bottom + top)
    pixel_width_deg = abs(right - left) / float(width)
    pixel_height_deg = abs(top - bottom) / float(height)
    pixel_x = (
        pixel_width_deg
        * 111.32
        * max(abs(float(np.cos(np.deg2rad(lat_center)))), 1.0e-6)
    )
    pixel_y = pixel_height_deg * 111.32
    return float(pixel_x), float(pixel_y)


def _spectra_for_paper_run(
    run: PaperSpectralRun,
    *,
    wavelength_edges: np.ndarray,
    basin_overlap_threshold: float,
    require_complete_patches: bool,
    layer_labels: dict[str, str],
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, int]]:
    """Compute all patch spectra for one paper-week variable run."""
    if not run.selected_patches_path.exists():
        return [], [], {"missing_selected_patches": 1}
    patches = pd.read_csv(run.selected_patches_path).to_dict(orient="records")
    calendar_year, month, season = _date_parts(run.selected_date)
    records: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    skip_counts = {
        "missing_selected_patches": 0,
        "missing_windows": 0,
        "incomplete_patches": 0,
        "empty_spectra": 0,
    }

    basin_cache: dict[str, str | None] = {}
    for layer_spec in run.layer_specs:
        with rasterio.open(layer_spec.path) as dataset:
            for row in patches:
                patch_key = str(row.get("patch_id", len(basin_cache)))
                if patch_key not in basin_cache:
                    basin_cache[patch_key] = assign_patch_basin_by_overlap(
                        row,
                        threshold=basin_overlap_threshold,
                    )
                patch_basin = basin_cache[patch_key]
                patch = _read_patch_window_from_dataset(
                    dataset,
                    row,
                    variable=run.variable,
                    band_index=layer_spec.band_index,
                    decode_uint8=layer_spec.decode_stretched_uint8,
                )
                if patch is None:
                    skip_counts["missing_windows"] += 1
                    continue
                if require_complete_patches and not np.all(np.isfinite(patch)):
                    skip_counts["incomplete_patches"] += 1
                    continue
                pixel_x_km, pixel_y_km = _pixel_sizes_km(row, patch.shape)
                spectrum_result = radial_wavenumber_spectrum(
                    patch,
                    pixel_size_x_km=pixel_x_km,
                    pixel_size_y_km=pixel_y_km,
                    wavelength_edges_km=wavelength_edges,
                    require_complete=require_complete_patches,
                )
                if spectrum_result is None:
                    skip_counts["empty_spectra"] += 1
                    continue
                spectrum, bin_counts = spectrum_result
                if not np.any(np.isfinite(spectrum)):
                    skip_counts["empty_spectra"] += 1
                    continue
                records.append(
                    {
                        "spectrum_index": len(spectra),
                        "variable": run.variable,
                        "layer": layer_spec.layer,
                        "layer_label": layer_labels.get(
                            layer_spec.layer, layer_spec.layer
                        ),
                        "source_kind": layer_spec.source_kind,
                        "run_dir": str(run.run_dir),
                        "selected_date": run.selected_date,
                        "iso_year": run.iso_year,
                        "iso_week": run.iso_week,
                        "year": calendar_year,
                        "month": month,
                        "season": season,
                        "patch_id": str(row.get("patch_id", "")),
                        "grid_y0": int(row.get("grid_y0", -1)),
                        "grid_x0": int(row.get("grid_x0", -1)),
                        "lon0": float(row.get("lon0", np.nan)),
                        "lon1": float(row.get("lon1", np.nan)),
                        "lat0": float(row.get("lat0", np.nan)),
                        "lat1": float(row.get("lat1", np.nan)),
                        "basin": "" if patch_basin is None else patch_basin,
                        "included_in_basin": bool(patch_basin is not None),
                        "depth_suffix": layer_spec.suffix,
                        "depth_label": layer_spec.label,
                        "requested_depth_m": layer_spec.requested_depth_m,
                        "actual_depth_m": layer_spec.actual_depth_m,
                        "channel_index": layer_spec.channel_index,
                        "raster_path": str(layer_spec.path),
                        "raster_band_index": int(layer_spec.band_index),
                        "finite_pixel_count": int(np.count_nonzero(np.isfinite(patch))),
                        "pixel_size_x_km": float(pixel_x_km),
                        "pixel_size_y_km": float(pixel_y_km),
                        "fft_bin_count_total": int(np.sum(bin_counts)),
                    }
                )
                spectra.append(spectrum.astype(np.float32, copy=False))
    return records, spectra, skip_counts


def _date_parts(date_value: int) -> tuple[int, int, str]:
    """Return calendar year, month, and meteorological season for YYYYMMDD."""
    text = str(int(date_value))
    parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    season = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }[int(parsed.month)]
    return int(parsed.year), int(parsed.month), season


def export_paper_method_wavenumber_spectra(
    *,
    paper_run_dirs: Sequence[Path],
    output_dir: Path,
    variables: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    prediction_method: str = DEFAULT_PREDICTION_METHOD,
    depth_suffixes: set[str] | None = None,
    depth_indices: set[int] | None = None,
    min_wavelength_km: float = DEFAULT_MIN_WAVELENGTH_KM,
    max_wavelength_km: float = DEFAULT_MAX_WAVELENGTH_KM,
    wavelength_bin_count: int = DEFAULT_WAVELENGTH_BIN_COUNT,
    basin_overlap_threshold: float = DEFAULT_BASIN_OVERLAP_THRESHOLD,
    require_complete_patches: bool = False,
    write_plots: bool = True,
    write_dashboard: bool = True,
    public_base_url: str | None = None,
    rclone_remote: str | None = None,
) -> dict[str, Any]:
    """Export one spectral dashboard from paper-week method bundles."""
    if not paper_run_dirs:
        raise ValueError("At least one paper run directory is required.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paper_paths = [Path(path) for path in paper_run_dirs]
    (
        spectral_runs,
        layer_labels,
        layer_order,
        line_styles,
        skipped_discovery,
    ) = discover_paper_spectral_runs(
        paper_paths,
        variables=variables,
        methods=methods,
        prediction_method=prediction_method,
        depth_suffixes=depth_suffixes,
        depth_indices=depth_indices,
    )
    if not spectral_runs:
        raise RuntimeError("No paper-week spectral layer runs were discovered.")

    edges = wavelength_bin_edges_km(
        min_wavelength_km=min_wavelength_km,
        max_wavelength_km=max_wavelength_km,
        bin_count=wavelength_bin_count,
    )
    centers = wavelength_bin_centers_km(edges)
    all_records: list[dict[str, Any]] = []
    all_spectra: list[np.ndarray] = []
    skipped_by_run: list[dict[str, Any]] = []
    for run in spectral_runs:
        records, spectra, skip_counts = _spectra_for_paper_run(
            run,
            wavelength_edges=edges,
            basin_overlap_threshold=basin_overlap_threshold,
            require_complete_patches=require_complete_patches,
            layer_labels=layer_labels,
        )
        offset = len(all_spectra)
        for record in records:
            record["spectrum_index"] = int(record["spectrum_index"]) + offset
        all_records.extend(records)
        all_spectra.extend(spectra)
        skipped_by_run.append(
            {
                "variable": run.variable,
                "run_dir": str(run.run_dir),
                **{key: int(value) for key, value in skip_counts.items()},
            }
        )

    spectra_array = (
        np.stack(all_spectra, axis=0).astype(np.float32, copy=False)
        if all_spectra
        else np.zeros((0, centers.size), dtype=np.float32)
    )
    records_df = pd.DataFrame.from_records(all_records)
    aggregated_df = aggregate_spectra(records_df, spectra_array, centers, edges)

    np.savez_compressed(
        output_path / "patch_spectra.npz",
        spectra=spectra_array,
        wavelength_bin_edges_km=edges.astype(np.float32),
        wavelength_bin_centers_km=centers.astype(np.float32),
    )
    records_df.to_csv(output_path / "patch_spectra_records.csv", index=False)
    aggregated_df.to_csv(output_path / "aggregated_spectra.csv", index=False)
    plot_paths = (
        write_spectrum_plots(
            aggregated_df,
            output_dir=output_path,
            layer_labels=layer_labels,
            line_styles=line_styles,
        )
        if write_plots
        else []
    )

    summary = {
        "schema_version": 1,
        "kind": "paper_method_wavenumber_spectra",
        "paper_run_dirs": [str(path) for path in paper_paths],
        "output_dir": str(output_path),
        "variables": sorted({run.variable for run in spectral_runs}),
        "methods": list(methods) if methods else None,
        "prediction_method": str(prediction_method),
        "layer_labels": layer_labels,
        "layer_order": layer_order,
        "line_styles": line_styles,
        "paper_week_count": len({str(run.run_dir) for run in spectral_runs}),
        "run_count": int(len(spectral_runs)),
        "spectrum_count": int(spectra_array.shape[0]),
        "wavelength_bin_count": int(centers.size),
        "wavelength_min_km": float(min_wavelength_km),
        "wavelength_max_km": float(max_wavelength_km),
        "basin_overlap_threshold": float(basin_overlap_threshold),
        "require_complete_patches": bool(require_complete_patches),
        "dashboard_enabled": bool(write_dashboard),
        "public_base_url": public_base_url,
        "upload_requested": rclone_remote is not None,
        "upload_remote": rclone_remote,
        "upload_ok": None,
        "upload_message": None,
        "skipped_discovery": skipped_discovery,
        "skipped_by_run": skipped_by_run,
        "artifacts": {
            "patch_spectra_npz": "patch_spectra.npz",
            "patch_spectra_records_csv": "patch_spectra_records.csv",
            "aggregated_spectra_csv": "aggregated_spectra.csv",
            "summary_json": "summary.json",
            "plot_count": int(len(plot_paths)),
            "plots_dir": "plots",
        },
    }
    if write_dashboard:
        dashboard_artifacts = write_spectral_dashboard_assets(
            aggregated_df,
            output_dir=output_path,
            run_paths=paper_paths,
            wavelength_edges=edges,
            wavelength_centers=centers,
            summary={
                key: value for key, value in summary.items() if key != "artifacts"
            },
            layer_labels=layer_labels,
            layer_order=layer_order,
            line_styles=line_styles,
        )
        summary["artifacts"].update(dashboard_artifacts)
    with (output_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    if rclone_remote is not None:
        ok, message = _sync_with_rclone(output_path, rclone_remote)
        summary["upload_ok"] = bool(ok)
        summary["upload_message"] = str(message)
        with (output_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    return summary


def _iso_week_count(year: int) -> int:
    """Return the number of ISO weeks in ``year``."""
    return int(date(int(year), 12, 28).isocalendar().week)


def _resolve_weeks(args: argparse.Namespace) -> list[int]:
    """Resolve requested ISO weeks from CLI arguments."""
    weeks: set[int] = set()
    if args.all_weeks:
        weeks.update(range(1, _iso_week_count(int(args.year)) + 1))
    if args.iso_week:
        weeks.update(int(week) for week in args.iso_week)
    if args.week_start is not None or args.week_end is not None:
        start = 1 if args.week_start is None else int(args.week_start)
        end = (
            _iso_week_count(int(args.year))
            if args.week_end is None
            else int(args.week_end)
        )
        weeks.update(range(start, end + 1))
    max_week = _iso_week_count(int(args.year))
    invalid = sorted(week for week in weeks if week < 1 or week > max_week)
    if invalid:
        raise ValueError(f"Invalid ISO weeks for {int(args.year)}: {invalid}")
    if not weeks:
        raise ValueError(
            "Select weeks with --all-weeks, --iso-week, or --week-start/--week-end."
        )
    return sorted(weeks)


def _paper_week_output_dir(output_dir: Path, year: int, iso_week: int) -> Path:
    """Return the paper-week bundle directory inside the combined output root."""
    return Path(output_dir) / "weeks" / f"{int(year)}_W{int(iso_week):02d}"


def _paper_week_argv(
    args: argparse.Namespace, *, iso_week: int, output_dir: Path
) -> list[str]:
    """Build argv for the existing paper-week exporter."""
    argv = [
        "--config",
        str(args.config),
        "--year",
        str(int(args.year)),
        "--iso-week",
        str(int(iso_week)),
        "--output-dir",
        str(output_dir),
        "--models-config",
        str(args.models_config),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--sigma",
        str(float(args.sigma)),
        "--full-sample-count",
        str(int(args.full_sample_count)),
        "--validation-year",
        str(int(args.validation_year)),
        "--en4-holdout-fraction",
        str(float(args.en4_holdout_fraction)),
        "--climatology-idw-power",
        str(float(args.climatology_idw_power)),
        "--climatology-idw-eps",
        str(float(args.climatology_idw_eps)),
        "--climatology-idw-neighbors",
        str(int(args.climatology_idw_neighbors)),
        "--climatology-idw-chunk-size",
        str(int(args.climatology_idw_chunk_size)),
        "--profile-chunk-size",
        str(int(args.profile_chunk_size)),
    ]
    if not bool(args.multi_gpu):
        argv.append("--no-multi-gpu")
    if bool(args.strict_load):
        argv.append("--strict-load")
    if bool(args.overwrite_climatology):
        argv.append("--overwrite-climatology")
    for override in args.config_overrides or []:
        argv.extend(["--set", str(override)])
    optional = {
        "--sampler": args.sampler,
        "--ddim-steps": args.ddim_num_timesteps,
        "--batch-size": args.batch_size,
        "--inference-num-workers": args.inference_num_workers,
        "--inference-prefetch-factor": args.inference_prefetch_factor,
        "--patch-stride": args.patch_stride,
        "--min-ocean-fraction": args.min_ocean_fraction,
        "--land-mask-path": args.land_mask_path,
    }
    for flag, value in optional.items():
        if value is not None:
            argv.extend([flag, str(value)])
    if args.rectangle is not None:
        argv.append("--rectangle")
        argv.extend(str(float(value)) for value in args.rectangle)
    return argv


def export_spectral_comparison_bundle(
    args: argparse.Namespace,
    *,
    paper_week_exporter: Callable[
        [argparse.Namespace], dict[str, Any]
    ] = export_paper_week,
) -> dict[str, Any]:
    """Run or reuse paper-week inference bundles, then export spectral data."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weeks = _resolve_weeks(args)
    paper_parser = _build_paper_week_parser()
    paper_run_dirs: list[Path] = []
    week_summaries: list[dict[str, Any]] = []
    for iso_week in weeks:
        week_dir = _paper_week_output_dir(output_dir, int(args.year), int(iso_week))
        manifest_path = week_dir / PAPER_MANIFEST_NAME
        if manifest_path.exists() and not bool(args.rerun_existing):
            manifest = _load_json(manifest_path)
            action = "reused"
        else:
            week_dir.mkdir(parents=True, exist_ok=True)
            paper_args = paper_parser.parse_args(
                _paper_week_argv(args, iso_week=int(iso_week), output_dir=week_dir)
            )
            manifest = paper_week_exporter(paper_args)
            action = "ran"
        paper_run_dirs.append(week_dir)
        week_summaries.append(
            {
                "year": int(args.year),
                "iso_week": int(iso_week),
                "output_dir": str(week_dir),
                "manifest_path": str(manifest_path),
                "selected_date": manifest.get("selected_date"),
                "action": action,
            }
        )

    spectral_output_dir = output_dir / str(args.wavenumber_output_name)
    spectral_summary = export_paper_method_wavenumber_spectra(
        paper_run_dirs=paper_run_dirs,
        output_dir=spectral_output_dir,
        variables=args.variables,
        methods=args.methods,
        prediction_method=str(args.prediction_method),
        depth_suffixes=set(args.depth_suffix or []) if args.depth_suffix else None,
        depth_indices=set(args.depth_index or []) if args.depth_index else None,
        min_wavelength_km=float(args.min_wavelength_km),
        max_wavelength_km=float(args.max_wavelength_km),
        wavelength_bin_count=int(args.wavelength_bin_count),
        basin_overlap_threshold=float(args.basin_overlap_threshold),
        require_complete_patches=bool(args.require_complete_patches),
        write_plots=not bool(args.no_plots),
        write_dashboard=not bool(args.no_dashboard),
        public_base_url=args.public_base_url,
        rclone_remote=None,
    )
    summary = {
        "schema_version": 1,
        "kind": "spectral_comparison_bundle",
        "output_dir": str(output_dir),
        "year": int(args.year),
        "weeks": week_summaries,
        "paper_run_dirs": [str(path) for path in paper_run_dirs],
        "spectral_output_dir": str(spectral_output_dir),
        "spectral_summary": spectral_summary,
        "upload_requested": args.rclone_remote is not None,
        "upload_scope": str(args.upload_scope),
        "upload_remote": args.rclone_remote,
        "upload_ok": None,
        "upload_message": None,
    }
    summary_path = output_dir / "spectral_comparison_bundle_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    if args.rclone_remote is not None:
        sync_source = (
            spectral_output_dir if args.upload_scope == "spectral" else output_dir
        )
        ok, message = _sync_with_rclone(sync_source, str(args.rclone_remote))
        summary["upload_ok"] = bool(ok)
        summary["upload_message"] = str(message)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the spectral comparison bundle CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all requested paper-week inference bundles and export one "
            "baseline-comparison wavenumber spectral dashboard."
        )
    )
    parser.add_argument("--config", type=str, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="TARGET=VALUE",
    )
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--iso-week", type=int, action="append", default=[])
    parser.add_argument("--week-start", type=int, default=None)
    parser.add_argument("--week-end", type=int, default=None)
    parser.add_argument("--all-weeks", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rerun-existing", action="store_true")
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
    parser.add_argument("--sampler", choices=INFERENCE_SAMPLERS, default="ddpm")
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
        "--climatology-idw-power", type=float, default=DEFAULT_CLIMATOLOGY_IDW_POWER
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
    parser.add_argument(
        "--variables", nargs="+", choices=("temperature", "salinity"), default=None
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--prediction-method", default=DEFAULT_PREDICTION_METHOD)
    parser.add_argument("--depth-suffix", action="append", default=[])
    parser.add_argument("--depth-index", type=int, action="append", default=[])
    parser.add_argument(
        "--wavenumber-output-name", default=DEFAULT_WAVENUMBER_OUTPUT_NAME
    )
    parser.add_argument(
        "--min-wavelength-km", type=float, default=DEFAULT_MIN_WAVELENGTH_KM
    )
    parser.add_argument(
        "--max-wavelength-km", type=float, default=DEFAULT_MAX_WAVELENGTH_KM
    )
    parser.add_argument(
        "--wavelength-bin-count", type=int, default=DEFAULT_WAVELENGTH_BIN_COUNT
    )
    parser.add_argument(
        "--basin-overlap-threshold", type=float, default=DEFAULT_BASIN_OVERLAP_THRESHOLD
    )
    parser.add_argument("--require-complete-patches", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--public-base-url", type=str, default=None)
    parser.add_argument("--rclone-remote", type=str, default=None)
    parser.add_argument(
        "--upload-scope", choices=("bundle", "spectral"), default="bundle"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the spectral comparison bundle CLI."""
    args = _build_parser().parse_args(argv)
    summary = export_spectral_comparison_bundle(args)
    print(
        "Wrote spectral comparison bundle: "
        f"{summary['spectral_summary']['spectrum_count']} spectra across "
        f"{summary['spectral_summary']['run_count']} variable-week runs to "
        f"{summary['spectral_output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
