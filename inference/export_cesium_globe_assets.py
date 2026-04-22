"""Convert one global inference export run into Cesium-hostable assets.

Typical CLI:
    /work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
      --run-dir inference/outputs/global_top_band_20100104 \
      --public-base-url https://example-bucket/path/global_top_band_20100104/globe/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import rasterio
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TEMPLATE_PATH = Path("inference/transforms/globe-config.template.json")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_existing_path(path_value: str | None, *, run_dir: Path) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        run_relative = run_dir / candidate.name
        if run_relative.exists():
            return run_relative
    return None


def _resolve_run_artifacts(run_dir: Path) -> tuple[Path, Path | None, Path | None, dict[str, Any]]:
    run_summary_path = run_dir / "run_summary.yaml"
    run_summary = _load_yaml(run_summary_path) if run_summary_path.exists() else {}

    prediction_path = _coerce_existing_path(
        run_summary.get("prediction_tif_path"),
        run_dir=run_dir,
    )
    if prediction_path is None:
        matches = sorted(run_dir.glob("*_prediction.tif"))
        if not matches:
            raise FileNotFoundError(
                "Could not locate the prediction GeoTIFF. "
                "Expected run_summary.yaml or one '*_prediction.tif' inside the run directory."
            )
        prediction_path = matches[0]

    ground_truth_path = _coerce_existing_path(
        run_summary.get("ground_truth_tif_path"),
        run_dir=run_dir,
    )
    if ground_truth_path is None:
        matches = sorted(run_dir.glob("*_glorys_top_band.tif"))
        ground_truth_path = matches[0] if matches else None

    points_path = _coerce_existing_path(
        run_summary.get("argo_points_geojson_path"),
        run_dir=run_dir,
    )
    if points_path is None:
        matches = sorted(run_dir.glob("*_argo_points.geojson"))
        points_path = matches[0] if matches else None

    return prediction_path, ground_truth_path, points_path, run_summary


def _read_raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        bounds = ds.bounds
        tags = ds.tags()
        center_lon = 0.5 * (float(bounds.left) + float(bounds.right))
        center_lat = 0.5 * (float(bounds.bottom) + float(bounds.top))
        span_degrees = max(
            abs(float(bounds.right) - float(bounds.left)),
            abs(float(bounds.top) - float(bounds.bottom)),
        )
        camera_height = max(2_000_000.0, span_degrees * 111_000.0 * 3.0)
        return {
            "west": float(bounds.left),
            "south": float(bounds.bottom),
            "east": float(bounds.right),
            "north": float(bounds.top),
            "default_camera_destination": {
                "lon": float(center_lon),
                "lat": float(center_lat),
                "height": float(camera_height),
            },
            "credit": str(tags.get("source", path.name)),
        }


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _iter_valid_values(path: Path):
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        for _, window in ds.block_windows(1):
            block = ds.read(1, window=window, masked=False)
            valid_mask = np.isfinite(block)
            if nodata is not None:
                valid_mask &= block != nodata
            if not np.any(valid_mask):
                continue
            yield block[valid_mask]


def _compute_scale_range(paths: list[Path]) -> tuple[float, float]:
    data_min: float | None = None
    data_max: float | None = None
    for path in paths:
        for values in _iter_valid_values(path):
            block_min = float(values.min())
            block_max = float(values.max())
            data_min = block_min if data_min is None else min(data_min, block_min)
            data_max = block_max if data_max is None else max(data_max, block_max)

    if data_min is None or data_max is None:
        raise RuntimeError("Could not determine a valid data range for raster tiling.")

    if data_min == data_max:
        # Keep a non-zero scale range so gdal_translate can still build a byte raster.
        epsilon = 1.0 if data_min == 0.0 else max(0.1, abs(data_min) * 0.01)
        data_min -= epsilon
        data_max += epsilon
    return data_min, data_max


def _convert_raster_to_byte(
    input_path: Path,
    output_path: Path,
    *,
    src_min: float,
    src_max: float,
) -> None:
    gdal_translate_exe = shutil.which("gdal_translate")
    if gdal_translate_exe is None:
        raise RuntimeError("gdal_translate was not found on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gdal_translate_exe,
        "-of",
        "GTiff",
        "-ot",
        "Byte",
        "-scale",
        f"{src_min}",
        f"{src_max}",
        "1",
        "255",
        "-a_nodata",
        "0",
        str(input_path),
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _run_gdal2tiles(input_path: Path, output_dir: Path) -> None:
    gdal2tiles_exe = shutil.which("gdal2tiles.py")
    if gdal2tiles_exe is None:
        raise RuntimeError("gdal2tiles.py was not found on PATH.")

    _ensure_clean_directory(output_dir)
    command = [
        gdal2tiles_exe,
        "-p",
        "mercator",
        "-w",
        "none",
        str(input_path),
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def _sync_with_rclone(local_dir: Path, remote: str) -> tuple[bool, str]:
    rclone_exe = shutil.which("rclone")
    if rclone_exe is None:
        return False, "rclone was not found on PATH. Skipping upload."

    command = [
        rclone_exe,
        "sync",
        str(local_dir),
        str(remote),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output_parts = [part.strip() for part in [exc.stdout, exc.stderr] if part and part.strip()]
        suffix = f" Output: {' | '.join(output_parts)}" if output_parts else ""
        return False, f"rclone sync failed with exit code {exc.returncode}.{suffix}"

    output_parts = [
        part.strip() for part in [completed.stdout, completed.stderr] if part and part.strip()
    ]
    message = "rclone sync completed successfully."
    if output_parts:
        message = f"{message} Output: {' | '.join(output_parts)}"
    return True, message


def _resolve_layer_url(name: str, *, public_base_url: str | None) -> str:
    if public_base_url is None:
        return f"./{name}"
    return f"{public_base_url.rstrip('/')}/{name}"


def build_globe_config(
    *,
    selected_date: int | None,
    prediction_tiles_url: str,
    ground_truth_tiles_url: str | None,
    argo_points_url: str | None,
    bounds: dict[str, Any],
    prediction_credit: str,
    ground_truth_credit: str | None,
    points_credit: str | None,
    template: dict[str, Any],
) -> dict[str, Any]:
    config = dict(template)
    config.update(
        {
            "selected_date": selected_date,
            "prediction_tiles_url": prediction_tiles_url,
            "ground_truth_tiles_url": ground_truth_tiles_url,
            "argo_points_url": argo_points_url,
            "west": float(bounds["west"]),
            "south": float(bounds["south"]),
            "east": float(bounds["east"]),
            "north": float(bounds["north"]),
            "default_camera_destination": dict(bounds["default_camera_destination"]),
        }
    )
    credits = dict(config.get("credits", {}))
    credits["prediction"] = prediction_credit
    if ground_truth_credit is not None:
        credits["ground_truth"] = ground_truth_credit
    if points_credit is not None:
        credits["points"] = points_credit
    config["credits"] = credits
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one exported global inference run into Cesium-hostable TMS imagery "
            "plus GeoJSON and a globe-config.json manifest."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="One completed inference run directory under inference/outputs/.",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help=(
            "Optional public base URL where the generated globe directory will be hosted. "
            "When omitted, the config uses local relative URLs."
        ),
    )
    parser.add_argument(
        "--globe-dir-name",
        type=str,
        default="globe",
        help="Subdirectory name created inside the run directory for the hosted assets.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to the JSON template used as the starting point for globe-config.json.",
    )
    parser.add_argument(
        "--rclone-remote",
        type=str,
        default=None,
        help=(
            "Optional rclone destination such as 'r2:depth-data/global_top_band_20150615/globe'. "
            "When provided, the generated globe directory is synced after export."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    prediction_path, ground_truth_path, points_path, run_summary = _resolve_run_artifacts(run_dir)

    globe_dir = run_dir / str(args.globe_dir_name)
    globe_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = globe_dir / ".tmp_byte_rasters"
    _ensure_clean_directory(temp_dir)

    scale_paths = [prediction_path]
    if ground_truth_path is not None:
        scale_paths.append(ground_truth_path)
    scale_min, scale_max = _compute_scale_range(scale_paths)

    prediction_byte_path = temp_dir / f"{prediction_path.stem}_byte.tif"
    _convert_raster_to_byte(
        prediction_path,
        prediction_byte_path,
        src_min=scale_min,
        src_max=scale_max,
    )
    prediction_tiles_dir = globe_dir / "prediction_tiles"
    _run_gdal2tiles(prediction_byte_path, prediction_tiles_dir)

    ground_truth_tiles_dir: Path | None = None
    if ground_truth_path is not None:
        ground_truth_byte_path = temp_dir / f"{ground_truth_path.stem}_byte.tif"
        _convert_raster_to_byte(
            ground_truth_path,
            ground_truth_byte_path,
            src_min=scale_min,
            src_max=scale_max,
        )
        ground_truth_tiles_dir = globe_dir / "ground_truth_tiles"
        _run_gdal2tiles(ground_truth_byte_path, ground_truth_tiles_dir)

    copied_points_path: Path | None = None
    if points_path is not None:
        copied_points_path = globe_dir / "argo_points.geojson"
        shutil.copy2(points_path, copied_points_path)

    prediction_meta = _read_raster_metadata(prediction_path)
    ground_truth_meta = (
        _read_raster_metadata(ground_truth_path) if ground_truth_path is not None else None
    )
    template = _load_template(Path(args.template_path))
    config = build_globe_config(
        selected_date=run_summary.get("selected_date"),
        prediction_tiles_url=_resolve_layer_url(
            "prediction_tiles",
            public_base_url=args.public_base_url,
        ),
        ground_truth_tiles_url=(
            None
            if ground_truth_tiles_dir is None
            else _resolve_layer_url(
                "ground_truth_tiles",
                public_base_url=args.public_base_url,
            )
        ),
        argo_points_url=(
            None
            if copied_points_path is None
            else _resolve_layer_url(
                copied_points_path.name,
                public_base_url=args.public_base_url,
            )
        ),
        bounds=prediction_meta,
        prediction_credit=prediction_meta["credit"],
        ground_truth_credit=None if ground_truth_meta is None else ground_truth_meta["credit"],
        points_credit=None if copied_points_path is None else "Observed Argo points",
        template=template,
    )
    config_path = globe_dir / "globe-config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Wrote globe assets to: {globe_dir}")
    print(f"- prediction tiles: {prediction_tiles_dir}")
    if ground_truth_tiles_dir is not None:
        print(f"- ground-truth tiles: {ground_truth_tiles_dir}")
    if copied_points_path is not None:
        print(f"- points GeoJSON: {copied_points_path}")
    print(f"- config: {config_path}")
    print(f"- shared byte scale range: [{scale_min:.4f}, {scale_max:.4f}]")
    if args.rclone_remote is not None:
        ok, message = _sync_with_rclone(globe_dir, args.rclone_remote)
        if ok:
            print(f"- rclone upload: {args.rclone_remote}")
            print(f"  {message}")
        else:
            print(f"WARNING: {message}")
            print(
                "WARNING: Globe assets were still created locally and can be uploaded later with:"
            )
            print(f"  rclone sync {globe_dir} {args.rclone_remote}")


if __name__ == "__main__":
    main()
