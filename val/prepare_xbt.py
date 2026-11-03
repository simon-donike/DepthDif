# /work/envs/depth/bin/python -m val.prepare_xbt val/results/xbt_import/profiles.csv --data-root /work/data/OceanVariableReconstruction --output val/results/xbt_reference/profiles.csv --manifest val/results/xbt_reference/manifest.json --year 2016 --max-time-delta-days 3.5 --max-vertical-gap-m 100
"""Attach GLORYS practical salinity needed for XBT TEOS-10 conversion."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import _decode_stretched_uint8
from val.observations import CANONICAL_COLUMNS, _model_depth_assignment


def _sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for one provenance file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_manifest(data_root: Path) -> tuple[Path, dict[str, Any], dict[int, Path]]:
    """Load the dataset manifest and resolve GLORYS salinity rasters by date."""
    manifest_path = data_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    try:
        stretch = dict(manifest["stretch"]["salinity"])
        entries = manifest["rasters"]["glorys"]["so"]
        depths = np.asarray(manifest["depth_axis_m"], dtype=float)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "dataset manifest has no GLORYS salinity definition"
        ) from error
    if (
        depths.ndim != 1
        or depths.size == 0
        or not np.isfinite(depths).all()
        or np.any(np.diff(depths) <= 0)
    ):
        raise ValueError("manifest depth_axis_m must be finite and increasing")
    paths: dict[int, Path] = {}
    for entry in entries:
        path = Path(entry["path"])
        paths[int(entry["date"])] = path if path.is_absolute() else data_root / path
    if not paths:
        raise ValueError("dataset manifest contains no GLORYS salinity rasters")
    return manifest_path, {"stretch": stretch, "depths": depths}, paths


def _validate_profiles(profiles: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """Validate canonical imported XBT rows and apply the requested year."""
    missing = sorted(set(CANONICAL_COLUMNS) - set(profiles.columns))
    if missing:
        raise ValueError("XBT profiles missing columns: " + ", ".join(missing))
    out = profiles.copy()
    out["observation_time"] = pd.to_datetime(
        out["observation_time"], utc=True, errors="coerce"
    )
    for column in ("longitude", "latitude", "depth_m"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["observation_time", "longitude", "latitude", "depth_m"]].isna().any().any():
        raise ValueError("XBT profiles contain invalid time, position, or depth")
    if (~out["variable"].astype(str).str.lower().eq("temperature")).any():
        raise ValueError("XBT reference preparation accepts temperature rows only")
    if (
        "temperature_kind" not in out
        or (~out["temperature_kind"].astype(str).str.lower().eq("in_situ")).any()
    ):
        raise ValueError("XBT rows must declare temperature_kind=in_situ")
    if year is not None:
        out = out.loc[out["observation_time"].dt.year.eq(year)].copy()
    if out.empty:
        raise ValueError(f"XBT profiles contain no rows for year {year}")
    if out.duplicated(["profile_id", "depth_m"]).any():
        raise ValueError("XBT profiles contain duplicate profile/depth rows")
    for profile_id, group in out.groupby("profile_id", sort=False):
        for column in ("observation_time", "longitude", "latitude"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"profile_id {profile_id!r} has inconsistent {column} metadata"
                )
    return out.reset_index(drop=True)


def _nearest_date(
    timestamp: pd.Timestamp,
    dates: np.ndarray,
    timestamps: dict[int, pd.Timestamp],
) -> tuple[int, float]:
    """Return the nearest product date and signed observation-minus-target offset."""
    target = min(
        dates.tolist(), key=lambda value: abs(timestamps[int(value)] - timestamp)
    )
    delta = (timestamp - timestamps[int(target)]).total_seconds() / 86400.0
    return int(target), float(delta)


def _sample_profile(
    source: rasterio.io.DatasetReader,
    path: Path,
    longitude: float,
    latitude: float,
    stretch: dict[str, Any],
    expected_depths: int,
) -> tuple[np.ndarray, int, int, float, float]:
    """Decode one complete salinity profile at the containing raster cell."""
    if source.count != expected_depths:
        raise ValueError(
            f"salinity raster band count {source.count} != depth count {expected_depths}: {path}"
        )
    row, col = source.index(longitude, latitude)
    if not (0 <= row < source.height and 0 <= col < source.width):
        raise ValueError("profile coordinate is outside the GLORYS raster")
    encoded = source.read(window=((row, row + 1), (col, col + 1)))[:, 0, 0]
    actual_longitude, actual_latitude = xy(source.transform, row, col, offset="center")
    decoded = _decode_stretched_uint8(encoded, stretch).astype(float)
    return decoded, int(row), int(col), float(actual_longitude), float(actual_latitude)


def prepare_xbt_reference(
    profiles_csv: str | Path,
    data_root: str | Path,
    output_csv: str | Path,
    manifest_json: str | Path,
    *,
    year: int | None = 2016,
    max_time_delta_days: float = 3.5,
    max_vertical_gap_m: float = 100.0,
) -> dict[str, Any]:
    """Write XBT rows with GLORYS salinity interpolated to raw depths."""
    profiles_path = Path(profiles_csv).resolve()
    root = Path(data_root).resolve()
    output = Path(output_csv)
    manifest_output = Path(manifest_json)
    exclusions_output = output.with_suffix(".exclusions.csv")
    if not profiles_path.is_file():
        raise FileNotFoundError(profiles_path)
    if any(path.exists() for path in (output, manifest_output, exclusions_output)):
        raise FileExistsError("refusing to overwrite prepared XBT outputs")
    if max_time_delta_days < 0 or max_vertical_gap_m <= 0:
        raise ValueError("time tolerance must be nonnegative and vertical gap positive")
    dataset_manifest, definition, raster_paths = _load_manifest(root)
    profiles = _validate_profiles(pd.read_csv(profiles_path), year)
    depths = definition["depths"]
    dates = np.asarray(sorted(raster_paths), dtype=np.int64)
    date_times = {
        int(value): pd.Timestamp(str(int(value)), tz="UTC") for value in dates
    }
    retained: list[pd.DataFrame] = []
    exclusions: list[dict[str, Any]] = []
    used_rasters: dict[Path, int] = {}
    with ExitStack() as stack:
        open_rasters: dict[Path, rasterio.io.DatasetReader] = {}
        for profile_id, group in profiles.groupby("profile_id", sort=False):
            timestamp = group["observation_time"].iloc[0]
            target_date, time_delta = _nearest_date(timestamp, dates, date_times)
            if abs(time_delta) > max_time_delta_days:
                exclusions.extend(
                    {
                        "profile_id": profile_id,
                        "depth_m": float(depth),
                        "reason": "no_salinity_date_within_tolerance",
                    }
                    for depth in group["depth_m"]
                )
                continue
            raster_path = raster_paths[target_date]
            if not raster_path.is_file():
                raise FileNotFoundError(
                    f"GLORYS salinity raster does not exist: {raster_path}"
                )
            if raster_path not in open_rasters:
                open_rasters[raster_path] = stack.enter_context(
                    rasterio.open(raster_path)
                )
            profile, grid_row, grid_col, actual_lon, actual_lat = _sample_profile(
                open_rasters[raster_path],
                raster_path,
                float(group["longitude"].iloc[0]),
                float(group["latitude"].iloc[0]),
                definition["stretch"],
                len(depths),
            )
            used_rasters[raster_path] = target_date
            kept_rows: list[dict[str, Any]] = []
            for row in group.to_dict(orient="records"):
                assignment = _model_depth_assignment(
                    float(row["depth_m"]), depths, max_vertical_gap_m
                )
                if assignment is None:
                    exclusions.append(
                        {
                            "profile_id": profile_id,
                            "depth_m": float(row["depth_m"]),
                            "reason": "outside_supported_salinity_depths_or_gap",
                        }
                    )
                    continue
                lower = int(assignment["depth_index_lower"])
                upper = int(assignment["depth_index_upper"])
                weight = float(assignment["depth_weight"])
                if not np.isfinite(profile[[lower, upper]]).all():
                    exclusions.append(
                        {
                            "profile_id": profile_id,
                            "depth_m": float(row["depth_m"]),
                            "reason": "salinity_bracket_is_not_wet",
                        }
                    )
                    continue
                row["reference_salinity"] = float(
                    (1.0 - weight) * profile[lower] + weight * profile[upper]
                )
                row["reference_salinity_kind"] = "practical"
                row["reference_salinity_target_date"] = target_date
                row["reference_salinity_time_delta_days"] = time_delta
                row["reference_salinity_grid_row"] = grid_row
                row["reference_salinity_grid_col"] = grid_col
                row["reference_salinity_longitude"] = actual_lon
                row["reference_salinity_latitude"] = actual_lat
                kept_rows.append(row)
            if kept_rows:
                retained.append(pd.DataFrame.from_records(kept_rows))
    if not retained:
        raise ValueError("no XBT rows have valid GLORYS salinity support")
    result = pd.concat(retained, ignore_index=True)
    if not np.isfinite(result["reference_salinity"].to_numpy(float)).all():
        raise ValueError("prepared XBT rows contain non-finite reference salinity")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame.from_records(
        exclusions, columns=("profile_id", "depth_m", "reason")
    ).to_csv(exclusions_output, index=False)
    output_sha256 = _sha256(output)
    manifest = {
        "input_profiles": str(profiles_path),
        "input_profiles_sha256": _sha256(profiles_path),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "year": year,
        "max_time_delta_days": max_time_delta_days,
        "max_vertical_gap_m": max_vertical_gap_m,
        "rows_input": int(len(profiles)),
        "rows_retained": int(len(result)),
        "rows_excluded": int(len(exclusions)),
        "profiles_retained": int(result["profile_id"].nunique()),
        "output_profiles": str(output.resolve()),
        "output_profiles_sha256": output_sha256,
        "exclusions": str(exclusions_output.resolve()),
        "exclusions_sha256": _sha256(exclusions_output),
        "salinity_rasters": [
            {
                "date": used_rasters[path],
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in sorted(used_rasters)
        ],
        "reference_salinity_kind": "practical",
        "reference_salinity_source": "GLORYS so at the same product date and containing grid cell",
        "scientific_gate": (
            "GLORYS salinity is an observation-operator assumption shared by every "
            "temperature comparison; report a salinity sensitivity analysis."
        ),
        "config_snippet": {
            "temperature_observation_operator": "in_situ_to_potential_teos10",
            "reference_salinity_kind": "practical",
            "reference_salinity_provenance": {
                "path": str(output.resolve()),
                "sha256": output_sha256,
            },
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Prepare the imported XBT table from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles_csv", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--max-time-delta-days", type=float, default=3.5)
    parser.add_argument("--max-vertical-gap-m", type=float, default=100.0)
    args = parser.parse_args()
    prepare_xbt_reference(
        args.profiles_csv,
        args.data_root,
        args.output,
        args.manifest,
        year=args.year,
        max_time_delta_days=args.max_time_delta_days,
        max_vertical_gap_m=args.max_vertical_gap_m,
    )


if __name__ == "__main__":
    main()
