# Example: /work/envs/depth/bin/python -m val.cugn --input evaluations/inputs/cugn/CUGN_line_66.nc --output val/results/cugn --year 2016 --max-depth-m 2000
"""Import processed CUGN section NetCDF files as explicitly external data.

Usage: ``/work/envs/depth/bin/python -m val.cugn --input evaluations/inputs/cugn/CUGN_line_66.nc --output val/results/cugn --year 2016 --max-depth-m 2000``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _name(names: set[str], candidates: tuple[str, ...]) -> str | None:
    """Choose a coordinate or variable name from common NetCDF conventions."""
    return next((candidate for candidate in candidates if candidate in names), None)


def inspect_file(path: Path) -> dict[str, Any]:
    """Return a schema summary without assuming CUGN files are raw casts."""
    import xarray as xr

    with xr.open_dataset(path, decode_times=True) as dataset:
        return {
            "path": str(path),
            "variables": sorted(str(name) for name in dataset.data_vars),
            "coordinates": sorted(str(name) for name in dataset.coords),
            "dims": {str(key): int(value) for key, value in dataset.sizes.items()},
            "processed_section": True,
            "source_type": "processed_gridded_section",
            "processing_level": dataset.attrs.get("processing_level"),
            "raw_casts": False,
            "deployment_metadata_present": any(
                name in dataset
                for name in ("platform", "platform_number", "deployment_id")
            ),
        }


def convert_file(
    path: Path,
    output_dir: Path,
    max_depth_m: float | None = None,
    year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert one processed section into canonical observation rows."""
    import xarray as xr

    with xr.open_dataset(path, decode_times=True) as dataset:
        names = set(dataset.variables)
        depth = _name(names, ("depth", "depth_m", "DEPTH", "z"))
        time = _name(names, ("time", "TIME", "datetime"))
        lat = _name(names, ("latitude", "lat", "LATITUDE"))
        lon = _name(names, ("longitude", "lon", "LONGITUDE"))
        temperature = _name(names, ("temperature", "temp", "TEMP", "thetao"))
        salinity = _name(names, ("salinity", "salt", "PSAL", "so"))
        if not all((depth, time, lat, lon)) or not (temperature or salinity):
            raise ValueError(
                f"cannot identify CUGN section coordinates/variables in {path}"
            )
        dimensions = dataset[temperature or salinity].dims
        depth_axis = dimensions.index(depth) if depth in dimensions else None
        if depth_axis is None or len(dimensions) != 2:
            raise ValueError(f"CUGN variable does not use depth coordinate: {path}")
        profile_dim = dimensions[1 - depth_axis]
        depths = np.asarray(dataset[depth].values, dtype=float).reshape(-1)
        profile_count = int(dataset.sizes[profile_dim])
        times = pd.to_datetime(dataset[time].values, utc=True, errors="coerce")
        latitudes = np.asarray(dataset[lat].values, dtype=float).reshape(-1)
        longitudes = np.asarray(dataset[lon].values, dtype=float).reshape(-1)
        values_by_variable = {}
        for variable, variable_name in (
            ("temperature", temperature),
            ("salinity", salinity),
        ):
            if variable_name:
                values_by_variable[variable] = np.asarray(
                    dataset[variable_name].transpose(depth, profile_dim).values,
                    dtype=float,
                )
        trajectory_values = (
            np.asarray(dataset["trajectory_index"].values).reshape(-1)
            if "trajectory_index" in dataset
            else np.arange(profile_count)
        )
        mission_values = (
            np.asarray(dataset["mission"].values).reshape(-1)
            if "mission" in dataset
            else np.arange(profile_count)
        )
        mission_profile_values = (
            np.asarray(dataset["mission_profile"].values).reshape(-1)
            if "mission_profile" in dataset
            else np.arange(profile_count)
        )
        trajectory_names = (
            np.asarray(dataset["mission_name"].values).reshape(-1)
            if "mission_name" in dataset
            else np.array([], dtype=object)
        )
        rows: list[dict[str, Any]] = []
        for profile_index in range(profile_count):
            timestamp = times[profile_index]
            latitude, longitude = float(latitudes[profile_index]), float(
                longitudes[profile_index]
            )
            if pd.isna(timestamp) or not np.isfinite(latitude + longitude):
                continue
            mission = mission_values[profile_index]
            mission_profile = mission_profile_values[profile_index]
            trajectory = trajectory_values[profile_index]
            trajectory_name = (
                trajectory_names[int(trajectory)]
                if trajectory_names.size and int(trajectory) < trajectory_names.size
                else trajectory
            )
            if isinstance(trajectory_name, (bytes, np.bytes_)):
                trajectory_name = trajectory_name.decode(errors="replace")
            section_id = (
                f"{path.stem}:trajectory{trajectory_name}:profile{mission_profile}"
            )
            if year is not None and int(timestamp.year) != int(year):
                continue
            for variable, variable_name in values_by_variable.items():
                values = values_by_variable[variable][:, profile_index]
                paired_salinity = (
                    values_by_variable.get("salinity", np.full_like(values, np.nan))[
                        :, profile_index
                    ]
                    if "salinity" in values_by_variable
                    else np.full_like(values, np.nan)
                )
                for depth_index, (depth_value, observation) in enumerate(
                    zip(depths, values)
                ):
                    if max_depth_m is not None and float(depth_value) > max_depth_m:
                        continue
                    if (
                        not np.isfinite(observation)
                        or not np.isfinite(depth_value)
                        or depth_value < 0
                    ):
                        continue
                    rows.append(
                        {
                            "profile_id": section_id,
                            "cluster_id": f"section:{path.stem}:trajectory{trajectory_name}",
                            "source": "cugn_processed_section",
                            "observation_time": timestamp,
                            "longitude": longitude,
                            "latitude": latitude,
                            "depth_m": float(depth_value),
                            "variable": variable,
                            "observation": float(observation),
                            "independence": "unknown",
                            "qc_flag": 1,
                            "temperature_kind": (
                                "in_situ" if variable == "temperature" else None
                            ),
                            "reference_salinity": (
                                float(paired_salinity[depth_index])
                                if variable == "temperature"
                                and np.isfinite(paired_salinity[depth_index])
                                else np.nan
                            ),
                            "reference_salinity_kind": (
                                "practical" if variable == "temperature" else None
                            ),
                            "case_id": path.stem,
                            "deployment_status": "unknown",
                            "processed_section": True,
                        }
                    )
    frame = pd.DataFrame(rows)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "source": "CUGN",
        "processed_section": True,
        "deployment_status": "unknown",
        "independence": "unknown",
        "temperature_kind": "in_situ",
        "temperature_observation_operator": "in_situ_to_potential_teos10",
        "reference_salinity_kind": "practical",
        "reference_salinity_provenance": {"path": str(path), "sha256": digest},
        "config_snippet": {
            "temperature_kind": "in_situ",
            "temperature_observation_operator": "in_situ_to_potential_teos10",
            "reference_salinity_kind": "practical",
            "reference_salinity_provenance": {"path": str(path), "sha256": digest},
        },
        "input": str(path),
        "n_rows": len(frame),
        "year_filter": year,
    }
    return frame, metadata


def import_cugn(
    paths: list[Path],
    output_dir: Path,
    max_depth_m: float | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Import configured CUGN files and write canonical rows plus provenance."""
    output_dir = Path(output_dir)
    if not str(output_dir.resolve()).startswith(str(Path("val").resolve())):
        raise ValueError("CUGN output must be inside val/")
    frames, metadata = [], []
    for path in paths:
        frame, info = convert_file(path, output_dir, max_depth_m, year)
        frames.append(frame)
        metadata.append(info)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "profiles.csv", index=False)
    manifest = {
        "source": "CUGN",
        "source_type": "processed_gridded_section",
        "raw_casts": False,
        "independence": "unknown",
        "files": metadata,
        "config_snippet": {
            "year": year,
            "max_depth_m": max_depth_m,
            "temperature_kind": "in_situ",
            "temperature_observation_operator": "in_situ_to_potential_teos10",
            "reference_salinity_kind": "practical",
            "reference_salinity_provenance": (
                metadata[0].get("reference_salinity_provenance") if metadata else None
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return {
        "status": "complete" if not result.empty else "partial",
        "rows": len(result),
        "profiles": int(result.profile_id.nunique()) if not result.empty else 0,
        "artifacts": [
            str(output_dir / "profiles.csv"),
            str(output_dir / "manifest.json"),
        ],
    }


def main() -> None:
    """Parse the standalone CUGN conversion command."""
    parser = argparse.ArgumentParser(
        description="Convert processed CUGN sections to validation profiles"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input NetCDF; repeat for multiple sections",
    )
    parser.add_argument("--output", required=True, help="Output directory under val/")
    parser.add_argument(
        "--year", type=int, default=None, help="Reserved provenance year filter"
    )
    parser.add_argument("--max-depth-m", type=float, default=None)
    args = parser.parse_args()
    print(
        import_cugn(
            [Path(value) for value in args.input],
            Path(args.output),
            args.max_depth_m,
            args.year,
        )
    )


if __name__ == "__main__":
    main()
