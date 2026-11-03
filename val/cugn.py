"""Import processed CUGN section NetCDF files as explicitly external data.

Usage: ``/work/envs/depth/bin/python -m val.cugn --input evaluations/inputs/cugn/CUGN_line_66.nc --output val/results/cugn --year 2016 --max-depth-m 2000``
"""

from __future__ import annotations

import argparse
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
            "deployment_metadata_present": any(name in dataset for name in ("platform", "platform_number", "deployment_id")),
        }


def convert_file(path: Path, output_dir: Path, max_depth_m: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
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
            raise ValueError(f"cannot identify CUGN section coordinates/variables in {path}")
        dimensions = dataset[temperature or salinity].dims
        depth_axis = dimensions.index(depth) if depth in dimensions else None
        if depth_axis is None:
            raise ValueError(f"CUGN variable does not use depth coordinate: {path}")
        rows: list[dict[str, Any]] = []
        values_by_variable = {"temperature": temperature, "salinity": salinity}
        for position in np.ndindex(*tuple(int(dataset.sizes[dim]) for dim in dimensions if dim != depth)):
            selectors = {dim: index for dim, index in zip((dim for dim in dimensions if dim != depth), position)}
            timestamp = pd.to_datetime(dataset[time].isel(selectors).values, utc=True, errors="coerce")
            latitude, longitude = float(dataset[lat].isel(selectors).values), float(dataset[lon].isel(selectors).values)
            if pd.isna(timestamp) or not np.isfinite(latitude + longitude):
                continue
            section_id = f"{path.stem}:{position[0] if position else 0}"
            for variable, variable_name in values_by_variable.items():
                if not variable_name:
                    continue
                values = np.asarray(dataset[variable_name].isel(selectors).values, dtype=float).reshape(-1)
                depths = np.asarray(dataset[depth].values, dtype=float).reshape(-1)
                for depth_value, observation in zip(depths, values):
                    if max_depth_m is not None and float(depth_value) > max_depth_m:
                        continue
                    if not np.isfinite(observation) or not np.isfinite(depth_value) or depth_value < 0:
                        continue
                    rows.append({"profile_id": section_id, "cluster_id": f"section:{path.stem}", "source": "cugn_processed_section", "observation_time": timestamp, "longitude": longitude, "latitude": latitude, "depth_m": float(depth_value), "variable": variable, "observation": float(observation), "independence": "unknown", "qc_flag": 1, "case_id": path.stem, "deployment_status": "unknown", "processed_section": True})
    frame = pd.DataFrame(rows)
    metadata = {"source": "CUGN", "processed_section": True, "deployment_status": "unknown", "independence": "unknown", "input": str(path), "n_rows": len(frame)}
    return frame, metadata


def import_cugn(paths: list[Path], output_dir: Path, max_depth_m: float | None = None) -> dict[str, Any]:
    """Import configured CUGN files and write canonical rows plus provenance."""
    output_dir = Path(output_dir)
    if not str(output_dir.resolve()).startswith(str(Path("val").resolve())):
        raise ValueError("CUGN output must be inside val/")
    frames, metadata = [], []
    for path in paths:
        frame, info = convert_file(path, output_dir, max_depth_m)
        frames.append(frame)
        metadata.append(info)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "profiles.csv", index=False)
    pd.DataFrame(metadata).to_json(output_dir / "manifest.json", orient="records", indent=2, date_format="iso")
    return {"status": "complete" if not result.empty else "partial", "rows": len(result), "profiles": int(result.profile_id.nunique()) if not result.empty else 0, "artifacts": [str(output_dir / "profiles.csv"), str(output_dir / "manifest.json")]}


def main() -> None:
    """Parse the standalone CUGN conversion command."""
    parser = argparse.ArgumentParser(description="Convert processed CUGN sections to validation profiles")
    parser.add_argument("--input", action="append", required=True, help="Input NetCDF; repeat for multiple sections")
    parser.add_argument("--output", required=True, help="Output directory under val/")
    parser.add_argument("--year", type=int, default=None, help="Reserved provenance year filter")
    parser.add_argument("--max-depth-m", type=float, default=None)
    args = parser.parse_args()
    print(import_cugn([Path(value) for value in args.input], Path(args.output), args.max_depth_m))


if __name__ == "__main__":
    main()
