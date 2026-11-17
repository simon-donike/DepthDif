# Example: /work/envs/depth/bin/python -m depth_recon.data.fit_target_bias --root /work/data/OceanVariableReconstruction --output /work/data/target_bias.npz --val-year 2016 --min-profiles 30 --chunk-size 50000
"""Fit a frozen, depth-wise EN4-minus-GLORYS target-bias artifact.

The compact profile export can contain the same source profile in several
training patches. This fitter counts a source profile once at each depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import yaml
import zarr

from depth_recon.data.dataset_argo_geotiff_gridded import (
    _decode_stretched_uint8,
    _kelvin_to_celsius,
)
from depth_recon.data.target_bias import file_sha256

REQUIRED_PROFILE_ARRAYS = (
    "profile_date",
    "target_date",
    "grid_row",
    "grid_col",
    "profile_source_file",
    "source_profile_idx",
    "argo_temp_kelvin_uint8",
    "argo_temp_valid",
)


def _resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a manifest path relative to its export root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _slice_starts(size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    """Yield bounded profile slices covering ``size`` rows."""
    for start in range(0, int(size), int(chunk_size)):
        yield start, min(start + int(chunk_size), int(size))


def _accepted_qc_mask(values: np.ndarray) -> np.ndarray:
    """Accept missing compact QC and the repository's accepted flags 1 and 2."""
    qc = np.asarray(values, dtype=np.int16)
    return (qc < 0) | np.isin(qc, np.asarray((1, 2), dtype=np.int16))


def _temperature_quality_mask(
    group: zarr.Group, start: int, stop: int, depth_size: int
) -> np.ndarray:
    """Return the temperature quality mask using the dataset loader's policy."""
    # Compact per-profile QC is preferred, matching ArgoGeoTIFFProfileStore.
    if "argo_temp_profile_qc" in group:
        profile_qc = np.asarray(group["argo_temp_profile_qc"][start:stop])
        return _accepted_qc_mask(profile_qc)[:, None]
    valid = np.ones((stop - start, depth_size), dtype=bool)
    for name in ("argo_depth_qc_on_glorys_depth", "argo_temp_qc_on_glorys_depth"):
        if name in group:
            valid &= _accepted_qc_mask(np.asarray(group[name][start:stop, :]))
    for name in ("argo_juld_qc", "argo_position_qc", "argo_profile_depth_qc"):
        if name in group:
            valid &= _accepted_qc_mask(np.asarray(group[name][start:stop]))[:, None]
    return valid


def _require_compact_schema(group: zarr.Group, depth: np.ndarray) -> dict[str, Any]:
    """Validate the compact POTM_CORRECTED profile schema before fitting."""
    missing = [name for name in REQUIRED_PROFILE_ARRAYS if name not in group]
    if missing:
        raise ValueError(f"Compact profile zarr is missing {missing}.")
    temperature = group["argo_temp_kelvin_uint8"]
    if depth.size > 64:
        raise ValueError("Target-bias deduplication supports at most 64 model depths.")
    if temperature.ndim != 2 or temperature.shape[1] != depth.size:
        raise ValueError(
            "Raw potential-temperature profile shape does not match manifest depth."
        )
    if temperature.dtype != np.dtype("uint8"):
        raise ValueError(
            "Compact potential-temperature profiles must be uint8 encoded."
        )
    attrs = dict(temperature.attrs)
    if attrs.get("source_variable") != "POTM_CORRECTED":
        raise ValueError("Compact temperature must declare POTM_CORRECTED provenance.")
    if attrs.get("standard_name") != "sea_water_potential_temperature":
        raise ValueError(
            "Compact temperature must declare potential-temperature units."
        )
    if attrs.get("units") != "K":
        raise ValueError(
            "Compact potential-temperature profiles must be stored in Kelvin."
        )
    if "minimum" not in attrs or "maximum" not in attrs:
        raise ValueError(
            "Compact potential-temperature profiles lack stretch metadata."
        )
    if group["argo_temp_valid"].shape != temperature.shape:
        raise ValueError("argo_temp_valid must have the temperature profile shape.")
    if group["profile_date"].shape[0] != temperature.shape[0]:
        raise ValueError(
            "Compact profile metadata arrays must share the profile dimension."
        )
    if "glorys_depth" not in group:
        raise ValueError("Compact profile zarr is missing glorys_depth.")
    compact_depth = np.asarray(group["glorys_depth"][:], dtype=np.float32).reshape(-1)
    if not np.array_equal(compact_depth, depth):
        raise ValueError(
            "Compact glorys_depth does not exactly match manifest depth_axis_m."
        )
    return attrs


def _source_keys(
    group: zarr.Group,
    *,
    profile_count: int,
    chunk_size: int,
    target_dates: set[int],
    val_year: int,
) -> tuple[np.ndarray, dict[str, int], int]:
    """Build source-profile keys without a Python set per depth."""
    # One uint64 key plus one uint64 seen-depth mask is enough for the 50-depth
    # compact export, avoiding roughly 470 million Python set entries.
    keys = np.empty(profile_count, dtype=np.uint64)
    source_ids: dict[str, int] = {}
    used = 0
    for start, stop in _slice_starts(profile_count, chunk_size):
        profile_date = np.asarray(group["profile_date"][start:stop], dtype=np.int64)
        target_date = np.asarray(group["target_date"][start:stop], dtype=np.int64)
        included = np.isin(target_date, np.fromiter(target_dates, dtype=np.int64))
        included &= (profile_date // 10000) != int(val_year)
        included &= (target_date // 10000) != int(val_year)
        if not included.any():
            continue
        raw_files = np.asarray(group["profile_source_file"][start:stop]).astype(str)
        source_index = np.asarray(
            group["source_profile_idx"][start:stop], dtype=np.int64
        )
        if np.any(source_index[included] < 0) or np.any(
            source_index[included] >= 2**32
        ):
            raise ValueError(
                "source_profile_idx must fit an unsigned 32-bit profile key."
            )
        names, inverse = np.unique(raw_files, return_inverse=True)
        file_ids = np.empty(stop - start, dtype=np.uint64)
        for local_id, name in enumerate(names.tolist()):
            if not name:
                raise ValueError("profile_source_file must be non-empty.")
            if name not in source_ids:
                source_ids[name] = len(source_ids)
            file_ids[inverse == local_id] = source_ids[name]
        if len(source_ids) >= 2**32:
            raise ValueError(
                "Too many profile source files to pack a source-profile key."
            )
        local_keys = (file_ids[included] << np.uint64(32)) | source_index[
            included
        ].astype(np.uint64)
        keys[used : used + local_keys.size] = local_keys
        used += int(local_keys.size)
    return np.unique(keys[:used]), source_ids, used


def _update_moments(
    difference: np.ndarray,
    profile_ids: np.ndarray,
    seen_depth: np.ndarray,
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
) -> None:
    """Merge unique profile-depth differences into streaming sample moments."""
    for depth_index in range(difference.shape[1]):
        finite_rows = np.flatnonzero(np.isfinite(difference[:, depth_index]))
        if finite_rows.size == 0:
            continue
        ids = profile_ids[finite_rows]
        _, first = np.unique(ids, return_index=True)
        finite_rows = finite_rows[first]
        ids = profile_ids[finite_rows]
        bit = np.uint64(1) << np.uint64(depth_index)
        fresh = (seen_depth[ids] & bit) == 0
        if not fresh.any():
            continue
        finite_rows = finite_rows[fresh]
        ids = ids[fresh]
        values = difference[finite_rows, depth_index].astype(np.float64, copy=False)
        np.bitwise_or.at(seen_depth, ids, bit)
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(np.square(values - batch_mean).sum())
        previous = int(count[depth_index])
        total = previous + batch_count
        delta = batch_mean - mean[depth_index]
        mean[depth_index] += delta * batch_count / total
        m2[depth_index] += batch_m2 + delta * delta * previous * batch_count / total
        count[depth_index] = total


def _hash_zarr_inputs(zarr_path: Path, names: Iterable[str]) -> str:
    """Return one stable digest over compact metadata and all read chunks."""
    digest = hashlib.sha256()
    for filename in (".zgroup", ".zattrs", ".zmetadata"):
        path = zarr_path / filename
        if path.is_file():
            digest.update(filename.encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
    for name in sorted(set(names)):
        array_dir = zarr_path / name
        if not array_dir.is_dir():
            continue
        for path in sorted(item for item in array_dir.rglob("*") if item.is_file()):
            digest.update(path.relative_to(zarr_path).as_posix().encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def fit_target_bias(
    root_dir: str | Path,
    output_path: str | Path,
    *,
    val_year: int = 2016,
    min_profiles: int = 30,
    chunk_size: int = 50000,
) -> Path:
    """Fit one additive EN4-minus-GLORYS offset per model depth."""
    if int(min_profiles) < 1:
        raise ValueError("min_profiles must be at least one.")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be at least one.")
    root = Path(root_dir)
    with (root / "manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest: dict[str, Any] = yaml.safe_load(handle)
    depth = np.asarray(manifest["depth_axis_m"], dtype=np.float32).reshape(-1)
    if depth.size == 0 or not np.isfinite(depth).all() or np.any(np.diff(depth) <= 0):
        raise ValueError(
            "Manifest depth_axis_m must be finite and strictly increasing."
        )
    argo_path = manifest.get("argo", {}).get("path")
    if not argo_path:
        raise ValueError("Manifest does not identify a compact ARGO profile zarr.")
    zarr_path = _resolve_path(root, argo_path)
    if not zarr_path.is_dir():
        raise FileNotFoundError(
            f"Compact ARGO profile zarr does not exist: {zarr_path}"
        )
    thetao_entries = manifest.get("rasters", {}).get("glorys", {}).get("thetao", [])
    raster_by_date = {
        int(item["date"]): _resolve_path(root, item["path"]) for item in thetao_entries
    }
    if not raster_by_date:
        raise ValueError("Manifest does not identify GLORYS thetao rasters.")
    for path in raster_by_date.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Manifest GLORYS thetao raster does not exist: {path}"
            )

    group = zarr.open_group(zarr_path, mode="r")
    stretch = _require_compact_schema(group, depth)
    profile_count = int(group["target_date"].shape[0])
    unique_keys, source_ids, included_rows = _source_keys(
        group,
        profile_count=profile_count,
        chunk_size=int(chunk_size),
        target_dates=set(raster_by_date),
        val_year=int(val_year),
    )
    seen_depth = np.zeros(unique_keys.size, dtype=np.uint64)
    count = np.zeros(depth.size, dtype=np.int64)
    mean = np.zeros(depth.size, dtype=np.float64)
    m2 = np.zeros(depth.size, dtype=np.float64)
    used_target_dates: set[int] = set()
    manifest_stretch = manifest["stretch"]["temperature_kelvin"]

    for start, stop in _slice_starts(profile_count, int(chunk_size)):
        profile_date = np.asarray(group["profile_date"][start:stop], dtype=np.int64)
        target_date = np.asarray(group["target_date"][start:stop], dtype=np.int64)
        grid_row = np.asarray(group["grid_row"][start:stop], dtype=np.int64)
        grid_col = np.asarray(group["grid_col"][start:stop], dtype=np.int64)
        included = np.isin(target_date, np.fromiter(raster_by_date, dtype=np.int64))
        included &= (profile_date // 10000) != int(val_year)
        included &= (target_date // 10000) != int(val_year)
        if not included.any():
            continue
        raw_files = np.asarray(group["profile_source_file"][start:stop]).astype(str)
        source_index = np.asarray(
            group["source_profile_idx"][start:stop], dtype=np.uint64
        )
        file_ids = np.asarray([source_ids[name] for name in raw_files], dtype=np.uint64)
        profile_keys = (file_ids << np.uint64(32)) | source_index
        profile_ids = np.searchsorted(unique_keys, profile_keys)
        if np.any(unique_keys[profile_ids[included]] != profile_keys[included]):
            raise RuntimeError(
                "Source profile key was not found in the deduplication index."
            )
        encoded = np.asarray(
            group["argo_temp_kelvin_uint8"][start:stop, :], dtype=np.uint8
        )
        en4 = _kelvin_to_celsius(_decode_stretched_uint8(encoded, stretch))
        valid = np.asarray(group["argo_temp_valid"][start:stop, :], dtype=bool)
        valid &= _temperature_quality_mask(group, start, stop, depth.size)
        en4[~valid] = np.nan

        for date in np.unique(target_date[included]):
            date = int(date)
            rows = np.flatnonzero(included & (target_date == date))
            raster_path = raster_by_date[date]
            with rasterio.open(raster_path) as raster:
                if raster.count != depth.size:
                    raise ValueError(
                        f"GLORYS thetao raster {raster_path} has {raster.count} bands; "
                        f"expected {depth.size}."
                    )
                in_bounds = (
                    (grid_row[rows] >= 0)
                    & (grid_row[rows] < raster.height)
                    & (grid_col[rows] >= 0)
                    & (grid_col[rows] < raster.width)
                )
                rows = rows[in_bounds]
                if rows.size == 0:
                    continue
                block_height, block_width = raster.block_shapes[0]
                blocks_per_row = (raster.width + block_width - 1) // block_width
                block_ids = (grid_row[rows] // block_height) * blocks_per_row + (
                    grid_col[rows] // block_width
                )
                order = np.argsort(block_ids, kind="stable")
                ordered_rows, ordered_blocks = rows[order], block_ids[order]
                block_starts = np.r_[
                    0, np.flatnonzero(np.diff(ordered_blocks)) + 1, ordered_rows.size
                ]
                for block_start, block_stop in zip(
                    block_starts[:-1], block_starts[1:], strict=False
                ):
                    block_rows = ordered_rows[block_start:block_stop]
                    row0 = int((grid_row[block_rows[0]] // block_height) * block_height)
                    col0 = int((grid_col[block_rows[0]] // block_width) * block_width)
                    window = rasterio.windows.Window(
                        col0,
                        row0,
                        min(block_width, raster.width - col0),
                        min(block_height, raster.height - row0),
                    )
                    local_rows, local_cols = (
                        grid_row[block_rows] - row0,
                        grid_col[block_rows] - col0,
                    )
                    # A native TIFF block can be much larger than its matched
                    # profiles. Gather encoded points before decoding to avoid
                    # allocating and converting the entire float32 block.
                    encoded_glorys = raster.read(window=window)[
                        :, local_rows, local_cols
                    ].T
                    glorys = _kelvin_to_celsius(
                        _decode_stretched_uint8(encoded_glorys, manifest_stretch)
                    )
                    difference = en4[block_rows] - glorys
                    _update_moments(
                        difference, profile_ids[block_rows], seen_depth, count, mean, m2
                    )
            used_target_dates.add(date)
            print(
                f"processed target date {date}: {rows.size} profiles, {int(count.sum())} unique valid band values",
                flush=True,
            )

    raster_hashes = {
        str(raster_by_date[date].relative_to(root)): file_sha256(raster_by_date[date])
        for date in sorted(used_target_dates)
    }
    used_arrays = list(REQUIRED_PROFILE_ARRAYS) + [
        "glorys_depth",
        "argo_temp_profile_qc",
    ]
    if "argo_temp_profile_qc" not in group:
        used_arrays.extend(
            name
            for name in (
                "argo_depth_qc_on_glorys_depth",
                "argo_temp_qc_on_glorys_depth",
                "argo_juld_qc",
                "argo_position_qc",
                "argo_profile_depth_qc",
            )
            if name in group
        )
    supported = count >= int(min_profiles)
    bias = np.where(supported, mean, 0.0).astype(np.float32)
    variance = np.where(count > 1, m2 / np.maximum(count - 1, 1), 0.0).astype(
        np.float32
    )
    metadata = {
        "kind": "en4_minus_glorys_target_bias",
        "units": "degree_C",
        "sign_convention": "EN4_minus_GLORYS; target=GLORYS+bias",
        "min_profiles": int(min_profiles),
        "excluded_years": [int(val_year)],
        "grid": manifest["grid"],
        "depth_axis_m": depth.tolist(),
        "profile_count_total": profile_count,
        "profile_row_count_eligible": int(included_rows),
        "profile_count_used": int(np.count_nonzero(seen_depth)),
        "source_profile_variable": "argo_temp_kelvin_uint8",
        "source_temperature": "EN4 potential temperature raw compact profile",
        "source_temperature_variable": "POTM_CORRECTED",
        "source_temperature_standard_name": "sea_water_potential_temperature",
        "qc_policy": "known compact/level QC flags 1 or 2; missing QC accepted",
        "deduplication": "profile_source_file + source_profile_idx, once per depth",
        "input_hashes": {
            "manifest.yaml": file_sha256(root / "manifest.yaml"),
            "compact_zarr_metadata": (
                file_sha256(zarr_path / ".zmetadata")
                if (zarr_path / ".zmetadata").is_file()
                else None
            ),
            "compact_zarr_used_chunks": _hash_zarr_inputs(zarr_path, used_arrays),
            "glorys_thetao": raster_hashes,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        bias=bias,
        count=count,
        variance=variance,
        supported=supported,
        depth_axis_m=depth,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return output


def main() -> None:
    """Parse CLI arguments and fit an artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-year", required=True, type=int)
    parser.add_argument("--min-profiles", default=30, type=int)
    parser.add_argument("--chunk-size", default=50000, type=int)
    args = parser.parse_args()
    fit_target_bias(
        args.root,
        args.output,
        val_year=args.val_year,
        min_profiles=args.min_profiles,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
