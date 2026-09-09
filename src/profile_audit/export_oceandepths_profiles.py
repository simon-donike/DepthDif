from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from profile_audit.common import valid_coordinates

FULL_STORE = Path("data/argo_glors_ostia_ssh.zarr")


def resolve_oceandepths_root(
    local_root: Path | None,
    *,
    repository: str = "ESA-philab/OceanDepths",
    revision: str | None = None,
) -> Path:
    if local_root is not None and Path(local_root).exists():
        return Path(local_root)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError("Local OceanDepths root is missing; install huggingface_hub to sync it.") from exc
    return Path(snapshot_download(repo_id=repository, repo_type="dataset", revision=revision))


def _datetime_from_juld(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(values, unit="D", origin=pd.Timestamp("1950-01-01"), utc=True)


def iter_oceandepths_chunks(
    store: Path,
    *,
    chunk_size: int = 50_000,
    max_profiles: int | None = None,
    include_values: bool = True,
) -> Iterator[pd.DataFrame]:
    ds = xr.open_zarr(store, consolidated=None)
    try:
        total = ds.sizes["profile"] if max_profiles is None else min(ds.sizes["profile"], max_profiles)
        depths = np.asarray(ds["glorys_depth"].values, dtype=np.float32)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            subset = ds.isel(profile=slice(start, stop))
            source_file = np.asarray(subset["profile_source_file"].values).astype(str)
            source_idx = np.asarray(subset["profile_idx"].values, dtype=np.int64)
            lat = np.asarray(subset["latitude"].values, dtype=np.float64)
            lon = np.asarray(subset["longitude"].values, dtype=np.float64)
            valid_geo = valid_coordinates(lat, lon)
            records = {
                "profile": np.asarray(subset["profile"].values, dtype=np.int64),
                "profile_id": np.char.add(np.char.add(source_file, ":"), source_idx.astype(str)),
                "profile_date": np.asarray(subset["profile_date"].values, dtype=np.int64),
                "datetime_utc": _datetime_from_juld(np.asarray(subset["profile_juld"].values, dtype=np.float64)),
                "profile_source_file": source_file,
                "source_profile_idx": source_idx,
                "latitude": lat,
                "longitude": lon,
                "valid_coordinate": valid_geo,
                "source_provider": "UK Met Office Hadley Centre",
                "source_product": "EN4.2.2 profile archive",
                "observation_family": "EN4",
            }
            for source, target in (
                ("argo_position_qc", "position_qc"),
                ("argo_profile_potm_qc", "temperature_profile_qc"),
                ("argo_profile_psal_qc", "salinity_profile_qc"),
            ):
                records[target] = np.asarray(subset[source].values, dtype=np.int8) if source in subset else np.full(stop - start, -1, dtype=np.int8)
            if include_values:
                records["depth_m"] = [depths] * (stop - start)
                for source, target in (
                    ("argo_temp_on_glorys_depth", "temperature_c"),
                    ("argo_potm_on_glorys_depth", "potential_temperature_c"),
                    ("argo_psal_on_glorys_depth", "salinity"),
                    ("argo_temp_qc_on_glorys_depth", "temperature_qc"),
                    ("argo_psal_qc_on_glorys_depth", "salinity_qc"),
                    ("glorys_thetao", "glorys_temperature_c"),
                ):
                    if source in subset:
                        records[target] = list(np.asarray(subset[source].values))
            yield pd.DataFrame(records)
    finally:
        ds.close()


def export_oceandepths_profiles(
    root: Path,
    output: Path,
    *,
    chunk_size: int = 50_000,
    max_profiles: int | None = None,
    include_values: bool = True,
) -> Path:
    root = Path(root)
    store = root if root.suffix == ".zarr" else root / FULL_STORE
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in iter_oceandepths_chunks(store, chunk_size=chunk_size, max_profiles=max_profiles, include_values=include_values):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("OceanDepths store contains no selected profiles.")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export EN4 profiles from the OceanDepths Zarr store.")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repository", default="ESA-philab/OceanDepths")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    root = resolve_oceandepths_root(args.root, repository=args.repository, revision=args.revision)
    print(export_oceandepths_profiles(root, args.output, chunk_size=args.chunk_size, max_profiles=args.max_profiles, include_values=not args.metadata_only))


if __name__ == "__main__":
    main()
