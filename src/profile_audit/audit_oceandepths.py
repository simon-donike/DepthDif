from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from profile_audit.common import haversine_km, valid_coordinates

EARTH_RADIUS_KM = 6371.0088
BROAD_RADIUS_KM = 25.0
BROAD_CHORD_RADIUS = 2.0 * np.sin(BROAD_RADIUS_KM / (2.0 * EARTH_RADIUS_KM))
OUTPUT_SCHEMA = pa.schema(
    [
        ("target_profile_id", pa.string()),
        ("target_profile", pa.int64()),
        ("profile_source_file", pa.string()),
        ("source_profile_idx", pa.int64()),
        ("datetime_utc", pa.timestamp("ns", tz="UTC")),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("expected_archive", pa.string()),
        ("coverage_status", pa.string()),
        ("historical_coverage_status", pa.string()),
        ("evidence_scope", pa.string()),
        ("audit_status", pa.string()),
        ("broad_candidate_count", pa.int64()),
        ("recovery_candidate_count", pa.int64()),
        ("primary_candidate_count", pa.int64()),
        ("best_candidate_profile_id", pa.string()),
        ("best_time_difference_minutes", pa.float64()),
        ("best_distance_km", pa.float64()),
        ("best_candidate_source_file", pa.string()),
        ("best_candidate_source_profile_index", pa.int64()),
    ]
)


def _unit_sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(latitude, dtype=np.float64))
    lon = np.radians(np.asarray(longitude, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def expected_archive(timestamp: pd.Timestamp) -> tuple[str | None, str]:
    if pd.isna(timestamp):
        return None, "invalid_time"
    if timestamp < pd.Timestamp("1993-01-01", tz="UTC"):
        return None, "outside_glorys_period"
    if timestamp < pd.Timestamp("2014-01-01", tz="UTC"):
        return "CORA4.1", "auditable_public_release"
    if timestamp < pd.Timestamp("2016-01-01", tz="UTC"):
        return "CORA5.0", "auditable_public_release"
    if timestamp < pd.Timestamp("2017-01-01", tz="UTC"):
        return "CORA5.1", "auditable_public_release"
    if timestamp < pd.Timestamp("2021-07-01", tz="UTC"):
        return None, "source_unresolved"
    return None, "internal_snapshot_unavailable"


class CandidateDayCache:
    def __init__(self, archives: dict[str, Path], max_items: int = 8) -> None:
        self.datasets = {name: pads.dataset(path, format="parquet") for name, path in archives.items()}
        self.max_items = max_items
        self.items: OrderedDict[tuple[str, str], pd.DataFrame] = OrderedDict()

    def get(self, archive: str, day: pd.Timestamp) -> pd.DataFrame:
        key = (archive, day.strftime("%Y%m%d"))
        if key in self.items:
            frame = self.items.pop(key)
            self.items[key] = frame
            return frame
        dataset = self.datasets[archive]
        required = {
            "profile_id", "datetime_utc", "latitude", "longitude", "source_file",
            "source_profile_index", "calendar_day",
        }
        missing = required - set(dataset.schema.names)
        if missing:
            raise ValueError(f"{archive} candidate index is missing columns: {sorted(missing)}")
        table = dataset.to_table(
            columns=sorted(required),
            filter=pads.field("calendar_day") == key[1],
        )
        frame = table.to_pandas()
        if not frame.empty:
            frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True, errors="coerce")
            mask = valid_coordinates(frame["latitude"], frame["longitude"])
            frame = frame.loc[mask & frame["datetime_utc"].notna()].reset_index(drop=True)
        self.items[key] = frame
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return frame


def _candidate_frame(cache: CandidateDayCache, archive: str, day: pd.Timestamp) -> pd.DataFrame:
    return pd.concat(
        [cache.get(archive, day + pd.Timedelta(days=offset)) for offset in (-1, 0, 1)],
        ignore_index=True,
    )


def _audit_target_day(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    archive: str,
    revision: str,
    historical_coverage_status: str = "auditable_public_release",
    proxy: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    candidate_tree = cKDTree(_unit_sphere(candidates["latitude"].to_numpy(), candidates["longitude"].to_numpy())) if not candidates.empty else None
    target_points = _unit_sphere(targets["latitude"].to_numpy(), targets["longitude"].to_numpy())
    neighbor_lists = candidate_tree.query_ball_point(target_points, BROAD_CHORD_RADIUS) if candidate_tree is not None else [[] for _ in range(len(targets))]
    candidate_times = candidates["datetime_utc"].astype("int64").to_numpy() if not candidates.empty else np.array([], dtype=np.int64)

    for position, (_, target) in enumerate(targets.iterrows()):
        target_id = f"{revision}:{target['profile_source_file']}:{int(target['profile_idx'])}"
        base = {
            "target_profile_id": target_id,
            "target_profile": int(target["profile"]),
            "profile_source_file": str(target["profile_source_file"]),
            "source_profile_idx": int(target["profile_idx"]),
            "datetime_utc": target["datetime_utc"],
            "latitude": float(target["latitude"]),
            "longitude": float(target["longitude"]),
            "expected_archive": archive,
            "coverage_status": "current_013030_proxy_nonhistorical" if proxy else "auditable_public_release",
            "historical_coverage_status": historical_coverage_status,
            "evidence_scope": "current_013030_proxy" if proxy else "historical_release_candidate",
        }
        neighbors = np.asarray(neighbor_lists[position], dtype=np.int64)
        if neighbors.size:
            target_ns = pd.Timestamp(target["datetime_utc"]).value
            minutes = np.abs(candidate_times[neighbors] - target_ns) / (60.0 * 1e9)
            distances = haversine_km(
                target["latitude"],
                target["longitude"],
                candidates.iloc[neighbors]["latitude"].to_numpy(),
                candidates.iloc[neighbors]["longitude"].to_numpy(),
            )
            keep = (minutes <= 24 * 60) & (distances <= BROAD_RADIUS_KM)
            neighbors, minutes, distances = neighbors[keep], minutes[keep], distances[keep]
        if neighbors.size == 0:
            records.append({
                **base,
                "audit_status": "proxy_no_spatiotemporal_candidate" if proxy else "no_spatiotemporal_candidate",
                "broad_candidate_count": 0,
                "recovery_candidate_count": 0,
                "primary_candidate_count": 0,
                "best_candidate_profile_id": None,
                "best_time_difference_minutes": np.nan,
                "best_distance_km": np.nan,
                "best_candidate_source_file": None,
                "best_candidate_source_profile_index": None,
            })
            continue
        primary = (minutes <= 10) & (distances <= 5)
        recovery = (minutes <= 60) & (distances <= 10)
        scores = 0.5 * (minutes / (24 * 60)) + 0.5 * (distances / BROAD_RADIUS_KM)
        best_local = int(np.argmin(scores))
        best = candidates.iloc[int(neighbors[best_local])]
        records.append({
            **base,
            "audit_status": "proxy_candidate_requires_fingerprint" if proxy else "candidate_requires_fingerprint",
            "broad_candidate_count": int(neighbors.size),
            "recovery_candidate_count": int(np.count_nonzero(recovery)),
            "primary_candidate_count": int(np.count_nonzero(primary)),
            "best_candidate_profile_id": str(best["profile_id"]),
            "best_time_difference_minutes": float(minutes[best_local]),
            "best_distance_km": float(distances[best_local]),
            "best_candidate_source_file": str(best["source_file"]),
            "best_candidate_source_profile_index": int(best["source_profile_index"]),
        })
    return pd.DataFrame.from_records(records)


def audit_oceandepths_profiles(
    oceandepths_index: Path,
    archives: dict[str, Path],
    output: Path,
    *,
    revision: str,
    batch_size: int = 50_000,
    max_profiles: int | None = None,
    overwrite: bool = False,
    proxy_013030: Path | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output}")
    partial.unlink(missing_ok=True)
    parquet = pq.ParquetFile(oceandepths_index)
    required = {"profile", "profile_source_file", "profile_idx", "profile_juld", "latitude", "longitude"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"OceanDepths index is missing columns: {sorted(missing)}")
    candidate_sources = dict(archives)
    if proxy_013030 is not None:
        candidate_sources["013_030_PROXY"] = Path(proxy_013030)
    cache = CandidateDayCache(candidate_sources)
    writer: pq.ParquetWriter | None = None
    processed = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=sorted(required)):
            targets = batch.to_pandas()
            if max_profiles is not None:
                targets = targets.iloc[: max(max_profiles - processed, 0)]
            if targets.empty:
                break
            targets["datetime_utc"] = pd.to_datetime(targets["profile_juld"], unit="D", origin="1950-01-01", utc=True, errors="coerce")
            targets["calendar_day"] = targets["datetime_utc"].dt.floor("D")
            target_geo_valid = valid_coordinates(targets["latitude"], targets["longitude"])
            output_chunks: list[pd.DataFrame] = []

            invalid = targets.loc[~target_geo_valid | targets["datetime_utc"].isna()]
            if not invalid.empty:
                output_chunks.append(_unavailable_rows(invalid, revision, "invalid_target_metadata", None))
            valid = targets.loc[target_geo_valid & targets["datetime_utc"].notna()].copy()
            timeline = valid["datetime_utc"].map(expected_archive)
            valid["expected_archive"] = [item[0] for item in timeline]
            valid["coverage_status"] = [item[1] for item in timeline]
            unavailable = valid[valid["coverage_status"] != "auditable_public_release"]
            proxy_statuses = {"source_unresolved", "internal_snapshot_unavailable"}
            proxy_targets = unavailable[unavailable["coverage_status"].isin(proxy_statuses)] if proxy_013030 is not None else unavailable.iloc[0:0]
            unavailable_without_proxy = unavailable.drop(index=proxy_targets.index)
            if not unavailable_without_proxy.empty:
                for status, group in unavailable_without_proxy.groupby("coverage_status", sort=False):
                    output_chunks.append(_unavailable_rows(group, revision, str(status), None))
            if not proxy_targets.empty:
                for (historical_status, day), group in proxy_targets.groupby(["coverage_status", "calendar_day"], sort=False):
                    output_chunks.append(
                        _audit_target_day(
                            group,
                            _candidate_frame(cache, "013_030_PROXY", day),
                            archive="013_030_PROXY",
                            revision=revision,
                            historical_coverage_status=str(historical_status),
                            proxy=True,
                        )
                    )
            auditable = valid[valid["coverage_status"] == "auditable_public_release"]
            for (archive, day), group in auditable.groupby(["expected_archive", "calendar_day"], sort=False):
                if archive not in archives:
                    output_chunks.append(_unavailable_rows(group, revision, "archive_not_supplied", str(archive)))
                    continue
                output_chunks.append(_audit_target_day(group, _candidate_frame(cache, str(archive), day), archive=str(archive), revision=revision))
            chunk = pd.concat(output_chunks, ignore_index=True).sort_values("target_profile", kind="stable")
            table = pa.Table.from_pandas(chunk, schema=OUTPUT_SCHEMA, preserve_index=False, safe=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
            processed += len(targets)
            if max_profiles is not None and processed >= max_profiles:
                break
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No OceanDepths profiles were audited.")
    partial.replace(output)
    return output


def _unavailable_rows(targets: pd.DataFrame, revision: str, status: str, archive: str | None) -> pd.DataFrame:
    return pd.DataFrame({
        "target_profile_id": [f"{revision}:{source}:{int(index)}" for source, index in zip(targets["profile_source_file"], targets["profile_idx"])],
        "target_profile": targets["profile"].astype(np.int64),
        "profile_source_file": targets["profile_source_file"].astype(str),
        "source_profile_idx": targets["profile_idx"].astype(np.int64),
        "datetime_utc": targets["datetime_utc"],
        "latitude": targets["latitude"].astype(float),
        "longitude": targets["longitude"].astype(float),
        "expected_archive": archive,
        "coverage_status": status,
        "historical_coverage_status": status,
        "evidence_scope": "unavailable",
        "audit_status": status,
        "broad_candidate_count": 0,
        "recovery_candidate_count": 0,
        "primary_candidate_count": 0,
        "best_candidate_profile_id": None,
        "best_time_difference_minutes": np.nan,
        "best_distance_km": np.nan,
        "best_candidate_source_file": None,
        "best_candidate_source_profile_index": None,
    })


def _archive_mapping(values: Sequence[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Archive mapping must be VERSION=PATH: {value}")
        version, path = value.split("=", 1)
        mapping[version] = Path(path)
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream an EN4-wide OceanDepths candidate-absence audit.")
    parser.add_argument("--oceandepths-index", type=Path, required=True)
    parser.add_argument("--archive", action="append", default=[], metavar="VERSION=PATH")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--proxy-013030", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    print(audit_oceandepths_profiles(
        args.oceandepths_index,
        _archive_mapping(args.archive),
        args.output,
        revision=args.revision,
        batch_size=args.batch_size,
        max_profiles=args.max_profiles,
        overwrite=args.overwrite,
        proxy_013030=args.proxy_013030,
    ))


if __name__ == "__main__":
    main()
