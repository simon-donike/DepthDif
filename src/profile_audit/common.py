from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_longitude_180(values: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    normalized = ((array + 180.0) % 360.0) - 180.0
    return float(normalized) if normalized.ndim == 0 else normalized


def normalize_longitude_360(values: Any) -> Any:
    array = np.asarray(values, dtype=np.float64) % 360.0
    return float(array) if array.ndim == 0 else array


def valid_coordinates(latitude: Any, longitude: Any) -> np.ndarray:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    return np.isfinite(lat) & np.isfinite(lon) & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """Great-circle distance with longitude differences wrapped at the dateline."""
    lat1_rad, lat2_rad = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))
    delta_lat = lat2_rad - lat1_rad
    delta_lon = np.radians(normalize_longitude_180(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)))
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def normalize_identifier(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def decode_char_array(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 1 and array.dtype.kind in {"S", "U"}:
        if array.dtype.kind == "S":
            return np.asarray([b"".join(row).decode("utf-8", "replace").strip() for row in array])
        return np.asarray(["".join(row).strip() for row in array])
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8", errors="replace")
    return array.astype(str)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"} or path.is_dir():
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def first_existing(names: Iterable[str], available: Iterable[str]) -> str | None:
    available_set = set(available)
    return next((name for name in names if name in available_set), None)
