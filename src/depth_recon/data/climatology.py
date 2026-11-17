"""Training-only monthly spatial climatology artifacts."""

from __future__ import annotations

import json
import hashlib
import os
from contextlib import contextmanager
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Any
import uuid
import zipfile

import numpy as np
from rasterio.enums import Resampling
import yaml

from depth_recon.data.dataset_argo_geotiff_gridded import (
    RasterDatasetCache,
    _decode_stretched_uint8,
    _kelvin_to_celsius,
    _records_by_date,
)

_UNLOADED = object()
_CACHE_ENVIRONMENT_VARIABLE = "DEPTH_RECON_CLIMATOLOGY_CACHE_DIR"
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _climatology_cache_root() -> Path:
    """Return a writable node-local cache, including for read-only data bundles."""
    configured = os.environ.get(_CACHE_ENVIRONMENT_VARIABLE)
    candidates = [
        Path(configured) if configured else None,
        (
            Path(os.environ["SLURM_TMPDIR"]) / "depthdif_climatology_cache"
            if os.environ.get("SLURM_TMPDIR")
            else None
        ),
        Path(tempfile.gettempdir()) / "depthdif_climatology_cache",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    # The last resort is per-process but keeps read-only source bundles usable.
    return Path(tempfile.mkdtemp(prefix="depthdif_climatology_cache_"))


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize cache publication between DataLoader workers on one node."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MonthlyClimatology:
    """Load and sample a monthly, spatial, depth climatology artifact."""

    def __init__(self, path: str | Path) -> None:
        """Open an artifact and validate its serialized metadata."""
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Climatology artifact does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            self.depth_axis_m = np.asarray(data["depth_axis_m"], dtype=np.float32)
            metadata = json.loads(str(data["metadata"]))
            self._members = set(data.files)
        self.metadata = metadata
        fields = set(metadata.get("fields", ()))
        artifact_field = metadata.get("field")
        self._available_fields = {
            field
            for field in ("temperature", "salinity")
            if field in self._members
            and not (artifact_field and artifact_field != field and field not in fields)
        }
        if "count" not in self._members:
            raise ValueError("Climatology artifact is missing count data.")
        self._temperature: np.ndarray | None | object = _UNLOADED
        self._salinity: np.ndarray | None | object = _UNLOADED
        self._count: np.ndarray | object = _UNLOADED
        self._validated_fields: set[str] = set()
        self._count_validated = False
        self._artifact_sha256: str | None = None
        self._cache_dir: Path | None = None
        self.spatial_stride = int(metadata.get("spatial_stride", 1))
        if self.spatial_stride < 1:
            raise ValueError("Climatology spatial_stride must be >= 1.")
        self.grid_shape = tuple(int(v) for v in metadata["grid_shape"])
        self.source_grid_shape = tuple(
            int(v) for v in metadata.get("source_grid_shape", self.grid_shape)
        )
        if (
            self.depth_axis_m.ndim != 1
            or self.depth_axis_m.size == 0
            or not np.isfinite(self.depth_axis_m).all()
            or np.any(np.diff(self.depth_axis_m) <= 0)
        ):
            raise ValueError(
                "Climatology depth axis must be finite and strictly increasing."
            )
        expected_grid = tuple(
            (size + self.spatial_stride - 1) // self.spatial_stride
            for size in self.source_grid_shape
        )
        if (
            len(self.source_grid_shape) != 2
            or min(self.source_grid_shape) < 1
            or self.grid_shape != expected_grid
        ):
            raise ValueError("Climatology source grid and downsampled grid disagree.")
        expected_shape = (12, len(self.depth_axis_m), *self.grid_shape)
        self._expected_shape = expected_shape

    @property
    def temperature(self) -> np.ndarray | None:
        """Return the temperature field as a shared read-only memory map."""
        return self._field("temperature")

    @property
    def salinity(self) -> np.ndarray | None:
        """Return the salinity field as a shared read-only memory map."""
        return self._field("salinity")

    @property
    def count(self) -> np.ndarray:
        """Return support counts without eagerly decoding the compressed archive."""
        if self._count is _UNLOADED:
            self._count = self._load_array("count", np.int32)
        if not self._count_validated:
            self._validate_count(self._count)
            self._count_validated = True
        return self._count

    def _field(self, field: str) -> np.ndarray | None:
        """Open and validate one active science field on first use only."""
        attribute = f"_{field}"
        values = getattr(self, attribute)
        if values is _UNLOADED:
            values = (
                self._load_array(field, np.float32)
                if field in self._available_fields
                else None
            )
            setattr(self, attribute, values)
        if values is not None and field not in self._validated_fields:
            self._validate_field(field, values)
            self._validated_fields.add(field)
        return values

    def _load_array(self, name: str, dtype: np.dtype[Any]) -> np.ndarray:
        """Extract one NPZ member once and reopen it as a process-shareable mmap."""
        array_path = self._materialize_member(name)
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if values.dtype == np.dtype(dtype):
            return values
        # Production artifacts already use these dtypes. Retain compatibility with
        # older small artifacts that relied on the previous coercion behaviour.
        return np.asarray(values, dtype=dtype)

    def _validate_field(self, field: str, values: np.ndarray) -> None:
        """Validate a memory-mapped field in slabs without allocating a full copy."""
        if values.shape != self._expected_shape:
            raise ValueError(
                f"Climatology {field} must be finite with shape {self._expected_shape}."
            )
        for month in values:
            if not np.isfinite(month).all():
                raise ValueError(
                    f"Climatology {field} must be finite with shape {self._expected_shape}."
                )

    def _validate_count(self, values: np.ndarray) -> None:
        """Validate count support in slabs while retaining the mmap backing."""
        if values.shape != self._expected_shape:
            raise ValueError(
                "Climatology counts must match the field shape and be nonnegative."
            )
        for month in values:
            if np.any(month < 0):
                raise ValueError(
                    "Climatology counts must match the field shape and be nonnegative."
                )

    def _materialize_member(self, name: str) -> Path:
        """Atomically copy one compressed NPZ member into the node-local cache."""
        member = f"{name}.npy"
        if name not in self._members:
            raise ValueError(f"Climatology artifact is missing {name!r} data.")
        cache_dir = self._ensure_cache_dir()
        destination = cache_dir / member
        if destination.is_file():
            return destination
        lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
        with _exclusive_file_lock(lock_path):
            if destination.is_file():
                return destination
            temporary = cache_dir / f".{member}.{uuid.uuid4().hex}.tmp"
            try:
                with zipfile.ZipFile(self.path) as archive:
                    with archive.open(member) as source, temporary.open("wb") as target:
                        while chunk := source.read(_COPY_CHUNK_BYTES):
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def _ensure_cache_dir(self) -> Path:
        """Create a cache namespace keyed by the immutable artifact SHA-256."""
        if self._cache_dir is not None:
            return self._cache_dir
        cache_root = _climatology_cache_root()
        source_sha256 = self._source_sha256(cache_root)
        cache_dir = cache_root / source_sha256
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity_path = cache_dir / "identity.json"
        lock_path = cache_root / f".{source_sha256}.lock"
        identity = {
            "sha256": source_sha256,
            "size_bytes": self.path.stat().st_size,
        }
        with _exclusive_file_lock(lock_path):
            if identity_path.is_file():
                try:
                    existing = json.loads(identity_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing != identity:
                    # A cache directory is private to one content hash. Refuse a
                    # surprising collision rather than using potentially stale data.
                    raise ValueError(
                        "Climatology cache identity does not match artifact."
                    )
            else:
                temporary = cache_dir / f".identity.{uuid.uuid4().hex}.tmp"
                temporary.write_text(
                    json.dumps(identity, sort_keys=True), encoding="utf-8"
                )
                os.replace(temporary, identity_path)
        self._cache_dir = cache_dir
        return cache_dir

    def _source_sha256(self, cache_root: Path) -> str:
        """Return a content hash while avoiding one full source read per worker."""
        if self._artifact_sha256 is None:
            fingerprint = self._source_fingerprint()
            fingerprint_bytes = json.dumps(
                fingerprint, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            index_name = hashlib.sha256(fingerprint_bytes).hexdigest()
            index_path = cache_root / f".source-{index_name}.json"
            lock_path = cache_root / f".source-{index_name}.lock"
            with _exclusive_file_lock(lock_path):
                if index_path.is_file():
                    try:
                        cached = json.loads(index_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        cached = None
                    if cached and cached.get("fingerprint") == fingerprint:
                        source_sha256 = cached.get("sha256")
                        if isinstance(source_sha256, str) and len(source_sha256) == 64:
                            self._artifact_sha256 = source_sha256
                            return source_sha256
                digest = hashlib.sha256()
                with self.path.open("rb") as source:
                    while chunk := source.read(_COPY_CHUNK_BYTES):
                        digest.update(chunk)
                if self._source_fingerprint() != fingerprint:
                    raise RuntimeError(
                        "Climatology artifact changed while its content hash was calculated."
                    )
                self._artifact_sha256 = digest.hexdigest()
                # Publish only after hashing immutable bytes; workers can now reuse
                # this source fingerprint instead of rereading the compressed artifact.
                temporary = cache_root / f".{index_path.name}.{uuid.uuid4().hex}.tmp"
                temporary.write_text(
                    json.dumps(
                        {"fingerprint": fingerprint, "sha256": self._artifact_sha256},
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, index_path)
        return self._artifact_sha256

    def _source_fingerprint(self) -> dict[str, int]:
        """Describe the immutable source inode used to reuse a verified hash."""
        stat = self.path.stat()
        return {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "mtime_ns": int(stat.st_mtime_ns),
            "size_bytes": int(stat.st_size),
        }

    def sample_patch(
        self,
        *,
        date: int,
        grid_y0: int,
        grid_x0: int,
        tile_size: int,
        field: str = "temperature",
    ) -> np.ndarray:
        """Return an absolute-value monthly patch on the native dataset grid."""
        if field not in {"temperature", "salinity"}:
            raise ValueError(f"Unsupported climatology field {field!r}.")
        source = self.temperature if field == "temperature" else self.salinity
        if source is None:
            raise ValueError(f"Climatology does not contain field {field!r}.")
        month = datetime.strptime(str(int(date)), "%Y%m%d").month - 1
        y0, x0, size = int(grid_y0), int(grid_x0), int(tile_size)
        height, width = self.source_grid_shape
        if size < 1 or min(y0, x0) < 0 or y0 + size > height or x0 + size > width:
            raise ValueError("Climatology patch extends outside artifact grid.")
        # Match GDAL's nearest-neighbour pixel-center mapping even when dimensions
        # are not divisible by spatial_stride, and retain non-aligned patch origins.
        rows = np.floor(
            (np.arange(y0, y0 + size) + 0.5) * self.grid_shape[0] / height
        ).astype(int)
        cols = np.floor(
            (np.arange(x0, x0 + size) + 0.5) * self.grid_shape[1] / width
        ).astype(int)
        return source[month][:, rows[:, None], cols[None, :]].astype(
            np.float32, copy=False
        )


def fit_monthly_climatology(
    geotiff_root_dir: str | Path,
    output_path: str | Path,
    *,
    val_year: int,
    spatial_stride: int = 4,
    field: str = "temperature",
) -> Path:
    """Fit a streaming monthly climatology from training-year GeoTIFF rasters."""
    if val_year is None:
        raise ValueError("val_year is required for training-only climatology fitting.")
    if field == "both":
        # Fit each variable independently so a missing salinity store cannot
        # contaminate the temperature accumulator.
        with tempfile.TemporaryDirectory(prefix="depthdif_climatology_") as temp_dir:
            temp_path = fit_monthly_climatology(
                geotiff_root_dir,
                Path(temp_dir) / "temperature.npz",
                val_year=val_year,
                spatial_stride=spatial_stride,
                field="temperature",
            )
            sal_path = fit_monthly_climatology(
                geotiff_root_dir,
                Path(temp_dir) / "salinity.npz",
                val_year=val_year,
                spatial_stride=spatial_stride,
                field="salinity",
            )
            with (
                np.load(temp_path, allow_pickle=False) as temp_data,
                np.load(sal_path, allow_pickle=False) as sal_data,
            ):
                metadata = json.loads(str(temp_data["metadata"]))
                salinity_metadata = json.loads(str(sal_data["metadata"]))
                if metadata.get("val_year_excluded") != salinity_metadata.get(
                    "val_year_excluded"
                ):
                    raise ValueError("Temperature and salinity provenance disagrees.")
                metadata["fields"] = ["temperature", "salinity"]
                metadata["unsupported_depth_indices"] = {
                    **metadata.get("unsupported_depth_indices", {}),
                    **salinity_metadata.get("unsupported_depth_indices", {}),
                }
                metadata["dates_temperature"] = metadata.get("dates", [])
                metadata["dates_salinity"] = salinity_metadata.get("dates", [])
                metadata["dates"] = sorted(
                    set(metadata["dates_temperature"]) | set(metadata["dates_salinity"])
                )
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output,
                    temperature=temp_data["temperature"],
                    salinity=sal_data["salinity"],
                    count=temp_data["count"],
                    depth_axis_m=temp_data["depth_axis_m"],
                    metadata=json.dumps(metadata),
                )
        return Path(output_path)
    if field not in {"temperature", "salinity"}:
        raise ValueError("field must be 'temperature', 'salinity', or 'both'.")
    root = Path(geotiff_root_dir)
    with (root / "manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest: dict[str, Any] = yaml.safe_load(handle)
    depth = np.asarray(manifest["depth_axis_m"], dtype=np.float32)
    if (
        depth.ndim != 1
        or depth.size == 0
        or not np.isfinite(depth).all()
        or np.any(np.diff(depth) <= 0)
    ):
        raise ValueError(
            "Manifest depth_axis_m must be finite and strictly increasing."
        )
    stride = int(spatial_stride)
    if stride < 1:
        raise ValueError("spatial_stride must be >= 1.")
    grid = manifest["grid"]
    source_grid_shape = (int(grid["height"]), int(grid["width"]))
    source_entries = manifest["rasters"]["glorys"]
    variable = "thetao" if field == "temperature" else "so"
    try:
        entries = source_entries[variable]
    except KeyError as error:
        raise ValueError(
            f"Manifest has no GLORYS {variable!r} rasters required for {field}."
        ) from error
    records = _records_by_date(entries, root)
    dates = sorted(records)
    if val_year is not None:
        dates = [d for d in dates if int(str(d)[:4]) != int(val_year)]
    if not dates:
        raise ValueError(
            "No training-year dates remain after validation-year exclusion."
        )
    cache = RasterDatasetCache(max_open=2)
    stretch_name = "temperature_kelvin" if field == "temperature" else "salinity"
    stretch = manifest["stretch"][stretch_name]
    sums = counts = None
    for date in dates:
        src = cache.get(records[date])
        if (src.height, src.width) != source_grid_shape:
            raise ValueError("Raster dimensions do not match the manifest grid.")
        if "transform" in grid and not np.allclose(
            tuple(src.transform)[:6], grid["transform"][:6]
        ):
            raise ValueError("Raster transform does not match the manifest grid.")
        if grid.get("crs") and str(src.crs) != str(grid["crs"]):
            raise ValueError("Raster CRS does not match the manifest grid.")
        encoded = src.read(
            out_shape=(
                src.count,
                (src.height + stride - 1) // stride,
                (src.width + stride - 1) // stride,
            ),
            resampling=Resampling.nearest,
        )
        expected_shape = (
            (source_grid_shape[0] + stride - 1) // stride,
            (source_grid_shape[1] + stride - 1) // stride,
        )
        if encoded.shape[-2:] != expected_shape:
            raise ValueError(
                f"{field} raster shape {encoded.shape[-2:]} does not match "
                f"manifest grid and spatial_stride ({expected_shape})."
            )
        values = _decode_stretched_uint8(encoded, stretch)
        if values.shape[0] != depth.size:
            raise ValueError(
                f"{field} raster depth count {values.shape[0]} does not match "
                f"manifest depth axis ({depth.size})."
            )
        if field == "temperature":
            values = _kelvin_to_celsius(values)
        values = values.astype(np.float32)
        if sums is None:
            shape = values.shape
            sums = np.zeros((12,) + shape, dtype=np.float64)
            counts = np.zeros((12,) + shape, dtype=np.int32)
        month = int(str(date)[4:6]) - 1
        valid = np.isfinite(values)
        sums[month][valid] += values[valid]
        counts[month][valid] += 1
    cache.close()
    assert sums is not None and counts is not None
    # Fill sparse months from the same pixel in other months, then from the
    # depth-wide mean. This avoids turning an absent month into a zero field.
    clim = np.full_like(sums, np.nan, dtype=np.float32)
    for month in range(12):
        for band in range(clim.shape[1]):
            valid = counts[month, band] > 0
            values = np.divide(
                sums[month, band],
                np.maximum(counts[month, band], 1),
                out=np.full_like(sums[month, band], np.nan, dtype=np.float64),
                where=counts[month, band] > 0,
            )
            other_counts = counts[:, band].sum(axis=0)
            other_sums = sums[:, band].sum(axis=0)
            pixel_fallback = np.divide(
                other_sums,
                np.maximum(other_counts, 1),
                out=np.full_like(other_sums, np.nan, dtype=np.float64),
                where=other_counts > 0,
            )
            depth_valid = other_counts > 0
            depth_fallback = (
                other_sums[depth_valid].sum() / other_counts[depth_valid].sum()
                if np.any(depth_valid)
                else np.nan
            )
            values = np.where(np.isfinite(values), values, pixel_fallback)
            values = np.where(np.isfinite(values), values, depth_fallback)
            clim[month, band] = values.astype(np.float32)
    unsupported_depths = np.flatnonzero(
        counts.sum(axis=(0, 2, 3), dtype=np.int64) == 0
    ).tolist()
    # Entirely empty source bands are retained only to preserve channel alignment.
    # The dataset rejects valid targets at these depths; zero is never a fitted mean.
    clim[:, unsupported_depths] = 0.0
    if not np.isfinite(clim).all():
        raise ValueError(f"{field} climatology contains nonfinite fitted values.")
    metadata = {
        "field": field,
        "val_year_excluded": val_year,
        "spatial_stride": stride,
        "grid_shape": list(clim.shape[-2:]),
        "source_grid_shape": list(source_grid_shape),
        "dates": dates,
        "depth_axis_m": depth.tolist(),
        "grid": grid,
        "unsupported_depth_indices": {field: unsupported_depths},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        temperature=clim if field == "temperature" else np.full_like(clim, np.nan),
        salinity=clim if field == "salinity" else np.full_like(clim, np.nan),
        count=counts,
        depth_axis_m=depth,
        metadata=json.dumps(metadata),
    )
    return output
