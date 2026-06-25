from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import xarray as xr
import yaml
import zarr

from depth_recon.data.dataset_grid_utils import (
    MISSING_TEXT_VALUES,
    _GridParams,
    _build_land_mask_patch_table,
    _center_lon_deg,
    _deep_update_config,
    _force_include_cache_hash,
    _normalize_lon,
    _parse_date_int,
    _parse_force_include_regions,
    _path_cache_hash,
    _sanitize_cache_text,
    _validate_grid_params,
)
from depth_recon.paths import config_path, resolve_config_path
from depth_recon.utils.normalizations import (
    CELSIUS_TO_KELVIN_OFFSET,
    salinity_normalize,
    temperature_normalize,
)

VALID_CODE_MAX = 254.0
NODATA_CODE = 255
DEFAULT_ACCEPTED_ARGO_QC_FLAGS = (1, 2)
ARGO_LEVEL_QC_VARS = {
    "depth": "argo_depth_qc_on_glorys_depth",
    "temp": "argo_temp_qc_on_glorys_depth",
    "psal": "argo_psal_qc_on_glorys_depth",
}
ARGO_PROFILE_QC_VARS = {
    "juld": "argo_juld_qc",
    "position": "argo_position_qc",
    "profile_depth": "argo_profile_depth_qc",
    "profile_potm": "argo_profile_potm_qc",
    "profile_psal": "argo_profile_psal_qc",
}
COMPACT_PROFILE_QC_VARS = {
    "temp": "argo_temp_profile_qc",
    "psal": "argo_psal_profile_qc",
}


def _decode_stretched_uint8(values: np.ndarray, stretch: dict[str, Any]) -> np.ndarray:
    """Decode uint8 GeoTIFF values into physical units from manifest metadata."""
    arr = np.asarray(values, dtype=np.uint8)
    nodata = int(stretch.get("nodata", NODATA_CODE))
    valid_code_max = float(stretch.get("valid_code_max", VALID_CODE_MAX))
    minimum = np.float32(stretch["minimum"])
    maximum = np.float32(stretch["maximum"])
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = arr != nodata
    out[valid] = minimum + (
        arr[valid].astype(np.float32)
        / np.float32(valid_code_max)
        * np.float32(maximum - minimum)
    )
    return out


def _kelvin_to_celsius(values: np.ndarray) -> np.ndarray:
    """Convert decoded Kelvin temperature values to Celsius for model normalization."""
    return np.asarray(values, dtype=np.float32) - np.float32(CELSIUS_TO_KELVIN_OFFSET)


def _resolve_manifest_path(root_dir: Path, raw_path: str | Path) -> Path:
    """Resolve a manifest path that may be absolute or export-root relative."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _resolve_land_mask_path(root_dir: Path, raw_path: str | Path) -> Path:
    """Resolve a land-mask path inside the packaged GeoTIFF dataset root."""
    export_path = _resolve_manifest_path(root_dir, raw_path)
    if not export_path.exists():
        raise FileNotFoundError(
            "Land-mask GeoTIFF must be present in the packaged dataset layout: "
            f"{export_path}"
        )
    return export_path


def _records_by_date(
    entries: Sequence[dict[str, Any]], root_dir: Path
) -> dict[int, Path]:
    """Map manifest raster entries by date."""
    records: dict[int, Path] = {}
    for entry in entries:
        records[int(entry["date"])] = _resolve_manifest_path(root_dir, entry["path"])
    return records


def _date_signature(dates: Sequence[int]) -> str:
    """Return a compact hashable date coverage signature."""
    if not dates:
        return "empty"
    raw = (int(min(dates)), int(max(dates)), int(len(dates)))
    return "-".join(str(value) for value in raw)


class RasterDatasetCache:
    """Small LRU cache for rasterio datasets opened by one worker process."""

    def __init__(self, max_open: int = 8) -> None:
        """Initialize a bounded raster path cache."""
        self.max_open = int(max_open)
        self._pid = os.getpid()
        self._items: OrderedDict[Path, rasterio.io.DatasetReader] = OrderedDict()

    def _ensure_current_process(self) -> None:
        """Drop inherited file handles after DataLoader worker forks."""
        pid = os.getpid()
        if pid == self._pid:
            return
        self.close()
        self._pid = pid

    def get(self, path: Path) -> rasterio.io.DatasetReader:
        """Return an opened raster dataset for ``path``."""
        self._ensure_current_process()
        path = Path(path)
        if path in self._items:
            src = self._items.pop(path)
            self._items[path] = src
            return src
        src = rasterio.open(path)
        self._items[path] = src
        while len(self._items) > self.max_open:
            _, old = self._items.popitem(last=False)
            old.close()
        return src

    def close(self) -> None:
        """Close all cached raster datasets."""
        for src in self._items.values():
            src.close()
        self._items.clear()


class GeoTIFFRasterStore:
    """Date-indexed GeoTIFF raster source for one exported variable."""

    def __init__(
        self,
        *,
        paths_by_date: dict[int, Path],
        stretch: dict[str, Any],
        cache: RasterDatasetCache,
        kelvin_temperature: bool,
    ) -> None:
        """Initialize a date-to-raster lookup."""
        self.paths_by_date = dict(paths_by_date)
        self.stretch = dict(stretch)
        self.cache = cache
        self.kelvin_temperature = bool(kelvin_temperature)

    @property
    def dates(self) -> set[int]:
        """Return available YYYYMMDD dates."""
        return set(int(value) for value in self.paths_by_date)

    def read_patch(
        self,
        *,
        target_date: int,
        grid_y0: int,
        grid_x0: int,
        tile_size: int,
    ) -> np.ndarray:
        """Read and decode one patch for ``target_date``."""
        path = self.paths_by_date[int(target_date)]
        src = self.cache.get(path)
        window = Window(
            col_off=int(grid_x0),
            row_off=int(grid_y0),
            width=int(tile_size),
            height=int(tile_size),
        )
        encoded = src.read(window=window)
        decoded = _decode_stretched_uint8(encoded, self.stretch)
        if self.kelvin_temperature:
            decoded = _kelvin_to_celsius(decoded)
        return decoded.astype(np.float32, copy=False)


def _normalize_accepted_qc_flags(values: Sequence[int] | None) -> tuple[int, ...]:
    """Return accepted ARGO QC flags as small integer codes."""
    if values is None:
        return DEFAULT_ACCEPTED_ARGO_QC_FLAGS
    flags = tuple(sorted({int(value) for value in values}))
    if not flags:
        raise ValueError("accepted_argo_qc_flags must contain at least one code.")
    return flags


class ArgoGeoTIFFProfileStore:
    """Profile-indexed ARGO zarr source exported with the GeoTIFF dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        include_salinity: bool = False,
        filter_bad_quality: bool = True,
        accepted_qc_flags: Sequence[int] | None = None,
    ) -> None:
        """Open a compact ARGO profile zarr store."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"ARGO profile zarr does not exist: {self.path}")
        self.include_salinity = bool(include_salinity)
        self.filter_bad_quality = bool(filter_bad_quality)
        self.accepted_qc_flags = _normalize_accepted_qc_flags(accepted_qc_flags)
        self._pid = os.getpid()
        self.ds = self._open_dataset()
        self._zarr_pid = os.getpid()
        self._zarr_group = self._open_zarr_group()
        required = {
            "target_date",
            "grid_row",
            "grid_col",
            "argo_temp_kelvin_uint8",
            "argo_temp_valid",
        }
        if self.include_salinity:
            required.update({"argo_psal_uint8", "argo_psal_valid"})
        missing = sorted(name for name in required if name not in self.ds)
        if missing:
            raise RuntimeError(
                f"ARGO profile zarr is missing required variables {missing}: {self.path}"
            )
        self.target_date = np.asarray(self.ds["target_date"].values, dtype=np.int32)
        self.grid_row = np.asarray(self.ds["grid_row"].values, dtype=np.int32)
        self.grid_col = np.asarray(self.ds["grid_col"].values, dtype=np.int32)
        self.depth_axis_m = np.asarray(
            self.ds["glorys_depth"].values, dtype=np.float32
        ).reshape(-1)
        temp_valid = np.asarray(self.ds["argo_temp_valid"].values, dtype=bool)
        if self.filter_bad_quality:
            temp_valid &= self._quality_mask_for_variable("temp")
        self._has_valid_temp = temp_valid.any(axis=1)
        (
            self._valid_profile_indices_by_date,
            self._profile_index_bounds_by_date,
        ) = self._build_valid_profile_index()
        self.temperature_stretch = self._temperature_stretch()
        self.salinity_stretch = (
            self._salinity_stretch() if self.include_salinity else None
        )

    def _open_dataset(self) -> xr.Dataset:
        """Open the zarr dataset in the current process."""
        return xr.open_zarr(self.path, consolidated=None)

    def _open_zarr_group(self) -> zarr.Group:
        """Open the zarr group used for direct array reads."""
        return zarr.open_group(self.path, mode="r")

    def _ensure_current_process(self) -> xr.Dataset:
        """Reopen zarr handles after DataLoader worker forks."""
        pid = os.getpid()
        if pid == self._pid:
            return self.ds
        # Do not close inherited xarray/zarr handles in a forked worker; closing
        # those locks after fork can block before the worker reads its first batch.
        self.ds = self._open_dataset()
        self._pid = pid
        return self.ds

    def _ensure_zarr_group(self) -> zarr.Group:
        """Return a direct zarr group opened in the current process."""
        pid = os.getpid()
        if pid != self._zarr_pid:
            self._zarr_group = self._open_zarr_group()
            self._zarr_pid = pid
        return self._zarr_group

    def _accepted_qc_mask(self, values: np.ndarray) -> np.ndarray:
        """Return True where QC is missing or one of the accepted flags."""
        qc = np.asarray(values, dtype=np.int16)
        missing = qc < 0
        accepted = np.isin(qc, np.asarray(self.accepted_qc_flags, dtype=np.int16))
        return missing | accepted

    def _profile_qc_names_for_variable(self, variable: str) -> tuple[str, ...]:
        """Return profile-level QC variables relevant to one ARGO variable."""
        names = ["juld", "position", "profile_depth"]
        if variable == "psal":
            names.append("profile_psal")
        return tuple(ARGO_PROFILE_QC_VARS[name] for name in names)

    def _level_qc_names_for_variable(self, variable: str) -> tuple[str, ...]:
        """Return level-level QC variables relevant to one ARGO variable."""
        if variable == "psal":
            return (ARGO_LEVEL_QC_VARS["depth"], ARGO_LEVEL_QC_VARS["psal"])
        return (ARGO_LEVEL_QC_VARS["depth"], ARGO_LEVEL_QC_VARS["temp"])

    def _quality_mask_for_variable(
        self,
        variable: str,
        *,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a profile-depth quality mask for one compact ARGO variable."""
        if indices is None:
            profile_count = int(self.target_date.size)
            mask = np.ones(
                (profile_count, int(self.depth_axis_m.size)),
                dtype=bool,
            )
            indexer: Any = slice(None)
        else:
            selected = np.asarray(indices, dtype=np.int64).reshape(-1)
            mask = np.ones(
                (int(selected.size), int(self.depth_axis_m.size)),
                dtype=bool,
            )
            indexer = selected
        ds = self._ensure_current_process()
        compact_name = COMPACT_PROFILE_QC_VARS.get(variable)
        if compact_name in ds:
            qc = np.asarray(ds[compact_name].isel(profile=indexer).values).reshape(-1)
            return mask & self._accepted_qc_mask(qc)[:, None]
        for name in self._level_qc_names_for_variable(variable):
            if name not in ds:
                continue
            qc = np.asarray(ds[name].isel(profile=indexer).values)
            mask &= self._accepted_qc_mask(qc)
        for name in self._profile_qc_names_for_variable(variable):
            if name not in ds:
                continue
            qc = np.asarray(ds[name].isel(profile=indexer).values).reshape(-1)
            # Profile-level QC rejects every depth for a bad date/position/profile.
            mask &= self._accepted_qc_mask(qc)[:, None]
        return mask

    def _build_valid_profile_index(
        self,
    ) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
        """Build date slices over valid-temperature profile indices."""
        valid_indices = np.flatnonzero(self._has_valid_temp).astype(np.int64)
        if valid_indices.size == 0:
            return valid_indices, {}

        # Querying per sample must not scan the full multi-million-profile store.
        order = np.argsort(self.target_date[valid_indices], kind="stable")
        sorted_indices = valid_indices[order]
        sorted_dates = self.target_date[sorted_indices]
        unique_dates, starts, counts = np.unique(
            sorted_dates, return_index=True, return_counts=True
        )
        bounds = {
            int(date): (int(start), int(start + count))
            for date, start, count in zip(
                unique_dates.tolist(),
                starts.tolist(),
                counts.tolist(),
                strict=False,
            )
        }
        return sorted_indices, bounds

    def _temperature_stretch(self) -> dict[str, Any]:
        """Read temperature stretch metadata from variable or dataset attributes."""
        ds = self._ensure_current_process()
        attrs = dict(ds["argo_temp_kelvin_uint8"].attrs)
        if "minimum" in attrs and "maximum" in attrs:
            return attrs
        ds_attrs = dict(ds.attrs)
        stretch = ds_attrs.get("temperature_stretch")
        if isinstance(stretch, dict):
            return stretch
        raise RuntimeError(
            f"ARGO profile zarr lacks temperature stretch metadata: {self.path}"
        )

    def _salinity_stretch(self) -> dict[str, Any]:
        """Read salinity stretch metadata from variable or dataset attributes."""
        ds = self._ensure_current_process()
        attrs = dict(ds["argo_psal_uint8"].attrs)
        if "minimum" in attrs and "maximum" in attrs:
            return attrs
        ds_attrs = dict(ds.attrs)
        stretch = ds_attrs.get("salinity_stretch")
        if isinstance(stretch, dict):
            return stretch
        raise RuntimeError(
            f"ARGO profile zarr lacks salinity stretch metadata: {self.path}"
        )

    def query_indices(
        self,
        *,
        target_date: int,
        grid_y0: int,
        grid_x0: int,
        tile_size: int,
    ) -> np.ndarray:
        """Return profile indices assigned to one date and grid patch."""
        y0 = int(grid_y0)
        x0 = int(grid_x0)
        tile = int(tile_size)
        bounds = self._profile_index_bounds_by_date.get(int(target_date))
        if bounds is None:
            return np.zeros((0,), dtype=np.int64)
        start, stop = bounds
        candidates = self._valid_profile_indices_by_date[start:stop]
        mask = (
            (self.grid_row[candidates] >= y0)
            & (self.grid_row[candidates] < y0 + tile)
            & (self.grid_col[candidates] >= x0)
            & (self.grid_col[candidates] < x0 + tile)
        )
        return candidates[mask].astype(np.int64, copy=False)

    def load_temperature_profiles(self, indices: np.ndarray) -> np.ndarray:
        """Load selected ARGO temperature profiles as Celsius arrays."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        depth_size = int(self.depth_axis_m.size)
        if indices.size == 0:
            return np.zeros((0, depth_size), dtype=np.float32)
        group = self._ensure_zarr_group()
        encoded = np.asarray(
            group["argo_temp_kelvin_uint8"].get_orthogonal_selection(
                (indices, slice(None))
            ),
            dtype=np.uint8,
        )
        valid = np.asarray(
            group["argo_temp_valid"].get_orthogonal_selection((indices, slice(None))),
            dtype=bool,
        )
        if self.filter_bad_quality:
            valid &= self._quality_mask_for_variable("temp", indices=indices)
        kelvin = _decode_stretched_uint8(encoded, self.temperature_stretch)
        kelvin[~valid] = np.nan
        return _kelvin_to_celsius(kelvin).astype(np.float32, copy=False)

    def load_salinity_profiles(self, indices: np.ndarray) -> np.ndarray:
        """Load selected ARGO salinity profiles as raw PSU arrays."""
        if self.salinity_stretch is None:
            raise RuntimeError(
                "ARGO salinity profiles were not enabled for this store."
            )
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        depth_size = int(self.depth_axis_m.size)
        if indices.size == 0:
            return np.zeros((0, depth_size), dtype=np.float32)
        group = self._ensure_zarr_group()
        encoded = np.asarray(
            group["argo_psal_uint8"].get_orthogonal_selection((indices, slice(None))),
            dtype=np.uint8,
        )
        valid = np.asarray(
            group["argo_psal_valid"].get_orthogonal_selection((indices, slice(None))),
            dtype=bool,
        )
        if self.filter_bad_quality:
            valid &= self._quality_mask_for_variable("psal", indices=indices)
        salinity = _decode_stretched_uint8(encoded, self.salinity_stretch)
        salinity[~valid] = np.nan
        return salinity.astype(np.float32, copy=False)

    def quality_cache_signature(self) -> str:
        """Return the ARGO quality-filter settings that affect support counts."""
        marker_text = "markers-" + "-".join(
            name for name in COMPACT_PROFILE_QC_VARS.values() if name in self.ds
        )
        flags_text = "-".join(str(value) for value in self.accepted_qc_flags)
        return _sanitize_cache_text(
            f"filter{int(self.filter_bad_quality)}_flags{flags_text}_{marker_text}"
        )

    def close(self) -> None:
        """Close the opened zarr dataset."""
        self.ds.close()


class GeoTIFFPatchIndex:
    """Build compact patch/date metadata rows for GeoTIFF training stores."""

    CACHE_VERSION = 3

    def __init__(
        self,
        *,
        root_dir: Path,
        dates: Sequence[int],
        argo_store: ArgoGeoTIFFProfileStore | None,
        cache_dir: str | Path | None,
        grid_params: _GridParams,
    ) -> None:
        """Initialize index inputs."""
        self.root_dir = Path(root_dir)
        self.dates = sorted(int(value) for value in dates)
        self.argo_store = argo_store
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.grid_params = grid_params
        _validate_grid_params(self.grid_params)
        if str(self.grid_params.patch_grid_source).strip().lower() != "land_mask":
            raise ValueError(
                "GeoTIFF datasets require grid.patch_grid_source='land_mask'."
            )

    def load_rows(self) -> list[dict[str, Any]]:
        """Load cached rows or build a fresh patch/date registry."""
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            return pd.read_csv(cache_path).to_dict(orient="records")

        patch_df = _build_land_mask_patch_table(self.grid_params)
        if self.grid_params.val_year is None:
            patch_records = patch_df.to_dict(orient="records")
            phases = self._split_phases(len(patch_records))
            for rec, phase in zip(patch_records, phases, strict=False):
                rec["split"] = phase
                rec["phase"] = phase
            patch_df = pd.DataFrame.from_records(patch_records)
        support_counts = self._build_support_counts(patch_df)
        rows: list[dict[str, Any]] = []
        export_index = 0
        for date_value in self.dates:
            for patch in patch_df.to_dict(orient="records"):
                patch_id = int(patch["patch_id"])
                row = dict(patch)
                row["date"] = int(date_value)
                row["export_index"] = int(export_index)
                if self.grid_params.val_year is not None:
                    phase = self._phase_for_date(int(date_value))
                    row["split"] = phase
                    row["phase"] = phase
                else:
                    phase = str(patch.get("split", patch.get("phase", "train")))
                    row["split"] = phase
                    row["phase"] = phase
                row["argo_profile_count"] = int(
                    support_counts.get((patch_id, int(date_value)), 0)
                )
                rows.append(row)
                export_index += 1

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(rows).to_csv(cache_path, index=False)
        return rows

    def _cache_path(self) -> Path | None:
        """Return the metadata cache path for these index settings."""
        if self.cache_dir is None:
            return None
        res_text = str(float(self.grid_params.resolution_deg)).replace(".", "p")
        land_text = str(float(self.grid_params.max_land_fraction)).replace(".", "p")
        grid_source = _sanitize_cache_text(self.grid_params.patch_grid_source)
        mask_hash = _path_cache_hash(self.grid_params.land_mask_path)
        force_hash = _force_include_cache_hash(self.grid_params.force_include_regions)
        root_hash = hashlib.sha1(str(self.root_dir).encode("utf-8")).hexdigest()[:8]
        split_text = (
            f"valyear{int(self.grid_params.val_year)}"
            if self.grid_params.val_year is not None
            else "patchsplit"
        )
        argo_quality_text = (
            "noargo"
            if self.argo_store is None
            else self.argo_store.quality_cache_signature()
        )
        name = (
            f"argo_geotiff_gridded_v{self.CACHE_VERSION}_root{root_hash}_"
            f"dates{_date_signature(self.dates)}_"
            f"tile{int(self.grid_params.tile_size)}_res{res_text}_"
            f"stride{int(self.grid_params.effective_patch_stride)}_"
            f"grid{grid_source}_land{land_text}_mask{mask_hash}_"
            f"force{force_hash}_{split_text}_argo{argo_quality_text}.csv"
        )
        return self.cache_dir / name

    def _phase_for_date(self, date_value: int) -> str:
        """Return the train/validation phase for one date."""
        year = int(date_value) // 10000
        return "val" if year == int(self.grid_params.val_year) else "train"

    def _split_phases(self, n_patches: int) -> list[str]:
        """Return deterministic spatial train/validation phases."""
        phases = np.full((int(n_patches),), "train", dtype=object)
        val_len = int(round(int(n_patches) * float(self.grid_params.val_fraction)))
        if n_patches > 1:
            val_len = min(
                max(val_len, 1 if self.grid_params.val_fraction > 0.0 else 0),
                int(n_patches) - 1,
            )
        else:
            val_len = 0
        if val_len > 0:
            rng = np.random.default_rng(int(self.grid_params.split_seed))
            val_indices = rng.permutation(np.arange(int(n_patches)))[:val_len]
            phases[val_indices] = "val"
        return [str(value) for value in phases.tolist()]

    def _build_support_counts(
        self,
        patch_df: pd.DataFrame,
    ) -> dict[tuple[int, int], int]:
        """Count ARGO profiles per overlapping patch/date row."""
        support_counts: dict[tuple[int, int], int] = {}
        if self.argo_store is None or patch_df.empty or not self.dates:
            return support_counts

        date_set = set(int(value) for value in self.dates)
        tile = int(self.grid_params.tile_size)
        patch_by_start = {
            (int(row["grid_y0"]), int(row["grid_x0"])): int(row["patch_id"])
            for row in patch_df.to_dict(orient="records")
        }
        y_starts = np.asarray(
            sorted({key[0] for key in patch_by_start}), dtype=np.int64
        )
        x_starts = np.asarray(
            sorted({key[1] for key in patch_by_start}), dtype=np.int64
        )
        for profile_idx in tqdm(
            range(int(self.argo_store.target_date.size)),
            desc="Counting ARGO overlap support",
            unit="profile",
            dynamic_ncols=True,
        ):
            if not bool(self.argo_store._has_valid_temp[profile_idx]):
                continue
            date_value = int(self.argo_store.target_date[profile_idx])
            if date_value not in date_set:
                continue
            row_idx = int(self.argo_store.grid_row[profile_idx])
            col_idx = int(self.argo_store.grid_col[profile_idx])
            y_candidates = y_starts[(y_starts <= row_idx) & (row_idx < y_starts + tile)]
            x_candidates = x_starts[(x_starts <= col_idx) & (col_idx < x_starts + tile)]
            for y0 in y_candidates.tolist():
                for x0 in x_candidates.tolist():
                    patch_id = patch_by_start.get((int(y0), int(x0)))
                    if patch_id is None:
                        continue
                    key = (int(patch_id), int(date_value))
                    support_counts[key] = support_counts.get(key, 0) + 1
        return support_counts


DEFAULT_DATASET_ROOT_DIR = Path("/work/data/OceanVariableReconstruction")
DEFAULT_GEOTIFF_ROOT_DIR = DEFAULT_DATASET_ROOT_DIR.as_posix()
DEFAULT_METADATA_CACHE_DIR = (DEFAULT_DATASET_ROOT_DIR / "depthdif_cache").as_posix()
DEFAULT_LAND_MASK_RELATIVE_PATH = "masks/world_land_mask_glorys_0p1.tif"
EO_SOURCE_DEFAULTS = {"ostia": "analysed_sst", "sss": "sos"}
EO_STRETCH_BY_SOURCE_VAR = {
    ("ostia", "analysed_sst"): ("temperature_kelvin", "temperature"),
    ("sss", "sos"): ("salinity", "salinity"),
}


class ArgoGeoTIFFGriddedPatchDataset(Dataset):
    """Dataset that lazily reads training patches from exported GeoTIFF stores."""

    DEFAULT_CONFIG_PATH = str(config_path("px_space", "training_super_config.yaml"))
    DEFAULT_GEOTIFF_ROOT_DIR = DEFAULT_DATASET_ROOT_DIR.as_posix()
    DEFAULT_METADATA_CACHE_DIR = (
        DEFAULT_DATASET_ROOT_DIR / "depthdif_cache"
    ).as_posix()

    def __init__(
        self,
        *,
        geotiff_root_dir: str | Path = DEFAULT_GEOTIFF_ROOT_DIR,
        metadata_cache_dir: str | Path | None = DEFAULT_METADATA_CACHE_DIR,
        split: str = "all",
        tile_size: int = 128,
        resolution_deg: float = 0.1,
        patch_grid_source: str = "land_mask",
        land_mask_path: str | Path | None = None,
        patch_stride: int | None = None,
        max_land_fraction: float = 0.30,
        force_include_regions: Sequence[dict[str, Any]] | None = None,
        finetune_sampling: dict[str, Any] | None = None,
        temporal_window_days: int = 7,
        glorys_var_name: str = "thetao",
        target_source: str = "glorys",
        ostia_var_name: str = "analysed_sst",
        eo_source: str = "ostia",
        eo_var_name: str | None = None,
        require_argo_for_train: bool = True,
        require_argo_for_val: bool = True,
        require_argo_for_all: bool = False,
        synthetic_mode: bool = False,
        synthetic_pixel_count: int = 250,
        return_info: bool = True,
        return_coords: bool = True,
        include_salinity: bool = False,
        output_fields: Sequence[str] | str | None = None,
        filter_bad_argo_quality: bool = True,
        accepted_argo_qc_flags: Sequence[int] | None = None,
        heldout_argo_locations: Sequence[tuple[int, int, int]] | None = None,
        random_seed: int = 7,
        cache_size: int = 8,
        val_fraction: float = 0.2,
        val_year: int | None = None,
    ) -> None:
        """Initialize the GeoTIFF-backed patch dataset."""
        self.split = str(split).strip().lower()
        if self.split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: 'all', 'train', 'val'")
        self.root_dir = Path(geotiff_root_dir)
        self.manifest_path = self.root_dir / "manifest.yaml"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"GeoTIFF manifest does not exist: {self.manifest_path}"
            )
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

        self.tile_size = int(tile_size)
        self.resolution_deg = float(resolution_deg)
        self.patch_grid_source = str(patch_grid_source)
        manifest_grid = self.manifest.get("grid", {})
        configured_land_mask = (
            land_mask_path
            or manifest_grid.get("source")
            or DEFAULT_LAND_MASK_RELATIVE_PATH
        )
        self.land_mask_path = _resolve_land_mask_path(
            self.root_dir,
            configured_land_mask,
        )
        self.patch_stride = None if patch_stride is None else int(patch_stride)
        self.max_land_fraction = float(max_land_fraction)
        self.force_include_regions = _parse_force_include_regions(force_include_regions)
        self.finetune_sampling = self._normalize_finetune_sampling(finetune_sampling)
        self.finetune_sampling_summary: dict[str, Any] = {
            "enabled": bool(self.finetune_sampling["enabled"]),
            "applied": False,
            "split": self.split,
        }
        self.temporal_window_days = int(temporal_window_days)
        self.glorys_var_name = str(glorys_var_name)
        self.target_source = self._normalize_target_source(target_source)
        self.ostia_var_name = str(ostia_var_name)
        self.eo_source, self.eo_var_name = self._normalize_eo_selection(
            eo_source=eo_source,
            eo_var_name=eo_var_name,
            ostia_var_name=self.ostia_var_name,
        )
        self.eo_stretch_name, self.eo_normalization = self._resolve_eo_metadata(
            self.eo_source, self.eo_var_name
        )
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
        self.output_fields = self._normalize_output_fields(
            output_fields, include_salinity=bool(include_salinity)
        )
        self.filter_bad_argo_quality = bool(filter_bad_argo_quality)
        self.accepted_argo_qc_flags = _normalize_accepted_qc_flags(
            accepted_argo_qc_flags
        )
        self.heldout_argo_location_keys: set[tuple[int, int, int]] = set()
        self.set_heldout_argo_locations(heldout_argo_locations)
        self.include_salinity = "salinity" in self.output_fields
        self._loads_temperature = "temperature" in self.output_fields
        self.random_seed = int(random_seed)
        self.require_argo_for_train = bool(require_argo_for_train)
        self.require_argo_for_val = bool(require_argo_for_val)
        self.require_argo_for_all = bool(require_argo_for_all)
        self.synthetic_mode = bool(synthetic_mode)
        self.synthetic_pixel_count = int(synthetic_pixel_count)
        if self.temporal_window_days < 1:
            raise ValueError("sampling.temporal_window_days must be >= 1.")
        if self.synthetic_pixel_count < 0:
            raise ValueError("synthetic.pixel_count must be >= 0.")

        self.raster_cache = RasterDatasetCache(max_open=cache_size)
        self._depth_axis_m = np.asarray(
            self.manifest.get("depth_axis_m", ()), dtype=np.float32
        ).reshape(-1)
        if self._depth_axis_m.size == 0:
            raise RuntimeError("GeoTIFF manifest is missing depth_axis_m.")

        self.argo_store = self._open_argo_store()
        if self.argo_store is not None and int(
            self.argo_store.depth_axis_m.size
        ) != int(self._depth_axis_m.size):
            raise RuntimeError(
                "ARGO profile zarr depth axis does not match GeoTIFF manifest depth_axis_m."
            )

        self.glorys_store, self.salinity_store, self.eo_store = (
            self._build_raster_stores()
        )
        # Backward-compatible alias for callers that still inspect the old name.
        self.ostia_store = self.eo_store
        self.available_dates = sorted(self.glorys_store.dates & self.eo_store.dates)
        if not self.available_dates:
            raise RuntimeError("No overlapping GeoTIFF raster dates were found.")
        if self.include_salinity:
            if self.salinity_store is None:
                raise RuntimeError("GeoTIFF salinity store was not initialized.")
            missing_salinity_dates = sorted(
                set(self.available_dates) - self.salinity_store.dates
            )
            if missing_salinity_dates:
                raise RuntimeError(
                    "GeoTIFF manifest is missing GLORYS salinity 'so' rasters "
                    f"for dates: {missing_salinity_dates[:5]}"
                )

        grid_params = _GridParams(
            tile_size=self.tile_size,
            resolution_deg=self.resolution_deg,
            invalid_threshold=0.5,
            invalid_mask_flags=("land",),
            val_fraction=float(val_fraction),
            val_year=None if val_year is None else int(val_year),
            split_seed=self.random_seed,
            patch_grid_source=self.patch_grid_source,
            land_mask_path=self.land_mask_path,
            patch_stride=self.patch_stride,
            max_land_fraction=self.max_land_fraction,
            force_include_regions=self._effective_force_include_regions(),
        )
        index = GeoTIFFPatchIndex(
            root_dir=self.root_dir,
            dates=self.available_dates,
            argo_store=self.argo_store,
            cache_dir=metadata_cache_dir,
            grid_params=grid_params,
        )
        rows = index.load_rows()
        rows = self._filter_rows(rows)
        rows = self._apply_finetune_sampling(rows)
        if not rows:
            raise RuntimeError("Dataset is empty after split/ARGO filtering.")
        self._rows = rows

    @staticmethod
    def _normalize_target_source(target_source: str) -> str:
        """Resolve the dense target raster group used as model supervision."""
        source = str(target_source or "glorys").strip().lower()
        if source not in {"glorys", "synthetic"}:
            raise ValueError("target_source must be one of: 'glorys', 'synthetic'")
        return source

    @staticmethod
    def _normalize_eo_selection(
        *,
        eo_source: str,
        eo_var_name: str | None,
        ostia_var_name: str,
    ) -> tuple[str, str]:
        """Resolve the dense surface EO raster group and variable."""
        source = str(eo_source or "ostia").strip().lower()
        if not source:
            source = "ostia"
        var_name = eo_var_name
        if var_name is None:
            var_name = (
                ostia_var_name if source == "ostia" else EO_SOURCE_DEFAULTS.get(source)
            )
        if var_name is None or not str(var_name).strip():
            raise ValueError(f"No EO variable configured for source {source!r}.")
        return source, str(var_name).strip()

    @staticmethod
    def _resolve_eo_metadata(eo_source: str, eo_var_name: str) -> tuple[str, str]:
        """Return manifest stretch and normalization family for one EO raster."""
        key = (str(eo_source).strip().lower(), str(eo_var_name).strip())
        metadata = EO_STRETCH_BY_SOURCE_VAR.get(key)
        if metadata is None:
            supported = ", ".join(
                f"{source}/{var}" for source, var in sorted(EO_STRETCH_BY_SOURCE_VAR)
            )
            raise ValueError(
                "Unsupported EO raster selection "
                f"{key[0]!r}/{key[1]!r}. Supported selections: {supported}."
            )
        return metadata

    def _open_argo_store(self) -> ArgoGeoTIFFProfileStore | None:
        """Open the optional compact ARGO zarr profile store."""
        argo_info = self.manifest.get("argo", {})
        raw_path = argo_info.get("path")
        if raw_path is None or str(raw_path).strip().lower() in MISSING_TEXT_VALUES:
            return None
        return ArgoGeoTIFFProfileStore(
            _resolve_manifest_path(self.root_dir, raw_path),
            include_salinity=self.include_salinity,
            filter_bad_quality=self.filter_bad_argo_quality,
            accepted_qc_flags=self.accepted_argo_qc_flags,
        )

    def _build_raster_stores(
        self,
    ) -> tuple[GeoTIFFRasterStore, GeoTIFFRasterStore | None, GeoTIFFRasterStore]:
        """Build date-indexed dense raster stores from manifest entries."""
        rasters = self.manifest.get("rasters", {})
        stretch = self.manifest.get("stretch", {})
        temp_stretch = stretch.get("temperature_kelvin")
        if not isinstance(temp_stretch, dict):
            raise RuntimeError(
                "GeoTIFF manifest is missing temperature_kelvin stretch."
            )
        eo_stretch = stretch.get(self.eo_stretch_name)
        if not isinstance(eo_stretch, dict):
            raise RuntimeError(
                "GeoTIFF manifest is missing EO stretch "
                f"{self.eo_stretch_name!r} for {self.eo_source}/{self.eo_var_name}."
            )
        target_rasters = rasters.get(self.target_source, {})
        target_entries = (
            target_rasters.get(self.glorys_var_name, [])
            if isinstance(target_rasters, dict)
            else []
        )
        eo_rasters = rasters.get(self.eo_source, {})
        eo_entries = (
            eo_rasters.get(self.eo_var_name, []) if isinstance(eo_rasters, dict) else []
        )
        if not target_entries or not eo_entries:
            raise RuntimeError(
                "GeoTIFF manifest is missing target/EO raster entries for "
                f"{self.target_source}/{self.glorys_var_name!r}/"
                f"{self.eo_source}/{self.eo_var_name}."
            )
        salinity_store = None
        if self.include_salinity:
            salinity_stretch = stretch.get("salinity")
            if not isinstance(salinity_stretch, dict):
                raise RuntimeError("GeoTIFF manifest is missing salinity stretch.")
            salinity_entries = (
                target_rasters.get("so", []) if isinstance(target_rasters, dict) else []
            )
            if not salinity_entries:
                raise RuntimeError(
                    "GeoTIFF manifest is missing target salinity 'so' raster entries "
                    f"for source {self.target_source!r}."
                )
            salinity_store = GeoTIFFRasterStore(
                paths_by_date=_records_by_date(salinity_entries, self.root_dir),
                stretch=salinity_stretch,
                cache=self.raster_cache,
                kelvin_temperature=False,
            )
        return (
            GeoTIFFRasterStore(
                paths_by_date=_records_by_date(target_entries, self.root_dir),
                stretch=temp_stretch,
                cache=self.raster_cache,
                kelvin_temperature=True,
            ),
            salinity_store,
            GeoTIFFRasterStore(
                paths_by_date=_records_by_date(eo_entries, self.root_dir),
                stretch=eo_stretch,
                cache=self.raster_cache,
                kelvin_temperature=self.eo_normalization == "temperature",
            ),
        )

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Return patch/date metadata rows."""
        return self._rows

    @property
    def depth_axis_m(self) -> np.ndarray:
        """Return the GLORYS depth axis in meters."""
        return self._depth_axis_m.copy()

    def set_heldout_argo_locations(
        self, locations: Sequence[tuple[int, int, int]] | None
    ) -> None:
        """Set EN4/ARGO locations excluded from sparse model inputs."""
        self.heldout_argo_location_keys = {
            (int(date_value), int(grid_row), int(grid_col))
            for date_value, grid_row, grid_col in (locations or ())
        }

    def load_heldout_argo_locations_csv(self, path: str | Path) -> None:
        """Load held-out EN4/ARGO location keys from a metrics CSV file."""
        df = pd.read_csv(path)
        required = {"date", "grid_row", "grid_col"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                "Held-out EN4 location CSV is missing required columns: "
                + ", ".join(missing)
            )
        self.set_heldout_argo_locations(
            [
                (int(row.date), int(row.grid_row), int(row.grid_col))
                for row in df.itertuples(index=False)
            ]
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        split: str = "all",
        dataset_overrides: dict[str, Any] | None = None,
    ) -> "ArgoGeoTIFFGriddedPatchDataset":
        """Build a GeoTIFF dataset from a YAML data config."""
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH
        with resolve_config_path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", cfg)
        ds_cfg = data_cfg.get("dataset", {})
        split_cfg = data_cfg.get("split", cfg.get("split", {}))
        if dataset_overrides:
            ds_cfg = _deep_update_config(ds_cfg, dataset_overrides)
        return cls(
            geotiff_root_dir=cls._cfg_get(
                ds_cfg,
                "core.geotiff_root_dir",
                "geotiff_root_dir",
                default=cls.DEFAULT_GEOTIFF_ROOT_DIR,
            ),
            metadata_cache_dir=cls._cfg_get(
                ds_cfg,
                "core.metadata_cache_dir",
                "metadata_cache_dir",
                default=cls.DEFAULT_METADATA_CACHE_DIR,
            ),
            split=split,
            tile_size=int(
                cls._cfg_get(ds_cfg, "grid.tile_size", "tile_size", default=128)
            ),
            resolution_deg=float(
                cls._cfg_get(
                    ds_cfg, "grid.resolution_deg", "resolution_deg", default=0.1
                )
            ),
            patch_grid_source=str(
                cls._cfg_get(
                    ds_cfg,
                    "grid.patch_grid_source",
                    "patch_grid_source",
                    default="land_mask",
                )
            ),
            land_mask_path=cls._cfg_get(
                ds_cfg,
                "grid.land_mask_path",
                "land_mask_path",
                default=None,
            ),
            patch_stride=cls._optional_int(
                cls._cfg_get(
                    ds_cfg,
                    "grid.patch_stride",
                    "patch_stride",
                    default=None,
                )
            ),
            max_land_fraction=float(
                cls._cfg_get(
                    ds_cfg,
                    "grid.max_land_fraction",
                    "max_land_fraction",
                    default=0.30,
                )
            ),
            force_include_regions=cls._cfg_get(
                ds_cfg,
                "grid.force_include_regions",
                "force_include_regions",
                default=None,
            ),
            finetune_sampling=cls._cfg_get(
                ds_cfg,
                "finetune_sampling",
                "finetune_sampling",
                default=None,
            ),
            temporal_window_days=int(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.temporal_window_days",
                    "temporal_window_days",
                    default=7,
                )
            ),
            glorys_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.glorys_var_name",
                    "glorys_var_name",
                    default="thetao",
                )
            ),
            target_source=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.target_source",
                    "target_source",
                    default="glorys",
                )
            ),
            ostia_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.ostia_var_name",
                    "ostia_var_name",
                    default="analysed_sst",
                )
            ),
            eo_source=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.eo_source",
                    "eo_source",
                    default="ostia",
                )
            ),
            eo_var_name=cls._cfg_get(
                ds_cfg,
                "sampling.eo_var_name",
                "eo_var_name",
                default=None,
            ),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
            val_year=cls._optional_int(split_cfg.get("val_year", None)),
            require_argo_for_train=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_train",
                    "require_argo_for_train",
                    default=True,
                )
            ),
            require_argo_for_val=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_val",
                    "require_argo_for_val",
                    default=True,
                )
            ),
            require_argo_for_all=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.require_argo_for_all",
                    "require_argo_for_all",
                    default=False,
                )
            ),
            filter_bad_argo_quality=bool(
                cls._cfg_get(
                    ds_cfg,
                    "selection.filter_bad_argo_quality",
                    "filter_bad_argo_quality",
                    default=True,
                )
            ),
            accepted_argo_qc_flags=cls._cfg_get(
                ds_cfg,
                "selection.accepted_argo_qc_flags",
                "accepted_argo_qc_flags",
                default=None,
            ),
            synthetic_mode=bool(
                cls._cfg_get(
                    ds_cfg, "synthetic.enabled", "synthetic_enabled", default=False
                )
            ),
            synthetic_pixel_count=int(
                cls._cfg_get(
                    ds_cfg,
                    "synthetic.pixel_count",
                    "synthetic_pixel_count",
                    default=250,
                )
            ),
            return_info=bool(
                cls._cfg_get(ds_cfg, "output.return_info", "return_info", default=True)
            ),
            return_coords=bool(
                cls._cfg_get(
                    ds_cfg, "output.return_coords", "return_coords", default=True
                )
            ),
            include_salinity=bool(
                cls._cfg_get(
                    ds_cfg,
                    "output.include_salinity",
                    "include_salinity",
                    default=False,
                )
            ),
            output_fields=cls._cfg_get(
                ds_cfg, "output.fields", "output_fields", default=None
            ),
            random_seed=int(
                cls._cfg_get(ds_cfg, "runtime.random_seed", "random_seed", default=7)
            ),
            cache_size=int(
                cls._cfg_get(ds_cfg, "runtime.cache_size", "cache_size", default=8)
            ),
        )

    @staticmethod
    def _cfg_get(
        cfg: dict[str, Any],
        nested_key: str,
        flat_key: str,
        *,
        default: Any,
    ) -> Any:
        """Read nested config values while keeping flat-key compatibility."""
        node: Any = cfg
        for part in nested_key.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
        _ = flat_key
        return default

    @staticmethod
    def _normalize_output_fields(
        output_fields: Sequence[str] | str | None,
        *,
        include_salinity: bool,
    ) -> tuple[str, ...]:
        """Resolve physical fields loaded for each dataset sample."""
        if output_fields is None:
            return ("temperature", "salinity") if include_salinity else ("temperature",)
        if isinstance(output_fields, str):
            fields = (output_fields,)
        else:
            fields = tuple(str(field) for field in output_fields)
        normalized = tuple(field.strip().lower() for field in fields if field.strip())
        if not normalized:
            raise ValueError("dataset.output.fields must contain at least one field.")
        unsupported = sorted(set(normalized) - {"temperature", "salinity"})
        if unsupported:
            raise ValueError(
                "dataset.output.fields contains unsupported fields: "
                f"{unsupported}. Supported fields are: temperature, salinity."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("dataset.output.fields cannot contain duplicates.")
        return normalized

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        """Parse nullable integer config values."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in MISSING_TEXT_VALUES:
            return None
        return int(value)

    @staticmethod
    def _normalize_finetune_sampling(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize optional hard-area finetuning row-sampling settings."""
        cfg = dict(raw_cfg or {})
        hard_fraction = float(cfg.get("hard_fraction", 0.75))
        if not (0.0 < hard_fraction <= 1.0):
            raise ValueError("finetune_sampling.hard_fraction must be in (0, 1].")
        default_max_land_fraction = float(cfg.get("default_max_land_fraction", 0.85))
        if not (0.0 <= default_max_land_fraction <= 1.0):
            raise ValueError(
                "finetune_sampling.default_max_land_fraction must be in [0, 1]."
            )

        raw_splits = cfg.get("apply_to_splits", ("train",))
        if isinstance(raw_splits, str):
            apply_to_splits = (raw_splits.strip().lower(),)
        else:
            apply_to_splits = tuple(str(value).strip().lower() for value in raw_splits)
        if not apply_to_splits or any(
            value not in {"all", "train", "val"} for value in apply_to_splits
        ):
            raise ValueError(
                "finetune_sampling.apply_to_splits must contain split names from "
                "{'all', 'train', 'val'}."
            )

        hard_regions: list[dict[str, Any]] = []
        for idx, raw_region in enumerate(cfg.get("hard_regions", ()) or ()):
            if not isinstance(raw_region, dict):
                raise ValueError(
                    "Each finetune_sampling.hard_regions item must be a mapping."
                )
            region = dict(raw_region)
            region["name"] = str(region.get("name", f"hard_region_{idx}"))
            region["lon_min"] = float(region["lon_min"])
            region["lon_max"] = float(region["lon_max"])
            region["lat_min"] = float(region["lat_min"])
            region["lat_max"] = float(region["lat_max"])
            region["max_land_fraction"] = float(
                region.get("max_land_fraction", default_max_land_fraction)
            )
            if not (0.0 <= region["max_land_fraction"] <= 1.0):
                raise ValueError(
                    "finetune_sampling.hard_regions[].max_land_fraction must be "
                    "in [0, 1]."
                )
            hard_regions.append(region)

        return {
            "enabled": bool(cfg.get("enabled", False)),
            "hard_fraction": hard_fraction,
            "apply_to_splits": apply_to_splits,
            "relax_land_filter": bool(cfg.get("relax_land_filter", True)),
            "default_max_land_fraction": default_max_land_fraction,
            "hard_regions": tuple(hard_regions),
        }

    def _finetune_applies_to_current_split(self) -> bool:
        """Return whether hard-area finetuning should filter this split."""
        if not bool(self.finetune_sampling["enabled"]):
            return False
        apply_to_splits = set(self.finetune_sampling["apply_to_splits"])
        return "all" in apply_to_splits or self.split in apply_to_splits

    def _effective_force_include_regions(self) -> tuple[Any, ...]:
        """Return force-include regions, extended by finetune boxes when needed."""
        if not (
            self._finetune_applies_to_current_split()
            and bool(self.finetune_sampling["relax_land_filter"])
        ):
            return self.force_include_regions

        merged = {region.name: region for region in self.force_include_regions}
        for raw_region in self.finetune_sampling["hard_regions"]:
            parsed_region = _parse_force_include_regions([raw_region])[0]
            existing = merged.get(parsed_region.name)
            if existing is not None:
                # Duplicate named boxes keep the most permissive finetune land cap.
                parsed_region = parsed_region.__class__(
                    name=parsed_region.name,
                    lon_min=parsed_region.lon_min,
                    lon_max=parsed_region.lon_max,
                    lat_min=parsed_region.lat_min,
                    lat_max=parsed_region.lat_max,
                    max_land_fraction=max(
                        float(existing.max_land_fraction),
                        float(parsed_region.max_land_fraction),
                    ),
                )
            merged[parsed_region.name] = parsed_region
        return tuple(merged.values())

    @staticmethod
    def _row_in_hard_region(
        row: dict[str, Any], regions: Sequence[dict[str, Any]]
    ) -> bool:
        """Return whether a patch center falls inside any hard finetune box."""
        lat_center = float(row.get("lat_center", np.nan))
        lon_center = _normalize_lon(float(row.get("lon_center", np.nan)))
        if not (np.isfinite(lat_center) and np.isfinite(lon_center)):
            return False
        for region in regions:
            lat_min = min(float(region["lat_min"]), float(region["lat_max"]))
            lat_max = max(float(region["lat_min"]), float(region["lat_max"]))
            lon_min = min(float(region["lon_min"]), float(region["lon_max"]))
            lon_max = max(float(region["lon_min"]), float(region["lon_max"]))
            if lat_min <= lat_center <= lat_max and lon_min <= lon_center <= lon_max:
                return True
        return False

    def _apply_finetune_sampling(
        self, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply deterministic hard/easy row filtering for finetuning runs."""
        if not self._finetune_applies_to_current_split():
            self.finetune_sampling_summary = {
                "enabled": bool(self.finetune_sampling["enabled"]),
                "applied": False,
                "split": self.split,
                "total_rows": len(rows),
            }
            return rows

        regions = self.finetune_sampling["hard_regions"]
        hard_indices = [
            idx
            for idx, row in enumerate(rows)
            if self._row_in_hard_region(row, regions)
        ]
        if not hard_indices:
            raise RuntimeError(
                "Finetune hard-area sampling matched no rows for split "
                f"{self.split!r}. Check data.dataset.finetune_sampling.hard_regions."
            )

        hard_fraction = float(self.finetune_sampling["hard_fraction"])
        hard_index_set = set(hard_indices)
        easy_indices = [idx for idx in range(len(rows)) if idx not in hard_index_set]
        requested_easy = int(
            round(len(hard_indices) * (1.0 - hard_fraction) / hard_fraction)
        )
        selected_easy: list[int] = []
        if requested_easy > 0 and easy_indices:
            sample_count = min(int(requested_easy), len(easy_indices))
            rng = np.random.default_rng(int(self.random_seed))
            selected_easy = sorted(
                int(value)
                for value in rng.choice(easy_indices, size=sample_count, replace=False)
            )

        selected_indices = sorted(hard_indices + selected_easy)
        filtered_rows = [rows[idx] for idx in selected_indices]
        actual_hard_fraction = len(hard_indices) / float(len(filtered_rows))
        self.finetune_sampling_summary = {
            "enabled": True,
            "applied": True,
            "split": self.split,
            "target_hard_fraction": hard_fraction,
            "actual_hard_fraction": actual_hard_fraction,
            "hard_rows": len(hard_indices),
            "easy_rows": len(selected_easy),
            "total_rows": len(filtered_rows),
            "available_easy_rows": len(easy_indices),
            "region_names": [str(region["name"]) for region in regions],
        }
        return filtered_rows

    def _filter_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply split and ARGO-support filters."""
        if self.split in {"train", "val"}:
            rows = [
                row
                for row in rows
                if str(row.get("split", row.get("phase", ""))).strip().lower()
                == self.split
            ]
        require_argo = self._require_argo_for_current_split()
        if require_argo:
            rows = [row for row in rows if int(row.get("argo_profile_count", 0)) > 0]
        return rows

    def _require_argo_for_current_split(self) -> bool:
        """Return whether the current split requires sparse ARGO support."""
        if self.synthetic_mode:
            return False
        if self.split == "train":
            return self.require_argo_for_train
        if self.split == "val":
            return self.require_argo_for_val
        return self.require_argo_for_all

    def __len__(self) -> int:
        """Return dataset row count."""
        return len(self._rows)

    def _load_y_patch(self, row: dict[str, Any]) -> np.ndarray:
        """Load the dense GLORYS target patch."""
        y_np = self.glorys_store.read_patch(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if y_np.ndim != 3:
            raise RuntimeError(
                f"Expected GLORYS patch shape (D,H,W), got {tuple(y_np.shape)}"
            )
        if int(y_np.shape[0]) != int(self._depth_axis_m.size):
            raise RuntimeError(
                "GLORYS raster band count does not match manifest depth_axis_m: "
                f"{int(y_np.shape[0])} != {int(self._depth_axis_m.size)}"
            )
        return y_np.astype(np.float32, copy=False)

    def _load_y_salinity_patch(self, row: dict[str, Any]) -> np.ndarray:
        """Load the dense GLORYS salinity target patch as raw PSU."""
        if self.salinity_store is None:
            raise RuntimeError("GeoTIFF salinity output is not enabled.")
        salinity_np = self.salinity_store.read_patch(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if salinity_np.ndim != 3:
            raise RuntimeError(
                "Expected GLORYS salinity patch shape (D,H,W), "
                f"got {tuple(salinity_np.shape)}"
            )
        if int(salinity_np.shape[0]) != int(self._depth_axis_m.size):
            raise RuntimeError(
                "GLORYS salinity raster band count does not match manifest "
                f"depth_axis_m: {int(salinity_np.shape[0])} != "
                f"{int(self._depth_axis_m.size)}"
            )
        return salinity_np.astype(np.float32, copy=False)

    def _load_land_mask_patch(self, row: dict[str, Any]) -> np.ndarray:
        """Load the configured on-disk world-mask patch as an ocean mask."""
        src = self.raster_cache.get(self.land_mask_path)
        window = Window(
            col_off=int(row["grid_x0"]),
            row_off=int(row["grid_y0"]),
            width=int(self.tile_size),
            height=int(self.tile_size),
        )
        land_np = src.read(1, window=window)
        expected_shape = (int(self.tile_size), int(self.tile_size))
        if land_np.shape != expected_shape:
            raise RuntimeError(
                "Land-mask patch shape does not match dataset tile_size: "
                f"{tuple(land_np.shape)} != {expected_shape}"
            )
        # The world raster stores 1 for land, while model masks use 1 for ocean.
        return (np.asarray(land_np, dtype=np.float32) <= 0.5).astype(
            np.float32,
            copy=False,
        )[None, ...]

    def _load_eo_patch(self, row: dict[str, Any]) -> np.ndarray:
        """Load the configured dense surface-context patch."""
        eo_np = self.eo_store.read_patch(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if eo_np.ndim == 3 and int(eo_np.shape[0]) == 1:
            eo_np = eo_np[0]
        if eo_np.ndim != 2:
            raise RuntimeError(
                f"Expected EO patch shape (H,W), got {tuple(eo_np.shape)}"
            )
        return eo_np.astype(np.float32, copy=False)[None, ...]

    def _normalize_eo_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the EO channel according to its physical variable family."""
        if self.eo_normalization == "temperature":
            return temperature_normalize(mode="norm", tensor=tensor)
        if self.eo_normalization == "salinity":
            return salinity_normalize(mode="norm", tensor=tensor)
        raise RuntimeError(f"Unsupported EO normalization: {self.eo_normalization}")

    def _spatial_support_from_valid_mask(
        self,
        valid_mask_np: np.ndarray,
        *,
        source_name: str,
    ) -> np.ndarray:
        """Collapse a per-band validity mask into one spatial ocean-support mask."""
        valid_np = np.asarray(valid_mask_np, dtype=bool)
        if valid_np.ndim == 3:
            spatial_mask = valid_np.any(axis=0, keepdims=True)
        elif valid_np.ndim == 2:
            spatial_mask = valid_np[None, ...]
        else:
            raise RuntimeError(
                f"{source_name} support must be shaped as (C,H,W) or (H,W), "
                f"got {tuple(valid_np.shape)}."
            )
        expected_shape = (1, int(self.tile_size), int(self.tile_size))
        if tuple(spatial_mask.shape) != expected_shape:
            raise RuntimeError(
                f"{source_name} support shape does not match dataset tile_size: "
                f"{tuple(spatial_mask.shape)} != {expected_shape}."
            )
        return spatial_mask.astype(np.float32, copy=False)

    def _build_land_mask_patch(
        self,
        row: dict[str, Any],
        *,
        y_valid_mask_np: np.ndarray | None,
        eo_np: np.ndarray | None,
    ) -> np.ndarray:
        """Build one spatial ocean mask from GLORYS, EO, or the on-disk mask."""
        if y_valid_mask_np is not None:
            return self._spatial_support_from_valid_mask(
                y_valid_mask_np,
                source_name="GLORYS target",
            )
        if eo_np is not None:
            return self._spatial_support_from_valid_mask(
                np.isfinite(eo_np),
                source_name="EO surface context",
            )
        if self.land_mask_path.exists():
            return self._load_land_mask_patch(row)
        raise RuntimeError(
            "Could not build land_mask: GLORYS target support was unavailable, "
            "EO support was unavailable, and the configured on-disk land mask "
            f"does not exist: {self.land_mask_path}"
        )

    def _empty_sparse_patch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return an empty sparse profile patch and validity mask."""
        depth_size = int(self._depth_axis_m.size)
        shape = (depth_size, self.tile_size, self.tile_size)
        return np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=bool)

    def _rasterize_profile_values(
        self,
        row: dict[str, Any],
        indices: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize selected profile values into one sparse patch."""
        depth_size = int(self._depth_axis_m.size)
        if indices.size == 0:
            return self._empty_sparse_patch()
        if values.ndim != 2 or int(values.shape[1]) != depth_size:
            raise RuntimeError(
                "ARGO profile values do not match manifest depth_axis_m: "
                f"{tuple(values.shape)}"
            )

        value_sum = np.zeros(
            (depth_size, self.tile_size, self.tile_size), dtype=np.float64
        )
        value_count = np.zeros(
            (depth_size, self.tile_size, self.tile_size), dtype=np.uint16
        )
        y0 = int(row["grid_y0"])
        x0 = int(row["grid_x0"])
        for local_idx, profile_idx in enumerate(indices.tolist()):
            row_idx = int(self.argo_store.grid_row[int(profile_idx)]) - y0
            col_idx = int(self.argo_store.grid_col[int(profile_idx)]) - x0
            if (
                row_idx < 0
                or row_idx >= self.tile_size
                or col_idx < 0
                or col_idx >= self.tile_size
            ):
                continue
            profile = values[int(local_idx)]
            valid = np.isfinite(profile)
            if not np.any(valid):
                continue
            # Multiple ARGO profiles can land on the same grid cell and depth.
            value_sum[valid, row_idx, col_idx] += profile[valid].astype(np.float64)
            value_count[valid, row_idx, col_idx] += 1

        value_np = np.full(value_sum.shape, np.nan, dtype=np.float32)
        value_valid = value_count > 0
        value_np[value_valid] = (
            value_sum[value_valid] / value_count[value_valid].astype(np.float64)
        ).astype(
            np.float32,
            copy=False,
        )
        return value_np, value_valid

    def _query_temperature_valid_argo_indices(self, row: dict[str, Any]) -> np.ndarray:
        """Return temperature-valid ARGO indices for the current patch."""
        if self.argo_store is None:
            return np.zeros((0,), dtype=np.int64)
        indices = self.argo_store.query_indices(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if indices.size == 0 or not self.heldout_argo_location_keys:
            return indices
        keep = np.ones(indices.shape, dtype=bool)
        for local_idx, profile_idx in enumerate(indices.tolist()):
            key = (
                int(self.argo_store.target_date[int(profile_idx)]),
                int(self.argo_store.grid_row[int(profile_idx)]),
                int(self.argo_store.grid_col[int(profile_idx)]),
            )
            # Held-out locations must be removed before rasterization so
            # overlapping patches cannot leak validation profiles into x.
            if key in self.heldout_argo_location_keys:
                keep[int(local_idx)] = False
        return indices[keep]

    def _rasterize_argo_patch(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize compact ARGO temperature observations into one patch."""
        indices = self._query_temperature_valid_argo_indices(row)
        if indices.size == 0 or self.argo_store is None:
            return self._empty_sparse_patch()
        values = self.argo_store.load_temperature_profiles(indices)
        return self._rasterize_profile_values(row, indices, values)

    def _rasterize_argo_salinity_patch(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize compact ARGO salinity observations into one patch."""
        if not self.include_salinity:
            raise RuntimeError("ARGO salinity output is not enabled.")
        indices = self._query_temperature_valid_argo_indices(row)
        if indices.size == 0 or self.argo_store is None:
            return self._empty_sparse_patch()
        # Keep salinity on the same temperature-valid support used for filtering.
        values = self.argo_store.load_salinity_profiles(indices)
        return self._rasterize_profile_values(row, indices, values)

    def _synthetic_rng_for_row(
        self,
        row: dict[str, Any],
        *,
        idx: int,
    ) -> np.random.Generator:
        """Build a deterministic synthetic-sampling RNG for one row."""
        seed = np.random.SeedSequence(
            [
                int(self.random_seed),
                int(row.get("patch_id", 0)),
                int(row.get("date", 0)),
                int(idx),
            ]
        )
        return np.random.default_rng(seed)

    def _build_synthetic_x_from_glorys(
        self,
        y_np: np.ndarray,
        y_valid_mask_np: np.ndarray,
        row: dict[str, Any],
        *,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sparse synthetic observations by sampling the dense target."""
        x_np = np.full(y_np.shape, np.nan, dtype=np.float32)
        x_valid = np.zeros(y_valid_mask_np.shape, dtype=bool)
        if self.synthetic_pixel_count == 0:
            return x_np, x_valid

        valid_columns = np.asarray(y_valid_mask_np, dtype=bool).any(axis=0)
        flat_valid_columns = np.flatnonzero(valid_columns.reshape(-1))
        if flat_valid_columns.size == 0:
            return x_np, x_valid

        sample_count = min(
            int(self.synthetic_pixel_count), int(flat_valid_columns.size)
        )
        rng = self._synthetic_rng_for_row(row, idx=idx)
        selected = rng.choice(flat_valid_columns, size=sample_count, replace=False)
        row_indices, col_indices = np.unravel_index(selected, valid_columns.shape)
        for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
            depth_valid = y_valid_mask_np[:, int(row_idx), int(col_idx)]
            if not np.any(depth_valid):
                continue
            # Synthetic mode uses decoded dense target values as sparse input.
            x_np[depth_valid, int(row_idx), int(col_idx)] = y_np[
                depth_valid,
                int(row_idx),
                int(col_idx),
            ]
            x_valid[depth_valid, int(row_idx), int(col_idx)] = True
        return x_np, x_valid

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one model-ready training sample."""
        row = self._rows[int(idx)]
        eo_np = self._load_eo_patch(row)
        temperature_payload: dict[str, torch.Tensor] | None = None
        salinity_payload: dict[str, torch.Tensor] | None = None
        land_support_np: np.ndarray | None = None

        if self._loads_temperature:
            y_np = self._load_y_patch(row)
            y_valid_mask_np = np.isfinite(y_np)
            if self.synthetic_mode:
                x_np, x_valid_mask_np = self._build_synthetic_x_from_glorys(
                    y_np,
                    y_valid_mask_np,
                    row,
                    idx=int(idx),
                )
            else:
                x_np, x_valid_mask_np = self._rasterize_argo_patch(row)

            x = temperature_normalize(mode="norm", tensor=torch.from_numpy(x_np))
            y = temperature_normalize(mode="norm", tensor=torch.from_numpy(y_np))
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            x_valid_mask = torch.from_numpy(
                x_valid_mask_np.astype(np.bool_, copy=False)
            )
            y_valid_mask = torch.from_numpy(
                y_valid_mask_np.astype(np.bool_, copy=False)
            )
            temperature_payload = {
                "x": x,
                "y": y,
                "x_valid_mask": x_valid_mask,
                "y_valid_mask": y_valid_mask,
                "x_valid_mask_1d": x_valid_mask.any(dim=0, keepdim=True),
            }
            land_support_np = y_valid_mask_np

        if self.include_salinity:
            y_salinity_np = self._load_y_salinity_patch(row)
            y_salinity_valid_mask_np = np.isfinite(y_salinity_np)
            if self.synthetic_mode:
                x_salinity_np, x_salinity_valid_mask_np = (
                    self._build_synthetic_x_from_glorys(
                        y_salinity_np,
                        y_salinity_valid_mask_np,
                        row,
                        idx=int(idx),
                    )
                )
            else:
                x_salinity_np, x_salinity_valid_mask_np = (
                    self._rasterize_argo_salinity_patch(row)
                )
            x_salinity = salinity_normalize(
                mode="norm", tensor=torch.from_numpy(x_salinity_np)
            )
            y_salinity = salinity_normalize(
                mode="norm", tensor=torch.from_numpy(y_salinity_np)
            )
            x_salinity = torch.nan_to_num(x_salinity, nan=0.0, posinf=0.0, neginf=0.0)
            y_salinity = torch.nan_to_num(y_salinity, nan=0.0, posinf=0.0, neginf=0.0)
            x_salinity_valid_mask = torch.from_numpy(
                x_salinity_valid_mask_np.astype(np.bool_, copy=False)
            )
            y_salinity_valid_mask = torch.from_numpy(
                y_salinity_valid_mask_np.astype(np.bool_, copy=False)
            )
            salinity_payload = {
                "x_salinity": x_salinity,
                "y_salinity": y_salinity,
                "x_salinity_valid_mask": x_salinity_valid_mask,
                "y_salinity_valid_mask": y_salinity_valid_mask,
                "x_salinity_valid_mask_1d": x_salinity_valid_mask.any(
                    dim=0, keepdim=True
                ),
            }
            if land_support_np is None:
                # Salinity-only runs should derive the spatial mask from salinity support.
                land_support_np = y_salinity_valid_mask_np

        land_mask_np = self._build_land_mask_patch(
            row,
            y_valid_mask_np=land_support_np,
            eo_np=eo_np,
        )
        eo = self._normalize_eo_tensor(torch.from_numpy(eo_np))
        eo = torch.nan_to_num(eo, nan=0.0, posinf=0.0, neginf=0.0)
        sample: dict[str, Any] = {
            "eo": eo,
            "land_mask": torch.from_numpy(land_mask_np),
            "date": _parse_date_int(row.get("date", 19700115)),
        }
        if temperature_payload is not None:
            sample.update(temperature_payload)
        if salinity_payload is not None:
            sample.update(salinity_payload)
        if self.return_coords:
            sample["coords"] = torch.tensor(
                [
                    0.5 * (float(row["lat0"]) + float(row["lat1"])),
                    _center_lon_deg(float(row["lon0"]), float(row["lon1"])),
                ],
                dtype=torch.float32,
            )
        if self.return_info:
            info = dict(row)
            info["target_source"] = self.target_source
            info["x_source"] = "glorys_synthetic" if self.synthetic_mode else "argo"
            info["synthetic_pixel_count"] = (
                int(self.synthetic_pixel_count) if self.synthetic_mode else 0
            )
            sample["info"] = info
        return sample
