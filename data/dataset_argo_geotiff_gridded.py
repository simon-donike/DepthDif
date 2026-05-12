from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
import xarray as xr
import yaml

from data.dataset_argo_netcdf_gridded import (
    DEFAULT_LAND_MASK_PATH,
    MISSING_TEXT_VALUES,
    _GridParams,
    _build_land_mask_patch_table,
    _center_lon_deg,
    _deep_update_config,
    _force_include_cache_hash,
    _parse_date_int,
    _parse_force_include_regions,
    _path_cache_hash,
    _sanitize_cache_text,
    _validate_grid_params,
)
from utils.normalizations import CELSIUS_TO_KELVIN_OFFSET, temperature_normalize

VALID_CODE_MAX = 254.0
NODATA_CODE = 255


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
        self._items: OrderedDict[Path, rasterio.io.DatasetReader] = OrderedDict()

    def get(self, path: Path) -> rasterio.io.DatasetReader:
        """Return an opened raster dataset for ``path``."""
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


class ArgoGeoTIFFProfileStore:
    """Profile-indexed ARGO zarr source exported with the GeoTIFF dataset."""

    def __init__(self, path: str | Path) -> None:
        """Open a compact ARGO profile zarr store."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"ARGO profile zarr does not exist: {self.path}")
        self.ds = xr.open_zarr(self.path, consolidated=None)
        required = {
            "target_date",
            "grid_row",
            "grid_col",
            "argo_temp_kelvin_uint8",
            "argo_temp_valid",
        }
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
        self._has_valid_temp = temp_valid.any(axis=1)
        self.temperature_stretch = self._temperature_stretch()

    def _temperature_stretch(self) -> dict[str, Any]:
        """Read temperature stretch metadata from variable or dataset attributes."""
        attrs = dict(self.ds["argo_temp_kelvin_uint8"].attrs)
        if "minimum" in attrs and "maximum" in attrs:
            return attrs
        ds_attrs = dict(self.ds.attrs)
        stretch = ds_attrs.get("temperature_stretch")
        if isinstance(stretch, dict):
            return stretch
        raise RuntimeError(
            f"ARGO profile zarr lacks temperature stretch metadata: {self.path}"
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
        mask = (
            (self.target_date == int(target_date))
            & self._has_valid_temp
            & (self.grid_row >= y0)
            & (self.grid_row < y0 + tile)
            & (self.grid_col >= x0)
            & (self.grid_col < x0 + tile)
        )
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    def load_temperature_profiles(self, indices: np.ndarray) -> np.ndarray:
        """Load selected ARGO temperature profiles as Celsius arrays."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        depth_size = int(self.depth_axis_m.size)
        if indices.size == 0:
            return np.zeros((0, depth_size), dtype=np.float32)
        encoded = np.asarray(
            self.ds["argo_temp_kelvin_uint8"].isel(profile=indices).values,
            dtype=np.uint8,
        )
        valid = np.asarray(
            self.ds["argo_temp_valid"].isel(profile=indices).values,
            dtype=bool,
        )
        kelvin = _decode_stretched_uint8(encoded, self.temperature_stretch)
        kelvin[~valid] = np.nan
        return _kelvin_to_celsius(kelvin).astype(np.float32, copy=False)

    def close(self) -> None:
        """Close the opened zarr dataset."""
        self.ds.close()


class GeoTIFFPatchIndex:
    """Build compact patch/date metadata rows for GeoTIFF training stores."""

    CACHE_VERSION = 1

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
        name = (
            f"argo_geotiff_gridded_v{self.CACHE_VERSION}_root{root_hash}_"
            f"dates{_date_signature(self.dates)}_"
            f"tile{int(self.grid_params.tile_size)}_res{res_text}_"
            f"stride{int(self.grid_params.effective_patch_stride)}_"
            f"grid{grid_source}_land{land_text}_mask{mask_hash}_"
            f"force{force_hash}_{split_text}.csv"
        )
        return self.cache_dir / name

    def _phase_for_date(self, date_value: int) -> str:
        """Return the train/validation phase for one date."""
        year = int(date_value) // 10000
        return "val" if year == int(self.grid_params.val_year) else "train"

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
        for profile_idx in range(int(self.argo_store.target_date.size)):
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


class ArgoGeoTIFFGriddedPatchDataset(Dataset):
    """Dataset that lazily reads training patches from exported GeoTIFF stores."""

    DEFAULT_CONFIG_PATH = "configs/px_space/data_ostia_argo_geotiff.yaml"
    DEFAULT_GEOTIFF_ROOT_DIR = "/work/data/depthdif"
    DEFAULT_METADATA_CACHE_DIR = "/work/data/depthdif/depthdif_cache"

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
        temporal_window_days: int = 7,
        glorys_var_name: str = "thetao",
        ostia_var_name: str = "analysed_sst",
        require_argo_for_train: bool = True,
        require_argo_for_val: bool = True,
        require_argo_for_all: bool = False,
        synthetic_mode: bool = False,
        synthetic_pixel_count: int = 250,
        return_info: bool = True,
        return_coords: bool = True,
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
            manifest_grid.get("source") if land_mask_path is None else land_mask_path
        )
        self.land_mask_path = Path(
            DEFAULT_LAND_MASK_PATH
            if configured_land_mask is None
            else configured_land_mask
        )
        self.patch_stride = None if patch_stride is None else int(patch_stride)
        self.max_land_fraction = float(max_land_fraction)
        self.force_include_regions = _parse_force_include_regions(force_include_regions)
        self.temporal_window_days = int(temporal_window_days)
        self.glorys_var_name = str(glorys_var_name)
        self.ostia_var_name = str(ostia_var_name)
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
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

        self.glorys_store, self.ostia_store = self._build_raster_stores()
        self.available_dates = sorted(self.glorys_store.dates & self.ostia_store.dates)
        if not self.available_dates:
            raise RuntimeError("No overlapping GeoTIFF raster dates were found.")

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
            force_include_regions=self.force_include_regions,
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
        if not rows:
            raise RuntimeError("Dataset is empty after split/ARGO filtering.")
        self._rows = rows

    def _open_argo_store(self) -> ArgoGeoTIFFProfileStore | None:
        """Open the optional compact ARGO zarr profile store."""
        argo_info = self.manifest.get("argo", {})
        raw_path = argo_info.get("path")
        if raw_path is None or str(raw_path).strip().lower() in MISSING_TEXT_VALUES:
            return None
        return ArgoGeoTIFFProfileStore(_resolve_manifest_path(self.root_dir, raw_path))

    def _build_raster_stores(self) -> tuple[GeoTIFFRasterStore, GeoTIFFRasterStore]:
        """Build date-indexed dense raster stores from manifest entries."""
        rasters = self.manifest.get("rasters", {})
        stretch = self.manifest.get("stretch", {})
        temp_stretch = stretch.get("temperature_kelvin")
        if not isinstance(temp_stretch, dict):
            raise RuntimeError(
                "GeoTIFF manifest is missing temperature_kelvin stretch."
            )
        glorys_entries = (
            rasters.get("glorys", {}).get(self.glorys_var_name, [])
            if isinstance(rasters.get("glorys", {}), dict)
            else []
        )
        ostia_entries = (
            rasters.get("ostia", {}).get(self.ostia_var_name, [])
            if isinstance(rasters.get("ostia", {}), dict)
            else []
        )
        if not glorys_entries or not ostia_entries:
            raise RuntimeError(
                "GeoTIFF manifest is missing GLORYS/OSTIA raster entries for "
                f"{self.glorys_var_name!r}/{self.ostia_var_name!r}."
            )
        return (
            GeoTIFFRasterStore(
                paths_by_date=_records_by_date(glorys_entries, self.root_dir),
                stretch=temp_stretch,
                cache=self.raster_cache,
                kelvin_temperature=True,
            ),
            GeoTIFFRasterStore(
                paths_by_date=_records_by_date(ostia_entries, self.root_dir),
                stretch=temp_stretch,
                cache=self.raster_cache,
                kelvin_temperature=True,
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
        with Path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg.get("dataset", {})
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
            ostia_var_name=str(
                cls._cfg_get(
                    ds_cfg,
                    "sampling.ostia_var_name",
                    "ostia_var_name",
                    default="analysed_sst",
                )
            ),
            val_fraction=float(cfg.get("split", {}).get("val_fraction", 0.2)),
            val_year=cls._optional_int(cfg.get("split", {}).get("val_year", None)),
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
    def _optional_int(value: Any) -> int | None:
        """Parse nullable integer config values."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in MISSING_TEXT_VALUES:
            return None
        return int(value)

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

    def _load_eo_patch(self, row: dict[str, Any]) -> np.ndarray:
        """Load the dense OSTIA surface-context patch."""
        eo_np = self.ostia_store.read_patch(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if eo_np.ndim == 3 and int(eo_np.shape[0]) == 1:
            eo_np = eo_np[0]
        if eo_np.ndim != 2:
            raise RuntimeError(
                f"Expected OSTIA patch shape (H,W), got {tuple(eo_np.shape)}"
            )
        return eo_np.astype(np.float32, copy=False)[None, ...]

    def _rasterize_argo_patch(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize compact ARGO profile observations into one patch."""
        depth_size = int(self._depth_axis_m.size)
        x_sum = np.zeros((depth_size, self.tile_size, self.tile_size), dtype=np.float64)
        x_count = np.zeros(
            (depth_size, self.tile_size, self.tile_size), dtype=np.uint16
        )
        if self.argo_store is None:
            return (
                np.full(x_sum.shape, np.nan, dtype=np.float32),
                np.zeros(x_sum.shape, dtype=bool),
            )
        indices = self.argo_store.query_indices(
            target_date=int(row["date"]),
            grid_y0=int(row["grid_y0"]),
            grid_x0=int(row["grid_x0"]),
            tile_size=self.tile_size,
        )
        if indices.size == 0:
            return (
                np.full(x_sum.shape, np.nan, dtype=np.float32),
                np.zeros(x_sum.shape, dtype=bool),
            )

        values = self.argo_store.load_temperature_profiles(indices)
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
            x_sum[valid, row_idx, col_idx] += profile[valid].astype(np.float64)
            x_count[valid, row_idx, col_idx] += 1

        x_np = np.full(x_sum.shape, np.nan, dtype=np.float32)
        x_valid = x_count > 0
        x_np[x_valid] = (x_sum[x_valid] / x_count[x_valid].astype(np.float64)).astype(
            np.float32,
            copy=False,
        )
        return x_np, x_valid

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
            # Synthetic mode uses decoded GLORYS Celsius values as sparse input.
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

        land_mask_np = y_valid_mask_np[:1].astype(np.float32, copy=False)
        eo = temperature_normalize(mode="norm", tensor=torch.from_numpy(eo_np))
        x = temperature_normalize(mode="norm", tensor=torch.from_numpy(x_np))
        y = temperature_normalize(mode="norm", tensor=torch.from_numpy(y_np))
        eo = torch.nan_to_num(eo, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x_valid_mask = torch.from_numpy(x_valid_mask_np.astype(np.bool_, copy=False))
        y_valid_mask = torch.from_numpy(y_valid_mask_np.astype(np.bool_, copy=False))
        land_mask = torch.from_numpy(land_mask_np)
        x_valid_mask_1d = x_valid_mask.any(dim=0, keepdim=True)

        sample: dict[str, Any] = {
            "eo": eo,
            "x": x,
            "y": y,
            "x_valid_mask": x_valid_mask,
            "y_valid_mask": y_valid_mask,
            "x_valid_mask_1d": x_valid_mask_1d,
            "land_mask": land_mask,
            "date": _parse_date_int(row.get("date", 19700115)),
        }
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
            info["x_source"] = "glorys_synthetic" if self.synthetic_mode else "argo"
            info["synthetic_pixel_count"] = (
                int(self.synthetic_pixel_count) if self.synthetic_mode else 0
            )
            sample["info"] = info
        return sample
