from __future__ import annotations

import csv
from pathlib import Path
import time
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import torch
import h5netcdf
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset

MISSING_TEXT_VALUES = frozenset({"", "__missing__", "nan", "none", "null"})


class OstiaArgoTileDataset(Dataset):
    """CSV-driven OSTIA patch dataset with Argo linkage metadata.

    The CSV is expected to contain patch bounds (`lat0`, `lat1`, `lon0`, `lon1`),
    split labels (`phase` or `split`), and an OSTIA file path column
    (`ostia_file_path` or `matched_ostia_file_path`).

    `__getitem__` returns the OSTIA surface condition patch and, when available,
    date-matched EN4 profiles loaded from the row-linked Argo monthly NetCDF.
    """

    REQUIRED_GEO_COLUMNS = ("lat0", "lat1", "lon0", "lon1")
    OSTIA_PATH_CANDIDATE_COLUMNS = ("ostia_file_path", "matched_ostia_file_path")
    SPLIT_CANDIDATE_COLUMNS = ("phase", "split")
    ARGO_PATH_CANDIDATE_COLUMNS = ("argo_file_path",)
    GLORYS_PATH_CANDIDATE_COLUMNS = ("matched_glorys_file_path",)

    def __init__(
        self,
        csv_path: str | Path,
        *,
        root_path: str | Path | None = None,
        split: str = "all",
        tile_size: int = 128,
        days: int = 1,
        sst_var_name: str = "analysed_sst",
        output_units: str = "celsius",
        return_argo_profiles: bool = True,
        argo_temp_var_name: str = "TEMP",
        argo_depth_var_name: str = "DEPH_CORRECTED",
        glorys_var_name: str = "thetao",
        return_info: bool = True,
        verbose_init: bool = True,
    ) -> None:
        # Store constructor args as normalized runtime config for all downstream methods.
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        if root_path is not None:
            # Accept either ".../depth_v2" or its parent as root_path input.
            self._depth_v2_root = self._normalize_depth_v2_root(Path(root_path).resolve())
        else:
            self._depth_v2_root = self._find_named_ancestor(self.csv_dir, "depth_v2")
        self.split = str(split).strip().lower()
        self.tile_size = int(tile_size)
        self.days = int(days)
        self.sst_var_name = str(sst_var_name)
        self.output_units = str(output_units).strip().lower()
        self.return_argo_profiles = bool(return_argo_profiles)
        self.argo_temp_var_name = str(argo_temp_var_name)
        self.argo_depth_var_name = str(argo_depth_var_name)
        self.glorys_var_name = str(glorys_var_name)
        self.return_info = bool(return_info)
        self.verbose_init = bool(verbose_init)

        if self.days < 1:
            raise ValueError("days must be >= 1.")
        if self.days % 2 == 0:
            self.days += 1
            self._log(f"days was even; auto-adjusted to odd window length: {self.days}")
        self._window_radius_days = self.days // 2

        self._log(
            "Initializing OstiaArgoTileDataset with CSV: "
            f"{self.csv_path} from root: {self._depth_v2_root}, split: '{self.split}', "
            f"tile_size: {self.tile_size}, days: {self.days}"
        )

        # Validate basic constructor constraints early so failures are explicit.
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if self.tile_size < 2:
            raise ValueError("tile_size must be >= 2.")
        if self.output_units not in {"kelvin", "celsius"}:
            raise ValueError("output_units must be one of {'kelvin', 'celsius'}.")

        # Load CSV once during dataset construction. Runtime item reads use self._rows.
        self._log(f"Loading CSV: {self.csv_path}")
        t0 = time.perf_counter()
        df = pd.read_csv(self.csv_path)
        self._log(f"CSV loaded with {len(df)} rows in {time.perf_counter() - t0:.2f}s")
        if df.empty:
            raise RuntimeError(f"CSV has no rows: {self.csv_path}")

        # These geo columns define the physical patch extents used by BOTH OSTIA and Argo mapping.
        missing_geo = [c for c in self.REQUIRED_GEO_COLUMNS if c not in df.columns]
        if missing_geo:
            raise RuntimeError(f"CSV is missing required geo columns: {missing_geo}")

        # Detect which OSTIA-path column exists in this CSV schema variant.
        self._ostia_path_col = self._first_present_column(
            df.columns.tolist(),
            self.OSTIA_PATH_CANDIDATE_COLUMNS,
        )
        if self._ostia_path_col is None:
            raise RuntimeError(
                "CSV is missing OSTIA path column. "
                f"Expected one of {list(self.OSTIA_PATH_CANDIDATE_COLUMNS)}."
            )

        # Detect split and Argo-path columns (if present) from candidate names.
        split_col = self._first_present_column(
            df.columns.tolist(),
            self.SPLIT_CANDIDATE_COLUMNS,
        )
        
        self._argo_path_col = self._first_present_column(
            df.columns.tolist(),
            self.ARGO_PATH_CANDIDATE_COLUMNS,
        )
        self._glorys_path_col = self._first_present_column(
            df.columns.tolist(),
            self.GLORYS_PATH_CANDIDATE_COLUMNS,
        )
        if self.split in {"train", "val"}:
            if split_col is None:
                raise RuntimeError(
                    "split='train'/'val' requested but CSV has no split column. "
                    f"Expected one of {list(self.SPLIT_CANDIDATE_COLUMNS)}."
                )
            # Apply split selection up front so downstream filtering/opening only touches target split.
            self._log(f"Applying split filter: split='{self.split}' via column '{split_col}'")
            df = df[df[split_col].astype(str).str.lower() == self.split].reset_index(drop=True)
            self._log(f"Rows after split filter: {len(df)}")
        elif self.split != "all":
            raise ValueError("split must be one of: 'all', 'train', 'val'")

        # Init-time Argo filtering is intentionally CSV-only (no NetCDF open here).
        # NetCDF opening/extraction is deferred to __getitem__ for on-demand loading.
        self._log("Filtering rows to valid date-linked Argo profiles...")
        t1 = time.perf_counter()
        df = self._filter_valid_argo_rows(df)
        self._log(
            "Finished Argo profile filtering: "
            f"{len(df)} rows kept in {time.perf_counter() - t1:.2f}s"
        )

        if len(df) == 0:
            raise RuntimeError("Dataset is empty after split filtering.")

        # Keep one in-memory dataframe view keyed by patch for temporal row filtering in __getitem__.
        self._lookup_rows_by_patch = self._build_lookup_rows_by_patch(df)
        self._log(f"Prepared temporal lookup for {len(self._lookup_rows_by_patch)} patch keys.")

        # Convert to row dictionaries once for very fast __getitem__ lookup.
        self._rows = df.to_dict(orient="records")
        self._log(f"Dataset ready with {len(self._rows)} rows.")

    @staticmethod
    def _first_present_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
        col_set = set(columns)
        for name in candidates:
            if name in col_set:
                return name
        return None

    def _log(self, msg: str) -> None:
        if self.verbose_init:
            print(f"[OstiaArgoTileDataset] {msg}", flush=True)

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _parse_date_int(value: Any) -> int:
        raw = str(value).strip()
        if raw == "":
            return 0
        try:
            parsed = int(raw)
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0

    @staticmethod
    def _normalize_index_text(value: Any) -> str:
        raw = str(value).strip()
        return "" if raw.lower() in MISSING_TEXT_VALUES else raw

    @staticmethod
    def _bbox_lookup_key(*, lat0: Any, lat1: Any, lon0: Any, lon1: Any) -> str:
        # Normalize bbox orientation so fallback patch keys stay stable across rows.
        lat_lo = min(float(lat0), float(lat1))
        lat_hi = max(float(lat0), float(lat1))
        lon_lo = min(float(lon0), float(lon1))
        lon_hi = max(float(lon0), float(lon1))
        return f"bbox:{lat_lo:.6f}|{lat_hi:.6f}|{lon_lo:.6f}|{lon_hi:.6f}"

    def _patch_lookup_key_from_row(self, row: dict[str, Any] | pd.Series) -> str:
        patch_id = str(row.get("patch_id", "")).strip()
        if patch_id != "":
            return f"patch:{patch_id}"
        return self._bbox_lookup_key(
            lat0=row["lat0"],
            lat1=row["lat1"],
            lon0=row["lon0"],
            lon1=row["lon1"],
        )

    def _build_lookup_rows_by_patch(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        lookup_df = df.copy()
        lookup_df["_target_date"] = pd.to_numeric(
            lookup_df["date"], errors="coerce"
        ).fillna(0).astype(np.int64)

        n_rows = int(len(lookup_df))
        patch_id_vals = (
            lookup_df["patch_id"].astype(str).str.strip().to_numpy()
            if "patch_id" in lookup_df.columns
            else np.full((n_rows,), "", dtype=object)
        )
        lat0_vals = pd.to_numeric(lookup_df["lat0"], errors="coerce").to_numpy(dtype=np.float64)
        lat1_vals = pd.to_numeric(lookup_df["lat1"], errors="coerce").to_numpy(dtype=np.float64)
        lon0_vals = pd.to_numeric(lookup_df["lon0"], errors="coerce").to_numpy(dtype=np.float64)
        lon1_vals = pd.to_numeric(lookup_df["lon1"], errors="coerce").to_numpy(dtype=np.float64)

        patch_keys: list[str] = []
        for i in range(n_rows):
            patch_id = str(patch_id_vals[i]).strip()
            if patch_id != "":
                patch_keys.append(f"patch:{patch_id}")
                continue
            patch_keys.append(
                self._bbox_lookup_key(
                    lat0=lat0_vals[i],
                    lat1=lat1_vals[i],
                    lon0=lon0_vals[i],
                    lon1=lon1_vals[i],
                )
            )
        lookup_df["_patch_lookup_key"] = patch_keys

        rows_by_patch: dict[str, pd.DataFrame] = {}
        for patch_key, patch_df in lookup_df.groupby("_patch_lookup_key", sort=False):
            rows_by_patch[str(patch_key)] = patch_df.sort_values("_target_date").reset_index(drop=True)
        return rows_by_patch

    def _window_date_bounds(self, target_date: int) -> tuple[int, int]:
        parsed = pd.to_datetime(str(int(target_date)), format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            return int(target_date), int(target_date)
        date_lo = parsed - pd.Timedelta(days=self._window_radius_days)
        date_hi = parsed + pd.Timedelta(days=self._window_radius_days)
        return int(date_lo.strftime("%Y%m%d")), int(date_hi.strftime("%Y%m%d"))

    def _select_temporal_rows(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        target_date = self._parse_date_int(row.get("date", 0))
        if self.days <= 1 or target_date <= 0:
            return [row]

        patch_key = self._patch_lookup_key_from_row(row)
        patch_df = self._lookup_rows_by_patch.get(patch_key)
        if patch_df is None or patch_df.empty:
            return [row]

        date_lo, date_hi = self._window_date_bounds(target_date)
        window_df = patch_df[
            (patch_df["_target_date"] >= int(date_lo))
            & (patch_df["_target_date"] <= int(date_hi))
        ]
        if window_df.empty:
            return [row]
        return window_df.to_dict(orient="records")

    @staticmethod
    def _find_named_ancestor(path: Path, target_name: str) -> Path | None:
        # Walk upward from CSV location to discover a movable dataset root anchor.
        curr = path.resolve()
        while True:
            if curr.name == target_name:
                return curr
            if curr.parent == curr:
                return None
            curr = curr.parent

    @staticmethod
    def _normalize_depth_v2_root(root_path: Path) -> Path:
        # root_path may point directly to depth_v2 or to its parent directory.
        return root_path if root_path.name == "depth_v2" else root_path / "depth_v2"

    def _resolve_index_path(self, value: Any) -> Path:
        # Support absolute paths, CSV-relative paths, and `depth_v2/...` anchored paths.
        normalized = self._normalize_index_text(value)
        path = Path(normalized)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == "depth_v2":
            if self._depth_v2_root is not None:
                return self._depth_v2_root / Path(*path.parts[1:])
        return self.csv_dir / path

    @staticmethod
    def _nearest_index(axis: np.ndarray, value: float) -> int:
        """Return nearest index on a monotonic axis (ascending or descending)."""
        if axis.ndim != 1 or axis.size == 0:
            raise RuntimeError("Axis must be a non-empty 1D array.")
        asc = bool(axis[0] <= axis[-1])
        if not asc:
            axis_work = axis[::-1]
            pos = int(np.searchsorted(axis_work, value, side="left"))
            pos = max(1, min(pos, int(axis_work.size) - 1))
            left = float(axis_work[pos - 1])
            right = float(axis_work[pos])
            idx_rev = pos if abs(value - right) < abs(value - left) else (pos - 1)
            return int(axis.size - 1 - idx_rev)
        pos = int(np.searchsorted(axis, value, side="left"))
        pos = max(1, min(pos, int(axis.size) - 1))
        left = float(axis[pos - 1])
        right = float(axis[pos])
        return int(pos if abs(value - right) < abs(value - left) else (pos - 1))

    @staticmethod
    def _parse_ostia_mask_flag_metadata(mask_attrs: dict[str, Any]) -> dict[str, int]:
        """Map OSTIA mask flag names to bit values from the NetCDF metadata."""
        raw_meanings = str(mask_attrs.get("flag_meanings", "")).strip()
        raw_masks = np.asarray(mask_attrs.get("flag_masks", ()), dtype=np.int64).reshape(-1)
        if raw_meanings == "" or raw_masks.size == 0:
            return {}

        flag_names = [part.strip() for part in raw_meanings.split() if part.strip()]
        if len(flag_names) != int(raw_masks.size):
            return {}
        return {
            flag_name: int(flag_mask)
            for flag_name, flag_mask in zip(flag_names, raw_masks.tolist())
        }

    @staticmethod
    def _decode_cf_numeric(values: np.ndarray, attrs: dict[str, Any] | None) -> np.ndarray:
        """Apply common CF packed-data decoding so fast low-level reads yield physical values."""
        out = np.asarray(values, dtype=np.float32)
        if attrs is None:
            return out

        fill_values: list[float] = []
        for key in ("_FillValue", "missing_value"):
            if key in attrs:
                raw = np.asarray(attrs[key]).reshape(-1)
                for value in raw.tolist():
                    try:
                        fill_values.append(float(value))
                    except (TypeError, ValueError):
                        continue
        if fill_values:
            for fill_value in fill_values:
                out[np.isclose(out, np.float32(fill_value))] = np.nan

        scale = attrs.get("scale_factor", None)
        offset = attrs.get("add_offset", None)
        if scale is not None:
            out = out * np.float32(scale)
        if offset is not None:
            out = out + np.float32(offset)
        return out

    def _load_ostia_patch(
        self,
        *,
        ostia_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        try:
            patch = self._load_ostia_patch_h5netcdf(
                ostia_path=ostia_path,
                lat0=lat0,
                lat1=lat1,
                lon0=lon0,
                lon1=lon1,
            )
        except Exception:
            # Keep the xarray path as a compatibility fallback for files h5netcdf cannot decode.
            patch = self._load_ostia_patch_xarray(
                ostia_path=ostia_path,
                lat0=lat0,
                lat1=lat1,
                lon0=lon0,
                lon1=lon1,
            )

        if patch.ndim != 2:
            raise RuntimeError(
                f"Expected 2D SST patch, got shape {tuple(patch.shape)} at {ostia_path}"
            )
        if patch.shape != (self.tile_size, self.tile_size):
            raise RuntimeError(
                f"Unexpected SST patch shape {tuple(patch.shape)} at {ostia_path}, "
                f"expected ({self.tile_size}, {self.tile_size})."
            )

        # Replace obvious fill values with NaN so invalid regions propagate cleanly.
        patch[(~np.isfinite(patch)) | (patch > 1.0e6)] = np.nan
        return patch

    def _apply_ostia_land_mask(
        self,
        *,
        native_patch: np.ndarray,
        native_mask: np.ndarray | None,
        mask_attrs: dict[str, Any] | None,
        lat_native: np.ndarray,
        lon_native: np.ndarray,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
    ) -> np.ndarray:
        """Mask OSTIA land pixels so they do not leak implausible SST values into eo."""
        patch = np.asarray(native_patch, dtype=np.float32)
        if native_mask is None or mask_attrs is None:
            return patch

        flag_bits = self._parse_ostia_mask_flag_metadata(mask_attrs)
        land_bit = int(flag_bits.get("land", 0))
        if land_bit == 0:
            return patch

        mask_patch = self._interp_native_patch_to_tile(
            native_patch=np.asarray(native_mask, dtype=np.float32),
            lat_native=lat_native,
            lon_native=lon_native,
            lat_axis=lat_axis,
            lon_axis=lon_axis,
        )
        mask_patch_i = np.rint(mask_patch).astype(np.int64, copy=False)
        land_pixels = np.isfinite(mask_patch) & ((mask_patch_i & land_bit) != 0)
        if np.any(land_pixels):
            # Use NaN here so temporal averaging ignores land, then __getitem__ collapses leftovers to 0.
            patch = patch.copy()
            patch[land_pixels] = np.nan
        return patch

    def _interp_native_patch_to_tile(
        self,
        *,
        native_patch: np.ndarray,
        lat_native: np.ndarray,
        lon_native: np.ndarray,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
    ) -> np.ndarray:
        native_patch = np.asarray(native_patch, dtype=np.float32)
        lat_native = np.asarray(lat_native, dtype=np.float64)
        lon_native = np.asarray(lon_native, dtype=np.float64)

        # RegularGridInterpolator requires ascending axes and avoids building transient xarray objects.
        if lat_native.size > 1 and lat_native[0] > lat_native[-1]:
            lat_native = lat_native[::-1]
            native_patch = native_patch[::-1, :]
        if lon_native.size > 1 and lon_native[0] > lon_native[-1]:
            lon_native = lon_native[::-1]
            native_patch = native_patch[:, ::-1]

        target_lat, target_lon = np.meshgrid(lat_axis, lon_axis, indexing="ij")
        target_points = np.column_stack((target_lat.reshape(-1), target_lon.reshape(-1)))
        linear_interp = RegularGridInterpolator(
            (lat_native, lon_native),
            native_patch,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        patch = np.asarray(
            linear_interp(target_points).reshape(self.tile_size, self.tile_size),
            dtype=np.float32,
        )
        if np.any(~np.isfinite(patch)):
            # Fill edge or convex-hull gaps without keeping a second full reader open.
            nearest_interp = RegularGridInterpolator(
                (lat_native, lon_native),
                native_patch,
                method="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            nearest_patch = np.asarray(
                nearest_interp(target_points).reshape(self.tile_size, self.tile_size),
                dtype=np.float32,
            )
            patch = np.where(np.isfinite(patch), patch, nearest_patch)
        return patch

    def _interp_native_cube_to_tile(
        self,
        *,
        native_cube: np.ndarray,
        lat_native: np.ndarray,
        lon_native: np.ndarray,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
    ) -> np.ndarray:
        native_cube = np.asarray(native_cube, dtype=np.float32)
        lat_native = np.asarray(lat_native, dtype=np.float64)
        lon_native = np.asarray(lon_native, dtype=np.float64)

        if native_cube.ndim != 3:
            raise RuntimeError(
                f"Expected native GLORYS cube with shape (depth,lat,lon), got {tuple(native_cube.shape)}"
            )

        # Reorder axes to (lat, lon, depth) so one interpolator call returns all depth channels.
        values = np.moveaxis(native_cube, 0, -1)
        if lat_native.size > 1 and lat_native[0] > lat_native[-1]:
            lat_native = lat_native[::-1]
            values = values[::-1, :, :]
        if lon_native.size > 1 and lon_native[0] > lon_native[-1]:
            lon_native = lon_native[::-1]
            values = values[:, ::-1, :]

        target_lat, target_lon = np.meshgrid(lat_axis, lon_axis, indexing="ij")
        target_points = np.column_stack((target_lat.reshape(-1), target_lon.reshape(-1)))
        linear_interp = RegularGridInterpolator(
            (lat_native, lon_native),
            values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        cube = np.asarray(
            linear_interp(target_points).reshape(self.tile_size, self.tile_size, values.shape[-1]),
            dtype=np.float32,
        )
        if np.any(~np.isfinite(cube)):
            # Fill edge or convex-hull gaps without dropping full depth columns.
            nearest_interp = RegularGridInterpolator(
                (lat_native, lon_native),
                values,
                method="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            nearest_cube = np.asarray(
                nearest_interp(target_points).reshape(
                    self.tile_size,
                    self.tile_size,
                    values.shape[-1],
                ),
                dtype=np.float32,
            )
            cube = np.where(np.isfinite(cube), cube, nearest_cube)
        return np.moveaxis(cube, -1, 0)

    def _load_glorys_patch(
        self,
        *,
        glorys_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        try:
            patch = self._load_glorys_patch_h5netcdf(
                glorys_path=glorys_path,
                lat0=lat0,
                lat1=lat1,
                lon0=lon0,
                lon1=lon1,
            )
        except Exception:
            patch = self._load_glorys_patch_xarray(
                glorys_path=glorys_path,
                lat0=lat0,
                lat1=lat1,
                lon0=lon0,
                lon1=lon1,
            )

        if patch.ndim != 3:
            raise RuntimeError(
                f"Expected 3D GLORYS patch, got shape {tuple(patch.shape)} at {glorys_path}"
            )
        patch[(~np.isfinite(patch)) | (patch > 1.0e6)] = np.nan
        return patch

    def _load_glorys_patch_h5netcdf(
        self,
        *,
        glorys_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        with h5netcdf.File(glorys_path, "r") as ds:
            if "latitude" not in ds.variables or "longitude" not in ds.variables:
                raise RuntimeError(
                    f"GLORYS file is missing latitude/longitude coordinates: {glorys_path}"
                )
            if self.glorys_var_name not in ds.variables:
                raise RuntimeError(
                    f"GLORYS file is missing variable '{self.glorys_var_name}': {glorys_path}"
                )

            lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
            lat_axis = self._build_patch_axis(float(lat0), float(lat1))
            lon_axis = self._build_patch_axis(float(lon0), float(lon1))
            lat_lo = min(float(lat0), float(lat1))
            lat_hi = max(float(lat0), float(lat1))
            lon_lo = min(float(lon0), float(lon1))
            lon_hi = max(float(lon0), float(lon1))

            lat_idx = np.flatnonzero((lat >= lat_lo) & (lat <= lat_hi))
            lon_idx = np.flatnonzero((lon >= lon_lo) & (lon <= lon_hi))
            if lat_idx.size == 0:
                lat_i0 = self._nearest_index(lat, lat_lo)
                lat_i1 = self._nearest_index(lat, lat_hi)
                lat_idx = np.arange(min(lat_i0, lat_i1), max(lat_i0, lat_i1) + 1)
            if lon_idx.size == 0:
                lon_i0 = self._nearest_index(lon, lon_lo)
                lon_i1 = self._nearest_index(lon, lon_hi)
                lon_idx = np.arange(min(lon_i0, lon_i1), max(lon_i0, lon_i1) + 1)

            lat_start = int(lat_idx.min())
            lat_stop = int(lat_idx.max()) + 1
            lon_start = int(lon_idx.min())
            lon_stop = int(lon_idx.max()) + 1
            thetao_var = ds.variables[self.glorys_var_name]
            native_cube = np.asarray(
                thetao_var[0, :, lat_start:lat_stop, lon_start:lon_stop],
                dtype=np.float32,
            )
            return self._interp_native_cube_to_tile(
                native_cube=native_cube,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )

    def _load_glorys_patch_xarray(
        self,
        *,
        glorys_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        with xr.open_dataset(
            glorys_path,
            engine="h5netcdf",
            decode_times=False,
            cache=False,
        ) as ds:
            if "latitude" not in ds.variables or "longitude" not in ds.variables:
                raise RuntimeError(
                    f"GLORYS file is missing latitude/longitude coordinates: {glorys_path}"
                )
            if self.glorys_var_name not in ds.variables:
                raise RuntimeError(
                    f"GLORYS file is missing variable '{self.glorys_var_name}': {glorys_path}"
                )

            lat = np.asarray(ds["latitude"].to_numpy(), dtype=np.float64)
            lon = np.asarray(ds["longitude"].to_numpy(), dtype=np.float64)
            thetao_var = ds[self.glorys_var_name]
            if "time" in thetao_var.dims:
                thetao_var = thetao_var.isel(time=0)
            lat_axis = self._build_patch_axis(float(lat0), float(lat1))
            lon_axis = self._build_patch_axis(float(lon0), float(lon1))
            lat_lo = min(float(lat0), float(lat1))
            lat_hi = max(float(lat0), float(lat1))
            lon_lo = min(float(lon0), float(lon1))
            lon_hi = max(float(lon0), float(lon1))

            lat_idx = np.flatnonzero((lat >= lat_lo) & (lat <= lat_hi))
            lon_idx = np.flatnonzero((lon >= lon_lo) & (lon <= lon_hi))
            if lat_idx.size == 0:
                lat_i0 = self._nearest_index(lat, lat_lo)
                lat_i1 = self._nearest_index(lat, lat_hi)
                lat_idx = np.arange(min(lat_i0, lat_i1), max(lat_i0, lat_i1) + 1)
            if lon_idx.size == 0:
                lon_i0 = self._nearest_index(lon, lon_lo)
                lon_i1 = self._nearest_index(lon, lon_hi)
                lon_idx = np.arange(min(lon_i0, lon_i1), max(lon_i0, lon_i1) + 1)

            lat_start = int(lat_idx.min())
            lat_stop = int(lat_idx.max()) + 1
            lon_start = int(lon_idx.min())
            lon_stop = int(lon_idx.max()) + 1
            native_cube = np.asarray(
                thetao_var.isel(
                    depth=slice(None),
                    latitude=slice(lat_start, lat_stop),
                    longitude=slice(lon_start, lon_stop),
                ).to_numpy(),
                dtype=np.float32,
            )
            return self._interp_native_cube_to_tile(
                native_cube=native_cube,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )

    def _load_ostia_patch_h5netcdf(
        self,
        *,
        ostia_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        with h5netcdf.File(ostia_path, "r") as ds:
            if "lat" not in ds.variables or "lon" not in ds.variables:
                raise RuntimeError(f"OSTIA file is missing lat/lon coordinates: {ostia_path}")
            if self.sst_var_name not in ds.variables:
                raise RuntimeError(
                    f"OSTIA file is missing variable '{self.sst_var_name}': {ostia_path}"
                )

            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            lat_axis = self._build_patch_axis(float(lat0), float(lat1))
            lon_axis = self._build_patch_axis(float(lon0), float(lon1))
            lat_lo = min(float(lat0), float(lat1))
            lat_hi = max(float(lat0), float(lat1))
            lon_lo = min(float(lon0), float(lon1))
            lon_hi = max(float(lon0), float(lon1))

            lat_idx = np.flatnonzero((lat >= lat_lo) & (lat <= lat_hi))
            lon_idx = np.flatnonzero((lon >= lon_lo) & (lon <= lon_hi))
            if lat_idx.size == 0:
                lat_i0 = self._nearest_index(lat, lat_lo)
                lat_i1 = self._nearest_index(lat, lat_hi)
                lat_idx = np.arange(min(lat_i0, lat_i1), max(lat_i0, lat_i1) + 1)
            if lon_idx.size == 0:
                lon_i0 = self._nearest_index(lon, lon_lo)
                lon_i1 = self._nearest_index(lon, lon_hi)
                lon_idx = np.arange(min(lon_i0, lon_i1), max(lon_i0, lon_i1) + 1)

            lat_start = int(lat_idx.min())
            lat_stop = int(lat_idx.max()) + 1
            lon_start = int(lon_idx.min())
            lon_stop = int(lon_idx.max()) + 1
            sst_var = ds.variables[self.sst_var_name]
            mask_var = ds.variables["mask"] if "mask" in ds.variables else None
            if "time" in getattr(sst_var, "dimensions", ()):
                native_patch = np.asarray(
                    sst_var[0, lat_start:lat_stop, lon_start:lon_stop],
                    dtype=np.float32,
                )
            else:
                native_patch = np.asarray(
                    sst_var[lat_start:lat_stop, lon_start:lon_stop],
                    dtype=np.float32,
                )
            native_patch = self._decode_cf_numeric(
                native_patch,
                dict(getattr(sst_var, "attrs", {})),
            )
            native_mask = None
            mask_attrs = None
            if mask_var is not None:
                if "time" in getattr(mask_var, "dimensions", ()):
                    native_mask = np.asarray(
                        mask_var[0, lat_start:lat_stop, lon_start:lon_stop],
                        dtype=np.float32,
                    )
                else:
                    native_mask = np.asarray(
                        mask_var[lat_start:lat_stop, lon_start:lon_stop],
                        dtype=np.float32,
                    )
                mask_attrs = dict(getattr(mask_var, "attrs", {}))

            patch = self._interp_native_patch_to_tile(
                native_patch=native_patch,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )
            return self._apply_ostia_land_mask(
                native_patch=patch,
                native_mask=native_mask,
                mask_attrs=mask_attrs,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )

    def _load_ostia_patch_xarray(
        self,
        *,
        ostia_path: Path,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> np.ndarray:
        # Keep a fallback reader for non-HDF5 files, but avoid xarray interpolation allocations.
        with xr.open_dataset(
            ostia_path,
            engine="h5netcdf",
            decode_times=False,
            cache=False,
        ) as ds:
            if "lat" not in ds.variables or "lon" not in ds.variables:
                raise RuntimeError(f"OSTIA file is missing lat/lon coordinates: {ostia_path}")
            if self.sst_var_name not in ds.variables:
                raise RuntimeError(
                    f"OSTIA file is missing variable '{self.sst_var_name}': {ostia_path}"
                )

            lat = np.asarray(ds["lat"].to_numpy(), dtype=np.float64)
            lon = np.asarray(ds["lon"].to_numpy(), dtype=np.float64)
            sst_var = ds[self.sst_var_name]
            if "time" in sst_var.dims:
                sst_var = sst_var.isel(time=0)
            mask_var = ds["mask"] if "mask" in ds.variables else None
            if mask_var is not None and "time" in mask_var.dims:
                mask_var = mask_var.isel(time=0)
            lat_axis = self._build_patch_axis(float(lat0), float(lat1))
            lon_axis = self._build_patch_axis(float(lon0), float(lon1))
            lat_lo = min(float(lat0), float(lat1))
            lat_hi = max(float(lat0), float(lat1))
            lon_lo = min(float(lon0), float(lon1))
            lon_hi = max(float(lon0), float(lon1))

            lat_idx = np.flatnonzero((lat >= lat_lo) & (lat <= lat_hi))
            lon_idx = np.flatnonzero((lon >= lon_lo) & (lon <= lon_hi))
            if lat_idx.size == 0:
                lat_i0 = self._nearest_index(lat, lat_lo)
                lat_i1 = self._nearest_index(lat, lat_hi)
                lat_idx = np.arange(min(lat_i0, lat_i1), max(lat_i0, lat_i1) + 1)
            if lon_idx.size == 0:
                lon_i0 = self._nearest_index(lon, lon_lo)
                lon_i1 = self._nearest_index(lon, lon_hi)
                lon_idx = np.arange(min(lon_i0, lon_i1), max(lon_i0, lon_i1) + 1)

            lat_start = int(lat_idx.min())
            lat_stop = int(lat_idx.max()) + 1
            lon_start = int(lon_idx.min())
            lon_stop = int(lon_idx.max()) + 1
            native_patch = np.asarray(
                sst_var.isel(
                    lat=slice(lat_start, lat_stop),
                    lon=slice(lon_start, lon_stop),
                ).to_numpy(),
                dtype=np.float32,
            )
            native_patch = self._decode_cf_numeric(native_patch, dict(sst_var.attrs))
            native_mask = None
            mask_attrs = None
            if mask_var is not None:
                native_mask = np.asarray(
                    mask_var.isel(
                        lat=slice(lat_start, lat_stop),
                        lon=slice(lon_start, lon_stop),
                    ).to_numpy(),
                    dtype=np.float32,
                )
                mask_attrs = dict(mask_var.attrs)

            patch = self._interp_native_patch_to_tile(
                native_patch=native_patch,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )
            return self._apply_ostia_land_mask(
                native_patch=patch,
                native_mask=native_mask,
                mask_attrs=mask_attrs,
                lat_native=lat[lat_start:lat_stop],
                lon_native=lon[lon_start:lon_stop],
                lat_axis=lat_axis,
                lon_axis=lon_axis,
            )

    def _build_patch_axis(self, start: float, end: float) -> np.ndarray:
        # Build pixel-center coordinates for one patch axis from bbox bounds.
        # Example: tile_size=128 yields 128 centers evenly spaced across [lo, hi].
        lo = min(float(start), float(end))
        hi = max(float(start), float(end))
        span = hi - lo
        if span <= 0.0:
            raise RuntimeError(f"Invalid patch extent with non-positive span: {start}, {end}")
        step = span / float(self.tile_size)
        return lo + (np.arange(self.tile_size, dtype=np.float64) + 0.5) * step

    @staticmethod
    def _juld_to_yyyymmdd(juld_days: np.ndarray) -> np.ndarray:
        # EN4 stores time as "days since 1950-01-01" (JULD). Convert to integer YYYYMMDD.
        # Invalid/out-of-range values remain 0 so callers can ignore them easily.
        out = np.zeros(juld_days.shape, dtype=np.int32)
        valid = np.isfinite(juld_days) & (juld_days < 90000.0) & (juld_days > -20000.0)
        if not np.any(valid):
            return out
        base = np.datetime64("1950-01-01", "D")
        days = np.floor(juld_days[valid]).astype(np.int64)
        dates = base + days.astype("timedelta64[D]")
        date_str = np.datetime_as_string(dates, unit="D")
        out[valid] = np.char.replace(date_str, "-", "").astype(np.int32)
        return out

    @staticmethod
    def _sanitize_float_array(values: np.ndarray, *, fill_abs_threshold: float = 1.0e6) -> np.ndarray:
        # Convert to float32 and turn fill-like/extreme values into NaN.
        # This keeps downstream validity logic uniform via np.isfinite checks.
        out = np.asarray(values, dtype=np.float32)
        out[(~np.isfinite(out)) | (np.abs(out) > fill_abs_threshold)] = np.nan
        return out

    @staticmethod
    def _replace_en4_fill_with_zero(values: np.ndarray) -> np.ndarray:
        # EN4 uses 99999.0 as a missing-data sentinel inside 400-slot profile rows.
        # Collapse those slots to 0.0 at tensor creation time so downstream code can
        # treat them as empty profile cells instead of extreme temperatures/depths.
        out = np.asarray(values, dtype=np.float32).copy()
        out[np.isclose(out, 99999.0)] = 0.0
        out[~np.isfinite(out)] = 0.0
        return out

    def _open_argo_dataset(self, argo_path: Path) -> xr.Dataset:
        # Try h5netcdf first (fast/common for EN4), then fallback to default backend.
        try:
            return xr.open_dataset(
                argo_path,
                engine="h5netcdf",
                decode_times=False,
                cache=False,
            )
        except Exception:
            # Some EN4 monthly files are not HDF5-backed, so keep a backend fallback.
            return xr.open_dataset(
                argo_path,
                decode_times=False,
                cache=False,
            )

    @staticmethod
    def _normalize_lon_array(lon: np.ndarray) -> np.ndarray:
        # Normalize longitude into [-180, 180) so CSV/Argo lon conventions are comparable.
        return ((lon + 180.0) % 360.0) - 180.0

    def _argo_profiles_to_x_grid(
        self,
        *,
        temperature: torch.Tensor,
        latitude: torch.Tensor,
        longitude: torch.Tensor,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # temperature is expected as (N_PROF, N_LEVELS): one vertical profile per row.
        temp_np = np.asarray(temperature.cpu().numpy(), dtype=np.float32)
        if temp_np.ndim != 2:
            raise RuntimeError(
                f"Expected Argo temperature shape (N_PROF, N_LEVELS), got {tuple(temp_np.shape)}"
            )
        n_prof, n_levels = temp_np.shape
        h = int(self.tile_size)
        w = int(self.tile_size)
        # When there are no vertical levels, return empty channel-first tensors.
        if n_levels <= 0:
            return (
                torch.empty((0, h, w), dtype=torch.float32),
                torch.empty((0, h, w), dtype=torch.bool),
                torch.empty((0, h, w), dtype=torch.int32),
            )

        # Accumulators are flattened per-pixel for easy vectorized channel updates.
        # x_sum[level, pixel] stores summed temperatures; x_count stores observation counts.
        x_sum = np.zeros((n_levels, h * w), dtype=np.float32)
        x_count = np.zeros((n_levels, h * w), dtype=np.int32)

        # Read profile positions (one lat/lon per profile).
        lat_np = np.asarray(latitude.cpu().numpy(), dtype=np.float64).reshape(-1)
        lon_np = np.asarray(longitude.cpu().numpy(), dtype=np.float64).reshape(-1)
        if lat_np.size != n_prof or lon_np.size != n_prof:
            raise RuntimeError(
                "Argo latitude/longitude size mismatch with profile count: "
                f"lat={lat_np.size}, lon={lon_np.size}, n_prof={n_prof}"
            )

        # Use the SAME patch bbox from CSV that is used for OSTIA interpolation.
        # This is the key alignment point between x (Argo) and eo (OSTIA).
        lat_lo = min(float(lat0), float(lat1))
        lat_hi = max(float(lat0), float(lat1))
        lon_lo = min(float(lon0), float(lon1))
        lon_hi = max(float(lon0), float(lon1))

        # Normalize longitudes so profile and bbox live in the same coordinate convention.
        lon_np = self._normalize_lon_array(lon_np)
        lon_lo = float(self._normalize_lon_array(np.asarray([lon_lo]))[0])
        lon_hi = float(self._normalize_lon_array(np.asarray([lon_hi]))[0])

        lat_span = lat_hi - lat_lo
        lon_span = lon_hi - lon_lo
        if lat_span <= 0.0 or lon_span <= 0.0:
            raise RuntimeError(
                f"Invalid patch bounds for gridding Argo profiles: "
                f"lat0={lat0}, lat1={lat1}, lon0={lon0}, lon1={lon1}"
            )

        # Rasterize each profile onto one patch pixel and aggregate means per depth level.
        # This yields sparse x where only observed profile locations carry values.
        for prof_idx in range(n_prof):
            prof_lat = lat_np[prof_idx]
            prof_lon = lon_np[prof_idx]
            # Skip invalid profile coordinates.
            if (not np.isfinite(prof_lat)) or (not np.isfinite(prof_lon)):
                continue
            # Keep only profiles physically inside this patch bbox.
            if prof_lat < lat_lo or prof_lat > lat_hi or prof_lon < lon_lo or prof_lon > lon_hi:
                continue

            # Convert geographic position -> pixel indices on the patch grid.
            y = int(np.floor(((prof_lat - lat_lo) / lat_span) * float(h)))
            x = int(np.floor(((prof_lon - lon_lo) / lon_span) * float(w)))
            y = min(max(y, 0), h - 1)
            x = min(max(x, 0), w - 1)
            pix = y * w + x

            # For this profile, push every valid depth-level value into the same pixel column.
            vals = temp_np[prof_idx]
            # Zero marks unobserved EN4 profile slots after fill-value sanitization.
            valid = np.isfinite(vals) & (vals != 0.0)
            if not np.any(valid):
                continue
            x_sum[valid, pix] += vals[valid]
            x_count[valid, pix] += 1

        # valid_mask marks (level, pixel) entries with at least one observation.
        valid_mask = x_count > 0
        # Fill x_grid with per-(level,pixel) means, leaving unobserved values at 0.
        x_grid = np.zeros_like(x_sum, dtype=np.float32)
        x_grid[valid_mask] = x_sum[valid_mask] / x_count[valid_mask].astype(np.float32)
        # Reshape back from flattened pixels to (levels, H, W).
        x_grid = x_grid.reshape(n_levels, h, w)
        valid_mask = valid_mask.reshape(n_levels, h, w)
        x_count = x_count.reshape(n_levels, h, w)

        return (
            torch.from_numpy(x_grid),
            torch.from_numpy(valid_mask),
            torch.from_numpy(x_count.astype(np.int32, copy=False)),
        )

    @staticmethod
    def _interpolate_sparse_x_level(
        x_level: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Interpolate sparse Argo level values onto the full 2D tile grid."""
        x_level = np.asarray(x_level, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if x_level.ndim != 2:
            raise RuntimeError(f"Expected 2D x_level, got shape {tuple(x_level.shape)}")
        if valid_mask.shape != x_level.shape:
            raise RuntimeError(
                "valid_mask shape must match x_level shape: "
                f"{tuple(valid_mask.shape)} != {tuple(x_level.shape)}"
            )

        filled = x_level.copy()
        observed_count = int(valid_mask.sum())
        if observed_count == 0:
            return filled

        yy, xx = np.indices(x_level.shape, dtype=np.float32)
        points = np.column_stack((yy[valid_mask], xx[valid_mask]))
        values = x_level[valid_mask].astype(np.float32, copy=False)
        target_mask = ~valid_mask
        if not np.any(target_mask):
            return filled

        if observed_count == 1:
            # With one observed point, nearest-neighbor interpolation is constant everywhere.
            filled[target_mask] = values[0]
            return filled

        from scipy.interpolate import griddata

        target_points = np.column_stack((yy[target_mask], xx[target_mask]))
        nearest_fill = griddata(points, values, target_points, method="nearest")
        if observed_count >= 3:
            # Prefer smooth linear interpolation where possible, fallback to nearest outside hull.
            try:
                linear_fill = griddata(points, values, target_points, method="linear")
                fill_values = np.where(np.isfinite(linear_fill), linear_fill, nearest_fill)
            except Exception:
                # Degenerate point layouts (e.g., profiles on one line) can fail linear triangulation.
                fill_values = nearest_fill
        else:
            fill_values = nearest_fill
        filled[target_mask] = np.asarray(fill_values, dtype=np.float32)
        return filled

    @staticmethod
    def _sanitize_filename_component(value: Any) -> str:
        raw = str(value).strip()
        if raw == "":
            return "na"

        sanitized_chars: list[str] = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_"}:
                sanitized_chars.append(ch)
            elif ch == ".":
                sanitized_chars.append("p")
            else:
                sanitized_chars.append("_")
        sanitized = "".join(sanitized_chars).strip("_")
        return sanitized or "na"

    def _export_basename_from_row(self, row: dict[str, Any]) -> str:
        date_token = self._sanitize_filename_component(row.get("date", "na"))
        patch_id = str(row.get("patch_id", "")).strip()
        if patch_id != "":
            patch_token = f"patch_{self._sanitize_filename_component(patch_id)}"
        else:
            patch_token = "_".join(
                [
                    "bbox",
                    self._sanitize_filename_component(f"{float(row['lat0']):.6f}"),
                    self._sanitize_filename_component(f"{float(row['lat1']):.6f}"),
                    self._sanitize_filename_component(f"{float(row['lon0']):.6f}"),
                    self._sanitize_filename_component(f"{float(row['lon1']):.6f}"),
                ]
            )
        return f"{date_token}_{patch_token}"

    @staticmethod
    def _north_up_array(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        # Internal patch tensors are built south->north; flip latitude so GeoTIFF row 0 is north.
        if arr.ndim == 2:
            return np.ascontiguousarray(arr[::-1, :])
        if arr.ndim == 3:
            return np.ascontiguousarray(arr[:, ::-1, :])
        raise RuntimeError(f"Expected 2D or 3D array for GeoTIFF export, got {tuple(arr.shape)}")

    @staticmethod
    def _write_geotiff(
        path: Path,
        values: np.ndarray,
        *,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
        band_descriptions: tuple[str, ...],
        tags: dict[str, Any] | None = None,
    ) -> None:
        import rasterio
        from rasterio.transform import from_bounds

        data = np.asarray(values, dtype=np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise RuntimeError(
                f"Expected GeoTIFF export array with shape (H,W) or (C,H,W), got {tuple(data.shape)}"
            )

        count, height, width = data.shape
        if len(band_descriptions) != count:
            raise RuntimeError(
                "Band description count must match raster band count: "
                f"{len(band_descriptions)} != {count}"
            )

        transform = from_bounds(
            min(float(lon0), float(lon1)),
            min(float(lat0), float(lat1)),
            max(float(lon0), float(lon1)),
            max(float(lat0), float(lat1)),
            int(width),
            int(height),
        )
        north_up = OstiaArgoTileDataset._north_up_array(data)

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=int(width),
            height=int(height),
            count=int(count),
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
            compress="deflate",
            predictor=3,
        ) as dst:
            dst.write(north_up)
            for band_idx, desc in enumerate(band_descriptions, start=1):
                dst.set_band_description(band_idx, str(desc))
            if tags:
                dst.update_tags(**{key: str(value) for key, value in tags.items() if value is not None})

    @staticmethod
    def _append_manifest_record(manifest_path: Path, record: dict[str, Any]) -> None:
        import fcntl
        import os

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(record.keys())
        with manifest_path.open("a+", newline="", encoding="utf-8") as f:
            # Use a file lock so multiple DataLoader workers can append safely on Linux.
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                header = next(csv.reader(f), None)
                f.seek(0, 2)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if header is None:
                    writer.writeheader()
                elif list(header) != fieldnames:
                    raise RuntimeError(
                        f"Manifest header mismatch at {manifest_path}. "
                        "Delete the CSV or keep the exported schema consistent."
                    )
                writer.writerow(record)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _build_export_record(
        self,
        *,
        idx: int,
        row: dict[str, Any],
        info: dict[str, Any],
        output_root: Path,
        manifest_path: Path,
        filename: str,
        argo_tif_path: Path,
        ostia_tif_path: Path,
        argo_depth_indices: tuple[int, ...],
        files_written: bool,
    ) -> dict[str, Any]:
        lat0 = float(row["lat0"])
        lat1 = float(row["lat1"])
        lon0 = float(row["lon0"])
        lon1 = float(row["lon1"])
        centroid_lat = 0.5 * (lat0 + lat1)
        centroid_lon = 0.5 * (lon0 + lon1)
        basename = argo_tif_path.stem

        record: dict[str, Any] = dict(row)
        record.update(
            {
                "export_index": int(idx),
                "sample_basename": basename,
                "filename": filename,
                "centroid_lat": float(centroid_lat),
                "centroid_lon": float(centroid_lon),
                "ostia_tif_path": ostia_tif_path.resolve().as_posix(),
                "argo_tif_path": argo_tif_path.resolve().as_posix(),
                "ostia_filename": ostia_tif_path.name,
                "argo_filename": argo_tif_path.name,
                "ostia_rel_path": ostia_tif_path.relative_to(output_root).as_posix(),
                "argo_rel_path": argo_tif_path.relative_to(output_root).as_posix(),
                "manifest_csv_path": manifest_path.resolve().as_posix(),
                "crs": "EPSG:4326",
                "tile_size": int(self.tile_size),
                "output_units": self.output_units,
                "argo_depth_indices": "|".join(str(depth_idx) for depth_idx in argo_depth_indices),
                "argo_band_count": int(len(argo_depth_indices)),
                "ostia_band_count": 1,
                "files_written": int(files_written),
                "resolved_ostia_source_path": str(self._resolve_index_path(row[self._ostia_path_col])),
                "resolved_argo_source_path": (
                    str(self._resolve_index_path(row[self._argo_path_col]))
                    if self._argo_path_col is not None
                    else ""
                ),
                "temporal_window_days": int(info.get("temporal_window_days", self.days)),
                "temporal_rows_count": int(info.get("temporal_rows_count", 0)),
                "temporal_ostia_days_used": int(info.get("temporal_ostia_days_used", 0)),
                "temporal_argo_days_used": int(info.get("temporal_argo_days_used", 0)),
                "valid_mask_1d_pixels": int(info.get("valid_mask_1d_pixels", 0)),
                "export_skipped_reason": str(info.get("export_skipped_reason", "")),
            }
        )
        return record

    def _write_sample_tiffs(
        self,
        *,
        row: dict[str, Any],
        x: torch.Tensor,
        eo: torch.Tensor,
        valid_mask: torch.Tensor,
        argo_depth_indices: tuple[int, ...],
        argo_tif_path: Path,
        ostia_tif_path: Path,
    ) -> None:
        argo_np = np.asarray(
            x[list(argo_depth_indices)].detach().cpu().numpy(),
            dtype=np.float32,
        )
        valid_np = np.asarray(
            valid_mask[list(argo_depth_indices)].detach().cpu().numpy(),
            dtype=bool,
        )
        # Missing Argo pixels are encoded as NaN so downstream loaders can infer validity directly.
        argo_np = np.where(valid_np, argo_np, np.nan).astype(np.float32, copy=False)
        eo_np = np.asarray(eo[0].detach().cpu().numpy(), dtype=np.float32)

        lat0 = float(row["lat0"])
        lat1 = float(row["lat1"])
        lon0 = float(row["lon0"])
        lon1 = float(row["lon1"])
        centroid_lat = 0.5 * (lat0 + lat1)
        centroid_lon = 0.5 * (lon0 + lon1)
        basename = argo_tif_path.stem

        common_tags = {
            "sample_basename": basename,
            "date": row.get("date"),
            "patch_id": row.get("patch_id"),
            "phase": row.get("phase", row.get("split")),
            "centroid_lat": f"{centroid_lat:.6f}",
            "centroid_lon": f"{centroid_lon:.6f}",
        }
        self._write_geotiff(
            ostia_tif_path,
            eo_np,
            lat0=lat0,
            lat1=lat1,
            lon0=lon0,
            lon1=lon1,
            band_descriptions=("ostia_sst",),
            tags={
                **common_tags,
                "product": "ostia",
                "source_file": str(self._resolve_index_path(row[self._ostia_path_col])),
            },
        )
        self._write_geotiff(
            argo_tif_path,
            argo_np,
            lat0=lat0,
            lat1=lat1,
            lon0=lon0,
            lon1=lon1,
            band_descriptions=tuple(
                f"argo_temperature_layer_{depth_idx}" for depth_idx in argo_depth_indices
            ),
            tags={
                **common_tags,
                "product": "argo",
                "source_file": (
                    str(self._resolve_index_path(row[self._argo_path_col]))
                    if self._argo_path_col is not None
                    else ""
                ),
                "argo_depth_indices": "|".join(str(depth_idx) for depth_idx in argo_depth_indices),
            },
        )

    def save_to_disk(
        self,
        idx: int,
        output_root: str | Path = "/work/data/depth_v3",
        *,
        manifest_path: str | Path | None = None,
        argo_depth_indices: tuple[int, ...] = (0, 1, 2),
        overwrite: bool = False,
        write_manifest: bool = True,
    ) -> dict[str, Any]:
        if not self.return_argo_profiles:
            raise RuntimeError("save_to_disk requires return_argo_profiles=True.")

        row = dict(self._rows[int(idx)])
        argo_depth_indices = tuple(int(depth_idx) for depth_idx in argo_depth_indices)
        if len(argo_depth_indices) == 0:
            raise ValueError("argo_depth_indices must contain at least one layer index.")
        if min(argo_depth_indices) < 0:
            raise ValueError("argo_depth_indices must be non-negative.")

        output_root = Path(output_root)
        argo_dir = output_root / "argo"
        ostia_dir = output_root / "ostia"
        argo_dir.mkdir(parents=True, exist_ok=True)
        ostia_dir.mkdir(parents=True, exist_ok=True)

        basename = self._export_basename_from_row(row)
        filename = f"{basename}.tif"
        argo_tif_path = argo_dir / filename
        ostia_tif_path = ostia_dir / filename
        if manifest_path is None:
            manifest_path = output_root / "ostia_argo_tiff_index.csv"
        else:
            manifest_path = Path(manifest_path)

        if argo_tif_path.exists() != ostia_tif_path.exists() and not overwrite:
            raise FileExistsError(
                "Found only one half of an exported OSTIA/Argo pair. "
                "Delete the partial export or rerun with overwrite=True."
            )

        if argo_tif_path.exists() and ostia_tif_path.exists() and not overwrite:
            # Resumed exports can skip disk-complete samples before expensive sample materialization.
            record = self._build_export_record(
                idx=int(idx),
                row=row,
                info={},
                output_root=output_root,
                manifest_path=manifest_path,
                filename=filename,
                argo_tif_path=argo_tif_path,
                ostia_tif_path=ostia_tif_path,
                argo_depth_indices=argo_depth_indices,
                files_written=False,
            )
            if write_manifest:
                self._append_manifest_record(manifest_path, record)
            return record

        sample = self.__getitem__(int(idx))
        x = sample["x"]
        eo = sample["eo"]
        valid_mask = sample["valid_mask"]
        valid_mask_1d = sample["valid_mask_1d"]
        info = sample.get("info", {})

        if x.ndim != 3 or eo.ndim != 3 or valid_mask.ndim != 3:
            raise RuntimeError(
                "Expected sample tensors with shape (C,H,W): "
                f"x={tuple(x.shape)}, eo={tuple(eo.shape)}, valid_mask={tuple(valid_mask.shape)}"
            )
        if eo.shape[0] != 1:
            raise RuntimeError(f"Expected single-band OSTIA tensor, got shape {tuple(eo.shape)}")
        if valid_mask.shape != x.shape:
            raise RuntimeError(
                "Expected valid_mask shape to match x shape: "
                f"{tuple(valid_mask.shape)} != {tuple(x.shape)}"
            )
        if valid_mask_1d.ndim != 3 or valid_mask_1d.shape != (1, self.tile_size, self.tile_size):
            raise RuntimeError(
                "Expected valid_mask_1d shape to be (1,H,W): "
                f"{tuple(valid_mask_1d.shape)} != {(1, self.tile_size, self.tile_size)}"
            )
        if x.shape[0] <= max(argo_depth_indices):
            raise RuntimeError(
                "Requested Argo layers are not available in this sample: "
                f"requested={argo_depth_indices}, available_channels={x.shape[0]}"
            )

        valid_mask_1d_pixels = int(valid_mask_1d.sum().item())
        if valid_mask_1d_pixels < 4:
            info = dict(info)
            info["valid_mask_1d_pixels"] = valid_mask_1d_pixels
            info["export_skipped_reason"] = "valid_mask_1d_pixels_lt_4"
            return self._build_export_record(
                idx=int(idx),
                row=row,
                info=info,
                output_root=output_root,
                manifest_path=manifest_path,
                filename=filename,
                argo_tif_path=argo_tif_path,
                ostia_tif_path=ostia_tif_path,
                argo_depth_indices=argo_depth_indices,
                files_written=False,
            )

        files_written = False
        if overwrite or (not argo_tif_path.exists() and not ostia_tif_path.exists()):
            self._write_sample_tiffs(
                row=row,
                x=x,
                eo=eo,
                valid_mask=valid_mask,
                argo_depth_indices=argo_depth_indices,
                argo_tif_path=argo_tif_path,
                ostia_tif_path=ostia_tif_path,
            )
            files_written = True

        record = self._build_export_record(
            idx=int(idx),
            row=row,
            info=info,
            output_root=output_root,
            manifest_path=manifest_path,
            filename=filename,
            argo_tif_path=argo_tif_path,
            ostia_tif_path=ostia_tif_path,
            argo_depth_indices=argo_depth_indices,
            files_written=files_written,
        )
        if write_manifest:
            self._append_manifest_record(manifest_path, record)
        return record

    def save_batch_plot_to_temp(
        self,
        idx: int = 0,
        output_path: str | Path = "temp/argo_ostia_batch_0.png",
    ) -> Path:
        """Save a single PNG with x[0], optional y[0], eo, and interpolated x[0]."""
        import matplotlib.pyplot as plt

        sample = self.__getitem__(int(idx))
        x = sample["x"]
        y = sample["y"]
        eo = sample["eo"]
        valid_mask = sample["valid_mask"]
        valid_mask_1d = sample["valid_mask_1d"]
        info = sample.get("info", {})

        if x.ndim != 3 or y.ndim != 3 or eo.ndim != 3 or valid_mask.ndim != 3:
            raise RuntimeError(
                "Expected sample tensors with shape (C,H,W): "
                f"x={tuple(x.shape)}, y={tuple(y.shape)}, eo={tuple(eo.shape)}, "
                f"valid_mask={tuple(valid_mask.shape)}"
            )
        if x.shape[0] == 0:
            raise RuntimeError("Cannot plot Argo channel 0 because x has no channels.")

        x0 = np.asarray(x[0].detach().cpu().numpy(), dtype=np.float32)
        eo0 = np.asarray(eo[0].detach().cpu().numpy(), dtype=np.float32)
        # For interpolation visualization, treat exact zeros as missing and fill all missing pixels.
        zero_invalid_mask = np.isfinite(x0) & (x0 != 0.0)
        x0_interpolated = self._interpolate_sparse_x_level(x0, zero_invalid_mask)
        valid_mask_1d_np = np.asarray(valid_mask_1d[0].detach().cpu().numpy(), dtype=np.float32)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if y.shape[0] > 0:
            y0 = np.asarray(y[0].detach().cpu().numpy(), dtype=np.float32)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
            panels = (
                (x0, "Argo x[0] (sparse)"),
                (y0, "GLORYS y[0]"),
                (eo0, "OSTIA eo[0]"),
                (x0_interpolated, "Argo x[0] (interpolated)"),
            )
            for ax, (img, title) in zip(axes, panels):
                im = ax.imshow(img, cmap="viridis")
                ax.set_title(title)
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
            panels = (
                (x0, "Argo x[0] (sparse)"),
                (eo0, "OSTIA eo[0]"),
                (x0_interpolated, "Argo x[0] (interpolated)"),
            )
            for ax, (img, title) in zip(axes[:3], panels):
                im = ax.imshow(img, cmap="viridis")
                ax.set_title(title)
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            axes[3].set_axis_off()
            axes[3].text(
                0.5,
                0.5,
                "GLORYS y unavailable\nfor this sample",
                ha="center",
                va="center",
                fontsize=12,
            )

        info_date = info.get("date", "N/A")
        fig.suptitle(f"OstiaArgoTileDataset idx={idx} date={info_date}", fontsize=12)
        info_text = textwrap.fill(str(info), width=160)
        fig.text(0.01, 0.01, info_text, ha="left", va="bottom", fontsize=8, family="monospace")
        # Keep collapsed validity available in the figure metadata area.
        valid_fraction = float(valid_mask_1d_np.mean())
        fig.text(0.99, 0.01, f"valid_mask_1d mean={valid_fraction:.4f}", ha="right", va="bottom", fontsize=8)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    def _filter_valid_argo_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method performs ONLY lightweight CSV-based filtering.
        # It does not open Argo NetCDF files.
        if self._argo_path_col is None:
            raise RuntimeError("CSV is missing required column 'argo_file_path'.")
        if "date" not in df.columns:
            raise RuntimeError("CSV is missing required column 'date'.")

        out = df.copy()
        # Build normalized helper columns used for filtering.
        out["_argo_path"] = out[self._argo_path_col].map(self._normalize_index_text)
        out["_target_date"] = pd.to_numeric(out["date"], errors="coerce").fillna(0).astype(np.int64)

        # Require non-empty path and valid date.
        valid_mask = (out["_argo_path"] != "") & (out["_target_date"] > 0)
        # Respect precomputed quality columns if present.
        if "argo_valid" in out.columns:
            argo_valid = pd.to_numeric(out["argo_valid"], errors="coerce").fillna(0).astype(np.int64)
            valid_mask &= argo_valid > 0
        if "argo_profile_count" in out.columns:
            argo_cnt = (
                pd.to_numeric(out["argo_profile_count"], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
            valid_mask &= argo_cnt > 0
        out = out.loc[valid_mask].reset_index(drop=True)
        self._log(f"Rows after basic Argo/date checks: {len(out)} / {len(df)}")
        self._log(
            "Skipping init-time NetCDF validation; Argo profile loading/checking is deferred to __getitem__."
        )
        return out.drop(columns=["_argo_path", "_target_date"], errors="ignore")

    def _empty_argo_payload(self) -> dict[str, torch.Tensor]:
        return {
            "profile_idx": torch.empty((0,), dtype=torch.long),
            "profile_dates": torch.empty((0,), dtype=torch.long),
            "temperature": torch.empty((0, 0), dtype=torch.float32),
            "depth": torch.empty((0, 0), dtype=torch.float32),
            "latitude": torch.empty((0,), dtype=torch.float32),
            "longitude": torch.empty((0,), dtype=torch.float32),
        }

    @staticmethod
    def _empty_argo_payload_with_levels(n_levels: int) -> dict[str, torch.Tensor]:
        return {
            "profile_idx": torch.empty((0,), dtype=torch.long),
            "profile_dates": torch.empty((0,), dtype=torch.long),
            "temperature": torch.empty((0, int(n_levels)), dtype=torch.float32),
            "depth": torch.empty((0, int(n_levels)), dtype=torch.float32),
            "latitude": torch.empty((0,), dtype=torch.float32),
            "longitude": torch.empty((0,), dtype=torch.float32),
        }

    @staticmethod
    def _dimension_size(dim: Any) -> int:
        if dim is None:
            return 0
        if isinstance(dim, (int, np.integer)):
            return int(dim)
        return int(len(dim))

    def _load_argo_profiles_for_date_h5netcdf(
        self,
        *,
        argo_path: Path,
        target_date: int,
    ) -> dict[str, torch.Tensor]:
        with h5netcdf.File(argo_path, "r") as ds:
            if "JULD" not in ds.variables:
                raise RuntimeError(f"Argo file is missing JULD variable: {argo_path}")
            if self.argo_temp_var_name not in ds.variables:
                raise RuntimeError(
                    f"Argo file is missing variable '{self.argo_temp_var_name}': {argo_path}"
                )
            if self.argo_depth_var_name not in ds.variables:
                raise RuntimeError(
                    f"Argo file is missing variable '{self.argo_depth_var_name}': {argo_path}"
            )

            juld = np.asarray(ds.variables["JULD"][:], dtype=np.float64)
            profile_dates = self._juld_to_yyyymmdd(juld)
            profile_idx = np.flatnonzero(profile_dates == int(target_date)).astype(np.int64)
            n_levels = self._dimension_size(ds.dimensions.get("N_LEVELS"))
            if profile_idx.size == 0:
                return self._empty_argo_payload_with_levels(n_levels)

            temp = self._replace_en4_fill_with_zero(self._sanitize_float_array(
                np.asarray(ds.variables[self.argo_temp_var_name][profile_idx, :], dtype=np.float32)
            ))
            depth = self._replace_en4_fill_with_zero(self._sanitize_float_array(
                np.asarray(ds.variables[self.argo_depth_var_name][profile_idx, :], dtype=np.float32)
            ))
            if "LATITUDE" in ds.variables:
                lat = self._sanitize_float_array(
                    np.asarray(ds.variables["LATITUDE"][profile_idx], dtype=np.float32)
                )
            else:
                lat = np.full((profile_idx.size,), np.nan, dtype=np.float32)
            if "LONGITUDE" in ds.variables:
                lon = self._sanitize_float_array(
                    np.asarray(ds.variables["LONGITUDE"][profile_idx], dtype=np.float32)
                )
            else:
                lon = np.full((profile_idx.size,), np.nan, dtype=np.float32)

            return {
                "profile_idx": torch.from_numpy(profile_idx),
                "profile_dates": torch.from_numpy(profile_dates[profile_idx].astype(np.int64)),
                "temperature": torch.from_numpy(temp),
                "depth": torch.from_numpy(depth),
                "latitude": torch.from_numpy(lat.astype(np.float32, copy=False)),
                "longitude": torch.from_numpy(lon.astype(np.float32, copy=False)),
            }

    def _load_argo_profiles_for_date(self, *, argo_path: Path, target_date: int) -> dict[str, torch.Tensor]:
        # Load one monthly EN4 file and keep only profiles for target_date.
        if target_date <= 0:
            return self._empty_argo_payload()

        try:
            return self._load_argo_profiles_for_date_h5netcdf(
                argo_path=argo_path,
                target_date=target_date,
            )
        except Exception:
            # Keep xarray as a fallback reader for monthly files that are not h5netcdf-compatible.
            pass

        with self._open_argo_dataset(argo_path) as ds:
            # Validate required variables for date filtering + profile payload.
            if "JULD" not in ds.variables:
                raise RuntimeError(f"Argo file is missing JULD variable: {argo_path}")
            if self.argo_temp_var_name not in ds.variables:
                raise RuntimeError(
                    f"Argo file is missing variable '{self.argo_temp_var_name}': {argo_path}"
                )
            if self.argo_depth_var_name not in ds.variables:
                raise RuntimeError(
                    f"Argo file is missing variable '{self.argo_depth_var_name}': {argo_path}"
                )

            # Convert monthly JULD vector -> YYYYMMDD and pick matching profile rows.
            juld = np.asarray(ds["JULD"].to_numpy(), dtype=np.float64)
            profile_dates = self._juld_to_yyyymmdd(juld)
            profile_idx = np.flatnonzero(profile_dates == int(target_date)).astype(np.int64)
            if profile_idx.size == 0:
                # Keep channel dimension consistent even when this date has no profiles.
                n_levels = int(ds.sizes.get("N_LEVELS", 0))
                return self._empty_argo_payload_with_levels(n_levels)

            # Pull selected profiles only (N_MATCHED, N_LEVELS) and sanitize fill/extreme values.
            temp = self._replace_en4_fill_with_zero(self._sanitize_float_array(
                np.asarray(
                    ds[self.argo_temp_var_name].isel(N_PROF=profile_idx.tolist()).to_numpy(),
                    dtype=np.float32,
                )
            ))
            depth = self._replace_en4_fill_with_zero(self._sanitize_float_array(
                np.asarray(
                    ds[self.argo_depth_var_name].isel(N_PROF=profile_idx.tolist()).to_numpy(),
                    dtype=np.float32,
                )
            ))

            # Profile positions are optional in some files; if absent, keep NaN placeholders.
            if "LATITUDE" in ds.variables:
                lat = self._sanitize_float_array(
                    np.asarray(ds["LATITUDE"].isel(N_PROF=profile_idx.tolist()).to_numpy())
                )
            else:
                lat = np.full((profile_idx.size,), np.nan, dtype=np.float32)

            if "LONGITUDE" in ds.variables:
                lon = self._sanitize_float_array(
                    np.asarray(ds["LONGITUDE"].isel(N_PROF=profile_idx.tolist()).to_numpy())
                )
            else:
                lon = np.full((profile_idx.size,), np.nan, dtype=np.float32)

            return {
                "profile_idx": torch.from_numpy(profile_idx),
                "profile_dates": torch.from_numpy(profile_dates[profile_idx].astype(np.int64)),
                "temperature": torch.from_numpy(temp),
                "depth": torch.from_numpy(depth),
                "latitude": torch.from_numpy(lat.astype(np.float32, copy=False)),
                "longitude": torch.from_numpy(lon.astype(np.float32, copy=False)),
            }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Read the prefiltered CSV row.
        row = self._rows[int(idx)]
        temporal_rows = self._select_temporal_rows(row)

        center_ostia_path = self._resolve_index_path(row[self._ostia_path_col])
        patch_sum = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        patch_count = np.zeros((self.tile_size, self.tile_size), dtype=np.uint16)
        ostia_days_used = 0
        for day_row in temporal_rows:
            day_ostia_raw = str(day_row.get(self._ostia_path_col, "")).strip()
            if day_ostia_raw == "":
                continue
            day_ostia_path = self._resolve_index_path(day_ostia_raw)
            if not day_ostia_path.exists():
                continue

            day_patch = self._load_ostia_patch(
                ostia_path=day_ostia_path,
                lat0=float(day_row["lat0"]),
                lat1=float(day_row["lat1"]),
                lon0=float(day_row["lon0"]),
                lon1=float(day_row["lon1"]),
            )
            if self.output_units == "celsius":
                day_patch = day_patch - np.float32(273.15)

            day_valid = np.isfinite(day_patch)
            if not np.any(day_valid):
                continue
            patch_sum[day_valid] += day_patch[day_valid]
            patch_count[day_valid] += 1
            ostia_days_used += 1

        if not np.any(patch_count > 0):
            raise FileNotFoundError(
                "No readable OSTIA files found inside temporal window for sample "
                f"idx={idx}, patch_id={row.get('patch_id')}, date={row.get('date')}, "
                f"center_path={center_ostia_path}"
            )

        # Spatial alignment flow for one sample:
        # 1) CSV lat/lon bounds define the physical patch extent for this row.
        # 2) OSTIA is cropped/interpolated onto tile_size x tile_size using those bounds.
        # 3) Argo profile lon/lat is converted into pixel indices using the same bounds.
        # 4) Because both eo and x use the same bbox, they overlap in the same physical area.
        patch = np.full((self.tile_size, self.tile_size), np.nan, dtype=np.float32)
        patch_valid = patch_count > 0
        patch[patch_valid] = patch_sum[patch_valid] / patch_count[patch_valid].astype(np.float32)
        # Any land-only or otherwise invalid surface pixels should stay benign for downstream consumers.
        patch[~np.isfinite(patch)] = np.float32(0.0)
        # eo is single-band (surface SST) in channel-first shape.
        eo = torch.from_numpy(patch).unsqueeze(0)

        y = torch.empty((0, self.tile_size, self.tile_size), dtype=torch.float32)
        glorys_days_used = 0
        center_glorys_path: Path | None = None
        if self._glorys_path_col is not None:
            center_glorys_raw = self._normalize_index_text(row.get(self._glorys_path_col, ""))
            if center_glorys_raw != "":
                center_glorys_path = self._resolve_index_path(center_glorys_raw)

            y_sum: np.ndarray | None = None
            y_count: np.ndarray | None = None
            for day_row in temporal_rows:
                day_glorys_raw = self._normalize_index_text(day_row.get(self._glorys_path_col, ""))
                if day_glorys_raw == "":
                    continue
                day_glorys_path = self._resolve_index_path(day_glorys_raw)
                if not day_glorys_path.exists():
                    continue

                # Use the same CSV bbox as eo/x so the GLORYS target is pixel-aligned spatially.
                day_glorys = self._load_glorys_patch(
                    glorys_path=day_glorys_path,
                    lat0=float(day_row["lat0"]),
                    lat1=float(day_row["lat1"]),
                    lon0=float(day_row["lon0"]),
                    lon1=float(day_row["lon1"]),
                )
                if y_sum is None or y_count is None:
                    y_sum = np.zeros_like(day_glorys, dtype=np.float32)
                    y_count = np.zeros_like(day_glorys, dtype=np.uint16)
                elif day_glorys.shape != y_sum.shape:
                    raise RuntimeError(
                        "Temporal GLORYS aggregation encountered inconsistent shapes: "
                        f"expected {tuple(y_sum.shape)}, got {tuple(day_glorys.shape)}"
                    )

                day_valid = np.isfinite(day_glorys)
                if not np.any(day_valid):
                    continue
                y_sum[day_valid] += day_glorys[day_valid]
                y_count[day_valid] += 1
                glorys_days_used += 1

            if y_sum is not None and y_count is not None and np.any(y_count > 0):
                y_np = np.full_like(y_sum, np.nan, dtype=np.float32)
                y_valid = y_count > 0
                y_np[y_valid] = y_sum[y_valid] / y_count[y_valid].astype(np.float32)
                y = torch.from_numpy(y_np)

        # Default x/valid_mask are empty and will be filled if Argo data is available.
        x = torch.empty((0, self.tile_size, self.tile_size), dtype=torch.float32)
        valid_mask = torch.empty(
            (0, self.tile_size, self.tile_size),
            dtype=torch.bool,
        )
        argo_days_used = 0
        if self.return_argo_profiles:
            x_sum: torch.Tensor | None = None
            x_count: torch.Tensor | None = None
            for day_row in temporal_rows:
                # Parse Argo file/date pointers from each temporal row.
                argo_path_raw = (
                    day_row.get(self._argo_path_col, "") if self._argo_path_col is not None else ""
                )
                argo_date = self._parse_date_int(day_row.get("date", 0))
                if argo_date <= 0 or str(argo_path_raw).strip() == "":
                    continue

                argo_path = self._resolve_index_path(argo_path_raw)
                if not argo_path.exists():
                    continue

                # Load date-matched profile table from EN4 monthly file.
                argo_payload = self._load_argo_profiles_for_date(
                    argo_path=argo_path,
                    target_date=argo_date,
                )
                # Rasterize profile table into patch tensor x and corresponding valid_mask.
                # IMPORTANT: we pass the same CSV bounds used for eo, so x/eo overlap physically.
                # The actual point->pixel conversion happens inside _argo_profiles_to_x_grid().
                x_day, _valid_mask_day, count_day = self._argo_profiles_to_x_grid(
                    temperature=argo_payload["temperature"],
                    latitude=argo_payload["latitude"],
                    longitude=argo_payload["longitude"],
                    lat0=float(day_row["lat0"]),
                    lat1=float(day_row["lat1"]),
                    lon0=float(day_row["lon0"]),
                    lon1=float(day_row["lon1"]),
                )

                if x_sum is None or x_count is None:
                    x_sum = torch.zeros_like(x_day)
                    x_count = torch.zeros_like(x_day, dtype=torch.int32)
                elif x_day.shape != x_sum.shape:
                    raise RuntimeError(
                        "Temporal Argo aggregation encountered inconsistent shapes: "
                        f"expected {tuple(x_sum.shape)}, got {tuple(x_day.shape)}"
                    )

                if count_day.shape != x_sum.shape:
                    raise RuntimeError(
                        "Temporal Argo aggregation encountered inconsistent count shape: "
                        f"expected {tuple(x_sum.shape)}, got {tuple(count_day.shape)}"
                    )

                # Convert per-day means back to sums so temporal aggregation is the true average
                # over all overlapping observations (not an unweighted average of day-means).
                x_sum += x_day * count_day.to(dtype=torch.float32)
                x_count += count_day
                argo_days_used += 1

            if x_sum is not None and x_count is not None:
                valid_mask = x_count > 0
                x = torch.zeros_like(x_sum, dtype=torch.float32)
                x[valid_mask] = x_sum[valid_mask] / x_count[valid_mask].to(dtype=torch.float32)
        # Collapse across all depth bands so each pixel is valid if any band has an observation.
        if valid_mask.shape[0] > 0:
            valid_mask_1d = valid_mask.any(dim=0, keepdim=True)
        else:
            valid_mask_1d = torch.zeros(
                (1, self.tile_size, self.tile_size),
                dtype=torch.bool,
            )

        surface_valid_pixels = 0
        surface_ostia_argo_mae_deg: float | None = None
        if x.shape[0] > 0 and valid_mask.shape[0] > 0:
            # Compute surface-only error on observed Argo pixels:
            # Argo level 0 vs. OSTIA surface eo[0].
            surface_valid = valid_mask[0]
            surface_valid_pixels = int(surface_valid.sum().item())
            if surface_valid_pixels > 0:
                surface_ostia_argo_mae_deg = float(
                    torch.abs(x[0][surface_valid] - eo[0][surface_valid]).mean().item()
                )

        # Final sample contract: Argo raster (x), OSTIA patch (eo), Argo valid mask, metadata.
        sample: dict[str, Any] = {
            "x": x,
            "y": y,
            "eo": eo,
            "valid_mask": valid_mask,
            "valid_mask_1d": valid_mask_1d,
            "info": {
                "index": int(idx),
                "patch_id": row.get("patch_id"),
                "date": row.get("date"),
                "phase": row.get("phase", row.get("split")),
                "ostia_file_path": str(center_ostia_path),
                "glorys_file_path": str(center_glorys_path) if center_glorys_path is not None else "",
                "argo_file_path": row.get("argo_file_path"),
                "argo_profile_count_world_daily": row.get("argo_profile_count"),
                "temporal_window_days": int(self.days),
                "temporal_rows_count": int(len(temporal_rows)),
                "temporal_ostia_days_used": int(ostia_days_used),
                "temporal_glorys_days_used": int(glorys_days_used),
                "temporal_argo_days_used": int(argo_days_used),
                "surface_ostia_argo_valid_pixels": int(surface_valid_pixels),
                "surface_ostia_argo_mae_deg": surface_ostia_argo_mae_deg,
                "valid_mask_1d_pixels": int(valid_mask_1d.sum().item()),
            },
        }
        return sample


if __name__ == "__main__":
    if False:
        # Example usage:
        dataset = OstiaArgoTileDataset("/work/data/depth_v2/ostia_patch_index_daily_0p1.csv", split="train",return_argo_profiles=True,days=7)
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"x shape: {tuple(sample['x'].shape)}")
        print(f"eo shape: {tuple(sample['eo'].shape)}")
        print(f"valid_mask shape: {tuple(sample['valid_mask'].shape)}")
        
        # save as PNG for a few random samples to visually inspect Argo/OSTIA alignment and interpolation behavior
        import random
        for i in range(10):
            rand_i = random.randint(0, len(dataset) - 1)
            dataset.save_batch_plot_to_temp(idx=rand_i, output_path=f"temp/argo_ostia/argo_ostia_sample_{i}.png")
            
        
        # calculate valid/invalid counts for Argo samples in the dataset
        c_inv = 0
        c_val = 0
        c_avg_num = []
        for i in range(1000):
            rand_i = random.randint(0, len(dataset) - 1)
            batch = dataset[rand_i]
            max_v = batch['x'].max().item()
            valid_pixels = batch["valid_mask_1d"].sum().item()
            if max_v > 0:
                c_val += 1
                c_avg_num.append(valid_pixels)
            else:
                c_inv += 1
            print(f"Percentage of invalid samples: {c_inv / (c_val + c_inv) * 100:.2f}%, Average valid pixels: {np.mean(c_avg_num):.2f}. {i}/1000", end="\r")

if __name__ == "__main__":
    if False:
        dataset = OstiaArgoTileDataset(
            "/work/data/depth_v2/ostia_patch_index_daily_0p1.csv",
            split="train",
            days=14,
            return_argo_profiles=True,
        )
        from tqdm import tqdm
        for i in tqdm(range(len(dataset))):
            record = dataset.save_to_disk(i)  # writes to /work/data/depth_v3 by default
            
if __name__ == "__main__":
    if True:
        # Get Dataset and get sample
        dataset = OstiaArgoTileDataset(
            "/work/data/depth_v2/ostia_patch_index_daily.csv",
            split="train",
            days=14,
            return_argo_profiles=True,
        )
        import random
        rand_i = random.randint(0, len(dataset) - 1)
        sample = dataset[rand_i]
        print(f"Fetched Sample: {rand_i}")
        dataset.save_batch_plot_to_temp(idx=rand_i, output_path=f"temp/argo_ostia/argo_ostia_modalities_sample_{rand_i}.png")
        
        # Save 3D plot for the sample
        from utils.visualizations import save_argo_profile_3d_plot
        save_argo_profile_3d_plot(
            profile_tensor=sample["x"],
            output_path=f"temp/argo_ostia/argo_ostia_sample_{rand_i}.png"
        )
        
        # Save 3D Flyaround GIF for the same sample
        from utils.visualizations import save_argo_profile_3d_flyaround_gif
        save_argo_profile_3d_flyaround_gif(
            profile_tensor=sample["x"],
            output_path=f"temp/argo_ostia/argo_ostia_sample_{rand_i}.gif",
            dpi=50,
            num_frames=36,
            frame_duration_ms=100,
        )
        
