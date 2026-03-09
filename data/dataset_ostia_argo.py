from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset


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

    def __init__(
        self,
        csv_path: str | Path,
        *,
        split: str = "all",
        tile_size: int = 128,
        sst_var_name: str = "analysed_sst",
        output_units: str = "celsius",
        return_argo_profiles: bool = True,
        argo_temp_var_name: str = "TEMP",
        argo_depth_var_name: str = "DEPH_CORRECTED",
        return_info: bool = True,
        verbose_init: bool = True,
    ) -> None:
        # Store constructor args as normalized runtime config for all downstream methods.
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.tile_size = int(tile_size)
        self.sst_var_name = str(sst_var_name)
        self.output_units = str(output_units).strip().lower()
        self.return_argo_profiles = bool(return_argo_profiles)
        self.argo_temp_var_name = str(argo_temp_var_name)
        self.argo_depth_var_name = str(argo_depth_var_name)
        self.return_info = bool(return_info)
        self.verbose_init = bool(verbose_init)

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

    def _resolve_index_path(self, value: Any) -> Path:
        # Support absolute paths and CSV-relative paths in the same index.
        path = Path(str(value))
        return path if path.is_absolute() else self.csv_dir / path

    def _load_ostia_arrays(self, ostia_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Open one OSTIA file for the current row only (no dataset-level caching).
        with xr.open_dataset(
            ostia_path,
            engine="h5netcdf",
            decode_times=False,
            cache=False,
        ) as ds:
            # Fail early when key geospatial axes or SST variable are absent.
            if "lat" not in ds.variables or "lon" not in ds.variables:
                raise RuntimeError(f"OSTIA file is missing lat/lon coordinates: {ostia_path}")
            if self.sst_var_name not in ds.variables:
                raise RuntimeError(
                    f"OSTIA file is missing variable '{self.sst_var_name}': {ostia_path}"
                )

            # Pull coordinate axes and the SST field into numpy arrays for interpolation.
            lat = np.asarray(ds["lat"].to_numpy(), dtype=np.float64)
            lon = np.asarray(ds["lon"].to_numpy(), dtype=np.float64)
            sst_var = ds[self.sst_var_name]
            # Daily OSTIA files typically have a singleton time dimension.
            if "time" in sst_var.dims:
                sst_var = sst_var.isel(time=0)
            sst = np.asarray(sst_var.to_numpy(), dtype=np.float32)

        # Interpolator expects a 2D data plane over (lat, lon).
        if sst.ndim != 2:
            raise RuntimeError(f"Expected 2D SST field, got shape {tuple(sst.shape)} at {ostia_path}")

        # Ensure axes are strictly increasing because RegularGridInterpolator assumes sorted axes.
        if lat.size > 1 and lat[0] > lat[-1]:
            lat = lat[::-1].copy()
            sst = sst[::-1, :]
        if lon.size > 1 and lon[0] > lon[-1]:
            lon = lon[::-1].copy()
            sst = sst[:, ::-1]

        # Replace obvious fill values with NaN so invalid regions propagate cleanly.
        sst[(~np.isfinite(sst)) | (sst > 1.0e6)] = np.nan

        return lat, lon, sst

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            )

        # Accumulators are flattened per-pixel for easy vectorized channel updates.
        # x_sum[level, pixel] stores summed temperatures; x_count stores observation counts.
        x_sum = np.zeros((n_levels, h * w), dtype=np.float32)
        x_count = np.zeros((n_levels, h * w), dtype=np.uint16)

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
            valid = np.isfinite(vals)
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

        return torch.from_numpy(x_grid), torch.from_numpy(valid_mask)

    def _filter_valid_argo_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method performs ONLY lightweight CSV-based filtering.
        # It does not open Argo NetCDF files.
        if self._argo_path_col is None:
            raise RuntimeError("CSV is missing required column 'argo_file_path'.")
        if "date" not in df.columns:
            raise RuntimeError("CSV is missing required column 'date'.")

        out = df.copy()
        # Build normalized helper columns used for filtering.
        out["_argo_path"] = out[self._argo_path_col].astype(str).str.strip()
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

    def _load_argo_profiles_for_date(self, *, argo_path: Path, target_date: int) -> dict[str, torch.Tensor]:
        # Load one monthly EN4 file and keep only profiles for target_date.
        if target_date <= 0:
            return self._empty_argo_payload()

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
                return {
                    "profile_idx": torch.empty((0,), dtype=torch.long),
                    "profile_dates": torch.empty((0,), dtype=torch.long),
                    "temperature": torch.empty((0, n_levels), dtype=torch.float32),
                    "depth": torch.empty((0, n_levels), dtype=torch.float32),
                    "latitude": torch.empty((0,), dtype=torch.float32),
                    "longitude": torch.empty((0,), dtype=torch.float32),
                }

            # Pull selected profiles only (N_MATCHED, N_LEVELS) and sanitize fill/extreme values.
            temp = self._sanitize_float_array(
                np.asarray(
                    ds[self.argo_temp_var_name].isel(N_PROF=profile_idx.tolist()).to_numpy(),
                    dtype=np.float32,
                )
            )
            depth = self._sanitize_float_array(
                np.asarray(
                    ds[self.argo_depth_var_name].isel(N_PROF=profile_idx.tolist()).to_numpy(),
                    dtype=np.float32,
                )
            )

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

        # Resolve and validate OSTIA path for this row.
        ostia_path = self._resolve_index_path(row[self._ostia_path_col])
        if not ostia_path.exists():
            raise FileNotFoundError(f"OSTIA file not found: {ostia_path}")

        # Load OSTIA field and build interpolator over physical (lat, lon) grid.
        lat, lon, sst = self._load_ostia_arrays(ostia_path)
        interp = RegularGridInterpolator(
            (lat, lon),
            sst,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Build the patch pixel-center axes from CSV bounds.
        # These axes define the physical footprint of eo for this sample.
        lat_axis = self._build_patch_axis(float(row["lat0"]), float(row["lat1"]))
        lon_axis = self._build_patch_axis(float(row["lon0"]), float(row["lon1"]))
        mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
        query_points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

        # Interpolate OSTIA SST onto the patch grid and convert units if requested.
        patch = interp(query_points).reshape(self.tile_size, self.tile_size).astype(
            np.float32, copy=False
        )
        if self.output_units == "celsius":
            patch = patch - np.float32(273.15)

        # eo is single-band (surface SST) in channel-first shape.
        eo = torch.from_numpy(patch).unsqueeze(0)
        # Default x/valid_mask are empty and will be filled if Argo data is available.
        x = torch.empty((0, self.tile_size, self.tile_size), dtype=torch.float32)
        valid_mask = torch.empty(
            (0, self.tile_size, self.tile_size),
            dtype=torch.bool,
        )
        if self.return_argo_profiles:
            # Parse Argo file/date pointers from CSV row.
            argo_path_raw = row.get(self._argo_path_col, "") if self._argo_path_col is not None else ""
            argo_date = int(row.get("date", 0)) if str(row.get("date", "")).strip() else 0
            if str(argo_path_raw).strip():
                argo_path = self._resolve_index_path(argo_path_raw)
                if argo_path.exists():
                    # Load date-matched profile table from EN4 monthly file.
                    argo_payload = self._load_argo_profiles_for_date(
                        argo_path=argo_path,
                        target_date=argo_date,
                    )
                    # Rasterize profile table into patch tensor x and corresponding valid_mask.
                    # IMPORTANT: we pass the same CSV bounds used for eo, so x/eo overlap physically.
                    x, valid_mask = self._argo_profiles_to_x_grid(
                        temperature=argo_payload["temperature"],
                        latitude=argo_payload["latitude"],
                        longitude=argo_payload["longitude"],
                        lat0=float(row["lat0"]),
                        lat1=float(row["lat1"]),
                        lon0=float(row["lon0"]),
                        lon1=float(row["lon1"]),
                    )
        # Final sample contract: Argo raster (x), OSTIA patch (eo), Argo valid mask, metadata.
        sample: dict[str, Any] = {
            "x": x,
            "eo": eo,
            "valid_mask": valid_mask,
            "info": {
                "index": int(idx),
                "patch_id": row.get("patch_id"),
                "date": row.get("date"),
                "phase": row.get("phase", row.get("split")),
                "ostia_file_path": str(ostia_path),
                "argo_file_path": row.get("argo_file_path"),
                "argo_profile_count": row.get("argo_profile_count"),
            },
        }
        return sample


if __name__ == "__main__":
    # Example usage:
    dataset = OstiaArgoTileDataset("/data1/datasets/depth_v2/ostia_patch_index_daily.csv", split="train")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"x shape: {tuple(sample['x'].shape)}")
    print(f"eo shape: {tuple(sample['eo'].shape)}")
    print(f"valid_mask shape: {tuple(sample['valid_mask'].shape)}")
