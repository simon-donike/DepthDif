from __future__ import annotations

from pathlib import Path
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

    For now, this dataset only returns the OSTIA surface condition patch.
    Argo fields from the CSV are passed through as metadata for future use.
    """

    REQUIRED_GEO_COLUMNS = ("lat0", "lat1", "lon0", "lon1")
    OSTIA_PATH_CANDIDATE_COLUMNS = ("ostia_file_path", "matched_ostia_file_path")
    SPLIT_CANDIDATE_COLUMNS = ("phase", "split")

    def __init__(
        self,
        csv_path: str | Path,
        *,
        split: str = "all",
        tile_size: int = 128,
        sst_var_name: str = "analysed_sst",
        output_units: str = "celsius",
        cache_last_ostia: bool = True,
        return_info: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.tile_size = int(tile_size)
        self.sst_var_name = str(sst_var_name)
        self.output_units = str(output_units).strip().lower()
        self.cache_last_ostia = bool(cache_last_ostia)
        self.return_info = bool(return_info)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if self.tile_size < 2:
            raise ValueError("tile_size must be >= 2.")
        if self.output_units not in {"kelvin", "celsius"}:
            raise ValueError("output_units must be one of {'kelvin', 'celsius'}.")

        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise RuntimeError(f"CSV has no rows: {self.csv_path}")

        missing_geo = [c for c in self.REQUIRED_GEO_COLUMNS if c not in df.columns]
        if missing_geo:
            raise RuntimeError(f"CSV is missing required geo columns: {missing_geo}")

        self._ostia_path_col = self._first_present_column(
            df.columns.tolist(),
            self.OSTIA_PATH_CANDIDATE_COLUMNS,
        )
        if self._ostia_path_col is None:
            raise RuntimeError(
                "CSV is missing OSTIA path column. "
                f"Expected one of {list(self.OSTIA_PATH_CANDIDATE_COLUMNS)}."
            )

        split_col = self._first_present_column(
            df.columns.tolist(),
            self.SPLIT_CANDIDATE_COLUMNS,
        )
        if self.split in {"train", "val"}:
            if split_col is None:
                raise RuntimeError(
                    "split='train'/'val' requested but CSV has no split column. "
                    f"Expected one of {list(self.SPLIT_CANDIDATE_COLUMNS)}."
                )
            df = df[df[split_col].astype(str).str.lower() == self.split].reset_index(drop=True)
        elif self.split != "all":
            raise ValueError("split must be one of: 'all', 'train', 'val'")

        if len(df) == 0:
            raise RuntimeError("Dataset is empty after split filtering.")

        self._rows = df.to_dict(orient="records")

        self._cached_ostia_path: Path | None = None
        self._cached_lat: np.ndarray | None = None
        self._cached_lon: np.ndarray | None = None
        self._cached_sst: np.ndarray | None = None

    @staticmethod
    def _first_present_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
        col_set = set(columns)
        for name in candidates:
            if name in col_set:
                return name
        return None

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_index_path(self, value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.csv_dir / path

    def _load_ostia_arrays(self, ostia_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.cache_last_ostia
            and self._cached_ostia_path is not None
            and ostia_path == self._cached_ostia_path
        ):
            if self._cached_lat is None or self._cached_lon is None or self._cached_sst is None:
                raise RuntimeError("Internal cache state is invalid.")
            return self._cached_lat, self._cached_lon, self._cached_sst

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
            # Daily OSTIA files typically have a singleton time dimension.
            if "time" in sst_var.dims:
                sst_var = sst_var.isel(time=0)
            sst = np.asarray(sst_var.to_numpy(), dtype=np.float32)

        if sst.ndim != 2:
            raise RuntimeError(f"Expected 2D SST field, got shape {tuple(sst.shape)} at {ostia_path}")

        # Ensure increasing axes for interpolation routines.
        if lat.size > 1 and lat[0] > lat[-1]:
            lat = lat[::-1].copy()
            sst = sst[::-1, :]
        if lon.size > 1 and lon[0] > lon[-1]:
            lon = lon[::-1].copy()
            sst = sst[:, ::-1]

        # Replace obvious fill values with NaN so invalid regions propagate cleanly.
        sst[(~np.isfinite(sst)) | (sst > 1.0e6)] = np.nan

        if self.cache_last_ostia:
            self._cached_ostia_path = ostia_path
            self._cached_lat = lat
            self._cached_lon = lon
            self._cached_sst = sst
        return lat, lon, sst

    def _build_patch_axis(self, start: float, end: float) -> np.ndarray:
        lo = min(float(start), float(end))
        hi = max(float(start), float(end))
        span = hi - lo
        if span <= 0.0:
            raise RuntimeError(f"Invalid patch extent with non-positive span: {start}, {end}")
        step = span / float(self.tile_size)
        return lo + (np.arange(self.tile_size, dtype=np.float64) + 0.5) * step

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        ostia_path = self._resolve_index_path(row[self._ostia_path_col])
        if not ostia_path.exists():
            raise FileNotFoundError(f"OSTIA file not found: {ostia_path}")

        lat, lon, sst = self._load_ostia_arrays(ostia_path)
        interp = RegularGridInterpolator(
            (lat, lon),
            sst,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        lat_axis = self._build_patch_axis(float(row["lat0"]), float(row["lat1"]))
        lon_axis = self._build_patch_axis(float(row["lon0"]), float(row["lon1"]))
        mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
        query_points = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

        patch = interp(query_points).reshape(self.tile_size, self.tile_size).astype(
            np.float32, copy=False
        )
        if self.output_units == "celsius":
            patch = patch - np.float32(273.15)

        condition = torch.from_numpy(patch).unsqueeze(0)
        sample: dict[str, Any] = {
            "condition": condition,
            "eo": condition,
        }
        if self.return_info:
            sample["info"] = {
                "index": int(idx),
                "patch_id": row.get("patch_id"),
                "date": row.get("date"),
                "phase": row.get("phase", row.get("split")),
                "ostia_file_path": str(ostia_path),
                "argo_file_path": row.get("argo_file_path"),
                "argo_profile_count": row.get("argo_profile_count"),
            }
        return sample


if __name__ == "__main__":
    # Example usage:
    dataset = OstiaArgoTileDataset("/data1/datasets/depth_v2/ostia_patch_index_daily.csv", split="train")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Condition shape: {sample['condition'].shape}")
    if "info" in sample:
        print(f"Sample info: {sample['info']}")