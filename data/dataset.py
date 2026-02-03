from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm


class DepthTileDataset(Dataset):
    def __init__(
        self,
        config_path: str = "configs/data_config.yaml",
        max_nodata_fraction: float = 0.2,
        rebuild_index: bool = False,
    ) -> None:
        cfg = self._load_config(config_path)
        ds_cfg = cfg["dataset"]

        self.root_dir = Path(ds_cfg["root_dir"])
        self.patch_size = int(ds_cfg["patch_size"])
        self.grid_size_deg = float(ds_cfg["grid_size_deg"])
        self.interpolation = str(ds_cfg.get("interpolation", "linear"))
        self.nan_fill_value = float(ds_cfg.get("nan_fill_value", 0.0))
        self.mask_fraction = float(ds_cfg.get("mask_fraction", 0.0))
        self.max_nodata_fraction = float(max_nodata_fraction)
        self.random_seed = int(ds_cfg.get("random_seed", 7))

        index_path_cfg = Path(ds_cfg["index_path"])
        if index_path_cfg.is_absolute():
            self.index_path = index_path_cfg
        else:
            self.index_path = Path.cwd() / index_path_cfg
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if rebuild_index or not self.index_path.exists():
            self._build_shared_parquet()

        self.tiles = pd.read_parquet(self.index_path)
        self.tiles = self.tiles[
            self.tiles["nodata_fraction"] <= self.max_nodata_fraction
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Any:
        row = self.tiles.iloc[idx]
        arr = np.frombuffer(
            row["tile_bytes"], dtype=np.dtype(row["tile_dtype"])
        ).reshape((int(row["height"]), int(row["width"])))
        tensor = torch.from_numpy(arr.copy()).unsqueeze(0).float()

        if self.mask_fraction <= 0.0:
            return tensor

        rng = np.random.default_rng(self.random_seed + idx)
        keep_mask = rng.random((self.patch_size, self.patch_size)) > self.mask_fraction
        masked = tensor.clone()
        masked[~torch.from_numpy(keep_mask).unsqueeze(0)] = self.nan_fill_value
        return masked, tensor

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_shared_parquet(self) -> None:
        nc_files = sorted(self.root_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {self.root_dir}")

        records: list[dict[str, Any]] = []
        tile_id = 0

        for _, nc_path in enumerate(nc_files):
            with xr.open_dataset(nc_path) as ds:
                data_var_name, lat_name, lon_name = self._pick_var_and_coords(ds)
                da = ds[data_var_name]
                da = self._reduce_to_2d(da, lat_name, lon_name)

                lats = da[lat_name].values
                lons = da[lon_name].values

                lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
                lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))

                lat_edges = np.arange(
                    np.floor(lat_min), np.ceil(lat_max), self.grid_size_deg
                )
                lon_edges = np.arange(
                    np.floor(lon_min), np.ceil(lon_max), self.grid_size_deg
                )

                lat_desc = bool(lats[0] > lats[-1])
                lon_desc = bool(lons[0] > lons[-1])

                fill_values = []
                if "_FillValue" in da.attrs:
                    fill_values.append(float(da.attrs["_FillValue"]))
                if "missing_value" in da.attrs:
                    fill_values.append(float(da.attrs["missing_value"]))

                for lat0 in tqdm(lat_edges,desc=f"Processing .nc file {_+1}/{len(nc_files)}: {nc_path.name}"):
                    lat1 = lat0 + self.grid_size_deg
                    lat_slice = slice(lat1, lat0) if lat_desc else slice(lat0, lat1)

                    for lon0 in lon_edges:
                        lon1 = lon0 + self.grid_size_deg
                        lon_slice = slice(lon1, lon0) if lon_desc else slice(lon0, lon1)

                        tile = da.sel({lat_name: lat_slice, lon_name: lon_slice})
                        if tile.sizes.get(lat_name, 0) < 2 or tile.sizes.get(lon_name, 0) < 2:
                            continue

                        lat_target = np.linspace(lat0, lat1, self.patch_size)
                        lon_target = np.linspace(lon0, lon1, self.patch_size)
                        tile = tile.interp(
                            {lat_name: lat_target, lon_name: lon_target},
                            method=self.interpolation,
                        )

                        arr = tile.values.astype(np.float32, copy=False)
                        if arr.ndim != 2:
                            continue

                        nodata_mask = np.isnan(arr)
                        for fv in fill_values:
                            nodata_mask |= np.isclose(arr, fv)
                        nodata_fraction = float(nodata_mask.mean())
                        if nodata_fraction > self.max_nodata_fraction:
                            continue

                        arr = np.where(nodata_mask, self.nan_fill_value, arr).astype(
                            np.float32, copy=False
                        )

                        records.append(
                            {
                                "tile_id": tile_id,
                                "source_file": nc_path.name,
                                "variable": data_var_name,
                                "lat_min": float(lat0),
                                "lat_max": float(lat1),
                                "lon_min": float(lon0),
                                "lon_max": float(lon1),
                                "height": int(arr.shape[0]),
                                "width": int(arr.shape[1]),
                                "tile_dtype": str(arr.dtype),
                                "nodata_fraction": nodata_fraction,
                                "tile_bytes": arr.tobytes(),
                            }
                        )
                        tile_id += 1

        if not records:
            raise RuntimeError("No valid tiles were generated from the provided .nc files.")

        pd.DataFrame.from_records(records).to_parquet(self.index_path, index=False)

    @staticmethod
    def _reduce_to_2d(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
        extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
        if extra_dims:
            da = da.isel({d: 0 for d in extra_dims})
        da = da.squeeze(drop=True)
        if da.dims != (lat_name, lon_name):
            da = da.transpose(lat_name, lon_name)
        return da

    @staticmethod
    def _pick_var_and_coords(ds: xr.Dataset) -> tuple[str, str, str]:
        lat_name = DepthTileDataset._pick_coord_name(ds, ("lat", "latitude"))
        lon_name = DepthTileDataset._pick_coord_name(ds, ("lon", "longitude"))

        for name in ds.data_vars:
            dims = ds[name].dims
            if lat_name in dims and lon_name in dims:
                return name, lat_name, lon_name
        raise RuntimeError("Could not find a data variable containing both lat/lon dimensions.")

    @staticmethod
    def _pick_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
        lower_map = {name.lower(): name for name in ds.coords}
        for key in candidates:
            if key in lower_map:
                return lower_map[key]
        for coord in ds.coords:
            c = coord.lower()
            if any(k in c for k in candidates):
                return coord
        raise RuntimeError(f"Could not infer coordinate name from candidates: {candidates}")


if __name__ == "__main__":
    dataset = DepthTileDataset(
        config_path="configs/data_config.yaml",
        max_nodata_fraction=0.2,
        rebuild_index=True,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape if isinstance(sample, torch.Tensor) else [s.shape for s in sample]}")