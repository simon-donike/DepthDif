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
        max_nodata_fraction: float | None = None,
        enforce_tile_validity: bool | None = None,
        rebuild_index: bool = False,
    ) -> None:
        # Load dataset behavior from YAML once so every setting is centralized in config.
        cfg = self._load_config(config_path)
        ds_cfg = cfg["dataset"]

        # Core tiling settings (where data lives and how big each returned sample should be).
        self.root_dir = Path(ds_cfg["root_dir"])
        self.patch_size = int(ds_cfg["patch_size"])
        self.grid_size_deg = float(ds_cfg["grid_size_deg"])
        self.interpolation = str(ds_cfg.get("interpolation", "linear"))
        self.nan_fill_value = float(ds_cfg.get("nan_fill_value", 0.0))
        self.mask_fraction = float(ds_cfg.get("mask_fraction", 0.0))
        self.random_seed = int(ds_cfg.get("random_seed", 7))

        # Prefer config-driven validity behavior, but still allow runtime override via constructor.
        config_enforce_validity = bool(ds_cfg.get("enforce_tile_validity", True))
        config_max_nodata = float(ds_cfg.get("max_nodata_fraction", 0.2))
        self.enforce_tile_validity = (
            config_enforce_validity
            if enforce_tile_validity is None
            else bool(enforce_tile_validity)
        )
        self.max_nodata_fraction = (
            config_max_nodata
            if max_nodata_fraction is None
            else float(max_nodata_fraction)
        )

        # Resolve the shared parquet path. Relative paths are anchored at repository root.
        index_path_cfg = Path(ds_cfg["index_path"])
        if index_path_cfg.is_absolute():
            self.index_path = index_path_cfg
        else:
            self.index_path = Path.cwd() / index_path_cfg
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the shared index on first run (or when explicitly requested).
        if rebuild_index or not self.index_path.exists():
            self._build_shared_parquet()

        # Load every tile entry sequentially from parquet metadata.
        self.tiles = pd.read_parquet(self.index_path)

        # Rebuild legacy indices that do not contain global temperature normalization stats.
        if not self._index_has_temperature_stats(self.tiles):
            self._build_shared_parquet()
            self.tiles = pd.read_parquet(self.index_path)

        # If enforcement is enabled but index was created in fast mode (no nodata stats), rebuild now.
        if self.enforce_tile_validity and not self._index_has_validity_stats(self.tiles):
            self._build_shared_parquet()
            self.tiles = pd.read_parquet(self.index_path)

        # Optional validity filter:
        # - enabled: keep only tiles with <= max_nodata_fraction no-data.
        # - disabled: keep all tiles (skip percentage validity filtering).
        if self.enforce_tile_validity:
            self.tiles = self.tiles[
                self.tiles["nodata_fraction"] <= self.max_nodata_fraction
            ].reset_index(drop=True)
        else:
            self.tiles = self.tiles.reset_index(drop=True)

        self.temp_global_min = float(self.tiles["temp_global_min"].iloc[0])
        self.temp_global_max = float(self.tiles["temp_global_max"].iloc[0])

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Any:
        # Read one tile entry and either decode precomputed bytes or lazily load from source netCDF.
        row = self.tiles.iloc[idx]
        arr = self._load_tile_array(row)
        y = torch.from_numpy(arr.copy()).unsqueeze(0).float()

        # Build masked input x by randomly zeroing mask_fraction of pixels.
        mask_fraction = float(np.clip(self.mask_fraction, 0.0, 1.0))
        if mask_fraction > 0.0:
            mask = torch.rand_like(y) < mask_fraction
            x = y.clone()
            x[mask] = 0.0
        else:
            x = y.clone()

        return {
            "x": x,
            "y": y,
            "info": self._row_to_info(row, idx),
        }

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _index_has_validity_stats(index_df: pd.DataFrame) -> bool:
        if "nodata_fraction" not in index_df.columns:
            return False
        return bool(index_df["nodata_fraction"].notna().any())

    @staticmethod
    def _index_has_temperature_stats(index_df: pd.DataFrame) -> bool:
        required = {"temp_global_min", "temp_global_max"}
        if not required.issubset(index_df.columns):
            return False
        if index_df.empty:
            return False
        return np.isfinite(index_df["temp_global_min"].iloc[0]) and np.isfinite(
            index_df["temp_global_max"].iloc[0]
        )

    def _build_shared_parquet(self) -> None:
        # Discover all netCDF files once and tile them into a single shared parquet index.
        nc_files = sorted(self.root_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {self.root_dir}")

        # Compute dataset-wide temperature min/max on deepest level once for normalization.
        temp_global_min, temp_global_max = self._compute_temperature_global_minmax(nc_files)

        records: list[dict[str, Any]] = []
        tile_id = 0

        # In fast mode we only write tile metadata (no per-tile extraction); in enforcement mode
        # we also open every tile to compute nodata fraction and cache bytes for fast reads.
        for file_idx, nc_path in enumerate(nc_files):
            with xr.open_dataset(nc_path) as ds:
                # Use deepest available temperature (thetao) slice, similar to:
                # thetao_deep = ds["thetao"].isel(depth=-1); thetao_deep_2d = thetao_deep.squeeze("time", drop=True)
                da, lat_name, lon_name = self._get_temperature_2d(ds)

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

                # Handle coordinates whether they are ascending or descending in the file.
                lat_desc = bool(lats[0] > lats[-1])
                lon_desc = bool(lons[0] > lons[-1])

                # Capture optional no-data markers for validity checks and replacement.
                fill_values = []
                if "_FillValue" in da.attrs:
                    fill_values.append(float(da.attrs["_FillValue"]))
                if "missing_value" in da.attrs:
                    fill_values.append(float(da.attrs["missing_value"]))

                lat_iter = lat_edges
                if self.enforce_tile_validity:
                    lat_iter = tqdm(
                        lat_edges,
                        desc=f"Tiling file {file_idx + 1}/{len(nc_files)}: {nc_path.name}",
                    )

                for lat0 in lat_iter:
                    lat1 = lat0 + self.grid_size_deg

                    for lon0 in lon_edges:
                        lon1 = lon0 + self.grid_size_deg

                        record = {
                            "tile_id": tile_id,
                            "source_file": nc_path.name,
                            "variable": "thetao",
                            "lat_name": lat_name,
                            "lon_name": lon_name,
                            "lat_desc": lat_desc,
                            "lon_desc": lon_desc,
                            "lat_min": float(lat0),
                            "lat_max": float(lat1),
                            "lon_min": float(lon0),
                            "lon_max": float(lon1),
                            "height": int(self.patch_size),
                            "width": int(self.patch_size),
                            "tile_dtype": "float32",
                            "nodata_fraction": np.nan,
                            "temp_global_min": temp_global_min,
                            "temp_global_max": temp_global_max,
                            "tile_bytes": b"",
                            "has_precomputed_tile": False,
                        }

                        if self.enforce_tile_validity:
                            arr, nodata_fraction = self._extract_tile_array(
                                da=da,
                                lat_name=lat_name,
                                lon_name=lon_name,
                                lat0=float(lat0),
                                lat1=float(lat1),
                                lon0=float(lon0),
                                lon1=float(lon1),
                                lat_desc=lat_desc,
                                lon_desc=lon_desc,
                                fill_values=fill_values,
                                temp_global_min=temp_global_min,
                                temp_global_max=temp_global_max,
                            )

                            # Skip empty/non-usable tiles while running strict validity mode.
                            if arr is None:
                                continue

                            record["nodata_fraction"] = nodata_fraction
                            record["tile_bytes"] = arr.tobytes()
                            record["has_precomputed_tile"] = True

                        records.append(record)
                        tile_id += 1

        if not records:
            raise RuntimeError("No valid tiles were generated from the provided .nc files.")

        # Write one shared parquet file so dataset access is fast and deterministic.
        pd.DataFrame.from_records(records).to_parquet(self.index_path, index=False)

    def _load_tile_array(self, row: pd.Series) -> np.ndarray:
        has_precomputed = bool(row.get("has_precomputed_tile", False))
        tile_bytes = row.get("tile_bytes", b"")

        # Fast path: decode cached bytes from parquet (used when enforcement mode built tiles).
        if has_precomputed and isinstance(tile_bytes, (bytes, bytearray)) and len(tile_bytes) > 0:
            return np.frombuffer(
                tile_bytes,
                dtype=np.dtype(row["tile_dtype"]),
            ).reshape((int(row["height"]), int(row["width"])))

        # Lazy path: in fast index mode, compute this tile now by opening only the required source file.
        nc_path = self.root_dir / str(row["source_file"])
        with xr.open_dataset(nc_path) as ds:
            variable = str(row.get("variable", ""))
            lat_name = str(row.get("lat_name", ""))
            lon_name = str(row.get("lon_name", ""))

            if variable not in ds.data_vars or lat_name not in ds.coords or lon_name not in ds.coords:
                variable, lat_name, lon_name = self._pick_var_and_coords(ds)

            da = self._reduce_to_2d(ds[variable], lat_name, lon_name)

            fill_values = []
            if "_FillValue" in da.attrs:
                fill_values.append(float(da.attrs["_FillValue"]))
            if "missing_value" in da.attrs:
                fill_values.append(float(da.attrs["missing_value"]))

            arr, _ = self._extract_tile_array(
                da=da,
                lat_name=lat_name,
                lon_name=lon_name,
                lat0=float(row["lat_min"]),
                lat1=float(row["lat_max"]),
                lon0=float(row["lon_min"]),
                lon1=float(row["lon_max"]),
                lat_desc=bool(row.get("lat_desc", False)),
                lon_desc=bool(row.get("lon_desc", False)),
                fill_values=fill_values,
                temp_global_min=float(row["temp_global_min"]),
                temp_global_max=float(row["temp_global_max"]),
            )

        if arr is None:
            # Rare edge case for empty bins near domain boundaries: return fill-only tile shape.
            return np.full((self.patch_size, self.patch_size), self.nan_fill_value, dtype=np.float32)

        return arr

    def _extract_tile_array(
        self,
        da: xr.DataArray,
        lat_name: str,
        lon_name: str,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
        lat_desc: bool,
        lon_desc: bool,
        fill_values: list[float],
        temp_global_min: float,
        temp_global_max: float,
    ) -> tuple[np.ndarray | None, float]:
        lat_slice = slice(lat1, lat0) if lat_desc else slice(lat0, lat1)
        lon_slice = slice(lon1, lon0) if lon_desc else slice(lon0, lon1)

        tile = da.sel({lat_name: lat_slice, lon_name: lon_slice})
        if tile.sizes.get(lat_name, 0) < 2 or tile.sizes.get(lon_name, 0) < 2:
            return None, 1.0

        # Normalize every degree tile to a fixed pixel grid for model input.
        lat_target = np.linspace(lat0, lat1, self.patch_size)
        lon_target = np.linspace(lon0, lon1, self.patch_size)
        tile = tile.interp(
            {lat_name: lat_target, lon_name: lon_target},
            method=self.interpolation,
        )

        arr = tile.values.astype(np.float32, copy=False)
        if arr.ndim != 2:
            return None, 1.0

        # Compute no-data fraction once; callers can decide whether to enforce filtering.
        nodata_mask = np.isnan(arr)
        for fv in fill_values:
            nodata_mask |= np.isclose(arr, fv)
        nodata_fraction = float(nodata_mask.mean())

        # Normalize valid temperatures to [0, 1] using global deepest-temperature min/max.
        arr_norm = np.empty_like(arr, dtype=np.float32)
        arr_norm[:] = np.nan
        valid_mask = ~nodata_mask
        denom = max(float(temp_global_max - temp_global_min), 1e-12)
        arr_norm[valid_mask] = (arr[valid_mask] - float(temp_global_min)) / denom
        arr_norm = np.clip(arr_norm, 0.0, 1.0)

        # Replace no-data pixels after normalization with configured fill value.
        arr = np.where(np.isnan(arr_norm), self.nan_fill_value, arr_norm).astype(
            np.float32, copy=False
        )
        return arr, nodata_fraction

    def _compute_temperature_global_minmax(self, nc_files: list[Path]) -> tuple[float, float]:
        global_min = np.inf
        global_max = -np.inf

        for nc_path in nc_files:
            with xr.open_dataset(nc_path) as ds:
                da, _, _ = self._get_temperature_2d(ds)
                arr = da.values.astype(np.float32, copy=False)

                nodata_mask = np.isnan(arr)
                if "_FillValue" in da.attrs:
                    nodata_mask |= np.isclose(arr, float(da.attrs["_FillValue"]))
                if "missing_value" in da.attrs:
                    nodata_mask |= np.isclose(arr, float(da.attrs["missing_value"]))

                valid = arr[~nodata_mask]
                if valid.size == 0:
                    continue

                global_min = min(global_min, float(np.min(valid)))
                global_max = max(global_max, float(np.max(valid)))

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise RuntimeError("Could not compute valid global min/max for thetao.")
        if global_max <= global_min:
            raise RuntimeError("Invalid global min/max for thetao: max must be > min.")
        return float(global_min), float(global_max)

    def _get_temperature_2d(self, ds: xr.Dataset) -> tuple[xr.DataArray, str, str]:
        if "thetao" not in ds.data_vars:
            raise RuntimeError("Expected temperature variable 'thetao' in dataset.")

        da = ds["thetao"]

        depth_dim = self._pick_dim_name(da.dims, ("depth",))
        if depth_dim is not None:
            # Use the smallest depth value (closest to the surface).
            depth_coord = da[depth_dim]
            depth_index = int(depth_coord.argmin().item())
            da = da.isel({depth_dim: depth_index})

        time_dim = self._pick_dim_name(da.dims, ("time",))
        if time_dim is not None:
            if int(da.sizes[time_dim]) == 1:
                da = da.squeeze(time_dim, drop=True)
            else:
                da = da.isel({time_dim: 0})

        lat_name = self._pick_dim_name(da.dims, ("lat", "latitude"))
        lon_name = self._pick_dim_name(da.dims, ("lon", "longitude"))
        if lat_name is None or lon_name is None:
            # Fallback to coordinate-based inference when dim names are uncommon.
            lat_name = self._pick_coord_name(ds, ("lat", "latitude"))
            lon_name = self._pick_coord_name(ds, ("lon", "longitude"))

        da = self._reduce_to_2d(da, lat_name, lon_name)
        return da, lat_name, lon_name

    @staticmethod
    def _pick_dim_name(dims: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
        dim_map = {d.lower(): d for d in dims}
        for candidate in candidates:
            if candidate in dim_map:
                return dim_map[candidate]
        for dim in dims:
            dim_l = dim.lower()
            if any(candidate in dim_l for candidate in candidates):
                return dim
        return None

    @staticmethod
    def _row_to_info(row: pd.Series, idx: int) -> dict[str, Any]:
        # Keep all parquet metadata except raw tile bytes to avoid large payloads per sample.
        info: dict[str, Any] = {"dataset_index": int(idx)}
        for key, value in row.items():
            if key == "tile_bytes":
                continue
            if isinstance(value, np.generic):
                info[key] = value.item()
            else:
                info[key] = value
        return info

    @staticmethod
    def _reduce_to_2d(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
        # If a variable has extra dimensions (time/depth/etc.), select the first slice.
        extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
        if extra_dims:
            da = da.isel({d: 0 for d in extra_dims})
        da = da.squeeze(drop=True)

        # Ensure dimension order is always (lat, lon) for consistent downstream slicing.
        if da.dims != (lat_name, lon_name):
            da = da.transpose(lat_name, lon_name)
        return da

    @staticmethod
    def _pick_var_and_coords(ds: xr.Dataset) -> tuple[str, str, str]:
        # Robustly infer coordinate names across common netCDF naming conventions.
        lat_name = DepthTileDataset._pick_coord_name(ds, ("lat", "latitude"))
        lon_name = DepthTileDataset._pick_coord_name(ds, ("lon", "longitude"))

        for name in ds.data_vars:
            dims = ds[name].dims
            if lat_name in dims and lon_name in dims:
                return name, lat_name, lon_name
        raise RuntimeError(
            "Could not find a data variable containing both lat/lon dimensions."
        )

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
    
    def _plot_example_image(self) -> None:
        try:
            import matplotlib.pyplot as plt
            rand_n = np.random.RandomState(self.random_seed)
            idx = rand_n.randint(0, len(self))
            sample = self.__getitem__(idx)
            x = sample["x"].squeeze().numpy()
            y = sample["y"].squeeze().numpy()
            # clean up tensor - remove nan, inf etc
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            # minmax normalization for better visualization
            x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(x, cmap="viridis")
            axes[0].set_title("Input x (masked)")
            axes[1].imshow(y, cmap="viridis")
            axes[1].set_title("Target y (ground truth)")
            plt.tight_layout()
            plt.savefig("temp/example_depth_tile.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot example image: {e}")


if __name__ == "__main__":
    # Quick test: build dataset and print some stats.
    dataset = DepthTileDataset(config_path="configs/data_config.yaml")
    dataset._plot_example_image() # plot random example image to temp folder
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    x,y = sample["x"], sample["y"]
    print(f"x shape: {sample['x'].shape}")
    print(f"y shape: {sample['y'].shape}")
    print(f"info keys: {list(sample['info'].keys())}")
