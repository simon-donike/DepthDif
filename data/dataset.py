from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm


class SurfaceTempPatchDataset(Dataset):
    """
    Patch dataset for exporting NetCDF tiles to disk (no corruption/stretching).
    Still a torch Dataset for easy iteration, but primarily used via save_dataset_to_disk.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        index_path: str | Path,
        bands: Sequence[str],
        edge_size: int,
        enforce_validity: bool,
        max_nodata_fraction: float,
        nan_fill_value: float,
        return_info: bool,
        return_coords: bool,
        rebuild_index: bool,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_path)
        if not bands:
            raise ValueError("At least one band must be provided.")
        self.bands = [str(band) for band in bands]
        self.edge_size = int(edge_size)
        self.stride = int(edge_size)  # non-overlapping tiles
        self.enforce_validity = bool(enforce_validity)
        self.max_nodata_fraction = float(max_nodata_fraction)
        self.nan_fill_value = float(nan_fill_value)
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)

        if rebuild_index or not self.index_path.exists():
            index_df = self._build_index()
            self._write_index(index_df, self.index_path)

        if self.index_path.suffix.lower() == ".parquet":
            self.tiles = pd.read_parquet(self.index_path)
        else:
            self.tiles = pd.read_csv(self.index_path)

        if (
            self.enforce_validity
            and "nodata_fraction" in self.tiles.columns
            and self.tiles["nodata_fraction"].notna().any()
        ):
            self.tiles = self.tiles[
                self.tiles["nodata_fraction"] <= self.max_nodata_fraction
            ].reset_index(drop=True)
        else:
            self.tiles = self.tiles.reset_index(drop=True)

        if len(self.tiles) == 0:
            raise RuntimeError("Dataset contains no patches after indexing/filtering.")
        self._init_index_arrays(self.tiles)
        self.tiles = None  # drop DataFrame to reduce worker overhead

    @classmethod
    def from_config(
        cls, config_path: str = "configs/data_config.yaml"
    ) -> "SurfaceTempPatchDataset":
        cfg = cls._load_config(config_path)
        ds_cfg = cfg["dataset"]
        return cls(
            root_dir=ds_cfg["root_dir"],
            index_path=ds_cfg["index_path"],
            bands=ds_cfg.get("bands", ["thetao"]),
            edge_size=int(ds_cfg["edge_size"]),
            enforce_validity=bool(ds_cfg.get("enforce_validity", True)),
            max_nodata_fraction=float(ds_cfg.get("max_nodata_fraction", 0.2)),
            nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
            return_info=bool(ds_cfg.get("return_info", False)),
            return_coords=bool(ds_cfg.get("return_coords", False)),
            rebuild_index=bool(ds_cfg.get("rebuild_index", False)),
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __len__(self) -> int:
        return self._num_tiles

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = int(idx)
        source_file = self._source_file_names[int(self._source_file_codes[row_idx])]
        nc_path = self.root_dir / source_file

        y0 = int(self._y0[row_idx])
        x0 = int(self._x0[row_idx])
        e = int(self._edge_size[row_idx])
        if self.return_coords:
            lat0 = float(self._lat0[row_idx])
            lat1 = float(self._lat1[row_idx])
            lon0 = float(self._lon0[row_idx])
            lon1 = float(self._lon1[row_idx])
            lat_center = 0.5 * (lat0 + lat1)
            lon_center = self._center_lon_deg(lon0, lon1)

        with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
            band_arrays: list[np.ndarray] = []
            band_valid_masks: list[np.ndarray] = []
            lat_name = lon_name = None
            for band in self.bands:
                da2d, lat_name, lon_name = self._band_2d(ds, band)
                patch = da2d.isel(
                    {lat_name: slice(y0, y0 + e), lon_name: slice(x0, x0 + e)}
                )
                arr = patch.values.astype(np.float32, copy=False)
                valid_mask_band = self._validity_mask(arr, da2d).astype(
                    np.float32, copy=False
                )
                np.nan_to_num(
                    arr,
                    copy=False,
                    nan=self.nan_fill_value,
                    posinf=self.nan_fill_value,
                    neginf=self.nan_fill_value,
                )
                band_arrays.append(arr)
                band_valid_masks.append(valid_mask_band)

            nodata_fraction = (
                np.nan
                if self._nodata_fraction is None
                else float(self._nodata_fraction[row_idx])
            )

        y_np = np.stack(band_arrays, axis=0)  # (C, H, W)
        valid_mask_np = np.logical_and.reduce(band_valid_masks).astype(np.float32)
        y = torch.from_numpy(y_np)
        valid_mask = torch.from_numpy(valid_mask_np)
        land_mask = (valid_mask <= 0.0).to(dtype=torch.float32)

        # No corruption: x == y
        x = y.clone()

        sample: Dict[str, Any] = {
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
        }
        if self.return_coords:
            sample["coords"] = torch.tensor(
                [lat_center, lon_center], dtype=torch.float32
            )
        if self.return_info:
            info = {
                key: (
                    self._info_columns[key][row_idx].item()
                    if isinstance(self._info_columns[key][row_idx], np.generic)
                    else self._info_columns[key][row_idx]
                )
                for key in self._info_columns
            }
            info["nodata_fraction_effective"] = nodata_fraction
            info["bands"] = self.bands
            sample["info"] = info
        return sample

    def _init_index_arrays(self, tiles: pd.DataFrame) -> None:
        required_columns = {"source_file", "y0", "x0", "edge_size"}
        if self.return_coords:
            required_columns.update({"lat0", "lat1", "lon0", "lon1"})
        missing_columns = required_columns.difference(tiles.columns)
        if missing_columns:
            raise RuntimeError(
                f"Index file is missing required columns: {sorted(missing_columns)}"
            )

        source_file_cat = tiles["source_file"].astype("category")
        self._source_file_names = source_file_cat.cat.categories.to_list()
        self._source_file_codes = source_file_cat.cat.codes.to_numpy(
            dtype=np.int32, copy=True
        )
        self._y0 = tiles["y0"].to_numpy(dtype=np.int32, copy=True)
        self._x0 = tiles["x0"].to_numpy(dtype=np.int32, copy=True)
        self._edge_size = tiles["edge_size"].to_numpy(dtype=np.int32, copy=True)
        self._nodata_fraction = (
            tiles["nodata_fraction"].to_numpy(dtype=np.float32, copy=True)
            if "nodata_fraction" in tiles.columns
            else None
        )
        if self.return_coords:
            self._lat0 = tiles["lat0"].to_numpy(dtype=np.float32, copy=True)
            self._lat1 = tiles["lat1"].to_numpy(dtype=np.float32, copy=True)
            self._lon0 = tiles["lon0"].to_numpy(dtype=np.float32, copy=True)
            self._lon1 = tiles["lon1"].to_numpy(dtype=np.float32, copy=True)
        self._num_tiles = int(self._source_file_codes.shape[0])

        if self.return_info:
            self._info_columns = {
                column: tiles[column].to_numpy(copy=True) for column in tiles.columns
            }
        else:
            self._info_columns = {}

    @staticmethod
    def _center_lon_deg(lon0: float, lon1: float) -> float:
        lon0_rad = np.deg2rad(lon0)
        lon1_rad = np.deg2rad(lon1)
        sin_sum = np.sin(lon0_rad) + np.sin(lon1_rad)
        cos_sum = np.cos(lon0_rad) + np.cos(lon1_rad)
        return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))

    def _build_index(self) -> pd.DataFrame:
        nc_files = sorted(self.root_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {self.root_dir}")

        template_path = nc_files[0]
        primary_band = self.bands[0]
        template_records: List[Dict[str, Any]] = []
        with xr.open_dataset(template_path, engine="h5netcdf", cache=False) as ds:
            da2d, lat_name, lon_name = self._band_2d(ds, primary_band)
            h = int(da2d.sizes[lat_name])
            w = int(da2d.sizes[lon_name])

            e = self.edge_size
            s = self.stride
            if h < e or w < e:
                raise RuntimeError(
                    "No patches indexed. edge_size is larger than input dimensions."
                )

            lats = ds[lat_name].values
            lons = ds[lon_name].values

            for y0 in tqdm(range(0, h - e + 1, s), desc="Indexing template rows"):
                for x0 in range(0, w - e + 1, s):
                    rec = {
                        "y0": int(y0),
                        "x0": int(x0),
                        "edge_size": int(e),
                        "lat0": float(lats[y0]),
                        "lat1": float(lats[y0 + e - 1]),
                        "lon0": float(lons[x0]),
                        "lon1": float(lons[x0 + e - 1]),
                        "nodata_fraction": np.nan,
                    }
                    if self.enforce_validity:
                        patch = da2d.isel(
                            {
                                lat_name: slice(y0, y0 + e),
                                lon_name: slice(x0, x0 + e),
                            }
                        )
                        arr = patch.astype("float32").to_numpy()
                        rec["nodata_fraction"] = self._nodata_fraction(arr, da2d)
                    template_records.append(rec)

        if not template_records:
            raise RuntimeError(
                "No patches indexed. Check edge_size/stride and data dimensions."
            )

        template_df = pd.DataFrame.from_records(template_records)
        if self.enforce_validity:
            template_df = template_df[
                template_df["nodata_fraction"] <= self.max_nodata_fraction
            ].reset_index(drop=True)

        per_file_frames = []
        for nc_path in nc_files:
            per_file_frames.append(template_df.assign(source_file=nc_path.name))

        df = pd.concat(per_file_frames, ignore_index=True)
        ordered_cols = [
            "source_file",
            "y0",
            "x0",
            "edge_size",
            "lat0",
            "lat1",
            "lon0",
            "lon1",
            "nodata_fraction",
        ]
        return df[ordered_cols]

    @staticmethod
    def _write_index(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=False)
        elif path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("index_path must end with .parquet or .csv")

    def _band_2d(self, ds: xr.Dataset, band: str) -> Tuple[xr.DataArray, str, str]:
        if band not in ds.data_vars:
            raise RuntimeError(f"Expected '{band}' in dataset.")

        da = ds[band]
        lat_name = self._pick_name(ds, da.dims, ("latitude", "lat"))
        lon_name = self._pick_name(ds, da.dims, ("longitude", "lon"))
        depth_dim = self._pick_name(ds, da.dims, ("depth",))
        time_dim = self._pick_name(ds, da.dims, ("time",))

        if time_dim is not None:
            da = da.isel({time_dim: 0})

        if depth_dim is not None:
            depth_coord = ds[depth_dim] if depth_dim in ds.coords else da[depth_dim]
            k0 = int(depth_coord.argmin().item())
            da = da.isel({depth_dim: k0})

        da = da.squeeze(drop=True)
        if da.dims != (lat_name, lon_name):
            da = da.transpose(lat_name, lon_name)
        return da, lat_name, lon_name

    @staticmethod
    def _pick_name(
        ds: xr.Dataset, dims: Tuple[str, ...], candidates: Tuple[str, ...]
    ) -> Optional[str]:
        low = {d.lower(): d for d in dims}
        for c in candidates:
            if c in low:
                return low[c]
        for d in dims:
            dl = d.lower()
            if any(c in dl for c in candidates):
                return d
        lowc = {c.lower(): c for c in ds.coords}
        for c in candidates:
            if c in lowc:
                return lowc[c]
        for c in ds.coords:
            cl = c.lower()
            if any(k in cl for k in candidates):
                return c
        return None

    @staticmethod
    def _nodata_fraction(arr: np.ndarray, da2d: xr.DataArray) -> float:
        mask = ~np.isfinite(arr)
        fill_values = []
        if "_FillValue" in da2d.attrs:
            fill_values.append(float(da2d.attrs["_FillValue"]))
        if "missing_value" in da2d.attrs:
            fill_values.append(float(da2d.attrs["missing_value"]))
        for fv in fill_values:
            mask |= np.isclose(arr, fv)
        return float(mask.mean())

    @staticmethod
    def _validity_mask(arr: np.ndarray, da2d: xr.DataArray) -> np.ndarray:
        valid = np.isfinite(arr)
        fill_values = []
        if "_FillValue" in da2d.attrs:
            fill_values.append(float(da2d.attrs["_FillValue"]))
        if "missing_value" in da2d.attrs:
            fill_values.append(float(da2d.attrs["missing_value"]))
        for fv in fill_values:
            valid &= ~np.isclose(arr, fv)
        return valid

    def _plot_example_image(self) -> None:
        try:
            import matplotlib.pyplot as plt

            rand_n = np.random.RandomState(42)
            idx = rand_n.randint(0, len(self))
            sample = self.__getitem__(idx)
            x_t = sample["x"]
            y_t = sample["y"]
            valid_mask_t = sample["valid_mask"]
            land_mask_t = sample["land_mask"]

            x = x_t[0] if x_t.ndim == 3 else x_t
            y = y_t[0] if y_t.ndim == 3 else y_t
            valid_mask = (
                valid_mask_t[0] if valid_mask_t.ndim == 3 else valid_mask_t
            )
            land_mask = land_mask_t[0] if land_mask_t.ndim == 3 else land_mask_t

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            im0 = axes[0].imshow(x, cmap="viridis")
            axes[0].set_title("Input x (band 0)")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(y, cmap="viridis")
            axes[1].set_title("Target y (band 0)")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            axes[2].imshow(valid_mask.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[2].set_title("Valid mask")
            axes[3].imshow(land_mask.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[3].set_title("Land mask")
            plt.tight_layout()
            plt.savefig("temp/example_depth_tile.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot example image: {e}")

    def _get_stats(self):
        y_min_overall = float("inf")
        y_max_overall = float("-inf")

        count = 0
        mean = 0.0
        M2 = 0.0

        for i in tqdm(range(len(self))):
            sample = self[i]
            y = sample["y"]  # (C, H, W)

            vals = y.reshape(-1)
            y_min = torch.min(vals)
            y_max = torch.max(vals)
            y_min_overall = min(y_min_overall, y_min.item())
            y_max_overall = max(y_max_overall, y_max.item())

            vals = vals.double()
            for v in vals:
                count += 1
                delta = v.item() - mean
                mean += delta / count
                delta2 = v.item() - mean
                M2 += delta * delta2

        variance = M2 / (count - 1) if count > 1 else 0.0
        std = variance**0.5

        print(f"Overall y min: {y_min_overall}")
        print(f"Overall y max: {y_max_overall}")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")

    def save_dataset_to_disk(
        self,
        dir: str | Path,
        *,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        flush_every: int = 500,
        valid_fraction_threshold: float = 0.25,
        bands: Sequence[str] | None = None,
    ) -> Path:
        out_dir = Path(dir)
        y_dir = out_dir / "y_npy"
        y_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "patch_index_with_paths.csv"

        if self.index_path.suffix.lower() == ".parquet":
            index_df = pd.read_parquet(self.index_path)
        else:
            index_df = pd.read_csv(self.index_path)
        index_df = index_df.reset_index(drop=True)

        val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
        valid_fraction_threshold = float(np.clip(valid_fraction_threshold, 0.0, 1.0))
        flush_every = max(1, int(flush_every))
        bands_to_save = [str(b) for b in (self.bands if bands is None else bands)]
        if not bands_to_save:
            raise ValueError("bands must contain at least one variable to save.")

        records: List[Dict[str, Any]] = []
        kept_count = 0
        for i in tqdm(range(len(index_df)), desc="Saving dataset to disk"):
            row_data = index_df.iloc[i]
            source_file = row_data["source_file"]
            nc_path = self.root_dir / str(source_file)
            y0 = int(row_data["y0"])
            x0 = int(row_data["x0"])
            e = int(row_data["edge_size"])

            with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
                band_arrays: list[np.ndarray] = []
                valid_masks: list[np.ndarray] = []
                for band in bands_to_save:
                    da2d, lat_name, lon_name = self._band_2d(ds, band)
                    patch = da2d.isel(
                        {lat_name: slice(y0, y0 + e), lon_name: slice(x0, x0 + e)}
                    )
                    arr = patch.values.astype(np.float32, copy=False)
                    valid_mask_band = self._validity_mask(arr, da2d).astype(
                        np.float32, copy=False
                    )
                    np.nan_to_num(
                        arr,
                        copy=False,
                        nan=self.nan_fill_value,
                        posinf=self.nan_fill_value,
                        neginf=self.nan_fill_value,
                    )
                    band_arrays.append(arr)
                    valid_masks.append(valid_mask_band)

                y_np = np.stack(band_arrays, axis=0)  # (C, H, W)
                valid_mask_np = np.logical_and.reduce(valid_masks).astype(np.float32)
                valid_fraction = float(valid_mask_np.mean())

            if valid_fraction <= valid_fraction_threshold:
                continue

            y_rel_path = Path("y_npy") / f"{kept_count:08d}.npy"
            y_abs_path = out_dir / y_rel_path
            np.save(y_abs_path, y_np)

            row = row_data.to_dict()
            row["sample_idx"] = kept_count
            row["y_npy_path"] = y_rel_path.as_posix()
            row["valid_fraction"] = valid_fraction
            row["bands"] = "|".join(bands_to_save)
            row["num_channels"] = y_np.shape[0]
            records.append(row)
            kept_count += 1
            if kept_count % flush_every == 0:
                pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

        n_samples = len(records)
        val_len = int(round(n_samples * val_fraction))
        if n_samples > 1:
            val_len = min(max(val_len, 1 if val_fraction > 0.0 else 0), n_samples - 1)
        else:
            val_len = 0
        rng = np.random.default_rng(split_seed)
        val_indices = set(rng.permutation(n_samples)[:val_len].tolist())
        for idx, row in enumerate(records):
            row["split"] = "val" if idx in val_indices else "train"

        pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
        return csv_path


if __name__ == "__main__":
    # Example 1: temperature only (thetao), config-driven paths
    ds_temp = SurfaceTempPatchDataset.from_config("configs/data_config.yaml")
    print(f"[thetao] Dataset length: {len(ds_temp)}")
    ds_temp.save_dataset_to_disk("/work/data/depth/extracted/thetao", val_fraction=0.2)

    # Example 2: temperature + salinity (thetao, so)
    ds_temp_sal = SurfaceTempPatchDataset(
        root_dir="/data1/datasets/depth/monthy/",
        index_path="data/patch_index.parquet",
        bands=["thetao", "so"],
        edge_size=128,
        enforce_validity=True,
        max_nodata_fraction=0.25,
        nan_fill_value=0.0,
        return_info=False,
        return_coords=True,
        rebuild_index=False,
    )
    print(f"[thetao, so] Dataset length: {len(ds_temp_sal)}")
    ds_temp_sal.save_dataset_to_disk(
        "/work/data/depth/extracted/thetao_so", val_fraction=0.2
    )

    # Example 3: salinity only (so) with on-the-fly index rebuild
    ds_sal = SurfaceTempPatchDataset(
        root_dir="/data1/datasets/depth/monthy/",
        index_path="data/patch_index_so.parquet",
        bands=["so"],
        edge_size=128,
        enforce_validity=True,
        max_nodata_fraction=0.25,
        nan_fill_value=0.0,
        return_info=True,
        return_coords=True,
        rebuild_index=True,
    )
    print(f"[so] Dataset length: {len(ds_sal)}")
    ds_sal.save_dataset_to_disk("/work/data/depth/extracted/so", val_fraction=0.2)

    # Example 4: write only temperature from a multi-band dataset using save-time override
    ds_temp_sal.save_dataset_to_disk(
        "/work/data/depth/extracted/thetao_only_from_multiband",
        val_fraction=0.2,
        bands=["thetao"],
    )
