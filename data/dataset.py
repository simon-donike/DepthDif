from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.normalizations import (
    PLOT_CMAP,
    temperature_standardize,
    temperature_to_plot_unit,
)


class SurfaceTempPatchDataset(Dataset):
    """
    Surface temperature patch dataset from thetao at time=0 and shallowest depth.
    Returns dict with masked input `x`, target `y`, and metadata `info`.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        index_path: str | Path,
        edge_size: int,
        enforce_validity: bool,
        max_nodata_fraction: float,
        nan_fill_value: float,
        mask_fraction: float,
        x_return_mode: str,
        rebuild_index: bool,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_path)
        self.edge_size = int(edge_size)
        # Always tile with non-overlapping patches.
        self.stride = int(edge_size)
        self.enforce_validity = bool(enforce_validity)
        self.max_nodata_fraction = float(max_nodata_fraction)
        self.nan_fill_value = float(nan_fill_value)
        self.mask_fraction = float(np.clip(mask_fraction, 0.0, 1.0))
        self.x_return_mode = str(x_return_mode)
        valid_x_modes = {"raw_plus_mask", "masked_plus_mask"}
        if self.x_return_mode not in valid_x_modes:
            raise ValueError(
                f"Invalid x_return_mode '{self.x_return_mode}'. "
                f"Choose one of: {sorted(valid_x_modes)}"
            )

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

    @classmethod
    def from_config(
        cls, config_path: str = "configs/data_config.yaml"
    ) -> "SurfaceTempPatchDataset":
        cfg = cls._load_config(config_path)
        ds_cfg = cfg["dataset"]
        return cls(
            root_dir=ds_cfg["root_dir"],
            index_path=ds_cfg["index_path"],
            edge_size=int(ds_cfg["edge_size"]),
            enforce_validity=bool(ds_cfg.get("enforce_validity", True)),
            max_nodata_fraction=float(ds_cfg.get("max_nodata_fraction", 0.2)),
            nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
            mask_fraction=float(ds_cfg.get("mask_fraction", 0.0)),
            x_return_mode=str(ds_cfg.get("x_return_mode", "masked_plus_mask")),
            rebuild_index=bool(ds_cfg.get("rebuild_index", False)),
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.tiles.iloc[int(idx)]
        nc_path = self.root_dir / str(row["source_file"])

        y0 = int(row["y0"])
        x0 = int(row["x0"])
        e = int(row["edge_size"])

        # netCDF4 is not installed in this env; use h5netcdf (declared in requirements).
        with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
            da2d, lat_name, lon_name = self._surface_thetao_2d(ds)
            patch = da2d.isel(
                {lat_name: slice(y0, y0 + e), lon_name: slice(x0, x0 + e)}
            )
            arr = patch.values.astype(np.float32, copy=False)
            nodata_fraction = (
                float(row["nodata_fraction"]) if "nodata_fraction" in row else np.nan
            )

            # v: validity mask from data itself (1=valid ocean, 0=invalid/land/no-data).
            v_np = self._validity_mask(arr, da2d).astype(np.float32, copy=False)

        arr = np.nan_to_num(
            arr,
            nan=self.nan_fill_value,
            posinf=self.nan_fill_value,
            neginf=self.nan_fill_value,
        )
        y = torch.from_numpy(arr).unsqueeze(0)
        v = torch.from_numpy(v_np).unsqueeze(0)

        # Apply the same random geometric augmentation to y and masks.
        k_rot, flip_h, flip_v = self._sample_aug_params()
        y = self._apply_geometric_augment(y, k_rot, flip_h, flip_v)
        v = self._apply_geometric_augment(v, k_rot, flip_h, flip_v)
        y = temperature_standardize(mode="norm", tensor=y)

        # m: inpainting keep-mask (1=known, 0=hide), sampled only inside valid ocean.
        m = torch.ones_like(v, dtype=y.dtype)
        if self.mask_fraction > 0.0:
            hide_inside_valid = (torch.rand_like(v) < self.mask_fraction) & (v > 0.5)
            m[hide_inside_valid] = 0.0

        # k: known pixels = valid ocean pixels kept by inpainting mask.
        k = (v * m).to(dtype=y.dtype)
        masked = y * k  # unknown ocean + all land/invalid become 0.0

        # x return mode variants (only 2 supported).
        if self.x_return_mode == "raw_plus_mask":
            x = torch.cat([y, k], dim=0)
        else:
            x = torch.cat([masked, k], dim=0)

        info = {
            k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()
        }
        info["nodata_fraction_effective"] = nodata_fraction

        return {"x": x, "y": y, "info": info}

    @staticmethod
    def _sample_aug_params() -> Tuple[int, bool, bool]:
        k_rot = int(torch.randint(0, 4, (1,)).item())
        flip_h = bool(torch.randint(0, 2, (1,)).item())
        flip_v = bool(torch.randint(0, 2, (1,)).item())
        return k_rot, flip_h, flip_v

    @staticmethod
    def _apply_geometric_augment(
        t: torch.Tensor, k_rot: int, flip_h: bool, flip_v: bool
    ) -> torch.Tensor:
        t = torch.rot90(t, k=k_rot, dims=(-2, -1))
        if flip_h:
            t = torch.flip(t, dims=(-1,))
        if flip_v:
            t = torch.flip(t, dims=(-2,))
        return t

    def _build_index(self) -> pd.DataFrame:
        nc_files = sorted(self.root_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {self.root_dir}")

        records: List[Dict[str, Any]] = []
        for nc_path in tqdm(nc_files, desc="Indexing files", unit="file"):
            with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
                da2d, lat_name, lon_name = self._surface_thetao_2d(ds)
                h = int(da2d.sizes[lat_name])
                w = int(da2d.sizes[lon_name])

                e = self.edge_size
                s = self.stride
                if h < e or w < e:
                    continue

                lats = ds[lat_name].values
                lons = ds[lon_name].values

                for y0 in range(0, h - e + 1, s):
                    for x0 in range(0, w - e + 1, s):
                        rec = {
                            "source_file": nc_path.name,
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
                            arr = patch.values.astype(np.float32, copy=False)
                            rec["nodata_fraction"] = self._nodata_fraction(arr, da2d)
                        records.append(rec)

        if not records:
            raise RuntimeError(
                "No patches indexed. Check edge_size/stride and data dimensions."
            )

        df = pd.DataFrame.from_records(records)
        if self.enforce_validity:
            df = df[df["nodata_fraction"] <= self.max_nodata_fraction].reset_index(
                drop=True
            )
        return df

    @staticmethod
    def _write_index(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=False)
        elif path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("index_path must end with .parquet or .csv")

    def _surface_thetao_2d(self, ds: xr.Dataset) -> Tuple[xr.DataArray, str, str]:
        if "thetao" not in ds.data_vars:
            raise RuntimeError("Expected 'thetao' in dataset.")

        da = ds["thetao"]
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

            # Plot the first band when channels are present.
            if x_t.ndim == 3:
                x = x_t[0].numpy()
            else:
                x = x_t.numpy()

            if y_t.ndim == 3:
                y = y_t[0].numpy()
            else:
                y = y_t.numpy()

            # Use fixed dataset-level visualization bounds for stable cross-sample contrast.
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            x = temperature_to_plot_unit(
                torch.from_numpy(x), tensor_is_standardized=True
            ).numpy()
            y = temperature_to_plot_unit(
                torch.from_numpy(y), tensor_is_standardized=True
            ).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(x, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[0].set_title("Input x (masked)")
            axes[1].imshow(y, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[1].set_title("Target y (ground truth)")
            plt.tight_layout()
            plt.savefig("temp/example_depth_tile.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot example image: {e}")

    def _get_stats(self):
        y_min_overall = float("inf")
        y_max_overall = float("-inf")

        # running stats
        count = 0
        mean = 0.0
        M2 = 0.0  # sum of squares of differences from the current mean

        for i in tqdm(range(len(self))):
            sample = self[i]
            y = sample["y"]  # (1, H, W)

            # flatten valid values
            vals = y.reshape(-1)

            # min / max
            y_min = torch.min(vals)
            y_max = torch.max(vals)
            y_min_overall = min(y_min_overall, y_min.item())
            y_max_overall = max(y_max_overall, y_max.item())

            # streaming mean / variance (Welford)
            vals = vals.double()  # improve numerical stability
            for v in vals:
                count += 1
                delta = v.item() - mean
                mean += delta / count
                delta2 = v.item() - mean
                M2 += delta * delta2

        variance = M2 / (count - 1)
        std = variance**0.5

        print(f"Overall y min: {y_min_overall}")
        print(f"Overall y max: {y_max_overall}")
        print(f"Mean: {mean}")
        print(f"Std:  {std}")


if __name__ == "__main__":
    dataset = SurfaceTempPatchDataset.from_config("configs/data_config.yaml")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataset._plot_example_image()
    # dataset._get_stats()

    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")
    print(f"Info: {sample['info']}")
