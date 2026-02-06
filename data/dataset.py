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
from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch


class SurfaceTempPatchDataset(Dataset):
    """
    Surface temperature patch dataset from thetao at time=0 and shallowest depth.
    Returns dict with corrupted input `x`, target `y`, `valid_mask`, `land_mask`,
    and metadata `info`.
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
        enable_transform: bool,
        x_return_mode: str,
        return_info: bool,
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
        self.enable_transform = bool(enable_transform)
        self.x_return_mode = str(x_return_mode)
        self.return_info = bool(return_info)
        valid_x_modes = {"corrputed", "currupted_plus_mask"}
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
        self._init_index_arrays(self.tiles)
        # Drop the DataFrame object to avoid per-worker pandas overhead in DataLoader workers.
        self.tiles = None

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
            enable_transform=bool(ds_cfg.get("enable_transform", True)),
            x_return_mode=str(ds_cfg.get("x_return_mode", "currupted_plus_mask")),
            return_info=bool(ds_cfg.get("return_info", False)),
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

        # use h5netcdf (declared in requirements).
        with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
            da2d, lat_name, lon_name = self._surface_thetao_2d(ds)
            patch = da2d.isel(
                {lat_name: slice(y0, y0 + e), lon_name: slice(x0, x0 + e)}
            )
            arr = patch.values.astype(np.float32, copy=False)
            nodata_fraction = (
                np.nan
                if self._nodata_fraction is None
                else float(self._nodata_fraction[row_idx])
            )

            # v: validity mask from data itself (1=valid ocean, 0=invalid/land/no-data).
            v_np = self._validity_mask(arr, da2d).astype(np.float32, copy=False)
            # do nan correction in place
            np.nan_to_num(
                arr,
                copy=False,
                nan=self.nan_fill_value,
                posinf=self.nan_fill_value,
                neginf=self.nan_fill_value,
            )
        y = torch.from_numpy(arr).unsqueeze(0)
        v = torch.from_numpy(v_np).unsqueeze(0)

        # Normalize Temperature: 0 mean, 1 stdev
        y = temperature_normalize(mode="norm", tensor=y)

        if self.enable_transform:
            # Apply the same random geometric augmentation to y and masks if enabled.
            k_rot, flip_h, flip_v = self._sample_aug_params()
            y = self._apply_geometric_augment(y, k_rot, flip_h, flip_v)
            v = self._apply_geometric_augment(v, k_rot, flip_h, flip_v)


        # Keep uncorrupted target
        y_clean = y  # (1, H, W)

        # Build known-mask (start from validity mask)
        valid_mask = v.clone()  # (1, H, W), float32 0/1 (or bool if you switch)
        land_mask = (v <= 0.0).to(dtype=torch.float32)

        # Create corrupted input as a single clone (minimum needed to keep y_clean intact)
        y_corrupt = y_clean.clone()

        if self.mask_fraction > 0.0:
            hide = torch.rand_like(y_corrupt) < self.mask_fraction  # random over whole image
            valid_mask[hide] = 0.0
            y_corrupt[hide] = 0.0

        # x is the corrupted input; y is the clean target
        x = y_corrupt  # (1, H, W)

        sample = {
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
        }
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
            sample["info"] = info
        return sample

    def _init_index_arrays(self, tiles: pd.DataFrame) -> None:
        required_columns = {"source_file", "y0", "x0", "edge_size"}
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
        self._num_tiles = int(self._source_file_codes.shape[0])

        if self.return_info:
            self._info_columns = {
                column: tiles[column].to_numpy(copy=True) for column in tiles.columns
            }
        else:
            self._info_columns = {}

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

        template_path = nc_files[0]
        template_records: List[Dict[str, Any]] = []
        with xr.open_dataset(template_path, engine="h5netcdf", cache=False) as ds:
            da2d, lat_name, lon_name = self._surface_thetao_2d(ds)
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
            valid_mask_t = sample["valid_mask"]
            land_mask_t = sample["land_mask"]

            # Plot the first band when channels are present.
            x = x_t[0] if x_t.ndim == 3 else x_t
            y = y_t[0] if y_t.ndim == 3 else y_t
            valid_mask = (
                valid_mask_t[0] if valid_mask_t.ndim == 3 else valid_mask_t
            )
            land_mask = land_mask_t[0] if land_mask_t.ndim == 3 else land_mask_t

            # Use fixed dataset-level visualization bounds for stable cross-sample contrast.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            x = temperature_normalize(mode="denorm", tensor=x)
            y = temperature_normalize(mode="denorm", tensor=y)
            x = minmax_stretch(x, mask=valid_mask).numpy()
            y = minmax_stretch(y, mask=valid_mask).numpy()

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(x, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[0].set_title("Input x (masked)")
            axes[1].imshow(y, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
            axes[1].set_title("Target y (ground truth)")
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
        
    def save_dataset_to_disk(
        self,
        dir: str | Path,
        *,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        flush_every: int = 500,
        valid_fraction_threshold: float = 0.25,
    ) -> Path:
        out_dir = Path(dir)
        y_dir = out_dir / "y_npy"
        y_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "patch_index_with_paths.csv"

        if self.index_path.suffix.lower() == ".parquet":
            index_df = pd.read_parquet(self.index_path)
        else:
            index_df = pd.read_csv(self.index_path)
        # No filtering: export every row from the index as-is.
        index_df = index_df.reset_index(drop=True)

        val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
        valid_fraction_threshold = float(np.clip(valid_fraction_threshold, 0.0, 1.0))
        flush_every = max(1, int(flush_every))

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
                da2d, lat_name, lon_name = self._surface_thetao_2d(ds)
                patch = da2d.isel(
                    {lat_name: slice(y0, y0 + e), lon_name: slice(x0, x0 + e)}
                )
                arr = patch.values.astype(np.float32, copy=False)
                valid_fraction = float(self._validity_mask(arr, da2d).mean())

            if valid_fraction <= valid_fraction_threshold:
                continue

            y_np = arr

            y_rel_path = Path("y_npy") / f"{kept_count:08d}.npy"
            y_abs_path = out_dir / y_rel_path
            np.save(y_abs_path, y_np)

            row = row_data.to_dict()
            row["sample_idx"] = kept_count
            row["y_npy_path"] = y_rel_path.as_posix()
            row["valid_fraction"] = valid_fraction
            records.append(row)
            kept_count += 1
            if (kept_count % flush_every == 0):
                pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

        # Assign train/val split after filtering so ratios are correct.
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
    dataset = SurfaceTempPatchDataset.from_config("configs/data_config.yaml")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataset._plot_example_image()
    dataset.save_dataset_to_disk("/work/data/depth/extracted/", val_fraction=0.25)
    #dataset._get_stats()

    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")
    print(f"Info: {sample['info']}")
