from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import yaml


class OstiaArgoTiffDataset(Dataset):
    """Dataset that loads paired OSTIA/Argo GeoTIFFs written by the export script."""

    DEFAULT_CONFIG_PATH = "configs/px_space/data_ostia.yaml"
    DEFAULT_CSV_PATH = "/work/data/depth_v3/ostia_argo_tiff_index.csv"
    REQUIRED_PATH_COLUMNS = ("ostia_tif_path", "argo_tif_path")
    SPLIT_CANDIDATE_COLUMNS = ("phase", "split")

    def __init__(
        self,
        csv_path: str | Path = "/work/data/depth_v3/ostia_argo_tiff_index.csv",
        *,
        split: str = "all",
        return_info: bool = True,
        return_coords: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if self.csv_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path)
        if df.empty:
            raise RuntimeError(f"CSV has no rows: {self.csv_path}")

        missing_cols = [col for col in self.REQUIRED_PATH_COLUMNS if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f"Index is missing required columns: {missing_cols}")

        split_col = None
        for candidate in self.SPLIT_CANDIDATE_COLUMNS:
            if candidate in df.columns:
                split_col = candidate
                break

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

    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
        *,
        split: str = "all",
    ) -> "OstiaArgoTiffDataset":
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH
        with Path(config_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        ds_cfg = cfg.get("dataset", {})
        csv_path = cls._cfg_get(
            ds_cfg,
            "core.manifest_csv_path",
            "manifest_csv_path",
            default=cls.DEFAULT_CSV_PATH,
        )
        return cls(
            csv_path=csv_path,
            split=split,
            return_info=bool(
                cls._cfg_get(ds_cfg, "output.return_info", "return_info", default=True)
            ),
            return_coords=bool(
                cls._cfg_get(ds_cfg, "output.return_coords", "return_coords", default=True)
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

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_index_path(self, path_value: Any) -> Path:
        path = Path(str(path_value))
        return path if path.is_absolute() else self.csv_dir / path

    @staticmethod
    def _load_tiff(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32, copy=False)
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected TIFF shape at {path}: {tuple(arr.shape)}")
        return arr

    @staticmethod
    def _parse_date_int(value: Any) -> int:
        raw = str(value).strip()
        if raw.isdigit():
            return int(raw)
        return 19700115

    @staticmethod
    def _center_lon_deg(lon0: float, lon1: float) -> float:
        lon0_rad = np.deg2rad(lon0)
        lon1_rad = np.deg2rad(lon1)
        sin_sum = np.sin(lon0_rad) + np.sin(lon1_rad)
        cos_sum = np.cos(lon0_rad) + np.cos(lon1_rad)
        return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        ostia_path = self._resolve_index_path(row["ostia_tif_path"])
        argo_path = self._resolve_index_path(row["argo_tif_path"])

        eo_np = self._load_tiff(ostia_path)
        x_np = self._load_tiff(argo_path)
        if eo_np.shape[0] != 1:
            raise RuntimeError(
                f"Expected single-band OSTIA GeoTIFF at {ostia_path}, got shape {tuple(eo_np.shape)}"
            )

        valid_mask_np = np.isfinite(x_np)
        eo = torch.from_numpy(np.nan_to_num(eo_np, nan=0.0, posinf=0.0, neginf=0.0))
        x = torch.from_numpy(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0))
        valid_mask = torch.from_numpy(valid_mask_np.astype(np.bool_, copy=False))
        # The exported dataset contains observed Argo layers only, so use them as both
        # conditioning and target to keep the existing training loop contract satisfied.
        y = x.clone()
        land_mask = valid_mask.clone()

        sample: dict[str, Any] = {
            "eo": eo,
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
            "date": self._parse_date_int(row.get("date", 19700115)),
        }
        if self.return_coords:
            lat0 = float(row["lat0"])
            lat1 = float(row["lat1"])
            lon0 = float(row["lon0"])
            lon1 = float(row["lon1"])
            lat_center = 0.5 * (lat0 + lat1)
            lon_center = self._center_lon_deg(lon0, lon1)
            sample["coords"] = torch.tensor([lat_center, lon_center], dtype=torch.float32)
        if self.return_info:
            sample["info"] = row
        return sample


if __name__ == "__main__":
    dataset = OstiaArgoTiffDataset()
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(
        f"eo shape: {tuple(sample['eo'].shape)}, x shape: {tuple(sample['x'].shape)}, "
        f"valid_mask shape: {tuple(sample['valid_mask'].shape)}"
    )
    print(f"Coords: {sample.get('coords', 'N/A')}")
