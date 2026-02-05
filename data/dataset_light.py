from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from utils.normalizations import temperature_normalize


class SurfaceTempPatchLightDataset(Dataset):
    """
    Lightweight patch dataset that reads pre-saved y patches (.npy) from a CSV.
    Returns dict with masked input `x`, target `y`, and optional metadata `info`.
    """

    def __init__(
        self,
        *,
        csv_path: str | Path,
        split: str = "all",
        mask_fraction: float = 0.0,
        enable_transform: bool = False,
        x_return_mode: str = "currupted_plus_mask",
        return_info: bool = False,
        nan_fill_value: float = 0.0,
        valid_from_fill_value: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.mask_fraction = float(np.clip(mask_fraction, 0.0, 1.0))
        self.enable_transform = bool(enable_transform)
        self.return_info = bool(return_info)
        self.valid_from_fill_value = bool(valid_from_fill_value)

        self.x_return_mode = str(x_return_mode)
        valid_x_modes = {"corrputed", "currupted_plus_mask"}
        if self.x_return_mode not in valid_x_modes:
            raise ValueError(
                f"Invalid x_return_mode '{self.x_return_mode}'. "
                f"Choose one of: {sorted(valid_x_modes)}"
            )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        import pandas as pd

        df = pd.read_csv(self.csv_path)
        if "y_npy_path" not in df.columns:
            raise RuntimeError("CSV is missing required column 'y_npy_path'.")

        if self.split in {"train", "val"}:
            if "split" not in df.columns:
                raise RuntimeError(
                    "CSV is missing 'split' column required for train/val filtering."
                )
            df = df[df["split"].astype(str).str.lower() == self.split].reset_index(
                drop=True
            )
        elif self.split != "all":
            raise ValueError("split must be one of: 'all', 'train', 'val'")

        if len(df) == 0:
            raise RuntimeError("Dataset is empty after split filtering.")

        self._rows = df.to_dict(orient="records")

        fill = torch.tensor([[[float(nan_fill_value)]]], dtype=torch.float32)
        self._normalized_fill_value = temperature_normalize(mode="norm", tensor=fill)

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/data_config.yaml",
        *,
        split: str = "all",
    ) -> "SurfaceTempPatchLightDataset":
        cfg = cls._load_config(config_path)
        ds_cfg = cfg["dataset"]
        csv_path = ds_cfg.get(
            "light_index_csv", "data/exported_patches/patch_index_with_paths.csv"
        )
        return cls(
            csv_path=csv_path,
            split=split,
            mask_fraction=float(ds_cfg.get("mask_fraction", 0.0)),
            enable_transform=bool(ds_cfg.get("enable_transform", False)),
            x_return_mode=str(ds_cfg.get("x_return_mode", "currupted_plus_mask")),
            return_info=bool(ds_cfg.get("return_info", False)),
            nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
            valid_from_fill_value=bool(ds_cfg.get("valid_from_fill_value", True)),
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        y_rel_path = Path(str(row["y_npy_path"]))
        y_abs_path = y_rel_path if y_rel_path.is_absolute() else self.csv_dir / y_rel_path

        y_np = np.load(y_abs_path).astype(np.float32, copy=False)
        y = torch.from_numpy(y_np)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        if y.ndim != 3 or y.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected y shape at {y_abs_path}: {tuple(y.shape)} (expected (1,H,W))"
            )
        y = temperature_normalize(mode="norm", tensor=y)

        if self.valid_from_fill_value:
            v = ~torch.isclose(y, self._normalized_fill_value, atol=1e-6, rtol=0.0)
            v = v.to(dtype=torch.float32)
        else:
            v = torch.ones_like(y, dtype=torch.float32)

        if self.enable_transform:
            k_rot, flip_h, flip_v = self._sample_aug_params()
            y = self._apply_geometric_augment(y, k_rot, flip_h, flip_v)
            v = self._apply_geometric_augment(v, k_rot, flip_h, flip_v)

        y_clean = y
        k = v.clone()
        y_corrupt = y_clean.clone()

        if self.mask_fraction > 0.0:
            hide = torch.rand_like(y_corrupt) < self.mask_fraction
            k[hide] = 0.0
            y_corrupt[hide] = 0.0

        if self.x_return_mode == "currupted_plus_mask":
            x = torch.cat([y_corrupt, k], dim=0)
        else:
            x = y_corrupt

        sample: dict[str, Any] = {"x": x, "y": y}
        if self.return_info:
            sample["info"] = row
        return sample

    @staticmethod
    def _sample_aug_params() -> tuple[int, bool, bool]:
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


if __name__ == "__main__":
    # Example usage
    dataset = SurfaceTempPatchLightDataset.from_config("configs/data_config.yaml")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")