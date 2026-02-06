from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch


class SurfaceTempPatchLightDataset(Dataset):
    """
    Lightweight patch dataset that reads pre-saved y patches (.npy) from a CSV.
    Returns dict with corrupted input `x`, target `y`, `valid_mask`, `land_mask`,
    and optional metadata `info`.
    """

    def __init__(
        self,
        *,
        csv_path: str | Path,
        split: str = "all",
        mask_fraction: float = 0.0,
        mask_patch_min: int = 3,
        mask_patch_max: int = 9,
        enable_transform: bool = False,
        x_return_mode: str = "currupted_plus_mask",
        return_info: bool = False,
        nan_fill_value: float = 0.0,
        valid_from_fill_value: bool = True,
        split_seed: int = 7,
        val_fraction: float = 0.2,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.mask_fraction = float(np.clip(mask_fraction, 0.0, 1.0))
        self.mask_patch_min = int(mask_patch_min)
        self.mask_patch_max = int(mask_patch_max)
        if self.mask_patch_min < 1 or self.mask_patch_max < 1:
            raise ValueError("mask_patch_min and mask_patch_max must be >= 1.")
        if self.mask_patch_min > self.mask_patch_max:
            raise ValueError("mask_patch_min must be <= mask_patch_max.")
        self.enable_transform = bool(enable_transform)
        self.return_info = bool(return_info)
        self.valid_from_fill_value = bool(valid_from_fill_value)
        self.split_seed = int(split_seed)
        self.val_fraction = float(np.clip(val_fraction, 0.0, 1.0))

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
                # Deterministic split if CSV is missing split column.
                n_samples = len(df)
                val_len = int(round(n_samples * self.val_fraction))
                if n_samples > 1:
                    val_len = min(
                        max(val_len, 1 if self.val_fraction > 0.0 else 0),
                        n_samples - 1,
                    )
                else:
                    val_len = 0
                rng = np.random.default_rng(self.split_seed)
                val_indices = set(rng.permutation(n_samples)[:val_len].tolist())
                split_col = [
                    "val" if i in val_indices else "train" for i in range(n_samples)
                ]
                df = df.copy()
                df["split"] = split_col
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
        split_cfg = cfg.get("split", {})
        csv_path = ds_cfg.get(
            "light_index_csv", "data/exported_patches/patch_index_with_paths.csv"
        )
        return cls(
            csv_path=csv_path,
            split=split,
            mask_fraction=float(ds_cfg.get("mask_fraction", 0.0)),
            mask_patch_min=int(ds_cfg.get("mask_patch_min", 3)),
            mask_patch_max=int(ds_cfg.get("mask_patch_max", 9)),
            enable_transform=bool(ds_cfg.get("enable_transform", False)),
            x_return_mode=str(ds_cfg.get("x_return_mode", "currupted_plus_mask")),
            return_info=bool(ds_cfg.get("return_info", False)),
            nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
            valid_from_fill_value=bool(ds_cfg.get("valid_from_fill_value", True)),
            split_seed=int(ds_cfg.get("random_seed", 7)),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
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
        y_abs_path = (
            y_rel_path if y_rel_path.is_absolute() else self.csv_dir / y_rel_path
        )

        y_np = np.load(y_abs_path).astype(np.float32, copy=False)
        # Land mask must be derived from raw on-disk values before any normalization.
        # Treat non-finite values as land (mask=0), finite as water (mask=1).
        land_mask_np = np.isfinite(y_np).astype(np.float32, copy=False)
        y = torch.from_numpy(y_np)
        land_mask = torch.from_numpy(land_mask_np)
        if y.ndim == 2:
            y = y.unsqueeze(0)
            land_mask = land_mask.unsqueeze(0)
        if y.ndim != 3 or y.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected y shape at {y_abs_path}: {tuple(y.shape)} (expected (1,H,W))"
            )
        y = temperature_normalize(mode="norm", tensor=y)

        if self.valid_from_fill_value:
            v = ~torch.isclose(y, self._normalized_fill_value, atol=1e-6, rtol=0.0)
            v = v.to(dtype=torch.float32)
            v = v * land_mask
        else:
            v = land_mask.clone()

        if self.enable_transform:
            k_rot, flip_h, flip_v = self._sample_aug_params()
            y = self._apply_geometric_augment(y, k_rot, flip_h, flip_v)
            v = self._apply_geometric_augment(v, k_rot, flip_h, flip_v)
            land_mask = self._apply_geometric_augment(
                land_mask, k_rot, flip_h, flip_v
            )

        y_clean = y
        valid_mask = v.clone()
        y_corrupt = y_clean.clone()

        if self.mask_fraction > 0.0:
            _, h, w = y_corrupt.shape
            target = int(round(self.mask_fraction * h * w))
            if target > 0:
                patch_mask = torch.zeros(
                    (h, w), dtype=torch.bool, device=y_corrupt.device
                )
                covered = 0
                while covered < target:
                    ph = int(
                        torch.randint(
                            self.mask_patch_min, self.mask_patch_max + 1, (1,)
                        ).item()
                    )
                    pw = int(
                        torch.randint(
                            self.mask_patch_min, self.mask_patch_max + 1, (1,)
                        ).item()
                    )
                    y0 = int(torch.randint(0, max(1, h - ph + 1), (1,)).item())
                    x0 = int(torch.randint(0, max(1, w - pw + 1), (1,)).item())
                    patch_mask[y0 : y0 + ph, x0 : x0 + pw] = True
                    covered = int(patch_mask.sum().item())
                hide = patch_mask.unsqueeze(0)
                valid_mask[hide] = 0.0
                y_corrupt[hide] = 0.0

        x = y_corrupt
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        sample: dict[str, Any] = {
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
        }
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

    def _plot_example_image(self, idx=None) -> None:
        try:
            import matplotlib.pyplot as plt

            if idx is None:
                rand_n = np.random.RandomState()
                idx = rand_n.randint(0, len(self))
            print("Plotting random example image from dataset at index:", idx)
            sample = self.__getitem__(idx)
            x_t = sample["x"]
            y_t = sample["y"]
            valid_mask_t = sample["valid_mask"]
            land_mask_t = sample["land_mask"]

            # Plot the first band when channels are present.
            x = x_t[0] if x_t.ndim != 1 else x_t
            y = y_t[0] if y_t.ndim != 1 else y_t
            valid_mask = (
                valid_mask_t[0] if valid_mask_t.ndim != 1 else valid_mask_t
            )
            land_mask = land_mask_t[0] if land_mask_t.ndim != 1 else land_mask_t

            # Use fixed dataset-level visualization bounds for stable cross-sample contrast.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            # denormlaize x and y
            x = temperature_normalize(mode="denorm", tensor=x)
            y = temperature_normalize(mode="denorm", tensor=y)
            x = minmax_stretch(x, mask=valid_mask, nodata_value=None).numpy()
            y = minmax_stretch(y, mask=valid_mask, nodata_value=None).numpy()

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
            plt.savefig("temp/example_depth_tile_light.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot example image: {e}")


if __name__ == "__main__":
    # Example usage
    dataset = SurfaceTempPatchLightDataset.from_config("configs/data_config.yaml")
    dataset._plot_example_image(idx=11)
    
    import time
    for i in range(10):
        dataset._plot_example_image()
        time.sleep(5)
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")


    sample=dataset[11]
