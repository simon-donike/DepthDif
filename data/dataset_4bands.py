from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch


class SurfaceTempPatch4BandsLightDataset(Dataset):
    """Dataset that loads EO-conditioned multi-band depth patches."""

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
        return_coords: bool = False,
        nan_fill_value: float = 0.0,
        valid_from_fill_value: bool = True,
        eo_random_scale_enabled: bool = False,
        eo_speckle_noise_enabled: bool = False,
        split_seed: int = 7,
        val_fraction: float = 0.2,
    ) -> None:
        """Initialize SurfaceTempPatch4BandsLightDataset with configured parameters.

        Args:
            csv_path (str | Path): Path to an input or output file.
            split (str): Input value.
            mask_fraction (float): Mask tensor controlling valid or known pixels.
            mask_patch_min (int): Mask tensor controlling valid or known pixels.
            mask_patch_max (int): Mask tensor controlling valid or known pixels.
            enable_transform (bool): Boolean flag controlling behavior.
            x_return_mode (str): Input value.
            return_info (bool): Boolean flag controlling behavior.
            return_coords (bool): Boolean flag controlling behavior.
            nan_fill_value (float): Input value.
            valid_from_fill_value (bool): Boolean flag controlling behavior.
            eo_random_scale_enabled (bool): Boolean flag controlling behavior.
            eo_speckle_noise_enabled (bool): Boolean flag controlling behavior.
            split_seed (int): Input value.
            val_fraction (float): Input value.

        Returns:
            None: No value is returned.
        """
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
        self.return_coords = bool(return_coords)
        self.valid_from_fill_value = bool(valid_from_fill_value)
        self.eo_random_scale_enabled = bool(eo_random_scale_enabled)
        self.eo_speckle_noise_enabled = bool(eo_speckle_noise_enabled)
        self.split_seed = int(split_seed)
        self.val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
        self.eo_dropout_prob = 0.0
        if self.enable_transform and self.return_coords:
            warnings.warn(
                "Geometric augmentation is enabled while return_coords=true. "
                "Patch data will be rotated/flipped but coords will remain the "
                "original patch center. Disable transforms if this is undesirable (likely).",
                stacklevel=2,
            )

        self.x_return_mode = str(x_return_mode)
        valid_x_modes = {"corrputed", "currupted_plus_mask"}
        if self.x_return_mode not in valid_x_modes:
            raise ValueError(
                f"Invalid x_return_mode '{self.x_return_mode}'. "
                f"Choose one of: {sorted(valid_x_modes)}"
            )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        # Read the patch index once; rows are resolved to absolute paths on demand in __getitem__.
        import pandas as pd

        if self.csv_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.csv_path)
        else:
            df = pd.read_csv(self.csv_path)
        if "y_npy_path" not in df.columns:
            raise RuntimeError("Index is missing required column 'y_npy_path'.")

        # Support either explicit split column or deterministic local split for reproducibility.
        if self.split in {"train", "val"}:
            if "split" not in df.columns:
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

        # Coordinate conditioning needs patch bounds from the index.
        if self.return_coords:
            required_cols = {"lat0", "lat1", "lon0", "lon1"}
            missing = required_cols.difference(df.columns)
            if missing:
                raise RuntimeError(
                    "Index is missing required coord columns: " f"{sorted(missing)}."
                )

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
    ) -> "SurfaceTempPatch4BandsLightDataset":
        """Compute from config and return the result.

        Args:
            config_path (str): Path to an input or output file.
            split (str): Input value.

        Returns:
            'SurfaceTempPatch4BandsLightDataset': Computed output value.
        """
        cfg = cls._load_config(config_path)
        ds_cfg = cfg["dataset"]
        split_cfg = cfg.get("split", {})
        csv_path = ds_cfg.get(
            "light_index_csv", "data/exported_patches/patch_index_with_paths.parquet"
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
            return_coords=bool(ds_cfg.get("return_coords", False)),
            nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
            valid_from_fill_value=bool(ds_cfg.get("valid_from_fill_value", True)),
            eo_random_scale_enabled=bool(ds_cfg.get("eo_random_scale_enabled", False)),
            eo_speckle_noise_enabled=bool(
                ds_cfg.get("eo_speckle_noise_enabled", False)
            ),
            split_seed=int(ds_cfg.get("random_seed", 7)),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load and return config data.

        Args:
            config_path (str): Path to an input or output file.

        Returns:
            dict[str, Any]: Dictionary containing computed outputs.
        """
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __len__(self) -> int:
        """Return the number of available samples.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            int: Computed scalar output.
        """
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Resolve relative file paths against the index location.
        """Load and return one sample for the given index.

        Args:
            idx (int): Zero-based index for selecting a sample or batch.

        Returns:
            dict[str, Any]: Dictionary containing computed outputs.
        """
        row = self._rows[int(idx)]
        y_rel_path = Path(str(row["y_npy_path"]))
        y_abs_path = (
            y_rel_path if y_rel_path.is_absolute() else self.csv_dir / y_rel_path
        )

        # Expect at least 4 channels: EO (surface) + 3 target depth bands.
        y_np_all = np.load(y_abs_path).astype(np.float32, copy=False)
        if y_np_all.ndim == 2:
            raise RuntimeError(
                f"Unexpected y shape at {y_abs_path}: {tuple(y_np_all.shape)} "
                "(expected (4,H,W), got 2D)."
            )
        if y_np_all.ndim != 3 or y_np_all.shape[0] < 4:
            raise RuntimeError(
                f"Unexpected y shape at {y_abs_path}: {tuple(y_np_all.shape)} "
                "(expected at least 4 channels)."
            )

        # Channel layout: [0]=EO condition, [1:4]=supervised depth targets.
        eo_np = y_np_all[0:1]
        y_np = y_np_all[1:4]

        # Keep a per-depth structural mask so shallow valid bands are preserved
        # even when deeper bands are invalid at the same pixel.
        land_mask_np = (
            np.isfinite(y_np) & (~np.isclose(y_np, 0.0, atol=1e-8))
        ).astype(np.float32, copy=False)

        # to Torch
        eo = torch.from_numpy(eo_np)
        y = torch.from_numpy(y_np)
        land_mask = torch.from_numpy(land_mask_np)
        
        # Add random scale or speckle noise
        if self.eo_random_scale_enabled:
            deg_offset = 2.0 # offset in degrees
            offset = torch.empty((), device=eo.device, dtype=eo.dtype).uniform_(-deg_offset, deg_offset)
            eo = eo + offset
        if self.eo_speckle_noise_enabled:
            noise_std = 0.01  # 1% local variation            
            eps = torch.randn_like(eo)
            multiplier = 1.0 + noise_std * eps
            multiplier = multiplier.clamp(0.9, 1.1)  # allow at most ±10% scaling
            eo = eo * multiplier
            
        # Normalize inputs
        eo = temperature_normalize(mode="norm", tensor=eo)
        y = temperature_normalize(mode="norm", tensor=y)

        # Compute validity per depth channel so mask shape matches y exactly.
        # A pixel is valid only when that specific channel is finite and non-zero.
        zero = torch.zeros((), dtype=y.dtype, device=y.device)
        valid_from_values = torch.isfinite(y) & (
            ~torch.isclose(y, zero, atol=1e-6, rtol=0.0)
        )
        if self.valid_from_fill_value:
            v_fill = ~torch.isclose(
                y, self._normalized_fill_value, atol=1e-6, rtol=0.0
            )
            v = valid_from_values & v_fill
        else:
            v = valid_from_values

        # Apply the same geometric transform to data and masks to keep them aligned.
        if self.enable_transform:
            k_rot, flip_h, flip_v = self._sample_aug_params()
            eo = self._apply_geometric_augment(eo, k_rot, flip_h, flip_v)
            y = self._apply_geometric_augment(y, k_rot, flip_h, flip_v)
            v = self._apply_geometric_augment(v, k_rot, flip_h, flip_v)
            land_mask = self._apply_geometric_augment(land_mask, k_rot, flip_h, flip_v)

        y_clean = y
        valid_mask = v.clone()
        y_corrupt = y_clean.clone()

        # Corrupt x with a per-band mask so validity is truly 3D: (band, H, W).
        if self.mask_fraction > 0.0:
            channels, h, w = y_corrupt.shape
            target = int(round(self.mask_fraction * h * w))
            if target > 0:
                for band_idx in range(channels):
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
                    valid_mask[band_idx, patch_mask] = False
                    y_corrupt[band_idx, patch_mask] = 0.0

        # Keep tensors finite before returning a batch dict.
        x = y_corrupt
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        eo = torch.nan_to_num(eo, nan=0.0, posinf=0.0, neginf=0.0)
        # Random dropout of EO conditioning for ablation/testing (after all other processing to keep it simple).
        if self.eo_dropout_prob > 0.0 and bool(torch.rand(()) < self.eo_dropout_prob):
            eo = torch.zeros_like(eo)

        date = self._parse_date_yyyymmdd(row.get("source_file"))

        sample: dict[str, Any] = {
            "eo": eo,
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "land_mask": land_mask,
            "date": date,
        }
        if self.return_coords:
            lat0 = float(row["lat0"])
            lat1 = float(row["lat1"])
            lon0 = float(row["lon0"])
            lon1 = float(row["lon1"])
            lat_center = 0.5 * (lat0 + lat1)
            lon_center = self._center_lon_deg(lon0, lon1)
            sample["coords"] = torch.tensor(
                [lat_center, lon_center], dtype=torch.float32
            )
        if self.return_info:
            sample["info"] = row
        return sample

    @staticmethod
    def _center_lon_deg(lon0: float, lon1: float) -> float:
        """Helper that computes center lon deg.

        Args:
            lon0 (float): Input value.
            lon1 (float): Input value.

        Returns:
            float: Computed scalar output.
        """
        lon0_rad = np.deg2rad(lon0)
        lon1_rad = np.deg2rad(lon1)
        sin_sum = np.sin(lon0_rad) + np.sin(lon1_rad)
        cos_sum = np.cos(lon0_rad) + np.cos(lon1_rad)
        return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))

    @staticmethod
    def _parse_date_yyyymmdd(source_file: Any) -> int:
        """Helper that computes parse date yyyymmdd.

        Args:
            source_file (Any): Input value.

        Returns:
            int: Computed scalar output.
        """
        source = Path(str(source_file))
        suffix = source.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            if len(suffix) == 8:
                year = int(suffix[:4])
                month = int(suffix[4:6])
                day = int(suffix[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return year * 10000 + month * 100 + day
            if len(suffix) == 6:
                year = int(suffix[:4])
                month = int(suffix[4:6])
                if 1 <= month <= 12:
                    # Monthly source files carry no day; fix to the middle of month for now.
                    return year * 10000 + month * 100 + 15
        # Keep invalid/missing source dates deterministic so batches remain collate-friendly.
        return 19700115

    @staticmethod
    def _sample_aug_params() -> tuple[int, bool, bool]:
        """Helper that computes sample aug params.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            tuple[int, bool, bool]: Tuple containing computed outputs.
        """
        k_rot = int(torch.randint(0, 4, (1,)).item())
        flip_h = bool(torch.randint(0, 2, (1,)).item())
        flip_v = bool(torch.randint(0, 2, (1,)).item())
        return k_rot, flip_h, flip_v

    @staticmethod
    def _apply_geometric_augment(
        t: torch.Tensor, k_rot: int, flip_h: bool, flip_v: bool
    ) -> torch.Tensor:
        """Helper that computes apply geometric augment.

        Args:
            t (torch.Tensor): Tensor input for the computation.
            k_rot (int): Input value.
            flip_h (bool): Boolean flag controlling behavior.
            flip_v (bool): Boolean flag controlling behavior.

        Returns:
            torch.Tensor: Tensor output produced by this call.
        """
        t = torch.rot90(t, k=k_rot, dims=(-2, -1))
        if flip_h:
            t = torch.flip(t, dims=(-1,))
        if flip_v:
            t = torch.flip(t, dims=(-2,))
        return t

    def _plot_example_image(self, idx: int | None = None) -> None:
        """Helper that computes plot example image.

        Args:
            idx (int | None): Zero-based index for selecting a sample or batch.

        Returns:
            None: No value is returned.
        """
        try:
            import matplotlib.pyplot as plt

            if idx is None:
                rand_n = np.random.RandomState()
                idx = rand_n.randint(0, len(self))
            print("Plotting random example image from dataset at index:", idx)
            sample = self.__getitem__(idx)

            eo_t = sample["eo"][0]
            x_t = sample["x"]
            y_t = sample["y"]
            valid_mask_t = sample["valid_mask"]
            land_mask_t = sample["land_mask"]

            eo = temperature_normalize(mode="denorm", tensor=eo_t)
            x = temperature_normalize(mode="denorm", tensor=x_t)
            y = temperature_normalize(mode="denorm", tensor=y_t)

            eo_img = minmax_stretch(eo, mask=valid_mask_t[0], nodata_value=None).numpy()

            num_bands = int(x.shape[0])
            fig, axes = plt.subplots(
                num_bands, 5, figsize=(17, 4 * num_bands), squeeze=False
            )
            for band_idx in range(num_bands):
                mask_band = valid_mask_t[band_idx]
                land_band = land_mask_t[band_idx]
                x_img = minmax_stretch(
                    x[band_idx], mask=mask_band, nodata_value=None
                ).numpy()
                y_img = minmax_stretch(
                    y[band_idx], mask=mask_band, nodata_value=None
                ).numpy()

                axes[band_idx, 0].imshow(eo_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
                axes[band_idx, 1].imshow(x_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
                axes[band_idx, 2].imshow(y_img, cmap=PLOT_CMAP, vmin=0.0, vmax=1.0)
                axes[band_idx, 3].imshow(
                    mask_band.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0
                )
                axes[band_idx, 4].imshow(
                    land_band.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0
                )

                if band_idx == 0:
                    axes[band_idx, 0].set_title("EO (band 0)")
                    axes[band_idx, 1].set_title("Input x")
                    axes[band_idx, 2].set_title("Target y")
                    axes[band_idx, 3].set_title("Valid mask")
                    axes[band_idx, 4].set_title("Land mask")
                axes[band_idx, 0].set_ylabel(f"Band {band_idx + 1}")

                for col in range(5):
                    axes[band_idx, col].set_axis_off()
            plt.tight_layout()
            plt.savefig("temp/example_depth_tile_4bands.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot example image: {e}")


if __name__ == "__main__":
    dataset = SurfaceTempPatch4BandsLightDataset.from_config(
        "configs/data_config_eo_4band.yaml"
    )
    dataset._plot_example_image()

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(
        f"eo shape: {sample['eo'].shape}, x shape: {sample['x'].shape}, "
        f"y shape: {sample['y'].shape}"
    )
    print(
        "Valid mask sum: "
        f"{sample['valid_mask'].sum().item()}, "
        f"Land mask sum: {sample['land_mask'].sum().item()}"
    )
    print(f"Coords: {sample.get('coords', 'N/A')}")

    # testing images
    if False:
        import time

        for i in range(5):
            dataset._plot_example_image()
            time.sleep(4)

    # looking at values
    if False:
        # count 0s in x: pixels are 0s after norm!
        zero_count = (sample["x"] == 0.0).sum().item()
        total_count = sample["x"].numel()
        print(
            f"Zero count in x: {zero_count} / {total_count} ({100 * zero_count / total_count:.2f}%)"
        )
        # count 0s in mask
        mask_zero_count = (sample["valid_mask"] == 0.0).sum().item()
        mask_total_count = sample["valid_mask"].numel()
        print(
            f"Zero count in valid_mask: {mask_zero_count} / {mask_total_count} ({100 * mask_zero_count / mask_total_count:.2f}%)"
        )
