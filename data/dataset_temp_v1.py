from __future__ import annotations

import math
from pathlib import Path
import warnings
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from utils.normalizations import PLOT_CMAP, temperature_normalize
from utils.stretching import minmax_stretch


class SurfaceTempPatchLightDataset(Dataset):
    """Dataset that loads single-band surface temperature patches."""

    def __init__(
        self,
        *,
        csv_path: str | Path,
        split: str = "all",
        mask_fraction: float = 0.0,
        mask_patch_min: int = 3,
        mask_patch_max: int = 9,
        mask_strategy: str = "tracks",
        track_count_min: int = 2,
        track_count_max: int = 8,
        track_width_min: int = 1,
        track_width_max: int = 4,
        track_step_min: int = 1,
        track_step_max: int = 3,
        track_turn_std_deg: float = 18.0,
        track_heading_persistence: float = 0.85,
        track_seed_mode: str = "per_sample_random",
        enable_transform: bool = False,
        x_return_mode: str = "currupted_plus_mask",
        return_info: bool = False,
        return_coords: bool = False,
        nan_fill_value: float = 0.0,
        valid_from_fill_value: bool = True,
        split_seed: int = 7,
        val_fraction: float = 0.2,
    ) -> None:
        """Initialize SurfaceTempPatchLightDataset with configured parameters.

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
        self.mask_strategy = str(mask_strategy).strip().lower()
        valid_mask_strategies = {"tracks", "rectangles"}
        if self.mask_strategy not in valid_mask_strategies:
            raise ValueError(
                f"Invalid mask_strategy '{self.mask_strategy}'. "
                f"Choose one of: {sorted(valid_mask_strategies)}"
            )
        self.track_count_min = int(track_count_min)
        self.track_count_max = int(track_count_max)
        self.track_width_min = int(track_width_min)
        self.track_width_max = int(track_width_max)
        self.track_step_min = int(track_step_min)
        self.track_step_max = int(track_step_max)
        self.track_turn_std_deg = float(track_turn_std_deg)
        self.track_turn_std_rad = math.radians(self.track_turn_std_deg)
        self.track_heading_persistence = float(
            np.clip(track_heading_persistence, 0.0, 1.0)
        )
        self.track_seed_mode = str(track_seed_mode).strip().lower()
        if self.track_count_min < 1 or self.track_count_max < 1:
            raise ValueError("track_count_min and track_count_max must be >= 1.")
        if self.track_count_min > self.track_count_max:
            raise ValueError("track_count_min must be <= track_count_max.")
        if self.track_width_min < 1 or self.track_width_max < 1:
            raise ValueError("track_width_min and track_width_max must be >= 1.")
        if self.track_width_min > self.track_width_max:
            raise ValueError("track_width_min must be <= track_width_max.")
        if self.track_step_min < 1 or self.track_step_max < 1:
            raise ValueError("track_step_min and track_step_max must be >= 1.")
        if self.track_step_min > self.track_step_max:
            raise ValueError("track_step_min must be <= track_step_max.")
        if self.track_seed_mode != "per_sample_random":
            raise ValueError(
                "track_seed_mode currently supports only 'per_sample_random'."
            )
        self.enable_transform = bool(enable_transform)
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
        self.nan_fill_value = float(nan_fill_value)
        self.valid_from_fill_value = bool(valid_from_fill_value)
        self.split_seed = int(split_seed)
        self.val_fraction = float(np.clip(val_fraction, 0.0, 1.0))
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

        # Read the patch index once and keep row metadata in memory.
        import pandas as pd

        df = pd.read_csv(self.csv_path)
        if "y_npy_path" not in df.columns:
            raise RuntimeError("CSV is missing required column 'y_npy_path'.")

        # If split labels are missing, derive a deterministic split so runs are repeatable.
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

        # Coordinate conditioning requires patch bounds in index metadata.
        if self.return_coords:
            required_cols = {"lat0", "lat1", "lon0", "lon1"}
            missing = required_cols.difference(df.columns)
            if missing:
                raise RuntimeError(
                    "CSV is missing required coord columns: "
                    f"{sorted(missing)}."
                )

        if len(df) == 0:
            raise RuntimeError("Dataset is empty after split filtering.")

        self._rows = df.to_dict(orient="records")

        fill = torch.tensor([[[self.nan_fill_value]]], dtype=torch.float32)
        self._normalized_fill_value = temperature_normalize(mode="norm", tensor=fill)

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/data_config.yaml",
        *,
        split: str = "all",
    ) -> "SurfaceTempPatchLightDataset":
        """Compute from config and return the result.

        Args:
            config_path (str): Path to an input or output file.
            split (str): Input value.

        Returns:
            'SurfaceTempPatchLightDataset': Computed output value.
        """
        cfg = cls._load_config(config_path)
        ds_cfg = cfg["dataset"]
        split_cfg = cfg.get("split", {})
        csv_path = cls._cfg_get(
            ds_cfg,
            "source.light_index_csv",
            "light_index_csv",
            default="data/exported_patches/patch_index_with_paths.csv",
        )
        return cls(
            csv_path=csv_path,
            split=split,
            mask_fraction=float(
                cls._cfg_get(ds_cfg, "degradation.mask_fraction", "mask_fraction", default=0.0)
            ),
            mask_patch_min=int(
                cls._cfg_get(ds_cfg, "degradation.mask_patch_min", "mask_patch_min", default=3)
            ),
            mask_patch_max=int(
                cls._cfg_get(ds_cfg, "degradation.mask_patch_max", "mask_patch_max", default=9)
            ),
            mask_strategy=str(
                cls._cfg_get(ds_cfg, "degradation.mask_strategy", "mask_strategy", default="tracks")
            ),
            track_count_min=int(
                cls._cfg_get(ds_cfg, "degradation.track_count_min", "track_count_min", default=2)
            ),
            track_count_max=int(
                cls._cfg_get(ds_cfg, "degradation.track_count_max", "track_count_max", default=8)
            ),
            track_width_min=int(
                cls._cfg_get(ds_cfg, "degradation.track_width_min", "track_width_min", default=1)
            ),
            track_width_max=int(
                cls._cfg_get(ds_cfg, "degradation.track_width_max", "track_width_max", default=4)
            ),
            track_step_min=int(
                cls._cfg_get(ds_cfg, "degradation.track_step_min", "track_step_min", default=1)
            ),
            track_step_max=int(
                cls._cfg_get(ds_cfg, "degradation.track_step_max", "track_step_max", default=3)
            ),
            track_turn_std_deg=float(
                cls._cfg_get(
                    ds_cfg,
                    "degradation.track_turn_std_deg",
                    "track_turn_std_deg",
                    default=18.0,
                )
            ),
            track_heading_persistence=float(
                cls._cfg_get(
                    ds_cfg,
                    "degradation.track_heading_persistence",
                    "track_heading_persistence",
                    default=0.85,
                )
            ),
            track_seed_mode=str(
                cls._cfg_get(
                    ds_cfg,
                    "degradation.track_seed_mode",
                    "track_seed_mode",
                    default="per_sample_random",
                )
            ),
            enable_transform=bool(
                cls._cfg_get(
                    ds_cfg, "augmentation.enable_transform", "enable_transform", default=False
                )
            ),
            x_return_mode=str(
                cls._cfg_get(
                    ds_cfg, "output.x_return_mode", "x_return_mode", default="currupted_plus_mask"
                )
            ),
            return_info=bool(
                cls._cfg_get(ds_cfg, "output.return_info", "return_info", default=False)
            ),
            return_coords=bool(
                cls._cfg_get(ds_cfg, "output.return_coords", "return_coords", default=False)
            ),
            nan_fill_value=float(
                cls._cfg_get(ds_cfg, "validity.nan_fill_value", "nan_fill_value", default=0.0)
            ),
            valid_from_fill_value=bool(
                cls._cfg_get(
                    ds_cfg,
                    "validity.valid_from_fill_value",
                    "valid_from_fill_value",
                    default=True,
                )
            ),
            split_seed=int(
                cls._cfg_get(ds_cfg, "runtime.random_seed", "random_seed", default=7)
            ),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        )

    @staticmethod
    def _cfg_get(
        cfg: dict[str, Any],
        nested_key: str,
        flat_key: str,
        *,
        default: Any,
    ) -> Any:
        """Read nested config keys."""
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
        # Resolve sample file path relative to the CSV folder.
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

        # Load target tile and precompute a land/finite mask before normalization.
        y_np = np.load(y_abs_path).astype(np.float32, copy=False)
        # Land/ocean mask derived before normalization; exclude configured fill value.
        # This keeps masked-loss ocean filtering consistent when disk exports use fill pixels.
        land_mask_np = (
            np.isfinite(y_np)
            & (~np.isclose(y_np, self.nan_fill_value, atol=1e-8))
        ).astype(np.float32, copy=False)
        y = torch.from_numpy(y_np)
        land_mask = torch.from_numpy(land_mask_np)
        if y.ndim == 2:
            y = y.unsqueeze(0)
            land_mask = land_mask.unsqueeze(0)
        if y.ndim != 3:
            raise RuntimeError(
                f"Unexpected y shape at {y_abs_path}: {tuple(y.shape)} (expected (C,H,W))"
            )
        y = temperature_normalize(mode="norm", tensor=y)

        # Validity mask marks learnable pixels (exclude fill values and invalid land pixels).
        if self.valid_from_fill_value:
            v = ~torch.isclose(y, self._normalized_fill_value, atol=1e-6, rtol=0.0)
            v = v.to(dtype=torch.float32)
            v = v * land_mask
        else:
            v = land_mask.clone()

        # Keep masks and data synchronized under random rotation/flip augmentation.
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

        # Generate sparse-observation inputs from trajectory-like sampling tracks.
        if self.mask_fraction > 0.0:
            _, h, w = y_corrupt.shape
            target = int(round(self.mask_fraction * h * w))
            if target > 0:
                preferred = ((valid_mask > 0.5) & (land_mask > 0.5)).any(dim=0)
                if self.mask_strategy == "tracks":
                    patch_mask = self._generate_track_mask((h, w), target, preferred)
                else:
                    patch_mask = self._generate_rectangle_mask((h, w), target)
                hide = patch_mask.unsqueeze(0)
                valid_mask[hide] = 0.0
                y_corrupt[hide] = 0.0

        # Final sanitize step to avoid propagating NaN/Inf into training.
        x = y_corrupt
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        sample: dict[str, Any] = {
            "x": x,
            "y": y,
            "date": self._parse_date_yyyymmdd(row.get("source_file")),
            "valid_mask": valid_mask,
            "land_mask": land_mask,
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
    def _sample_point_from_mask(mask_2d: torch.Tensor) -> tuple[int, int] | None:
        """Sample one pixel location from a 2D boolean mask."""
        coords = torch.nonzero(mask_2d, as_tuple=False)
        if coords.numel() == 0:
            return None
        pick = int(torch.randint(0, coords.size(0), (1,)).item())
        yx = coords[pick]
        return int(yx[0].item()), int(yx[1].item())

    @staticmethod
    def _stamp_disk(mask_2d: torch.Tensor, cy: int, cx: int, radius: int) -> None:
        """Paint a filled disk into the boolean mask in-place."""
        if radius <= 0:
            mask_2d[cy, cx] = True
            return
        h, w = mask_2d.shape
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        yy = torch.arange(y0, y1, device=mask_2d.device, dtype=torch.int64)
        xx = torch.arange(x0, x1, device=mask_2d.device, dtype=torch.int64)
        # Disk stamping avoids blocky corruption and creates track-like footprints.
        yy_grid, xx_grid = torch.meshgrid(yy, xx, indexing="ij")
        disk = (yy_grid - cy) ** 2 + (xx_grid - cx) ** 2 <= radius * radius
        mask_2d[y0:y1, x0:x1] |= disk

    def _generate_rectangle_mask(
        self, spatial_shape: tuple[int, int], target: int
    ) -> torch.Tensor:
        """Generate a rectangle mask (legacy fallback strategy)."""
        h, w = spatial_shape
        patch_mask = torch.zeros((h, w), dtype=torch.bool)
        covered = 0
        while covered < target:
            ph = int(
                torch.randint(self.mask_patch_min, self.mask_patch_max + 1, (1,)).item()
            )
            pw = int(
                torch.randint(self.mask_patch_min, self.mask_patch_max + 1, (1,)).item()
            )
            y0 = int(torch.randint(0, max(1, h - ph + 1), (1,)).item())
            x0 = int(torch.randint(0, max(1, w - pw + 1), (1,)).item())
            patch_mask[y0 : y0 + ph, x0 : x0 + pw] = True
            covered = int(patch_mask.sum().item())
        return patch_mask

    def _generate_track_mask(
        self,
        spatial_shape: tuple[int, int],
        target: int,
        preferred_start_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a trajectory-style mask with smooth random heading updates."""
        h, w = spatial_shape
        patch_mask = torch.zeros((h, w), dtype=torch.bool)
        if target <= 0:
            return patch_mask

        max_pixels = h * w
        target = max(1, min(target, max_pixels))
        preferred = (
            preferred_start_mask.bool()
            if preferred_start_mask is not None
            else torch.zeros((h, w), dtype=torch.bool)
        )
        covered = 0
        track_count = int(
            torch.randint(self.track_count_min, self.track_count_max + 1, (1,)).item()
        )
        max_steps_per_track = max(16, int((h + w) * 2))

        for _ in range(track_count):
            start = self._sample_point_from_mask(preferred)
            if start is None:
                start = (
                    int(torch.randint(0, h, (1,)).item()),
                    int(torch.randint(0, w, (1,)).item()),
                )
            y, x = start
            heading = float(torch.empty(()).uniform_(-math.pi, math.pi).item())
            turn_state = 0.0
            for _ in range(max_steps_per_track):
                brush_width = int(
                    torch.randint(
                        self.track_width_min, self.track_width_max + 1, (1,)
                    ).item()
                )
                brush_radius = max(0, int(round((brush_width - 1) * 0.5)))
                self._stamp_disk(patch_mask, y, x, brush_radius)
                covered = int(patch_mask.sum().item())
                if covered >= target:
                    return patch_mask

                turn_noise = float(torch.randn(()).item()) * self.track_turn_std_rad
                # Keep turns correlated to avoid jagged random-walk artifacts.
                turn_state = (
                    self.track_heading_persistence * turn_state
                    + (1.0 - self.track_heading_persistence) * turn_noise
                )
                heading = heading + turn_state
                step_len = int(
                    torch.randint(self.track_step_min, self.track_step_max + 1, (1,)).item()
                )
                new_y = int(round(y + step_len * math.sin(heading)))
                new_x = int(round(x + step_len * math.cos(heading)))
                new_y = min(max(new_y, 0), h - 1)
                new_x = min(max(new_x, 0), w - 1)

                interp_steps = max(1, step_len)
                for s in range(1, interp_steps + 1):
                    frac = float(s) / float(interp_steps)
                    yi = int(round(y + (new_y - y) * frac))
                    xi = int(round(x + (new_x - x) * frac))
                    self._stamp_disk(patch_mask, yi, xi, brush_radius)
                y, x = new_y, new_x

        # Top-up ensures final masked coverage remains close to the requested fraction.
        top_up_budget = max(32, target * 2)
        attempts = 0
        while covered < target and attempts < top_up_budget:
            start = self._sample_point_from_mask(preferred)
            if start is None:
                start = (
                    int(torch.randint(0, h, (1,)).item()),
                    int(torch.randint(0, w, (1,)).item()),
                )
            brush_width = int(
                torch.randint(self.track_width_min, self.track_width_max + 1, (1,)).item()
            )
            brush_radius = max(0, int(round((brush_width - 1) * 0.5)))
            self._stamp_disk(patch_mask, start[0], start[1], brush_radius)
            covered = int(patch_mask.sum().item())
            attempts += 1
        return patch_mask

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
    print(f"Valid mask sum: {sample['valid_mask'].sum().item()}, Land mask sum: {sample['land_mask'].sum().item()}")
    print(f"Coords: {sample.get('coords', 'N/A')}")
