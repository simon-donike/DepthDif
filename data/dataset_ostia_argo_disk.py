from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import yaml

MISSING_TEXT_VALUES = frozenset({"", "__missing__", "nan", "none", "null"})


class OstiaArgoTiffDataset(Dataset):
    """Dataset that loads georeferenced OSTIA/Argo/GLORYS GeoTIFF exports."""

    DEFAULT_CONFIG_PATH = "configs/px_space/data_ostia.yaml"
    DEFAULT_CSV_PATH = "/work/data/depth_v3/ostia_argo_tiff_index.csv"
    REQUIRED_PATH_COLUMNS = ("ostia_tif_path", "argo_tif_path", "glorys_tif_path")
    GLORYS_PATH_CANDIDATE_COLUMNS = ("glorys_tif_path",)
    SPLIT_CANDIDATE_COLUMNS = ("phase", "split")
    GLORYS_PACK_SCALE = 100.0
    GLORYS_PACK_NODATA = np.int16(-32768)
    EXPORT_SKIPPED_REASON_COLUMN = "export_skipped_reason"

    def __init__(
        self,
        csv_path: str | Path = "/work/data/depth_v3/ostia_argo_tiff_index.csv",
        *,
        split: str = "all",
        return_info: bool = True,
        return_coords: bool = True,
        synthetic_mode: bool = False,
        synthetic_pixel_count: int = 20,
        random_seed: int = 7,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.split = str(split).strip().lower()
        self.return_info = bool(return_info)
        self.return_coords = bool(return_coords)
        self.synthetic_mode = bool(synthetic_mode)
        self.synthetic_pixel_count = int(synthetic_pixel_count)
        self.random_seed = int(random_seed)

        if self.synthetic_pixel_count < 1:
            raise ValueError("synthetic_pixel_count must be >= 1")

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
        self._glorys_path_col = None
        for candidate in self.GLORYS_PATH_CANDIDATE_COLUMNS:
            if candidate in df.columns:
                self._glorys_path_col = candidate
                break
        if self._glorys_path_col is None:
            raise RuntimeError(
                "Index is missing GLORYS TIFF path column. "
                f"Expected one of {list(self.GLORYS_PATH_CANDIDATE_COLUMNS)}."
            )

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

        df = self._filter_available_exports(df)
        if len(df) == 0:
            raise RuntimeError("Dataset is empty after split/export-availability filtering.")

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
            synthetic_mode=bool(
                cls._cfg_get(ds_cfg, "synthetic.enabled", "synthetic_enabled", default=False)
            ),
            synthetic_pixel_count=int(
                cls._cfg_get(ds_cfg, "synthetic.pixel_count", "synthetic_pixel_count", default=20)
            ),
            random_seed=int(
                cls._cfg_get(ds_cfg, "runtime.random_seed", "random_seed", default=7)
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

    @classmethod
    def _filter_available_exports(cls, df: pd.DataFrame) -> pd.DataFrame:
        if cls.EXPORT_SKIPPED_REASON_COLUMN not in df.columns:
            return df.reset_index(drop=True)

        skipped_reason = (
            df[cls.EXPORT_SKIPPED_REASON_COLUMN].fillna("").astype(str).str.strip().str.lower()
        )
        # Export manifests can include rows that were intentionally skipped and therefore
        # never produced TIFFs. Keep only rows without a concrete skip reason.
        available_mask = skipped_reason.isin(MISSING_TEXT_VALUES)
        return df.loc[available_mask].reset_index(drop=True)

    def _resolve_index_path(self, path_value: Any) -> Path:
        path = Path(str(path_value))
        return path if path.is_absolute() else self.csv_dir / path

    @staticmethod
    def _load_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        with rasterio.open(path) as ds:
            arr = ds.read()
            meta = {
                "height": int(ds.height),
                "width": int(ds.width),
                "transform": ds.transform,
                "crs": ds.crs.to_string() if ds.crs is not None else "",
                "band_descriptions": tuple(str(desc or "") for desc in ds.descriptions),
                "dtype": str(ds.dtypes[0]).lower() if ds.count > 0 else "",
                "nodata": ds.nodata,
                "tags": dict(ds.tags()),
            }
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected TIFF shape at {path}: {tuple(arr.shape)}")
        return arr, meta

    @classmethod
    def _decode_glorys_tiff(
        cls,
        *,
        path: Path,
        arr: np.ndarray,
        meta: dict[str, Any],
    ) -> np.ndarray:
        dtype = str(meta.get("dtype", "")).lower()
        tags = meta.get("tags", {})
        encoding = str(tags.get("value_encoding", "")).strip().lower()
        expected_encoding = "packed_int16_celsius_x100"
        if dtype != "int16" or encoding != expected_encoding:
            raise RuntimeError(
                "Packed-only GLORYS loading expects int16 GeoTIFFs tagged with "
                f"value_encoding={expected_encoding!r}, got dtype={dtype!r}, "
                f"encoding={encoding!r} at {path}"
            )

        scale = float(tags.get("scale_factor", "0.01"))
        add_offset = float(tags.get("add_offset", "0.0"))
        packed_nodata = int(tags.get("packed_nodata", str(int(cls.GLORYS_PACK_NODATA))))
        out = arr.astype(np.float32, copy=False)
        nodata_mask = arr == np.int16(packed_nodata)
        out = out * np.float32(scale) + np.float32(add_offset)
        out[nodata_mask] = np.nan
        return out

    @staticmethod
    def _glorys_horizontal_ocean_mask(glorys_patch: np.ndarray) -> np.ndarray:
        """Build a single-band ocean mask from the GLORYS surface layer."""
        patch = np.asarray(glorys_patch, dtype=np.float32)
        if patch.ndim != 3:
            raise RuntimeError(
                "Expected GLORYS patch with shape (C,H,W) when building land mask, "
                f"got {tuple(patch.shape)}"
            )
        if patch.shape[0] == 0:
            raise RuntimeError("Cannot build GLORYS land mask from an empty depth stack.")

        # Use the shallowest GLORYS layer only so the mask stays purely horizontal.
        return np.isfinite(patch[:1]).astype(np.float32, copy=False)

    @staticmethod
    def _assert_raster_alignment(
        *,
        reference_path: Path,
        reference_meta: dict[str, Any],
        other_path: Path,
        other_meta: dict[str, Any],
    ) -> None:
        if (
            int(reference_meta["height"]) != int(other_meta["height"])
            or int(reference_meta["width"]) != int(other_meta["width"])
        ):
            raise RuntimeError(
                "GeoTIFF shape mismatch between exported rasters: "
                f"{reference_path} ({reference_meta['height']}x{reference_meta['width']}) vs "
                f"{other_path} ({other_meta['height']}x{other_meta['width']})"
            )
        if str(reference_meta["crs"]) != str(other_meta["crs"]):
            raise RuntimeError(
                "GeoTIFF CRS mismatch between exported rasters: "
                f"{reference_path} ({reference_meta['crs']}) vs "
                f"{other_path} ({other_meta['crs']})"
            )

        ref_transform = reference_meta["transform"]
        other_transform = other_meta["transform"]
        if not ref_transform.almost_equals(other_transform):
            raise RuntimeError(
                "GeoTIFF transform mismatch between exported rasters: "
                f"{reference_path} ({tuple(ref_transform)}) vs "
                f"{other_path} ({tuple(other_transform)})"
            )

    @staticmethod
    def _normalize_index_text(value: Any) -> str:
        raw = str(value).strip()
        return "" if raw.lower() in MISSING_TEXT_VALUES else raw

    @staticmethod
    def _parse_date_int(value: Any) -> int:
        raw = str(value).strip()
        if raw.isdigit():
            date_int = int(raw)
            month = (date_int // 100) % 100
            day = date_int % 100
            # Keep dataset dates compatible with the model's fixed non-leap calendar.
            if month == 2 and day == 29:
                return date_int - 1
            return date_int
        return 20100101

    @staticmethod
    def _center_lon_deg(lon0: float, lon1: float) -> float:
        lon0_rad = np.deg2rad(lon0)
        lon1_rad = np.deg2rad(lon1)
        sin_sum = np.sin(lon0_rad) + np.sin(lon1_rad)
        cos_sum = np.cos(lon0_rad) + np.cos(lon1_rad)
        return float(np.rad2deg(np.arctan2(sin_sum, cos_sum)))

    def _build_synthetic_x_from_glorys(
        self,
        *,
        y_np: np.ndarray,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        horizontal_valid = np.isfinite(y_np).any(axis=0)
        candidate_flat = np.flatnonzero(horizontal_valid.reshape(-1))
        if candidate_flat.size == 0:
            raise RuntimeError("Synthetic mode found no finite GLORYS pixels to sample from.")

        rng = np.random.default_rng(self.random_seed + int(idx))
        pixel_count_std = max(1.0, 0.1 * float(self.synthetic_pixel_count))
        sampled_pixel_count = int(np.rint(rng.normal(self.synthetic_pixel_count, pixel_count_std)))
        sampled_pixel_count = int(
            np.clip(
                sampled_pixel_count,
                max(1, int(np.floor(0.9 * self.synthetic_pixel_count))),
                max(1, int(np.ceil(1.1 * self.synthetic_pixel_count))),
            )
        )
        sampled_pixel_count = min(sampled_pixel_count, int(candidate_flat.size))

        selected_flat = rng.choice(candidate_flat, size=sampled_pixel_count, replace=False)
        selected_mask_2d = np.zeros(horizontal_valid.shape, dtype=bool)
        selected_mask_2d.reshape(-1)[selected_flat] = True

        x_np = np.full_like(y_np, np.nan, dtype=np.float32)
        selected_mask_3d = np.broadcast_to(selected_mask_2d[None, :, :], y_np.shape)
        # Copy full GLORYS depth columns at sampled pixels so x becomes a sparse profile tensor.
        x_np[selected_mask_3d] = y_np[selected_mask_3d]
        valid_mask_np = np.isfinite(x_np)
        return x_np, valid_mask_np, int(sampled_pixel_count)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[int(idx)]
        ostia_path = self._resolve_index_path(row["ostia_tif_path"])
        argo_path = self._resolve_index_path(row["argo_tif_path"])
        glorys_raw = self._normalize_index_text(row.get(self._glorys_path_col, ""))
        if glorys_raw == "":
            raise RuntimeError(
                "Packed GLORYS disk dataset requires a non-empty glorys_tif_path in every row."
            )
        glorys_path = self._resolve_index_path(glorys_raw)
        missing_paths = [
            str(path) for path in (ostia_path, argo_path, glorys_path) if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Manifest row points to missing exported TIFFs. "
                f"idx={int(idx)}, export_index={row.get('export_index', '')}, "
                f"missing_paths={missing_paths}"
            )

        eo_np_raw, eo_meta = self._load_tiff(ostia_path)
        x_np_raw, x_meta = self._load_tiff(argo_path)
        eo_np = eo_np_raw.astype(np.float32, copy=False)
        x_np = x_np_raw.astype(np.float32, copy=False)
        if eo_np.shape[0] != 1:
            raise RuntimeError(
                f"Expected single-band OSTIA GeoTIFF at {ostia_path}, got shape {tuple(eo_np.shape)}"
            )
        self._assert_raster_alignment(
            reference_path=ostia_path,
            reference_meta=eo_meta,
            other_path=argo_path,
            other_meta=x_meta,
        )

        y_np_raw, y_meta = self._load_tiff(glorys_path)
        self._assert_raster_alignment(
            reference_path=ostia_path,
            reference_meta=eo_meta,
            other_path=glorys_path,
            other_meta=y_meta,
        )
        y_np = self._decode_glorys_tiff(
            path=glorys_path,
            arr=y_np_raw,
            meta=y_meta,
        )

        if y_np.shape != x_np.shape:
            raise RuntimeError(
                "Expected GLORYS and Argo GeoTIFFs to share the same band layout: "
                f"{tuple(y_np.shape)} != {tuple(x_np.shape)}"
            )

        synthetic_pixel_count: int | None = None
        if self.synthetic_mode:
            x_np, valid_mask_np, synthetic_pixel_count = self._build_synthetic_x_from_glorys(
                y_np=y_np,
                idx=int(idx),
            )
        else:
            valid_mask_np = np.isfinite(x_np)
        # Use only the GLORYS surface support so the ocean mask is 2D and cannot vary by depth.
        land_mask_np = self._glorys_horizontal_ocean_mask(y_np)
        eo = torch.from_numpy(np.nan_to_num(eo_np, nan=0.0, posinf=0.0, neginf=0.0))
        x = torch.from_numpy(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0))
        y = torch.from_numpy(np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0))
        valid_mask = torch.from_numpy(valid_mask_np.astype(np.bool_, copy=False))
        land_mask = torch.from_numpy(land_mask_np)
        valid_mask_1d = valid_mask.any(dim=0, keepdim=True)

        sample: dict[str, Any] = {
            "eo": eo,
            "x": x,
            "y": y,
            "valid_mask": valid_mask,
            "valid_mask_1d": valid_mask_1d,
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
            info = dict(row)
            if self.synthetic_mode:
                info["synthetic_mode"] = True
                info["synthetic_pixel_count"] = int(synthetic_pixel_count or 0)
            sample["info"] = info
        return sample

    def save_sample_figure_to_temp(
        self,
        idx: int,
        output_path: str | Path | None = None,
        depth_level: int = 0,
    ) -> Path:
        """Save one sample as a single matplotlib figure in the repo temp directory."""
        import matplotlib.pyplot as plt

        sample = self.__getitem__(int(idx))
        panels: list[tuple[np.ndarray, str]] = []

        eo = sample["eo"]
        x = sample["x"]
        valid_mask = sample["valid_mask"]
        valid_mask_1d = sample["valid_mask_1d"]
        land_mask = sample["land_mask"]
        y = sample["y"]
        info = sample.get("info", {})
        depth_level = int(depth_level)

        if (
            eo.ndim != 3
            or x.ndim != 3
            or valid_mask.ndim != 3
            or valid_mask_1d.ndim != 3
            or land_mask.ndim != 3
            or y.ndim != 3
        ):
            raise RuntimeError(
                "Expected image-like sample tensors with shape (C,H,W): "
                f"eo={tuple(eo.shape)}, x={tuple(x.shape)}, y={tuple(y.shape)}, "
                f"valid_mask={tuple(valid_mask.shape)}, valid_mask_1d={tuple(valid_mask_1d.shape)}, "
                f"land_mask={tuple(land_mask.shape)}"
            )
        if depth_level < 0 or depth_level >= int(x.shape[0]):
            raise RuntimeError(
                f"depth_level={depth_level} is out of range for sample with {int(x.shape[0])} depth bands."
            )
        if int(y.shape[0]) <= depth_level or int(valid_mask.shape[0]) <= depth_level:
            raise RuntimeError(
                "Expected x, y, and valid_mask to all include the requested depth level: "
                f"depth_level={depth_level}, x={tuple(x.shape)}, y={tuple(y.shape)}, "
                f"valid_mask={tuple(valid_mask.shape)}"
            )
        if int(land_mask.shape[0]) not in {1, int(x.shape[0])}:
            raise RuntimeError(
                "Expected land_mask to be either a single horizontal mask or per-depth mask: "
                f"x={tuple(x.shape)}, land_mask={tuple(land_mask.shape)}"
            )
        land_mask_level = 0 if int(land_mask.shape[0]) == 1 else depth_level

        panels.append((np.asarray(eo[0].detach().cpu().numpy(), dtype=np.float32), "eo[0]"))
        panels.append(
            (
                np.asarray(x[depth_level].detach().cpu().numpy(), dtype=np.float32),
                f"x[{depth_level}]",
            )
        )
        panels.append(
            (
                np.asarray(y[depth_level].detach().cpu().numpy(), dtype=np.float32),
                f"y[{depth_level}]",
            )
        )
        # Cast boolean masks to float so imshow produces a stable binary visualization.
        panels.append(
            (
                np.asarray(valid_mask_1d[0].detach().cpu().numpy(), dtype=np.float32),
                "valid_mask_1d[0]",
            )
        )
        panels.append(
            (
                np.asarray(land_mask[land_mask_level].detach().cpu().numpy(), dtype=np.float32),
                "land_mask[0]" if land_mask_level == 0 and int(land_mask.shape[0]) == 1 else f"land_mask[{depth_level}]",
            )
        )

        if output_path is None:
            output_path = Path("temp") / f"ostia_argo_tiff_sample_{int(idx)}.png"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_panels = len(panels) + 1
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.6 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        axes_flat = axes.reshape(-1)

        for ax, (img, title) in zip(axes_flat, panels):
            im = ax.imshow(img, cmap="viridis")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        info_ax = axes_flat[len(panels)]
        info_ax.set_axis_off()
        info_text = textwrap.fill(str(info), width=80)
        coords = sample.get("coords", None)
        coords_text = "N/A" if coords is None else np.array2string(coords.detach().cpu().numpy(), precision=4)
        info_ax.text(
            0.0,
            1.0,
            f"idx={int(idx)}\ndepth_level={depth_level}\ndate={sample['date']}\ncoords={coords_text}\n\ninfo={info_text}",
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            wrap=True,
        )

        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()

        fig.suptitle(f"OstiaArgoTiffDataset sample {int(idx)} depth {depth_level}", fontsize=12)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    dataset = OstiaArgoTiffDataset(csv_path="/work/data/depth_prod/ostia_argo_tiff_index.csv", split="all")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataset.save_sample_figure_to_temp(1)
    print(f"Sample keys: {list(sample.keys())}")
    print(
        f"eo shape: {tuple(sample['eo'].shape)}, x shape: {tuple(sample['x'].shape)}, "
        f"y shape: {tuple(sample['y'].shape)}, "
        f"valid_mask shape: {tuple(sample['valid_mask'].shape)}"
    )
    print(f"Coords: {sample.get('coords', 'N/A')}")
