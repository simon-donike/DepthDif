from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm


@dataclass
class _ExportParams:
    """Container for dataset export configuration parameters."""
    root_dir: Path
    output_dir: Path
    variable: str
    bands: tuple[int, ...]
    edge_size: int
    enforce_validity: bool
    max_nodata_fraction: float
    nan_fill_value: float
    valid_fraction_threshold: float
    val_fraction: float
    split_seed: int
    flush_every: int


class _NetCDFPatchExporter:
    """Exporter that writes NetCDF patch data and metadata to disk."""
    def __init__(self, params: _ExportParams) -> None:
        """Initialize _NetCDFPatchExporter with configured parameters.

        Args:
            params (_ExportParams): Input value.

        Returns:
            None: No value is returned.
        """
        self.params = params

    def run(self) -> Path:
        """Compute run and return the result.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            Path: Computed output value.
        """
        nc_files = sorted(self.params.root_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {self.params.root_dir}")

        self.params.output_dir.mkdir(parents=True, exist_ok=True)
        y_dir = self.params.output_dir / "y_npy"
        y_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.params.output_dir / "patch_index_with_paths.csv"

        # Always build a fresh index from one template image, then replicate for all files.
        index_df = self._build_full_index(nc_files)
        index_df = index_df[
            index_df["template_valid_fraction"] > self.params.valid_fraction_threshold
        ].reset_index(drop=True)
        if index_df.empty:
            raise RuntimeError(
                "No tensors were saved. "
                "All patches were filtered out by valid_fraction_threshold."
            )

        records: list[dict[str, Any]] = []
        kept_count = 0

        grouped = index_df.groupby("source_file", sort=False)
        with tqdm(total=len(index_df), desc="Saving dataset to disk") as pbar:
            for source_file, rows in grouped:
                nc_path = self.params.root_dir / str(source_file)

                with xr.open_dataset(nc_path, engine="h5netcdf", cache=False) as ds:
                    da3d, lat_name, lon_name, selected_idx, selected_depths = (
                        self._extract_depth_stack(ds)
                    )

                    for row in rows.itertuples(index=False):
                        y0 = int(row.y0)
                        x0 = int(row.x0)
                        edge = int(row.edge_size)
                        valid_fraction = float(row.template_valid_fraction)

                        patch = da3d.isel(
                            {
                                lat_name: slice(y0, y0 + edge),
                                lon_name: slice(x0, x0 + edge),
                            }
                        )
                        y_np = patch.values.astype(np.float32, copy=False)

                        np.nan_to_num(
                            y_np,
                            copy=False,
                            nan=self.params.nan_fill_value,
                            posinf=self.params.nan_fill_value,
                            neginf=self.params.nan_fill_value,
                        )

                        y_rel_path = Path("y_npy") / f"{kept_count:08d}.npy"
                        np.save(self.params.output_dir / y_rel_path, y_np)

                        rec = row._asdict()
                        rec.update(
                            {
                                "sample_idx": kept_count,
                                "y_npy_path": y_rel_path.as_posix(),
                                "valid_fraction": valid_fraction,
                                "variable": self.params.variable,
                                "bands": "|".join(str(b) for b in self.params.bands),
                                "selected_depth_indices": "|".join(
                                    str(i) for i in selected_idx.tolist()
                                ),
                                "selected_depth_values": "|".join(
                                    f"{float(v):.6f}" for v in selected_depths.tolist()
                                ),
                                "num_channels": int(y_np.shape[0]),
                            }
                        )
                        records.append(rec)
                        kept_count += 1
                        pbar.update(1)

                        if kept_count % self.params.flush_every == 0:
                            pd.DataFrame.from_records(records).to_csv(
                                csv_path, index=False
                            )

        if not records:
            raise RuntimeError("No tensors were saved after indexing.")

        self._assign_split(records)
        pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
        return csv_path

    def _build_full_index(self, nc_files: list[Path]) -> pd.DataFrame:
        """Helper that computes build full index.

        Args:
            nc_files (list[Path]): Input value.

        Returns:
            pd.DataFrame: Computed output value.
        """
        template_rows = self._build_template_rows(nc_files[0])
        template_df = pd.DataFrame.from_records(template_rows)
        per_file_frames = [template_df.assign(source_file=nc_path.name) for nc_path in nc_files]
        full_df = pd.concat(per_file_frames, ignore_index=True)

        cols = [
            "source_file",
            "y0",
            "x0",
            "edge_size",
            "lat0",
            "lat1",
            "lon0",
            "lon1",
            "nodata_fraction",
            "template_valid_fraction",
        ]
        return full_df[cols]

    def _build_template_rows(self, template_path: Path) -> list[dict[str, Any]]:
        """Helper that computes build template rows.

        Args:
            template_path (Path): Path to an input or output file.

        Returns:
            list[dict[str, Any]]: List containing computed outputs.
        """
        rows: list[dict[str, Any]] = []
        edge = int(self.params.edge_size)

        with xr.open_dataset(template_path, engine="h5netcdf", cache=False) as ds:
            da3d, lat_name, lon_name, _, _ = self._extract_depth_stack(ds)
            da2d = da3d.isel({da3d.dims[0]: 0})

            h = int(da2d.sizes[lat_name])
            w = int(da2d.sizes[lon_name])
            if h < edge or w < edge:
                raise RuntimeError(
                    "No patches indexed. edge_size is larger than input dimensions."
                )

            lats = ds[lat_name].values
            lons = ds[lon_name].values

            y_positions = list(range(0, h - edge + 1, edge))
            x_positions = list(range(0, w - edge + 1, edge))
            total_patches = len(y_positions) * len(x_positions)

            with tqdm(total=total_patches, desc="Building nodata fraction index") as pbar:
                for y0 in y_positions:
                    for x0 in x_positions:
                        rec: dict[str, Any] = {
                            "y0": int(y0),
                            "x0": int(x0),
                            "edge_size": edge,
                            "lat0": float(lats[y0]),
                            "lat1": float(lats[y0 + edge - 1]),
                            "lon0": float(lons[x0]),
                            "lon1": float(lons[x0 + edge - 1]),
                            "nodata_fraction": np.nan,
                            "template_valid_fraction": np.nan,
                        }
                        patch3d = da3d.isel(
                            {
                                lat_name: slice(y0, y0 + edge),
                                lon_name: slice(x0, x0 + edge),
                            }
                        )
                        arr3d = patch3d.values.astype(np.float32, copy=False)
                        valid_mask = np.ones(arr3d.shape[1:], dtype=bool)
                        for c in range(arr3d.shape[0]):
                            valid_mask &= self._validity_mask(arr3d[c], da3d)
                        template_valid_fraction = float(valid_mask.mean())
                        rec["template_valid_fraction"] = template_valid_fraction
                        rec["nodata_fraction"] = 1.0 - template_valid_fraction
                        rows.append(rec)
                        pbar.update(1)

        if not rows:
            raise RuntimeError(
                "No patches indexed. Check edge_size and source data dimensions."
            )

        if self.params.enforce_validity:
            rows = [
                r
                for r in rows
                if float(r["nodata_fraction"]) <= self.params.max_nodata_fraction
            ]
            if not rows:
                raise RuntimeError(
                    "All patches were removed by max_nodata_fraction filtering."
                )

        return rows

    def _extract_depth_stack(
        self, ds: xr.Dataset
    ) -> tuple[xr.DataArray, str, str, np.ndarray, np.ndarray]:
        """Helper that computes extract depth stack.

        Args:
            ds (xr.Dataset): Input value.

        Returns:
            tuple[xr.DataArray, str, str, np.ndarray, np.ndarray]: Tuple containing computed outputs.
        """
        if self.params.variable not in ds.data_vars:
            raise RuntimeError(
                f"Expected variable '{self.params.variable}' in dataset. "
                f"Available: {sorted(ds.data_vars)}"
            )

        da = ds[self.params.variable]
        lat_name = self._pick_name(ds, da.dims, ("latitude", "lat"))
        lon_name = self._pick_name(ds, da.dims, ("longitude", "lon"))
        depth_name = self._pick_name(ds, da.dims, ("depth",))
        time_name = self._pick_name(ds, da.dims, ("time",))

        if lat_name is None or lon_name is None or depth_name is None:
            raise RuntimeError(
                f"Could not resolve lat/lon/depth dims for '{self.params.variable}'. "
                f"Found dims: {tuple(da.dims)}"
            )

        if time_name is not None:
            da = da.isel({time_name: 0})

        depth_coord = ds[depth_name] if depth_name in ds.coords else da[depth_name]
        depth_values = np.asarray(depth_coord.values, dtype=np.float64)
        depth_order = np.argsort(depth_values)

        max_rank = int(depth_order.size) - 1
        if any(b < 0 or b > max_rank for b in self.params.bands):
            raise ValueError(
                f"bands contains out-of-range depth indices. "
                f"Valid range is [0, {max_rank}]"
            )

        selected_idx = depth_order[np.asarray(self.params.bands, dtype=np.int64)]
        selected_depths = depth_values[selected_idx]

        da = da.isel({depth_name: selected_idx.tolist()})
        da = da.transpose(depth_name, lat_name, lon_name)
        return da, lat_name, lon_name, selected_idx, selected_depths

    @staticmethod
    def _pick_name(
        ds: xr.Dataset, dims: tuple[str, ...], candidates: tuple[str, ...]
    ) -> str | None:
        """Helper that computes pick name.

        Args:
            ds (xr.Dataset): Input value.
            dims (tuple[str, ...]): Input value.
            candidates (tuple[str, ...]): Input value.

        Returns:
            str | None: Computed output value.
        """
        low_dims = {d.lower(): d for d in dims}
        for cand in candidates:
            if cand in low_dims:
                return low_dims[cand]

        for dim in dims:
            dim_low = dim.lower()
            if any(cand in dim_low for cand in candidates):
                return dim

        low_coords = {c.lower(): c for c in ds.coords}
        for cand in candidates:
            if cand in low_coords:
                return low_coords[cand]

        for coord in ds.coords:
            coord_low = coord.lower()
            if any(cand in coord_low for cand in candidates):
                return coord

        return None

    @staticmethod
    def _fill_values(da: xr.DataArray) -> list[float]:
        """Helper that computes fill values.

        Args:
            da (xr.DataArray): Input value.

        Returns:
            list[float]: List containing computed outputs.
        """
        values: list[float] = []
        if "_FillValue" in da.attrs:
            values.append(float(da.attrs["_FillValue"]))
        if "missing_value" in da.attrs:
            values.append(float(da.attrs["missing_value"]))
        return values

    def _validity_mask(self, arr: np.ndarray, da: xr.DataArray) -> np.ndarray:
        """Helper that computes validity mask.

        Args:
            arr (np.ndarray): Input value.
            da (xr.DataArray): Input value.

        Returns:
            np.ndarray: Computed output value.
        """
        valid = np.isfinite(arr)
        for fill_value in self._fill_values(da):
            valid &= ~np.isclose(arr, fill_value)
        return valid

    def _assign_split(self, records: list[dict[str, Any]]) -> None:
        """Helper that computes assign split.

        Args:
            records (list[dict[str, Any]]): Input value.

        Returns:
            None: No value is returned.
        """
        n_samples = len(records)
        val_len = int(round(n_samples * self.params.val_fraction))
        if n_samples > 1:
            val_len = min(max(val_len, 1 if self.params.val_fraction > 0.0 else 0), n_samples - 1)
        else:
            val_len = 0

        rng = np.random.default_rng(self.params.split_seed)
        val_indices = set(rng.permutation(n_samples)[:val_len].tolist())
        for i, rec in enumerate(records):
            rec["split"] = "val" if i in val_indices else "train"



def to_disk(
    *,
    root_dir: str | Path,
    output_dir: str | Path,
    bands: Sequence[int],
    variable: str = "thetao",
    edge_size: int = 128,
    enforce_validity: bool = True,
    max_nodata_fraction: float = 0.25,
    nan_fill_value: float = 0.0,
    valid_fraction_threshold: float = 0.25,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    flush_every: int = 500,
) -> Path:
    """Export dataset patches and index metadata to disk.

    Args:
        root_dir (str | Path): Input value.
        output_dir (str | Path): Input value.
        bands (Sequence[int]): Input value.
        variable (str): Input value.
        edge_size (int): Input value.
        enforce_validity (bool): Boolean flag controlling behavior.
        max_nodata_fraction (float): Input value.
        nan_fill_value (float): Input value.
        valid_fraction_threshold (float): Input value.
        val_fraction (float): Input value.
        split_seed (int): Input value.
        flush_every (int): Input value.

    Returns:
        Path: Computed output value.
    """
    if not bands:
        raise ValueError("bands must contain at least one depth index.")

    band_tuple = tuple(int(b) for b in bands)
    if len(set(band_tuple)) != len(band_tuple):
        raise ValueError("bands must not contain duplicates.")

    params = _ExportParams(
        root_dir=Path(root_dir),
        output_dir=Path(output_dir),
        variable=str(variable),
        bands=band_tuple,
        edge_size=int(edge_size),
        enforce_validity=bool(enforce_validity),
        max_nodata_fraction=float(max_nodata_fraction),
        nan_fill_value=float(nan_fill_value),
        valid_fraction_threshold=float(np.clip(valid_fraction_threshold, 0.0, 1.0)),
        val_fraction=float(np.clip(val_fraction, 0.0, 1.0)),
        split_seed=int(split_seed),
        flush_every=max(1, int(flush_every)),
    )
    return _NetCDFPatchExporter(params).run()



def _load_config(config_path: str = "configs/data_config.yaml") -> dict[str, Any]:
    """Load and return config data.

    Args:
        config_path (str): Path to an input or output file.

    Returns:
        dict[str, Any]: Dictionary containing computed outputs.
    """
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = _load_config("configs/data_config_eo_4band.yaml")
    ds_cfg = cfg.get("dataset", {})

    csv_path = to_disk(
        root_dir="/data1/datasets/depth/monthy/",
        output_dir="/work/data/depth/4_bands_v2",
        bands=[0, 4, 10, 17],
        variable=str(ds_cfg.get("bands", ["thetao"])[0]),
        edge_size=int(ds_cfg.get("edge_size", 128)),
        enforce_validity=bool(ds_cfg.get("enforce_validity", True)),
        max_nodata_fraction=float(ds_cfg.get("max_nodata_fraction", 0.25)),
        nan_fill_value=float(ds_cfg.get("nan_fill_value", 0.0)),
    )
    print(f"Wrote CSV index: {csv_path}")
