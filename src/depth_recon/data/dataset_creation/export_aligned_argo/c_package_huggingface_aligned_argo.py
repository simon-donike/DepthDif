# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo \
#   --input-zarr /data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr \
#   --output-dir /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh \
#   --zarr-name argo_glors_ostia_ssh.zarr \
#   --file-mode hardlink \
#   --overwrite
"""Package an enriched ARGO profile Zarr as a Hugging Face dataset folder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

DEFAULT_INPUT_ZARR = Path(
    "/data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh"
)
DEFAULT_ZARR_NAME = "argo_glors_ostia_ssh.zarr"
DEFAULT_DATASET_SLUG = "argo-glors-ostia-ssh"
DEFAULT_DATA_SUBDIR = Path("data")

PROFILE_SCALAR_COLUMNS = (
    "profile_source_file",
    "profile_idx",
    "profile_date",
    "profile_juld",
    "latitude",
    "longitude",
    "valid_observed_depth_count",
    "glorys_temporal_status",
    "ostia_temporal_status",
    "sealevel_temporal_status",
    "sss_temporal_status",
)
PROFILE_VALID_MASKS = {
    "argo_temp_valid_on_glorys_depth": "argo_temp_valid_depth_count",
    "argo_potm_valid_on_glorys_depth": "argo_potm_valid_depth_count",
    "argo_psal_valid_on_glorys_depth": "argo_psal_valid_depth_count",
}


def _json_safe(value: Any) -> Any:
    """Convert numpy/path values into JSON/YAML-safe Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _reset_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create an empty package directory."""
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite."
            )
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _stage_file(source: Path, target: Path, *, file_mode: str) -> None:
    """Stage one file by hardlinking when possible or copying when requested."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        # Hardlinks can fail across filesystems; keep the command usable by copying.
        shutil.copy2(source, target)


def _stage_zarr_tree(source_zarr: Path, target_zarr: Path, *, file_mode: str) -> None:
    """Stage the complete Zarr directory tree into the HF package."""
    if not source_zarr.exists():
        raise FileNotFoundError(f"Input Zarr does not exist: {source_zarr}")
    if not source_zarr.is_dir():
        raise NotADirectoryError(f"Input Zarr must be a directory store: {source_zarr}")
    if target_zarr.exists():
        raise FileExistsError(f"Target Zarr already exists: {target_zarr}")

    target_zarr.mkdir(parents=True, exist_ok=False)
    for source_path in source_zarr.rglob("*"):
        relative = source_path.relative_to(source_zarr)
        target_path = target_zarr / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            _stage_file(source_path, target_path, file_mode=file_mode)


def _as_1d_values(ds: xr.Dataset, name: str, profile_size: int) -> np.ndarray:
    """Read one profile-length variable into memory for Parquet export."""
    values = np.asarray(ds[name].values).reshape(-1)
    if values.size != int(profile_size):
        raise RuntimeError(f"{name} is not profile-length in the input Zarr.")
    if values.dtype.kind in {"S", "U", "O"}:
        return values.astype(str)
    return values


def _write_profiles_index(ds: xr.Dataset, output_path: Path) -> pd.DataFrame:
    """Write a lightweight one-row-per-profile Parquet index."""
    profile_size = int(ds.sizes.get("profile", 0))
    if "profile" in ds.coords:
        profile_values = np.asarray(ds["profile"].values, dtype=np.int64).reshape(-1)
    else:
        profile_values = np.arange(profile_size, dtype=np.int64)
    profile_df = pd.DataFrame({"profile": profile_values})

    for name in PROFILE_SCALAR_COLUMNS:
        if name in ds and ds[name].dims == ("profile",):
            profile_df[name] = _as_1d_values(ds, name, profile_size)

    for source_name, output_name in PROFILE_VALID_MASKS.items():
        if source_name not in ds:
            continue
        dims = ds[source_name].dims
        if "profile" not in dims:
            continue
        depth_dims = [dim for dim in dims if dim != "profile"]
        if len(depth_dims) != 1:
            continue
        counts = ds[source_name].sum(dim=depth_dims[0]).values
        profile_df[output_name] = np.asarray(counts, dtype=np.int16).reshape(-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_parquet(output_path, index=False)
    return profile_df


def _array_kind(ds: xr.Dataset, name: str) -> str:
    """Return whether a Zarr array is a coordinate or data variable."""
    if name in ds.coords:
        return "coordinate"
    return "data_variable"


def _write_variables_index(
    ds: xr.Dataset,
    output_path: Path,
    *,
    zarr_relative_path: Path,
) -> pd.DataFrame:
    """Write variable-level metadata for the packaged Zarr."""
    records: list[dict[str, Any]] = []
    names = list(ds.coords) + [name for name in ds.data_vars if name not in ds.coords]
    for name in names:
        array = ds[name]
        attrs = dict(array.attrs)
        records.append(
            {
                "name": name,
                "kind": _array_kind(ds, name),
                "dims": ",".join(str(dim) for dim in array.dims),
                "shape": json.dumps([int(size) for size in array.shape]),
                "dtype": str(array.dtype),
                "units": attrs.get("units", attrs.get("value_units")),
                "long_name": attrs.get("long_name"),
                "standard_name": attrs.get("standard_name"),
                "source_product": attrs.get("source_product"),
                "source_variable": attrs.get("source_variable"),
                "description": attrs.get("description"),
                "zarr_array_path": (zarr_relative_path / name).as_posix(),
            }
        )

    variables_df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables_df.to_parquet(output_path, index=False)
    return variables_df


def _yyyymmdd_to_iso(value: int | None) -> str | None:
    """Convert an integer YYYYMMDD date to an ISO date string."""
    if value is None:
        return None
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _date_bounds(profile_df: pd.DataFrame) -> tuple[int | None, int | None]:
    """Return profile-date min/max if that column is available."""
    if "profile_date" not in profile_df or profile_df.empty:
        return None, None
    dates = pd.to_numeric(profile_df["profile_date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return int(dates.min()), int(dates.max())


def _bbox(profile_df: pd.DataFrame) -> list[float]:
    """Return the geographic bounding box for indexed profiles."""
    if profile_df.empty or not {"longitude", "latitude"}.issubset(profile_df.columns):
        return [-180.0, -90.0, 180.0, 90.0]
    lon = pd.to_numeric(profile_df["longitude"], errors="coerce").dropna()
    lat = pd.to_numeric(profile_df["latitude"], errors="coerce").dropna()
    if lon.empty or lat.empty:
        return [-180.0, -90.0, 180.0, 90.0]
    return [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]


def _write_readme(
    output_dir: Path,
    *,
    dataset_slug: str,
    zarr_relative_path: Path,
    profile_df: pd.DataFrame,
) -> None:
    """Write the Hugging Face dataset card."""
    start_date, end_date = _date_bounds(profile_df)
    card_metadata = {
        "license": "other",
        "pretty_name": "DepthDif aligned ARGO profile collocation dataset",
        "tags": [
            "oceanography",
            "argo",
            "glorys",
            "ostia",
            "sea-level",
            "sea-surface-salinity",
            "zarr",
        ],
        "configs": [
            {
                "config_name": "profile-index",
                "data_files": [
                    {"split": "profiles", "path": "indices/profiles.parquet"},
                    {"split": "variables", "path": "indices/variables.parquet"},
                ],
            }
        ],
    }
    body = f"""# {dataset_slug}

This dataset package contains the DepthDif enriched ARGO profile Zarr store and
lightweight Parquet indices for Hugging Face preview/search workflows.

Main data:

```python
import xarray as xr

ds = xr.open_zarr("{zarr_relative_path.as_posix()}", consolidated=None)
```

Profile index:

```python
import pandas as pd

profiles = pd.read_parquet("indices/profiles.parquet")
```

The Zarr schema is unchanged from
`depth_recon.data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles`.
GeoTIFF dataset creation can consume this packaged Zarr directly by passing:

```bash
--enriched-argo-zarr {zarr_relative_path.as_posix()}
```

Coverage:

- Profiles: {len(profile_df)}
- Profile date range: {_yyyymmdd_to_iso(start_date)} to {_yyyymmdd_to_iso(end_date)}

The package collocates EN4/ARGO profiles with GLORYS, OSTIA, sea-level, and SSS
source fields. Upstream product licenses and citation requirements still apply.
"""
    readme = f"---\n{yaml.safe_dump(card_metadata, sort_keys=False)}---\n\n{body}"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_examples(output_dir: Path, *, zarr_relative_path: Path) -> None:
    """Write small usage examples into the package."""
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    zarr_text = zarr_relative_path.as_posix()
    (examples_dir / "open_with_xarray.py").write_text(
        f"""from pathlib import Path

import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
ds = xr.open_zarr(ROOT / "{zarr_text}", consolidated=None)
print(ds)
""",
        encoding="utf-8",
    )
    (examples_dir / "subset_by_region_time.py").write_text(
        f"""from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
profiles = pd.read_parquet(ROOT / "indices/profiles.parquet")
subset = profiles[
    (profiles["profile_date"] >= 20100101)
    & (profiles["profile_date"] <= 20101231)
    & (profiles["latitude"].between(30.0, 46.0))
    & (profiles["longitude"].between(-6.0, 37.0))
]

ds = xr.open_zarr(ROOT / "{zarr_text}", consolidated=None)
subset_ds = ds.sel(profile=subset["profile"].to_numpy())
print(subset_ds)
""",
        encoding="utf-8",
    )


def _write_metadata_files(
    output_dir: Path,
    *,
    ds: xr.Dataset,
    profile_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    dataset_slug: str,
    zarr_relative_path: Path,
) -> None:
    """Write JSON/CFF/STAC metadata sidecars."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    start_date, end_date = _date_bounds(profile_df)
    bbox = _bbox(profile_df)
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    description = {
        "name": dataset_slug,
        "created_utc": created_utc,
        "zarr_path": zarr_relative_path.as_posix(),
        "profile_count": int(ds.sizes.get("profile", 0)),
        "glorys_depth_count": int(ds.sizes.get("glorys_depth", 0)),
        "profile_date_range": {
            "start": start_date,
            "end": end_date,
            "start_iso": _yyyymmdd_to_iso(start_date),
            "end_iso": _yyyymmdd_to_iso(end_date),
        },
        "bbox": bbox,
        "variables": variables_df["name"].tolist(),
        "zarr_attrs": _json_safe(dict(ds.attrs)),
    }
    (metadata_dir / "dataset_description.json").write_text(
        json.dumps(_json_safe(description), indent=2) + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "citation.cff").write_text(
        """cff-version: 1.2.0
message: "If you use this dataset, cite DepthDif and the upstream EN4/ARGO, GLORYS, OSTIA, sea-level, and SSS products."
title: "DepthDif aligned ARGO profile collocation dataset"
authors:
  - family-names: "DepthDif contributors"
license: "other"
""",
        encoding="utf-8",
    )
    stac_item = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": dataset_slug,
        "bbox": bbox,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]
            ],
        },
        "properties": {
            "datetime": None,
            "start_datetime": _yyyymmdd_to_iso(start_date),
            "end_datetime": _yyyymmdd_to_iso(end_date),
            "created": created_utc,
        },
        "assets": {
            "zarr": {
                "href": zarr_relative_path.as_posix(),
                "type": "application/vnd+zarr",
                "title": "Aligned ARGO profile Zarr",
            },
            "profiles": {
                "href": "indices/profiles.parquet",
                "type": "application/x-parquet",
                "title": "Profile index",
            },
            "variables": {
                "href": "indices/variables.parquet",
                "type": "application/x-parquet",
                "title": "Variable index",
            },
        },
    }
    (metadata_dir / "stac-item.json").write_text(
        json.dumps(_json_safe(stac_item), indent=2) + "\n",
        encoding="utf-8",
    )


def _write_license(output_dir: Path) -> None:
    """Write the package license note."""
    (output_dir / "LICENSE").write_text(
        """This package aggregates collocated oceanographic data derived from upstream public products.

Use is subject to the terms and citation requirements of the upstream EN4/ARGO,
GLORYS, OSTIA, sea-level, and sea-surface-salinity products. This file is a
dataset license notice, not a replacement for upstream product licenses.
""",
        encoding="utf-8",
    )


def build_huggingface_aligned_argo_package(
    *,
    input_zarr: str | Path = DEFAULT_INPUT_ZARR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    zarr_name: str = DEFAULT_ZARR_NAME,
    file_mode: str = "hardlink",
    overwrite: bool = False,
) -> Path:
    """Build a Hugging Face-style folder around an enriched ARGO Zarr store."""
    input_zarr = Path(input_zarr)
    output_dir = Path(output_dir)
    if file_mode not in {"hardlink", "copy"}:
        raise ValueError("file_mode must be one of: hardlink, copy")

    _reset_output_dir(output_dir, overwrite=overwrite)
    zarr_relative_path = DEFAULT_DATA_SUBDIR / str(zarr_name)
    target_zarr = output_dir / zarr_relative_path
    _stage_zarr_tree(input_zarr, target_zarr, file_mode=file_mode)

    ds = xr.open_zarr(target_zarr, consolidated=None)
    try:
        profile_df = _write_profiles_index(ds, output_dir / "indices/profiles.parquet")
        variables_df = _write_variables_index(
            ds,
            output_dir / "indices/variables.parquet",
            zarr_relative_path=zarr_relative_path,
        )
        _write_readme(
            output_dir,
            dataset_slug=DEFAULT_DATASET_SLUG,
            zarr_relative_path=zarr_relative_path,
            profile_df=profile_df,
        )
        _write_examples(output_dir, zarr_relative_path=zarr_relative_path)
        _write_metadata_files(
            output_dir,
            ds=ds,
            profile_df=profile_df,
            variables_df=variables_df,
            dataset_slug=DEFAULT_DATASET_SLUG,
            zarr_relative_path=zarr_relative_path,
        )
        _write_license(output_dir)
    finally:
        ds.close()

    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for HF ARGO package creation."""
    parser = argparse.ArgumentParser(
        description="Package an enriched ARGO profile Zarr as a Hugging Face dataset folder."
    )
    parser.add_argument("--input-zarr", type=Path, default=DEFAULT_INPUT_ZARR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zarr-name", default=DEFAULT_ZARR_NAME)
    parser.add_argument(
        "--file-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hardlinks for large Zarr files when possible, or copy bytes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run Hugging Face package creation from the command line."""
    args = _build_parser().parse_args()
    output_dir = build_huggingface_aligned_argo_package(
        input_zarr=args.input_zarr,
        output_dir=args.output_dir,
        zarr_name=args.zarr_name,
        file_mode=args.file_mode,
        overwrite=args.overwrite,
    )
    print(f"Wrote Hugging Face aligned ARGO package: {output_dir}")


if __name__ == "__main__":
    main()
