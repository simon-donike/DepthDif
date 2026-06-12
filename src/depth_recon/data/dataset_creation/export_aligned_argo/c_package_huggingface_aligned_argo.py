# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo \
#   --input-zarr /work/data/depthdif/enriched_argo_profiles.zarr \
#   --raster-root /work/data/depthdif/rasters \
#   --compact-argo-zarr /work/data/depthdif/argo/argo_profiles_on_grid.zarr \
#   --manifest-path /work/data/depthdif/manifest.yaml \
#   --masks-dir /work/data/depthdif/masks \
#   --output-dir /work/data/OceanVariableReconstruction \
#   --zarr-name argo_glors_ostia_ssh.zarr \
#   --file-mode copy \
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

DEFAULT_GEOTIFF_ROOT = Path("/work/data/depthdif")
DEFAULT_INPUT_ZARR = DEFAULT_GEOTIFF_ROOT / "enriched_argo_profiles.zarr"
DEFAULT_RASTER_ROOT = DEFAULT_GEOTIFF_ROOT / "rasters"
DEFAULT_COMPACT_ARGO_ZARR = DEFAULT_GEOTIFF_ROOT / "argo" / "argo_profiles_on_grid.zarr"
DEFAULT_MANIFEST_PATH = DEFAULT_GEOTIFF_ROOT / "manifest.yaml"
DEFAULT_MASKS_DIR = DEFAULT_GEOTIFF_ROOT / "masks"
DEFAULT_OUTPUT_DIR = Path("/work/data/OceanVariableReconstruction")
DEFAULT_ZARR_NAME = "argo_glors_ostia_ssh.zarr"
DEFAULT_DATASET_SLUG = "OceanVariableReconstruction"
DEFAULT_DATA_SUBDIR = Path("data")
REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_CARD_ASSETS = (
    (
        Path("docs/assets/figures/depthdif_schema.webp"),
        Path("assets/figures/depthdif_schema.webp"),
    ),
    (
        Path("docs/assets/data/geotiff_dataset_random100_surface.webp"),
        Path("assets/data/geotiff_dataset_random100_surface.webp"),
    ),
    (
        Path("docs/assets/data/argo_on_glorys_grid_3D.gif"),
        Path("assets/data/argo_on_glorys_grid_3D.gif"),
    ),
    (
        Path("docs/assets/data/profile_comparison_good_alignment.webp"),
        Path("assets/data/profile_comparison_good_alignment.webp"),
    ),
    (
        Path("docs/assets/data/profile_comparison_bad_alignment.webp"),
        Path("assets/data/profile_comparison_bad_alignment.webp"),
    ),
)

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


def _stage_directory_tree(
    source_dir: Path,
    target_dir: Path,
    *,
    file_mode: str,
) -> None:
    """Stage a complete directory tree into the package root."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {source_dir}")
    if target_dir.exists():
        raise FileExistsError(f"Target directory already exists: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=False)
    for source_path in source_dir.rglob("*"):
        relative = source_path.relative_to(source_dir)
        target_path = target_dir / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            _stage_file(source_path, target_path, file_mode=file_mode)


def _stage_manifest_file(source: Path, target: Path, *, output_dir: Path) -> None:
    """Stage a manifest and point its output_dir at the package root."""
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        shutil.copy2(source, target)
        return
    manifest["output_dir"] = output_dir.as_posix()
    target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _stage_optional_geotiff_assets(
    *,
    output_dir: Path,
    raster_root: Path | None,
    compact_argo_zarr: Path | None,
    manifest_path: Path | None,
    masks_dir: Path | None,
    file_mode: str,
) -> bool:
    """Stage optional GeoTIFF training-store assets into the upload root."""
    staged = False
    if raster_root is not None:
        _stage_directory_tree(
            Path(raster_root),
            output_dir / "rasters",
            file_mode=file_mode,
        )
        staged = True
    if compact_argo_zarr is not None:
        _stage_zarr_tree(
            Path(compact_argo_zarr),
            output_dir / "argo" / "argo_profiles_on_grid.zarr",
            file_mode=file_mode,
        )
        staged = True
    if manifest_path is not None:
        _stage_manifest_file(
            Path(manifest_path),
            output_dir / "manifest.yaml",
            output_dir=output_dir,
        )
        staged = True
    if masks_dir is not None:
        _stage_directory_tree(
            Path(masks_dir), output_dir / "masks", file_mode=file_mode
        )
        staged = True
    return staged


def _stage_dataset_card_assets(output_dir: Path, *, file_mode: str) -> list[Path]:
    """Stage README illustration assets into the upload package."""
    staged_assets: list[Path] = []
    for source_relative_path, target_relative_path in DATASET_CARD_ASSETS:
        source_path = REPO_ROOT / source_relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset card asset does not exist: {source_path}")
        _stage_file(source_path, output_dir / target_relative_path, file_mode=file_mode)
        staged_assets.append(target_relative_path)
    return staged_assets


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
    include_geotiff_assets: bool,
    dataset_card_assets: list[Path],
) -> None:
    """Write the Hugging Face dataset card."""
    start_date, end_date = _date_bounds(profile_df)
    assets = {path.as_posix() for path in dataset_card_assets}
    schema_asset = "assets/figures/depthdif_schema.webp"
    raster_asset = "assets/data/geotiff_dataset_random100_surface.webp"
    argo_grid_asset = "assets/data/argo_on_glorys_grid_3D.gif"
    good_alignment_asset = "assets/data/profile_comparison_good_alignment.webp"
    bad_alignment_asset = "assets/data/profile_comparison_bad_alignment.webp"
    overview_section = ""
    if schema_asset in assets:
        overview_section = f"""
## Dataset Overview

<p align="center">
  <img src="{schema_asset}" width="85%" alt="DepthDif dataset and model overview" />
</p>
"""
    raster_example_section = ""
    if include_geotiff_assets and raster_asset in assets:
        raster_example_section = f"""
## Raster Example

Representative surface-level training patches from the exported GeoTIFF store:

<p align="center">
  <img src="{raster_asset}" width="85%" alt="Random surface-level training dataset patches" />
</p>
"""
    alignment_images: list[str] = []
    if argo_grid_asset in assets:
        alignment_images.append(f"""<p align="center">
  <img src="{argo_grid_asset}" width="70%" alt="Depth-aligned ARGO values on the GLORYS grid" />
</p>""")
    if good_alignment_asset in assets:
        alignment_images.append(f"""<p align="center">
  <img src="{good_alignment_asset}" width="72%" alt="Example of good EN4-to-GLORYS temperature and salinity profile alignment" />
</p>""")
    if bad_alignment_asset in assets:
        alignment_images.append(f"""<p align="center">
  <img src="{bad_alignment_asset}" width="72%" alt="Example of weaker ARGO-to-GLORYS profile alignment" />
</p>""")
    alignment_section = ""
    if alignment_images:
        alignment_section = f"""
## EN4 Alignment Examples

EN4 profiles are projected onto the fixed 50-level GLORYS depth axis before
spatial rasterization. The examples below show the grid-indexed EN4
representation and profile-level alignment quality.

{chr(10).join(alignment_images)}
"""
    tags = [
        "oceanography",
        "argo",
        "glorys",
        "ostia",
        "sea-level",
        "sea-surface-salinity",
        "zarr",
    ]
    if include_geotiff_assets:
        tags.append("geotiff")
    pretty_name = "DepthDif aligned ARGO profile collocation dataset"
    if include_geotiff_assets:
        pretty_name = "DepthDif GeoTIFF raster and aligned ARGO dataset"
    card_metadata = {
        "license": "other",
        "pretty_name": pretty_name,
        "tags": tags,
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
    if include_geotiff_assets:
        body = f"""# DepthDif GeoTIFF Raster and Aligned ARGO Dataset

This dataset package contains the model-ready DepthDif GeoTIFF raster store and
the enriched ARGO profile Zarr used to create it.
{overview_section}

## Layout

```text
assets/
  figures/depthdif_schema.webp
  data/geotiff_dataset_random100_surface.webp
  data/argo_on_glorys_grid_3D.gif
  data/profile_comparison_good_alignment.webp
  data/profile_comparison_bad_alignment.webp
rasters/
  glorys/thetao/
  glorys/so/
  ostia/analysed_sst/
  sealevel/adt/
  sss/sos/
  sss/dos/
argo/
  argo_profiles_on_grid.zarr/
data/
  {zarr_relative_path.name}/
indices/
  profiles.parquet
  variables.parquet
metadata/
  dataset_description.json
  citation.cff
  stac-item.json
examples/
  open_with_xarray.py
  subset_by_region_time.py
manifest.yaml
masks/
```

The `rasters/` directory is intentionally at the repository root. It contains
the aligned uint8 GeoTIFF products used by the pixel-space dataloader. The
compact `argo/argo_profiles_on_grid.zarr` store is the grid-indexed EN4 input
used by that dataloader.
{raster_example_section}

## Raster Products

All GeoTIFF rasters are exported on the GLORYS 0.1 degree global grid
(`EPSG:4326`, 3600 x 1800 pixels, west-to-east longitudes from -180 to 180 and
north-to-south latitudes from 90 to -90). Files are named
`<variable>_YYYYMMDD.tif` and use the weekly GLORYS target dates.

The GLORYS variables are depth-resolved 50-band GeoTIFFs:

- `rasters/glorys/thetao/`: potential temperature, encoded as Kelvin.
- `rasters/glorys/so/`: salinity, encoded as PSU.

The surface products are single-band GeoTIFFs aggregated to the same weekly
target dates with a centered 7-day mean window:

- `rasters/ostia/analysed_sst/`: OSTIA analysed sea-surface temperature in Kelvin.
- `rasters/sealevel/adt/`: absolute dynamic topography in meters.
- `rasters/sss/sos/`: sea-surface salinity in PSU.
- `rasters/sss/dos/`: sea-surface density in kg/m3.

Raster pixels are stored as `uint8` with `255` reserved for nodata. Valid codes
`0..254` are linearly decoded using the stretch ranges in `manifest.yaml`;
per-file statistics, source filenames, compression, target dates, and the full
depth axis are also recorded there.

The full enriched profile-level ARGO collocation dataset is available at:

```python
import xarray as xr

ds = xr.open_zarr("{zarr_relative_path.as_posix()}", consolidated=None)
```

The lightweight Parquet indices are included for preview and filtering:

```python
import pandas as pd

profiles = pd.read_parquet("indices/profiles.parquet")
variables = pd.read_parquet("indices/variables.parquet")
```

Coverage:

- Profiles: {len(profile_df)}
- Enriched ARGO profile date range: {_yyyymmdd_to_iso(start_date)} to {_yyyymmdd_to_iso(end_date)}
- GLORYS depth levels: 50

Upstream product licenses and citation requirements for EN4/ARGO, GLORYS,
OSTIA, sea-level, and sea-surface-salinity products still apply.
{alignment_section}
"""
    else:
        body = f"""# {dataset_slug}

This dataset package contains the DepthDif enriched ARGO profile Zarr store and
lightweight Parquet indices for Hugging Face preview/search workflows.
{overview_section}

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
{alignment_section}
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
    include_geotiff_assets: bool,
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
        "includes_geotiff_assets": bool(include_geotiff_assets),
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
    if include_geotiff_assets:
        stac_item["assets"].update(
            {
                "rasters": {
                    "href": "rasters/",
                    "type": "image/tiff; application=geotiff",
                    "title": "Root-level GeoTIFF raster store",
                },
                "compact_argo": {
                    "href": "argo/argo_profiles_on_grid.zarr",
                    "type": "application/vnd+zarr",
                    "title": "Compact grid-indexed EN4 Zarr",
                },
                "manifest": {
                    "href": "manifest.yaml",
                    "type": "application/x-yaml",
                    "title": "GeoTIFF export manifest",
                },
            }
        )
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


def _write_gitignore(output_dir: Path) -> None:
    """Write local-only ignore rules for the Hugging Face dataset repo."""
    (output_dir / ".gitignore").write_text(
        """# Local upload/cache state.
.cache/

# Synthetic pretraining rasters are generated locally and are not part of the public dataset.
rasters/synthetic/
manifest.yaml.synthetic_backup_*
""",
        encoding="utf-8",
    )


def build_huggingface_aligned_argo_package(
    *,
    input_zarr: str | Path = DEFAULT_INPUT_ZARR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    zarr_name: str = DEFAULT_ZARR_NAME,
    file_mode: str = "copy",
    overwrite: bool = False,
    raster_root: str | Path | None = None,
    compact_argo_zarr: str | Path | None = None,
    manifest_path: str | Path | None = None,
    masks_dir: str | Path | None = None,
) -> Path:
    """Build a Hugging Face-style folder around DepthDif dataset outputs."""
    input_zarr = Path(input_zarr)
    output_dir = Path(output_dir)
    if file_mode not in {"hardlink", "copy"}:
        raise ValueError("file_mode must be one of: hardlink, copy")

    _reset_output_dir(output_dir, overwrite=overwrite)
    zarr_relative_path = DEFAULT_DATA_SUBDIR / str(zarr_name)
    target_zarr = output_dir / zarr_relative_path
    _stage_zarr_tree(input_zarr, target_zarr, file_mode=file_mode)
    include_geotiff_assets = _stage_optional_geotiff_assets(
        output_dir=output_dir,
        raster_root=None if raster_root is None else Path(raster_root),
        compact_argo_zarr=(
            None if compact_argo_zarr is None else Path(compact_argo_zarr)
        ),
        manifest_path=None if manifest_path is None else Path(manifest_path),
        masks_dir=None if masks_dir is None else Path(masks_dir),
        file_mode=file_mode,
    )

    ds = xr.open_zarr(target_zarr, consolidated=None)
    try:
        profile_df = _write_profiles_index(ds, output_dir / "indices/profiles.parquet")
        variables_df = _write_variables_index(
            ds,
            output_dir / "indices/variables.parquet",
            zarr_relative_path=zarr_relative_path,
        )
        dataset_card_assets = _stage_dataset_card_assets(
            output_dir, file_mode=file_mode
        )
        _write_readme(
            output_dir,
            dataset_slug=DEFAULT_DATASET_SLUG,
            zarr_relative_path=zarr_relative_path,
            profile_df=profile_df,
            include_geotiff_assets=include_geotiff_assets,
            dataset_card_assets=dataset_card_assets,
        )
        _write_examples(output_dir, zarr_relative_path=zarr_relative_path)
        _write_metadata_files(
            output_dir,
            ds=ds,
            profile_df=profile_df,
            variables_df=variables_df,
            dataset_slug=DEFAULT_DATASET_SLUG,
            zarr_relative_path=zarr_relative_path,
            include_geotiff_assets=include_geotiff_assets,
        )
        _write_license(output_dir)
        _write_gitignore(output_dir)
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
    parser.add_argument("--raster-root", type=Path, default=DEFAULT_RASTER_ROOT)
    parser.add_argument(
        "--compact-argo-zarr",
        type=Path,
        default=DEFAULT_COMPACT_ARGO_ZARR,
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument(
        "--no-geotiff-assets",
        action="store_true",
        help="Package only the enriched ARGO Zarr and HF metadata.",
    )
    parser.add_argument(
        "--file-mode",
        choices=("hardlink", "copy"),
        default="copy",
        help="Copy bytes by default; hardlink is available for local staging only.",
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
        raster_root=None if args.no_geotiff_assets else args.raster_root,
        compact_argo_zarr=(None if args.no_geotiff_assets else args.compact_argo_zarr),
        manifest_path=None if args.no_geotiff_assets else args.manifest_path,
        masks_dir=None if args.no_geotiff_assets else args.masks_dir,
    )
    print(f"Wrote Hugging Face aligned ARGO package: {output_dir}")


if __name__ == "__main__":
    main()
