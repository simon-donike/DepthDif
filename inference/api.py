# Example:
# /work/envs/depth/bin/python -m inference.api infer-week --year 2015 --iso-week 25 --rectangle -20 30 10 50 --device cuda --config-repo simon-donike/DepthDif
"""Public inference API for PyPI and notebook usage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from typing import Callable, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import yaml

from data.dataset_argo_netcdf_gridded import (
    DEFAULT_LAND_MASK_PATH,
    ArgoNetCDFStore,
    PatchAxes,
    TimedNetCDFStore,
    _build_land_mask_patch_table,
    _center_lon_deg,
    _GridParams,
    _normalize_lon,
    _parse_force_include_regions,
)
from inference.core import (
    build_model,
    choose_device,
    load_yaml,
    resolve_checkpoint_path,
)
from inference.export_global import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAIN_CONFIG,
    ExportRunResult,
    ExportSelection,
    GeoJSONPointWriter,
    _accumulate_patch_into_arrays,
    _argo_point_features_for_patch,
    _build_parser as _build_export_parser,
    _cleanup_accumulator,
    _flush_accumulator,
    _summary_artifact_path,
    _to_device,
    build_mosaic_layout,
    create_raster_accumulator,
    filter_selection_by_rectangle,
    load_land_mask_for_layout,
    resolve_depth_export_levels,
    run_global_inference,
    write_global_top_band_geotiff,
)
from utils.normalizations import temperature_normalize

DEFAULT_HF_REPO_ID = "simon-donike/DepthDif"
DEFAULT_HF_REVISION = "main"
DEFAULT_HF_MODEL_CONFIG = "model_config.yaml"
DEFAULT_HF_DATA_CONFIG = DEFAULT_DATA_CONFIG
DEFAULT_HF_TRAIN_CONFIG = DEFAULT_TRAIN_CONFIG
DEFAULT_HF_CHECKPOINT = "depthdif_v1.ckpt"
DEFAULT_HF_LAND_MASK = DEFAULT_LAND_MASK_PATH
DEFAULT_EN4_BASE_URL = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1"
DEFAULT_OSTIA_DATASET_CANDIDATES = (
    "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    "METOFFICE-GLO-SST-L4-REP-OBS-SST-V2",
    "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
)
DEFAULT_GLORYS_DEPTH_AXIS_M = (
    0.494,
    1.541,
    2.646,
    3.819,
    5.078,
    6.441,
    7.93,
    9.573,
    11.405,
    13.468,
    15.81,
    18.495,
    21.599,
    25.211,
    29.444,
    34.434,
    40.344,
    47.373,
    55.764,
    65.808,
    77.853,
    92.326,
    109.729,
    130.666,
    155.851,
    186.126,
    222.475,
    266.04,
    318.127,
    380.213,
    453.938,
    541.089,
    643.567,
    763.333,
    902.339,
    1062.44,
    1245.291,
    1452.251,
    1684.284,
    1941.893,
    2225.078,
    2533.336,
    2865.703,
    3220.82,
    3597.032,
    3992.484,
    4405.224,
    4833.291,
    5274.784,
    5727.917,
)

Downloader = Callable[[str, Path], Path]
CommandRunner = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class InferenceAssets:
    """Local paths to model/config artifacts used by public inference."""

    model_config: Path
    data_config: Path
    train_config: Path
    checkpoint: Path


def _default_cache_dir() -> Path:
    """Return the default DepthDif artifact cache directory."""
    return Path.home() / ".cache" / "depthdif"


def _download_url(url: str, output_path: Path) -> Path:
    """Download one URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def _hf_url(repo_id: str, revision: str, repo_path: str) -> str:
    """Build a Hugging Face resolve URL for one repository artifact."""
    clean_path = str(repo_path).lstrip("/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{clean_path}"


def _resolve_hf_file(
    *,
    repo_id: str,
    revision: str,
    repo_path: str,
    cache_dir: Path,
    downloader: Downloader,
    force_download: bool,
) -> Path:
    """Resolve one Hugging Face artifact into the local cache."""
    local_path = cache_dir / repo_id.replace("/", "--") / revision / repo_path
    if local_path.exists() and not force_download:
        return local_path
    return downloader(_hf_url(repo_id, revision, repo_path), local_path)


def resolve_hf_assets(
    *,
    config_repo: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    model_config_path: str = DEFAULT_HF_MODEL_CONFIG,
    data_config_path: str = DEFAULT_HF_DATA_CONFIG,
    train_config_path: str = DEFAULT_HF_TRAIN_CONFIG,
    checkpoint_path: str = DEFAULT_HF_CHECKPOINT,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> InferenceAssets:
    """Download or reuse configs and checkpoint from Hugging Face."""
    target_cache = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    fetch = _download_url if downloader is None else downloader
    return InferenceAssets(
        model_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=model_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        data_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=data_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        train_config=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=train_config_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
        checkpoint=_resolve_hf_file(
            repo_id=config_repo,
            revision=revision,
            repo_path=checkpoint_path,
            cache_dir=target_cache,
            downloader=fetch,
            force_download=force_download,
        ),
    )


def resolve_hf_land_mask(
    *,
    config_repo: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    land_mask_path: str = DEFAULT_HF_LAND_MASK,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Download or reuse the public land-mask GeoTIFF from Hugging Face."""
    target_cache = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    fetch = _download_url if downloader is None else downloader
    return _resolve_hf_file(
        repo_id=config_repo,
        revision=revision,
        repo_path=land_mask_path,
        cache_dir=target_cache,
        downloader=fetch,
        force_download=force_download,
    )


def _week_months(
    year: int, iso_week: int, radius_days: int = 3
) -> set[tuple[int, int]]:
    """Return calendar year/month pairs touched by an ISO-week window."""
    center = date.fromisocalendar(int(year), int(iso_week), 3)
    return {
        (
            (center + timedelta(days=offset)).year,
            (center + timedelta(days=offset)).month,
        )
        for offset in range(-int(radius_days), int(radius_days) + 1)
    }


def _en4_zip_url(base_url: str, year: int) -> str:
    """Return the EN4 yearly profile ZIP URL for one calendar year."""
    return f"{base_url.rstrip('/')}/EN.4.2.2.profiles.g10.{int(year)}.zip"


def _extract_matching_en4_months(
    *,
    zip_path: Path,
    output_dir: Path,
    months: set[tuple[int, int]],
) -> list[Path]:
    """Extract EN4 monthly NetCDF files matching requested months."""
    extracted: list[Path] = []
    month_tokens = {f"{year:04d}{month:02d}" for year, month in months}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not name.endswith(".nc"):
                continue
            if not any(f".{token}." in name for token in month_tokens):
                continue
            target_path = output_dir / name
            if not target_path.exists():
                # Extract through ZipFile.open to avoid trusting archive paths.
                with archive.open(member) as src, target_path.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(target_path)
    return extracted


def download_argo_for_week(
    year: int,
    iso_week: int,
    output_dir: str | Path,
    *,
    base_url: str = DEFAULT_EN4_BASE_URL,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Download and extract EN4/ARGO profile files needed for one ISO week."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    months = _week_months(int(year), int(iso_week))
    missing_months = {
        (month_year, month)
        for month_year, month in months
        if not list(output_path.glob(f"*{month_year:04d}{month:02d}*.nc"))
    }
    if not missing_months:
        return output_path

    archive_cache = (
        _default_cache_dir() / "en4" if cache_dir is None else Path(cache_dir)
    )
    archive_cache.mkdir(parents=True, exist_ok=True)
    fetch = _download_url if downloader is None else downloader

    for archive_year in sorted({year_value for year_value, _ in missing_months}):
        zip_path = archive_cache / f"EN.4.2.2.profiles.g10.{archive_year}.zip"
        if force_download or not zip_path.exists():
            fetch(_en4_zip_url(base_url, archive_year), zip_path)
        extracted = _extract_matching_en4_months(
            zip_path=zip_path,
            output_dir=output_path,
            months={item for item in missing_months if item[0] == archive_year},
        )
        if not extracted:
            raise RuntimeError(
                "Downloaded EN4 archive did not contain the requested month(s): "
                f"{sorted(missing_months)}"
            )
    return output_path


def _run_subprocess_command(cmd: Sequence[str]) -> None:
    """Run one external command and raise on failure."""
    subprocess.run([str(part) for part in cmd], check=True)


def _copernicusmarine_command() -> list[str]:
    """Return the preferred Copernicus Marine CLI command."""
    command = shutil.which("copernicusmarine")
    if command is not None:
        return [command]
    return [sys.executable, "-m", "copernicusmarine"]


def _ostia_day_tag(year: int, iso_week: int) -> str:
    """Return the YYYYMMDD tag for the ISO-week Wednesday OSTIA field."""
    selected = date.fromisocalendar(int(year), int(iso_week), 3)
    return selected.strftime("%Y%m%d")


def _existing_ostia_files(output_dir: Path, day_tag: str) -> list[Path]:
    """Return existing OSTIA NetCDF files for one day tag."""
    return sorted(Path(output_dir).glob(f"*{day_tag}*.nc"))


def _ostia_filter_for_day(day_tag: str) -> str:
    """Build the Copernicus Marine file filter for one OSTIA day."""
    year = day_tag[:4]
    month = day_tag[4:6]
    return (
        f"*/{year}/{month}/*{day_tag}120000-UKMO-L4_GHRSST-SSTfnd-"
        "OSTIA-GLOB_REP-v02.0-fv02.0.nc"
    )


def download_ostia_for_week(
    year: int,
    iso_week: int,
    output_dir: str | Path,
    *,
    dataset_candidates: Sequence[str] = DEFAULT_OSTIA_DATASET_CANDIDATES,
    force_download: bool = False,
    username: str | None = None,
    password: str | None = None,
    token: str | None = None,
    runner: CommandRunner | None = None,
) -> Path:
    """Download the OSTIA SST file needed for one ISO-week inference date."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    day_tag = _ostia_day_tag(year, iso_week)
    if _existing_ostia_files(output_path, day_tag) and not force_download:
        return output_path

    run_command = _run_subprocess_command if runner is None else runner
    base_cmd = _copernicusmarine_command()
    # The Copernicus Marine toolbox accepts the API key/token through its
    # password channel, so keep the old name and offer token as a clearer alias.
    effective_password = password
    if effective_password is None or str(effective_password).strip() == "":
        effective_password = token
    ostia_filter = _ostia_filter_for_day(day_tag)
    attempted: list[str] = []
    for dataset_id in dataset_candidates:
        clean_dataset_id = str(dataset_id).strip()
        if clean_dataset_id == "":
            continue
        attempted.append(clean_dataset_id)
        cmd = [
            *base_cmd,
            "get",
            "-i",
            clean_dataset_id,
            "--filter",
            ostia_filter,
            "-o",
            str(output_path),
            "-nd",
            "--log-level",
            "ERROR",
        ]
        if username is not None and str(username).strip() != "":
            cmd.extend(["--username", str(username)])
        if effective_password is not None and str(effective_password).strip() != "":
            cmd.extend(["--password", str(effective_password)])
        try:
            run_command(cmd)
        except Exception:
            continue
        if _existing_ostia_files(output_path, day_tag):
            return output_path

    raise RuntimeError(
        "Could not download OSTIA for "
        f"{int(year)}-W{int(iso_week):02d} ({day_tag}). "
        "Install/configure copernicusmarine credentials and check dataset IDs: "
        f"{attempted}."
    )


def _cfg_path_value(cfg: dict, path: str, *, default: object) -> object:
    """Read a nested config value, returning a default when absent."""
    node: object = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _selected_iso_week_date(year: int, iso_week: int) -> int:
    """Return the Wednesday date used for one ISO-week inference request."""
    selected = date.fromisocalendar(int(year), int(iso_week), 3)
    return int(selected.strftime("%Y%m%d"))


def _patch_axes_for_row(
    row: dict,
    *,
    tile_size: int,
    resolution_deg: float,
) -> PatchAxes:
    """Build latitude and longitude axes for one patch metadata row."""
    top = max(float(row["lat0"]), float(row["lat1"]))
    left = min(float(row["lon0"]), float(row["lon1"]))
    half = 0.5 * float(resolution_deg)
    lat_axis = top - half - (np.arange(int(tile_size)) * float(resolution_deg))
    lon_axis = left + half + (np.arange(int(tile_size)) * float(resolution_deg))
    return PatchAxes(lat_axis=lat_axis, lon_axis=lon_axis)


def _build_public_argo_rows(
    *,
    data_cfg: dict,
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None,
    land_mask_path: str | Path,
    min_ocean_fraction: float,
) -> tuple[list[dict], dict]:
    """Build selected public-inference patch rows without GLORYS source files."""
    tile_size = int(_cfg_path_value(data_cfg, "dataset.grid.tile_size", default=128))
    resolution_deg = float(
        _cfg_path_value(data_cfg, "dataset.grid.resolution_deg", default=0.1)
    )
    invalid_threshold = float(
        _cfg_path_value(data_cfg, "dataset.grid.invalid_threshold", default=0.5)
    )
    invalid_flags = tuple(
        _cfg_path_value(data_cfg, "dataset.grid.invalid_mask_flags", default=("land",))
    )
    val_fraction = float(_cfg_path_value(data_cfg, "split.val_fraction", default=0.2))
    random_seed = int(
        _cfg_path_value(data_cfg, "dataset.runtime.random_seed", default=7)
    )
    force_regions = _parse_force_include_regions(
        _cfg_path_value(data_cfg, "dataset.grid.force_include_regions", default=None)
    )
    patch_stride = max(1, int(tile_size) // 4)
    max_land_fraction = 1.0 - float(min_ocean_fraction)
    if not 0.0 <= max_land_fraction <= 1.0:
        raise ValueError("min_ocean_fraction must be in [0, 1].")

    grid_params = _GridParams(
        tile_size=tile_size,
        resolution_deg=resolution_deg,
        invalid_threshold=invalid_threshold,
        invalid_mask_flags=invalid_flags,
        val_fraction=val_fraction,
        val_year=None,
        split_seed=random_seed,
        patch_grid_source="land_mask",
        land_mask_path=land_mask_path,
        patch_stride=patch_stride,
        max_land_fraction=max_land_fraction,
        force_include_regions=force_regions,
    )
    patch_df = _build_land_mask_patch_table(grid_params)
    target_date = _selected_iso_week_date(year, iso_week)
    rows = patch_df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        row["date"] = int(target_date)
        row["export_index"] = int(idx)
        row["split"] = "inference"
        row["phase"] = "inference"
        row["argo_profile_count"] = 0

    selection = ExportSelection(
        selected_date=target_date,
        iso_year=int(year),
        iso_week=int(iso_week),
        indices=list(range(len(rows))),
    )
    selection = filter_selection_by_rectangle(rows, selection, rectangle)
    selected_rows = [rows[int(idx)] for idx in selection.indices]
    metadata = {
        "tile_size": int(tile_size),
        "resolution_deg": float(resolution_deg),
        "patch_stride": int(patch_stride),
        "patch_overlap_fraction": 1.0 - (float(patch_stride) / float(tile_size)),
        "min_ocean_fraction": float(min_ocean_fraction),
        "max_land_fraction": float(max_land_fraction),
        "patch_grid_source": "land_mask",
        "land_mask_path": str(land_mask_path),
        "require_argo_for_all": False,
    }
    return selected_rows, metadata


def _rasterize_public_argo_patch(
    *,
    argo_store: ArgoNetCDFStore,
    row: dict,
    tile_size: int,
    resolution_deg: float,
    temporal_window_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize ARGO profiles for one public-inference patch."""
    depth_size = int(argo_store.depth_axis_m.size)
    x_sum = np.zeros((depth_size, tile_size, tile_size), dtype=np.float64)
    x_count = np.zeros((depth_size, tile_size, tile_size), dtype=np.uint16)
    indices = argo_store.query_indices(
        target_date=int(row["date"]),
        temporal_window_days=int(temporal_window_days),
        lat0=float(row["lat0"]),
        lat1=float(row["lat1"]),
        lon0=float(row["lon0"]),
        lon1=float(row["lon1"]),
    )
    row["argo_profile_count"] = int(indices.size)
    if indices.size == 0:
        return (
            np.full(x_sum.shape, np.nan, dtype=np.float32),
            np.zeros(x_sum.shape, dtype=bool),
        )

    values = argo_store.load_temperature_profiles(indices)
    top = max(float(row["lat0"]), float(row["lat1"]))
    left = min(float(row["lon0"]), float(row["lon1"]))
    for local_idx, profile_idx in enumerate(indices.tolist()):
        lat = float(argo_store.latitude[int(profile_idx)])
        lon = _normalize_lon(float(argo_store.longitude[int(profile_idx)]))
        row_idx = int(np.floor((top - lat) / float(resolution_deg)))
        col_idx = int(np.floor((_normalize_lon(lon) - left) / float(resolution_deg)))
        if (
            row_idx < 0
            or row_idx >= int(tile_size)
            or col_idx < 0
            or col_idx >= int(tile_size)
        ):
            continue
        profile = values[int(local_idx)]
        valid = np.isfinite(profile)
        if not np.any(valid):
            continue
        x_sum[valid, row_idx, col_idx] += profile[valid].astype(np.float64)
        x_count[valid, row_idx, col_idx] += 1

    x_np = np.full(x_sum.shape, np.nan, dtype=np.float32)
    x_valid = x_count > 0
    x_np[x_valid] = (x_sum[x_valid] / x_count[x_valid].astype(np.float64)).astype(
        np.float32,
        copy=False,
    )
    return x_np, x_valid


def _ostia_values_look_kelvin(values: np.ndarray) -> bool:
    """Return True when an OSTIA patch appears to be encoded in Kelvin."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return False
    return float(np.nanmedian(finite)) > 100.0


def _load_public_eo_patch(
    *,
    ostia_store: TimedNetCDFStore,
    row: dict,
    axes: PatchAxes,
    tile_size: int,
    ostia_var_name: str,
) -> torch.Tensor:
    """Load an OSTIA patch as normalized EO conditioning."""
    eo_np = ostia_store.read_patch(
        target_date=int(row["date"]),
        var_name=ostia_var_name,
        axes=axes,
        categorical=False,
    )
    if eo_np.ndim != 2:
        raise RuntimeError(f"Expected OSTIA patch shape (H,W), got {eo_np.shape}.")
    eo_np = eo_np.astype(np.float32, copy=False)
    if _ostia_values_look_kelvin(eo_np):
        eo_np = eo_np - np.float32(273.15)
    eo = temperature_normalize(mode="norm", tensor=torch.from_numpy(eo_np[None, ...]))
    return torch.nan_to_num(eo, nan=0.0, posinf=0.0, neginf=0.0)


def _build_public_argo_sample(
    *,
    argo_store: ArgoNetCDFStore,
    ostia_store: TimedNetCDFStore,
    row: dict,
    tile_size: int,
    resolution_deg: float,
    temporal_window_days: int,
    ostia_var_name: str,
) -> dict[str, object]:
    """Build one model-ready public-inference sample from ARGO inputs."""
    axes = _patch_axes_for_row(
        row,
        tile_size=int(tile_size),
        resolution_deg=float(resolution_deg),
    )
    x_np, x_valid_mask_np = _rasterize_public_argo_patch(
        argo_store=argo_store,
        row=row,
        tile_size=int(tile_size),
        resolution_deg=float(resolution_deg),
        temporal_window_days=int(temporal_window_days),
    )
    eo = _load_public_eo_patch(
        ostia_store=ostia_store,
        row=row,
        axes=axes,
        tile_size=int(tile_size),
        ostia_var_name=ostia_var_name,
    )
    x = temperature_normalize(mode="norm", tensor=torch.from_numpy(x_np))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_valid_mask = torch.from_numpy(x_valid_mask_np.astype(np.bool_, copy=False))
    x_valid_mask_1d = x_valid_mask.any(dim=0, keepdim=True)
    y_valid_mask = torch.ones_like(x_valid_mask, dtype=torch.bool)
    land_mask = torch.ones((1, int(tile_size), int(tile_size)), dtype=torch.bool)
    return {
        "eo": eo,
        "x": x,
        "x_valid_mask": x_valid_mask,
        "y_valid_mask": y_valid_mask,
        "x_valid_mask_1d": x_valid_mask_1d,
        "land_mask": land_mask,
        "coords": torch.tensor(
            [
                0.5 * (float(row["lat0"]) + float(row["lat1"])),
                _center_lon_deg(float(row["lon0"]), float(row["lon1"])),
            ],
            dtype=torch.float32,
        ),
        "date": int(row["date"]),
    }


def _public_patch_feature_for_row(row: dict) -> dict:
    """Build a GeoJSON polygon feature for one public inference patch."""
    left = min(float(row["lon0"]), float(row["lon1"]))
    right = max(float(row["lon0"]), float(row["lon1"]))
    bottom = min(float(row["lat0"]), float(row["lat1"]))
    top = max(float(row["lat0"]), float(row["lat1"]))
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                    [left, top],
                ]
            ],
        },
        "properties": {
            "date": int(row["date"]),
            "patch_id": str(row.get("patch_id", "")),
            "export_index": int(row.get("export_index", -1)),
            "split": "inference",
        },
    }


def _write_public_data_config(
    *,
    source_config: Path,
    output_dir: Path,
    argo_dir: str | Path | None,
    glorys_dir: str | Path | None,
    ostia_dir: str | Path | None,
    sealevel_dir: str | Path | None,
    metadata_cache_dir: str | Path | None,
    land_mask_path: str | Path | None,
) -> Path:
    """Write a data config copy with user-provided source directories."""
    with Path(source_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg.setdefault("dataset", {})
    core_cfg = dataset_cfg.setdefault("core", {})
    grid_cfg = dataset_cfg.setdefault("grid", {})
    for key, value in (
        ("argo_dir", argo_dir),
        ("glorys_dir", glorys_dir),
        ("ostia_dir", ostia_dir),
        ("sealevel_dir", sealevel_dir),
        ("metadata_cache_dir", metadata_cache_dir),
    ):
        if value is not None:
            core_cfg[key] = str(value)
    if land_mask_path is not None:
        grid_cfg["land_mask_path"] = str(land_mask_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = output_dir / "depthdif_public_data_config.yaml"
    with public_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return public_config


def _export_args_from_public_api(
    *,
    assets: InferenceAssets,
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None,
    output_root: str | Path,
    device: str,
    batch_size: int | None,
    export_ground_truth: bool,
    full_sample_count: int,
    land_mask_path: str | Path,
    min_ocean_fraction: float,
    sigma: float,
    strict_load: bool,
) -> argparse.Namespace:
    """Build exporter args using the same defaults as the global CLI parser."""
    parser = _build_export_parser()
    args = parser.parse_args(
        ["--year", str(int(year)), "--iso-week", str(int(iso_week))]
    )
    args.model_config = str(assets.model_config)
    args.data_config = str(assets.data_config)
    args.train_config = str(assets.train_config)
    args.checkpoint_path = str(assets.checkpoint)
    args.rectangle = (
        None if rectangle is None else [float(value) for value in rectangle]
    )
    args.output_root = Path(output_root)
    args.device = str(device)
    args.batch_size = batch_size
    args.export_ground_truth = bool(export_ground_truth)
    args.full_sample_count = int(full_sample_count)
    args.land_mask_path = Path(land_mask_path)
    args.min_ocean_fraction = float(min_ocean_fraction)
    args.sigma = float(sigma)
    args.strict_load = bool(strict_load)
    return args


def run_week_inference(
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    checkpoint: str | Path | None = None,
    config_repo: str = DEFAULT_HF_REPO_ID,
    *,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    argo_dir: str | Path | None = None,
    glorys_dir: str | Path | None = None,
    ostia_dir: str | Path | None = None,
    sealevel_dir: str | Path | None = None,
    metadata_cache_dir: str | Path | None = None,
    auto_download_argo: bool = False,
    auto_download_ostia: bool = True,
    copernicus_username: str | None = None,
    copernicus_password: str | None = None,
    copernicus_token: str | None = None,
    export_ground_truth: bool = True,
    full_sample_count: int = 0,
    batch_size: int | None = None,
    land_mask_path: str | Path = DEFAULT_LAND_MASK_PATH,
    min_ocean_fraction: float = 0.05,
    sigma: float = DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    strict_load: bool = False,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Run DepthDif inference for one ISO week and return the run directory."""
    if glorys_dir is None:
        public_land_mask_path = (
            None
            if str(land_mask_path) == DEFAULT_LAND_MASK_PATH
            and not Path(land_mask_path).exists()
            else land_mask_path
        )
        return run_argo_week_inference(
            year=year,
            iso_week=iso_week,
            rectangle=rectangle,
            output_root=output_root,
            device=device,
            checkpoint=checkpoint,
            config_repo=config_repo,
            revision=revision,
            cache_dir=cache_dir,
            argo_dir=argo_dir,
            ostia_dir=ostia_dir,
            auto_download_argo=bool(auto_download_argo or argo_dir is None),
            auto_download_ostia=auto_download_ostia,
            copernicus_username=copernicus_username,
            copernicus_password=copernicus_password,
            copernicus_token=copernicus_token,
            batch_size=batch_size,
            land_mask_path=public_land_mask_path,
            min_ocean_fraction=min_ocean_fraction,
            sigma=sigma,
            strict_load=strict_load,
            force_download=force_download,
            downloader=downloader,
        )

    assets = resolve_hf_assets(
        config_repo=config_repo,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        downloader=downloader,
    )
    if checkpoint is not None:
        assets = InferenceAssets(
            model_config=assets.model_config,
            data_config=assets.data_config,
            train_config=assets.train_config,
            checkpoint=Path(checkpoint),
        )
    if auto_download_argo:
        argo_target = (
            Path(argo_dir)
            if argo_dir is not None
            else (_default_cache_dir() / "en4_profiles")
        )
        argo_dir = download_argo_for_week(
            year,
            iso_week,
            argo_target,
            cache_dir=None if cache_dir is None else Path(cache_dir) / "en4",
            force_download=force_download,
            downloader=downloader,
        )
    if auto_download_ostia:
        target_cache = _default_cache_dir() if cache_dir is None else Path(cache_dir)
        ostia_target = target_cache / "ostia" if ostia_dir is None else Path(ostia_dir)
        ostia_dir = download_ostia_for_week(
            year,
            iso_week,
            ostia_target,
            force_download=force_download,
            username=copernicus_username,
            password=copernicus_password,
            token=copernicus_token,
        )

    output_root_path = Path(output_root)
    data_config = _write_public_data_config(
        source_config=assets.data_config,
        output_dir=output_root_path,
        argo_dir=argo_dir,
        glorys_dir=glorys_dir,
        ostia_dir=ostia_dir,
        sealevel_dir=sealevel_dir,
        metadata_cache_dir=metadata_cache_dir,
        land_mask_path=land_mask_path,
    )
    public_assets = InferenceAssets(
        model_config=assets.model_config,
        data_config=data_config,
        train_config=assets.train_config,
        checkpoint=assets.checkpoint,
    )
    args = _export_args_from_public_api(
        assets=public_assets,
        year=year,
        iso_week=iso_week,
        rectangle=rectangle,
        output_root=output_root_path,
        device=device,
        batch_size=batch_size,
        export_ground_truth=export_ground_truth,
        full_sample_count=full_sample_count,
        land_mask_path=land_mask_path,
        min_ocean_fraction=min_ocean_fraction,
        sigma=sigma,
        strict_load=strict_load,
    )
    result: ExportRunResult = run_global_inference(args)
    return result.run_dir


def run_argo_week_inference(
    year: int,
    iso_week: int,
    rectangle: Sequence[float] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "auto",
    checkpoint: str | Path | None = None,
    config_repo: str = DEFAULT_HF_REPO_ID,
    *,
    revision: str = DEFAULT_HF_REVISION,
    cache_dir: str | Path | None = None,
    argo_dir: str | Path | None = None,
    ostia_dir: str | Path | None = None,
    auto_download_argo: bool = True,
    auto_download_ostia: bool = True,
    copernicus_username: str | None = None,
    copernicus_password: str | None = None,
    copernicus_token: str | None = None,
    batch_size: int | None = None,
    land_mask_path: str | Path | None = None,
    min_ocean_fraction: float = 0.05,
    sigma: float = DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    strict_load: bool = False,
    force_download: bool = False,
    downloader: Downloader | None = None,
) -> Path:
    """Run public ARGO+OSTIA inference for one ISO week and return the run directory."""
    assets = resolve_hf_assets(
        config_repo=config_repo,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        downloader=downloader,
    )
    if checkpoint is not None:
        assets = InferenceAssets(
            model_config=assets.model_config,
            data_config=assets.data_config,
            train_config=assets.train_config,
            checkpoint=Path(checkpoint),
        )

    target_cache = _default_cache_dir() if cache_dir is None else Path(cache_dir)
    if land_mask_path is None:
        land_mask_path = resolve_hf_land_mask(
            config_repo=config_repo,
            revision=revision,
            cache_dir=target_cache,
            force_download=force_download,
            downloader=downloader,
        )

    if auto_download_argo:
        argo_target = (
            target_cache / "en4_profiles" if argo_dir is None else Path(argo_dir)
        )
        argo_dir = download_argo_for_week(
            year,
            iso_week,
            argo_target,
            cache_dir=target_cache / "en4",
            force_download=force_download,
            downloader=downloader,
        )
    if argo_dir is None:
        raise ValueError("argo_dir is required when auto_download_argo=false.")
    if auto_download_ostia:
        ostia_target = target_cache / "ostia" if ostia_dir is None else Path(ostia_dir)
        ostia_dir = download_ostia_for_week(
            year,
            iso_week,
            ostia_target,
            force_download=force_download,
            username=copernicus_username,
            password=copernicus_password,
            token=copernicus_token,
        )
    if ostia_dir is None or str(ostia_dir).strip() == "":
        raise ValueError("ostia_dir is required when auto_download_ostia=false.")

    data_cfg = load_yaml(assets.data_config)
    training_cfg = load_yaml(assets.train_config)
    model_cfg = load_yaml(assets.model_config)
    rows, inference_grid_metadata = _build_public_argo_rows(
        data_cfg=data_cfg,
        year=year,
        iso_week=iso_week,
        rectangle=rectangle,
        land_mask_path=land_mask_path,
        min_ocean_fraction=min_ocean_fraction,
    )
    if not rows:
        raise RuntimeError("No public inference patches were selected.")

    tile_size = int(inference_grid_metadata["tile_size"])
    resolution_deg = float(inference_grid_metadata["resolution_deg"])
    temporal_window_days = int(
        _cfg_path_value(
            data_cfg,
            "dataset.sampling.temporal_window_days",
            default=7,
        )
    )
    ostia_var_name = str(
        _cfg_path_value(
            data_cfg, "dataset.sampling.ostia_var_name", default="analysed_sst"
        )
    )
    argo_temp_var_name = str(
        _cfg_path_value(data_cfg, "dataset.sampling.argo_temp_var_name", default="TEMP")
    )
    argo_depth_var_name = str(
        _cfg_path_value(
            data_cfg,
            "dataset.sampling.argo_depth_var_name",
            default="DEPH_CORRECTED",
        )
    )
    batch_size_value = int(
        batch_size
        if batch_size is not None
        else training_cfg.get("dataloader", {}).get("val_batch_size", 4)
    )
    if batch_size_value < 1:
        raise ValueError("batch_size must be >= 1.")

    selected_date = _selected_iso_week_date(year, iso_week)
    run_stem = f"depthdif_argo_{selected_date}"
    output_root_path = Path(output_root)
    run_dir = output_root_path / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = run_dir / ".scratch"

    depth_axis_m = np.asarray(DEFAULT_GLORYS_DEPTH_AXIS_M, dtype=np.float64)
    depth_export_levels = resolve_depth_export_levels(depth_axis_m)
    depth_channel_indices = tuple(
        int(level.channel_index) for level in depth_export_levels
    )
    layout = build_mosaic_layout(rows, patch_shape=(tile_size, tile_size))
    land_mask = load_land_mask_for_layout(
        land_mask_path=Path(land_mask_path),
        layout=layout,
    )
    pred_accumulators = {
        level.suffix: create_raster_accumulator(
            root_dir=scratch_dir,
            stem=f"prediction_{level.suffix}",
            layout=layout,
        )
        for level in depth_export_levels
    }
    argo_points_geojson_path = run_dir / f"{run_stem}_argo_points.geojson"
    patch_splits_geojson_path = run_dir / f"{run_stem}_patch_splits.geojson"
    argo_points_writer = GeoJSONPointWriter(argo_points_geojson_path)
    patch_splits_writer = GeoJSONPointWriter(patch_splits_geojson_path)
    argo_points_writer.open()
    patch_splits_writer.open()

    argo_store = ArgoNetCDFStore(
        argo_dir,
        depth_axis_m=depth_axis_m.astype(np.float32, copy=False),
        temp_var_name=argo_temp_var_name,
        depth_var_name=argo_depth_var_name,
    )
    ostia_store = TimedNetCDFStore(ostia_dir)

    model = build_model(
        model_config_path=str(assets.model_config),
        data_config_path=str(assets.data_config),
        training_config_path=str(assets.train_config),
        model_cfg=model_cfg,
        datamodule=None,
    )
    ckpt_path = resolve_checkpoint_path(str(assets.checkpoint), model_cfg)
    if ckpt_path is None:
        raise RuntimeError("No public inference checkpoint was resolved.")
    checkpoint_payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = (
        checkpoint_payload["state_dict"]
        if "state_dict" in checkpoint_payload
        else checkpoint_payload
    )
    model.load_state_dict(state_dict, strict=bool(strict_load))
    target_device = choose_device(device)
    model = model.to(target_device)
    model.eval()

    print(
        "Preparing public ARGO+OSTIA inference: "
        f"selected_date={selected_date}, "
        f"iso_week={int(year)}-W{int(iso_week):02d}, "
        f"selected_patches={len(rows)}, "
        f"batch_size={batch_size_value}, "
        "ostia_conditioning=enabled, "
        f"rectangle={rectangle}"
    )

    for start in range(0, len(rows), batch_size_value):
        batch_rows = rows[start : start + batch_size_value]
        samples = [
            _build_public_argo_sample(
                argo_store=argo_store,
                ostia_store=ostia_store,
                row=row,
                tile_size=tile_size,
                resolution_deg=resolution_deg,
                temporal_window_days=temporal_window_days,
                ostia_var_name=ostia_var_name,
            )
            for row in batch_rows
        ]
        batch = {
            key: (
                torch.stack([sample[key] for sample in samples], dim=0)
                if torch.is_tensor(samples[0][key])
                else torch.as_tensor([sample[key] for sample in samples])
            )
            for key in samples[0]
        }
        with torch.no_grad():
            outputs = model.predict_step(_to_device(batch, target_device), batch_idx=0)

        prediction_depth_batch = (
            outputs["y_hat_denorm"][:, depth_channel_indices]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        for local_idx, row in enumerate(batch_rows):
            patch_splits_writer.write_feature(_public_patch_feature_for_row(row))
            observed_mask_2d = (
                batch["x_valid_mask_1d"][local_idx, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool, copy=False)
            )
            for feature in _argo_point_features_for_patch(
                row=row,
                observed_mask_2d=observed_mask_2d,
            ):
                argo_points_writer.write_feature(feature)
            for depth_idx, level in enumerate(depth_export_levels):
                accumulator = pred_accumulators[level.suffix]
                _accumulate_patch_into_arrays(
                    accumulator.sum_array,
                    accumulator.count_array,
                    row=row,
                    patch_values=prediction_depth_batch[local_idx, depth_idx],
                    layout=layout,
                )

    for accumulator in pred_accumulators.values():
        _flush_accumulator(accumulator)
    argo_points_writer.close()
    patch_splits_writer.close()
    pd.DataFrame.from_records(rows).to_csv(
        run_dir / "selected_patches.csv", index=False
    )

    depth_export_records: list[dict[str, object]] = []
    for level in depth_export_levels:
        prediction_tif_path = run_dir / f"{run_stem}_prediction_{level.suffix}.tif"
        common_depth_tags = {
            "selected_date": str(int(selected_date)),
            "selected_patch_count": str(int(len(rows))),
            "depth_label": level.label,
            "requested_depth_m": f"{float(level.requested_depth_m):.3f}",
            "actual_depth_m": f"{float(level.actual_depth_m):.3f}",
            "channel_index": str(int(level.channel_index)),
            "land_mask_path": str(land_mask_path),
            "land_zeroed": "true",
        }
        write_global_top_band_geotiff(
            output_path=prediction_tif_path,
            accumulator=pred_accumulators[level.suffix],
            layout=layout,
            nodata=-9999.0,
            band_description=f"predicted_{level.suffix}_celsius",
            tags={
                **common_depth_tags,
                "source": "DepthDif public ARGO+OSTIA weekly inference export",
                "checkpoint_path": str(ckpt_path),
                "kind": "prediction",
                "prediction_runs_per_patch": "1",
                "extra_gaussian_blur_sigma": f"{float(sigma):.3f}",
            },
            extra_gaussian_blur_sigma=float(sigma),
            land_mask=land_mask,
        )
        depth_export_records.append(
            {
                "suffix": level.suffix,
                "label": level.label,
                "requested_depth_m": float(level.requested_depth_m),
                "actual_depth_m": float(level.actual_depth_m),
                "channel_index": int(level.channel_index),
                "prediction_tif_path": _summary_artifact_path(prediction_tif_path),
                "ground_truth_tif_path": None,
            }
        )

    run_summary = {
        "selected_date": int(selected_date),
        "target_date": int(selected_date),
        "iso_year": int(year),
        "iso_week": int(iso_week),
        "selected_patch_count": int(len(rows)),
        "rectangle": (
            None if rectangle is None else [float(value) for value in rectangle]
        ),
        "inference_grid": inference_grid_metadata,
        "land_mask_path": str(land_mask_path),
        "land_zeroed": True,
        "checkpoint_path": str(ckpt_path),
        "model_config": str(assets.model_config),
        "data_config": str(assets.data_config),
        "train_config": str(assets.train_config),
        "device": str(target_device),
        "run_dir": str(run_dir),
        "prediction_tif_path": str(depth_export_records[0]["prediction_tif_path"]),
        "ground_truth_tif_path": None,
        "depth_exports": depth_export_records,
        "argo_points_geojson_path": _summary_artifact_path(argo_points_geojson_path),
        "argo_point_count": int(argo_points_writer.feature_count),
        "patch_splits_geojson_path": _summary_artifact_path(patch_splits_geojson_path),
        "patch_split_count": int(patch_splits_writer.feature_count),
        "ostia_dir": str(ostia_dir),
        "argo_dir": str(argo_dir),
        "glorys_dir": None,
        "glorys_required": False,
    }
    with (run_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, sort_keys=False)

    for accumulator in pred_accumulators.values():
        _cleanup_accumulator(accumulator)
    scratch_dir.rmdir()
    return run_dir


def _build_public_parser() -> argparse.ArgumentParser:
    """Build CLI parser for public inference commands."""
    parser = argparse.ArgumentParser(description="DepthDif public inference helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer-week", help="Run one ISO-week inference.")
    infer.add_argument("--year", type=int, required=True)
    infer.add_argument("--iso-week", type=int, required=True)
    infer.add_argument("--rectangle", type=float, nargs=4, default=None)
    infer.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    infer.add_argument("--checkpoint", type=Path, default=None)
    infer.add_argument("--config-repo", default=DEFAULT_HF_REPO_ID)
    infer.add_argument("--revision", default=DEFAULT_HF_REVISION)
    infer.add_argument("--cache-dir", type=Path, default=None)
    infer.add_argument("--argo-dir", type=Path, default=None)
    infer.add_argument("--glorys-dir", type=Path, default=None)
    infer.add_argument("--ostia-dir", type=Path, default=None)
    infer.add_argument("--sealevel-dir", type=Path, default=None)
    infer.add_argument("--metadata-cache-dir", type=Path, default=None)
    infer.add_argument("--auto-download-argo", action="store_true")
    infer.add_argument("--no-auto-download-ostia", action="store_true")
    infer.add_argument("--copernicus-username", default=None)
    infer.add_argument("--copernicus-password", default=None)
    infer.add_argument("--copernicus-token", default=None)
    infer.add_argument("--no-export-ground-truth", action="store_true")
    infer.add_argument("--batch-size", type=int, default=None)
    infer.add_argument("--full-sample-count", type=int, default=0)
    infer.add_argument(
        "--land-mask-path", type=Path, default=Path(DEFAULT_LAND_MASK_PATH)
    )
    infer.add_argument("--min-ocean-fraction", type=float, default=0.05)
    infer.add_argument(
        "--sigma", type=float, default=DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA
    )
    infer.add_argument("--strict-load", action="store_true")
    infer.add_argument("--force-download", action="store_true")

    argo = subparsers.add_parser("download-argo", help="Download EN4/ARGO for a week.")
    argo.add_argument("--year", type=int, required=True)
    argo.add_argument("--iso-week", type=int, required=True)
    argo.add_argument("--output-dir", type=Path, required=True)
    argo.add_argument("--base-url", default=DEFAULT_EN4_BASE_URL)
    argo.add_argument("--cache-dir", type=Path, default=None)
    argo.add_argument("--force-download", action="store_true")

    ostia = subparsers.add_parser("download-ostia", help="Download OSTIA for a week.")
    ostia.add_argument("--year", type=int, required=True)
    ostia.add_argument("--iso-week", type=int, required=True)
    ostia.add_argument("--output-dir", type=Path, required=True)
    ostia.add_argument("--copernicus-username", default=None)
    ostia.add_argument("--copernicus-password", default=None)
    ostia.add_argument("--copernicus-token", default=None)
    ostia.add_argument("--force-download", action="store_true")
    return parser


def infer_week_cli(argv: Sequence[str] | None = None) -> None:
    """Run the public week-inference console command."""
    parser = _build_public_parser()
    args = parser.parse_args(["infer-week", *(sys.argv[1:] if argv is None else argv)])
    run_dir = run_week_inference(
        year=args.year,
        iso_week=args.iso_week,
        rectangle=args.rectangle,
        output_root=args.output_root,
        device=args.device,
        checkpoint=args.checkpoint,
        config_repo=args.config_repo,
        revision=args.revision,
        cache_dir=args.cache_dir,
        argo_dir=args.argo_dir,
        glorys_dir=args.glorys_dir,
        ostia_dir=args.ostia_dir,
        sealevel_dir=args.sealevel_dir,
        metadata_cache_dir=args.metadata_cache_dir,
        auto_download_argo=args.auto_download_argo,
        auto_download_ostia=not bool(args.no_auto_download_ostia),
        copernicus_username=args.copernicus_username,
        copernicus_password=args.copernicus_password,
        copernicus_token=args.copernicus_token,
        export_ground_truth=not bool(args.no_export_ground_truth),
        full_sample_count=args.full_sample_count,
        batch_size=args.batch_size,
        land_mask_path=args.land_mask_path,
        min_ocean_fraction=args.min_ocean_fraction,
        sigma=args.sigma,
        strict_load=args.strict_load,
        force_download=args.force_download,
    )
    print(run_dir)


def download_argo_cli(argv: Sequence[str] | None = None) -> None:
    """Run the public ARGO download console command."""
    parser = _build_public_parser()
    args = parser.parse_args(
        ["download-argo", *(sys.argv[1:] if argv is None else argv)]
    )
    output_dir = download_argo_for_week(
        args.year,
        args.iso_week,
        args.output_dir,
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
    print(output_dir)


def download_ostia_cli(argv: Sequence[str] | None = None) -> None:
    """Run the public OSTIA download console command."""
    parser = _build_public_parser()
    args = parser.parse_args(
        ["download-ostia", *(sys.argv[1:] if argv is None else argv)]
    )
    output_dir = download_ostia_for_week(
        args.year,
        args.iso_week,
        args.output_dir,
        force_download=args.force_download,
        username=args.copernicus_username,
        password=args.copernicus_password,
        token=args.copernicus_token,
    )
    print(output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Run public API subcommands when invoked as a module."""
    parser = _build_public_parser()
    args = parser.parse_args(argv)
    if args.command == "infer-week":
        run_dir = run_week_inference(
            year=args.year,
            iso_week=args.iso_week,
            rectangle=args.rectangle,
            output_root=args.output_root,
            device=args.device,
            checkpoint=args.checkpoint,
            config_repo=args.config_repo,
            revision=args.revision,
            cache_dir=args.cache_dir,
            argo_dir=args.argo_dir,
            glorys_dir=args.glorys_dir,
            ostia_dir=args.ostia_dir,
            sealevel_dir=args.sealevel_dir,
            metadata_cache_dir=args.metadata_cache_dir,
            auto_download_argo=args.auto_download_argo,
            auto_download_ostia=not bool(args.no_auto_download_ostia),
            copernicus_username=args.copernicus_username,
            copernicus_password=args.copernicus_password,
            copernicus_token=args.copernicus_token,
            export_ground_truth=not bool(args.no_export_ground_truth),
            full_sample_count=args.full_sample_count,
            batch_size=args.batch_size,
            land_mask_path=args.land_mask_path,
            min_ocean_fraction=args.min_ocean_fraction,
            sigma=args.sigma,
            strict_load=args.strict_load,
            force_download=args.force_download,
        )
        print(run_dir)
    elif args.command == "download-argo":
        output_dir = download_argo_for_week(
            args.year,
            args.iso_week,
            args.output_dir,
            base_url=args.base_url,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
        print(output_dir)
    elif args.command == "download-ostia":
        output_dir = download_ostia_for_week(
            args.year,
            args.iso_week,
            args.output_dir,
            force_download=args.force_download,
            username=args.copernicus_username,
            password=args.copernicus_password,
            token=args.copernicus_token,
        )
        print(output_dir)


if __name__ == "__main__":
    main()
