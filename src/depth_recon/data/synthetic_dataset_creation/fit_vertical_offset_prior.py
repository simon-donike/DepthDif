# Example with all options:
# /work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior \
#   --geotiff-root-dir /work/data/OceanVariableReconstruction \
#   --output-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
#   --metadata-cache-dir /work/data/OceanVariableReconstruction/depthdif_cache \
#   --start-year 2000 --end-year 2024 --exclude-year 2018 --tile-size 128 \
#   --patch-stride 128 --max-land-fraction 0.30 --max-patches 4000 \
#   --max-supervised-depth-m 1000 --random-seed 7 --overwrite --no-progress
"""Fit mean GLORYS depth-minus-surface offsets for deterministic pretraining."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from depth_recon.data.dataset_argo_geotiff_gridded import (
    ArgoGeoTIFFGriddedPatchDataset,
)
from depth_recon.data.synthetic_dataset_creation.vertical_offset_prior import (
    VerticalOffsetAccumulator,
    VerticalOffsetPrior,
)


def _manifest_surface_window_days(manifest: Mapping[str, Any]) -> int:
    """Read and validate the centered surface-composite width."""
    aggregation = manifest.get("surface_temporal_aggregation")
    if not isinstance(aggregation, Mapping) or "window_days" not in aggregation:
        raise RuntimeError(
            "GeoTIFF manifest is missing surface_temporal_aggregation.window_days."
        )
    value = int(aggregation["window_days"])
    if value < 1 or float(aggregation["window_days"]) != float(value):
        raise ValueError("surface temporal window_days must be a positive integer.")
    return value


def _centered_window_avoids_excluded_years(
    date: int, *, window_days: int, excluded_years: Sequence[int]
) -> bool:
    """Reject dates whose complete centered surface window touches held-out years."""
    center = datetime.strptime(str(int(date)), "%Y%m%d")
    radius = timedelta(days=(int(window_days) - 1) / 2.0)
    touched = set(range((center - radius).year, (center + radius).year + 1))
    return touched.isdisjoint(int(value) for value in excluded_years)


def _select_rows(
    rows: pd.DataFrame, *, max_patches: int | None, random_seed: int
) -> pd.DataFrame:
    """Select reproducible rows approximately balanced across calendar months."""
    selected = rows.copy()
    selected["_month"] = (selected["date"].astype(np.int64) // 100) % 100
    if max_patches is None or int(max_patches) >= len(selected):
        return selected.sort_values(["date", "grid_y0", "grid_x0"])
    grouped = list(selected.groupby("_month", sort=True))
    if int(max_patches) < len(grouped):
        raise ValueError(
            "max_patches is smaller than the number of represented months."
        )
    rng = np.random.default_rng(int(random_seed))
    base, remainder = divmod(int(max_patches), len(grouped))
    pieces: list[pd.DataFrame] = []
    for group_index, (_, group) in enumerate(grouped):
        count = min(len(group), base + int(group_index < remainder))
        positions = rng.choice(len(group), size=count, replace=False)
        pieces.append(group.iloc[np.sort(positions)])
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["date", "grid_y0", "grid_x0"]
    )


def fit_vertical_offset_prior(
    *,
    geotiff_root_dir: str | Path,
    output_path: str | Path,
    metadata_cache_dir: str | Path | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    excluded_years: Sequence[int] = (2018,),
    tile_size: int = 128,
    patch_stride: int | None = None,
    max_land_fraction: float = 0.30,
    max_patches: int | None = 4000,
    max_supervised_depth_m: float = 1000.0,
    random_seed: int = 7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> VerticalOffsetPrior:
    """Fit and save one scalar temperature/salinity offset per depth level."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Vertical-offset artifact already exists: {output}")
    excluded = tuple(sorted({int(value) for value in excluded_years}))
    dataset = ArgoGeoTIFFGriddedPatchDataset(
        geotiff_root_dir=geotiff_root_dir,
        metadata_cache_dir=metadata_cache_dir,
        split="all",
        tile_size=int(tile_size),
        patch_stride=(int(tile_size) if patch_stride is None else int(patch_stride)),
        max_land_fraction=float(max_land_fraction),
        require_argo_for_all=False,
        surface_conditioning={"sources": ["sst", "sss", "adt"]},
        pretraining_prior={"enabled": False},
        output_fields=("temperature", "salinity"),
        random_seed=int(random_seed),
    )
    rows = dataset._rows.copy()
    years = rows["date"].astype(np.int64) // 10000
    keep = pd.Series(True, index=rows.index, dtype=bool)
    if start_year is not None:
        keep &= years >= int(start_year)
    if end_year is not None:
        keep &= years <= int(end_year)
    window_days = _manifest_surface_window_days(dataset.manifest)
    keep &= rows["date"].map(
        lambda date: _centered_window_avoids_excluded_years(
            int(date), window_days=window_days, excluded_years=excluded
        )
    )
    rows = _select_rows(
        rows.loc[keep].reset_index(drop=True),
        max_patches=max_patches,
        random_seed=random_seed,
    )
    accumulator = VerticalOffsetAccumulator(
        depth_axis_m=dataset.depth_axis_m,
        excluded_years=excluded,
        provenance={
            "source": "mean GLORYS depth-minus-surface offsets",
            "geotiff_root_dir": str(Path(geotiff_root_dir).resolve()),
            "manifest_sha256": hashlib.sha256(
                dataset.manifest_path.read_bytes()
            ).hexdigest(),
            "selected_start_date": int(rows["date"].min()),
            "selected_end_date": int(rows["date"].max()),
            "surface_temporal_aggregation_window_days": int(window_days),
            "random_seed": int(random_seed),
        },
    )
    iterator = tqdm(
        rows.to_dict(orient="records"),
        desc="Fitting vertical offsets",
        unit="patch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for row in iterator:
        accumulator.update(
            temperature_c=dataset._load_y_patch(row),
            salinity_psu=dataset._load_y_salinity_patch(row),
            date=int(row["date"]),
        )
    prior = accumulator.finalize(max_supervised_depth_m=float(max_supervised_depth_m))
    prior.to_npz(output)
    dataset.raster_cache.close()
    return prior


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geotiff-root-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metadata-cache-dir", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--exclude-year", type=int, action="append", default=None)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--max-land-fraction", type=float, default=0.30)
    parser.add_argument("--max-patches", type=int, default=4000)
    parser.add_argument("--max-supervised-depth-m", type=float, default=1000.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    """Fit vertical offsets from command-line arguments."""
    args = _build_parser().parse_args()
    fit_vertical_offset_prior(
        geotiff_root_dir=args.geotiff_root_dir,
        output_path=args.output_path,
        metadata_cache_dir=args.metadata_cache_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        excluded_years=(
            args.exclude_year if args.exclude_year is not None else (2018,)
        ),
        tile_size=args.tile_size,
        patch_stride=args.patch_stride,
        max_land_fraction=args.max_land_fraction,
        max_patches=args.max_patches,
        max_supervised_depth_m=args.max_supervised_depth_m,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
