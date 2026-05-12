# Production Dataset

This page documents the maintained production dataset path. DepthDif now builds
patches directly from source NetCDF files through
`ArgoNetCDFGriddedPatchDataset`; there is no precomputed patch export stage.

Use [Data Sources](data-source.md) for native product properties and
[Depth Alignment](depth-alignment.md) for ARGO-to-GLORYS vertical resampling.

## Source Inputs

Expected source trees:

- `/data1/datasets/depth_v2/ostia`
- `/data1/datasets/depth_v2/en4_profiles`
- `/data1/datasets/depth_v2/glorys_weekly`
- `/data1/datasets/depth_v2/sealevel_daily`

The active config is `configs/px_space/data_ostia_argo_netcdf.yaml`.

## Dataset Assembly

At runtime, the loader:

1. Scans NetCDF source roots and caches compact date/path indexes.
2. Builds a deterministic land-mask-derived patch grid, with configurable
   overlap and maximum land fraction.
3. Assigns train/val from `split.val_year` when overlapping patches are enabled.
4. Expands source dates into `(patch, date)` rows.
5. Optionally precomputes ARGO-support flags for fast split filtering.
6. Reads GLORYS, OSTIA, ARGO, and sea-level source files lazily per sample.

Only metadata caches are written under `dataset.core.metadata_cache_dir`.
Model-facing tensors are produced on demand.

## Patch Grid Concept

A patch is a fixed-size window on the 0.1 degree GLORYS grid. The production
configuration uses `tile_size: 128`, so each patch covers 128 by 128 grid cells.
Instead of placing every patch once in a non-overlapping grid, the loader moves
the window by `patch_stride`. The default stride is 64 cells, which means nearby
patches overlap by half their width and height.

The same world grid is reused every time the dataset is instantiated. That makes
the patch locations deterministic: changing the date range changes which
timesteps are available, but it does not move the patch boundaries.

![Global patch grid overview](assets/data/patch_grid/patch_grid_global_overview.png)

The global view shows all candidate patch windows. Transparent outlines make
overlap visible: darker regions are covered by more candidate patches. Retained
patches pass the ocean/land rules, force-included patches are retained by a
regional override, and rejected patches are land-heavy candidates that are not
used for training or validation rows.

## Patch Filtering

The grid is built from the committed GLORYS-aligned land-mask GeoTIFF:
`data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif`.
In that mask, `1` means land and `0` means ocean. For each candidate patch, the
dataset computes the fraction of land pixels and keeps the patch when
`land_fraction <= dataset.grid.max_land_fraction`.

The default cap is `0.30`, so normal retained patches are at least 70% ocean.
Mediterranean-centered patches are force-included with a relaxed land cap through
`dataset.grid.force_include_regions`, because that basin is narrow and would
otherwise lose useful ocean context around coastlines.

![Land fraction filter examples](assets/data/patch_grid/land_fraction_filter_examples.png)

![Mediterranean overlap example](assets/data/patch_grid/patch_overlap_regional_example.png)

Overlapping patches mean an ARGO profile is not tied to only one spatial
context. If a profile falls inside several retained patch bounds, it can
contribute support to each matching `(patch, date)` row.

![ARGO profile in multiple patch contexts](assets/data/patch_grid/argo_profile_multiple_contexts.png)

## Patch Registry Storage

During dataset instantiation, the loader expands the retained patch table across
the available OSTIA dates, then filters those dates to the GLORYS and sea-level
coverage already present on disk. The resulting registry is a table of
`(patch_id, date)` rows and becomes `dataset.rows`.

Each patch row stores the grid indices, latitude/longitude bounds, center
coordinates, `land_fraction`, `ocean_fraction`, `invalid_fraction`, and any
force-include metadata. Each date row stores the timestep, split assignment, and
optional ARGO-support count. Cache filenames include the grid source, stride,
tile size, land threshold, temporal window, split policy, and mask metadata so a
changed configuration creates a new cache instead of reusing stale rows.

The cache is metadata only. It records where patches are and which dates are
valid; it does not store precomputed GLORYS, OSTIA, sea-level, or ARGO tensors.

## Sample Read Path

When training asks for an item, `__getitem__` reads one registry row, converts
the stored patch bounds back into source-file slices, and lazily loads the
matching GLORYS, OSTIA, and sea-level data for that date. ARGO profiles are
selected by the patch bounds and the configured temporal window, projected onto
the GLORYS depth axis, then rasterized into the sample tensors and validity
masks.

The Zarr dataset uses the same patch-grid and registry rules. Its source reads
come from Zarr stores instead of NetCDF files, but the conceptual registry is the
same: deterministic patch windows crossed with valid timesteps.

## Spatial And Temporal Semantics

- `dataset.grid.tile_size` controls patch height/width.
- `dataset.grid.resolution_deg` controls patch pixel spacing.
- `dataset.grid.patch_stride` controls patch overlap; values below `tile_size`
  require `split.val_year`.
- `dataset.grid.max_land_fraction` filters land-heavy patches from the
  committed GLORYS-aligned world mask.
- `dataset.grid.force_include_regions` keeps Mediterranean-centered patches up
  to a relaxed land fraction so the training registry retains that basin.
- `dataset.sampling.temporal_window_days` controls the centered ARGO profile
  search window for each patch date.
- `dataset.selection.require_argo_for_train` and
  `dataset.selection.require_argo_for_val` default to `true`.
- `dataset.selection.require_argo_for_all` defaults to `false` so global
  inference can cover rows without ARGO observations.
- `split.val_year` defaults to `2018`, assigning that year to validation and
  all other years to training.

## Depth Semantics

- GLORYS `thetao` defines the dense target `y`.
- ARGO `TEMP` is projected from `DEPH_CORRECTED` samples onto the GLORYS depth
  axis before rasterization.
- `dataset.depth_axis_m` exposes the physical GLORYS depth levels to inference
  and export code.

## Output Contract

Each sample returns `eo`, `x`, `y`, `x_valid_mask`, `y_valid_mask`,
`x_valid_mask_1d`, `land_mask`, `date`, and optional `coords`/`info`.

See [Data Contract](data-contract.md) for the full tensor contract.
