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
- `/data1/datasets/depth_v2/glorys`
- `/data1/datasets/depth_v2/sealevel_daily`

The active config is `configs/px_space/data_ostia_argo_netcdf.yaml`.

## Dataset Assembly

At runtime, the loader:

1. Scans NetCDF source roots and caches compact date/path indexes.
2. Builds a deterministic OSTIA-derived patch grid.
3. Assigns a deterministic patch-level train/val split.
4. Expands source dates into `(patch, date)` rows.
5. Optionally precomputes ARGO-support flags for fast split filtering.
6. Reads GLORYS, OSTIA, ARGO, and sea-level source files lazily per sample.

Only metadata caches are written under `dataset.core.metadata_cache_dir`.
Model-facing tensors are produced on demand.

## Spatial And Temporal Semantics

- `dataset.grid.tile_size` controls patch height/width.
- `dataset.grid.resolution_deg` controls patch pixel spacing.
- `dataset.sampling.temporal_window_days` controls the centered ARGO profile
  search window for each patch date.
- `dataset.selection.require_argo_for_train` and
  `dataset.selection.require_argo_for_val` default to `true`.
- `dataset.selection.require_argo_for_all` defaults to `false` so global
  inference can cover rows without ARGO observations.

## Depth Semantics

- GLORYS `thetao` defines the dense target `y`.
- ARGO `TEMP` is projected from `DEPH_CORRECTED` samples onto the GLORYS depth
  axis before rasterization.
- `dataset.depth_axis_m` exposes the physical GLORYS depth levels to inference
  and export code.

## Output Contract

Each sample returns `eo`, `x`, `y`, `x_valid_mask`, `y_valid_mask`,
`x_valid_mask_1d`, `land_mask`, `date`, and optional `coords`/`info`.

See [NetCDF Patch Dataset](data.md) for the full tensor contract.
