# Data Overview

DepthDif learns to densify sparse in-situ ocean profiles into gridded
temperature fields. The active GeoTIFF workflow can also train a salinity-only or joint temperature + salinity pixel model when `--scenario salinity` or `--scenario joint` is selected. The training data combines satellite surface context,
profile observations, normal dense targets or aggregate prior statistics, and three dense surface predictors:
SST, SSS, and ADT on one shared global grid.

Use [Data Sources](data-source.md) for product-specific details,
[Dataset Downloads](data-download.md) for raw and packaged download commands,
[Depth Alignment](depth-alignment.md) for vertical profile projection,
[Data Export](data-export.md) for the GeoTIFF training store, and
[Data Contract](data-contract.md) for exact model tensor shapes.

## Modalities

The current training workflow is built around these modalities:

| Modality | Source | Variables | Role |
| --- | --- | --- | --- |
| Sparse profile temperature | EN4 / ARGO | `TEMP`, projected to GLORYS depths | Sparse subsurface conditioning signal. |
| Sparse profile salinity | EN4 / ARGO | `PSAL_CORRECTED`, projected to GLORYS depths | Auxiliary profile context stored with the training data. |
| Surface temperature | OSTIA | `analysed_sst` | First dense surface-conditioning channel (`sst`). |
| Target ocean temperature | GLORYS | `thetao` | Normal dense target; approved fitting years may also contribute aggregate-only prior statistics. |
| Reanalysis salinity | GLORYS | `so` | Normal dense salinity target and aggregate-only prior-fitting source. |
| Sea-surface height | Sea Level L4 | `adt` | Third dense surface-conditioning channel (`adt`). |
| Sea-surface salinity/density | SSS MULTIOBS | `sos`, `dos` | `sos` is the second conditioning channel (`sss`); `dos` remains auxiliary. |
| Land/ocean mask | Rasterized world polygons | `output_land_mask` | Defines patch candidates and provides final output cleanup support; model-facing `land_mask` is derived from finite GLORYS target support. |

Temperature is kept physically in Kelvin in the exported GeoTIFF dataset, then
converted or normalized by the loader as needed for model training. Salinity is
stored in PSU and normalized only when the GeoTIFF dataloader is configured with
`--scenario salinity` or `--scenario joint`.

## Shared Axes

All dense exported rasters use the committed global 0.1 degree land-mask grid:

```text
src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

This grid gives a single spatial reference for GLORYS, OSTIA, sea-level, SSS
fields, and ARGO profile locations. The vertical axis is the 50-level GLORYS depth
coordinate. ARGO profiles are interpolated onto those same depths before they
are rasterized into model patches.

## Temporal Cadence

GLORYS weekly dates define the training timeline. For every GLORYS target date:

- GLORYS `thetao` and `so` are saved for that weekly date.
- OSTIA `analysed_sst` is saved as a centered 7-day mean around that date.
- Sea-level `adt` is saved as a centered 7-day mean around that date.
- SSS `sos` and `dos` are saved as centered 7-day means around that date.
- ARGO profiles are assigned to the nearest GLORYS weekly date inside the same
  centered temporal window.

## Training View

The model-facing training sample is a patch cut from the shared grid. By
default it contains sparse ARGO temperature observations and dense `[SST, SSS, ADT]` surface
context. Normal dense training uses the GLORYS temperature target; optional Stage 1 uses an online deterministic surface-offset target. Masks tell the model
which values are observed, supervised, ocean, or missing. With
`--scenario salinity`, the loader skips temperature tensors and returns normalized
`x_salinity` and `y_salinity` plus salinity-specific masks. With
`--scenario joint`, it returns both temperature and salinity tensors, and
`PixelDiffusionConditional` stacks them at the model boundary.

The precise tensor contract is intentionally separated from this overview. See
[Data Contract](data-contract.md) for shapes, normalization, masks, and loader
assembly rules.
