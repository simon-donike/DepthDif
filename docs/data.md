# Data Overview

DepthDif learns to densify sparse in-situ ocean profiles into gridded
temperature fields. The training data combines satellite surface context,
profile observations, ocean reanalysis targets, and auxiliary sea-surface
height on one shared global grid.

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
| Surface temperature | OSTIA | `analysed_sst` | Dense surface conditioning signal. |
| Target ocean temperature | GLORYS | `thetao` | Dense 3D supervision target. |
| Reanalysis salinity | GLORYS | `so` | Dense aligned ocean state variable stored alongside temperature. |
| Sea-surface height | Sea Level L4 | `adt` | Dense surface-height context stored on the same grid. |
| Land/ocean mask | Rasterized world polygons | `land_mask` | Defines patch candidates and excludes land-heavy regions. |

Temperature is kept physically in Kelvin in the exported GeoTIFF dataset, then
converted or normalized by the loader as needed for model training.

## Shared Axes

All dense exported rasters use the committed global 0.1 degree land-mask grid:

```text
src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif
```

This grid gives a single spatial reference for GLORYS, OSTIA, sea-level fields,
and ARGO profile locations. The vertical axis is the 50-level GLORYS depth
coordinate. ARGO profiles are interpolated onto those same depths before they
are rasterized into model patches.

## Temporal Cadence

GLORYS weekly dates define the training timeline. For every GLORYS target date:

- GLORYS `thetao` and `so` are saved for that weekly date.
- OSTIA `analysed_sst` is saved as a centered 7-day mean around that date.
- Sea-level `adt` is saved as a centered 7-day mean around that date.
- ARGO profiles are assigned to the nearest GLORYS weekly date inside the same
  centered temporal window.

## Training View

The model-facing training sample is a patch cut from the shared grid. It
contains sparse ARGO temperature observations, dense OSTIA surface temperature,
the dense GLORYS temperature target, and masks that tell the model which values
are observed, supervised, ocean, or missing.

The precise tensor contract is intentionally separated from this overview. See
[Data Contract](data-contract.md) for shapes, normalization, masks, and loader
assembly rules.
