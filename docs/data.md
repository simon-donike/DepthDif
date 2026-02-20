# Data
Currently, monthly tiles from 2000 - 2025 from the [Global Ocean Physics Reanalysis dataset](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1M-m_202311&path=GLOBAL_MULTIYEAR_PHY_001_030%2Fcmems_mod_glo_phy_my_0.083deg_P1M-m_202311%2F2024%2F) have been downloaded and are manually masked to simulate real sparse observations. Excluding patches with >20% NoData values, ~106k samples are avaialble (128x128, 1/12 °). Download the data by installing the `copernicusmarine` package, then use the CLI like so `copernicusmarine get -i cmems_mod_glo_phy_my_0.083deg_P1M-m  --filter "*2021/*"`  
The of the obstructions and the coverage percentage are selectable in the `data_config.yaml`.

The validation split is geographically coherent: the same spatial locations/windows across timesteps are assigned to either train or val, not both.

Dataset example for 50% occlusion:  
![img](assets/dataset_50percMask.png)  

## Implemented dataset/task modes
There are currently two implemented training tasks:

1. **Straight corrupted -> uncorrupted (single-band)**  
   Uses `SurfaceTempPatchLightDataset` (`temp_v1` style):
   - `x`: corrupted temperature band
   - `y`: clean temperature band (ground truth)
   - optional masks: `valid_mask`, `land_mask`

2. **EO-conditioned multiband reconstruction**  
   Uses `SurfaceTempPatch4BandsLightDataset` (`eo_4band` style):
   - `eo`: first band used as extra condition (surface/EO-like observation)
   - `x`: corrupted deeper temperature bands
   - `valid_mask`: mask channel used as additional condition
   - `y`: clean deeper temperature bands (ground truth)
   - optional masks: `valid_mask`, `land_mask`

EO + multiband dataloader example (`eo` + deeper levels as corrupted/target):
![img](assets/eo_dataset_example.png)

In config, switch this via `dataset.dataset_variant`:
- `temp_v1` for single-band corrupted->clean
- `eo_4band` for EO + multiband conditioning

## Dataset tweaks
- Synthetic occlusion pipeline to create sparse observations with configurable `mask_fraction`.
- Patch-based masking with min/max patch sizes (`mask_patch_min`, `mask_patch_max`) instead of single-pixel drops.
- Validity/land masks derived from nodata or fill values; invalid pixels are tracked separately from corruption.
- Optional filtering of tiles by `max_nodata_fraction` to avoid overly invalid patches.
- Corrupted input + mask channel return modes for conditional modeling (`x_return_mode`).
- Z-score temperature normalization and optional geometric augmentation (rotations/flips) applied consistently to data and masks.
- Dataset index build with nodata-fraction metadata for fast filtering.
- Geographically coherent split support via the index `split` column (same location kept in one split over time).
- Optional patch-center coordinate return (`return_coords`) using index columns (`lat0/lat1/lon0/lon1`) with dateline-safe longitude center computation.
