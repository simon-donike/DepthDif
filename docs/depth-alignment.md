# Depth Alignment  
This page documents how EN4 / ARGO temperature profiles are projected onto the GLORYS depth grid.  

Use [Data Sources](data-source.md) for native product properties and [Production Dataset](production-datasets.md) for the spatial and temporal assembly pipeline.  

## Native Vertical Coordinates  
- GLORYS uses one fixed, monotonic 50-level `depth` coordinate.  
- EN4 / ARGO stores profile samples at profile-specific `DEPH_CORRECTED` depths.  
- EN4 profile arrays may have up to `400` storage slots, but those slots are not a shared physical depth axis.  

## Target Grid  
- The raw dataset uses the full 50 GLORYS depth levels as the target channel axis.  
- Depth alignment is applied profile-by-profile before spatial rasterization.  
- `x`, `y`, `x_valid_mask`, and `y_valid_mask` therefore share the same GLORYS-aligned depth layout.  

## Per-Profile Alignment Procedure  
1. Read finite `(DEPH_CORRECTED, POTM_CORRECTED)` pairs from one EN4 / ARGO
   profile. `TEMP` remains available only for explicitly requested legacy
   in-situ exports.
2. Sort the samples by depth and collapse duplicate depths.  
3. For each GLORYS target depth inside the observed profile range, linearly interpolate temperature.  
4. Accept the interpolated value only when the nearest observed ARGO depth satisfies `abs(nearest_depth - target_depth) <= max(0.1 * target_depth, 10 m)`.  
5. Leave out-of-range or rejected targets invalid; no depth extrapolation is applied.  

## Output Semantics  
- During GeoTIFF export, ARGO is resampled onto the GLORYS depth axis before patch rasterization.
- `x_valid_mask` marks aligned Argo depths that passed the profile-range and nearest-depth checks.  
- The model-facing `x`, `y`, `x_valid_mask`, and `y_valid_mask` tensors share this GLORYS depth axis.  

## Diagnostics

The contributor plotting CLIs can inspect corrected-depth distributions,
ARGO-to-GLORYS channel mapping, and target-alignment shifts directly from a
selected dataset. Invoke the `plot_argo_corrected_depth_*`,
`plot_argo_glorys_depth_mapping`, or `plot_glorys_target_alignment_shift`
modules listed in the [CLI reference](cli.md). Generated figures and reports are
run artifacts, not part of the saved dataset contract.
