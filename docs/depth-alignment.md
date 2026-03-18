# Depth Alignment  
This page explains how the vertical dimensions of EN4 / ARGO and GLORYS relate to each other and how the production workflow should handle them.  
  
The central issue is simple:  
- GLORYS has one fixed, monotonic 50-level depth grid  
- EN4 / ARGO profiles are irregular in physical depth  
- raw EN4 storage-slot indices are therefore not a valid production depth grid by themselves  
  
Use [Data Sources](data-source.md) for the raw-product overview and [Production Dataset](production-datasets.md) for the spatial and temporal sampling pipeline.  
  
## Why Alignment Is Needed  
If ARGO is used as the ground truth, the model still needs a stable channel layout. That means every sample must ultimately live on one shared target depth grid.  
  
The recommended production choice is:  
- use the fixed GLORYS `depth` axis as the common target grid  
- interpolate each ARGO profile independently onto that grid  
- keep a validity mask for target depths outside the supported ARGO range  
  
This keeps the physical meaning clean:  
- ARGO provides the target values  
- GLORYS provides the canonical depth coordinates  
- both sources then share one stable channel definition  
  
## What `DEPH_CORRECTED` Means  
`DEPH_CORRECTED` is the corrected physical depth assigned to each observation after EN4 processing. It should be interpreted as:  
- the depth of that particular observed sample  
- not a universal ARGO depth axis  
- not a guarantee that slot `k` has the same physical meaning across all profiles  
  
## Why The EN4 Grid Is Tricky  
EN4 stores profile arrays in rectangular form:  
- shape like `(N_PROF, 400)`  
- `400` is storage capacity, not a common physical 400-level ocean grid  
- each individual profile only fills the slots it actually observed  
- different profiles can therefore have:  
  - different valid depth counts  
  - different exact depth values  
  - slightly different local monotonic behavior after QC and correction  
  
This is why raw EN4 slot indices should not be used directly as production depth channels.  
  
## Archive-Wide Corrected-Depth Distribution  
Archive-wide corrected-depth distribution:  
![img](assets/argo_corrected_depth_distribution.png)  
  
How to read this figure:  
- top panel: how many valid profile values contribute to each EN4 slot index across the scanned archive  
- lower-left heatmap: where corrected depths concentrate for each EN4 slot index; brighter regions mean more observations  
- white line: approximate median corrected depth per EN4 slot  
- dashed lines: approximate `p10` and `p90` depth envelopes per slot  
- right panel: overall depth histogram across all valid corrected-depth samples  
  
Interpretation:  
- if EN4 slot index were a universal physical depth grid, the median curve would rise smoothly and monotonically  
- the visible local dips and broad spread show that EN4 slot index is only a storage coordinate and that the realized physical depth varies across profiles  
- this is the main reason ARGO must be interpolated onto a shared target depth grid instead of treating raw slot indices as stable channels  
  
## Example ARGO Profile Structure  
Example 3D ARGO profile visualization:  
![img](assets/argo_profile_3D.gif)  
  
How to read this figure:  
- each occupied voxel corresponds to one observed ARGO value in a rasterized profile tensor  
- the vertical axis is depth level, with sparse occupied cells showing that observations do not fill a dense regular 3D volume  
  
Interpretation:  
- this gives an intuitive view of why ARGO behaves like sparse, irregular profile observations rather than a complete dense depth cube  
- it is a qualitative illustration of the sampling structure, not a canonical depth-grid definition  
  
## Recommended Production Handling  
If ARGO is the ground truth, the clean production choice is:  
1. Treat the fixed GLORYS `depth` axis as the common target grid.  
2. For each EN4 / ARGO profile independently:  
   - extract valid `(DEPH_CORRECTED, TEMP)` pairs  
   - sort by depth  
   - interpolate temperature onto the GLORYS depth levels  
3. Mask target depths outside the supported ARGO range instead of extrapolating fake values.  
  
This preserves the scientific meaning:  
- ARGO still provides the ground-truth values  
- GLORYS only provides the common depth coordinates  
- every aligned profile then has one stable channel layout  
  
## Current Production Depth-Grid Decision  
For the current production setup, the working decision is:  
- restrict the aligned target grid to the GLORYS depths below `1000 m`  
- for each retained GLORYS target depth, use the nearest available ARGO depth recording as the observation-side match  
- keep this truncated shallow-to-mid-depth regime as the model depth axis so the vertical scale stays focused on the part of the water column where sampling density and practical signal are strongest  
  
Current consequence of that decision:  
- the production-aligned depth axis keeps `39` depth levels  
  
Interpretation:  
- this is a pragmatic modeling choice, not a claim that EN4 / ARGO becomes a fixed 39-level native grid  
- the underlying ARGO observations remain irregular in depth  
- the retained 39 levels are the production target channels chosen after restricting the common grid to the sub-`1000 m` regime  
- deeper GLORYS levels are intentionally excluded from the current aligned target definition  
  
## Profile-Level Nearest-Match Diagnostic  
This diagnostic is computed profile-by-profile rather than from EN4 slot-index aggregates:  
- for each EN4 / ARGO profile and each retained GLORYS target depth, find the closest observed `DEPH_CORRECTED` value  
- if a maximum valid-distance cutoff is active, discard nearest matches that are still too far away  
- aggregate only the accepted matches across profiles  
  
![img](assets/glorys_target_alignment_depth_summary.png)  
  
How to read this figure:  
- top panel: the fixed GLORYS target depth at each retained level, overlaid with the mean nearest accepted ARGO depth across profiles  
- blue band: `±1 std` of the nearest accepted ARGO depth, showing how much that nearest observation varies across the archive  
- bottom panel: mean absolute distance between each GLORYS target depth and the closest accepted ARGO observation, again with a `±1 std` band  
- when a cutoff is active, profiles whose nearest ARGO observation is still too far away are excluded from both panels rather than being averaged in  
  
Interpretation:  
- this is the more physically meaningful diagnostic because it works at the individual-profile level rather than treating EN4 slot index as if it were a shared depth axis  
- the mean nearest ARGO curve broadly follows the GLORYS depth curve, which shows that the shallow-to-mid-depth GLORYS grid is a workable target grid for resampled ARGO profiles  
- the spread and average nearest-depth mismatch both grow with depth, which indicates that the fixed GLORYS levels become a looser approximation of the irregular ARGO sampling deeper in the water column  
- in the saved summary, shallow levels are matched much more consistently than deep ones: the accepted-match fraction is above roughly `96%` near the surface and falls to about `36%` by the deepest retained levels, while the mean absolute accepted mismatch grows from roughly `1-3 m` near the surface to around `5 m` close to `900 m`  
- this supports the production choice to restrict the aligned depth axis to the sub-`1000 m` regime and to reject nearest matches that are still too far away, instead of forcing every GLORYS target depth to use a weak ARGO proxy  
  
### Why Not Use Raw EN4 Slot Indices As Channels  
Because EN4 slot index is a storage coordinate, not a physical vertical coordinate:  
- aggregating by slot index across all profiles mixes different realized depths  
- representative curves by slot index can show local dips or other non-monotonic artifacts  
- those artifacts are properties of the archive-wide aggregation, not a valid universal ocean depth grid  
  
### Why GLORYS Is The Practical Canonical Grid  
- fixed and monotonic across files  
- already physically meaningful  
- avoids inventing a second custom ARGO-derived grid  
- lets both sources live on the same depth channels after interpolation  
  
## Saved Alignment Artifacts  
The following artifacts summarize how the raw EN4 archive relates to the GLORYS grid:  
- `data/glorys_argo_alignment/argo_to_glorys_channel_mapping.json`  
- `data/glorys_argo_alignment/glorys_argo_alignment_report.txt`  
- `data/glorys_argo_alignment/argo_glorys_depth_alignment.csv`  
- `data/glorys_argo_alignment/argo_depth_level_summary.csv`  
- `data/glorys_argo_alignment/glorys_depth_coverage_summary.csv`  
- `data/glorys_argo_alignment/argo_corrected_depth_distribution.png`  
- `data/glorys_argo_alignment/figures/glorys_target_alignment_depth_summary.png`  
- `data/glorys_argo_alignment/figures/glorys_target_alignment_shift_summary.csv`  
- `data/glorys_argo_alignment/figures/glorys_target_alignment_within_cutoff_fraction.png`  
  
Helper scripts:  
- `data/EDA_glorys_argo_alignment.py`  
- `data/plot_argo_corrected_depth_distribution.py`  
- `utils/plot_argo_glorys_depth_mapping.py`  
- `utils/plot_glorys_target_alignment_shift.py`  
  
## Alignment Diagnostics  
![img](assets/argo_glorys_depth_vs_index.png)  
![img](assets/argo_glorys_absolute_difference.png)  
![img](assets/argo_glorys_depth_scatter.png)  
![img](assets/argo_level_valid_profile_count.png)  
  
How to read these diagnostics:  
- `argo_glorys_depth_vs_index.png`:  
  - compares the representative ARGO depth assigned to each EN4 slot against the nearest matched GLORYS depth  
  - use it to see where the two vertical layouts broadly track each other and where the approximation becomes coarse  
- `argo_glorys_absolute_difference.png`:  
  - shows the absolute depth mismatch between representative ARGO depth and matched GLORYS depth for every EN4 slot  
  - low values indicate good local alignment; peaks indicate ranges where nearest-neighbor channel matching is physically rough  
- `argo_glorys_depth_scatter.png`:  
  - parity-style comparison between representative ARGO depths and matched GLORYS depths  
  - points near the diagonal mean close agreement; visible offsets from the diagonal indicate systematic mismatch at those depths  
- `argo_level_valid_profile_count.png`:  
  - shows how many profiles contributed to the representative depth estimate for each EN4 slot  
  - high counts mean the representative depth is supported by many profiles; low counts indicate that the estimate is based on rarer deep observations and should be interpreted more cautiously  
