# GLORYS/Argo Aligned Experiments  

The maintained GLORYS/ARGO workflow now uses  
`ArgoNetCDFGriddedPatchDataset` directly against NetCDF sources. It keeps  
GLORYS `thetao` as `y`, ARGO `TEMP` projected onto the GLORYS depth axis as  
`x`, and OSTIA `analysed_sst` as `eo`.  

Use `src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml` for new aligned runs.  
Depth-axis details are documented in [Depth Alignment](depth-alignment.md).  
