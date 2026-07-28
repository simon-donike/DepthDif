# Synthetic Pretraining GeoTIFFs

Create SST/SSS-guided synthetic dense targets inside an existing packaged GeoTIFF dataset.
The exporter writes `uint8` multiband GeoTIFFs under `rasters/synthetic/` and updates
`manifest.yaml` with `rasters.synthetic.thetao` and `rasters.synthetic.so`. Original
`rasters.glorys` entries are preserved. ARGO/EN4 profiles provide vertical deltas
only; per-depth delta outliers are trimmed before IDW interpolation. Salinity uses
same-date SSS `sos` as its dense surface anchor before ARGO vertical salinity
deltas are applied. Missing per-profile delta depths are filled by vertical
interpolation with edge holding before spatial IDW. Synthetic nodata masks are
copied from the matching GLORYS target rasters after synthesis. No Gaussian
smoothing is applied in the current experiment; spatial interpolation uses CUDA
IDW when available and falls back to CPU otherwise.

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.synthetic_pretraining_geotiff \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --workers 1 \
  --overwrite-synthetic
```

Training uses real GLORYS targets by default. To use these synthetic targets, set
`data.dataset.sampling.target_source: synthetic` in the pixel training config or an
override config.
