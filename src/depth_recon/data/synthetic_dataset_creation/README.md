# Synthetic Pretraining GeoTIFFs

Create SST/SSS-guided synthetic dense targets inside an existing packaged GeoTIFF dataset.
The exporter writes `uint8` multiband GeoTIFFs under `rasters/synthetic/` and updates
`manifest.yaml` with `rasters.synthetic.thetao` and `rasters.synthetic.so`. Original
`rasters.glorys` entries are preserved. ARGO/EN4 profiles provide vertical deltas
only; salinity uses same-date SSS `sos` as its dense surface anchor before ARGO
vertical salinity deltas are applied. Synthetic nodata masks are copied from the
matching GLORYS target rasters after synthesis, and final smoothing is intentionally
light.

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.synthetic_pretraining_geotiff \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --workers 4 \
  --overwrite-synthetic
```

Training uses real GLORYS targets by default. To use these synthetic targets, set
`data.dataset.sampling.target_source: synthetic` in the pixel training config or an
override config.
