# Packaged Dataset Downloads

This folder is reserved for future scripts that download packaged DepthDif
datasets, for example from Hugging Face. Raw upstream source-data download
scripts live in `../data_download_raw/`.

## Packaged Dataset Downloaders

Download and extract the hosted aligned ARGO zarr archive from Hugging Face:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
  --output-dir /work/data/depthdif/aligned_argo
```

Download and extract the future exported GeoTIFF dataset zip from public Google
Drive hosting. The default URL is still a placeholder, so pass the hosted link
with `--url` once it is available:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
  --output-dir /work/data/depthdif/geotiff_export \
  --url https://drive.google.com/file/d/GOOGLE_DRIVE_FILE_ID/view?usp=sharing
```
