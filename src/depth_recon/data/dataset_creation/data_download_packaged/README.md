# Packaged Dataset Downloads

This folder is reserved for future scripts that download packaged DepthDif
datasets, for example from Hugging Face. Raw upstream source-data download
scripts live in `../data_download_raw/`.

## Packaged Dataset Downloaders

Hosted dataset links are read from `dataset_links.yaml`. Edit that file to
change where these scripts download from.

Download and extract the hosted aligned ARGO zarr archive from Hugging Face:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
  --output-dir /work/data/depthdif/aligned_argo
```

Download and extract the exported GeoTIFF dataset zip from the public Google
Drive link configured in `dataset_links.yaml`:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
  --output-dir /work/data/depthdif/geotiff_export
```
