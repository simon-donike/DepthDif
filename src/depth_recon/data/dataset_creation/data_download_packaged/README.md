# Packaged Dataset Downloads

This folder contains scripts that download packaged Ocean Depth Reconstruction datasets, including
the Hugging Face aligned ARGO package. Raw upstream source-data download scripts
live in `../data_download_raw/`.

## Packaged Dataset Downloaders

Hosted dataset links are read from `dataset_links.yaml`. Edit that file to
change where these scripts download from.

Download the hosted Hugging Face aligned ARGO package folder:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
  --output-dir ./data/aligned_argo
```

The downloaded package keeps the HF layout on disk. The enriched ARGO zarr is
located at:

```text
./data/aligned_argo/data/aligned_argo_profiles.zarr
```

Use that zarr directly as `--enriched-argo-zarr` for the GeoTIFF export. It
contains the GLORYS, OSTIA, sea-level, and SSS profile-context variables from
the original enriched ARGO export.

Download and extract the exported GeoTIFF dataset zip from the public Google
Drive link configured in `dataset_links.yaml`:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
  --output-dir ./data/ocean_depth_reconstruction
```
