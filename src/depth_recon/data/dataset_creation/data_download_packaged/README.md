# Packaged Dataset Downloads

This folder contains scripts that download packaged DepthDif datasets from the
official Hugging Face dataset repository. Raw upstream source-data download scripts
live in `../data_download_raw/`.

## Packaged Dataset Downloaders

Hosted dataset links are read from `dataset_links.yaml`. Edit that file to
change where these scripts download from.

Download the hosted Hugging Face aligned ARGO package folder:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr \
  --output-dir /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh
```

The downloaded package keeps the HF layout on disk. The enriched ARGO zarr is
located at:

```text
/data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
```

Use that zarr directly as `--enriched-argo-zarr` for the GeoTIFF export. It
contains the GLORYS, OSTIA, sea-level, and SSS profile-context variables from
the original enriched ARGO export.

Download the full exported GeoTIFF training dataset from the official Hugging
Face repository configured in `dataset_links.yaml`:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset \
  --output-dir /work/data/OceanVariableReconstruction
```
