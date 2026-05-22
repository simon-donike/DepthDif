# Aligned ARGO Export

`b_export_enriched_argo_profiles.py` writes the enriched ARGO profile zarr used
as input by the GeoTIFF export. It performs the ARGO-to-GLORYS depth alignment
and samples GLORYS, OSTIA, sea-level, and SSS context at each profile point.
`c_package_huggingface_aligned_argo.py` only repackages that finished zarr for
Hugging Face; it does not redo alignment or change the schema. To package the
zarr, run:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo \
  --input-zarr /data1/datasets/depth_v2/aligned_argo/enriched_argo_profiles.zarr \
  --output-dir /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh \
  --zarr-name argo_glors_ostia_ssh.zarr \
  --file-mode hardlink \
  --overwrite
```

The package folder contains:

```text
hf_argo_glors_ostia_ssh/
  README.md
  LICENSE
  data/argo_glors_ostia_ssh.zarr/
  indices/profiles.parquet
  indices/variables.parquet
  examples/open_with_xarray.py
  examples/subset_by_region_time.py
  metadata/dataset_description.json
  metadata/citation.cff
  metadata/stac-item.json
  assets/figures/depthdif_schema.png
  assets/data/geotiff_dataset_random100_surface.png
  assets/data/argo_on_glorys_grid_3D.gif
  assets/data/profile_comparison_good_alignment.png
  assets/data/profile_comparison_bad_alignment.png
```

The zarr store is unchanged, including the SSS variables `sss_sos`, `sss_dos`,
`sss_sea_ice_fraction`, and `sss_temporal_status`. The GeoTIFF exporter can read
the packaged copy:

```bash
--enriched-argo-zarr /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
```
