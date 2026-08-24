# Aligned ARGO Export

This directory owns the three aligned-profile workflow steps:

1. `a_check_export_sourcefiles.py` validates date coverage and source files.
2. `b_export_enriched_argo_profiles.py` aligns profiles to GLORYS depths and
   writes both the enriched profile Zarr and compact grid-indexed training Zarr.
3. `c_package_huggingface_aligned_argo.py` assembles the profile and optional
   raster products without changing the saved schemas.

The production temperature source is corrected EN4 `POTM_CORRECTED`, sampled at
`DEPH_CORRECTED` and projected onto GLORYS depths. Use
`--temperature-source in-situ` only for explicit reproduction of older `TEMP`
exports.

Create the aligned stores:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles \
  --argo-dir /data1/datasets/depth_v2/en4_profiles \
  --glorys-dir /data1/datasets/depth_v2/glorys_weekly \
  --ostia-dir /data1/datasets/depth_v2/ostia \
  --sealevel-dir /data1/datasets/depth_v2/sealevel_daily \
  --sss-dir /data1/datasets/depth_v2/sss_daily \
  --output-zarr /work/data/depthdif/enriched_argo_profiles.zarr \
  --compact-output-zarr /work/data/depthdif/argo/argo_profiles_on_grid.zarr \
  --compact-land-mask-path src/depth_recon/data/dataset_creation/data_download_raw/get_world/world_land_mask_glorys_0p1.tif \
  --start-date 20100101 --end-date 20240731 \
  --temperature-source potential --workers 4 --overwrite
```

To package only the aligned Zarr, run:

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
  assets/figures/depthdif_schema.webp
  assets/data/geotiff_dataset_random100_surface.webp
  assets/data/argo_on_glorys_grid_3D.gif
  assets/data/profile_comparison_good_alignment.webp
  assets/data/profile_comparison_bad_alignment.webp
```

The enriched Zarr store is unchanged, including the SSS variables `sss_sos`, `sss_dos`,
`sss_sea_ice_fraction`, and `sss_temporal_status`. The GeoTIFF exporter can read
the packaged copy:

```bash
--enriched-argo-zarr /data1/datasets/depth_v2/aligned_argo/hf_argo_glors_ostia_ssh/data/argo_glors_ostia_ssh.zarr
```

For the self-contained training package, also pass `--raster-root`,
`--compact-argo-zarr`, `--manifest-path`, and `--masks-dir` as shown in the
parent dataset-creation README.
