# Ocean Depth Reconstruction Dataset Code

This repository branch contains the code associated with the dataset-paper submission: raw source downloads, ARGO/profile alignment, GeoTIFF dataset export, the dataset loader, and IDW, LSTM, and 3D U-Net baselines. It intentionally excludes the original diffusion model, hosted dashboard/export tooling, notebooks, deployment assets, install/build metadata, and a separate docs site.

The source tree keeps the import name `depth_recon` for compatibility with the existing scripts. Default hosted artifact links have been anonymized; replace placeholder dataset URLs in `src/depth_recon/data/dataset_creation/data_download_packaged/dataset_links.yaml` with review-safe artifact URLs.

## Environment

Use Python 3.12 from the repository root. Install dependencies into your environment, then run scripts with `src` on `PYTHONPATH` instead of installing this repository as a package:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
```

## Repository Layout

- `src/depth_recon/data/dataset_creation/data_download_raw/`: shell/Python helpers for raw EN4/ARGO, GLORYS, OSTIA, sea-level, SSS, and land-mask sources.
- `src/depth_recon/data/dataset_creation/data_download_packaged/`: helpers for downloading hosted review artifacts listed in `dataset_links.yaml`.
- `src/depth_recon/data/dataset_creation/export_aligned_argo/`: source checks, ARGO/profile alignment, enriched Zarr export, compact ARGO-on-grid export, and optional review-artifact staging.
- `src/depth_recon/data/dataset_creation/export_dataset_geotiff/`: model-ready GeoTIFF raster export, masks, and manifest writing.
- `src/depth_recon/data/dataset_argo_geotiff_gridded.py`: lazy GeoTIFF plus aligned-ARGO patch dataset used by baselines.
- `src/depth_recon/models/baselines/`: IDW, point-wise LSTM, and 3D U-Net baselines.
- `src/depth_recon/configs/px_space/`: scenario-aware super-configs for temperature, salinity, and joint runs.
- `train.py`: baseline training entry point.

## Dataset Pipeline

The dataset pipeline is implemented under `src/depth_recon/data/dataset_creation/`. Raw source download scripts collect upstream EN4/ARGO, GLORYS, OSTIA, sea-level, SSS, and land-mask inputs. The alignment exporter maps ARGO profiles onto the GLORYS depth axis, collocates auxiliary fields, and writes an enriched Zarr store plus a compact ARGO-on-grid Zarr used by the training dataset. The GeoTIFF exporter writes uint8-stretched rasters, masks, and `manifest.yaml` for lazy patch loading.

Packaged download helpers read artifact URLs from `data_download_packaged/dataset_links.yaml`; those URLs are placeholders in this anonymous branch and should be replaced with review-safe hosted artifacts.

1. Download raw source files with the scripts under `data_download_raw/`.
2. Check source coverage:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles
```

3. Align ARGO profiles to GLORYS depth levels and collocate auxiliary variables:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles \
  --argo-dir ./data/raw/en4_profiles \
  --glorys-dir ./data/raw/glorys_weekly \
  --ostia-dir ./data/raw/ostia \
  --sealevel-dir ./data/raw/sealevel_daily \
  --sss-dir ./data/raw/sss_daily \
  --output-zarr ./data/ocean_depth_reconstruction/enriched_argo_profiles.zarr \
  --compact-output-zarr ./data/ocean_depth_reconstruction/argo/argo_profiles_on_grid.zarr
```

4. Export the model-ready GeoTIFF dataset:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff \
  --glorys-dir ./data/raw/glorys_weekly \
  --ostia-dir ./data/raw/ostia \
  --sealevel-dir ./data/raw/sealevel_daily \
  --sss-dir ./data/raw/sss_daily \
  --enriched-argo-zarr ./data/ocean_depth_reconstruction/enriched_argo_profiles.zarr \
  --output-dir ./data/ocean_depth_reconstruction
```

5. Optionally assemble the anonymous review artifact folder:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo \
  --input-zarr ./data/ocean_depth_reconstruction/enriched_argo_profiles.zarr \
  --raster-root ./data/ocean_depth_reconstruction/rasters \
  --compact-argo-zarr ./data/ocean_depth_reconstruction/argo/argo_profiles_on_grid.zarr \
  --manifest-path ./data/ocean_depth_reconstruction/manifest.yaml \
  --masks-dir ./data/ocean_depth_reconstruction/masks \
  --output-dir ./data/review_artifact
```

## Baselines

All baselines consume batches from `ArgoGeoTIFFGriddedPatchDataset` and return the same normalized and denormalized prediction keys used by the retained evaluation helpers.

- `idw_baseline`: checkpoint-free inverse-distance interpolation over sparse ARGO observations.
- `lstm_baseline`: trainable point-wise vertical LSTM with optional EO surface context.
- `unet_baseline`: trainable 3D U-Net that treats field, depth, height, and width as a volume-inpainting problem.

Train a baseline by selecting `model.model_type` in the super-config or CLI override. Validation loaders remain shuffled by default in the provided configs.

```bash
/work/envs/depth/bin/python train.py --scenario temperature --set model.model_type=idw_baseline
/work/envs/depth/bin/python train.py --scenario temperature --set model.model_type=lstm_baseline
/work/envs/depth/bin/python train.py --scenario temperature --set model.model_type=unet_baseline
```

The `--scenario` selector supports `temperature`, `salinity`, and `joint`; it derives output fields, salinity loading, EO source selection, generated channels, and baseline condition channels.

## Tests

Run the retained test suite with:

```bash
tests/run_tests.sh
```
