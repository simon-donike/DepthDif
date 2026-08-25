# CLI Reference

Run repository commands with `/work/envs/depth/bin/python`. Every parser listed
below supports `--help`; that output is authoritative for optional flags and
defaults.

## Installed console scripts

| Command | Purpose |
| --- | --- |
| `depth-recon-infer-week` | Run public legacy-checkpoint inference for one ISO week. |
| `depth-recon-download-argo` | Download the EN4 archives needed by one ISO week. |
| `depth-recon-download-ostia` | Download the OSTIA inputs needed by one ISO week. |
| `depth-recon-export-paper-week` | Export a configured paper-week model bundle. |
| `depth-recon-export-paper-metrics` | Compute metrics from paper-week artifacts. |

```bash
depth-recon-infer-week --help
```

## Training and diagnostics

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train_autoencoder.py
/work/envs/depth/bin/python src/depth_recon/scripts/benchmark_dataloader_settings.py --help
```

`train.py` is the maintained pixel/baseline training entry point. Autoencoder
and latent components are experimental; see [Autoencoder](autoencoder.md).

The complete held-out-2016 baseline workflow uses two independent GPU workers,
stable W&B run identities, resumable checkpoints, and the paper/spectral export
pipeline:

```bash
/work/envs/depth/bin/python src/depth_recon/scripts/run_baseline_2016_suite.py \
  --phase all --gpu-indices 0 1 \
  --output-root logs/baseline_2016_global \
  --resume-incomplete
```

Use `--dry-run` to inspect all model/scenario commands without requiring CUDA or
W&B. Normal execution requires both requested GPUs and valid online W&B
authentication; it never falls back to CPU.

## Packaged data downloads

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_aligned_argo_zarr --help
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.data_download_packaged.download_exported_geotiff_dataset --help
```

Raw Copernicus/EN4 download shell workflows live below
`data_download_raw/`. They are documented in [Dataset downloads](data-download.md)
because their credentials, product candidates, and date ranges differ.

## Dataset production

Check source coverage, export enriched profiles, export dense rasters, and package
the result in that order:

```bash
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.a_check_export_sourcefiles --help
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.b_export_enriched_argo_profiles --help
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_dataset_geotiff.export_dataset_geotiff --help
/work/envs/depth/bin/python -m depth_recon.data.dataset_creation.export_aligned_argo.c_package_huggingface_aligned_argo --help
```

The production default temperature source is `potential`, backed by EN4
`POTM_CORRECTED`. Select `in-situ` only to reproduce older `TEMP` exports.

## Synthetic prior

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior --help
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.plot_vertical_offset_examples --help
```

## Model inference and exports

```bash
/work/envs/depth/bin/python -m depth_recon.inference.run_single
/work/envs/depth/bin/python -m depth_recon.inference.export_global --help
/work/envs/depth/bin/python -m depth_recon.inference.export_global_variables --help
/work/envs/depth/bin/python -m depth_recon.inference.export_validation_error_summary --help
```

The single-variable exporter writes one stitched run. The paired wrapper owns the
temperature/salinity production bundle and can trigger temporal and spectral
products.

## Paper and comparative evaluation

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_paper_week --help
/work/envs/depth/bin/python -m depth_recon.inference.export_paper_metrics --help
/work/envs/depth/bin/python -m depth_recon.inference.export_spectral_comparison_bundle --help
```

These commands require explicit models/checkpoints and preserve their run
metadata. `export_paper_metrics` has both bundle and legacy per-method modes;
prefer bundle mode for new results.

## Analysis bundles

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_error_analysis_dashboard --help
/work/envs/depth/bin/python -m depth_recon.inference.export_cesium_globe_assets --help
/work/envs/depth/bin/python -m depth_recon.inference.export_wavenumber_spectra --help
/work/envs/depth/bin/python -m depth_recon.inference.export_temporal_global_variables --help
/work/envs/depth/bin/python -m depth_recon.inference.export_temporal_consistency_dashboard --help
/work/envs/depth/bin/python -m depth_recon.inference.export_temporal_cesium_globe_assets --help
```

`export_wavenumber_spectra` includes incomplete ocean patches by default and
masks invalid pixels during processing. Pass `--require-complete-patches` for the
stricter alternate policy.

## Developer visualization utilities

Sampler-comparison and plotting CLIs are contributor diagnostics rather than
production inference entry points:

```bash
/work/envs/depth/bin/python -m depth_recon.utils.compare_ddpm_ddim_sampling --help
/work/envs/depth/bin/python -m depth_recon.experiments.compare_ddpm_ddim_step_grid --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_argo_corrected_depth_distribution --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_argo_corrected_depth_histogram --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_argo_glorys_depth_mapping --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_glorys_target_alignment_shift --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_land_fraction_filter_examples --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.plot_loss_explanations --help
/work/envs/depth/bin/python -m depth_recon.utils.visualization.create_paper_header_image --help
```

Examples with all options are kept at the top of each task-specific script. The
diagnostic samplers do not define quality thresholds or a recommended universal
step count.
