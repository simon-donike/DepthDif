# Tests

This page documents the repository test coverage added around diffusion math, dataset behavior, config wiring, and one-batch model smoke runs.

## How To Run

Run the full local test suite with the repository Python environment:

```bash
/work/envs/depth/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Run one test file:

```bash
/work/envs/depth/bin/python -m unittest tests.test_diffusion_math -v
```

Run one specific test:

```bash
/work/envs/depth/bin/python -m unittest tests.test_model_dry_runs.TestModelDryRuns.test_pixel_diffusion_completes_one_training_batch -v
```

## Test Files

### `tests/test_diffusion_math.py`

This file checks the core math and masking behavior in the diffusion stack.

- `test_beta_schedule_variants_match_their_implementations`
  Confirms that `linear`, `quadratic`, `sigmoid`, and `cosine` schedule selection returns the expected beta tensors.
  Why it matters: schedule mistakes silently change the entire forward/reverse diffusion process and can invalidate training comparisons.

- `test_beta_schedule_rejects_invalid_ranges_and_unknown_variants`
  Verifies that invalid beta ranges and unknown schedule names fail loudly.
  Why it matters: bad config values should fail early instead of producing unstable training.

- `test_build_valid_mask_aligns_channels_and_inverts_missing_mode`
  Checks mask broadcasting and the `observed` vs `missing` inversion logic.
  Why it matters: supervision masks are used in multiple places, and shape or polarity errors would corrupt both training and inference behavior.

- `test_build_ambient_further_valid_mask_keeps_subset_and_respects_minimum_pixels`
  Verifies that ambient further-corruption only removes observed pixels and still keeps a minimum amount of supervision.
  Why it matters: the ambient objective becomes degenerate if the further mask can remove everything.

- `test_build_task_supervision_mask_switches_between_standard_and_ambient_targets`
  Confirms that standard mode supervises `y_valid_mask`, while ambient mode supervises `x_valid_mask ∩ y_valid_mask`.
  Why it matters: this is the central difference between the normal and ambient objectives.

- `test_p_loss_averages_only_over_supervised_ocean_pixels`
  Checks the masked MSE calculation, including the land/ocean gate.
  Why it matters: this verifies that the loss is actually reduced only over intended pixels.

- `test_p_loss_returns_zero_when_mask_selects_nothing`
  Verifies the guarded zero-loss path when the supervision mask is empty.
  Why it matters: batches with no valid supervised support should not produce NaNs or crash.

- `test_p_loss_applies_ambient_further_mask_to_the_noisy_branch`
  Confirms that the optional ambient noisy-branch corruption is applied to the denoiser input.
  Why it matters: this is an objective-defining behavior, not just a logging detail.

### `tests/test_datasets_and_wiring.py`

This file covers fake-data dataset loading, datamodule behavior, and config plumbing.

- `test_surface_temp_4band_dataset_builds_masks_coords_and_dates`
  Loads a minimal fake `.npy` sample and checks `x`, `y`, `x_valid_mask`, `y_valid_mask`, `land_mask`, `coords`, and parsed date.
  Why it matters: it validates the base light-dataset contract that training depends on.

- `test_surface_temp_4band_dataset_can_hide_every_x_pixel`
  Uses maximal corruption and checks that all `x` support can be hidden while `y` remains valid.
  Why it matters: sparse-input training should still behave correctly at extreme corruption levels.

- `test_surface_temp_ostia_dataset_resamples_eo_and_zeroes_land_pixels`
  Verifies OSTIA EO resizing and land masking.
  Why it matters: EO conditioning must match target resolution and should not leak values onto invalid land regions.

- `test_ostia_argo_tiff_dataset_synthetic_mode_rebuilds_sparse_x`
  Builds fake OSTIA/Argo/GLORYS GeoTIFFs and checks synthetic sparse-input generation from GLORYS support.
  Why it matters: this is the fake-data path that allows testing and training smoke runs on machines without the real local dataset.

- `test_datamodule_split_is_deterministic_and_loader_settings_are_applied`
  Checks deterministic train/val splitting and dataloader options such as batch size and shuffle behavior.
  Why it matters: reproducibility and config wiring are both easy to break here.

- `test_config_override_helpers_and_dataset_builder_use_nested_settings`
  Verifies CLI-style nested config overrides and that `build_dataset()` applies nested dataset settings correctly.
  Why it matters: this protects the training entrypoint from silently ignoring user configuration.

### `tests/test_model_dry_runs.py`

This file exercises one-batch model execution and config-based instantiation.

- `test_pixel_diffusion_from_config_wires_nested_settings`
  Instantiates the pixel diffusion model from YAML files and checks scheduler, ambient, sampler, blur, and noise settings.
  Why it matters: config wiring bugs can otherwise look like model-performance issues.

- `test_pixel_training_step_uses_standard_target_and_passes_land_mask`
  Confirms that the standard training path supervises the correct target and forwards `land_mask` into `p_loss`.
  Why it matters: this protects the masked-loss implementation and the standard objective.

- `test_pixel_validation_step_uses_ambient_target_and_intersection_mask`
  Confirms that ambient validation uses the `x` target, the correct supervision mask, and the further-valid mask.
  Why it matters: ambient mode has different semantics and needs separate protection.

- `test_pixel_diffusion_completes_one_training_batch`
  Runs a real Lightning `fit()` call for one batch with synthetic data.
  Why it matters: this is the practical smoke test that catches integration failures across datamodule, model, optimizer, and validation hooks.

- `test_latent_diffusion_completes_one_training_batch`
  Runs one synthetic latent-diffusion training batch.
  Why it matters: latent mode has different mask/channel behavior and must remain runnable end-to-end.

- `test_autoencoder_lightning_completes_one_training_batch`
  Runs one autoencoder training batch.
  Why it matters: latent diffusion depends on this component behaving correctly.

- `test_autoencoder_from_configs_wires_loss_and_scheduler_settings`
  Checks that AE loss weights and scheduler options are loaded from config files.
  Why it matters: AE training should be reproducible from configs just like the diffusion model.

### `tests/test_dataset_ostia_argo_save_to_disk.py`

This file covers the older OSTIA/Argo export path.

- Alignment tests check interpolation onto GLORYS depth levels, duplicate-depth collapse, and the shallow-depth floor behavior.
  Why it matters: depth alignment is fundamental to producing consistent full-stack targets.

- `test_getitem_returns_glorys_aligned_channel_shapes`
  Verifies that exported samples use full GLORYS-aligned band layouts and expose the expected info metadata.
  Why it matters: downstream disk export assumes strict shape alignment.

- Save/export tests cover skipping existing exports, detecting partial exports, overwriting exports, and GeoTIFF metadata writing.
  Why it matters: the disk export path is operational infrastructure, and metadata regressions would break later dataset consumption.

## What These Tests Protect

At a high level, the suite is intended to catch four classes of regressions:

- Diffusion math regressions:
  bad schedules, wrong target selection, incorrect ambient corruption, or masked-loss mistakes.

- Dataset contract regressions:
  wrong shapes, invalid-mask semantics, broken synthetic sparse-input generation, or bad date/coord parsing.

- Config wiring regressions:
  settings defined in YAML but not actually applied at runtime.

- End-to-end integration regressions:
  models that instantiate but fail to complete even one real batch.

## Notes

- The tests use `unittest`, not `pytest`, because the pinned repository environment does not include `pytest`.
- The dry-run tests intentionally use fake data and tiny models so they can run without the production dataset.
- The Lightning smoke tests may print warnings about disabled loggers or low dataloader worker counts. Those warnings are expected in these minimal test runs.
