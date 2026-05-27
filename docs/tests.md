# Tests

This page documents the current repository test coverage around diffusion math,
GeoTIFF dataset behavior, config wiring, inference exports, and one-batch model
smoke runs.

## How To Run

Run the full local suite through the repository wrapper:

```bash
bash tests/run_tests.sh
```

The wrapper uses `unittest` discovery in the configured environment.
It also prepends `src` to `PYTHONPATH` so tests exercise the src-layout package
without requiring a prior editable install.

## Main Test Areas

### Diffusion Math

`tests/test_diffusion_math.py` checks schedule construction, mask polarity,
ambient further-corruption, target selection, masked loss behavior, and the
guarded zero-loss path.

### Model Dry Runs

`tests/test_model_dry_runs.py` instantiates small in-memory datasets and tiny
models to verify one training/validation batch for pixel diffusion, latent
diffusion, and the autoencoder.

### Inference Exports

The global export and Cesium packaging tests verify that inference code uses
public dataset metadata (`rows`, `depth_axis_m`) and produces consistent raster,
GeoJSON, summary, and globe asset outputs.

### Config And CLI Wiring

Training, inference, validation-summary, and override tests protect the active pixel super-config path, scenario derivation, effective config materialization, and GeoTIFF builder selection.

## Notes

- Tests use `unittest`, not `pytest`.
- Some smoke tests intentionally use tiny in-memory fixtures so model wiring can
  be checked quickly without local source archives.
