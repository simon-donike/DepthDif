# Tests

DepthDif uses the standard-library `unittest` runner through a repository wrapper.
Run the complete suite from the repository root:

```bash
tests/run_tests.sh
```

The wrapper uses `/work/envs/depth/bin/python`. Targeted test execution is useful
while diagnosing a failure, but substantive changes are accepted against the full
suite.

## Coverage map

| Area | Representative coverage |
| --- | --- |
| Data creation | EN4 enrichment, potential-temperature selection, GeoTIFF export, compact Zarr, packaged downloads, and Hugging Face layout. |
| Dataset contract | Quantization, masks, fields, scenarios, synthetic targets, hard-region sampling, splits, and datamodule wiring. |
| Model | Diffusion math, schedules, DDPM/DDIM samplers, model dry runs, losses, coordinate/date conditioning, EMA, and baselines. |
| Configuration | Pixel scenario resolution, preset invariants, checkpoint compatibility, and CLI override parsing. |
| Public API | Asset resolution, source downloads, weekly patch selection, mosaics, metadata, and package exports. |
| Evaluation | Global inference, EN4 holdouts, hard regions, validation summaries, paper-week bundles, and metrics. |
| Analysis products | Cesium assets, comparison globe, spatial/temporal dashboards, temporal globe, and wavenumber spectra. |

Tests use temporary directories and small synthetic fixtures where possible. They
do not establish scientific skill or validate external hosted data; they verify
the repository contracts and deterministic transformations.

## Documentation checks

Build with API rendering enabled so malformed docstrings are not hidden:

```bash
ENABLE_MKDOCSTRINGS=true /work/envs/depth/bin/python -m mkdocs build --strict
```

For notebook changes, validate the JSON and compile each Python code cell without
running downloads or GPU inference.
