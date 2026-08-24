# Development

## Environment

Use `/work/envs/depth/bin/python` for Python commands in this repository. Install
the project dependencies from the root metadata:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Training CLI and Lightning orchestration. |
| `src/depth_recon/configs/` | Pixel and latent YAML configs plus scenario resolvers. |
| `src/depth_recon/data/` | Active GeoTIFF/Zarr dataset and data-production tools. |
| `src/depth_recon/models/` | Diffusion, EMA, autoencoder, and baseline models. |
| `src/depth_recon/inference/` | Public API and export/analysis workflows. |
| `docs/` | MkDocs pages, static viewers, JavaScript, styles, and assets. |
| `tests/` | Complete unittest-based regression suite. |

`dataset_argo_netcdf_gridded.py` is a legacy dataset. New documentation and
features should use `ArgoGeoTIFFGriddedPatchDataset` and do not need compatibility
with the legacy loader.

## Working conventions

- Keep edits focused and follow nearby naming, docstring, and error-handling style.
- Keep validation dataloaders shuffled; this is intentional repository behavior.
- Add an all-options invocation comment to new task-specific CLI scripts.
- Explain non-obvious logic inline and document new functions.
- Do not treat generated synthetic targets as observations or scientific truth.
- Update docs whenever a config key, artifact schema, CLI, or public interface changes.

## Checks

Format Python changes:

```bash
/work/envs/depth/bin/python -m black .
```

Build the complete API-enabled site:

```bash
ENABLE_MKDOCSTRINGS=true /work/envs/depth/bin/python -m mkdocs build --strict
```

Run the whole test suite for substantive changes:

```bash
tests/run_tests.sh
```

See [Tests](tests.md) for the coverage map and [CLI reference](cli.md) for
maintained commands.
