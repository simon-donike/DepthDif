# Public Inference Package

The `depth-recon` package provides a stable one-week inference API and three
console commands. Its default `depthdif_v1.ckpt` is the legacy single-OSTIA
model. Current repository checkpoints use SST, SSS, and ADT and require their
matching scenario-resolved configs and data pipeline.

## Install

```bash
python -m pip install depth-recon
```

## Run one ISO week

```python
from depth_recon import run_week_inference

run_dir = run_week_inference(
    year=2015,
    iso_week=25,
    rectangle=(-20.0, 30.0, 10.0, 50.0),
    device="cuda",
)
```

The selected date is the ISO-week Wednesday. The no-GLORYS path uses a
0.1° land-mask grid with 128×128 non-overlapping public patches and a default
minimum ocean fraction of 0.05. A rectangle is `(west, east, south, north)`.

## Assets and cache

By default, asset resolution reads `simon-donike/DepthDif` at revision `main`:

- `model_config.yaml`
- `data_config.yaml`
- `training_config.yaml`
- `depthdif_v1.ckpt`
- `world_land_mask_glorys_0p1.tif`

The default cache is `~/.cache/depthdif`. Existing files are reused unless
`force_download=True`. Resolve them without starting inference:

```python
from depth_recon import resolve_public_inference_assets

bundle = resolve_public_inference_assets()
print(bundle.assets.checkpoint)
print(bundle.land_mask_path)
```

## Source inputs

When `argo_dir` is omitted, `run_week_inference` downloads each EN.4.2.2 monthly
profile archive touched by the ISO week. OSTIA is also downloaded by default.
Copernicus credentials may come from the environment or the API arguments
`copernicus_username` and `copernicus_token`; `copernicus_password` remains an
alias accepted by the toolbox integration.

Disable automatic OSTIA download for ARGO-only conditioning:

```python
run_week_inference(
    year=2015,
    iso_week=25,
    auto_download_ostia=False,
)
```

Supplying `glorys_dir` selects the repository-backed export branch, where GLORYS
can be included as ground truth. GLORYS is not required for the standard public
path.

## Outputs

The call returns the run directory, normally
`inference/outputs/depthdif_argo_<YYYYMMDD>/`. Prediction GeoTIFFs are exported
for Surface, 10, 50, 100, 250, 500, 1000, 2000, 2500, and 5000 m, with nearest
native source-depth metadata. The run also records configuration, sampling, grid,
and source provenance.

## Sampling and uncertainty

The bundled public training metadata defaults to DDPM. Passing a DDIM step count
without a sampler selects DDIM. Uncertainty has independent sampler overrides:

```python
run_week_inference(
    year=2015,
    iso_week=25,
    sampler="ddim",
    ddim_num_timesteps=100,
    export_uncertainty=True,
    uncertainty_num_samples=20,
    uncertainty_sampler="ddim",
    uncertainty_ddim_num_timesteps=50,
)
```

The public uncertainty product is one depth-collapsed population-standard-
deviation raster. Set `uncertainty_only=True` to omit the ordinary prediction.

## Console commands

```bash
depth-recon-download-argo \
  --year 2015 --iso-week 25 --output-dir ./en4_profiles

depth-recon-download-ostia \
  --year 2015 --iso-week 25 --output-dir ./ostia

depth-recon-infer-week \
  --year 2015 --iso-week 25 \
  --rectangle -20 30 10 50 \
  --device cuda
```

Use `--help` for the complete option set, including sampler overrides,
uncertainty-only mode, local assets, credentials, cache control, and strict
checkpoint loading.
