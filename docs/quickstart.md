# Quick Start
Use this page for the shortest path from setup to first train/inference run.

## Environment & Dependencies
- Python: **3.12.3**
- Install runtime dependencies:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

The root `requirements.txt` installs this repository from the curated
dependencies in `pyproject.toml`.

## Quick Training
Pixel-space GeoTIFF training with the default super-config:

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature
```

Scenario choices:

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
```

The default training config is `src/depth_recon/configs/px_space/training_super_config.yaml`. `--scenario` derives the field list, salinity data flag, generated channels, and condition channels. Use `--set data.*`, `--set model.*`, or `--set training.*` for one-off overrides after scenario resolution.

Latent diffusion uses the separate `src/depth_recon/configs/lat_space/` configs. See [Autoencoder + Latent Diffusion](autoencoder.md) for architecture, goals, limitations, and workflow details.

## Quick Inference
For public inference from PyPI, install the package and run one ISO week:

```bash
python -m pip install depth-recon
```

```python
from depth_recon import run_week_inference

run_dir = run_week_inference(
    year=2015,
    iso_week=25,
    rectangle=(-20.0, 30.0, 10.0, 50.0),
    device="cuda",
)
print(run_dir)
```

The package downloads the public model artifacts from Hugging Face, downloads
EN4/ARGO and OSTIA source files when needed, and writes prediction GeoTIFFs under
`inference/outputs/depthdif_argo_<YYYYMMDD>/`. See
[Public Inference Package](public-inference-package.md) for the full API and CLI.

For repository-local smoke checks, set config/checkpoint constants at the top of
`src/depth_recon/inference/run_single.py`, then run:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.run_single
```

For EO multiband runs, use:
- `CONFIG_PATH = "src/depth_recon/configs/px_space/inference_super_config.yaml"`
- `SCENARIO = "temperature"`, `"salinity"`, or `"joint"`

The inference super-config has top-level `data`, `model`, `training`, and `inference` sections. Scenario resolution derives the same model/data channel contract used by training.

To export one stitched world raster and prepare the hosted Cesium assets afterward, use:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.export_global --year 2010 --iso-week 1
/work/envs/depth/bin/python -m depth_recon.inference.export_cesium_globe_assets \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```
