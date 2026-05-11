# Quick Start
Use this page for the shortest path from setup to first train/inference run.

## Environment & Dependencies
- Python: **3.12.3**
- Install runtime dependencies:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

## Quick Training
OSTIA + Argo NetCDF training (`dataset.core.dataset_variant="argo_netcdf_gridded"`):

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/px_space/training_config.yaml \
  --model-config configs/px_space/model_config.yaml
```

Latent diffusion workflow:

```bash
/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config configs/lat_space/ae_config.yaml \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/lat_space/training_config.yaml

/work/envs/depth/bin/python train.py \
  --data-config configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config configs/lat_space/training_config.yaml \
  --model-config configs/lat_space/model_config.yaml
```

Equivalent script wrappers:
- `./scripts/train_autoencoder.sh`
- `./scripts/train_latent_diffusion.sh`

See [Autoencoder + Latent Diffusion](autoencoder.md) for architecture, goals, limitations, and workflow details.

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
`inference/run_single.py`, then run:

```bash
/work/envs/depth/bin/python inference/run_single.py
```

For EO multiband runs, use:
- `MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"`
- `DATA_CONFIG_PATH = "configs/px_space/data_ostia_argo_netcdf.yaml"`
- `TRAIN_CONFIG_PATH = "configs/px_space/training_config.yaml"`
Remember to wire through your dataloaders in the config. Alternatively, pass the inputs individually through PL's `predict_step`.

To export one stitched world raster and prepare the hosted Cesium assets afterward, use:

```bash
/work/envs/depth/bin/python inference/export_global.py --year 2010 --iso-week 1
/work/envs/depth/bin/python inference/export_cesium_globe_assets.py \
  --run-dir inference/outputs/global_top_band_<YYYYMMDD> \
  --public-base-url https://<bucket-or-site>/inference_production/globe/ \
  --rclone-remote r2:<bucket>/inference_production/globe \
  --rclone-sync-scope globe
```
