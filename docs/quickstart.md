# Quick Start

## Install

Use Python 3.12 or newer. In this repository, use the configured environment:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

For only the published public inference API:

```bash
python -m pip install depth-recon
```

## Train a maintained model

Choose exactly one scenario; the resolver derives the data fields and model
channel counts.

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
```

The default super-config is
`src/depth_recon/configs/px_space/training_super_config.yaml`. Override a value
after scenario resolution with `--set`:

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set training.wandb.run_name=temperature_debug \
  --set training.trainer.max_epochs=2
```

The default config expects the packaged GeoTIFF/Zarr dataset at its configured
root. See [dataset downloads](data-download.md) and [training](training.md) before
changing data paths or target modes.

## Run public inference

```python
from depth_recon import run_week_inference

run_dir = run_week_inference(
    year=2015,
    iso_week=25,
    rectangle=(-20.0, 30.0, 10.0, 50.0),
    device="cuda",
)
```

The public call uses the legacy single-OSTIA `depthdif_v1.ckpt`, not the current
three-surface training architecture. It resolves cached Hugging Face assets and
downloads weekly EN4/ARGO and OSTIA inputs when required.

To also export the default 20-member collapsed uncertainty map:

```python
run_week_inference(
    year=2015,
    iso_week=25,
    export_uncertainty=True,
)
```

See [public inference](public-inference-package.md) for credentials, caching,
sampler overrides, output depths, and ARGO-only mode.

## Run a repository checkpoint

For a smoke test, configure the constants in
`src/depth_recon/inference/run_single.py`, then run:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.run_single
```

For stitched exports, use the commands in [inference](inference.md) and inspect
all available options with `--help`.
