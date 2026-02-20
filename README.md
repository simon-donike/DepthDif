<p align="center">
  <img src="docs/assets/banner_depthdif.png" width="70%" style="border-radius: 12px;" />
</p>

<p align="center">
  <a href="docs/">
    <img src="https://img.shields.io/badge/Open-Documentation-0b2e4f?style=for-the-badge" alt="Open Documentation" />
  </a>
  <a href="docs/experiments.md">
    <img src="https://img.shields.io/badge/Open-Experiments-0f3f68?style=for-the-badge" alt="Open Experiments" />
  </a>
</p>

# DepthDif

DepthDif is a conditional diffusion project for densifying sparse ocean temperature observations.

## Installation

This project uses Python 3.12.3.

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

For docs tooling:

```bash
/work/envs/depth/bin/python -m pip install -r docs/requirements.txt
```

## Model Overview

- Model: `PixelDiffusionConditional` (conditional pixel-space diffusion with ConvNeXt U-Net denoiser).
- Main task modes:
  - `temp_v1`: single-band corrupted input `x` to clean target `y`.
  - `eo_4band`: EO-conditioned multiband reconstruction (`[eo, x, valid_mask] -> y`).
- Default configs live in `configs/` and are selected via CLI.

## Training

Default single-band training:

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_config.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

EO + multiband training:

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_config_eo_4band.yaml \
  --train-config configs/training_config_eo_4band.yaml \
  --model-config configs/model_config_eo_4band.yaml
```

Notes:
- `--train-config` and `--training-config` are equivalent.
- Training outputs are written under `logs/<timestamp>/` with `best.ckpt` and `last.ckpt`.

## Inference

Use `inference.py`:

1. Set config/checkpoint constants at the top of `inference.py` (`MODEL_CONFIG_PATH`, `DATA_CONFIG_PATH`, `TRAIN_CONFIG_PATH`, `CHECKPOINT_PATH`).
2. Choose `MODE` (`"dataloader"` or `"random"`).
3. Run:

```bash
/work/envs/depth/bin/python inference.py
```

## Documentation

- Full documentation: `docs/` (or build/serve with MkDocs).
- Experiments page: `docs/experiments.md`.
