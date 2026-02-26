<p align="center">
  <img src="docs/assets/banner_depthdif.png" width="65%" style="border-radius: 12px;" />
</p>

<p align="center">
  <a href="https://depthdif.donike.net/">
    <img src="https://img.shields.io/badge/Visit-Documentation-0b2e4f?style=for-the-badge" alt="Open Documentation" />
  </a>
  <a href="https://depthdif.donike.net/experiments/">
    <img src="https://img.shields.io/badge/Open-Experiments-0f3f68?style=for-the-badge" alt="Check Experiments" />
  </a>
</p>

# DepthDif

DepthDif is a conditional diffusion project for densifying sparse ocean temperature observations. Visit the [Documentation](https://depthdif.donike.net/) for more info on the models, datasets, and auxiliary data - or follow along with the [Experiments](https://depthdif.donike.net/experiments/).



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
  - `eo_4band`: EO-conditioned multiband reconstruction (`[eo, x, valid_mask] -> y`).  
- Default configs live in `configs/` and are selected via CLI.  

DepthDif is a conditional diffusion model: it reconstructs dense depth fields from corrupted submarine observations, conditioned on EO (surface) data plus sparse corrupted subsurface input. Synthetic sparse inputs are generated with continuous curved trajectory masks to mimic submarine movement; in the current dataset version, each track keeps one measurement every few pixels (random 2-8 pixel stride) until the configured corruption percentage is reached. It can inject coordinate/date context via FiLM conditioning and reconstruct the full target image.

![depthdif_schema](docs/assets/depthdif_schema.png)

## Training

EO + multiband training:

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_ostia.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

Legacy same-source EO config: `configs/data.yaml`  

Notes:
- `--train-config` and `--training-config` are equivalent.  
- Training outputs are written under `logs/<timestamp>/` with `best.ckpt` and `last.ckpt`.  
- `model.resume_checkpoint` resumes full Lightning state; `model.load_checkpoint` warm-starts by loading only model weights.  

## Inference

Use `inference.py`:

1. Set config/checkpoint constants at the top of `inference.py` (`MODEL_CONFIG_PATH`, `DATA_CONFIG_PATH`, `TRAIN_CONFIG_PATH`, `CHECKPOINT_PATH`).  
   For the active EO setup in this repository, use:
   `configs/model_config.yaml`, `configs/data_ostia.yaml`, `configs/training_config.yaml`  
   (legacy same-source EO uses `configs/data.yaml`).
2. Choose `MODE` (`"dataloader"` or `"random"`).  
3. Run:  

```bash
/work/envs/depth/bin/python inference.py
```

## Documentation

- Full documentation: `docs/` (or build/serve with MkDocs).  
- Experiments page: `docs/experiments.md`.  
