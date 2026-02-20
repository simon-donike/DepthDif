# Training
Train with `train.py`. You can now choose which config files to use from CLI.

## CLI config selection
```bash
python3 train.py --data-config configs/data_config_eo_4band.yaml --train-config configs/training_config_eo_4band.yaml --model-config configs/model_config_eo_4band.yaml
```

Notes:
- `--train-config` and `--training-config` are equivalent.
- If omitted, all three arguments default to:
  - `configs/data_config.yaml`
  - `configs/training_config.yaml`
  - `configs/model_config.yaml`

## EO + 3-band conditional training
Use the EO/4-band config set to train with:
- condition = `eo` (1 band)  + corrupted `x` (3 bands) + `valid_mask` (1 band) = 5 condition channels
- target = `y` (3 clean temperature bands)
- `dataset.eo_dropout_prob` enables EO dropout (randomly zeroes EO for a subset of samples in both train and val)
  - reasoning: this reduces EO shortcut learning so the model does not over-rely on EO and still reconstructs from the actual corrupted `x` (+ mask context).

```bash
python3 train.py \
  --data-config configs/data_config_eo_4band.yaml \
  --train-config configs/training_config_eo_4band.yaml \
  --model-config configs/model_config_eo_4band.yaml
```

## Straight corrupted -> uncorrupted training
Use the default single-band setup:

```bash
python3 train.py \
  --data-config configs/data_config.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

## What happens during training
- A timestamped run folder is created under `logs/`.
- The exact config files used for the run are copied into that folder, and checkpointing keeps `best.ckpt` (by `trainer.ckpt_monitor`) plus `last.ckpt`.
- The only supported model type is `cond_px_dif` (`PixelDiffusionConditional`).
- Training resumes automatically when `model.resume_checkpoint` is set to a valid `.ckpt` path in `configs/model_config.yaml`.
