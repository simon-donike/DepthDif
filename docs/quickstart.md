# Quick Start
Use this page for the shortest path from setup to first train/inference run.

## Environment & Dependencies
- Python: **3.12.3**  
- Install runtime dependencies:   

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

- Optional docs dependencies:  

```bash
/work/envs/depth/bin/python -m pip install -r docs/requirements.txt
```

## Quick Training
EO-conditioned multiband training (`eo_4band` or `ostia`, depending on `dataset.core.dataset_variant`):  

```bash
/work/envs/depth/bin/python train.py \
  --data-config configs/data_ostia.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

Legacy same-source EO setup: use `--data-config configs/data.yaml`.

## Quick Inference
Set config/checkpoint constants at the top of `inference.py`, then run:  

```bash
/work/envs/depth/bin/python inference.py
```

For EO multiband runs, use:
- `MODEL_CONFIG_PATH = "configs/model_config.yaml"`  
- `DATA_CONFIG_PATH = "configs/data_ostia.yaml"`  
- `TRAIN_CONFIG_PATH = "configs/training_config.yaml"`  
