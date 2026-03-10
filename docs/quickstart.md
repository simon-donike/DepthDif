# Quick Start  
Use this page for the shortest path from setup to first train/inference run.  
  
## Environment & Dependencies  
- Python: **3.12.3**  
- Install runtime dependencies:  
  
```bash  
/work/envs/depth/bin/python -m pip install -r requirements.txt  
```  
  
## Quick Training  
EO-conditioned multiband training (`eo_4band` or `ostia`, depending on `dataset.core.dataset_variant`):  
  
```bash  
python train.py \  
  --data-config configs/px_space/data_ostia.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config.yaml  
```  
  
Legacy same-source EO setup: use `--data-config configs/px_space/data_config.yaml`.  
Latent diffusion workflow:  
  
```bash  
/work/envs/depth/bin/python train_autoencoder.py \  
  --ae-config configs/lat_space/ae_config.yaml \  
  --data-config configs/lat_space/data_config.yaml \  
  --train-config configs/lat_space/training_config.yaml  
  
/work/envs/depth/bin/python train.py \  
  --data-config configs/lat_space/data_config.yaml \  
  --train-config configs/lat_space/training_config.yaml \  
  --model-config configs/lat_space/model_config.yaml  
```  
  
Equivalent script wrappers:  
- `./scripts/train_autoencoder.sh`  
- `./scripts/train_latent_diffusion.sh`  
  
See [Autoencoder + Latent Diffusion](autoencoder.md) for architecture, goals, limitations, and workflow details.  
  
## Quick Inference  
Set config/checkpoint constants at the top of `inference.py`, then run:  
  
```bash  
python inference.py  
```  
  
For EO multiband runs, use:  
- `MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"`  
- `DATA_CONFIG_PATH = "configs/px_space/data_ostia.yaml"`  
- `TRAIN_CONFIG_PATH = "configs/px_space/training_config.yaml"`  
Remember to wire through your dataloaders in the config. Alternatively, pass the inputs individually through PL's `predict_step`.  
