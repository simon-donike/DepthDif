# Quick Start  
Use this page for the shortest path from setup to first train/inference run.  
  
## Environment & Dependencies  
- Python: **3.12.3**  
- Install runtime dependencies:  
  
```bash  
/work/envs/depth/bin/python -m pip install -r requirements.txt  
```  
  
## Quick Training  
OSTIA + Argo disk training (`dataset.core.dataset_variant="ostia_argo_disk"`):  
  
```bash  
python train.py \  
  --data-config configs/px_space/data_ostia_argo_disk.yaml \  
  --train-config configs/px_space/training_config.yaml \  
  --model-config configs/px_space/model_config.yaml  
```  
  
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
Set config/checkpoint constants at the top of `inference/run_single.py`, then run:  

```bash  
/work/envs/depth/bin/python inference/run_single.py  
```  

For EO multiband runs, use:  
- `MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"`  
- `DATA_CONFIG_PATH = "configs/px_space/data_ostia_argo_disk.yaml"`  
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
  
