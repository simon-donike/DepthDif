# Inference  
There are two practical inference workflows in this repository:  
- run the standalone script `inference.py`  
- call `PixelDiffusionConditional.predict_step(...)` directly  
  
DepthDif supports pixel-space configs (`configs/px_space/*`) and latent-workflow configs (`configs/lat_space/*`).  
For latent workflow setup and command flow, see [Autoencoder + Latent Diffusion](autoencoder.md).  
  
## Workflow 1: Use `inference.py`  
`inference.py` is a configurable script for quick prediction sanity checks.  
  
### What it supports  
- load config files and instantiate model/datamodule  
- load checkpoint (explicit override or `model.load_checkpoint` / `model.resume_checkpoint`)  
- run from:  
  - dataloader sample (`MODE="dataloader"`)  
  - synthetic random batch (`MODE="random"`)  
- optional intermediate sample capture  
  
### Important script settings  
At the top of `inference.py`, set:  
- `MODEL_CONFIG_PATH`  
- `DATA_CONFIG_PATH`  
- `TRAIN_CONFIG_PATH`  
- `CHECKPOINT_PATH` (or keep `None` to use `model.load_checkpoint` then `model.resume_checkpoint`)  
- `MODE`, `LOADER_SPLIT`, `DEVICE`, `INCLUDE_INTERMEDIATES`  
  
### Note on default paths  
The script constants should be set explicitly. In this repository, the actively used configs are:  
- OSTIA + Argo disk setup: `configs/px_space/model_config.yaml`, `configs/px_space/data_ostia_argo_disk.yaml`, `configs/px_space/training_config.yaml`  
  
## Workflow 2: Direct `predict_step`  
The model inference entry point is:  
- `PixelDiffusionConditional.predict_step(batch, batch_idx=0)`  
  
Minimum required batch key:  
- `x`  
  
Common optional keys:  
- `eo`  
- `valid_mask`  
- `land_mask`  
- `coords`  
- `date`  
- `sampler`  
- `clamp_known_pixels`  
- `return_intermediates`  
- `intermediate_step_indices`  
  
### Returned outputs  
`predict_step` returns a dictionary containing:  
- `y_hat`: standardized prediction  
- `y_hat_denorm`: temperature-denormalized prediction  
- `denoise_samples`: reverse samples (if requested)  
- `x0_denoise_samples`: per-step x0 predictions (if requested)  
- `sampler`: sampler used for prediction  
  
## Example (`ostia_argo_disk` config)  
```python  
import torch  
  
from data.datamodule import DepthTileDataModule  
from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset  
from models.difFF import PixelDiffusionConditional  
  
model_config = "configs/px_space/model_config.yaml"  
data_config = "configs/px_space/data_ostia_argo_disk.yaml"  
train_config = "configs/px_space/training_config.yaml"  
ckpt_path = "logs/<run>/best-epochXXX.ckpt"  
  
train_dataset = OstiaArgoTiffDataset.from_config(data_config, split="train")  
val_dataset = OstiaArgoTiffDataset.from_config(data_config, split="val")  
datamodule = DepthTileDataModule(dataset=train_dataset, val_dataset=val_dataset)  
  
model = PixelDiffusionConditional.from_config(  
    model_config_path=model_config,  
    data_config_path=data_config,  
    training_config_path=train_config,  
    datamodule=datamodule,  
)  
  
state = torch.load(ckpt_path, map_location="cpu")  
state_dict = state["state_dict"] if "state_dict" in state else state  
model.load_state_dict(state_dict, strict=False)  
model.eval()  
  
batch = next(iter(datamodule.val_dataloader()))  
with torch.no_grad():  
    pred = model.predict_step(batch, batch_idx=0)  
  
y_hat = pred["y_hat"]  
y_hat_denorm = pred["y_hat_denorm"]  
```  
  
## Sampler Choice  
Validation/inference sampler can be switched via training config:  
- `training.validation_sampling.sampler: "ddpm"` or `"ddim"`  
- DDIM controls:  
  - `ddim_num_timesteps`  
  - `ddim_eta`  
  
The same sampler can also be injected per batch through `batch["sampler"]` in direct prediction calls.  
  