# Inference
There are two practical inference workflows in this repository:  
- run the standalone script `inference.py`  
- call `PixelDiffusionConditional.predict_step(...)` directly  

## Workflow 1: Use `inference.py`
`inference.py` is a configurable script for quick prediction sanity checks.

### What it supports
- load config files and instantiate model/datamodule  
- load checkpoint (explicit override or `model.resume_checkpoint`)  
- run from:  
  - dataloader sample (`MODE="dataloader"`)  
  - synthetic random batch (`MODE="random"`)  
- optional intermediate sample capture  

### Important script settings
At the top of `inference.py`, set:  
- `MODEL_CONFIG_PATH`  
- `DATA_CONFIG_PATH`  
- `TRAIN_CONFIG_PATH`  
- `CHECKPOINT_PATH` (or keep `None` to use config resume path)  
- `MODE`, `LOADER_SPLIT`, `DEVICE`, `INCLUDE_INTERMEDIATES`  

### Note on default paths
The script constants should be set explicitly. In this repository, the actively used configs are:
- EO/OSTIA setup: `configs/model_config.yaml`, `configs/data_ostia.yaml`, `configs/training_config.yaml`  
- legacy same-source EO setup: swap to `configs/data.yaml`  

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

## Example (`eo_4band` config)
```python
import torch

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from models.difFF import PixelDiffusionConditional

model_config = "configs/model_config.yaml"
data_config = "configs/data_ostia.yaml"
train_config = "configs/training_config.yaml"
ckpt_path = "logs/<run>/best-epochXXX.ckpt"

dataset = SurfaceTempPatch4BandsLightDataset.from_config(data_config, split="all")
datamodule = DepthTileDataModule(dataset=dataset)
datamodule.setup("fit")

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

For `dataset_variant="ostia"`, use `SurfaceTempPatchOstiaLightDataset` instead and point `data_config.dataset.source.light_index_csv` to the OSTIA overlap CSV with `ostia_npy_path`.

## Sampler Choice
Validation/inference sampler can be switched via training config:
- `training.validation_sampling.sampler: "ddpm"` or `"ddim"`  
- DDIM controls:  
  - `ddim_num_timesteps`  
  - `ddim_eta`  

The same sampler can also be injected per batch through `batch["sampler"]` in direct prediction calls.
