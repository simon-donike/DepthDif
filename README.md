# Densifying Sparse Ocean Depth Observations
This implementation is a first test, checking the feasability of densifying sparse ocean measurements.

## Environment & Dependencies

- The project uses **Python 3.12.3**.
- All Python dependencies are listed in a single `requirements.txt` file located at the **repository root**.
- Install dependencies with:
```bash
pip install -r requirements.txt
```


## Data
Currently, monthly tiles from 2000 - 2025 from the [Global Ocean Physics Reanalysis dataset](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1M-m_202311&path=GLOBAL_MULTIYEAR_PHY_001_030%2Fcmems_mod_glo_phy_my_0.083deg_P1M-m_202311%2F2024%2F) have been downloaded and are manually masked to simulate real sparse observations. Excluding patches with >20% NoData values, ~106k samples are avaialble (128x128, 1/12 °). Download the data by installing the `copernicusmarine` package, then use the CLI like so `copernicusmarine get -i cmems_mod_glo_phy_my_0.083deg_P1M-m  --filter "*2021/*"`  
The of the obstructions and the coverage percentage are selectable in the `data_config.yaml`.

Dataset example for 50% occlusion:  
![img](assets/dataset_50percMask.png)  

Current Status:
- 1-band only for experimentation
- 128x128 hardcoded

### Dataset tweaks
- Synthetic occlusion pipeline to create sparse observations with configurable `mask_fraction`.
- Patch-based masking with min/max patch sizes (`mask_patch_min`, `mask_patch_max`) instead of single-pixel drops.
- Validity/land masks derived from nodata or fill values; invalid pixels are tracked separately from corruption.
- Optional filtering of tiles by `max_nodata_fraction` to avoid overly invalid patches.
- Corrupted input + mask channel return modes for conditional modeling (`x_return_mode`).
- Z-score temperature normalization and optional geometric augmentation (rotations/flips) applied consistently to data and masks.
- Dataset index build with nodata-fraction metadata for fast filtering.
- Optional patch-center coordinate return (`return_coords`) using index columns (`lat0/lat1/lon0/lon1`) with dateline-safe longitude center computation.

## Model
As a first prototype, a conditional pixel-space Diffuser is modeled after [DiffusionFF](https://github.com/mikonvergence/DiffusionFastForward).  

The model is trained on 1-channel temp + valid pixel mask. Loss can be pulled including or excluding the mask.

### Major Model Settings + Where to Configure
These are the core model behaviors in this repo and where they are wired in config.

- **Input channels with mask (conditioning channels)**  
  The conditional diffusion model receives the corrupted data plus a mask channel.  
  Configure in `configs/model_config.yaml`:
  - `model.condition_channels`: total conditioning channels passed to the denoiser (data + mask).
  - `model.condition_mask_channels`: number of those channels that are mask semantics (excluded from normalization).  
  This is built from `x` and `valid_mask` in the dataset.

- **Masked pixel loss computation**  
  Loss can be restricted to valid (ocean) pixels to avoid land/no‑data bias.  
  Configure in `configs/model_config.yaml`:
  - `model.mask_loss_with_valid_pixels: true`  
  When enabled, the conditional loss is multiplied by the `valid_mask` and normalized by its sum.

- **Coordinate encoding + FiLM injection**  
  Patch-center coordinates are encoded and injected via FiLM scale/shift in every ConvNeXt block.  
  Configure in:
  - `configs/data_config.yaml`: `dataset.return_coords: true` (adds coords to each batch)
  - `configs/model_config.yaml`: `model.coord_conditioning.enabled: true`
  - `model.coord_conditioning.encoding`: `unit_sphere`, `sincos`, or `raw`
  - `model.coord_conditioning.embed_dim`: embedding width (defaults to `unet.dim` if null)

- **Anchoring known pixels at inference (inpainting clamp)**  
  During sampling, known pixels are overwritten at every diffusion step for stability. See image below: top-left shows conditioning with mask, the other images show 15 intermediate denoising steps with noise only injected into thos epixels where no observations are present to anchor values.  
  Configure in `configs/model_config.yaml`:
  - `model.clamp_known_pixels: true`  
  This uses the conditioning input + mask to keep known values fixed while denoising.

  ![img](assets/clamped_pixels.png)  

### Minor Model Settings + Where to Configure
These are the model training behaviors in this repo and where they are wired in config.

- **Learning-rate scheduler (ReduceLROnPlateau)**  
  Configure in `configs/training_config.yaml`:
  - `scheduler.reduce_on_plateau.enabled`
  - `scheduler.reduce_on_plateau.monitor`, `mode`, `factor`, `patience`, `threshold`, `cooldown`
  - `trainer.lr_logging_interval` (learning-rate monitor cadence)

- **Checkpointing + resume**  
  Configure in:
  - `configs/model_config.yaml`: `model.resume_checkpoint` (false/null or a `.ckpt` path)
  - `configs/training_config.yaml`: `trainer.ckpt_monitor` (best-checkpoint metric)

- **W&B logging (metrics/images/watch)**  
  Configure in `configs/training_config.yaml`:
  - `wandb.project`, `wandb.entity`, `wandb.run_name`, `wandb.log_model`
  - `wandb.verbose`, `wandb.log_images_every_n_steps`, `wandb.log_stats_every_n_steps`
  - `wandb.watch_gradients`, `wandb.watch_parameters`, `wandb.watch_log_freq`, `wandb.watch_log_graph`

- **Validation-time sampling + diagnostics**  
  Configure in `configs/training_config.yaml`:
  - `training.validation_sampling.sampler` (`ddpm` or `ddim`)
  - `training.validation_sampling.ddim_num_timesteps`, `ddim_eta`
  Notes: PSNR/SSIM are computed when `skimage` is available; one cached val example per epoch is used for full reconstruction logging.

## Training
Train with `train.py`. You can now choose which config files to use from CLI.

### CLI config selection
```bash
python3 train.py \
  --data-config configs/data_config.yaml \
  --train-config configs/training_config.yaml \
  --model-config configs/model_config.yaml
```

Notes:
- `--train-config` and `--training-config` are equivalent.
- `--model-config` also accepts `--mdoel-config` (typo alias).
- If omitted, all three arguments default to:
  - `configs/data_config.yaml`
  - `configs/training_config.yaml`
  - `configs/model_config.yaml`

### What happens during training
- A timestamped run folder is created under `logs/`.
- The exact config files used for the run are copied into that folder.
- Model type is selected from `model.model_type`:
  - `cond_px_dif` -> `PixelDiffusionConditional`
  - `px_dif` -> `PixelDiffusion`
- Training resumes automatically when `model.resume_checkpoint` is set to a valid `.ckpt` path in `configs/model_config.yaml`.

## Inference
Inference can be run by loading a checkpoint into the same model class and calling the model's `predict_step`.

### 1) Build model from config + load checkpoint
```python
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer

from data.datamodule import DepthTileDataModule
from data.dataset_temp_v1 import SurfaceTempPatchLightDataset
from models.difFF import PixelDiffusion, PixelDiffusionConditional


def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


model_config_path = "configs/model_config.yaml"
data_config_path = "configs/data_config.yaml"
training_config_path = "configs/training_config.yaml"
ckpt_path = "logs/<run>/best.ckpt"

model_cfg = load_yaml(model_config_path)
model_type = model_cfg.get("model", {}).get("model_type", "cond_px_dif")

dataset = SurfaceTempPatchLightDataset.from_config(data_config_path, split="all")
datamodule = DepthTileDataModule(dataset=dataset)
datamodule.setup("fit")

if model_type == "cond_px_dif":
    model = PixelDiffusionConditional.from_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )
elif model_type == "px_dif":
    model = PixelDiffusion.from_config(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        datamodule=datamodule,
    )
else:
    raise ValueError(f"Unsupported model_type: {model_type}")

checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

### 2) Run a single `predict_step` directly
For `PixelDiffusionConditional`, `predict_step` expects at least `x`; it will also use `valid_mask` and optional `coords` if present.

```python
batch = next(iter(datamodule.val_dataloader()))
batch = {k: (v if not torch.is_tensor(v) else v.to(model.device)) for k, v in batch.items()}

with torch.no_grad():
    pred = model.predict_step(batch, batch_idx=0)

y_hat = pred["y_hat"]               # standardized output
y_hat_denorm = pred["y_hat_denorm"] # temperature-denormalized output
```

### 3) Run prediction through Lightning
```python
trainer = Trainer(accelerator="auto", devices="auto", logger=False)
predictions = trainer.predict(model=model, dataloaders=datamodule.val_dataloader())
```

Each prediction item includes:
- `y_hat`
- `y_hat_denorm`
- `denoise_samples` (if intermediates requested)
- `sampler`

## Results
Preliminary results for sub-surface reconstruction, 50% pixelated occlusion (clustered), 24hr train time. Valid masks for training, land mask only for vosualization. Loss calculated over whole image. No inpainting pixel anchoring in DDPM sampling. PSNR ~40dB, SSIM ~0.90.
![img](assets/prelim_results2.png)  
  
Here is the same checkoint, applied to an image with 75% occlusion:
![img](assets/prelim_results_75perc.png)  



# Comments

## Known Issues
- `mask_loss_with_valid_pixels` does the inverse? 😂 - Fixed ✅, not yet tested in training run
![img](assets/val_issue.png)  
- somewhat speckled, noisy output. Ideas: DDIM sampling, structure-aware weighted loss, x0 parameterization. 

## Untested Imlpementations:
- `mask_loss_with_valid_pixels` - doesnt work - fixed ✅
- `coord_conditioning` - neither tested nor run - only implemented - works, tested ✅
- new dataset_light and datamodule not yet tested. - works ✅
- new x0 parameterization is implemented, but not tested yet

## Notes
none currently.

## ToDos
- [x] Include Deps file
- [ ] DDIM Sampling implemented but doesnt work! switching from DDPM to DDIM sampling might mess up noise schedules, but for now a DDPM checkpoint doesnt work with DDIM sampling
- [x] in dataset, implmeent bigger boxes of corruption instead of pixels
- [x] make dataset.py a save-to-disk funcitonality, then load straight form tensors
- [x] Implement masked loss for train/val for land pixels  
- [x] Implement masked loss for train/val for reconstruction pixels?
- [x] Implement two masks: known land pixels and  missing pixels? Add land to known?
- [ ] Increase unet.dim (e.g., 64 → 96 or 128), deeper level by extending dim_mults (e.g., [1, 2, 4, 8, 8])
- [x] Add known‑pixel clamping during sampling (inpainting‑style diffusion): at each step, overwrite known pixels with observed values.
- [ ] Add a frequency-aware loss like L2 on gradients or PSD loss to get rid of speckle noise in output
- [ ] Try out x0 instead of epsilon param
- [ ] Activate and test EMA Weights

## RoadMap
#### Tier 1
- [x] Aux data: coords, other priors:  
  Patch‑level [FiLM](https://arxiv.org/abs/1709.07871) conditioning with patch-center (lat, lon) embedding and ConvNeXt scale/shift injection for global geophysical priors.
- [ ] Simulate EO data img + sparse in-situ observation: 1 band surface temp + multiple bands (corrupted) for depth profile. 
- [ ] Reduce resolution to something that we could expect from Argo profiles

#### Tier 2
- [ ] Check more CopernicusMarine products like ARMOR3D as alternative data sources. 
- [x] More sophisticated way to feed masks to model, how to do it? masks * img?   
- [ ] more capable backbone?   

## Appendix: FiLM Coordinate Injection Details

### Coordinate Encoding Options
Encoding options (set with `model.coord_conditioning.encoding`):
- `unit_sphere`: Convert lat/lon to a 3D unit vector (x,y,z). This avoids lon wrap discontinuity and is the default.
- `sincos`: Use sin/cos for lat and lon (4D). Also wrap-safe, slightly higher dimensional.
- `raw`: Normalize degrees to [-1, 1] (lat/90, lon/180). Simplest but can be discontinuous at +/-180.

### Exact Injection Mechanism (Scale-Shift)
The coordinate embedding is injected via a per-channel FiLM scale and shift inside each `ConvNextBlock`.

Inside `ConvNextBlock`:
```python
self.coord_mlp = nn.Sequential(nn.GELU(), nn.Linear(coord_emb_dim, dim * 2))
...
scale_shift = self.coord_mlp(coord_emb)   # (B, 2*dim)
scale, shift = scale_shift.chunk(2, dim=1) # each (B, dim)

h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
```

That is:
```
h[b,c,x,y] <- h[b,c,x,y] * (1 + s[b,c]) + t[b,c]
```

Notes:
- `scale` and `shift` are per-sample, per-channel and broadcast to `(B, C, H, W)`.
- Applied after the depthwise conv (`ds_conv`) and before the main conv stack (`self.net`).
- This is classic FiLM conditioning: coordinates decide how strongly each channel is amplified/suppressed and offset.
- Why `1 + scale`? It keeps the identity map easy: if `scale=0` and `shift=0`, coords do nothing. This is more stable than multiplying by `scale` directly.

### Interaction With Time Conditioning
Time conditioning is additive:
```python
condition = self.mlp(time_emb)   # (B, dim)
h = h + condition[:, :, None, None]
```

So:
- Time adds a bias per channel.
- Coords do a scale-and-shift per channel.
- These are compatible: time tells the block where it is in diffusion, coords tell it where on Earth the sample belongs.
