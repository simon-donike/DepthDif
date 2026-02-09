# Densifying Sparse Ocean Depth Observations
This implementation is a first test, checking the feasability of densifying sparse ocean measurements.

## Data
Currently, monthly tiles from 2000 - 2025 from the [Global Ocean Physics Reanalysis dataset](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/files?subdataset=cmems_mod_glo_phy_my_0.083deg_P1M-m_202311&path=GLOBAL_MULTIYEAR_PHY_001_030%2Fcmems_mod_glo_phy_my_0.083deg_P1M-m_202311%2F2024%2F) have been downloaded and are manually masked to simulate real sparse observations. Excluding patches with >20% NoData values, ~106k samples are avaialble (128x128, 1/12 °). Download the data by installing the `copernicusmarine` package, then use the CLI like so `copernicusmarine get -i cmems_mod_glo_phy_my_0.083deg_P1M-m  --filter "*2021/*"`  
The of the obstructions and the coverage percentage are selectable in the `data_config.yaml`.

Dataset example for 50% occlusion:  
![img](assets/dataset_50percMask.png)  

Current Status:
- 1-band only for experimentation
- 128x128 hardcoded

### Repository Tweaks (Data)
- Synthetic occlusion pipeline to create sparse observations with configurable `mask_fraction`.
- Patch-based masking with min/max patch sizes (`mask_patch_min`, `mask_patch_max`) instead of single-pixel drops.
- Validity/land masks derived from nodata or fill values; invalid pixels are tracked separately from corruption.
- Optional filtering of tiles by `max_nodata_fraction` to avoid overly invalid patches.
- Corrupted input + mask channel return modes for conditional modeling (`x_return_mode`).
- Z-score temperature normalization and optional geometric augmentation (rotations/flips) applied consistently to data and masks.
- Dataset index build with nodata-fraction metadata for fast filtering.

## Model
As a first prototype, a conditional pixel-space Diffuser is modeled after [DiffusionFF](https://github.com/mikonvergence/DiffusionFastForward).  

The model is trained on 1-channel temp + valid pixel mask. Loss can be pulled including or excluding the mask.

### Repository Tweaks (Model)
- Condition channels include both data and mask (`condition_channels`, `condition_mask_channels`).
- Masked loss option that restricts training loss to valid pixels (`mask_loss_with_valid_pixels`).
- Inpainting-style known-pixel clamping during sampling for stability (`clamp_known_pixels`).
- ReduceLROnPlateau learning-rate scheduler configuration and logging support.

### Repository Tweaks (Training/Logging)
- PSNR and SSIM computed during validation (when `skimage` is available).
- Validation-time sampling (DDPM or DDIM) for qualitative reconstruction checks.
- Per-epoch cached validation example used for full reconstruction logging.
- W&B image logging for inputs, targets, predictions, masks, and reconstruction grids.
- Periodic stats logging (e.g., masked fraction, stdev etc) during train/val.
- Checkpointing + resume support, plus learning-rate monitoring callbacks.
- Optional W&B `watch` settings for gradients/parameters/graphs.

### Sampling
- DDIM/DDPM sampling possible
- inpaitning-style injection of known values during generation can be turned on

## Results
Preliminary results for sub-surface reconstruction, 50% pixelated occlusion (clustered), 24hr train time. Valid masks for training, land mask only for vosualization. Loss calculated over whole image. No inpainting pixel anchoring in DDPM sampling.
![img](assets/prelim_results2.png)  

## Environment & Dependencies

- The project uses **Python 3.12.3**.
- All Python dependencies are listed in a single `requirements.txt` file located at the **repository root**.
- Install dependencies with:
```bash
pip install -r requirements.txt
```

# Comments

## Issues
- `mask_loss_with_valid_pixels` does the inverse? 😂

## Notes
Currently num_workers=0 and pin_mermory=False due to previous PID datalader death. This way, GPUs arent saturated. Find this error and put up again for effective training. ✅ - reduced val workers to 0, increased num_workers and pin_memory=True, bac to good saturation.


## ToDos
- [x] Include Deps file
- [x] DDIM Sampling
- [ ] Reduce resolution to something that we could expect from Argo profiles
- [x] in dataset, implmeent bigger boxes of corruption instead of pixels
- [ ] make dataset.py a save-to-disk funcitonality, then load straight form tensors
- [x] Implement masked loss for train/val for land pixels  
- [x] Implement masked loss for train/val for reconstruction pixels?
- [x] Implement two masks: known land pixels and  missing pixels? Add land to known?
- [ ] Increase unet.dim (e.g., 64 → 96 or 128), deeper level by extending dim_mults (e.g., [1, 2, 4, 8, 8])

## RoadMap
#### Tier 1
- [ ] Aux data: coords, other priors, etc: How to to include them? Idea:  
    - Patch‑level [FiLM](https://arxiv.org/abs/1709.07871) conditioning:
        - Compute patch center (lat, lon), embed with an MLP, and inject via FiLM (scale/shift) in ConvNeXt blocks.
        - => global geophysical priors without a full coord grid.
        - edit UnetConvNextBlock to accept an extra embedding and applying it inside blocks.
- [ ] Simulate EO data img + sparse in-situ observation: 1 band surface temp + multiple bands (corrupted) for depth profile. 
- [x] Add known‑pixel clamping during sampling (inpainting‑style diffusion): at each step, overwrite known pixels with observed values.

#### Tier 2
- [ ] Check more CopernicusMarine products like ARMOR3D as alternative data sources. 
- [ ] More sophisticated way to feed masks to model, how to do it? masks * img?   
- [ ] more capable backbone?   
