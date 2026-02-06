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

## Model
As a first prototype, a conditional pixel-space Diffuser is modeled after [DiffusionFF](https://github.com/mikonvergence/DiffusionFastForward)

## Results
Preliminary results for sub-surface reconstruction, 50% occlusion, 3hr train time
![img](assets/prelim_results.png)  

## Environment & Dependencies

- The project uses **Python 3.12.3**.
- All Python dependencies are listed in a single `requirements.txt` file located at the **repository root**.
- Install dependencies with:
```bash
pip install -r requirements.txt
```

# Comments

## Notes
Currently num_workers=0 and pin_mermory=False due to previous PID datalader death. This way, GPUs arent saturated. Find this error and put up again for effective training. ✅ - reduced val workers to 0, increased num_workers and pin_memory=True, bac to good saturation.


## ToDos
- [x] Include Deps file
- [x] DDIM Sampling
- [ ] Reduce resolution to something that we could expect from Argo profiles
- [ ] in dataset, implmeent bigger boxes of corruption instead of pixels
- [ ] make dataset.py a save-to-disk funcitonality, then load straight form tensors
- [x] Implement masked loss for train/val for land pixels  
- [x] Implement masked loss for train/val for reconstruction pixels?
- [x] Implement two masks: known land pixels and  missing pixels? Add land to known?
- [ ] Increase unet.dim (e.g., 64 → 96 or 128), deeper level by extending dim_mults (e.g., [1, 2, 4, 8, 8])

## RoadMap
#### Tier 1
- [ ] Simulate EO data img + sparse in-situ observation: 1 band surface temp + multiple bands (corrupted) for depth profile. 
- [ ] Aux data: coords, other priors, etc: How to to include them? Idea:  
    - Patch‑level [FiLM](https://arxiv.org/abs/1709.07871) conditioning:
        - Compute patch center (lat, lon), embed with an MLP, and inject via FiLM (scale/shift) in ConvNeXt blocks.
        - => global geophysical priors without a full coord grid.
        - edit UnetConvNextBlock to accept an extra embedding and applying it inside blocks.
- [x] Add known‑pixel clamping during sampling (inpainting‑style diffusion): at each step, overwrite known pixels with observed values.

#### Tier 2
- [ ] Check more CopernicusMarine products like ARMOR3D as alternative data sources. 
- [ ] More sophisticated way to feed masks to model, how to do it? masks * img?   
- [ ] more capable backbone?   
