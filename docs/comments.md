# Comments

## Known Issues
- somewhat speckled, noisy output. Ideas: DDIM sampling, structure-aware weighted loss, x0 parameterization. 
- patches with large land areas make generation struggle everywhere in image

## Untested Imlpementations:
- `mask_loss_with_valid_pixels` - doesnt work - fixed ✅
- `coord_conditioning` - neither tested nor run - only implemented - works, tested ✅
- new dataset_light and datamodule not yet tested. - works ✅
- new x0 parameterization is implemented, but not tested yet  - works very well ✅
- Date + Coord embedding simplemented, but not yet tested 

## Notes
none currently.

## ToDos
- [ ] DDIM Sampling implemented but doesnt work! switching from DDPM to DDIM sampling might mess up noise schedules, but for now a DDPM checkpoint doesnt work with DDIM sampling
- [ ] Increase unet.dim (e.g., 64 → 96 or 128), deeper level by extending dim_mults (e.g., [1, 2, 4, 8, 8])
- [ ] Add a frequency-aware loss like L2 on gradients or PSD loss to get rid of speckle noise in output
- [ ] Activate and test EMA Weights

**Done**:
- [x] Encode timestamp somehow: merged in mlp with coord embeddings
- [x] Try out x0 instead of epsilon param
- [x] **Important**: Make val set geographically consistent. As in, select ~20 perc of geographic locations for val, keep them the same over time
- [x] Add known‑pixel clamping during sampling (inpainting‑style diffusion): at each step, overwrite known pixels with observed values.
- [x] in dataset, implmeent bigger boxes of corruption instead of pixels
- [x] make dataset.py a save-to-disk funcitonality, then load straight form tensors
- [x] Implement masked loss for train/val for land pixels  
- [x] Implement masked loss for train/val for reconstruction pixels?
- [x] Implement two masks: known land pixels and  missing pixels? Add land to known?
- [x] Include Deps file

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
