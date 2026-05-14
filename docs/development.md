# Development  
This page tracks current limitations, implementation status, and roadmap items.  
  
## Known Issues  
- Outputs can still look somewhat speckled/noisy.  
- Patches with large land coverage can degrade generation quality across the full patch.  
  
Potential mitigation directions already identified:  
- DDIM sampling for better compute/fidelity tradeoff  
- structure-aware or frequency-aware losses  
- parameterization/schedule tuning  
  
## Implementation Status Notes  
- `mask_loss_with_valid_pixels`: implemented and working  
- `coord_conditioning`: implemented and tested  
- lightweight dataset + datamodule path: implemented and working  
- `x0` parameterization: implemented and working well in current experiments  
- combined date + coordinate embedding: implemented, now exercised in EO config  

## Repository Layout  
- installable package code lives under `src/depth_recon/`  
- bundled configs live under `src/depth_recon/configs/`  
- local launcher scripts live under `src/depth_recon/scripts/`  
- root-level `train.py`, `train_autoencoder.py`, and `experiments.py` remain direct local entry points  
- package modules should be imported through the `depth_recon.*` namespace  
- generated runtime outputs default to root-level `inference/outputs/`; historical moved outputs under `src/depth_recon/inference/outputs/` are also ignored and excluded from package builds  
  
## ToDos  
- [ ] DDIM sampling path still needs deeper validation across checkpoints/settings  
- [ ] Increase U-Net capacity (for example `dim: 64 -> 96/128`, deeper `dim_mults`)  
- [ ] Add frequency-aware objectives (for example gradient/PSD losses) to reduce speckle noise  
- [ ] Validate and tune EMA weights in full training runs  
  
## Done  
- [x] Encode timestamps together with coordinate embedding  
- [x] Add and test `x0` parameterization path  
- [x] Establish geographically consistent window split tooling  
- [x] Implement known-pixel clamping mechanism for sampling  
- [x] Use larger corruption patches instead of isolated single pixels  
- [x] Add dataset-to-disk export pipeline  
- [x] Implement masked loss support for land/validity handling  
- [x] Maintain dependency list in repository  
  
## Roadmap  
### Tier 1  
- [x] Aux priors via patch-level FiLM conditioning from coordinates (and optional date)  
- [x] Increase sparse-input stress test to `mask_fraction=0.975` as a standard comparison setting  
- [x] Implement trajectory-style corruption ("walk" masks) to better simulate submarine-like movement across each patch  
- [x] Simulate EO observation + sparse in-situ measurement setup more systematically: trajectory & OSTIA dataset  
- [ ] Evaluate lower-resolution setups aligned with expected sparse in-situ measurement density  
  
### Tier 2  
- [ ] Evaluate additional Copernicus Marine products (for example ARMOR3D)  
- [x] Improve mask handling design in conditional inputs  
  
