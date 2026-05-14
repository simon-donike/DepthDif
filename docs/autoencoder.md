# Autoencoder  
This page describes the autoencoder-first latent diffusion workflow used for high-band depth reconstruction.  

## Why Autoencoder + Latent Diffusion  
With many depth levels (for example 50 bands), direct pixel-space diffusion becomes expensive in memory and training time.  
The latent workflow solves this by:  
- compressing the depth stack into a compact latent representation  
- running diffusion in that latent space  
- decoding the generated latent back to full-band depth fields  

In this repository, the latent workflow config lives in:  
- `src/depth_recon/configs/lat_space/model_config.yaml`  
- `src/depth_recon/configs/lat_space/training_config.yaml`  
- `src/depth_recon/configs/lat_space/ae_config.yaml`  
It uses the active NetCDF dataset config at  
`src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml`.  

## Task  
Learn a compressed latent representation for multiband depth fields that preserves:  
- vertical profile structure across depth levels  
- spatial coherence of ocean patterns  
- compatibility with sparse-conditioning diffusion inputs  

## Goal  
Use the autoencoder as the representation bridge so latent diffusion can model fewer channels than the original depth stack while keeping reconstruction quality high.  

## Architecture  
The autoencoder configuration is defined in `src/depth_recon/configs/lat_space/ae_config.yaml`.  

Current default design:  
- model type: `depth_band_ae`  
- input channels: `50`  
- latent channels: `12`  
- spatial downsample: `1` (band-only compression, no spatial downsampling)  
- encoder hidden channels: `[64, 96, 128]`  
- decoder hidden channels: `[128, 96, 64]`  

Data contract for this default:  
- the active NetCDF dataset returns 50 GLORYS depth channels in `y`  
- `ae.in_channels` should match the dataset target channel count  

Loss controls:  
- `ae.loss.recon_l1_weight`  
- `ae.loss.recon_l2_weight`  
- `ae.loss.masked_only`  

Training controls:  
- `ae.training.lr`  
- `ae.training.batch_size`  
- `ae.training.max_epochs`  

## End-to-End Latent Workflow  
1. Train the autoencoder on full-band depth targets.  
2. Freeze or partially freeze the AE (configured via latent model settings).  
3. Train latent diffusion with:  
   - `model.model_type: "latent_cond_dif"`  
   - latent conditioning channels (`z_x`, EO, valid mask)  
4. Decode predicted latent outputs back to full-band depth space.  

## Scripts and Commands  
Use `/work/envs/depth/bin/python` for all training commands.  

### Command: Autoencoder Training  
```bash
/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml
```

### Command: Latent Diffusion Training  
```bash
/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --model-config src/depth_recon/configs/lat_space/model_config.yaml
```

### Script: `src/depth_recon/scripts/train_autoencoder.sh`  
```bash
#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train_autoencoder.py \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml
```

### Script: `src/depth_recon/scripts/train_latent_diffusion.sh`  
```bash
#!/usr/bin/env bash
set -euo pipefail

/work/envs/depth/bin/python train.py \
  --data-config src/depth_recon/configs/px_space/data_ostia_argo_netcdf.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --model-config src/depth_recon/configs/lat_space/model_config.yaml
```

## Limitations  
- Reconstruction fidelity is bounded by AE quality; diffusion cannot recover information the AE discards.  
- Very aggressive compression can smooth fine vertical gradients.  
- If spatial downsampling is introduced later, small-scale structures become harder to preserve.  
- Latent diffusion training stability is sensitive to AE normalization, latent channel count, and mask handling consistency.  

