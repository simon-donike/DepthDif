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
It can use the active pixel training super-config at
`src/depth_recon/configs/px_space/training_super_config.yaml`; the autoencoder script unwraps the top-level `data` section before constructing the dataset.

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
- the active pixel dataset returns 50 GLORYS depth channels in `y` for the temperature scenario
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
  --data-config src/depth_recon/configs/px_space/training_super_config.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml
```

### Latent Diffusion Training
`train.py` is now the pixel-super-config entry point. Latent diffusion still has its `lat_space` config files and model code, but the old shell launchers were removed with the pixel config cleanup. Reintroduce a dedicated latent launcher before starting new latent diffusion runs.

## Limitations
- Reconstruction fidelity is bounded by AE quality; diffusion cannot recover information the AE discards.
- Very aggressive compression can smooth fine vertical gradients.
- If spatial downsampling is introduced later, small-scale structures become harder to preserve.
- Latent diffusion training stability is sensitive to AE normalization, latent channel count, and mask handling consistency.

