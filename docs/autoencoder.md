# Autoencoder and Latent Components

The repository contains a depth-band autoencoder, latent diffusion model code,
configs, and dry-run tests. These are experimental building blocks: there is a
maintained autoencoder training command, but no dedicated end-to-end latent
diffusion launcher.

## Autoencoder contract

`src/depth_recon/configs/lat_space/ae_config.yaml` currently configures:

- model type `depth_band_ae`;
- 50 input depth channels and 12 latent channels;
- band-only compression with spatial downsample 1;
- encoder widths `[64, 96, 128]`;
- decoder widths `[128, 96, 64]`.

For temperature training, `ae.in_channels` must match the active dataset's 50
target channels. Reconstruction uses the configured L1/L2 weights and valid-mask
policy.

## Train the autoencoder

```bash
/work/envs/depth/bin/python train_autoencoder.py \
  --data-config src/depth_recon/configs/px_space/training_super_config.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml
```

The script unwraps the `data` section of the pixel super-config before building
the active dataset.

## Latent diffusion status

Latent configs and `latent_cond_dif` model code describe compression, latent
conditioning, and decoding. Older shell launchers were removed during the pixel
config cleanup. A new supported workflow would need a dedicated launcher that
resolves the current scenario/data contract and validates an autoencoder
checkpoint before training.

Until that exists, do not present latent diffusion as a reproducible production
workflow. Existing components remain useful for architecture tests and controlled
development.

## Constraints

- Diffusion cannot recover depth information discarded by the autoencoder.
- Changing latent channels or normalization invalidates downstream checkpoints.
- Spatial downsampling would require explicit output alignment and mask behavior.
- Joint temperature/salinity latent semantics are not defined by the current
  autoencoder preset.
