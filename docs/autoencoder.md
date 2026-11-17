# Autoencoder and latent diffusion

The supported workflow first trains a mask-aware depth autoencoder, then freezes
it for dense-target conditional latent diffusion. The default temperature preset
compresses 50 depths into 12 channels without spatial downsampling. There is no
latent-size sweep. Sparse ARGO observations condition the diffusion model; dense
GLORYS fields provide its training targets.

## Autoencoder inputs and losses

The encoder receives three separate channel groups: normalized values,
per-depth observation validity, and per-depth physical wet-domain support. Invalid
values are sanitized before encoding; a valid zero remains distinguishable from a
missing value. The existing spatial convolutions mix neighboring profiles. The
decoder reconstructs the original physical field channels.

Training combines three independently normalized L1/L2 reconstruction losses:

- Dense GLORYS input, supervised on valid GLORYS values.
- GLORYS sampled using ARGO profile/depth coverage, supervised on valid GLORYS
  values. Patches without observations use randomly located partial profiles.
- Real sparse ARGO input, supervised only on observed ARGO values.

Each case has weight 1 by default. Missing cells never become zero-valued targets;
empty support contributes zero loss. Sparse and dense encodings are not forced to
be identical. The model's guesses between observations are not treated as labels.

The AE runner uses explicit dataset `train` and `val` splits, including the
configured validation year. Validation loading remains shuffled. A separate fixed
subset with reproducible corruption seeds reports physical-unit errors by depth
for dense reconstruction, sparse-input dense reconstruction, visible observations,
and held-out whole profiles. Patches with fewer than two observed profiles do not
contribute to held-profile scores. Best checkpoints use **dense round-trip
reconstruction** on this fixed subset; sparse and held-profile scores are separate
diagnostics.

## Stage 1: train and export the autoencoder

Run from the repository root. Set dataset paths in the latent super-config first.
The default logger operates offline and does not upload runs.

```bash
/work/envs/depth/bin/python train_autoencoder.py \
  --data-config src/depth_recon/configs/lat_space/training_super_config.yaml \
  --train-config src/depth_recon/configs/lat_space/training_config.yaml \
  --ae-config src/depth_recon/configs/lat_space/ae_config.yaml \
  --run-dir logs/ae_latent
```

Use `--resume-checkpoint logs/ae_latent/last.ckpt` to restore training state, or
`--load-checkpoint logs/ae_latent/final.ckpt` for weight-only initialization.
Externally launched DDP requires the same explicit `--run-dir` on every rank.

After fitting, the runner reloads the best reconstruction checkpoint, estimates
per-channel latent mean/std on up to `ae.training.calibration_batches` **training
batches only**, and writes `logs/ae_latent/autoencoder_calibrated.ckpt`. This export
contains AE weights, latent statistics, calibration provenance, field ordering,
normalization constants, and the versioned mask contract. Calibration excludes
columns without valid targets and never uses validation samples.

AE optimizer settings and epoch count come from `ae.training`; loader and trainer
settings come from the training config. Effective resolved configs are retained
under the run directory.

## Stage 2: train latent diffusion

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/lat_space/training_super_config.yaml \
  --run-dir logs/latent_temperature \
  --set model.latent.ae_checkpoint=logs/ae_latent/autoencoder_calibrated.ckpt
```

The resolver derives physical channels independently from the 12 denoised latent
channels. The default condition contains 12 encoded observation channels, three
EO channels `[SST, SSS, ADT]`, 50 observation masks, 50 wet-domain masks, and one
spatial land/ocean mask. Coordinates and date use the existing conditioning path.

Dense targets and sparse conditions use the same frozen encoder and fixed latent
normalization. The latent ocean domain is the union of wet depths at each
horizontal location: encoded channels mix depths and cannot have separate physical
seafloor masks. Final outputs and observation losses retain per-depth support.

The optional `model.latent.decoded_observation_weight` adds L1 supervision against
real observations after decoding the predicted clean latent. Default: zero. The
decoder stays frozen while allowing gradients to flow back into the denoiser.

Checkpoint selection uses the existing fixed full-reconstruction callback on
**decoded physical fields**, not latent denoising loss. To resume, add:

```text
--set model.resume_checkpoint=logs/latent_temperature/last.ckpt
```

Keep the compatible calibrated AE export available when constructing or restoring
a latent model. Old mask-unaware AE/latent checkpoints are incompatible and must be
retrained; frozen random autoencoders and uncalibrated exports are rejected.

## Inference

Run a single validation-batch prediction with:

```bash
/work/envs/depth/bin/python -m depth_recon.inference.run_single \
  --config src/depth_recon/configs/lat_space/training_super_config.yaml \
  --checkpoint logs/latent_temperature/last.ckpt \
  --set model.latent.ae_checkpoint=logs/ae_latent/autoencoder_calibrated.ckpt
```

The latent super-config also includes an `inference` section and works with the
existing inference config resolver and model factory. For programmatic inference,
load the resolved model and diffusion checkpoint with the existing
`depth_recon.inference.core` functions, then call `predict_step` with physical
observations, their depth masks, static wet-domain masks, EO, coordinates/date,
and target/output support masks. Returned `y_hat_<field>_denorm` tensors are in
physical units. DDIM/DDPM sampling and ensemble `uncertainty_step` share this path.
Diffusion checkpoint loading validates the AE contract and strictly matches weights.

## Scenarios and supported options

Temperature, salinity, and joint tensor bookkeeping is supported. Set the data
super-config scenario and AE `output_fields` together; use 50 AE input channels for
a single field or 100 for `[temperature, salinity]`. Each scenario requires its own
trained AE export. The supplied runnable preset is temperature.

Climatology residuals are supported only when both AE and diffusion are configured
for residuals and the dataset supplies a training-only fitted background. Subtraction
happens before encoding and restoration after decoding.

Ambient corruption, observation clamping in latent space, spatial downsampling,
synthetic-prior confidence targets, and pixel-space auxiliary losses are rejected.
They cannot be transferred by interpreting latent channels as physical depths.
Compression may discard fine vertical structure; inspect the AE's depth and
held-profile errors before committing to a long diffusion run. This workflow does
not establish that latent diffusion outperforms the pixel model.
