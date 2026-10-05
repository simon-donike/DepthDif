# Training

## Launch

Pixel-space training requires one scenario:

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
```

The resolver applies the scenario first and explicit `--set` overrides afterward.
This keeps dataset fields, salinity loading, generated channels, and condition
channels aligned.

The maintained pixel diffusion presets enable three depth-aware domain options:
`condition_per_depth_valid_mask` supplies one sparse-observation mask per output
field/depth, `condition_use_wet_mask` supplies one static physical wet-domain
channel per output field/depth, and `mask_diffusion_with_wet_mask` keeps dry
cells outside the diffusion domain during training and sampling. These options require
`data.dataset.wet_domain.enabled=true` and change the denoiser input
contract, so these options require a fresh model initialization. Evaluate existing
checkpoints using their original saved configurations. Baseline
models ignore these diffusion-only flags and retain their own channel contracts.

`model.climatology_residual=true` is an optional mode that predicts departures
from a monthly/spatial climatology. It requires
`data.dataset.climatology.enabled=true` and a non-empty artifact `path`; the
climatology must be fitted from training years only, excluding the validation
year. Fit an artifact with `depth_recon.data.fit_climatology`, then compare
fixed-sample per-depth baselines with
`depth_recon.scripts.evaluate_depth_baselines`. The current presets leave this
mode disabled. These changes affect diffusion inputs and the optional output
representation; the loss formula, weights, and supervised support remain unchanged.

See [depth diagnostics](depth-diagnostics.md) for fitting and evaluation commands.

## Maintained presets

The default local `training_super_config.yaml` and explicit
`training_super_config_standard.yaml` select the
`glorys_dense_reproduction_2016_2x3090_slow_lr` scratch experiment. They hold out **2016**
and learn dense GLORYS targets from ARGO-containing patches, matching the original
scratch run's row selection. Ambient training, synthetic targets, regional
fine-tuning, and coastal loss are disabled. Temperature uses one SST conditioning
channel; the salinity scenario uses SSS.

The local recipe uses two GPUs with DDP, mixed precision, batch size 48 per GPU
(effective batch 96), two training workers per GPU, and seed 7. The learning rate
starts at `1e-4`. The plateau scheduler checks validation loss once per epoch,
with patience 5 before halving the LR. This avoids counting a stale validation
metric as a new non-improvement on every optimizer step. ARGO filtering remains
enabled for both splits and counts temperature-valid profiles.
The HPC presets remain separate; the command below selects the local recipe.

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config.yaml \
  --scenario temperature
```

Validation stays shuffled and runs four times per epoch. Cheap denoising loss
uses at most 64 batches per GPU. Full DDPM-1000 reconstruction uses only **one
cached patch per GPU for each of raw and EMA weights**: two chains per GPU per
validation check. The EN4 candidate and hard-region inference callbacks are
disabled for this experiment, and sanity checking skips full reconstruction.
W&B logs both `val_imgs/x_y_full_reconstruction_standard` and
`val_imgs/x_y_full_reconstruction_ema`, plus separate profile plots, reusing
those same predictions. The local inference preset also selects DDPM, with
one preview patch and export batches of two.

EMA decay is `0.999`, reducing its approximate averaging timescale from 10,000
to 1,000 optimizer updates compared with `0.9999`. This reduces startup lag at
the cost of less smoothing. EMA still updates every optimizer step and remains
the default validation/checkpoint metric; compare the raw and EMA panels during
training. The original larger decay can be restored with
`--set model.ema.decay=0.9999`.

If ambient training is enabled later, samples without usable observations
contribute no supervised gradient. An entirely empty masked batch returns a
graph-connected zero so backward/DDP can complete on every rank. Keep
`require_argo_for_train=true` to avoid spending training compute on empty rows.

The configured dataset root is `/work/data/OceanVariableReconstruction`; it must
be mounted or downloaded before launching. Configuring this experiment does not
start training.

The HPC preset enables deterministic synthetic targets and disables ambient and
hard-region modes. It uses automatic visible devices with DDP, offline W&B,
batch size 96, 48 training workers, and a 10,000-epoch ceiling.

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config_hpc.yaml \
  --scenario temperature
```

The SpaceHPC GLORYS preset has the same resource envelope but supervises directly
against paired GLORYS fields instead of the synthetic prior.

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config_spacehpc_glorys.yaml \
  --scenario temperature
```

## Two-stage initialization

Stage 1 can initialize the same three-surface architecture with a deterministic
monthly/spatial surface-offset target. Stage 2 loads those weights and returns to
the observation-supported ambient objective.

```bash
# Stage 1
/work/envs/depth/bin/python train.py --scenario temperature \
  --set data.dataset.synthetic_target.enabled=true \
  --set data.dataset.selection.require_argo_for_train=false \
  --set model.ambient_occlusion.enabled=false \
  --set model.resume_checkpoint=false

# Stage 2
/work/envs/depth/bin/python train.py --scenario temperature \
  --set data.dataset.synthetic_target.enabled=false \
  --set data.dataset.selection.require_argo_for_train=true \
  --set model.ambient_occlusion.enabled=true \
  --set model.resume_checkpoint=/absolute/path/to/stage1/best.ckpt \
  --set model.load_checkpoint_only=true
```

The synthetic target is an initialization objective, not an observation or
scientific truth. Its fitter excludes 2016 and rejects source windows touching
that held-out year. See [Synthetic prior](vertical-offset-pretraining.md).

## Startup and outputs

`train.py` loads the selected YAML, resolves the scenario and overrides, builds
the active GeoTIFF dataset/datamodule, constructs the selected diffusion or
baseline model, validates checkpoint compatibility, and launches Lightning.

Each run writes under `logs/<timestamp>/`:

- `best.ckpt` and `last.ckpt` according to checkpoint configuration;
- the original super-config;
- resolved effective data, model, and training YAML snapshots;
- W&B metadata and callback outputs when enabled.

`model.resume_checkpoint` selects a checkpoint. With
`model.load_checkpoint_only=true`, only compatible weights are loaded; otherwise
Lightning restores full training state.

`train.py --run-dir <path>` selects a stable local run directory and
`--validate-only` runs the configured validation callbacks without fitting.
Optional `training.trainer.seed` and `training.trainer.early_stopping` settings
support reproducible, convergence-limited baseline runs. W&B accepts stable
`run_id`, `resume`, `group`, `job_type`, and `tags` metadata from the training
config.

## Two-GPU baseline suite

`run_baseline_2016_suite.py` trains temperature and salinity LSTM, profile-CNN,
3D U-Net, and 2D U-Net checkpoints from scratch, then validates checkpoint-free
IDW. Two workers independently bind to GPU 0 and GPU 1 and dequeue the longest
remaining jobs first. The suite fixes the validation year to 2016, disables the
hard/easy row sampler, and retains shuffled validation. By default, a logical
epoch exposes approximately 100,000 examples, validation runs after each logical
epoch, and checkpoint selection uses patience 2 under an eight-logical-epoch
cap. Recovery state is written every 5,000 optimizer steps and each training
task has a six-hour Lightning wall-time ceiling.

Use `--validation-examples`, `--max-epochs`, `--patience`,
`--checkpoint-every-n-train-steps`, and `--max-task-hours` to adjust this budget.
`--skip-models unet3d` records both 3D tasks as intentionally skipped and omits
that method from the generated evaluation configuration.

Each task logs losses, reconstruction metrics/images, EN4 candidate profiles,
and hard-region comparisons to one resumable W&B group. After training, the
best checkpoint is validated under the same W&B run ID. The `all` phase also
exports 2016-W25 EN4/GLORYS tables and spectral comparisons and uploads the
evaluation tables, plots, configs, and dashboard as a W&B artifact.

## Validation

Validation loading remains shuffled intentionally. Normal Lightning validation
can be supplemented by two configured monitors:

- EN4 candidate evaluation uniformly selects deterministic patches that retain at
  least the configured number of QC-valid input profiles, then holds out candidate
  locations within only those patches. The external candidate parquet is a
  provenance allowlist; both the remaining sparse inputs and exact held-out profile
  values come from the compact EN4/ARGO store. W&B logs profile comparisons and
  full-patch input/GLORYS/reconstruction/error images at configured depths.
- Hard-region evaluation samples deterministic 2016 patches from provisional
  hand-authored polygons and compares against GLORYS. These regions are useful
  diagnostics, not literature-backed scientific boundaries.

Both monitors are enabled in the current pixel presets. Their results are model
diagnostics and should not be presented as independent scientific validation.
