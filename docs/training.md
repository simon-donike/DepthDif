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

## Maintained presets

The local `training_super_config.yaml` and explicit standard preset train on real
sparse EN4/ARGO support. They enable the ambient objective, EMA, a 50/50
hard-region/easy-region row mix for both training and validation, and 100-step
DDIM validation reconstruction. Synthetic targets and coastal loss are disabled.

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config.yaml \
  --scenario temperature
```

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
