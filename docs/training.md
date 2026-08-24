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

## Validation

Validation loading remains shuffled intentionally. Normal Lightning validation
can be supplemented by two configured monitors:

- EN4 candidate evaluation holds out deterministic profile locations from the
  sparse input and compares reconstructed values at those exact locations.
- Hard-region evaluation samples deterministic 2016 patches from provisional
  hand-authored polygons and compares against GLORYS. These regions are useful
  diagnostics, not literature-backed scientific boundaries.

Both monitors are enabled in the current pixel presets. Their results are model
diagnostics and should not be presented as independent scientific validation.
