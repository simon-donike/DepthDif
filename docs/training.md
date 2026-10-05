# Training

## Launch

Latent diffusion uses a trained, calibrated mask-aware autoencoder and its own
super-config. Follow the [two-stage latent workflow](autoencoder.md) before using
`model_type: latent_cond_dif`; pixel ambient presets cannot be reused unchanged.

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

## Checkpoint selection from full reconstructions

All runs launched by `train.py` (temperature, salinity, joint, pixel/latent
diffusion and baselines) and `train_autoencoder.py` select the best checkpoint by
`val/full_reconstruction_score`, rather than denoising or training loss.
At every real validation check, a separate fixed subset of **128 unique validation
patches globally** is reconstructed in batches of four. Smaller validation datasets
use every patch. The regular validation loader remains shuffled, and its batch
limit and the preview-image count do not limit this evaluation. Sanity checks skip it.

For diffusion, `predict_step` runs the configured validation sampler from noise to
the final physical fields (DDPM-1000 in the maintained pixel presets). Baselines
run their normal prediction path; autoencoders encode and decode the complete target.
The AE callback additionally reports sparse-input and whole-profile holdout errors
on the same fixed subset, separately from the dense compression checkpoint score.
The score uses the existing validation targets and dated validity masks, intersected
with spatial ocean support. It does not change the training targets, loss or weights.
Synthetic-target runs are therefore scored against their synthetic validation
targets; the separate GLORYS comparison remains a diagnostic.

Squared errors and valid counts are pooled across patches and GPUs **before**
computing each depth's RMSE. The checkpoint score averages supported depth RMSEs,
divides each field by its existing normalization scale (10.9334°C for temperature,
1.15827 PSU for salinity), then averages fields equally. This prevents the large
number of shallow pixels or different field units from dominating selection.
Unsupported depths are excluded; an entirely unsupported field fails evaluation.
Non-finite predictions on valid water produce an infinite score, never an
artificial improvement. Physical MAE, RMSE, equal-depth errors, valid counts and
the actual patch count are also logged under `val/full_reconstruction/`.

The same 128 reconstructions also produce depth-wise mean absolute error plots
with ±1 population-standard-deviation bands, and tables of MAE, RMSE, error
standard deviation and valid counts. Temperature uses °C; salinity uses PSU.
Each field has a plot against its validation target and, where observations are
available, a paired **Prediction / GLORYS versus gridded EN4** plot. The latter
uses the conditioning observations on common observed, GLORYS-valid ocean support;
it is a conditioning-consistency diagnostic, **not independent held-out EN4
validation**. Synthetic-target runs use the separate GLORYS reference for this
comparison. Observation MAE/RMSE and equal-depth metrics are logged under
`val/full_reconstruction/en4_conditioning/`; these do not change the checkpoint
score or training loss. Statistics are pooled across GPUs before plotting on
rank zero. Unsupported depths remain empty, and invalid scored predictions are
penalized rather than dropped. These diagnostics add no inference passes and do
not change the fixed sample count, regular shuffled loader or existing previews.

Patch selection and sampling noise are seeded separately from training, with
no duplicate patches between ranks. The selected indices are recorded in
`full_reconstruction_selection.json` beside the checkpoints. When EMA evaluation
is enabled, the score uses EMA weights; raw weights are restored before saving
the normal resume checkpoint, which also retains the configured EMA callback state.
Preview plots remain independent and may use different patches.

Training retains the three lowest-scoring reconstruction checkpoints. Filenames
include both epoch and optimizer step so checks within one epoch remain distinct.
The recovery checkpoint is separate. `train.py` also writes `final.ckpt` with
the exact final training state when fitting returns, including a configured time
limit that falls between periodic recovery saves. For a comparison or deployment, copy the
chosen checkpoint, record its SHA-256, step, and raw/EMA weight choice, and use
that snapshot rather than a mutable `last.ckpt` or historical epoch-only name.
See the [checkpoint experiment record](experiments/2026-10-05-checkpoint-selection/README.md)
for the current candidate registry, evaluation protocol, and decisions.

The global sample count is a bounded default, not a guarantee of statistical
coverage. Increase it when per-depth support or checkpoint rankings are unstable:

```bash
/work/envs/depth/bin/python train.py --scenario joint \
  --set training.training.reconstruction_eval.sample_count=256 \
  --set training.training.reconstruction_eval.batch_size=4 \
  --set training.training.reconstruction_eval.seed=7
```

These settings increase validation cost, especially with DDPM. Old saved training
configs using `val/loss` or `val/loss_ckpt` for checkpoint selection are migrated
to the new score when loaded. The old loss-based best score is not comparable and
is not reused when resuming. Recovery `last.ckpt` saving and loss-based learning-rate
scheduling/early stopping retain their existing behavior.

## Held-out EN4 with no historical archive candidate

An opt-in benchmark logs under `val/en4_archive_holdout/`, separately from the
128-patch checkpoint evaluator and the conditioning-consistency plots. It uses
up to **32 fixed patches globally**, in inference batches of four, at the **first
non-sanity validation check of each epoch**. Later checks in that epoch skip it;
the callback saves its last evaluated epoch for mid-epoch resume. It uses the
active validation weights, including EMA when configured, and logs that choice.

```bash
/work/envs/depth/bin/python train.py --scenario temperature \
  --set training.training.en4_archive_holdout.enabled=true \
  --set training.training.en4_candidate_eval.enabled=false
```

Configure the audit file with
`training.training.en4_archive_holdout.candidate_profiles_path`. The default is
`instructions/en4_no_spatiotemporal_candidate_profiles.parquet`. The file must
contain `profile_source_file`, `source_profile_idx`, `datetime_utc` and
`audit_status`. Only `no_spatiotemporal_candidate` rows in the validation year
are eligible; `proxy_no_spatiotemporal_candidate` is explicitly excluded. Exact
provenance keys must match the local compact profile store. Scored references
always apply the configured accepted ARGO QC flags, even if training disables
QC filtering; the existing treatment of missing QC flags is preserved.

Selection cycles through seeded, shuffled validation dates and patch candidates
to spread the bounded subset across the year. Each patch holds out 20% of its
available candidate locations, rounded with a minimum of one, while retaining
at least eight observed grid locations. A global date/location holdout set
removes every depth of temperature and salinity, including colocated duplicate
records and overlapping patches. Only copied conditioning tensors and their
masks are edited. The shared dataset, training inputs, normal shuffled loader,
128-patch checkpoint score and loss remain unchanged. If fewer patches qualify,
the actual count is logged; an empty selection fails explicitly.

The benchmark scores the stored individual EN4 profiles on common finite
EN4/GLORYS ocean support. Each profile is assigned to exactly one reconstruction
patch. Prediction failures on scored support produce infinite error rather than
being discarded. DDP distributes explicit global batch indices without sampler
padding and pools error sums/counts across all ranks. It logs physical MAE/RMSE,
equal-depth errors, `1 - prediction_RMSE / GLORYS_RMSE` skill, supported profile
and location counts, depth MAE plots with standard-deviation bands, per-depth
metric tables and up to four profile examples per field. Temperature, salinity
and joint scenarios use the same evaluation machinery.

`en4_archive_holdout_selection.json` beside the checkpoints records the audit
file SHA-256, evidence label, QC flags, seed, budget, retained support, patch
coordinates and exact held-out source-profile identities. W&B also receives
the audit hash, selection seed and a date/location/profile table. Rank selections
must agree, and resuming into an existing run cannot silently replace its cohort.
The display label is **Held-out EN4 — no historical archive candidate**: the
audit indicates absence within its declared archive/time/distance search, not
confirmed non-assimilation by GLORYS. This selected cohort is not a claim of
uniform geographic or instrument coverage.

`sample_count`, `batch_size`, `seed`, `holdout_fraction`, `min_input_profiles`
(counted as distinct grid locations) and `max_profiles_to_plot` are configurable
under `training.training.en4_archive_holdout`. The legacy `en4_candidate_eval`
callback must be disabled because it changes the shared validation dataset.
Enabling the new benchmark adds at most 32 reconstruction patches per epoch with
the defaults; it never reconstructs the entire validation set. It does not
change or restart already-running jobs.

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
uses at most 64 batches per GPU. Preview logging uses **one cached patch per GPU
for each of raw and EMA weights**, in addition to the 128-patch checkpoint
evaluation described above. The EN4 candidate and hard-region inference callbacks are
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

- step-specific best checkpoints, recovery `last.ckpt`, and `final.ckpt` when `train.py` finishes;
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
