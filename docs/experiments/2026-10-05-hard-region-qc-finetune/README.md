# Standard and residual hard-region/QC fine-tuning — 5 October 2026

Status: **both runs training on separate GPUs with offline tracking; initial
training losses verified finite**. The user requested one standard-model
fine-tuning run on GPU 0 and one climatology-residual run on GPU 1, both with
hard-region sampling and ARGO quality filtering enabled.

## Checkpoint lineage

| Run | GPU | Parent checkpoint | Starting tensors | Local W&B run ID |
|---|---:|---|---|---|
| Standard | 0 | Sept 30 reproduction, step 56,652, epoch 1 | EMA | `cc38592a` |
| Residual | 1 | Oct 3 residual, step 113,304, epoch 2 | Raw | `1419568d` |

[experiment.yaml](experiment.yaml) pins parent and initialization-file SHA-256
hashes, configuration hashes, source W&B runs, log directories, and GPU/session
assignments. The standard parent is copied read-only before extracting its EMA
tensors. The residual uses the existing verified raw-weight export. Each
initialization file has exactly one weight variant; the EMA-preferring loader
cannot silently choose different tensors. Original checkpoints remain intact.

The standard parent is the best available local standard checkpoint from the
earlier comparison; the HPC checkpoint remains unavailable. The residual parent
is the numerical field winner of the 128-patch comparison, effectively tied with
early EMA. See the [selection record](../2026-10-05-checkpoint-selection/README.md).
These runs preserve their respective architectures (standard 103 total input
channels; residual 252). They are not an isolated test of adding climatology:
their histories and mask-conditioning architectures also differ.

## Shared fine-tuning policy

- Load selected weights into a **new fine-tuning stage**, with fresh optimizer,
  scheduler, and EMA state. Child optimizer steps start at zero; parent steps
  are recorded separately. This avoids restoring the old optimizer's learning
  rate and denoising-loss scheduler into the changed data distribution.
- Learning rate **1e-5**, ten times lower than the parents' 1e-4; seed 7;
  batch size 48 per run, one GPU each, fp16 mixed precision. The parent runs
  used two GPUs and an effective batch of 96. Both new runs use the same batch
  size of 48. One loader worker and prefetch factor one bound host-memory use.
- Enable existing hard-region sampling with a **50% hard / 50% general** row
  target, existing region boxes, and relaxed coastal inclusion. Apply this to
  training only; keep global validation coverage. The actual counts/fraction
  are checked rather than assuming the requested mix was attained.
- Enable QC for both training and validation **before profile rasterization**.
  Accepted known QC codes are 1 and 2; the existing loader also accepts missing
  negative QC values. Known QC 4 profiles identified in the audit must be rejected.
  Build fresh QC-aware metadata caches shared by the two runs.
- Keep 2016 held out and the ordinary validation loader **shuffled**. Run
  validation every 2,000 training batches, with 64 ordinary validation batches,
  fixed 128-patch DDPM-1000 reconstruction scoring, and the existing hard-region
  diagnostic callback. Regional polygons are the repository's provisional
  development definitions. The fixed global reconstruction subset is selected
  from the QC-filtered validation population.
- Keep the **three best checkpoints** by `val/full_reconstruction_score`,
  with optimizer steps in filenames. Save recovery checkpoints every 1,000
  optimizer steps and on exceptions; save `final.ckpt` at normal completion,
  including the configured time limit. The plateau scheduler watches the
  same reconstruction score. EMA evaluation is enabled for both child runs.
- Initial stage: **24-hour training limit**, subject to the user's requested
  duration preference, with the inherited 100-epoch ceiling. This is a bounded
  first stage, not a claim that 24 hours is the optimum training duration.

The residual keeps the frozen training-only monthly climatology and depth-aware
wet mask. The standard keeps its original absolute-temperature model and mask
contract. These runs implement the user's requested continuation; no extra
ablations or ensemble evaluation are launched.

## Evidence and operation

Exact launch configs: [standard.yaml](standard.yaml), [residual.yaml](residual.yaml).
The six `*_parent_*.yaml` files preserve each parent's effective configurations.
Preflight verifies strict model loading and exact tensor equality, matching row
selection settings, split years, actual hard/easy counts, and rejection of the
five known QC-4 source profiles. Its result is recorded in [preflight.json](preflight.json):
442,638 hard training rows plus 442,638 general rows, and 173,137 validation rows.

`outputs/hard_region_qc_finetune_20261005/provenance/` stores the full source
snapshot, initial working-tree diff/status, preflight helper/log, formatter output,
and full test logs. The initial suite passed **389 tests in 19.688 seconds**.
The only production change for this launch saves `final.ckpt` after training
returns, so reaching the time limit cannot leave only an older periodic recovery
checkpoint. The full suite then passed **389 tests in 20.111 seconds**. Required `black .` output,
including its Python target-version warning, is retained in the log.

Training commands, run from the repository root in separate persistent sessions:

```bash
CUDA_VISIBLE_DEVICES=0 /work/envs/depth/bin/python -u train.py \
  --config docs/experiments/2026-10-05-hard-region-qc-finetune/standard.yaml \
  --run-dir logs/2026-10-05_hard_region_qc_standard
CUDA_VISIBLE_DEVICES=1 /work/envs/depth/bin/python -u train.py \
  --config docs/experiments/2026-10-05-hard-region-qc-finetune/residual.yaml \
  --run-dir logs/2026-10-05_hard_region_qc_residual
```

The actual launch record includes thread/cache environment settings and detached
session names. Do not execute these commands again while those sessions run.
Fresh W&B IDs prevent the fine-tuning stage from being merged into a parent run.
Compare subsequent checkpoint scores within this new QC-aware protocol; the old
unfiltered benchmark and profile errors are preserved as historical evidence.

## Launch events and tracking permission

The proposed online launch was rejected by automatic approval review before any
process started: exporting metrics, configs, checkpoint provenance, and validation
figures to W&B required explicit approval of that payload and destination. The
user was asked about uploading to `esa-phi-lab/DepthDif_Simon`; approval is pending.
Both jobs therefore use `offline: true`, `WANDB_MODE=offline`, and `log_model: false`.
No model/checkpoint binaries or dataset files are configured for upload. Proposed
online configs and the blocked attempt remain archived; future syncing must wait
for explicit approval. The URLs in the manifest are planned destinations, not
evidence of active remote runs.

The first offline launcher started the standard session, then failed to record its
PID because a `tmux display-message` target returned an empty string. The live
process was verified using `list-panes` and retained. After correcting only the
launch helper, the residual session started; no duplicate standard process or
training restart occurred. Session identities and launch times are recorded in
the manifest. Local W&B data is under the repository's `wandb/offline-run-*` paths;
the logger's save directory takes precedence over the suggested `WANDB_DIR`.

[startup_verification.json](startup_verification.json) records the initial flushed
training metrics and checks of the configurations actually loaded by each job.
Standard reached at least logged step 49; residual at least step 24, with finite
training losses. These are startup checks, not validation results. The live local
W&B files buffer records, so those steps are lower bounds on current progress.
Processes 2726741 and 2727215 were verified on physical GPUs 0 and 1 respectively.

Attach to the persistent sessions with:

```bash
tmux attach -t depthdif_ft_standard_20261005
tmux attach -t depthdif_ft_residual_20261005
```

Output is redirected to `outputs/hard_region_qc_finetune_20261005/standard.log`
and `residual.log`. Fine-tuning checkpoints are written to
`logs/2026-10-05_hard_region_qc_standard/` and
`logs/2026-10-05_hard_region_qc_residual/`.
