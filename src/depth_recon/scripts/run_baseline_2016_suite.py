# Example:
# /work/envs/depth/bin/python src/depth_recon/scripts/run_baseline_2016_suite.py --phase all --config src/depth_recon/configs/px_space/training_super_config.yaml --inference-config src/depth_recon/configs/px_space/inference_super_config.yaml --output-root logs/baseline_2016_global --gpu-indices 0 1 --project DepthDif_Simon --entity esa-phi-lab --candidate-profiles instructions/en4_no_spatiotemporal_candidate_profiles.parquet --year 2016 --iso-week 25 --seed 7 --max-epochs 8 --patience 2 --validation-examples 100000 --checkpoint-every-n-train-steps 5000 --max-task-hours 6 --skip-models unet3d --resume-incomplete --max-wandb-images 100
"""Run the complete two-GPU 2016 baseline training and evaluation suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Sequence

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = Path("/work/envs/depth/bin/python")
DEFAULT_CONFIG = (
    REPO_ROOT / "src/depth_recon/configs/px_space/training_super_config.yaml"
)
DEFAULT_INFERENCE_CONFIG = (
    REPO_ROOT / "src/depth_recon/configs/px_space/inference_super_config.yaml"
)
DEFAULT_CANDIDATE_PROFILES = (
    REPO_ROOT / "instructions/en4_no_spatiotemporal_candidate_profiles.parquet"
)
MANIFEST_NAME = "baseline_suite_manifest.json"


@dataclass(frozen=True)
class ModelSpec:
    """One trainable or checkpoint-free baseline architecture."""

    name: str
    label: str
    model_type: str
    train_batch_size: int
    val_batch_size: int
    trainable: bool = True


@dataclass(frozen=True)
class SuiteTask:
    """One model-variable training or validation unit."""

    model: ModelSpec
    scenario: str

    @property
    def key(self) -> str:
        """Return the stable task identifier used by manifests and W&B."""
        return f"{self.model.name}_{self.scenario}"


MODEL_SPECS = (
    ModelSpec("unet3d", "U-Net", "unet_baseline", 8, 4),
    ModelSpec("unet2d", "U-Net 2D", "unet2d_baseline", 48, 6),
    ModelSpec("cnn", "CNN", "cnn_baseline", 9, 9),
    ModelSpec("lstm", "LSTM", "lstm_baseline", 1, 1),
    ModelSpec("idw", "IDW", "idw_baseline", 1, 6, trainable=False),
)
SCENARIOS = ("temperature", "salinity")


def _build_parser() -> argparse.ArgumentParser:
    """Build the baseline-suite command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the two-GPU 2016 DepthDif baseline suite."
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--inference-config", type=Path, default=DEFAULT_INFERENCE_CONFIG
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu-indices", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--project", default="DepthDif_Simon")
    parser.add_argument("--entity", default="esa-phi-lab")
    parser.add_argument(
        "--candidate-profiles", type=Path, default=DEFAULT_CANDIDATE_PROFILES
    )
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--iso-week", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--validation-examples", type=int, default=100_000)
    parser.add_argument("--checkpoint-every-n-train-steps", type=int, default=5_000)
    parser.add_argument("--max-task-hours", type=int, default=6)
    parser.add_argument(
        "--skip-models",
        nargs="*",
        choices=tuple(model.name for model in MODEL_SPECS if model.trainable),
        default=(),
    )
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-wandb-images", type=int, default=100)
    return parser


def _suite_id(output_root: Path) -> str:
    """Return a W&B-safe stable suite identifier from the output directory."""
    cleaned = "".join(
        character if character.isalnum() else "-" for character in output_root.name
    ).strip("-")
    return (cleaned or "baseline-2016-global")[:80]


def _task_run_id(suite_id: str, task_key: str) -> str:
    """Return a deterministic compact W&B run id for a suite task."""
    digest = hashlib.sha1(f"{suite_id}:{task_key}".encode("utf-8")).hexdigest()[:16]
    return f"b16{digest}"


def _default_output_root() -> Path:
    """Return a timestamped default suite output directory."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return REPO_ROOT / "logs" / f"baseline_2016_global_{stamp}"


def _task_list() -> list[SuiteTask]:
    """Return longest-first tasks so two GPUs minimize total wall time."""
    return [
        SuiteTask(model=model, scenario=scenario)
        for model in MODEL_SPECS
        for scenario in SCENARIOS
    ]


def _set_arg(path: str, value: Any) -> list[str]:
    """Return one strict train.py YAML override argument pair."""
    # JSON scalars/lists are valid YAML and avoid PyYAML's scalar ``...`` trailer.
    return ["--set", f"{path}={json.dumps(value)}"]


def _logical_epoch_batches(args: argparse.Namespace, task: SuiteTask) -> int:
    """Return batches that expose approximately the requested example count."""
    validation_examples = int(args.validation_examples)
    if validation_examples < 1:
        raise ValueError("--validation-examples must be >= 1.")
    batch_size = int(task.model.train_batch_size)
    return max(1, (validation_examples + batch_size - 1) // batch_size)


def _common_overrides(
    *,
    args: argparse.Namespace,
    task: SuiteTask,
    suite_id: str,
    job_type: str,
) -> list[str]:
    """Build the shared fair-training and W&B overrides for one task."""
    checkpoint_interval = int(args.checkpoint_every_n_train_steps)
    max_task_hours = int(args.max_task_hours)
    if checkpoint_interval < 1:
        raise ValueError("--checkpoint-every-n-train-steps must be >= 1.")
    if max_task_hours < 1:
        raise ValueError("--max-task-hours must be >= 1.")
    run_name = f"{suite_id}-{task.model.name}-{task.scenario}"
    tags = [
        "baseline",
        "validation-2016",
        "global-rows",
        task.model.name,
        task.scenario,
    ]
    values = (
        ("model.model_type", task.model.model_type),
        ("data.split.val_year", int(args.year)),
        ("data.dataset.finetune_sampling.enabled", False),
        ("data.dataset.synthetic_target.enabled", False),
        ("training.trainer.seed", int(args.seed)),
        ("training.trainer.max_epochs", int(args.max_epochs)),
        ("training.trainer.accelerator", "gpu"),
        ("training.trainer.devices", 1),
        ("training.trainer.strategy", "auto"),
        ("training.trainer.precision", "16-mixed"),
        ("training.trainer.val_check_interval", 1.0),
        ("training.trainer.limit_train_batches", _logical_epoch_batches(args, task)),
        (
            "training.trainer.checkpoint_every_n_train_steps",
            checkpoint_interval,
        ),
        (
            "training.trainer.max_time",
            f"00:{max_task_hours:02d}:00:00",
        ),
        ("training.dataloader.batch_size", task.model.train_batch_size),
        ("training.dataloader.val_batch_size", task.model.val_batch_size),
        ("training.wandb.offline", False),
        ("training.wandb.project", str(args.project)),
        ("training.wandb.entity", str(args.entity)),
        ("training.wandb.run_name", run_name),
        ("training.wandb.run_id", _task_run_id(suite_id, task.key)),
        ("training.wandb.resume", "allow"),
        ("training.wandb.group", suite_id),
        ("training.wandb.job_type", job_type),
        ("training.wandb.tags", tags),
        ("training.wandb.log_model", False),
        ("training.wandb.verbose", True),
    )
    overrides: list[str] = []
    for path, value in values:
        overrides.extend(_set_arg(path, value))
    return overrides


def _training_command(
    *,
    args: argparse.Namespace,
    task: SuiteTask,
    suite_id: str,
    run_dir: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    """Build a scratch or full-state-resume training command."""
    command = [
        str(PYTHON_BIN),
        str(REPO_ROOT / "train.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--scenario",
        task.scenario,
        "--run-dir",
        str(run_dir),
    ]
    command.extend(
        _common_overrides(
            args=args,
            task=task,
            suite_id=suite_id,
            job_type="training",
        )
    )
    command.extend(_set_arg("training.trainer.early_stopping.enabled", True))
    command.extend(_set_arg("training.trainer.early_stopping.monitor", "val/loss_ckpt"))
    command.extend(_set_arg("training.trainer.early_stopping.mode", "min"))
    command.extend(
        _set_arg("training.trainer.early_stopping.patience", int(args.patience))
    )
    command.extend(
        _set_arg(
            "model.resume_checkpoint",
            False if resume_checkpoint is None else str(resume_checkpoint.resolve()),
        )
    )
    command.extend(_set_arg("model.load_checkpoint_only", False))
    return command


def _validation_command(
    *,
    args: argparse.Namespace,
    task: SuiteTask,
    suite_id: str,
    run_dir: Path,
    checkpoint: Path | None,
) -> list[str]:
    """Build one checkpoint-selected or checkpoint-free validation command."""
    command = [
        str(PYTHON_BIN),
        str(REPO_ROOT / "train.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--scenario",
        task.scenario,
        "--run-dir",
        str(run_dir),
        "--validate-only",
    ]
    command.extend(
        _common_overrides(
            args=args,
            task=task,
            suite_id=suite_id,
            job_type="validation" if checkpoint is None else "training",
        )
    )
    command.extend(_set_arg("training.trainer.early_stopping.enabled", False))
    command.extend(
        _set_arg(
            "model.resume_checkpoint",
            False if checkpoint is None else str(checkpoint.resolve()),
        )
    )
    command.extend(_set_arg("model.load_checkpoint_only", checkpoint is not None))
    return command


def _initial_manifest(output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Build the durable suite manifest before any GPU task starts."""
    suite_id = _suite_id(output_root)
    return {
        "schema_version": 1,
        "kind": "baseline_2016_training_suite",
        "suite_id": suite_id,
        "output_root": str(output_root.resolve()),
        "created_at": datetime.now().isoformat(),
        "config": str(Path(args.config).resolve()),
        "inference_config": str(Path(args.inference_config).resolve()),
        "candidate_profiles": str(Path(args.candidate_profiles).resolve()),
        "year": int(args.year),
        "iso_week": int(args.iso_week),
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "validation_examples": int(args.validation_examples),
        "checkpoint_every_n_train_steps": int(args.checkpoint_every_n_train_steps),
        "max_task_hours": int(args.max_task_hours),
        "gpu_indices": [int(value) for value in args.gpu_indices],
        "wandb": {
            "project": str(args.project),
            "entity": str(args.entity),
            "group": suite_id,
        },
        "tasks": {
            task.key: {
                "model": task.model.name,
                "label": task.model.label,
                "model_type": task.model.model_type,
                "scenario": task.scenario,
                "trainable": bool(task.model.trainable),
                "status": "pending",
                "wandb_run_id": _task_run_id(suite_id, task.key),
            }
            for task in _task_list()
        },
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist suite state so interrupted tasks can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_or_create_manifest(
    output_root: Path, args: argparse.Namespace, *, dry_run: bool
) -> dict[str, Any]:
    """Load a matching manifest or initialize a fresh suite."""
    manifest_path = output_root / MANIFEST_NAME
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("year", -1)) != int(args.year):
            raise RuntimeError("Existing suite manifest uses a different year.")
        return payload
    payload = _initial_manifest(output_root, args)
    if not dry_run:
        _write_manifest(manifest_path, payload)
    return payload


def _gpu_environment(gpu_index: int) -> dict[str, str]:
    """Return a subprocess environment exposing exactly one physical GPU."""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    environment["WANDB_MODE"] = "online"
    return environment


def _run_command(
    command: Sequence[str],
    *,
    gpu_index: int,
    log_path: Path,
    task_key: str,
) -> int:
    """Run one command while streaming prefixed output to console and disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_gpu_environment(gpu_index),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            prefixed = f"[gpu{gpu_index}:{task_key}] {line}"
            print(prefixed, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return int(process.wait())


def _best_checkpoint(run_dir: Path) -> Path:
    """Return the unique best checkpoint produced by one completed run."""
    checkpoints = sorted(run_dir.glob("best-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one best checkpoint in {run_dir}, found {checkpoints}."
        )
    return checkpoints[0]


def _preflight(args: argparse.Namespace) -> None:
    """Require two working GPUs, the dataset inputs, and online W&B authentication."""
    if not PYTHON_BIN.is_file():
        raise FileNotFoundError(f"Required Python environment is missing: {PYTHON_BIN}")
    for required_path in (args.config, args.inference_config, args.candidate_profiles):
        if not Path(required_path).is_file():
            raise FileNotFoundError(f"Required suite input is missing: {required_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU baseline training.")
    gpu_count = int(torch.cuda.device_count())
    unavailable = [
        index for index in args.gpu_indices if index < 0 or index >= gpu_count
    ]
    if unavailable:
        raise RuntimeError(
            f"Requested GPU indices {unavailable} are unavailable; device_count={gpu_count}."
        )
    try:
        import wandb

        api = wandb.Api(timeout=30)
        _ = api.viewer
    except Exception as exc:
        raise RuntimeError(
            f"Online W&B authentication preflight failed: {exc}"
        ) from exc


def _execute_task(
    *,
    args: argparse.Namespace,
    task: SuiteTask,
    gpu_index: int,
    output_root: Path,
    manifest: dict[str, Any],
    manifest_lock: threading.Lock,
) -> bool:
    """Execute, checkpoint, and best-validate one queued task."""
    suite_id = str(manifest["suite_id"])
    task_state = manifest["tasks"][task.key]
    run_dir = output_root / "runs" / task.model.name / task.scenario
    with manifest_lock:
        task_state.update(
            {
                "status": "running",
                "gpu_index": int(gpu_index),
                "run_dir": str(run_dir.resolve()),
                "started_at": datetime.now().isoformat(),
            }
        )
        _write_manifest(output_root / MANIFEST_NAME, manifest)

    try:
        if task.model.trainable:
            recorded_checkpoint = task_state.get("best_checkpoint")
            checkpoint = Path(str(recorded_checkpoint)) if recorded_checkpoint else None
            if checkpoint is None or not checkpoint.is_file():
                resume_checkpoint = None
                if run_dir.exists():
                    last_checkpoint = run_dir / "last.ckpt"
                    if args.resume_incomplete and last_checkpoint.is_file():
                        resume_checkpoint = last_checkpoint
                    elif args.resume_incomplete:
                        # A monitored validation checkpoint is also full trainer state.
                        best_checkpoints = sorted(run_dir.glob("best-epoch*.ckpt"))
                        if best_checkpoints:
                            resume_checkpoint = best_checkpoints[-1]
                        else:
                            raise RuntimeError(
                                f"No resumable checkpoint exists in {run_dir}."
                            )
                    else:
                        raise RuntimeError(
                            f"Incomplete run directory already exists: {run_dir}. "
                            "Use --resume-incomplete when last.ckpt is available."
                        )
                train_command = _training_command(
                    args=args,
                    task=task,
                    suite_id=suite_id,
                    run_dir=run_dir,
                    resume_checkpoint=resume_checkpoint,
                )
                return_code = _run_command(
                    train_command,
                    gpu_index=gpu_index,
                    log_path=run_dir / "training.log",
                    task_key=task.key,
                )
                if return_code != 0:
                    raise RuntimeError(f"Training exited with status {return_code}.")
                checkpoint = _best_checkpoint(run_dir)
                with manifest_lock:
                    task_state.update(
                        {
                            "status": "validating",
                            "best_checkpoint": str(checkpoint.resolve()),
                        }
                    )
                    _write_manifest(output_root / MANIFEST_NAME, manifest)
            validation_dir = run_dir / "final_validation"
            validation_command = _validation_command(
                args=args,
                task=task,
                suite_id=suite_id,
                run_dir=validation_dir,
                checkpoint=checkpoint,
            )
            return_code = _run_command(
                validation_command,
                gpu_index=gpu_index,
                log_path=validation_dir / "validation.log",
                task_key=f"{task.key}-best",
            )
            if return_code != 0:
                raise RuntimeError(
                    f"Best-checkpoint validation exited with status {return_code}."
                )
        else:
            validation_command = _validation_command(
                args=args,
                task=task,
                suite_id=suite_id,
                run_dir=run_dir,
                checkpoint=None,
            )
            return_code = _run_command(
                validation_command,
                gpu_index=gpu_index,
                log_path=run_dir / "validation.log",
                task_key=task.key,
            )
            if return_code != 0:
                raise RuntimeError(f"IDW validation exited with status {return_code}.")
        with manifest_lock:
            task_state.update(
                {"status": "complete", "completed_at": datetime.now().isoformat()}
            )
            _write_manifest(output_root / MANIFEST_NAME, manifest)
        return True
    except Exception as exc:
        with manifest_lock:
            task_state.update(
                {
                    "status": "failed",
                    "failed_at": datetime.now().isoformat(),
                    "error": str(exc),
                }
            )
            _write_manifest(output_root / MANIFEST_NAME, manifest)
        print(f"Task {task.key} failed on GPU {gpu_index}: {exc}", file=sys.stderr)
        return False


def _run_training_queue(
    *,
    args: argparse.Namespace,
    output_root: Path,
    manifest: dict[str, Any],
) -> None:
    """Run pending tasks on two fixed-GPU workers with fail-fast dequeueing."""
    manifest.update(
        {
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "validation_examples": int(args.validation_examples),
            "checkpoint_every_n_train_steps": int(args.checkpoint_every_n_train_steps),
            "max_task_hours": int(args.max_task_hours),
        }
    )
    skipped_models = set(args.skip_models)
    for task in _task_list():
        if task.model.name not in skipped_models:
            continue
        task_state = manifest["tasks"][task.key]
        if task_state.get("status") == "complete":
            continue
        task_state.update(
            {
                "status": "skipped",
                "skipped_at": datetime.now().isoformat(),
                "reason": "Explicitly excluded with --skip-models.",
            }
        )
    if skipped_models and not args.dry_run:
        _write_manifest(output_root / MANIFEST_NAME, manifest)
    tasks = [
        task
        for task in _task_list()
        if manifest["tasks"][task.key].get("status") not in {"complete", "skipped"}
    ]
    if args.dry_run:
        suite_id = str(manifest["suite_id"])
        for task_index, task in enumerate(tasks):
            gpu_index = int(args.gpu_indices[task_index % len(args.gpu_indices)])
            run_dir = output_root / "runs" / task.model.name / task.scenario
            command = (
                _training_command(
                    args=args,
                    task=task,
                    suite_id=suite_id,
                    run_dir=run_dir,
                    resume_checkpoint=None,
                )
                if task.model.trainable
                else _validation_command(
                    args=args,
                    task=task,
                    suite_id=suite_id,
                    run_dir=run_dir,
                    checkpoint=None,
                )
            )
            print(f"GPU {gpu_index}: {' '.join(command)}")
        return
    if not tasks:
        return

    pending: queue.Queue[SuiteTask] = queue.Queue()
    for task in tasks:
        pending.put(task)
    stop_dequeueing = threading.Event()
    manifest_lock = threading.Lock()

    def worker(gpu_index: int) -> None:
        while not stop_dequeueing.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            succeeded = _execute_task(
                args=args,
                task=task,
                gpu_index=gpu_index,
                output_root=output_root,
                manifest=manifest,
                manifest_lock=manifest_lock,
            )
            pending.task_done()
            if not succeeded:
                stop_dequeueing.set()
                return

    workers = [
        threading.Thread(target=worker, args=(int(gpu_index),), daemon=False)
        for gpu_index in args.gpu_indices
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    if stop_dequeueing.is_set():
        raise RuntimeError(
            "The baseline queue stopped after a task failure; inspect the manifest."
        )


def _write_models_config(output_root: Path, manifest: dict[str, Any]) -> Path:
    """Write paper-workflow method specs from completed suite checkpoints."""
    methods: dict[str, Any] = {"idw": {"label": "IDW", "model_type": "idw_baseline"}}
    for model in MODEL_SPECS:
        if not model.trainable:
            continue
        model_states = [
            manifest["tasks"][f"{model.name}_{scenario}"] for scenario in SCENARIOS
        ]
        if all(state.get("status") == "skipped" for state in model_states):
            continue
        checkpoints: dict[str, str] = {}
        for scenario in SCENARIOS:
            task_state = manifest["tasks"][f"{model.name}_{scenario}"]
            checkpoint = task_state.get("best_checkpoint")
            if task_state.get("status") != "complete" or not checkpoint:
                raise RuntimeError(
                    f"Cannot evaluate incomplete task {model.name}_{scenario}."
                )
            checkpoints[f"{scenario}_checkpoint"] = str(checkpoint)
        methods[model.name if model.name != "unet3d" else "unet"] = {
            "label": model.label,
            "model_type": model.model_type,
            **checkpoints,
        }
    path = output_root / "evaluation" / "baseline_models.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"methods": methods}, sort_keys=False), encoding="utf-8"
    )
    return path


def _evaluation_commands(
    *, args: argparse.Namespace, output_root: Path, models_config: Path
) -> tuple[list[str], list[str], Path, Path]:
    """Build spectral inference and scalar paper-metric export commands."""
    spectral_root = output_root / "evaluation" / "spectral_comparison"
    week_dir = spectral_root / "weeks" / f"{int(args.year)}_W{int(args.iso_week):02d}"
    metrics_dir = output_root / "evaluation" / "paper_metrics"
    spectral_command = [
        str(PYTHON_BIN),
        "-m",
        "depth_recon.inference.export_spectral_comparison_bundle",
        "--config",
        str(Path(args.inference_config).resolve()),
        "--models-config",
        str(models_config.resolve()),
        "--year",
        str(int(args.year)),
        "--iso-week",
        str(int(args.iso_week)),
        "--output-dir",
        str(spectral_root),
        "--device",
        "cuda",
        "--strict-load",
        "--batch-size",
        "8",
        "--inference-num-workers",
        "4",
        "--inference-prefetch-factor",
        "2",
        "--patch-stride",
        "128",
        "--min-ocean-fraction",
        "0.05",
        "--sampler",
        "ddim",
        "--ddim-steps",
        "100",
        "--seed",
        str(int(args.seed)),
        "--validation-year",
        str(int(args.year)),
        "--en4-holdout-fraction",
        "0.2",
        "--en4-candidate-profiles",
        str(Path(args.candidate_profiles).resolve()),
        "--prediction-method",
        "unet2d",
        "--min-wavelength-km",
        "30",
        "--max-wavelength-km",
        "1000",
        "--wavelength-bin-count",
        "32",
    ]
    metrics_command = [
        str(PYTHON_BIN),
        "-m",
        "depth_recon.inference.export_paper_metrics",
        "--paper-run-dir",
        str(week_dir),
        "--output-dir",
        str(metrics_dir),
        "--validation-year",
        str(int(args.year)),
        "--max-depth-m",
        "2000",
        "--seed",
        str(int(args.seed)),
    ]
    return spectral_command, metrics_command, spectral_root, metrics_dir


def _upload_evaluation_to_wandb(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    output_root: Path,
    models_config: Path,
    metrics_dir: Path,
) -> None:
    """Log final metric tables, preview images, and evaluation artifacts to W&B."""
    import wandb

    suite_id = str(manifest["suite_id"])
    run_id = _task_run_id(suite_id, "paper-evaluation")
    run = wandb.init(
        project=str(args.project),
        entity=str(args.entity),
        name=f"{suite_id}-paper-evaluation",
        id=run_id,
        resume="allow",
        group=suite_id,
        job_type="evaluation",
        tags=["baseline", "validation-2016", "paper-metrics", "spectral"],
        config={
            "year": int(args.year),
            "iso_week": int(args.iso_week),
            "seed": int(args.seed),
            "depth_averaging": "equal_depth_mean",
            "max_depth_m": 2000.0,
        },
    )
    try:
        for table_name, filename in (
            ("paper_metrics_summary", "paper_metrics_summary.csv"),
            ("paper_metrics_by_depth", "paper_metrics_by_depth.csv"),
            ("en4_holdout_metrics", "en4_holdout_metrics.csv"),
            ("glorys_field_metrics", "glorys_field_metrics.csv"),
        ):
            table_path = metrics_dir / filename
            if table_path.is_file():
                run.log({table_name: wandb.Table(dataframe=pd.read_csv(table_path))})

        image_paths = sorted(
            path
            for path in (output_root / "evaluation").rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )[: max(0, int(args.max_wandb_images))]
        for offset in range(0, len(image_paths), 16):
            payload = {
                f"evaluation_images/{path.relative_to(output_root / 'evaluation')}": (
                    wandb.Image(str(path))
                )
                for path in image_paths[offset : offset + 16]
            }
            if payload:
                run.log(payload)

        artifact = wandb.Artifact(
            name=f"{suite_id}-evaluation",
            type="baseline-evaluation",
            metadata={"year": int(args.year), "iso_week": int(args.iso_week)},
        )
        artifact.add_file(str(models_config), name=models_config.name)
        artifact.add_file(str(output_root / MANIFEST_NAME), name=MANIFEST_NAME)
        artifact.add_dir(str(output_root / "evaluation"), name="evaluation")
        run.log_artifact(artifact)
    finally:
        run.finish()


def _run_evaluation(
    *, args: argparse.Namespace, output_root: Path, manifest: dict[str, Any]
) -> None:
    """Run paper-week inference, scalar metrics, spectra, and W&B upload."""
    # A dry run must remain useful before training has produced any checkpoints.
    models_config = (
        output_root / "evaluation" / "baseline_models.yaml"
        if args.dry_run
        else _write_models_config(output_root, manifest)
    )
    spectral_command, metrics_command, _, metrics_dir = _evaluation_commands(
        args=args,
        output_root=output_root,
        models_config=models_config,
    )
    if args.dry_run:
        print("Evaluation: " + " ".join(spectral_command))
        print("Metrics: " + " ".join(metrics_command))
        return
    evaluation_log_dir = output_root / "evaluation" / "logs"
    spectral_status = _run_command(
        spectral_command,
        gpu_index=int(args.gpu_indices[0]),
        log_path=evaluation_log_dir / "spectral.log",
        task_key="spectral",
    )
    if spectral_status != 0:
        raise RuntimeError(f"Spectral evaluation exited with status {spectral_status}.")
    metrics_status = _run_command(
        metrics_command,
        gpu_index=int(args.gpu_indices[0]),
        log_path=evaluation_log_dir / "paper_metrics.log",
        task_key="paper-metrics",
    )
    if metrics_status != 0:
        raise RuntimeError(f"Paper metrics exited with status {metrics_status}.")
    _upload_evaluation_to_wandb(
        args=args,
        manifest=manifest,
        output_root=output_root,
        models_config=models_config,
        metrics_dir=metrics_dir,
    )
    manifest["evaluation"] = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(),
        "models_config": str(models_config.resolve()),
        "metrics_dir": str(metrics_dir.resolve()),
    }
    _write_manifest(output_root / MANIFEST_NAME, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected baseline-suite phases."""
    args = _build_parser().parse_args(argv)
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root is not None
        else _default_output_root()
    )
    output_root = output_root.resolve()
    manifest = _load_or_create_manifest(output_root, args, dry_run=bool(args.dry_run))
    if not args.dry_run:
        _preflight(args)
    if args.phase in {"train", "all"}:
        _run_training_queue(args=args, output_root=output_root, manifest=manifest)
    if args.phase in {"evaluate", "all"}:
        _run_evaluation(args=args, output_root=output_root, manifest=manifest)
    print(f"Baseline suite root: {output_root}", flush=True)


if __name__ == "__main__":
    main()
