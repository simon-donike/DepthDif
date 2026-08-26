from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import yaml

from depth_recon.scripts.run_baseline_2016_suite import (
    MODEL_SPECS,
    _initial_manifest,
    _logical_epoch_batches,
    _run_evaluation,
    _run_training_queue,
    _suite_id,
    _task_list,
    _training_command,
    _validation_command,
    _write_models_config,
)


def _args(output_root: Path) -> argparse.Namespace:
    """Return compact suite CLI arguments for unit tests."""
    return argparse.Namespace(
        phase="all",
        config=Path("src/depth_recon/configs/px_space/training_super_config.yaml"),
        inference_config=Path(
            "src/depth_recon/configs/px_space/inference_super_config.yaml"
        ),
        output_root=output_root,
        gpu_indices=(0, 1),
        project="DepthDif_Simon",
        entity="esa-phi-lab",
        candidate_profiles=Path(
            "instructions/en4_no_spatiotemporal_candidate_profiles.parquet"
        ),
        year=2016,
        iso_week=25,
        seed=7,
        max_epochs=8,
        patience=2,
        validation_examples=100_000,
        checkpoint_every_n_train_steps=5_000,
        max_task_hours=6,
        skip_models=(),
        resume_incomplete=False,
        dry_run=False,
        max_wandb_images=10,
    )


class TestBaseline2016Suite(unittest.TestCase):
    def test_task_order_is_longest_first_and_covers_both_variables(self) -> None:
        tasks = _task_list()

        self.assertEqual(len(tasks), 10)
        self.assertEqual(
            [task.key for task in tasks[:4]],
            [
                "unet3d_temperature",
                "unet3d_salinity",
                "unet2d_temperature",
                "unet2d_salinity",
            ],
        )
        self.assertEqual(
            [task.key for task in tasks[-2:]], ["idw_temperature", "idw_salinity"]
        )

    def test_training_and_validation_commands_lock_fair_suite_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            args = _args(output_root)
            task = _task_list()[0]
            suite_id = _suite_id(output_root)
            train_command = _training_command(
                args=args,
                task=task,
                suite_id=suite_id,
                run_dir=output_root / "run",
                resume_checkpoint=None,
            )
            checkpoint = output_root / "best.ckpt"
            validation_command = _validation_command(
                args=args,
                task=task,
                suite_id=suite_id,
                run_dir=output_root / "validation",
                checkpoint=checkpoint,
            )

        train_text = " ".join(train_command)
        validation_text = " ".join(validation_command)
        self.assertIn("data.split.val_year=2016", train_text)
        self.assertIn("data.dataset.finetune_sampling.enabled=false", train_text)
        self.assertIn("training.trainer.devices=1", train_text)
        self.assertIn("training.trainer.early_stopping.patience=2", train_text)
        self.assertIn("training.trainer.limit_train_batches=12500", train_text)
        self.assertIn("training.trainer.val_check_interval=1.0", train_text)
        self.assertIn(
            "training.trainer.checkpoint_every_n_train_steps=5000", train_text
        )
        self.assertIn('training.trainer.max_time="00:06:00:00"', train_text)
        self.assertIn("model.resume_checkpoint=false", train_text)
        self.assertIn("--validate-only", validation_command)
        self.assertIn("model.load_checkpoint_only=true", validation_text)
        self.assertIn('training.wandb.resume="allow"', validation_text)

    def test_logical_epochs_expose_comparable_example_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            batches = {
                task.model.name: _logical_epoch_batches(args, task)
                for task in _task_list()[::2]
            }

        self.assertEqual(batches["unet3d"], 12_500)
        self.assertEqual(batches["unet2d"], 2_084)
        self.assertEqual(batches["cnn"], 11_112)
        self.assertEqual(batches["lstm"], 100_000)

    def test_two_gpu_queue_stops_dequeueing_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            args = _args(output_root)
            manifest = _initial_manifest(output_root, args)
            failure_started = threading.Event()
            both_started = threading.Barrier(2)
            calls: list[tuple[str, int]] = []
            calls_lock = threading.Lock()

            def execute_task(**kwargs: object) -> bool:
                task = kwargs["task"]
                gpu_index = int(kwargs["gpu_index"])
                with calls_lock:
                    calls.append((task.key, gpu_index))
                both_started.wait(timeout=2.0)
                if gpu_index == 0:
                    failure_started.set()
                    return False
                failure_started.wait(timeout=2.0)
                return True

            with patch(
                "depth_recon.scripts.run_baseline_2016_suite._execute_task",
                side_effect=execute_task,
            ):
                with self.assertRaisesRegex(RuntimeError, "task failure"):
                    _run_training_queue(
                        args=args,
                        output_root=output_root,
                        manifest=manifest,
                    )

        self.assertEqual(len(calls), 2)
        self.assertEqual({gpu_index for _, gpu_index in calls}, {0, 1})

    def test_models_config_uses_completed_temperature_and_salinity_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            args = _args(output_root)
            manifest = _initial_manifest(output_root, args)
            for model in MODEL_SPECS:
                for scenario in ("temperature", "salinity"):
                    state = manifest["tasks"][f"{model.name}_{scenario}"]
                    state["status"] = "complete"
                    if model.trainable:
                        checkpoint = output_root / f"{model.name}_{scenario}.ckpt"
                        checkpoint.touch()
                        state["best_checkpoint"] = str(checkpoint)

            config_path = _write_models_config(output_root, manifest)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            list(config["methods"]), ["idw", "unet", "unet2d", "cnn", "lstm"]
        )
        self.assertEqual(config["methods"]["idw"]["model_type"], "idw_baseline")
        self.assertTrue(
            config["methods"]["unet2d"]["temperature_checkpoint"].endswith(
                "unet2d_temperature.ckpt"
            )
        )

    def test_skipped_architecture_is_omitted_from_models_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            args = _args(output_root)
            manifest = _initial_manifest(output_root, args)
            for model in MODEL_SPECS:
                for scenario in ("temperature", "salinity"):
                    state = manifest["tasks"][f"{model.name}_{scenario}"]
                    if model.name == "unet3d":
                        state["status"] = "skipped"
                    else:
                        state["status"] = "complete"
                        if model.trainable:
                            checkpoint = output_root / f"{model.name}_{scenario}.ckpt"
                            checkpoint.touch()
                            state["best_checkpoint"] = str(checkpoint)

            config_path = _write_models_config(output_root, manifest)
            methods = yaml.safe_load(config_path.read_text(encoding="utf-8"))["methods"]

        self.assertEqual(list(methods), ["idw", "unet2d", "cnn", "lstm"])

    def test_evaluation_dry_run_does_not_require_completed_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            args = _args(output_root)
            args.dry_run = True
            manifest = _initial_manifest(output_root, args)

            with patch("builtins.print") as print_mock:
                _run_evaluation(
                    args=args,
                    output_root=output_root,
                    manifest=manifest,
                )

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("Evaluation:", output)
        self.assertIn("--en4-candidate-profiles", output)
        self.assertIn("Metrics:", output)


if __name__ == "__main__":
    unittest.main()
