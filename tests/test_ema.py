from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from depth_recon.inference.core import extract_ema_state_dict, load_checkpoint_weights
from depth_recon.models.diffusion import EMA
from depth_recon.models.diffusion.PixelDiffusion import PixelDiffusionConditional
from train import build_ema_callback


class _TinyLightningModule(pl.LightningModule):
    """Minimal LightningModule exposing parameters and buffers for EMA tests."""

    def __init__(self) -> None:
        """Initialize a tiny module with one weight and one integer buffer."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("counter", torch.tensor([1], dtype=torch.long))


class _TinyFitModule(pl.LightningModule):
    """Minimal module for end-to-end Lightning EMA callback tests."""

    def __init__(self) -> None:
        """Initialize one learnable scalar and validation capture state."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.validation_weights: list[torch.Tensor] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Return a deterministic loss that moves weight from 1.0 to 5.0."""
        return (self.weight - 3.0).square().sum()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Record the weight value visible to validation."""
        self.validation_weights.append(self.weight.detach().clone().cpu())
        return self.weight.detach().abs().sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use SGD so the expected one-step update is exact."""
        return torch.optim.SGD(self.parameters(), lr=1.0)

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return one training batch."""
        return DataLoader([torch.zeros(1)], batch_size=1)

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        """Return one validation batch."""
        return DataLoader([torch.zeros(1)], batch_size=1)


class _FakeEMAForValidationLog:
    """Tiny EMA stand-in that records raw/EMA swap state."""

    def __init__(self, *, evaluate_ema_weights_instead: bool, applied: bool) -> None:
        """Initialize fake EMA state for validation logging tests."""
        self.ema_initialized = True
        self.evaluate_ema_weights_instead = bool(evaluate_ema_weights_instead)
        self.weights_are_applied = bool(applied)
        self.events: list[str] = []

    def replace_model_weights(self, pl_module: object) -> None:
        """Record that EMA weights were applied."""
        self.events.append("replace")
        if self.weights_are_applied:
            raise AssertionError("EMA weights already applied")
        self.weights_are_applied = True

    def restore_original_weights(self, pl_module: object) -> None:
        """Record that raw weights were restored."""
        self.events.append("restore")
        if not self.weights_are_applied:
            raise AssertionError("EMA weights were not applied")
        self.weights_are_applied = False


class _FakeValidationLoggerModel:
    """Small object that runs PixelDiffusion's EMA validation wrapper."""

    _run_single_image_full_reconstruction_and_log = (
        PixelDiffusionConditional._run_single_image_full_reconstruction_and_log
    )

    def __init__(self, ema_callback: _FakeEMAForValidationLog) -> None:
        """Initialize fake validation model with a cached example marker."""
        self._cached_val_example = object()
        self.ema_callback = ema_callback
        self.calls: list[tuple[bool, dict[str, object]]] = []

    def _get_ema_callback(self) -> _FakeEMAForValidationLog:
        """Return the fake EMA callback."""
        return self.ema_callback

    def _run_single_image_full_reconstruction_for_current_weights(
        self, **kwargs: object
    ) -> None:
        """Record logging kwargs and whether EMA weights were active."""
        self.calls.append((self.ema_callback.weights_are_applied, dict(kwargs)))

    def _log_full_reconstruction_placeholders(self) -> None:
        """Placeholder method required by the wrapped PixelDiffusion method."""
        self.calls.append(
            (self.ema_callback.weights_are_applied, {"placeholder": True})
        )


class TestEMA(unittest.TestCase):
    def test_build_ema_callback_from_model_config(self) -> None:
        self.assertIsNone(build_ema_callback({"model": {}}))
        self.assertIsNone(build_ema_callback({"model": {"ema": {"enabled": False}}}))

        callback = build_ema_callback(
            {
                "model": {
                    "ema": {
                        "enabled": True,
                        "decay": 0.95,
                        "apply_every_n_steps": 2,
                        "start_step": 3,
                        "save_ema_weights_in_callback_state": True,
                        "evaluate_ema_weights_instead": False,
                    }
                }
            }
        )

        self.assertIsInstance(callback, EMA)
        self.assertEqual(callback.decay, 0.95)
        self.assertEqual(callback.apply_ema_every_n_steps, 2)
        self.assertEqual(callback.start_step, 3)
        self.assertTrue(callback.save_ema_weights_in_callback_state)
        self.assertFalse(callback.evaluate_ema_weights_instead)

    def test_ema_updates_and_restores_model_state(self) -> None:
        module = _TinyLightningModule()
        callback = EMA(
            decay=0.5,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )
        callback.on_fit_start(None, module)

        with torch.no_grad():
            module.weight.fill_(3.0)
            module.counter.fill_(2)
        callback.ema(module)

        ema_state = callback.state_dict()["ema_weights"]
        self.assertTrue(torch.equal(ema_state["counter"], torch.tensor([2])))
        self.assertTrue(torch.allclose(ema_state["weight"], torch.tensor([2.0])))

        callback.replace_model_weights(module)
        self.assertTrue(torch.allclose(module.weight, torch.tensor([2.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([2])))

        callback.restore_original_weights(module)
        self.assertTrue(torch.allclose(module.weight, torch.tensor([3.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([2])))

    def test_ema_logs_raw_vs_shadow_weight_metrics(self) -> None:
        module = _TinyLightningModule()
        callback = EMA(decay=0.5)
        callback.on_fit_start(None, module)

        with torch.no_grad():
            module.weight.fill_(3.0)
        callback.ema(module)

        logged: dict[str, torch.Tensor] = {}

        def fake_log(name: str, value: torch.Tensor, **kwargs: object) -> None:
            logged[name] = value.detach().cpu()

        with patch.object(module, "log", fake_log):
            callback.log_weight_delta_metrics(None, module)

        self.assertIn("ema/weight_mean_abs_delta", logged)
        self.assertIn("ema/weight_rms_delta", logged)
        self.assertIn("ema/weight_relative_rms_delta", logged)
        self.assertIn("ema/weight_max_abs_delta", logged)
        self.assertIn("ema/tracked_floating_tensors", logged)
        self.assertTrue(
            torch.allclose(logged["ema/weight_mean_abs_delta"], torch.tensor(1.0))
        )
        self.assertTrue(
            torch.allclose(logged["ema/tracked_floating_tensors"], torch.tensor(1.0))
        )

    def test_ema_update_math_uses_previous_shadow_weights(self) -> None:
        module = _TinyLightningModule()
        callback = EMA(decay=0.25, save_ema_weights_in_callback_state=True)
        callback.on_fit_start(None, module)

        with torch.no_grad():
            module.weight.fill_(5.0)
            module.counter.fill_(2)
        callback.ema(module)
        self.assertTrue(
            torch.allclose(
                callback.state_dict()["ema_weights"]["weight"], torch.tensor([4.0])
            )
        )
        self.assertTrue(
            torch.equal(
                callback.state_dict()["ema_weights"]["counter"], torch.tensor([2])
            )
        )

        with torch.no_grad():
            module.weight.fill_(9.0)
            module.counter.fill_(3)
        callback.ema(module)

        # 0.25 * previous EMA 4.0 + 0.75 * current weight 9.0 = 7.75.
        self.assertTrue(
            torch.allclose(
                callback.state_dict()["ema_weights"]["weight"], torch.tensor([7.75])
            )
        )
        self.assertTrue(
            torch.equal(
                callback.state_dict()["ema_weights"]["counter"], torch.tensor([3])
            )
        )

    def test_ema_cadence_and_start_step_gate_updates(self) -> None:
        callback = EMA(decay=0.5, apply_ema_every_n_steps=3, start_step=2)

        self.assertFalse(callback.should_apply_ema(0))
        self.assertFalse(callback.should_apply_ema(1))
        self.assertFalse(callback.should_apply_ema(2))
        self.assertTrue(callback.should_apply_ema(3))
        callback._cur_step = 3
        self.assertFalse(callback.should_apply_ema(3))
        self.assertFalse(callback.should_apply_ema(4))
        self.assertTrue(callback.should_apply_ema(6))

    def test_ema_state_dict_resume_preserves_loaded_shadow_weights(self) -> None:
        module = _TinyLightningModule()
        callback = EMA(decay=0.5, save_ema_weights_in_callback_state=True)
        callback.on_fit_start(None, module)
        with torch.no_grad():
            module.weight.fill_(3.0)
        callback.ema(module)
        saved_state = callback.state_dict()

        resumed_module = _TinyLightningModule()
        with torch.no_grad():
            resumed_module.weight.fill_(99.0)
        resumed = EMA(decay=0.5, save_ema_weights_in_callback_state=True)
        resumed.load_state_dict(saved_state)
        resumed.on_fit_start(None, resumed_module)

        self.assertTrue(
            torch.allclose(
                resumed.state_dict()["ema_weights"]["weight"], torch.tensor([2.0])
            )
        )

    def test_checkpoint_loader_prefers_callback_ema_weights(self) -> None:
        module = _TinyLightningModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "manual_ema.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "weight": torch.tensor([5.0]),
                        "counter": torch.tensor([2], dtype=torch.long),
                    },
                    "callbacks": {
                        "depth_recon.models.diffusion.EMA": {
                            "ema_weights": {
                                "weight": torch.tensor([3.0]),
                                "counter": torch.tensor([4], dtype=torch.long),
                            }
                        }
                    },
                },
                checkpoint_path,
            )

            ema_state = extract_ema_state_dict(
                torch.load(str(checkpoint_path), map_location="cpu")
            )
            weight_source = load_checkpoint_weights(
                module,
                checkpoint_path,
                strict=True,
            )

        self.assertIsNotNone(ema_state)
        self.assertTrue(torch.allclose(ema_state["weight"], torch.tensor([3.0])))
        self.assertEqual(weight_source, "ema")
        self.assertTrue(torch.allclose(module.weight.detach(), torch.tensor([3.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([4])))

    def test_checkpoint_loader_falls_back_to_standard_weights_without_ema(
        self,
    ) -> None:
        module = _TinyLightningModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "manual_standard.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "weight": torch.tensor([5.0]),
                        "counter": torch.tensor([7], dtype=torch.long),
                    }
                },
                checkpoint_path,
            )

            weight_source = load_checkpoint_weights(
                module,
                checkpoint_path,
                strict=True,
            )

        self.assertEqual(weight_source, "standard")
        self.assertTrue(torch.allclose(module.weight.detach(), torch.tensor([5.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([7])))

    def test_lightning_checkpoint_saves_and_inference_loads_ema_weights(
        self,
    ) -> None:
        module = _TinyFitModule()
        callback = EMA(
            decay=0.5,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                callbacks=[callback],
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                default_root_dir=tmpdir,
            )
            trainer.fit(module)

            checkpoint_path = Path(tmpdir) / "ema_round_trip.ckpt"
            trainer.save_checkpoint(str(checkpoint_path))
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            ema_state = extract_ema_state_dict(checkpoint)

            loaded = _TinyFitModule()
            weight_source = load_checkpoint_weights(
                loaded,
                checkpoint_path,
                strict=True,
            )

        self.assertIsNotNone(ema_state)
        self.assertTrue(
            torch.allclose(checkpoint["state_dict"]["weight"], torch.tensor([5.0]))
        )
        self.assertTrue(torch.allclose(ema_state["weight"], torch.tensor([3.0])))
        self.assertEqual(weight_source, "ema")
        self.assertTrue(torch.allclose(loaded.weight.detach(), torch.tensor([3.0])))

    def test_lightning_fit_evaluates_with_ema_and_restores_raw_weights(self) -> None:
        module = _TinyFitModule()
        callback = EMA(
            decay=0.5,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            callbacks=[callback],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        trainer.fit(module)

        # Raw SGD update: 1.0 - 1.0 * grad((w - 3)^2) = 5.0.
        self.assertTrue(torch.allclose(module.weight.detach(), torch.tensor([5.0])))
        self.assertFalse(callback.weights_are_applied)

        # EMA update after the optimizer step: 0.5 * 1.0 + 0.5 * 5.0 = 3.0.
        ema_weight = callback.state_dict()["ema_weights"]["weight"]
        self.assertTrue(torch.allclose(ema_weight, torch.tensor([3.0])))

        # Validation ran with EMA weights, not the raw post-step weight.
        self.assertEqual(len(module.validation_weights), 1)
        self.assertTrue(
            torch.allclose(module.validation_weights[0], torch.tensor([3.0]))
        )

    def test_validation_logging_restores_raw_and_reapplies_ema_when_eval_uses_ema(
        self,
    ) -> None:
        ema_callback = _FakeEMAForValidationLog(
            evaluate_ema_weights_instead=True,
            applied=True,
        )
        model = _FakeValidationLoggerModel(ema_callback)

        model._run_single_image_full_reconstruction_and_log()

        self.assertEqual(ema_callback.events, ["restore", "replace"])
        self.assertTrue(ema_callback.weights_are_applied)
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(model.calls[0][0])
        self.assertTrue(model.calls[1][0])
        self.assertEqual(model.calls[0][1]["metric_prefix"], "val_standard")
        self.assertEqual(model.calls[0][1]["image_key_suffix"], "standard")
        self.assertFalse(model.calls[0][1]["log_default_metrics"])
        self.assertFalse(model.calls[0][1]["log_common_metrics"])
        self.assertEqual(model.calls[1][1]["metric_prefix"], "val_ema")
        self.assertEqual(model.calls[1][1]["image_key_suffix"], "ema")
        self.assertTrue(model.calls[1][1]["log_default_metrics"])
        self.assertTrue(model.calls[1][1]["log_common_metrics"])

    def test_validation_logging_applies_and_restores_ema_when_eval_uses_standard(
        self,
    ) -> None:
        ema_callback = _FakeEMAForValidationLog(
            evaluate_ema_weights_instead=False,
            applied=False,
        )
        model = _FakeValidationLoggerModel(ema_callback)

        model._run_single_image_full_reconstruction_and_log()

        self.assertEqual(ema_callback.events, ["replace", "restore"])
        self.assertFalse(ema_callback.weights_are_applied)
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(model.calls[0][0])
        self.assertTrue(model.calls[1][0])
        self.assertEqual(model.calls[0][1]["metric_prefix"], "val_standard")
        self.assertTrue(model.calls[0][1]["log_default_metrics"])
        self.assertTrue(model.calls[0][1]["log_common_metrics"])
        self.assertEqual(model.calls[1][1]["metric_prefix"], "val_ema")
        self.assertFalse(model.calls[1][1]["log_default_metrics"])
        self.assertFalse(model.calls[1][1]["log_common_metrics"])


if __name__ == "__main__":
    unittest.main()
