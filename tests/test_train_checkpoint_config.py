from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from train import (
    load_weights_only_checkpoint,
    resolve_load_checkpoint_only,
    resolve_resume_ckpt_path,
)


class _TinyCheckpointModule(torch.nn.Module):
    """Tiny module with one parameter and one buffer for checkpoint loading tests."""

    def __init__(self) -> None:
        """Initialize deterministic raw module state."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("counter", torch.tensor([1], dtype=torch.long))


class TestTrainCheckpointConfig(unittest.TestCase):
    def test_resume_checkpoint_false_starts_from_scratch(self) -> None:
        model_cfg = {"model": {"resume_checkpoint": False}}

        self.assertIsNone(resolve_resume_ckpt_path(model_cfg))

    def test_resume_checkpoint_path_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "last.ckpt"
            ckpt_path.touch()
            model_cfg = {"model": {"resume_checkpoint": str(ckpt_path)}}

            self.assertEqual(resolve_resume_ckpt_path(model_cfg), str(ckpt_path))

    def test_load_checkpoint_only_is_boolean_mode(self) -> None:
        self.assertTrue(
            resolve_load_checkpoint_only({"model": {"load_checkpoint_only": True}})
        )
        self.assertFalse(
            resolve_load_checkpoint_only({"model": {"load_checkpoint_only": False}})
        )

    def test_load_checkpoint_only_rejects_checkpoint_paths(self) -> None:
        model_cfg = {"model": {"load_checkpoint_only": "weights.ckpt"}}

        with self.assertRaisesRegex(
            ValueError, "model.load_checkpoint_only must be true or false"
        ):
            resolve_load_checkpoint_only(model_cfg)

    def test_weights_only_checkpoint_loads_standard_state_dict(self) -> None:
        module = _TinyCheckpointModule()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ema.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "weight": torch.tensor([5.0]),
                        "counter": torch.tensor([2], dtype=torch.long),
                    },
                    "callbacks": {
                        "EMA": {
                            "ema_weights": {
                                "weight": torch.tensor([3.0]),
                                "counter": torch.tensor([4], dtype=torch.long),
                            }
                        }
                    },
                },
                checkpoint_path,
            )

            weight_source = load_weights_only_checkpoint(module, str(checkpoint_path))

        self.assertEqual(weight_source, "standard")
        self.assertTrue(torch.allclose(module.weight.detach(), torch.tensor([5.0])))
        self.assertTrue(torch.equal(module.counter, torch.tensor([2])))


if __name__ == "__main__":
    unittest.main()
