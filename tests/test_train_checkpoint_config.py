from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from train import resolve_load_checkpoint_only, resolve_resume_ckpt_path


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


if __name__ == "__main__":
    unittest.main()
