"""Check replay evidence gates and physical image stack ordering."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from val.export_spatial import (
    ChunkedLSTM,
    support_locations,
    verify_predictions,
    write_stack,
)


class SpatialExportTests(unittest.TestCase):
    """Prevent unverified fields or incorrect depth ordering from being published."""

    def test_support_markers_preserve_masked_cells_and_depth_availability(self):
        """Masked targets stay absent; retained cells carry only valid input levels."""
        mask = torch.zeros((3, 2, 4), dtype=torch.bool)
        mask[0, 0, 1] = True
        mask[2, 0, 1] = True
        mask[1, 1, 3] = True
        self.assertEqual(
            support_locations(mask),
            [
                {"row": 0, "col": 1, "levels": [0, 2]},
                {"row": 1, "col": 3, "levels": [1]},
            ],
        )
        self.assertEqual(support_locations(torch.zeros_like(mask)), [])
        with self.assertRaisesRegex(ValueError, "depth/row/column"):
            support_locations(mask[0])

    def test_lstm_chunking_preserves_depth_sequences_and_states(self):
        """Splitting independent pixels preserves outputs and both recurrent states."""
        torch.manual_seed(17)
        recurrent = torch.nn.LSTM(3, 5, batch_first=True, bidirectional=True).eval()
        values = torch.randn(7, 11, 3)
        with torch.inference_mode():
            expected, states = recurrent(values)
            actual, actual_states = ChunkedLSTM(recurrent, chunk_size=2)(values)
        torch.testing.assert_close(actual, expected)
        for actual_state, state in zip(actual_states, states):
            torch.testing.assert_close(actual_state, state)

    def test_point_replay_rejects_changed_prediction(self):
        """A spatial output must reproduce the original native-depth observation operator."""
        assignments = pd.DataFrame(
            {
                "point_id": ["p"],
                "batch_index": [0],
                "row": [0],
                "col": [0],
                "depth_index_lower": [0],
                "depth_index_upper": [1],
                "depth_weight": [0.25],
            }
        )
        field = torch.tensor([[[[2.0]], [[6.0]]]])
        saved = pd.DataFrame({"point_id": ["p"], "prediction_mean": [3.0]})
        self.assertEqual(
            verify_predictions(field, assignments, saved, "residual")["status"],
            "passed",
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            verify_predictions(field + 1, assignments, saved, "residual")

    def test_stack_preserves_depth_order_and_missing_support(self):
        """Different depth planes remain distinct and missing ocean cells stay transparent."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.png"
            field = np.array([[[0.0, 1.0]], [[2.0, np.nan]]])
            valid = np.array([[[True, False]], [[True, True]]])
            write_stack(path, field, valid, [0, 2])
            pixels = np.array(Image.open(path))
            self.assertEqual(pixels.shape, (2, 2, 4))
            self.assertFalse(np.array_equal(pixels[0, 0, :3], pixels[1, 0, :3]))
            np.testing.assert_array_equal(pixels[:, :, 3], [[255, 0], [255, 0]])
