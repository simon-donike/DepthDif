"""Protect the outcome-independent selection used for the short EN4 campaign."""

import unittest

import pandas as pd

from val.quick_en4 import choose_pairs


class QuickEN4SelectionTests(unittest.TestCase):
    """Check coverage and frozen batch selection before any GPU inference."""

    def test_selection_ignores_observations_and_prediction_values(self):
        """Changing measured or predicted temperatures cannot change the subset."""
        selection = {"patches": [{"index": i} for i in range(48)]}
        points = pd.DataFrame(
            {
                "patch_index": range(48),
                "region": [str(i % 6) for i in range(48)],
                "target_time": [f"2016-{i // 2 % 12 + 1:02d}-15" for i in range(48)],
                "observation": range(48),
                "prediction": range(48),
            }
        )
        chosen = choose_pairs(selection, points, 24, 7)
        altered = points.sample(frac=1, random_state=123).assign(
            observation=-999, prediction=999
        )
        self.assertEqual(chosen, choose_pairs(selection, altered, 24, 7))
        self.assertEqual(len(set(chosen)), 12)
        indices = [i for batch in chosen for i in (2 * batch, 2 * batch + 1)]
        months = pd.to_datetime(
            points[points.patch_index.isin(indices)].target_time
        ).dt.month
        self.assertEqual(set(months), set(range(1, 13)))

    def test_partial_original_batches_are_rejected(self):
        """Never change the residual noise stream by cutting an original pair."""
        for count in (0, 3, 6):
            with self.subTest(count=count), self.assertRaises(ValueError):
                choose_pairs(
                    {"patches": [{"index": i} for i in range(4)]},
                    pd.DataFrame(),
                    count,
                    7,
                )


if __name__ == "__main__":
    unittest.main()
