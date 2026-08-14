from __future__ import annotations

import unittest

from depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior import (
    _centered_window_avoids_excluded_years,
    _manifest_surface_window_days,
)


class TestVerticalOffsetFitterLeakageGuard(unittest.TestCase):
    def test_centered_window_rejects_adjacent_year_overlap(self) -> None:
        """Surface composites touching the held-out year must not enter fitting."""
        self.assertFalse(
            _centered_window_avoids_excluded_years(
                20171230, window_days=7, excluded_years=(2018,)
            )
        )
        self.assertTrue(
            _centered_window_avoids_excluded_years(
                20171220, window_days=7, excluded_years=(2018,)
            )
        )

    def test_manifest_window_is_required(self) -> None:
        """The fitter must fail closed when temporal aggregation is unknown."""
        with self.assertRaises(RuntimeError):
            _manifest_surface_window_days({})
        self.assertEqual(
            _manifest_surface_window_days(
                {"surface_temporal_aggregation": {"window_days": 7}}
            ),
            7,
        )


if __name__ == "__main__":
    unittest.main()
