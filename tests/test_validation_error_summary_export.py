from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from inference.export_validation_error_summary import (
    build_validation_error_summary_dataframe,
    create_validation_error_summary_accumulator,
    filter_validation_summary_dataset_by_iso_week,
    update_validation_error_summary_accumulator,
)
from utils.validation_denoise import (
    save_average_glorys_profile_and_error_plot,
    save_average_glorys_profile_error_plot,
)


class TestValidationErrorSummaryExport(unittest.TestCase):
    def test_accumulator_uses_expected_masks_for_glorys_and_argo(self) -> None:
        accumulator = create_validation_error_summary_accumulator(depth_size=2)

        x_denorm = torch.tensor(
            [[[[1.0], [100.0]], [[10.0], [200.0]]]],
            dtype=torch.float32,
        )
        y_denorm = torch.tensor(
            [[[[2.0], [3.0]], [[20.0], [30.0]]]],
            dtype=torch.float32,
        )
        y_hat_denorm = torch.tensor(
            [[[[2.5], [4.0]], [[float("nan")], [33.0]]]],
            dtype=torch.float32,
        )
        x_valid_mask = torch.tensor(
            [[[[True], [False]], [[False], [True]]]],
            dtype=torch.bool,
        )
        y_valid_mask = torch.tensor(
            [[[[True], [True]], [[True], [False]]]],
            dtype=torch.bool,
        )

        update_validation_error_summary_accumulator(
            accumulator,
            x_denorm=x_denorm,
            y_denorm=y_denorm,
            y_hat_denorm=y_hat_denorm,
            x_valid_mask=x_valid_mask,
            y_valid_mask=y_valid_mask,
        )

        self.assertEqual(
            [
                len(bucket)
                for bucket in accumulator.abs_error_prediction_vs_glorys_values
            ],
            [1, 0],
        )
        self.assertEqual(
            [
                values.size
                for values in accumulator.abs_error_prediction_vs_glorys_values[0]
            ],
            [2],
        )
        self.assertEqual(
            [len(bucket) for bucket in accumulator.abs_error_prediction_vs_argo_values],
            [1, 1],
        )

        df = build_validation_error_summary_dataframe(
            accumulator,
            depth_axis_m=np.asarray([0.0, 100.0], dtype=np.float64),
        )
        self.assertEqual(
            list(df.columns),
            [
                "depth_index",
                "depth_m",
                "median_abs_error_prediction_vs_glorys_c",
                "count_prediction_vs_glorys",
                "median_abs_error_prediction_vs_argo_c",
                "count_prediction_vs_argo",
                "median_prediction_profile_c",
                "count_prediction_profile",
                "median_glorys_profile_c",
                "count_glorys_profile",
                "median_argo_profile_c",
                "count_argo_profile",
            ],
        )
        self.assertAlmostEqual(
            float(df.loc[0, "median_abs_error_prediction_vs_glorys_c"]),
            0.75,
        )
        self.assertTrue(
            np.isnan(float(df.loc[1, "median_abs_error_prediction_vs_glorys_c"]))
        )
        self.assertAlmostEqual(
            float(df.loc[1, "median_abs_error_prediction_vs_argo_c"]),
            167.0,
        )
        self.assertAlmostEqual(
            float(df.loc[0, "median_prediction_profile_c"]),
            3.25,
        )
        self.assertAlmostEqual(
            float(df.loc[0, "median_glorys_profile_c"]),
            2.5,
        )
        self.assertAlmostEqual(
            float(df.loc[0, "median_argo_profile_c"]),
            1.0,
        )

    def test_average_plot_writers_create_png_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            depth_axis_m = np.asarray([0.0, 50.0, 100.0], dtype=np.float64)
            error_plot_path = tmp_path / "validation_median_absolute_error_by_depth.png"
            profile_error_plot_path = (
                tmp_path / "validation_median_profile_and_error_by_depth.png"
            )

            save_average_glorys_profile_error_plot(
                output_path=error_plot_path,
                mean_abs_error_prediction_vs_glorys=np.asarray(
                    [0.5, 0.75, 1.0], dtype=np.float64
                ),
                mean_abs_error_prediction_vs_argo=np.asarray(
                    [0.6, np.nan, 1.2], dtype=np.float64
                ),
                depth_axis=depth_axis_m,
                figure_title="Validation summary",
            )
            save_average_glorys_profile_and_error_plot(
                output_path=profile_error_plot_path,
                mean_argo_profile_c=np.asarray([18.0, np.nan, 7.0], dtype=np.float64),
                mean_prediction_profile_c=np.asarray(
                    [18.5, 11.0, 7.5], dtype=np.float64
                ),
                mean_glorys_profile_c=np.asarray([18.2, 10.5, 7.2], dtype=np.float64),
                mean_abs_error_prediction_vs_glorys=np.asarray(
                    [0.5, 0.75, 1.0], dtype=np.float64
                ),
                mean_abs_error_prediction_vs_argo=np.asarray(
                    [0.6, np.nan, 1.2], dtype=np.float64
                ),
                depth_axis=depth_axis_m,
                figure_title="Validation summary",
            )

            self.assertTrue(error_plot_path.is_file())
            self.assertGreater(error_plot_path.stat().st_size, 0)
            self.assertTrue(profile_error_plot_path.is_file())
            self.assertGreater(profile_error_plot_path.stat().st_size, 0)

    def test_filter_validation_summary_dataset_by_iso_week_keeps_requested_week(
        self,
    ) -> None:
        dataset = type(
            "DatasetStub",
            (),
            {
                "_rows": [
                    {"phase": "val", "date": 20150615},
                    {"phase": "val", "date": 20150618},
                    {"phase": "val", "date": 20150625},
                ],
                "split": "val",
            },
        )()
        selected_iso_year, selected_iso_week = (
            filter_validation_summary_dataset_by_iso_week(
                dataset,
                iso_year=2015,
                iso_week=25,
            )
        )

        self.assertEqual((selected_iso_year, selected_iso_week), (2015, 25))
        self.assertEqual(len(dataset._rows), 2)
        self.assertEqual(
            [int(row["date"]) for row in dataset._rows], [20150615, 20150618]
        )

    def test_filter_validation_summary_dataset_by_iso_week_requires_both_fields(
        self,
    ) -> None:
        dataset = type("DatasetStub", (), {"_rows": [], "split": "val"})()
        with self.assertRaisesRegex(ValueError, "--year and --iso-week"):
            filter_validation_summary_dataset_by_iso_week(
                dataset,
                iso_year=2015,
                iso_week=None,
            )


if __name__ == "__main__":
    unittest.main()
