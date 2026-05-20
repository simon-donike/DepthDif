from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from torch import nn

from depth_recon.inference.export_global import (
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA,
    DEFAULT_FULL_SAMPLE_COUNT,
    DEFAULT_UNCERTAINTY_NUM_SAMPLES,
    EXPORT_VARIABLE_SPECS,
    ExportInferenceWrapper,
    FullProfileSample,
    MosaicLayout,
    _cleanup_accumulator,
    _argo_point_features_for_patch,
    _build_parser,
    _build_inference_loader,
    _patch_split_feature_for_row,
    _full_profile_feature_for_sample,
    _default_run_stem,
    filter_selection_by_rectangle,
    _normalize_cli_args,
    _profile_graph_figure_title,
    _prepare_run_directory,
    _promote_production_run,
    _load_ground_truth_patch_celsius,
    _load_ground_truth_patch_for_variable,
    _prediction_zeros_to_nan,
    _repair_small_nodata_gaps_2d,
    create_raster_accumulator,
    build_global_mosaic,
    global_inference_dataset_overrides,
    resolve_depth_export_levels,
    select_export_indices,
    write_absolute_error_geotiff,
    write_global_top_band_geotiff,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize


class _TinyInferenceDataset:
    """Small map-style dataset used to test inference-loader collation."""

    def __len__(self) -> int:
        """Return the sample count."""
        return 3

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return one deterministic tensor sample."""
        value = float(idx)
        return {
            "x": torch.full((1, 1, 1), value, dtype=torch.float32),
            "date": 20260105 + int(idx),
        }


class _IncrementingPredictModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        _ = batch, batch_idx
        self.call_count += 1
        call_value = float(self.call_count)
        y_hat_denorm = torch.tensor(
            [[[[call_value]], [[call_value + 10.0]], [[call_value + 20.0]]]],
            dtype=torch.float32,
        )
        return {"y_hat_denorm": y_hat_denorm}


class _UncertaintyPredictModel(nn.Module):
    """Small model double that records uncertainty export calls."""

    def __init__(self) -> None:
        super().__init__()
        self.uncertainty_calls: list[tuple[int, int]] = []

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Return a deterministic salinity prediction tensor."""
        _ = batch, batch_idx
        return {"y_hat_salinity_denorm": torch.ones((1, 3, 2, 2), dtype=torch.float32)}

    def uncertainty_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        num_samples: int,
    ) -> dict[str, torch.Tensor]:
        """Return one field-specific 1-channel uncertainty map."""
        _ = batch
        self.uncertainty_calls.append((int(batch_idx), int(num_samples)))
        return {
            "uncertainty_salinity": torch.tensor(
                [[[[0.25, 0.50], [0.75, 1.00]]]], dtype=torch.float32
            )
        }


class _DecodedGlorysDataset:
    def _load_y_patch(self, row: dict[str, int]) -> np.ndarray:
        """Return one already decoded Celsius GLORYS patch."""
        return np.asarray(
            [
                [[float(row["date"] % 100), 11.0]],
                [[20.0, 21.0]],
            ],
            dtype=np.float32,
        )


class _DecodedSalinityDataset:
    def _load_y_salinity_patch(self, row: dict[str, int]) -> np.ndarray:
        """Return one already decoded PSU GLORYS salinity patch."""
        return np.asarray(
            [
                [[float(row["date"] % 100) + 30.0, 34.0]],
                [[35.0, 36.0]],
            ],
            dtype=np.float32,
        )


class _SalinityPredictModel(nn.Module):
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        _ = batch, batch_idx
        return {
            "y_hat_denorm": torch.full((1, 3, 1, 1), -1.0, dtype=torch.float32),
            "y_hat_salinity_denorm": torch.tensor(
                [[[[34.0]], [[35.0]], [[36.0]]]], dtype=torch.float32
            ),
        }


class TestGlobalInferenceExport(unittest.TestCase):
    def test_export_inference_wrapper_runs_one_prediction_per_batch(self) -> None:
        model = _IncrementingPredictModel()
        wrapper = ExportInferenceWrapper(
            model,
            variable_spec=EXPORT_VARIABLE_SPECS["temperature"],
            export_ground_truth=False,
            export_full_prediction_stack=True,
            depth_channel_indices=(0, 2),
        )

        outputs = wrapper({"y": torch.zeros((1, 3, 1, 1), dtype=torch.float32)})

        self.assertEqual(model.call_count, 1)
        torch.testing.assert_close(
            outputs["prediction_depth_stack"],
            torch.tensor([[[[1.0]], [[21.0]]]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            outputs["prediction_full_stack"],
            torch.tensor([[[[1.0]], [[11.0]], [[21.0]]]], dtype=torch.float32),
        )

    def test_export_inference_wrapper_denormalizes_ground_truth_to_celsius(
        self,
    ) -> None:
        model = _IncrementingPredictModel()
        wrapper = ExportInferenceWrapper(
            model,
            variable_spec=EXPORT_VARIABLE_SPECS["temperature"],
            export_ground_truth=True,
            export_full_prediction_stack=False,
            depth_channel_indices=(0, 2),
        )
        y_celsius = torch.tensor(
            [[[[4.0]], [[12.0]], [[18.0]]]],
            dtype=torch.float32,
        )
        y_norm = temperature_normalize(mode="norm", tensor=y_celsius)

        outputs = wrapper(
            {
                "y": y_norm,
                "y_valid_mask": torch.ones_like(y_norm, dtype=torch.bool),
            }
        )

        torch.testing.assert_close(
            outputs["ground_truth_depth_stack"],
            torch.tensor([[[[4.0]], [[18.0]]]], dtype=torch.float32),
        )

    def test_export_inference_wrapper_uses_salinity_keys_and_denormalization(
        self,
    ) -> None:
        wrapper = ExportInferenceWrapper(
            _SalinityPredictModel(),
            variable_spec=EXPORT_VARIABLE_SPECS["salinity"],
            export_ground_truth=True,
            export_full_prediction_stack=True,
            depth_channel_indices=(0, 2),
        )
        y_salinity_psu = torch.tensor(
            [[[[33.0]], [[34.0]], [[35.0]]]],
            dtype=torch.float32,
        )
        y_salinity_norm = salinity_normalize(mode="norm", tensor=y_salinity_psu)

        outputs = wrapper(
            {
                "y_salinity": y_salinity_norm,
                "y_salinity_valid_mask": torch.tensor(
                    [[[[True]], [[True]], [[False]]]],
                    dtype=torch.bool,
                ),
            }
        )

        torch.testing.assert_close(
            outputs["prediction_depth_stack"],
            torch.tensor([[[[34.0]], [[36.0]]]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            outputs["prediction_full_stack"],
            torch.tensor([[[[34.0]], [[35.0]], [[36.0]]]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            outputs["ground_truth_depth_stack"],
            torch.tensor([[[[33.0]], [[float("nan")]]]], dtype=torch.float32),
            equal_nan=True,
        )

    def test_export_inference_wrapper_exports_field_uncertainty_map(self) -> None:
        model = _UncertaintyPredictModel()
        wrapper = ExportInferenceWrapper(
            model,
            variable_spec=EXPORT_VARIABLE_SPECS["salinity"],
            export_ground_truth=False,
            export_full_prediction_stack=False,
            export_uncertainty=True,
            uncertainty_num_samples=DEFAULT_UNCERTAINTY_NUM_SAMPLES,
            depth_channel_indices=(0,),
        )

        outputs = wrapper({})

        self.assertEqual(
            model.uncertainty_calls, [(0, DEFAULT_UNCERTAINTY_NUM_SAMPLES)]
        )
        torch.testing.assert_close(
            outputs["uncertainty_map"],
            torch.tensor([[[[0.25, 0.50], [0.75, 1.00]]]], dtype=torch.float32),
        )

    def test_load_ground_truth_patch_celsius_uses_decoded_dataset_values(self) -> None:
        patch = _load_ground_truth_patch_celsius(
            _DecodedGlorysDataset(),
            {"date": 20260105},
        )

        assert patch is not None
        np.testing.assert_allclose(
            patch,
            np.asarray([[[5.0, 11.0]], [[20.0, 21.0]]], dtype=np.float32),
        )

    def test_load_ground_truth_patch_for_salinity_uses_salinity_loader(self) -> None:
        patch = _load_ground_truth_patch_for_variable(
            _DecodedSalinityDataset(),
            {"date": 20260105},
            EXPORT_VARIABLE_SPECS["salinity"],
        )

        assert patch is not None
        np.testing.assert_allclose(
            patch,
            np.asarray([[[35.0, 34.0]], [[35.0, 36.0]]], dtype=np.float32),
        )

    def test_default_full_sample_count_exports_all_locations(self) -> None:
        self.assertEqual(DEFAULT_FULL_SAMPLE_COUNT, -1)

    def test_default_inference_config_uses_super_config(self) -> None:
        self.assertTrue(
            DEFAULT_INFERENCE_CONFIG.endswith("inference_super_config.yaml")
        )

    def test_default_export_gaussian_blur_sigma_is_zero(self) -> None:
        self.assertEqual(DEFAULT_EXPORT_GAUSSIAN_BLUR_SIGMA, 0.0)

    def test_inference_loader_defaults_are_parallel_prefetch(self) -> None:
        parser = _build_parser()

        args = parser.parse_args(["--year", "2026", "--iso-week", "2"])

        self.assertEqual(args.config_path, DEFAULT_INFERENCE_CONFIG)
        self.assertIsNone(args.scenario)
        self.assertEqual(args.config_overrides, [])
        self.assertIsNone(args.inference_num_workers)
        self.assertIsNone(args.inference_prefetch_factor)
        self.assertFalse(args.export_uncertainty)
        self.assertEqual(args.uncertainty_num_samples, DEFAULT_UNCERTAINTY_NUM_SAMPLES)

    def test_build_inference_loader_collates_selected_grid_rows(self) -> None:
        rows = [
            {"date": 20260105, "patch_id": "a"},
            {"date": 20260106, "patch_id": "b"},
        ]
        loader = _build_inference_loader(
            dataset=_TinyInferenceDataset(),
            indices=[2, 0],
            rows=rows,
            batch_size=2,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=False,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["dataset_indices"], [2, 0])
        self.assertEqual(batch["rows"], rows)
        torch.testing.assert_close(
            batch["batch"]["x"],
            torch.tensor([[[[2.0]]], [[[0.0]]]], dtype=torch.float32),
        )

    def test_normalize_cli_args_accepts_sigma_colon_zero(self) -> None:
        self.assertEqual(
            _normalize_cli_args(["--device", "cpu", "sigma:0"]),
            ["--device", "cpu", "--sigma", "0"],
        )

    def test_select_export_indices_picks_iso_week_wednesday(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260107, "lat0": 0.0, "lat1": 1.0, "lon0": 1.0, "lon1": 2.0},
            {"date": 20260108, "lat0": 1.0, "lat1": 2.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260112, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
        ]

        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        self.assertEqual(selection.selected_date, 20260107)
        self.assertEqual(selection.iso_year, 2026)
        self.assertEqual(selection.iso_week, 2)
        self.assertEqual(selection.indices, [1])

    def test_select_export_indices_picks_closest_available_iso_week_date(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260108, "lat0": 0.0, "lat1": 1.0, "lon0": 1.0, "lon1": 2.0},
            {"date": 20260109, "lat0": 1.0, "lat1": 2.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20260112, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
        ]

        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        self.assertEqual(selection.selected_date, 20260108)
        self.assertEqual(selection.iso_year, 2026)
        self.assertEqual(selection.iso_week, 2)
        self.assertEqual(selection.indices, [1])

    def test_select_export_indices_handles_iso_year_boundary_wednesday(self) -> None:
        rows = [
            {"date": 20241231, "lat0": 0.0, "lat1": 1.0, "lon0": 0.0, "lon1": 1.0},
            {"date": 20250101, "lat0": 0.0, "lat1": 1.0, "lon0": 1.0, "lon1": 2.0},
            {"date": 20250102, "lat0": 1.0, "lat1": 2.0, "lon0": 0.0, "lon1": 1.0},
        ]

        selection = select_export_indices(rows, iso_year=2025, iso_week=1)

        self.assertEqual(selection.selected_date, 20250101)
        self.assertEqual(selection.iso_year, 2025)
        self.assertEqual(selection.iso_week, 1)
        self.assertEqual(selection.indices, [1])

    def test_filter_selection_by_rectangle_keeps_intersecting_patches(self) -> None:
        rows = [
            {"date": 20260107, "lat0": 0.0, "lat1": 10.0, "lon0": 0.0, "lon1": 10.0},
            {"date": 20260107, "lat0": 0.0, "lat1": 10.0, "lon0": 10.0, "lon1": 20.0},
            {"date": 20260107, "lat0": 20.0, "lat1": 30.0, "lon0": 0.0, "lon1": 10.0},
        ]
        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        filtered = filter_selection_by_rectangle(
            rows,
            selection,
            rectangle=(9.0, 1.0, 12.0, 9.0),
        )

        self.assertEqual(filtered.indices, [0, 1])

    def test_filter_selection_by_rectangle_counts_touching_edges(self) -> None:
        rows = [
            {"date": 20260107, "lat0": 0.0, "lat1": 10.0, "lon0": 0.0, "lon1": 10.0},
            {"date": 20260107, "lat0": 0.0, "lat1": 10.0, "lon0": 20.0, "lon1": 30.0},
        ]
        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        filtered = filter_selection_by_rectangle(
            rows,
            selection,
            rectangle=(10.0, 2.0, 15.0, 8.0),
        )

        self.assertEqual(filtered.indices, [0])

    def test_filter_selection_by_rectangle_supports_antimeridian(self) -> None:
        rows = [
            {"date": 20260107, "lat0": -5.0, "lat1": 5.0, "lon0": 170.0, "lon1": 179.0},
            {
                "date": 20260107,
                "lat0": -5.0,
                "lat1": 5.0,
                "lon0": -179.0,
                "lon1": -170.0,
            },
            {"date": 20260107, "lat0": -5.0, "lat1": 5.0, "lon0": -20.0, "lon1": -10.0},
        ]
        selection = select_export_indices(rows, iso_year=2026, iso_week=2)

        filtered = filter_selection_by_rectangle(
            rows,
            selection,
            rectangle=(175.0, -2.0, -175.0, 2.0),
        )

        self.assertEqual(filtered.indices, [1, 0])

    def test_global_inference_dataset_overrides_force_full_overlap_grid(self) -> None:
        overrides, metadata = global_inference_dataset_overrides(
            {"dataset": {"grid": {"tile_size": 128}}},
            land_mask_path="mask.tif",
        )

        self.assertEqual(overrides["grid"]["patch_grid_source"], "land_mask")
        self.assertEqual(overrides["grid"]["patch_stride"], 32)
        self.assertEqual(overrides["grid"]["max_land_fraction"], 0.95)
        self.assertFalse(overrides["selection"]["require_argo_for_all"])
        self.assertEqual(metadata["overlap_stitching"], "weighted")
        self.assertAlmostEqual(metadata["patch_overlap_fraction"], 0.75)
        self.assertAlmostEqual(metadata["min_ocean_fraction"], 0.05)

    def test_global_inference_dataset_overrides_accept_custom_min_ocean_fraction(
        self,
    ) -> None:
        overrides, metadata = global_inference_dataset_overrides(
            {"dataset": {"grid": {"tile_size": 100}}},
            land_mask_path="mask.tif",
            min_ocean_fraction=0.20,
        )

        self.assertEqual(overrides["grid"]["patch_stride"], 25)
        self.assertAlmostEqual(overrides["grid"]["max_land_fraction"], 0.80)
        self.assertAlmostEqual(metadata["min_ocean_fraction"], 0.20)

    def test_global_inference_dataset_overrides_accept_custom_patch_stride(
        self,
    ) -> None:
        overrides, metadata = global_inference_dataset_overrides(
            {"dataset": {"grid": {"tile_size": 128}}},
            land_mask_path="mask.tif",
            patch_stride=64,
        )

        self.assertEqual(overrides["grid"]["patch_stride"], 64)
        self.assertAlmostEqual(metadata["patch_overlap_fraction"], 0.50)

    def test_parser_accepts_one_command_upload_args(self) -> None:
        args = _build_parser().parse_args(
            [
                "--year",
                "2026",
                "--iso-week",
                "2",
                "--public-base-url",
                "https://example.test/globe",
                "--rclone-remote",
                "r2:bucket/globe",
                "--rclone-sync-scope",
                "globe",
                "--extra-zoom-levels",
                "1",
                "--min-ocean-fraction",
                "0.10",
            ]
        )

        self.assertEqual(args.year, 2026)
        self.assertEqual(args.iso_week, 2)
        self.assertEqual(args.public_base_url, "https://example.test/globe")
        self.assertEqual(args.rclone_remote, "r2:bucket/globe")
        self.assertEqual(args.extra_zoom_levels, 1)
        self.assertEqual(args.min_ocean_fraction, 0.10)
        self.assertFalse(args.export_uncertainty)

    def test_build_global_mosaic_places_adjacent_tiles_in_expected_grid(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 2.0, "lon0": 0.0, "lon1": 2.0},
            {"date": 20260105, "lat0": 0.0, "lat1": 2.0, "lon0": 2.0, "lon1": 4.0},
        ]
        patches = [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        ]

        mosaic, transform = build_global_mosaic(
            rows=rows,
            top_band_predictions=patches,
            nodata=-9999.0,
        )

        np.testing.assert_allclose(
            mosaic,
            np.asarray([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(transform.a, 1.0, places=6)
        self.assertAlmostEqual(transform.e, -1.0, places=6)
        self.assertAlmostEqual(transform.c, 0.0, places=6)
        self.assertAlmostEqual(transform.f, 2.0, places=6)

    def test_build_global_mosaic_weight_stitches_overlapping_tiles(self) -> None:
        rows = [
            {"date": 20260105, "lat0": 0.0, "lat1": 4.0, "lon0": 0.0, "lon1": 4.0},
            {"date": 20260105, "lat0": 0.0, "lat1": 4.0, "lon0": 2.0, "lon1": 6.0},
        ]
        patches = [
            np.full((4, 4), 2.0, dtype=np.float32),
            np.full((4, 4), 6.0, dtype=np.float32),
        ]

        mosaic, _ = build_global_mosaic(
            rows=rows,
            top_band_predictions=patches,
            nodata=-9999.0,
        )

        np.testing.assert_allclose(
            mosaic,
            np.asarray(
                [
                    [2.0, 2.0, 10.0 / 3.0, 14.0 / 3.0, 6.0, 6.0],
                    [2.0, 2.0, 10.0 / 3.0, 14.0 / 3.0, 6.0, 6.0],
                    [2.0, 2.0, 10.0 / 3.0, 14.0 / 3.0, 6.0, 6.0],
                    [2.0, 2.0, 10.0 / 3.0, 14.0 / 3.0, 6.0, 6.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_repair_small_nodata_gaps_fills_only_tiny_internal_seams(self) -> None:
        raster = np.asarray(
            [
                [7.0, 7.0, -9999.0, 7.0, 7.0],
                [7.0, 7.0, -9999.0, 7.0, 7.0],
            ],
            dtype=np.float32,
        )

        repaired, repaired_mask = _repair_small_nodata_gaps_2d(
            raster,
            nodata=-9999.0,
        )

        self.assertTrue(repaired_mask[:, 2].all())
        np.testing.assert_allclose(repaired, np.full((2, 5), 7.0, dtype=np.float32))

    def test_write_global_top_band_geotiff_repairs_small_internal_seam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            scratch_dir = tmp_path / "scratch"
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=32.0,
                top=32.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=32,
                height=32,
                patch_width=32,
                patch_height=32,
                transform=from_origin(0.0, 32.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=scratch_dir,
                stem="prediction_top_band",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 0
                accumulator.sum_array[:, :16] = 7.0
                accumulator.count_array[:, :16] = 1
                accumulator.sum_array[:, 17:] = 7.0
                accumulator.count_array[:, 17:] = 1

                tif_path = tmp_path / "prediction.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_top_band_celsius",
                    tags={"kind": "prediction"},
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)
                    self.assertEqual(ds.nodata, -9999.0)

                np.testing.assert_allclose(
                    band, np.full((32, 32), 7.0, dtype=np.float32)
                )
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_blurs_completed_raster_when_enabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_blur",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[8, 8] = 9.0

                tif_path = tmp_path / "prediction_blurred.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={"kind": "prediction"},
                    extra_gaussian_blur_sigma=1.0,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertLess(band[8, 8], 9.0)
                self.assertGreater(band[8, 7], 0.0)
                self.assertEqual(band[0, 0], -9999.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_skips_extra_blur_when_sigma_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_no_blur",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[8, 8] = 9.0

                tif_path = tmp_path / "prediction_unblurred.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={"kind": "prediction"},
                    extra_gaussian_blur_sigma=0.0,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertEqual(band[8, 8], 9.0)
                self.assertEqual(band[8, 7], -9999.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_prediction_zeros_to_nan_masks_exact_zero_values(self) -> None:
        patch = np.asarray(
            [[0.0, 1.0], [-0.0, 0.0001], [1.0e-7, -1.0e-7]], dtype=np.float32
        )

        masked = _prediction_zeros_to_nan(patch)

        self.assertTrue(np.isnan(masked[0, 0]))
        self.assertTrue(np.isnan(masked[1, 0]))
        self.assertTrue(np.isnan(masked[2, 0]))
        self.assertTrue(np.isnan(masked[2, 1]))
        self.assertEqual(masked[0, 1], 1.0)
        self.assertEqual(masked[1, 1], np.float32(0.0001))

    def test_write_global_top_band_geotiff_keeps_zero_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="default_zero",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 1.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[4, 4] = 0.0

                tif_path = tmp_path / "default_zero.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="glorys_surface_celsius",
                    tags={"kind": "ground_truth"},
                    prediction_zero_masked_to_nodata=False,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertEqual(band[4, 4], 0.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_masks_prediction_zero_to_nodata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_zero",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 2.0
                accumulator.count_array[:] = 1
                accumulator.sum_array[4, 4] = 0.0

                tif_path = tmp_path / "prediction_zero.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={
                        "kind": "prediction",
                        "prediction_zero_masked_to_nodata": "true",
                    },
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)
                    tags = ds.tags()

                self.assertEqual(band[4, 4], -9999.0)
                self.assertEqual(tags["prediction_zero_masked_to_nodata"], "true")
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_global_top_band_geotiff_masks_land_to_nodata_after_stitching(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=4.0,
                top=4.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=4,
                height=4,
                patch_width=4,
                patch_height=4,
                transform=from_origin(0.0, 4.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_land",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 0.0
                accumulator.count_array[:] = 0
                accumulator.sum_array[0, 0] = 8.0
                accumulator.count_array[0, 0] = 2
                accumulator.sum_array[0, 1] = 9.0
                accumulator.count_array[0, 1] = 1
                land_mask = np.zeros((4, 4), dtype=bool)
                land_mask[0, 1] = True
                land_mask[1, 1] = True

                tif_path = tmp_path / "prediction_land.tif"
                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_celsius",
                    tags={"kind": "prediction"},
                    land_mask=land_mask,
                )

                with rasterio.open(tif_path) as ds:
                    band = ds.read(1)

                self.assertEqual(band[0, 0], 4.0)
                self.assertEqual(band[0, 1], -9999.0)
                self.assertEqual(band[1, 1], -9999.0)
                self.assertEqual(band[3, 3], -9999.0)
            finally:
                _cleanup_accumulator(accumulator)

    def test_write_absolute_error_geotiff_computes_valid_celsius_difference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            transform = from_origin(0.0, 16.0, 1.0, 1.0)
            prediction_path = tmp_path / "prediction.tif"
            ground_truth_path = tmp_path / "glorys.tif"
            prediction = np.full((16, 16), 10.0, dtype=np.float32)
            ground_truth = np.full((16, 16), 7.0, dtype=np.float32)
            prediction[0, 1] = -9999.0
            ground_truth[0, 2] = -9999.0

            for path, data in (
                (prediction_path, prediction),
                (ground_truth_path, ground_truth),
            ):
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=16,
                    width=16,
                    count=1,
                    dtype="float32",
                    nodata=-9999.0,
                    crs="EPSG:4326",
                    transform=transform,
                ) as ds:
                    ds.write(data, 1)

            error_path = tmp_path / "absolute_error.tif"
            write_absolute_error_geotiff(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                output_path=error_path,
                nodata=-9999.0,
                band_description="absolute_error_surface_celsius",
                tags={"kind": "absolute_error"},
            )

            with rasterio.open(error_path) as ds:
                band = ds.read(1)
                tags = ds.tags()
                description = ds.descriptions[0]

        self.assertEqual(band[0, 0], 3.0)
        self.assertEqual(band[0, 1], -9999.0)
        self.assertEqual(band[0, 2], -9999.0)
        self.assertEqual(tags["kind"], "absolute_error")
        self.assertEqual(description, "absolute_error_surface_celsius")

    def test_resolve_depth_export_levels_uses_nearest_glorys_depths(self) -> None:
        levels = resolve_depth_export_levels(
            np.asarray(
                [
                    0.5,
                    9.8,
                    52.0,
                    97.0,
                    247.0,
                    505.0,
                    980.0,
                    1990.0,
                    2502.0,
                    4997.0,
                ],
                dtype=np.float64,
            )
        )

        self.assertEqual(
            [level.suffix for level in levels],
            [
                "surface",
                "10m",
                "50m",
                "100m",
                "250m",
                "500m",
                "1000m",
                "2000m",
                "2500m",
                "5000m",
            ],
        )
        self.assertEqual(
            [level.channel_index for level in levels],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        self.assertAlmostEqual(levels[3].requested_depth_m, 100.0)
        self.assertAlmostEqual(levels[3].actual_depth_m, 97.0)

    def test_depth_geotiff_metadata_records_requested_and_actual_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=16.0,
                top=16.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=16,
                height=16,
                patch_width=16,
                patch_height=16,
                transform=from_origin(0.0, 16.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="prediction_100m",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 12.5
                accumulator.count_array[:] = 1
                tif_path = tmp_path / "global_top_band_20260105_prediction_100m.tif"

                write_global_top_band_geotiff(
                    output_path=tif_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_100m_celsius",
                    tags={
                        "kind": "prediction",
                        "depth_label": "100m",
                        "requested_depth_m": "100.000",
                        "actual_depth_m": "97.000",
                        "channel_index": "1",
                        "value_units": "degree_Celsius",
                        "value_space": "denormalized_dequantized_celsius",
                    },
                )

                with rasterio.open(tif_path) as ds:
                    tags = ds.tags()
                    self.assertEqual(ds.descriptions[0], "predicted_100m_celsius")
                    self.assertEqual(tags["depth_label"], "100m")
                    self.assertEqual(tags["requested_depth_m"], "100.000")
                    self.assertEqual(tags["actual_depth_m"], "97.000")
                    self.assertEqual(tags["channel_index"], "1")
                    self.assertEqual(tags["value_units"], "degree_Celsius")
                    self.assertEqual(
                        tags["value_space"], "denormalized_dequantized_celsius"
                    )
            finally:
                _cleanup_accumulator(accumulator)

    def test_salinity_geotiff_metadata_records_units_and_error_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout = MosaicLayout(
                left=0.0,
                bottom=0.0,
                right=4.0,
                top=4.0,
                pixel_width=1.0,
                pixel_height=1.0,
                width=4,
                height=4,
                patch_width=4,
                patch_height=4,
                transform=from_origin(0.0, 4.0, 1.0, 1.0),
            )
            accumulator = create_raster_accumulator(
                root_dir=tmp_path / "scratch",
                stem="salinity_prediction",
                layout=layout,
            )
            try:
                accumulator.sum_array[:] = 35.5
                accumulator.count_array[:] = 1
                prediction_path = tmp_path / "salinity_prediction_surface.tif"
                write_global_top_band_geotiff(
                    output_path=prediction_path,
                    accumulator=accumulator,
                    layout=layout,
                    nodata=-9999.0,
                    band_description="predicted_surface_psu",
                    tags={
                        "kind": "prediction",
                        "variable": "salinity",
                        "value_units": "PSU",
                        "value_unit_label": "PSU",
                        "value_space": "denormalized_dequantized_psu",
                        "source_value_transform": "model_prediction_denormalized_to_psu",
                    },
                )
                ground_truth_path = tmp_path / "salinity_glorys_surface.tif"
                with rasterio.open(prediction_path) as prediction_ds:
                    profile = prediction_ds.profile
                with rasterio.open(ground_truth_path, "w", **profile) as ds:
                    ds.write(np.full((4, 4), 34.0, dtype=np.float32), 1)

                error_path = tmp_path / "salinity_absolute_error_surface.tif"
                write_absolute_error_geotiff(
                    prediction_path=prediction_path,
                    ground_truth_path=ground_truth_path,
                    output_path=error_path,
                    nodata=-9999.0,
                    band_description="absolute_error_surface_psu",
                    tags={
                        "kind": "absolute_error",
                        "variable": "salinity",
                        "value_units": "PSU",
                        "value_space": "absolute_error_psu",
                        "source_value_transform": "abs(prediction_psu_minus_glorys_psu)",
                    },
                )

                with rasterio.open(prediction_path) as ds:
                    prediction_tags = ds.tags()
                    self.assertEqual(ds.descriptions[0], "predicted_surface_psu")
                with rasterio.open(error_path) as ds:
                    error_tags = ds.tags()
                    error_band = ds.read(1)
                    self.assertEqual(ds.descriptions[0], "absolute_error_surface_psu")

                self.assertEqual(prediction_tags["value_units"], "PSU")
                self.assertEqual(
                    prediction_tags["value_space"], "denormalized_dequantized_psu"
                )
                self.assertEqual(
                    prediction_tags["source_value_transform"],
                    "model_prediction_denormalized_to_psu",
                )
                self.assertEqual(error_tags["value_space"], "absolute_error_psu")
                self.assertEqual(
                    error_tags["source_value_transform"],
                    "abs(prediction_psu_minus_glorys_psu)",
                )
                self.assertEqual(error_band[0, 0], 1.5)
            finally:
                _cleanup_accumulator(accumulator)

    def test_argo_point_features_use_pixel_centers(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-7",
            "export_index": 11,
            "lat0": 0.0,
            "lat1": 2.0,
            "lon0": 10.0,
            "lon1": 12.0,
        }
        observed_mask = np.asarray([[True, False], [False, True]], dtype=bool)

        features = _argo_point_features_for_patch(
            row=row,
            observed_mask_2d=observed_mask,
        )

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["geometry"]["coordinates"], [10.5, 1.5])
        self.assertEqual(features[1]["geometry"]["coordinates"], [11.5, 0.5])
        self.assertEqual(features[0]["properties"]["date"], 20260105)
        self.assertEqual(features[0]["properties"]["patch_id"], "patch-7")
        self.assertEqual(features[0]["properties"]["export_index"], 11)
        self.assertNotIn("observed_depth_index", features[0]["properties"])
        self.assertNotIn("observed_temp_c", features[0]["properties"])

    def test_patch_split_feature_builds_closed_polygon_and_keeps_split(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-9",
            "export_index": 13,
            "lat0": -2.0,
            "lat1": 1.0,
            "lon0": 10.0,
            "lon1": 14.0,
            "split": "train",
        }

        feature = _patch_split_feature_for_row(row)

        self.assertIsNotNone(feature)
        assert feature is not None
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(
            feature["geometry"]["coordinates"][0],
            [[10.0, 1.0], [14.0, 1.0], [14.0, -2.0], [10.0, -2.0], [10.0, 1.0]],
        )
        self.assertEqual(feature["properties"]["split"], "train")
        self.assertEqual(feature["properties"]["patch_id"], "patch-9")

    def test_patch_split_feature_skips_rows_without_train_val_label(self) -> None:
        row = {
            "date": 20260105,
            "patch_id": "patch-10",
            "export_index": 14,
            "lat0": -2.0,
            "lat1": 1.0,
            "lon0": 10.0,
            "lon1": 14.0,
            "split": "invalid",
        }

        self.assertIsNone(_patch_split_feature_for_row(row))

    def test_full_profile_feature_keeps_depth_stacks_and_graph_path(self) -> None:
        sample = FullProfileSample(
            dataset_index=4,
            row={
                "date": 20260105,
                "patch_id": "patch-12",
                "export_index": 18,
                "lat0": 0.0,
                "lat1": 2.0,
                "lon0": 10.0,
                "lon1": 12.0,
            },
            point_row=1,
            point_col=0,
            patch_height=2,
            patch_width=2,
            lon=10.5,
            lat=0.5,
            x_profile_c=np.asarray([12.0, 13.0, 14.0], dtype=np.float32),
            y_hat_profile_c=np.asarray([11.5, 12.5, 13.5], dtype=np.float32),
            y_target_profile_c=np.asarray([10.5, 11.5, 12.5], dtype=np.float32),
            ostia_sst_c=19.25,
            observed_profile=np.asarray([True, False, True], dtype=bool),
            target_valid_profile=np.asarray([True, True, False], dtype=bool),
        )

        feature = _full_profile_feature_for_sample(
            sample=sample,
            location_id="full_sample_001",
            graph_png_path="graphs/full_sample_001.png",
            depth_axis_m=np.asarray([0.5, 5.0, 25.0], dtype=np.float64),
        )

        self.assertEqual(feature["geometry"]["coordinates"], [10.5, 0.5])
        self.assertEqual(feature["properties"]["date"], 20260105)
        self.assertEqual(feature["properties"]["location_id"], "full_sample_001")
        self.assertEqual(
            feature["properties"]["graph_png_path"], "graphs/full_sample_001.png"
        )
        self.assertEqual(feature["properties"]["depth_m"], [0.5, 5.0, 25.0])
        self.assertEqual(feature["properties"]["ostia_sst_c"], 19.25)
        self.assertEqual(feature["properties"]["argo_profile_c"], [12.0, None, 14.0])
        self.assertEqual(
            feature["properties"]["prediction_profile_c"], [11.5, 12.5, None]
        )
        self.assertEqual(feature["properties"]["glorys_profile_c"], [10.5, 11.5, None])
        self.assertEqual(feature["properties"]["variable"], "temperature")
        self.assertEqual(feature["properties"]["value_units"], "degree_Celsius")
        self.assertEqual(feature["properties"]["argo_profile"], [12.0, None, 14.0])
        self.assertEqual(
            feature["properties"]["prediction_profile"], [11.5, 12.5, None]
        )
        self.assertEqual(feature["properties"]["glorys_profile"], [10.5, 11.5, None])

    def test_profile_graph_title_uses_iso_week_and_geographic_coordinates_only(
        self,
    ) -> None:
        title = _profile_graph_figure_title(
            sample_date=20260630,
            lat=-12.345678,
            lon=45.125,
        )

        self.assertEqual(
            title,
            "Week: ISO week 2026-W27 (Jul)\nLocation: 12.3457 deg S, 45.1250 deg E",
        )
        self.assertNotIn("Pixel", title)

    def test_default_run_stem_keeps_selected_date_in_artifact_names(self) -> None:
        self.assertEqual(_default_run_stem(20260105), "global_top_band_20260105")

    def test_prepare_run_directory_defaults_to_date_stamped_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"

            run_dir, production_dir = _prepare_run_directory(
                output_root,
                run_stem="global_top_band_20260105",
                output_name=None,
            )

            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir, output_root / "global_top_band_20260105")
            self.assertIsNone(production_dir)

    def test_promote_production_run_strips_date_from_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            staging_dir = Path(tmp_dir) / "staging"
            production_dir = Path(tmp_dir) / "production"
            staging_dir.mkdir(parents=True, exist_ok=True)

            (staging_dir / "global_top_band_20260105_prediction.tif").write_text(
                "prediction", encoding="utf-8"
            )
            (staging_dir / "global_top_band_20260105_prediction_100m.tif").write_text(
                "prediction depth", encoding="utf-8"
            )
            (staging_dir / "global_top_band_20260105_absolute_error.tif").write_text(
                "absolute error", encoding="utf-8"
            )
            (
                staging_dir / "global_top_band_20260105_absolute_error_100m.tif"
            ).write_text("absolute error depth", encoding="utf-8")
            (staging_dir / "global_top_band_20260105_argo_points.geojson").write_text(
                "points", encoding="utf-8"
            )
            (staging_dir / "global_top_band_20260105_patch_splits.geojson").write_text(
                "splits", encoding="utf-8"
            )
            with (staging_dir / "run_summary.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "selected_date": 20260105,
                        "run_dir": str(staging_dir),
                        "prediction_tif_path": "global_top_band_20260105_prediction.tif",
                        "ground_truth_tif_path": None,
                        "absolute_error_tif_path": "global_top_band_20260105_absolute_error.tif",
                        "depth_exports": [
                            {
                                "suffix": "100m",
                                "prediction_tif_path": "global_top_band_20260105_prediction_100m.tif",
                                "ground_truth_tif_path": None,
                                "absolute_error_tif_path": "global_top_band_20260105_absolute_error_100m.tif",
                            }
                        ],
                        "argo_points_geojson_path": "global_top_band_20260105_argo_points.geojson",
                        "patch_splits_geojson_path": "global_top_band_20260105_patch_splits.geojson",
                    },
                    f,
                    sort_keys=False,
                )

            _promote_production_run(staging_dir, production_dir)

            self.assertFalse(staging_dir.exists())
            self.assertTrue(
                (production_dir / "global_top_band_prediction.tif").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_prediction_100m.tif").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_absolute_error.tif").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_absolute_error_100m.tif").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_argo_points.geojson").exists()
            )
            self.assertTrue(
                (production_dir / "global_top_band_patch_splits.geojson").exists()
            )

            with (production_dir / "run_summary.yaml").open("r", encoding="utf-8") as f:
                summary = yaml.safe_load(f)

            self.assertEqual(summary["run_dir"], str(production_dir))
            self.assertEqual(
                summary["prediction_tif_path"], "global_top_band_prediction.tif"
            )
            self.assertEqual(
                summary["depth_exports"][0]["prediction_tif_path"],
                "global_top_band_prediction_100m.tif",
            )
            self.assertEqual(
                summary["absolute_error_tif_path"],
                "global_top_band_absolute_error.tif",
            )
            self.assertEqual(
                summary["depth_exports"][0]["absolute_error_tif_path"],
                "global_top_band_absolute_error_100m.tif",
            )
            self.assertEqual(
                summary["argo_points_geojson_path"],
                "global_top_band_argo_points.geojson",
            )
            self.assertEqual(
                summary["patch_splits_geojson_path"],
                "global_top_band_patch_splits.geojson",
            )


if __name__ == "__main__":
    unittest.main()
