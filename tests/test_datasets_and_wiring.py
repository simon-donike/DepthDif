from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import yaml

from data.datamodule import DepthTileDataModule
from data.dataset_4bands import SurfaceTempPatch4BandsLightDataset
from data.dataset_ostia import SurfaceTempPatchOstiaLightDataset
from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset
from train import apply_config_overrides, build_dataset, parse_config_override
from utils.normalizations import temperature_normalize, temperature_to_plot_unit
from utils.validation_denoise import (
    _temperature_band_to_plot_image,
    save_glorys_profile_comparison_plot,
)

matplotlib.use("Agg")
os.environ.setdefault("WANDB_MODE", "disabled")


class _RangeDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_base_index(
    tmp_path: Path,
    *,
    y_all: np.ndarray,
    extra_columns: dict[str, object] | None = None,
) -> Path:
    y_path = tmp_path / "target.npy"
    np.save(y_path, y_all.astype(np.float32))
    row: dict[str, object] = {
        "y_npy_path": y_path.name,
        "source_file": "depth_patch_20240105.nc",
        "lat0": -10.0,
        "lat1": 10.0,
        "lon0": 170.0,
        "lon1": -170.0,
    }
    if extra_columns is not None:
        row.update(extra_columns)
    csv_path = tmp_path / "index.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return csv_path


def _write_tiff(
    path: Path,
    array: np.ndarray,
    *,
    nodata: float | int | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=int(data.shape[-1]),
        height=int(data.shape[-2]),
        count=int(data.shape[0]),
        dtype=str(data.dtype),
        crs="EPSG:4326",
        transform=from_origin(10.0, 50.0, 1.0, 1.0),
        nodata=nodata,
    ) as ds:
        ds.write(data)
        if tags:
            ds.update_tags(**tags)


class TestDatasetsAndWiring(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

    def test_surface_temp_4band_dataset_builds_masks_coords_and_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            y_all = np.array(
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[10.0, 11.0], [0.0, np.nan]],
                    [[14.0, 15.0], [16.0, np.nan]],
                    [[18.0, 19.0], [20.0, np.nan]],
                ],
                dtype=np.float32,
            )
            csv_path = _write_base_index(tmp_path, y_all=y_all)
            dataset = SurfaceTempPatch4BandsLightDataset(
                csv_path=csv_path,
                split="all",
                mask_fraction=0.0,
                return_coords=True,
            )

            sample = dataset[0]

            expected_y = torch.from_numpy(y_all[1:])
            expected_mask = torch.tensor(
                [
                    [[True, True], [False, False]],
                    [[True, True], [True, False]],
                    [[True, True], [True, False]],
                ]
            )
            self.assertEqual(sample["eo"].shape, (1, 2, 2))
            self.assertEqual(sample["y"].shape, (3, 2, 2))
            self.assertTrue(torch.equal(sample["x_valid_mask"], expected_mask))
            self.assertTrue(torch.equal(sample["y_valid_mask"], expected_mask))
            self.assertTrue(
                torch.equal(
                    sample["x_valid_mask_1d"], expected_mask.any(dim=0, keepdim=True)
                )
            )
            self.assertTrue(
                torch.equal(
                    sample["land_mask"],
                    torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                )
            )
            denorm_y = temperature_normalize(mode="denorm", tensor=sample["y"])
            self.assertTrue(
                torch.allclose(
                    denorm_y[expected_mask], expected_y[expected_mask], atol=1e-5
                )
            )
            self.assertEqual(sample["date"], 20240105)
            self.assertTrue(
                torch.allclose(sample["coords"], torch.tensor([0.0, 180.0]), atol=1e-5)
            )

    def test_surface_temp_4band_dataset_can_hide_every_x_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            y_all = np.array(
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[10.0, 11.0], [12.0, 13.0]],
                    [[14.0, 15.0], [16.0, 17.0]],
                    [[18.0, 19.0], [20.0, 21.0]],
                ],
                dtype=np.float32,
            )
            csv_path = _write_base_index(tmp_path, y_all=y_all)
            dataset = SurfaceTempPatch4BandsLightDataset(
                csv_path=csv_path,
                split="all",
                mask_fraction=1.0,
                mask_strategy="tracks",
            )

            sample = dataset[0]

            self.assertFalse(bool(sample["x_valid_mask"].any().item()))
            self.assertTrue(bool(sample["y_valid_mask"].any().item()))
            self.assertTrue(torch.equal(sample["x"], torch.zeros_like(sample["x"])))

    def test_surface_temp_ostia_dataset_resamples_eo_and_zeroes_land_pixels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            y_all = np.array(
                [
                    [
                        [2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0],
                        [14.0, 15.0, 16.0, 17.0],
                    ],
                    [
                        [10.0, 11.0, 12.0, np.nan],
                        [13.0, 14.0, 15.0, np.nan],
                        [16.0, 17.0, 18.0, np.nan],
                        [19.0, 20.0, 21.0, np.nan],
                    ],
                    [
                        [22.0, 23.0, 24.0, np.nan],
                        [25.0, 26.0, 27.0, np.nan],
                        [28.0, 29.0, 30.0, np.nan],
                        [31.0, 32.0, 33.0, np.nan],
                    ],
                ],
                dtype=np.float32,
            )
            ostia = np.array([[5.0, 7.0], [9.0, 11.0]], dtype=np.float32)
            ostia_path = tmp_path / "ostia.npy"
            np.save(ostia_path, ostia)
            csv_path = _write_base_index(
                tmp_path,
                y_all=y_all,
                extra_columns={"ostia_npy_path": ostia_path.name},
            )
            dataset = SurfaceTempPatchOstiaLightDataset(
                csv_path=csv_path,
                split="all",
                target_band_start=1,
                target_band_end=3,
            )

            sample = dataset[0]

            self.assertEqual(sample["eo"].shape, (1, 4, 4))
            self.assertEqual(sample["y"].shape, (2, 4, 4))
            self.assertTrue(
                torch.equal(
                    sample["eo"][0, :, -1], torch.zeros(4, dtype=sample["eo"].dtype)
                )
            )
            self.assertTrue(bool((sample["eo"][0, :, 0] != 0).all().item()))

    def test_ostia_argo_tiff_dataset_synthetic_mode_rebuilds_sparse_x(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ostia = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
            argo = np.array(
                [
                    [[1.0, np.nan], [np.nan, 4.0]],
                    [[5.0, 6.0], [np.nan, 8.0]],
                ],
                dtype=np.float32,
            )
            glorys_celsius = np.array(
                [
                    [[10.0, 11.0], [12.0, np.nan]],
                    [[20.0, 21.0], [22.0, np.nan]],
                ],
                dtype=np.float32,
            )
            packed_glorys = np.where(
                np.isfinite(glorys_celsius),
                np.rint(glorys_celsius * 100.0),
                OstiaArgoTiffDataset.GLORYS_PACK_NODATA,
            ).astype(np.int16)
            tags = {
                "value_encoding": "packed_int16_celsius_x100",
                "scale_factor": "0.01",
                "add_offset": "0.0",
                "packed_nodata": str(int(OstiaArgoTiffDataset.GLORYS_PACK_NODATA)),
            }

            ostia_path = tmp_path / "ostia.tif"
            argo_path = tmp_path / "argo.tif"
            glorys_path = tmp_path / "glorys.tif"
            _write_tiff(ostia_path, ostia)
            _write_tiff(argo_path, argo)
            _write_tiff(
                glorys_path,
                packed_glorys,
                nodata=int(OstiaArgoTiffDataset.GLORYS_PACK_NODATA),
                tags=tags,
            )

            manifest_path = tmp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "ostia_tif_path": ostia_path.name,
                        "argo_tif_path": argo_path.name,
                        "glorys_tif_path": glorys_path.name,
                        "lat0": -4.0,
                        "lat1": 2.0,
                        "lon0": 10.0,
                        "lon1": 20.0,
                        "date": 20240229,
                        "phase": "train",
                        "export_skipped_reason": "",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = OstiaArgoTiffDataset(
                csv_path=manifest_path,
                split="train",
                return_info=True,
                return_coords=True,
                synthetic_mode=True,
                synthetic_pixel_count=1,
                random_seed=3,
            )
            sample = dataset[0]

            x_mask = sample["x_valid_mask"]
            y_mask = sample["y_valid_mask"]
            self.assertTrue(
                torch.equal(sample["x_valid_mask_1d"], x_mask.any(dim=0, keepdim=True))
            )
            self.assertTrue(torch.all(x_mask <= y_mask))
            self.assertGreaterEqual(int(sample["info"]["synthetic_pixel_count"]), 1)
            self.assertLessEqual(int(sample["info"]["synthetic_pixel_count"]), 2)
            self.assertTrue(bool(sample["info"]["synthetic_mode"]))
            denorm_x = temperature_normalize(mode="denorm", tensor=sample["x"])
            denorm_y = temperature_normalize(mode="denorm", tensor=sample["y"])
            self.assertTrue(
                torch.allclose(denorm_x[x_mask], denorm_y[x_mask], atol=1e-5)
            )
            self.assertTrue(
                torch.equal(
                    sample["land_mask"],
                    torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                )
            )
            self.assertEqual(sample["date"], 20240228)
            self.assertTrue(
                torch.allclose(sample["coords"], torch.tensor([-1.0, 15.0]), atol=1e-5)
            )

    def test_ostia_argo_tiff_dataset_border_helper_repairs_only_full_edges(
        self,
    ) -> None:
        ostia_patch = np.array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        repaired_ostia, repaired_ostia_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                ostia_patch,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(
            repaired_ostia[2:, :],
            np.repeat(ostia_patch[1:2, :], 2, axis=0),
            atol=0.0,
        )
        np.testing.assert_allclose(
            repaired_ostia[:2, :], ostia_patch[:2, :], atol=0.0
        )
        np.testing.assert_array_equal(
            repaired_ostia_mask[2:, :], np.ones((2, 4), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_ostia_mask[:2, :], np.zeros((2, 4), dtype=bool)
        )

        four_border_rows = np.array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        repaired_four_border_rows, repaired_four_border_rows_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                four_border_rows,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(
            repaired_four_border_rows[2:, :],
            np.repeat(four_border_rows[1:2, :], 4, axis=0),
            atol=0.0,
        )
        np.testing.assert_array_equal(
            repaired_four_border_rows_mask[2:, :], np.ones((4, 4), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_four_border_rows_mask[:2, :], np.zeros((2, 4), dtype=bool)
        )

        too_many_border_rows = np.array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        repaired_too_many, repaired_too_many_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                too_many_border_rows,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(
            repaired_too_many[:6, :],
            np.vstack(
                [
                    too_many_border_rows[:2, :],
                    np.repeat(too_many_border_rows[1:2, :], 4, axis=0),
                ]
            ),
            atol=0.0,
        )
        np.testing.assert_allclose(
            repaired_too_many[6:, :], too_many_border_rows[6:, :], atol=0.0
        )
        np.testing.assert_array_equal(
            repaired_too_many_mask[:2, :], np.zeros((2, 4), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_too_many_mask[2:6, :], np.ones((4, 4), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_too_many_mask[6:, :], np.zeros((1, 4), dtype=bool)
        )

        partial_edge = np.array(
            [
                [0.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
            dtype=np.float32,
        )
        repaired_partial, repaired_partial_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                partial_edge,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(repaired_partial, partial_edge, atol=0.0)
        np.testing.assert_array_equal(
            repaired_partial_mask, np.zeros_like(partial_edge, dtype=bool)
        )

        blocked_top = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [np.nan, 2.0, 3.0, 4.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
            dtype=np.float32,
        )
        repaired_blocked_top, repaired_blocked_top_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                blocked_top,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(repaired_blocked_top, blocked_top, atol=0.0)
        np.testing.assert_array_equal(
            repaired_blocked_top_mask, np.zeros_like(blocked_top, dtype=bool)
        )

        mixed_edges = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 6.0, 7.0, 8.0],
                [np.nan, 10.0, 11.0, 12.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        repaired_mixed, repaired_mixed_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_2d(
                mixed_edges,
                zero_border_is_artifact=True,
            )
        )
        np.testing.assert_allclose(
            repaired_mixed[0, :],
            mixed_edges[1, :],
            atol=0.0,
        )
        np.testing.assert_array_equal(
            repaired_mixed_mask[0, :], np.ones(4, dtype=bool)
        )
        np.testing.assert_allclose(repaired_mixed[1:3, :], mixed_edges[1:3, :], atol=0.0)
        np.testing.assert_allclose(repaired_mixed[-1, :], mixed_edges[-1, :], atol=0.0)
        np.testing.assert_array_equal(
            repaired_mixed_mask[1:, :], np.zeros((3, 4), dtype=bool)
        )

    def test_ostia_argo_tiff_dataset_border_helper_repairs_only_affected_glorys_bands(
        self,
    ) -> None:
        glorys = np.array(
            [
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [14.0, 15.0, 16.0, 17.0],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, 32.0, 33.0],
                    [np.nan, np.nan, 36.0, 37.0],
                    [np.nan, np.nan, 40.0, 41.0],
                    [np.nan, np.nan, 44.0, 45.0],
                ],
                [
                    [18.0, 19.0, 20.0, 21.0],
                    [22.0, np.nan, 24.0, 25.0],
                    [26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0],
                ],
            ],
            dtype=np.float32,
        )
        repaired, repaired_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_stack(
                glorys,
                zero_border_is_artifact=False,
            )
        )
        np.testing.assert_allclose(
            repaired[0, 2:, :],
            np.repeat(glorys[0, 1:2, :], 2, axis=0),
            atol=0.0,
        )
        np.testing.assert_allclose(
            repaired[1, :, :2],
            np.repeat(glorys[1, :, 2:3], 2, axis=1),
            atol=0.0,
        )
        np.testing.assert_allclose(repaired[2], glorys[2], atol=0.0)
        np.testing.assert_array_equal(
            repaired_mask[0, 2:, :], np.ones((2, 4), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_mask[1, :, :2], np.ones((4, 2), dtype=bool)
        )
        np.testing.assert_array_equal(
            repaired_mask[2], np.zeros_like(glorys[2], dtype=bool)
        )

        four_nan_cols = np.array(
            [
                [
                    [10.0, 11.0, np.nan, np.nan, np.nan, np.nan],
                    [14.0, 15.0, np.nan, np.nan, np.nan, np.nan],
                    [18.0, 19.0, np.nan, np.nan, np.nan, np.nan],
                    [22.0, 23.0, np.nan, np.nan, np.nan, np.nan],
                ]
            ],
            dtype=np.float32,
        )
        repaired_four_nan_cols, repaired_four_nan_cols_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_stack(
                four_nan_cols,
                zero_border_is_artifact=False,
            )
        )
        np.testing.assert_allclose(
            repaired_four_nan_cols[0, :, 2:],
            np.repeat(four_nan_cols[0, :, 1:2], 4, axis=1),
            atol=0.0,
        )
        np.testing.assert_array_equal(
            repaired_four_nan_cols_mask[0, :, 2:],
            np.ones((4, 4), dtype=bool),
        )
        np.testing.assert_array_equal(
            repaired_four_nan_cols_mask[0, :, :2],
            np.zeros((4, 2), dtype=bool),
        )

        too_many_nan_cols = np.array(
            [
                [
                    [10.0, 11.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [14.0, 15.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [18.0, 19.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [22.0, 23.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ],
            dtype=np.float32,
        )
        repaired_too_many_nan_cols, repaired_too_many_nan_cols_mask = (
            OstiaArgoTiffDataset._repair_full_border_artifacts_stack(
                too_many_nan_cols,
                zero_border_is_artifact=False,
            )
        )
        np.testing.assert_allclose(
            repaired_too_many_nan_cols[0, :, :6],
            np.concatenate(
                [
                    too_many_nan_cols[0, :, :2],
                    np.repeat(too_many_nan_cols[0, :, 1:2], 4, axis=1),
                ],
                axis=1,
            ),
            atol=0.0,
        )
        np.testing.assert_allclose(
            repaired_too_many_nan_cols[0, :, 6:],
            too_many_nan_cols[0, :, 6:],
            atol=0.0,
        )
        np.testing.assert_array_equal(
            repaired_too_many_nan_cols_mask[0, :, :2],
            np.zeros((4, 2), dtype=bool),
        )
        np.testing.assert_array_equal(
            repaired_too_many_nan_cols_mask[0, :, 2:6],
            np.ones((4, 4), dtype=bool),
        )
        np.testing.assert_array_equal(
            repaired_too_many_nan_cols_mask[0, :, 6:],
            np.zeros((4, 2), dtype=bool),
        )

    def test_ostia_argo_tiff_dataset_repairs_returned_images_without_touching_masks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ostia = np.array(
                [
                    [
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
            argo = np.array(
                [
                    [
                        [1.0, 2.0, np.nan, 4.0],
                        [5.0, np.nan, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                    ],
                    [
                        [21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0],
                        [29.0, 30.0, 31.0, 32.0],
                        [33.0, 34.0, 35.0, 36.0],
                    ],
                ],
                dtype=np.float32,
            )
            glorys_celsius = np.array(
                [
                    [
                        [10.0, 11.0, 12.0, 13.0],
                        [14.0, 15.0, 16.0, 17.0],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ],
                    [
                        [np.nan, np.nan, 32.0, 33.0],
                        [np.nan, np.nan, 36.0, 37.0],
                        [np.nan, np.nan, 40.0, 41.0],
                        [np.nan, np.nan, 44.0, 45.0],
                    ],
                ],
                dtype=np.float32,
            )
            packed_glorys = np.where(
                np.isfinite(glorys_celsius),
                np.rint(glorys_celsius * 100.0),
                OstiaArgoTiffDataset.GLORYS_PACK_NODATA,
            ).astype(np.int16)
            tags = {
                "value_encoding": "packed_int16_celsius_x100",
                "scale_factor": "0.01",
                "add_offset": "0.0",
                "packed_nodata": str(int(OstiaArgoTiffDataset.GLORYS_PACK_NODATA)),
            }

            ostia_path = tmp_path / "ostia.tif"
            argo_path = tmp_path / "argo.tif"
            glorys_path = tmp_path / "glorys.tif"
            _write_tiff(ostia_path, ostia)
            _write_tiff(argo_path, argo)
            _write_tiff(
                glorys_path,
                packed_glorys,
                nodata=int(OstiaArgoTiffDataset.GLORYS_PACK_NODATA),
                tags=tags,
            )

            manifest_path = tmp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "ostia_tif_path": ostia_path.name,
                        "argo_tif_path": argo_path.name,
                        "glorys_tif_path": glorys_path.name,
                        "lat0": -4.0,
                        "lat1": 2.0,
                        "lon0": 10.0,
                        "lon1": 20.0,
                        "date": 20240215,
                        "phase": "train",
                        "export_skipped_reason": "",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = OstiaArgoTiffDataset(
                csv_path=manifest_path,
                split="train",
                return_info=False,
                return_coords=False,
                synthetic_mode=False,
            )
            sample = dataset[0]

            denorm_eo = temperature_normalize(mode="denorm", tensor=sample["eo"])
            denorm_y = temperature_normalize(mode="denorm", tensor=sample["y"])
            denorm_x = temperature_normalize(mode="denorm", tensor=sample["x"])

            expected_bottom = torch.tensor(
                [9.0, 10.0, 11.0, 12.0], dtype=torch.float32
            )
            expected_center_cols = torch.tensor(
                [
                    [32.0, 32.0, 32.0, 33.0],
                    [36.0, 36.0, 36.0, 37.0],
                    [40.0, 40.0, 40.0, 41.0],
                    [44.0, 44.0, 44.0, 45.0],
                ],
                dtype=torch.float32,
            )
            self.assertTrue(
                torch.allclose(denorm_eo[0, -1, :], expected_bottom, atol=1e-5)
            )
            self.assertTrue(
                torch.allclose(
                    denorm_eo[0, -2, :],
                    expected_bottom,
                    atol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(
                    denorm_y[0, -2:, :],
                    torch.tensor(
                        [
                            [14.0, 15.0, 16.0, 17.0],
                            [14.0, 15.0, 16.0, 17.0],
                        ],
                        dtype=torch.float32,
                    ),
                    atol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(denorm_y[1], expected_center_cols, atol=1e-5)
            )

            self.assertTrue(
                torch.equal(
                    sample["y_valid_mask"][0, -2:, :],
                    torch.ones((2, 4), dtype=torch.bool),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample["y_valid_mask"][1, :, :2],
                    torch.ones((4, 2), dtype=torch.bool),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample["land_mask"][0, -2:, :], torch.ones((2, 4), dtype=torch.float32)
                )
            )
            self.assertTrue(
                torch.equal(sample["x_valid_mask"], torch.from_numpy(np.isfinite(argo)))
            )
            self.assertTrue(
                torch.equal(
                    sample["x_valid_mask_1d"],
                    sample["x_valid_mask"].any(dim=0, keepdim=True),
                )
            )
            valid_x = sample["x_valid_mask"]
            self.assertTrue(
                torch.allclose(
                    denorm_x[valid_x],
                    torch.from_numpy(argo)[valid_x],
                    atol=1e-5,
                )
            )

    def test_temperature_plotting_uses_shared_celsius_scale(self) -> None:
        cool_context = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        warm_context = torch.tensor([[20.0, 30.0]], dtype=torch.float32)

        cool_plot = _temperature_band_to_plot_image(cool_context)
        warm_plot = _temperature_band_to_plot_image(warm_context)
        expected_twenty = float(
            temperature_to_plot_unit(
                torch.tensor(20.0, dtype=torch.float32), tensor_is_normalized=False
            ).item()
        )

        self.assertAlmostEqual(float(cool_plot[0, 1]), expected_twenty, places=6)
        self.assertAlmostEqual(float(warm_plot[0, 0]), expected_twenty, places=6)

    def test_save_glorys_profile_comparison_plot_writes_two_panel_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.png"

            saved_path = save_glorys_profile_comparison_plot(
                output_path=output_path,
                x_profile=np.asarray([15.0, np.nan, 10.0, 8.0], dtype=np.float32),
                y_hat_profile=np.asarray([14.0, 13.0, 11.0, 7.5], dtype=np.float32),
                y_target_profile=np.asarray([14.5, 12.5, 10.5, 7.0], dtype=np.float32),
                observed_profile=np.asarray([True, False, True, True], dtype=bool),
                depth_axis=np.asarray([0.0, 50.0, 100.0, 250.0], dtype=np.float64),
                ostia_sst_c=16.0,
                figure_title=(
                    "Week: ISO week 2026-W27 (Jul)\n"
                    "Location: 12.3457 deg S, 45.1250 deg E"
                ),
            )

            image = matplotlib.image.imread(saved_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.exists())
            self.assertGreater(int(image.shape[1]), int(image.shape[0]))

    def test_datamodule_split_is_deterministic_and_loader_settings_are_applied(
        self,
    ) -> None:
        dataloader_cfg = {
            "batch_size": 3,
            "val_batch_size": 2,
            "num_workers": 0,
            "val_num_workers": 0,
            "shuffle": False,
            "val_shuffle": True,
            "pin_memory": False,
        }
        dataset = _RangeDataset(10)
        first = DepthTileDataModule(
            dataset=dataset,
            dataloader_cfg=dataloader_cfg,
            val_fraction=0.2,
            seed=11,
        )
        second = DepthTileDataModule(
            dataset=_RangeDataset(10),
            dataloader_cfg=dataloader_cfg,
            val_fraction=0.2,
            seed=11,
        )
        first.setup("fit")
        second.setup("fit")

        self.assertEqual(
            list(first.train_dataset.indices), list(second.train_dataset.indices)
        )
        self.assertEqual(
            list(first.val_dataset.indices), list(second.val_dataset.indices)
        )
        self.assertEqual(len(first.train_dataset), 8)
        self.assertEqual(len(first.val_dataset), 2)

        train_loader = first.train_dataloader()
        val_loader = first.val_dataloader()
        self.assertEqual(train_loader.batch_size, 3)
        self.assertEqual(val_loader.batch_size, 2)
        self.assertIsInstance(train_loader.sampler, SequentialSampler)
        self.assertIsInstance(val_loader.sampler, RandomSampler)

    def test_config_override_helpers_and_dataset_builder_use_nested_settings(
        self,
    ) -> None:
        root = {
            "data": {"dataset": {"core": {"dataset_variant": "ostia_argo_disk"}}},
            "training": {"training": {"noise": {"num_timesteps": 10}}},
            "model": {"model": {"ambient_occlusion": {"enabled": False}}},
        }
        self.assertEqual(
            parse_config_override("training.training.noise.num_timesteps=7"),
            (
                "training",
                ["training", "noise", "num_timesteps"],
                7,
            ),
        )
        apply_config_overrides(
            [
                "training.training.noise.num_timesteps=7",
                "model.model.ambient_occlusion.enabled=true",
            ],
            root,
        )
        self.assertEqual(root["training"]["training"]["noise"]["num_timesteps"], 7)
        self.assertTrue(root["model"]["model"]["ambient_occlusion"]["enabled"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ostia = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
            argo = np.array([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
            glorys = np.array([[[1000, 1100], [1200, -32768]]], dtype=np.int16)
            tags = {
                "value_encoding": "packed_int16_celsius_x100",
                "scale_factor": "0.01",
                "add_offset": "0.0",
                "packed_nodata": "-32768",
            }
            ostia_path = tmp_path / "ostia.tif"
            argo_path = tmp_path / "argo.tif"
            glorys_path = tmp_path / "glorys.tif"
            _write_tiff(ostia_path, ostia)
            _write_tiff(argo_path, argo)
            _write_tiff(glorys_path, glorys, nodata=-32768, tags=tags)
            manifest_path = tmp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "ostia_tif_path": ostia_path.name,
                        "argo_tif_path": argo_path.name,
                        "glorys_tif_path": glorys_path.name,
                        "phase": "all",
                        "lat0": 0.0,
                        "lat1": 1.0,
                        "lon0": 2.0,
                        "lon1": 3.0,
                        "date": 20240115,
                    }
                ]
            ).to_csv(manifest_path, index=False)
            data_config_path = tmp_path / "data.yaml"
            payload = {
                "dataset": {
                    "core": {
                        "dataset_variant": "ostia_argo_disk",
                        "manifest_csv_path": str(manifest_path),
                    },
                    "output": {"return_info": True, "return_coords": False},
                    "synthetic": {"enabled": True, "pixel_count": 1},
                    "runtime": {"random_seed": 13},
                }
            }
            _write_yaml(data_config_path, payload)

            dataset = build_dataset(
                str(data_config_path), payload["dataset"], split="all"
            )

            self.assertIsInstance(dataset, OstiaArgoTiffDataset)
            self.assertTrue(dataset.synthetic_mode)
            self.assertEqual(dataset.synthetic_pixel_count, 1)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)
            self.assertEqual(dataset.random_seed, 13)


if __name__ == "__main__":
    unittest.main()
