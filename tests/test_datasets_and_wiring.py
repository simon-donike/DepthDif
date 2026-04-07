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
from utils.normalizations import temperature_normalize


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
                torch.equal(sample["x_valid_mask_1d"], expected_mask.any(dim=0, keepdim=True))
            )
            self.assertTrue(
                torch.equal(
                    sample["land_mask"],
                    torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                )
            )
            denorm_y = temperature_normalize(mode="denorm", tensor=sample["y"])
            self.assertTrue(
                torch.allclose(denorm_y[expected_mask], expected_y[expected_mask], atol=1e-5)
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

    def test_surface_temp_ostia_dataset_resamples_eo_and_zeroes_land_pixels(self) -> None:
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
                torch.equal(sample["eo"][0, :, -1], torch.zeros(4, dtype=sample["eo"].dtype))
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
            self.assertTrue(torch.equal(sample["x_valid_mask_1d"], x_mask.any(dim=0, keepdim=True)))
            self.assertTrue(torch.all(x_mask <= y_mask))
            self.assertGreaterEqual(int(sample["info"]["synthetic_pixel_count"]), 1)
            self.assertLessEqual(int(sample["info"]["synthetic_pixel_count"]), 2)
            self.assertTrue(bool(sample["info"]["synthetic_mode"]))
            denorm_x = temperature_normalize(mode="denorm", tensor=sample["x"])
            denorm_y = temperature_normalize(mode="denorm", tensor=sample["y"])
            self.assertTrue(torch.allclose(denorm_x[x_mask], denorm_y[x_mask], atol=1e-5))
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

    def test_datamodule_split_is_deterministic_and_loader_settings_are_applied(self) -> None:
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

        self.assertEqual(list(first.train_dataset.indices), list(second.train_dataset.indices))
        self.assertEqual(list(first.val_dataset.indices), list(second.val_dataset.indices))
        self.assertEqual(len(first.train_dataset), 8)
        self.assertEqual(len(first.val_dataset), 2)

        train_loader = first.train_dataloader()
        val_loader = first.val_dataloader()
        self.assertEqual(train_loader.batch_size, 3)
        self.assertEqual(val_loader.batch_size, 2)
        self.assertIsInstance(train_loader.sampler, SequentialSampler)
        self.assertIsInstance(val_loader.sampler, RandomSampler)

    def test_config_override_helpers_and_dataset_builder_use_nested_settings(self) -> None:
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

            dataset = build_dataset(str(data_config_path), payload["dataset"], split="all")

            self.assertIsInstance(dataset, OstiaArgoTiffDataset)
            self.assertTrue(dataset.synthetic_mode)
            self.assertEqual(dataset.synthetic_pixel_count, 1)
            self.assertTrue(dataset.return_info)
            self.assertFalse(dataset.return_coords)
            self.assertEqual(dataset.random_seed, 13)


if __name__ == "__main__":
    unittest.main()
