from __future__ import annotations

import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.utils.hard_region_validation import (
    HardRegionValidationCallback,
    load_hard_region_definitions,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from tests.test_argo_geotiff_gridded_dataset import _make_geotiff_dataset


class TestHardRegionValidation(unittest.TestCase):
    def _write_regions(self, root: Path, *, evaluation_year: int) -> Path:
        """Write one region covering the complete tiny test raster."""
        path = root / "hard_regions.yaml"
        path.write_text(
            "\n".join(
                [
                    "type: FeatureCollection",
                    "properties:",
                    f"  evaluation_year: {int(evaluation_year)}",
                    "features:",
                    "- type: Feature",
                    "  properties:",
                    "    id: hard_ocean",
                    "    label: Hard Ocean",
                    "  geometry:",
                    "    type: Polygon",
                    "    coordinates:",
                    "    - - [10.0, 0.0]",
                    "      - [12.0, 0.0]",
                    "      - [12.0, 2.0]",
                    "      - [10.0, 2.0]",
                    "      - [10.0, 0.0]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_callback_logs_exact_prediction_vs_glorys_during_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir, cache_dir, land_mask_path = _make_geotiff_dataset(root)
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output_dir,
                metadata_cache_dir=cache_dir,
                split="all",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land_mask_path,
                patch_stride=2,
                max_land_fraction=1.0,
                val_year=2024,
                require_argo_for_all=False,
                include_salinity=True,
            )
            callback = HardRegionValidationCallback(
                dataset=dataset,
                regions_path=self._write_regions(root, evaluation_year=2024),
                evaluation_year=2024,
                max_patches_per_region=1,
                random_seed=7,
            )

            class ExactGlorysModel(pl.LightningModule):
                output_fields = ("temperature", "salinity")

                def transfer_batch_to_device(self, batch, device, dataloader_idx):
                    return batch

                def validation_step(self, batch, batch_idx):
                    return torch.zeros((), device=self.device)

                def predict_step(self, batch, batch_idx):
                    return {
                        "y_hat_temperature_denorm": temperature_normalize(
                            mode="denorm", tensor=batch["y"]
                        ),
                        "y_hat_salinity_denorm": salinity_normalize(
                            mode="denorm", tensor=batch["y_salinity"]
                        ),
                    }

            model = ExactGlorysModel()
            results = callback.evaluate(model)

            self.assertAlmostEqual(results["temperature"]["hard_ocean"].rmse, 0.0)
            self.assertAlmostEqual(results["salinity"]["hard_ocean"].mae, 0.0)
            self.assertGreater(results["temperature"]["hard_ocean"].count, 0)
            figure = callback._region_figure(
                callback._latest_image_data[("hard_ocean", "temperature")]
            )
            try:
                # Each unique mapped depth has three panels and three colorbars.
                self.assertEqual(len(figure.axes), 12)
            finally:
                plt.close(figure)

            image_logs: list[dict[str, object]] = []
            fake_wandb = SimpleNamespace(Image=lambda image: image)
            image_trainer = SimpleNamespace(
                is_global_zero=True,
                logger=SimpleNamespace(
                    experiment=SimpleNamespace(log=image_logs.append)
                ),
            )
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                callback._log_region_figures(image_trainer)
            self.assertEqual(len(image_logs), 1)
            self.assertIn(
                "hard_region_eval/hard_ocean/temperature_comparison", image_logs[0]
            )
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                logger=False,
                callbacks=[callback],
                limit_val_batches=1,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            trainer.validate(
                model,
                dataloaders=DataLoader(dataset, batch_size=1, num_workers=0),
                verbose=False,
            )
            self.assertIn(
                "hard_region_eval/hard_ocean/temperature_rmse",
                trainer.callback_metrics,
            )
            self.assertAlmostEqual(
                float(trainer.callback_metrics["hard_region_eval/temperature_rmse"]),
                0.0,
            )

    def test_committed_regions_are_provisional_and_fixed_to_2016(self) -> None:
        regions, evaluation_year = load_hard_region_definitions(
            Path("src/depth_recon/configs/evaluation/hard_regions_2016.yaml")
        )

        self.assertEqual(evaluation_year, 2016)
        self.assertEqual(
            [region.region_id for region in regions],
            ["greenland", "california_baja", "beaufort_arctic"],
        )


if __name__ == "__main__":
    unittest.main()
