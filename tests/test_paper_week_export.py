from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from depth_recon.inference.export_paper_week import (
    _build_parser,
    _load_method_specs,
    export_paper_week,
)


class TestPaperWeekExport(unittest.TestCase):
    def _write_models_config(self, path: Path) -> Path:
        """Write a tiny paper-week methods config."""
        path.write_text(
            yaml.safe_dump(
                {
                    "methods": {
                        "idw": {
                            "label": "IDW",
                            "model_type": "idw_baseline",
                        },
                        "cnn": {
                            "label": "CNN",
                            "model_type": "cnn_baseline",
                            "temperature_checkpoint": "cnn_temperature.ckpt",
                            "salinity_checkpoint": "cnn_salinity.ckpt",
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_method_specs_require_checkpoints_except_idw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            idw_only = root / "idw.yaml"
            idw_only.write_text(
                yaml.safe_dump(
                    {
                        "methods": {
                            "idw": {
                                "label": "IDW",
                                "model_type": "idw_baseline",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            specs = _load_method_specs(idw_only)

            self.assertEqual(specs[0].name, "idw")
            self.assertIsNone(specs[0].temperature_checkpoint)
            self.assertIsNone(specs[0].salinity_checkpoint)

            missing_checkpoint = root / "missing_checkpoint.yaml"
            missing_checkpoint.write_text(
                yaml.safe_dump(
                    {
                        "methods": {
                            "cnn": {
                                "label": "CNN",
                                "model_type": "cnn_baseline",
                                "temperature_checkpoint": "temp.ckpt",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "salinity_checkpoint"):
                _load_method_specs(missing_checkpoint)

    def test_export_paper_week_writes_holdout_before_inference_and_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "paper_2024_W02"
            models_config = self._write_models_config(root / "models.yaml")
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            calls = []

            args = _build_parser().parse_args(
                [
                    "--config",
                    str(root / "config.yaml"),
                    "--year",
                    "2024",
                    "--iso-week",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--models-config",
                    str(models_config),
                    "--device",
                    "cpu",
                    "--sampler",
                    "ddim",
                    "--ddim-steps",
                    "3",
                    "--batch-size",
                    "2",
                    "--seed",
                    "11",
                    "--en4-holdout-fraction",
                    "0.5",
                    "--set",
                    "model.model_type=ignored_by_wrapper",
                ]
            )

            holdout_df = pd.DataFrame.from_records(
                [
                    {
                        "date": 20240108,
                        "grid_row": 0,
                        "grid_col": 0,
                        "lon": 10.5,
                        "lat": 1.5,
                        "profile_index": 0,
                        "temperature_valid_depth_count": 2,
                        "salinity_valid_depth_count": 2,
                        "holdout_fraction": 0.5,
                        "split_seed": 11,
                    }
                ]
            )
            fake_context = SimpleNamespace(
                root=dataset_root,
                depth_axis_m=np.asarray([0.0, 10.0], dtype=np.float32),
            )

            def fake_profiles_csv(*, context, holdout_df, output_path):
                """Write the held-out profile CSV without opening a profile store."""
                pd.DataFrame.from_records(
                    [
                        {
                            "date": 20240108,
                            "grid_row": 0,
                            "grid_col": 0,
                            "profile_index": 0,
                            "variable": variable,
                            "channel_index": channel_index,
                            "depth_m": 0.0 if channel_index == 0 else 10.0,
                            "value": value,
                        }
                        for variable, values in {
                            "temperature": [10.0, 20.0],
                            "salinity": [35.0, 36.0],
                        }.items()
                        for channel_index, value in enumerate(values)
                    ]
                ).to_csv(output_path, index=False)
                return Path(output_path)

            def fake_climatology(**kwargs):
                """Write a tiny climatology summary referenced by the bundle manifest."""
                clim_dir = Path(kwargs["output_dir"])
                clim_dir.mkdir(parents=True, exist_ok=True)
                summary_path = clim_dir / "climatology_summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "kind": "en4_climatology_idw",
                            "artifacts": {
                                "temperature": "climatology_temperature.tif",
                                "salinity": "climatology_salinity.tif",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (clim_dir / "climatology_temperature.tif").write_text(
                    "temp", encoding="utf-8"
                )
                (clim_dir / "climatology_salinity.tif").write_text(
                    "sal", encoding="utf-8"
                )
                return SimpleNamespace(summary_path=summary_path)

            def fake_run_inference(single_args):
                """Record parsed global-export args and write a fake run summary."""
                holdout_path = Path(single_args.en4_holdout_locations_csv)
                self.assertTrue(holdout_path.is_file())
                self.assertEqual(single_args.depth_export_mode, "native")
                method_name = Path(single_args.output_root).name
                expected_model_type = {
                    "idw": "idw_baseline",
                    "cnn": "cnn_baseline",
                }[method_name]
                self.assertIn(
                    "model.model_type=ignored_by_wrapper",
                    single_args.config_overrides[:-1],
                )
                self.assertEqual(
                    single_args.config_overrides[-1],
                    f"model.model_type={expected_model_type}",
                )
                calls.append(single_args)
                run_dir = Path(single_args.output_root) / str(single_args.output_name)
                run_dir.mkdir(parents=True, exist_ok=True)
                depth_exports = []
                for channel_index in range(2):
                    suffix = f"depth_{channel_index:03d}"
                    prediction = run_dir / f"{single_args.scenario}_{suffix}.tif"
                    prediction.write_text("prediction", encoding="utf-8")
                    ground_truth_name = None
                    if bool(single_args.persist_ground_truth_rasters):
                        ground_truth = (
                            run_dir / f"{single_args.scenario}_{suffix}_glorys.tif"
                        )
                        ground_truth.write_text("glorys", encoding="utf-8")
                        ground_truth_name = ground_truth.name
                    depth_exports.append(
                        {
                            "suffix": suffix,
                            "label": suffix,
                            "requested_depth_m": 0.0 if channel_index == 0 else 10.0,
                            "actual_depth_m": 0.0 if channel_index == 0 else 10.0,
                            "channel_index": channel_index,
                            "prediction_tif_path": prediction.name,
                            "ground_truth_tif_path": ground_truth_name,
                        }
                    )
                (run_dir / "selected_patches.csv").write_text(
                    "patch_id,grid_y0,grid_x0\n0,0,0\n", encoding="utf-8"
                )
                summary_path = run_dir / "run_summary.yaml"
                summary_path.write_text(
                    yaml.safe_dump(
                        {
                            "variable": single_args.scenario,
                            "selected_date": 20240108,
                            "depth_export_mode": "native",
                            "depth_exports": depth_exports,
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    run_dir=run_dir,
                    summary_path=summary_path,
                    selected_date=20240108,
                )

            with (
                patch(
                    "depth_recon.inference.export_paper_week._resolve_context_bundle",
                    return_value=(object(), {}, 20240108, dataset_root),
                ),
                patch(
                    "depth_recon.inference.export_paper_week.load_dataset_context",
                    return_value=fake_context,
                ),
                patch(
                    "depth_recon.inference.export_paper_week.select_en4_holdout_locations",
                    return_value=holdout_df,
                ),
                patch(
                    "depth_recon.inference.export_paper_week.write_en4_holdout_profiles_csv",
                    side_effect=fake_profiles_csv,
                ),
                patch(
                    "depth_recon.inference.export_paper_week.build_climatology_artifacts",
                    side_effect=fake_climatology,
                ),
            ):
                manifest = export_paper_week(args, run_inference=fake_run_inference)

            manifest_path = output_dir / "paper_week_manifest.json"
            manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(calls), 4)
            self.assertIsNone(calls[0].checkpoint_path)
            self.assertIsNone(calls[1].checkpoint_path)
            self.assertEqual(calls[2].checkpoint_path, "cnn_temperature.ckpt")
            self.assertEqual(calls[3].checkpoint_path, "cnn_salinity.ckpt")
            self.assertTrue(calls[0].persist_ground_truth_rasters)
            self.assertTrue(calls[1].persist_ground_truth_rasters)
            self.assertFalse(calls[2].persist_ground_truth_rasters)
            self.assertFalse(calls[3].persist_ground_truth_rasters)
            self.assertEqual(manifest["method_order"], ["climatology", "idw", "cnn"])
            self.assertEqual(manifest_json["depth_export_mode"], "native")
            self.assertEqual(
                set(manifest_json["methods"]["idw"]["variables"]),
                {"temperature", "salinity"},
            )
            self.assertEqual(
                set(manifest_json["methods"]["cnn"]["variables"]),
                {"temperature", "salinity"},
            )
            self.assertEqual(
                set(manifest_json["references"]["glorys"]), {"temperature", "salinity"}
            )
            self.assertEqual(
                len(
                    manifest_json["references"]["glorys"]["temperature"][
                        "depth_exports"
                    ]
                ),
                2,
            )
            self.assertTrue(
                (output_dir / "references" / "en4_holdout_locations.csv").is_file()
            )
            self.assertTrue(
                (output_dir / "references" / "en4_holdout_profiles.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
