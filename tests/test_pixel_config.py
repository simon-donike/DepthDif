from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from depth_recon.configs.config_resolver_pixel import (
    apply_config_overrides,
    load_pixel_inference_config,
    load_pixel_training_config,
    load_yaml,
    resolve_pixel_scenario,
)
from depth_recon.models.baselines import UNetInfillingBaseline
from tests.test_argo_geotiff_gridded_dataset import _make_geotiff_dataset
from train import build_dataset


def _minimal_super_config(
    tmp_path: Path, *, scenario: str = "temperature"
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "data": {
            "dataset": {
                "core": {
                    "dataset_variant": "argo_geotiff_gridded",
                    "dataloader_type": "light",
                    "geotiff_root_dir": str(tmp_path / "geotiff_training"),
                    "metadata_cache_dir": str(tmp_path / "cache"),
                },
                "grid": {
                    "tile_size": 2,
                    "resolution_deg": 1.0,
                    "patch_grid_source": "land_mask",
                    "land_mask_path": str(tmp_path / "land_mask.tif"),
                    "patch_stride": 2,
                    "max_land_fraction": 1.0,
                },
                "sampling": {
                    "temporal_window_days": 7,
                    "glorys_var_name": "thetao",
                    "ostia_var_name": "analysed_sst",
                    "eo_source": "ostia",
                    "eo_var_name": "analysed_sst",
                },
                "selection": {
                    "require_argo_for_train": True,
                    "require_argo_for_val": True,
                    "require_argo_for_all": False,
                },
                "synthetic": {"enabled": False, "pixel_count": 1},
                "output": {
                    "return_info": False,
                    "return_coords": True,
                    "include_salinity": False,
                },
                "runtime": {"random_seed": 7, "cache_size": 2},
            },
            "split": {"val_year": 2018, "val_fraction": 0.2},
            "dataloader": {"num_workers": 0, "prefetch_factor": 2, "val_shuffle": True},
        },
        "model": {
            "model_type": "unet_baseline",
            "depth_channels": 50,
            "resume_checkpoint": False,
            "load_checkpoint_only": False,
            "condition_mask_channels": 1,
            "condition_include_eo": True,
            "condition_use_valid_mask": True,
            "condition_use_land_mask": True,
            "clamp_known_pixels": False,
            "mask_loss_with_valid_pixels": True,
            "parameterization": "x0",
            "log_intermediates": False,
            "ema": {"enabled": False},
            "ambient_occlusion": {"enabled": False},
            "post_process": {
                "gaussian_blur": {"enabled": False, "sigma": 0.75, "kernel_size": 5}
            },
            "coord_conditioning": {"enabled": False, "include_date": False},
            "unet_baseline": {
                "base_channels": 8,
                "channel_mults": [1],
                "norm_groups": 4,
                "dropout": 0.0,
                "lr": None,
                "weight_decay": 1.0e-4,
                "per_channel_valid_mask": True,
            },
            "unet": {"dim": 8, "dim_mults": [1], "with_time_emb": True},
        },
        "inference": {
            "grid": {
                "patch_stride": 96,
                "min_ocean_fraction": 0.05,
                "land_mask_path": str(tmp_path / "land_mask.tif"),
            },
            "dataloader": {"batch_size": 64, "num_workers": 6, "prefetch_factor": 2},
        },
        "training": {
            "training": {
                "lr": 1.0e-4,
                "batch_size": 1,
                "noise": {
                    "num_timesteps": 2,
                    "schedule": "linear",
                    "beta_start": 1.0e-4,
                    "beta_end": 2.0e-2,
                },
                "validation_sampling": {"sampler": "ddim", "ddim_num_timesteps": 2},
            },
            "trainer": {},
            "wandb": {"verbose": False},
            "dataloader": {"batch_size": 1, "val_batch_size": 1},
            "scheduler": {"reduce_on_plateau": {"enabled": False}},
        },
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


class TestPixelConfig(unittest.TestCase):
    def test_super_config_derives_temperature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            _write_yaml(
                config_path, _minimal_super_config(tmp_path, scenario="temperature")
            )

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                runtime_config_dir=tmp_path / "runtime",
                write_snapshots=False,
            )

        self.assertEqual(bundle.scenario, "temperature")
        self.assertEqual(bundle.model_cfg["model"]["output_fields"], ["temperature"])
        self.assertEqual(bundle.model_cfg["model"]["generated_channels"], 50)
        self.assertEqual(bundle.model_cfg["model"]["condition_channels"], 4)
        self.assertFalse(bundle.data_cfg["dataset"]["output"]["include_salinity"])
        self.assertEqual(bundle.data_cfg["dataset"]["sampling"]["eo_source"], "ostia")
        self.assertEqual(
            bundle.data_cfg["dataset"]["sampling"]["eo_var_name"],
            "analysed_sst",
        )

    def test_super_config_derives_salinity_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            _write_yaml(
                config_path, _minimal_super_config(tmp_path, scenario="salinity")
            )

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                runtime_config_dir=tmp_path / "runtime",
                write_snapshots=False,
            )

        self.assertEqual(bundle.model_cfg["model"]["output_fields"], ["salinity"])
        self.assertEqual(bundle.model_cfg["model"]["generated_channels"], 50)
        self.assertEqual(bundle.model_cfg["model"]["condition_channels"], 4)
        self.assertEqual(bundle.data_cfg["dataset"]["output"]["fields"], ["salinity"])
        self.assertTrue(bundle.data_cfg["dataset"]["output"]["include_salinity"])
        self.assertEqual(bundle.data_cfg["dataset"]["sampling"]["eo_source"], "sss")
        self.assertEqual(bundle.data_cfg["dataset"]["sampling"]["eo_var_name"], "sos")

    def test_super_config_derives_joint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            _write_yaml(config_path, _minimal_super_config(tmp_path, scenario="joint"))

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                runtime_config_dir=tmp_path / "runtime",
                write_snapshots=False,
            )

        self.assertEqual(
            bundle.model_cfg["model"]["output_fields"], ["temperature", "salinity"]
        )
        self.assertEqual(bundle.model_cfg["model"]["generated_channels"], 100)
        self.assertEqual(bundle.model_cfg["model"]["condition_channels"], 6)
        self.assertEqual(
            bundle.data_cfg["dataset"]["output"]["fields"],
            ["temperature", "salinity"],
        )
        self.assertTrue(bundle.data_cfg["dataset"]["output"]["include_salinity"])

    def test_cli_scenario_overrides_super_config_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            _write_yaml(
                config_path, _minimal_super_config(tmp_path, scenario="temperature")
            )

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                scenario_override="joint",
                runtime_config_dir=tmp_path / "runtime",
                write_snapshots=False,
            )

        self.assertEqual(bundle.scenario, "joint")
        self.assertEqual(bundle.model_cfg["model"]["generated_channels"], 100)

    def test_set_override_applies_after_scenario_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            _write_yaml(config_path, _minimal_super_config(tmp_path, scenario="joint"))

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                overrides=["model.generated_channels=12"],
                runtime_config_dir=tmp_path / "runtime",
                write_snapshots=False,
            )

        self.assertEqual(
            bundle.model_cfg["model"]["output_fields"], ["temperature", "salinity"]
        )
        self.assertEqual(bundle.model_cfg["model"]["generated_channels"], 12)

    def test_unet_baseline_override_derives_per_channel_condition_channels(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "super.yaml"
            config = _minimal_super_config(tmp_path, scenario="temperature")
            config["model"]["depth_channels"] = 2
            _write_yaml(config_path, config)

            training_bundle = load_pixel_training_config(
                config_path_value=config_path,
                overrides=["model.model_type=unet_baseline"],
                runtime_config_dir=tmp_path / "training_runtime",
                write_snapshots=False,
            )
            inference_bundle = load_pixel_inference_config(
                config_path_value=config_path,
                overrides=["model.model_type=unet_baseline"],
                runtime_config_dir=tmp_path / "inference_runtime",
                write_snapshots=False,
            )

            for bundle in (training_bundle, inference_bundle):
                model_section = bundle.model_cfg["model"]
                self.assertEqual(model_section["model_type"], "unet_baseline")
                self.assertEqual(model_section["generated_channels"], 2)
                self.assertEqual(model_section["condition_mask_channels"], 1)
                self.assertEqual(model_section["condition_channels"], 4)

    def test_selected_scenarios_materialize_expected_training_and_inference_settings(
        self,
    ) -> None:
        expected = {
            "temperature": {
                "fields": ["temperature"],
                "include_salinity": False,
                "generated_channels": 50,
                "condition_channels": 4,
                "eo_source": "ostia",
                "eo_var_name": "analysed_sst",
            },
            "salinity": {
                "fields": ["salinity"],
                "include_salinity": True,
                "generated_channels": 50,
                "condition_channels": 4,
                "eo_source": "sss",
                "eo_var_name": "sos",
            },
            "joint": {
                "fields": ["temperature", "salinity"],
                "include_salinity": True,
                "generated_channels": 100,
                "condition_channels": 6,
                "eo_source": "ostia",
                "eo_var_name": "analysed_sst",
            },
        }
        for selected_scenario, contract in expected.items():
            with self.subTest(selected_scenario=selected_scenario):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    config_path = tmp_path / "super.yaml"
                    _write_yaml(
                        config_path,
                        _minimal_super_config(tmp_path, scenario="temperature"),
                    )

                    training_bundle = load_pixel_training_config(
                        config_path_value=config_path,
                        scenario_override=selected_scenario,
                        runtime_config_dir=tmp_path / "training_runtime",
                        write_snapshots=False,
                    )
                    inference_bundle = load_pixel_inference_config(
                        config_path_value=config_path,
                        scenario_override=selected_scenario,
                        runtime_config_dir=tmp_path / "inference_runtime",
                        write_snapshots=False,
                    )

                    for bundle in (training_bundle, inference_bundle):
                        effective_model = load_yaml(bundle.effective_model_config_path)
                        effective_data = load_yaml(bundle.effective_data_config_path)
                        model_section = effective_model["model"]
                        data_output = effective_data["dataset"]["output"]

                        self.assertEqual(bundle.scenario, selected_scenario)
                        self.assertEqual(model_section["scenario"], selected_scenario)
                        self.assertEqual(
                            model_section["output_fields"], contract["fields"]
                        )
                        self.assertEqual(
                            model_section["generated_channels"],
                            contract["generated_channels"],
                        )
                        self.assertEqual(
                            model_section["condition_channels"],
                            contract["condition_channels"],
                        )
                        self.assertEqual(data_output["fields"], contract["fields"])
                        self.assertEqual(
                            data_output["include_salinity"],
                            contract["include_salinity"],
                        )
                        self.assertEqual(
                            effective_data["dataset"]["sampling"]["eo_source"],
                            contract["eo_source"],
                        )
                        self.assertEqual(
                            effective_data["dataset"]["sampling"]["eo_var_name"],
                            contract["eo_var_name"],
                        )

    def test_invalid_scenario_and_override_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported pixel scenario"):
            resolve_pixel_scenario({"scenario": "oxygen"})

        with self.assertRaisesRegex(KeyError, "does_not_exist"):
            apply_config_overrides(
                ["model.does_not_exist=true"],
                {"model": {"known": False}, "data": {}, "training": {}},
            )

    def test_materialized_effective_configs_instantiate_geotiff_dataset_and_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            geotiff_root, _cache_root, land_mask = _make_geotiff_dataset(tmp_path)
            config = _minimal_super_config(tmp_path, scenario="temperature")
            config["data"]["dataset"]["core"]["geotiff_root_dir"] = str(geotiff_root)
            config["data"]["dataset"]["grid"]["land_mask_path"] = str(land_mask)
            config["model"]["depth_channels"] = 2
            config["model"]["condition_include_eo"] = False
            config["model"]["condition_use_land_mask"] = False
            config_path = tmp_path / "super.yaml"
            _write_yaml(config_path, config)

            bundle = load_pixel_training_config(
                config_path_value=config_path,
                runtime_config_dir=tmp_path / "runtime",
                snapshot_dir=tmp_path / "snapshots",
            )
            dataset = build_dataset(
                bundle.effective_data_config_path,
                bundle.data_cfg["dataset"],
                split="train",
            )
            model = UNetInfillingBaseline.from_config(
                bundle.effective_model_config_path,
                bundle.effective_data_config_path,
                bundle.effective_training_config_path,
            )

            self.assertGreater(len(dataset), 0)
            self.assertEqual(model.output_fields, ("temperature",))
            self.assertEqual(model.generated_channels, 2)
            self.assertEqual(model.condition_channels, 2)
            self.assertTrue((tmp_path / "snapshots" / "super.yaml").is_file())
            self.assertTrue(
                (tmp_path / "snapshots" / "data_config_effective.yaml").is_file()
            )

    def test_inference_super_config_derives_all_scenarios(self) -> None:
        expected = {
            "temperature": (["temperature"], 50, 4, False, "ostia", "analysed_sst"),
            "salinity": (["salinity"], 50, 4, True, "sss", "sos"),
            "joint": (
                ["temperature", "salinity"],
                100,
                6,
                True,
                "ostia",
                "analysed_sst",
            ),
        }
        for scenario, (
            fields,
            generated,
            condition,
            include_salinity,
            eo_source,
            eo_var_name,
        ) in expected.items():
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    config_path = tmp_path / "inference_super.yaml"
                    _write_yaml(
                        config_path,
                        _minimal_super_config(tmp_path, scenario=scenario),
                    )

                    bundle = load_pixel_inference_config(
                        config_path_value=config_path,
                        runtime_config_dir=tmp_path / "runtime",
                        write_snapshots=False,
                    )

                    self.assertEqual(bundle.scenario, scenario)
                    self.assertEqual(bundle.model_cfg["model"]["output_fields"], fields)
                    self.assertEqual(
                        bundle.model_cfg["model"]["generated_channels"], generated
                    )
                    self.assertEqual(
                        bundle.model_cfg["model"]["condition_channels"], condition
                    )
                    self.assertEqual(
                        bundle.data_cfg["dataset"]["output"]["include_salinity"],
                        include_salinity,
                    )
                    self.assertEqual(
                        bundle.data_cfg["dataset"]["sampling"]["eo_source"],
                        eo_source,
                    )
                    self.assertEqual(
                        bundle.data_cfg["dataset"]["sampling"]["eo_var_name"],
                        eo_var_name,
                    )

    def test_inference_super_config_materializes_and_propagates_settings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            geotiff_root, _cache_root, land_mask = _make_geotiff_dataset(tmp_path)
            config = _minimal_super_config(tmp_path, scenario="salinity")
            config["data"]["dataset"]["core"]["geotiff_root_dir"] = str(geotiff_root)
            config["data"]["dataset"]["grid"]["land_mask_path"] = str(land_mask)
            config["data"]["dataloader"]["val_shuffle"] = True
            config["model"]["depth_channels"] = 2
            config["model"]["condition_include_eo"] = False
            config["model"]["condition_use_valid_mask"] = True
            config["model"]["condition_mask_channels"] = 2
            config["model"]["condition_use_land_mask"] = True
            config["model"]["unet"]["dim"] = 12
            config["training"]["dataloader"]["batch_size"] = 3
            config["training"]["training"]["validation_sampling"][
                "ddim_num_timesteps"
            ] = 1
            config["inference"]["grid"]["patch_stride"] = 4
            config["inference"]["grid"]["min_ocean_fraction"] = 0.25
            config["inference"]["dataloader"]["batch_size"] = 5
            config_path = tmp_path / "inference_super.yaml"
            _write_yaml(config_path, config)

            bundle = load_pixel_inference_config(
                config_path_value=config_path,
                overrides=[
                    "inference.dataloader.num_workers=0",
                    "model.condition_mask_channels=3",
                    "model.condition_channels=5",
                ],
                runtime_config_dir=tmp_path / "runtime",
                snapshot_dir=tmp_path / "snapshots",
            )
            effective_model = load_yaml(bundle.effective_model_config_path)
            effective_data = load_yaml(bundle.effective_data_config_path)
            effective_training = load_yaml(bundle.effective_training_config_path)
            effective_inference = load_yaml(bundle.effective_inference_config_path)
            dataset = build_dataset(
                bundle.effective_data_config_path,
                bundle.data_cfg["dataset"],
                split="train",
            )
            model = UNetInfillingBaseline.from_config(
                bundle.effective_model_config_path,
                bundle.effective_data_config_path,
                bundle.effective_training_config_path,
            )

            self.assertGreater(len(dataset), 0)
            self.assertEqual(dataset.eo_source, "sss")
            self.assertEqual(dataset.eo_var_name, "sos")
            self.assertTrue(effective_data["dataloader"]["val_shuffle"])
            self.assertEqual(effective_model["model"]["output_fields"], ["salinity"])
            self.assertEqual(effective_model["model"]["generated_channels"], 2)
            self.assertEqual(effective_model["model"]["condition_channels"], 5)
            self.assertEqual(effective_model["model"]["condition_mask_channels"], 3)
            self.assertEqual(
                effective_data["dataset"]["output"]["fields"], ["salinity"]
            )
            self.assertTrue(effective_data["dataset"]["output"]["include_salinity"])
            self.assertEqual(effective_training["dataloader"]["batch_size"], 3)
            self.assertEqual(
                effective_inference["inference"]["grid"]["patch_stride"], 4
            )
            self.assertEqual(
                effective_inference["inference"]["grid"]["min_ocean_fraction"],
                0.25,
            )
            self.assertEqual(
                effective_inference["inference"]["dataloader"]["batch_size"], 5
            )
            self.assertEqual(
                effective_inference["inference"]["dataloader"]["num_workers"], 0
            )
            self.assertEqual(model.output_fields, ("salinity",))
            self.assertEqual(model.generated_channels, 2)
            self.assertEqual(model.condition_channels, 5)
            self.assertTrue((tmp_path / "snapshots" / "inference_super.yaml").is_file())
            self.assertTrue(
                (tmp_path / "snapshots" / "inference_config_effective.yaml").is_file()
            )


if __name__ == "__main__":
    unittest.main()
