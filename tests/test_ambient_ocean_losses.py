from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from depth_recon.losses import (
    AmbientOceanLoss,
    SpectralEnergyFloorLoss,
    StructureFunctionPriorLoss,
    sparse_increment_loss,
    aux_timestep_weight,
    sparse_observation_loss,
)
from depth_recon.models.diffusion.DenoisingDiffusionProcess.DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
)
from depth_recon.models.diffusion.PixelDiffusion import PixelDiffusionConditional


class _FakeForward(nn.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise
        self.num_timesteps = 2
        self.register_buffer("alphas_cumprod", torch.tensor([1.0, 0.25]))

    def forward(
        self,
        output: torch.Tensor,
        t: torch.Tensor,
        return_noise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        noisy = output + 0.5
        if return_noise:
            return noisy, self.noise.to(device=output.device, dtype=output.dtype)
        return noisy


class _StaticPredictor(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = prediction

    def forward(
        self,
        model_input: torch.Tensor,
        t: torch.Tensor,
        coord_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = t, coord_emb
        return self.prediction.to(device=model_input.device, dtype=model_input.dtype)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestAmbientOceanLosses(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_sparse_observation_loss_uses_masked_charbonnier_mean(self) -> None:
        x0_pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        obs_grid = torch.zeros_like(x0_pred)
        obs_mask = torch.tensor([[[[True, False], [True, False]]]])

        loss = sparse_observation_loss(
            x0_pred, obs_grid=obs_grid, obs_mask_grid=obs_mask, eps=1.0e-3
        )

        expected = (
            torch.sqrt(torch.tensor(1.0 + 1.0e-6))
            + torch.sqrt(torch.tensor(9.0 + 1.0e-6))
        ) / 2.0
        self.assertTrue(torch.isclose(loss, expected))

    def test_sparse_increment_loss_builds_vertical_pairs(self) -> None:
        x0_pred = torch.tensor([[[[0.0]], [[2.0]], [[5.0]]]])
        obs_grid = torch.tensor([[[[0.0]], [[1.0]], [[4.0]]]])
        obs_mask = torch.ones_like(obs_grid, dtype=torch.bool)

        loss = sparse_increment_loss(
            x0_pred,
            obs_grid=obs_grid,
            obs_mask_grid=obs_mask,
            eps=1.0e-3,
            max_pairs_per_sample=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_structure_function_prior_uses_toy_reference_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "s2.pt"
            torch.save(
                {
                    "distance_bins": torch.tensor([0.0, 2.0, 8.0]),
                    "s2_ref": torch.ones(2, 2),
                },
                ref_path,
            )
            loss_fn = StructureFunctionPriorLoss(
                reference_path=ref_path, num_pairs=64, per_depth=True
            )
            field = torch.randn(1, 2, 4, 4)

            loss = loss_fn(field, valid_mask=torch.ones(1, 1, 4, 4))

            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss), 0.0)

    def test_spectral_energy_floor_hinge_penalizes_low_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = Path(tmpdir) / "spectral.pt"
            torch.save({"energy_ref": torch.ones(2, 3)}, ref_path)
            loss_fn = SpectralEnergyFloorLoss(
                reference_path=ref_path, min_band=0, per_depth=True
            )
            field = torch.zeros(1, 2, 4, 4)

            loss = loss_fn(field)

            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(loss), 0.0)

    def test_enabled_glorys_prior_requires_reference_path(self) -> None:
        with self.assertRaises(ValueError):
            AmbientOceanLoss(
                {"structure_function_prior": {"enabled": True, "reference_path": None}}
            )

    def test_aux_timestep_weight_disabled_returns_one(self) -> None:
        weight = aux_timestep_weight(
            {"enabled": False},
            t=None,
            alphas_cumprod=None,
            reference=torch.tensor(2.0),
        )

        self.assertTrue(torch.isclose(weight, torch.tensor(1.0)))

    def test_aux_timestep_weight_linear_prefers_clean_timesteps(self) -> None:
        alphas = torch.linspace(1.0, 0.1, 5)
        clean = aux_timestep_weight(
            {
                "enabled": True,
                "mode": "linear",
                "linear_start_weight": 0.0,
                "linear_end_weight": 1.0,
            },
            t=torch.tensor([0]),
            alphas_cumprod=alphas,
            reference=torch.tensor(0.0),
        )
        noisy = aux_timestep_weight(
            {
                "enabled": True,
                "mode": "linear",
                "linear_start_weight": 0.0,
                "linear_end_weight": 1.0,
            },
            t=torch.tensor([4]),
            alphas_cumprod=alphas,
            reference=torch.tensor(0.0),
        )

        self.assertTrue(torch.isclose(clean, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(noisy, torch.tensor(0.0)))

    def test_aux_timestep_weight_snr_uses_gamma_and_clamps(self) -> None:
        weight = aux_timestep_weight(
            {
                "enabled": True,
                "mode": "snr",
                "snr_gamma": 5.0,
                "min_weight": 0.2,
                "max_weight": 0.8,
            },
            t=torch.tensor([0]),
            alphas_cumprod=torch.tensor([0.9, 0.5, 0.01]),
            reference=torch.tensor(0.0),
        )

        self.assertTrue(torch.isclose(weight, torch.tensor(0.8)))

    def test_ambient_ocean_loss_weights_only_auxiliary_terms_by_timestep(self) -> None:
        loss_fn = AmbientOceanLoss(
            {
                "ambient": {"weight": 2.0},
                "aux_timestep_weighting": {
                    "enabled": True,
                    "mode": "linear",
                    "linear_start_weight": 0.0,
                    "linear_end_weight": 1.0,
                },
                "sparse_observation": {
                    "enabled": True,
                    "weight": 3.0,
                    "eps": 0.0,
                },
            }
        )
        total, components = loss_fn(
            loss_ambient=torch.tensor(5.0),
            x0_pred=torch.ones(1, 1, 1, 1) * 2.0,
            obs_grid=torch.zeros(1, 1, 1, 1),
            obs_mask_grid=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            t=torch.tensor([1]),
            alphas_cumprod=torch.tensor([1.0, 0.5]),
        )

        self.assertTrue(
            torch.isclose(components["loss_aux_timestep_weight"], torch.tensor(0.0))
        )
        self.assertTrue(torch.isclose(total, torch.tensor(10.0)))

    def test_pixel_diffusion_forwards_timestep_context_to_ocean_loss(self) -> None:
        model = PixelDiffusionConditional(
            generated_channels=1,
            condition_channels=1,
            condition_include_eo=False,
            condition_use_valid_mask=False,
            parameterization="x0",
            num_timesteps=2,
            unet_dim=8,
            unet_dim_mults=(1,),
            wandb_verbose=False,
            losses_config={"sparse_observation": {"enabled": True}},
        )
        captured: dict[str, torch.Tensor | None] = {}

        class _CapturingOceanLoss(nn.Module):
            def forward(self, **kwargs):
                captured["t"] = kwargs.get("t")
                captured["alphas_cumprod"] = kwargs.get("alphas_cumprod")
                loss = kwargs["loss_ambient"]
                return loss, {
                    "loss_total": loss,
                    "loss_aux_timestep_weight": torch.ones_like(loss),
                }

        model.ocean_loss = _CapturingOceanLoss()
        model.log = lambda *args, **kwargs: None
        loss = model._combine_and_log_ocean_losses(
            prefix="train",
            loss_ambient=torch.tensor(1.0),
            diffusion_context={
                "x0_pred": torch.zeros(1, 1, 1, 1),
                "t": torch.tensor([1]),
            },
            target=torch.zeros(1, 1, 1, 1),
            model_batch={"x": torch.zeros(1, 1, 1, 1)},
            land_mask=None,
            on_step=True,
            on_epoch=True,
        )

        self.assertTrue(torch.isclose(loss, torch.tensor(1.0)))
        self.assertTrue(torch.equal(captured["t"], torch.tensor([1])))
        self.assertTrue(
            torch.equal(
                captured["alphas_cumprod"], model.model.forward_process.alphas_cumprod
            )
        )

    def test_p_loss_return_context_preserves_default_scalar_behavior(self) -> None:
        process = DenoisingDiffusionConditionalProcess(
            generated_channels=1,
            condition_channels=1,
            parameterization="x0",
            num_timesteps=2,
            unet_dim=8,
            unet_dim_mults=(1,),
        )
        output = torch.ones(1, 1, 2, 2)
        condition = torch.zeros_like(output)
        prediction = torch.full_like(output, 2.0)
        process.forward_process = _FakeForward(noise=torch.zeros_like(output))
        process.model = _StaticPredictor(prediction)

        scalar = process.p_loss(output, condition)
        contextual, context = process.p_loss(output, condition, return_context=True)

        self.assertTrue(torch.is_tensor(scalar))
        self.assertTrue(torch.isclose(scalar, contextual))
        self.assertTrue(torch.equal(context["x0_pred"], prediction))

    def test_auxiliary_losses_reject_joint_output_fields(self) -> None:
        with self.assertRaises(ValueError):
            PixelDiffusionConditional(
                generated_channels=4,
                condition_channels=5,
                output_fields=["temperature", "salinity"],
                condition_include_eo=False,
                parameterization="x0",
                num_timesteps=2,
                unet_dim=8,
                unet_dim_mults=(1,),
                losses_config={"increment": {"enabled": True}},
            )

    def test_pixel_diffusion_from_config_wires_loss_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model.yaml"
            data_path = tmp_path / "data.yaml"
            training_path = tmp_path / "training.yaml"
            _write_yaml(
                model_path,
                {
                    "model": {
                        "generated_channels": 2,
                        "condition_channels": 3,
                        "condition_mask_channels": 1,
                        "condition_include_eo": False,
                        "condition_use_valid_mask": True,
                        "parameterization": "x0",
                        "unet": {"dim": 8, "dim_mults": [1]},
                        "losses": {
                            "sparse_observation": {"enabled": True, "weight": 1.25},
                            "increment": {"enabled": True, "weight": 0.75},
                        },
                    }
                },
            )
            _write_yaml(data_path, {"dataset": {"grid": {"tile_size": 8}}})
            _write_yaml(
                training_path,
                {
                    "training": {
                        "noise": {"num_timesteps": 2, "schedule": "linear"},
                        "validation_sampling": {
                            "sampler": "ddim",
                            "ddim_num_timesteps": 2,
                        },
                    },
                    "wandb": {"verbose": False},
                },
            )

            model = PixelDiffusionConditional.from_config(
                str(model_path), str(data_path), str(training_path)
            )

        self.assertTrue(model.ocean_loss.any_extra_enabled())
        self.assertEqual(model.ocean_loss.obs_cfg["weight"], 1.25)
        self.assertEqual(model.ocean_loss.increment_cfg["weight"], 0.75)


if __name__ == "__main__":
    unittest.main()
