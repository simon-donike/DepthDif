from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from depth_recon.models.diffusion.PixelDiffusion import PixelDiffusionConditional
from depth_recon.utils.normalizations import temperature_normalize


def _model(**overrides: object) -> PixelDiffusionConditional:
    settings: dict[str, object] = {
        "generated_channels": 2,
        "condition_channels": 3,
        "condition_mask_channels": 1,
        "condition_use_valid_mask": True,
        "parameterization": "x0",
        "num_timesteps": 2,
        "unet_dim": 8,
        "unet_dim_mults": (1,),
        "full_reconstruction_logging_enabled": False,
        "log_intermediates": False,
        "wandb_verbose": False,
    }
    settings.update(overrides)
    return PixelDiffusionConditional(**settings)


def _field_batch(channels: int = 2) -> dict[str, torch.Tensor]:
    shape = (1, channels, 2, 2)
    return {
        "x": torch.ones(shape),
        "y": torch.ones(shape) * 2.0,
        "x_valid_mask": torch.ones(shape),
        "y_valid_mask": torch.ones(shape),
    }


class _CaptureOceanLoss(nn.Module):
    """Capture the absolute-field context passed to auxiliary losses."""

    def __init__(self) -> None:
        super().__init__()
        self.x0_pred: torch.Tensor | None = None

    def any_extra_enabled(self) -> bool:
        return True

    def forward(
        self, **kwargs: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.x0_pred = kwargs["x0_pred"]
        loss = kwargs["loss_ambient"]
        assert loss is not None
        return loss, {}


class TestPixelDepthContext(unittest.TestCase):
    def test_depth_aware_residual_model_runs_forward_backward_and_sampling(
        self,
    ) -> None:
        """Exercise the actual denoiser with wet masks and residuals together."""
        model = _model(
            condition_channels=8,
            condition_mask_channels=2,
            condition_per_depth_valid_mask=True,
            condition_use_wet_mask=True,
            mask_diffusion_with_wet_mask=True,
            mask_loss_with_valid_pixels=True,
            climatology_residual=True,
            clamp_known_pixels=False,
        )
        shape = (1, 2, 8, 8)
        wet = torch.ones(shape, dtype=torch.bool)
        wet[:, 1, :, -2:] = False
        observed = wet.clone()
        observed[:, :, ::2, ::2] = False
        batch = {
            "x": torch.where(observed, 0.1, 0.0),
            "y": torch.where(wet, 0.3, 0.0),
            "x_valid_mask": observed,
            "y_valid_mask": wet,
            "wet_mask": wet,
            "climatology": torch.full(shape, 0.2),
        }
        with patch.object(model, "log"):
            loss = model.training_step(batch, batch_idx=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        result = model.eval().predict_step(batch, batch_idx=0)
        self.assertTrue(torch.isfinite(result["y_hat_denorm"][wet]).all())
        self.assertTrue(torch.isnan(result["y_hat_denorm"][~wet]).all())

    def test_per_depth_condition_mask_requires_matching_mask_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "one mask per output channel"):
            _model(condition_per_depth_valid_mask=True, condition_mask_channels=1)

        model = _model(
            condition_per_depth_valid_mask=True,
            condition_mask_channels=2,
            condition_channels=4,
        )
        x = torch.zeros(1, 2, 2, 2)
        valid = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 1.0], [0.0, 1.0]]]])
        valid = valid.permute(1, 0, 2, 3)
        condition = model._prepare_condition_for_model(x, valid)
        self.assertEqual(tuple(condition.shape), (1, 4, 2, 2))
        self.assertTrue(torch.equal(condition[:, 2:], valid))

    def test_wet_mask_and_climatology_are_per_depth_and_dry_background_is_zero(
        self,
    ) -> None:
        model = _model(
            condition_use_wet_mask=True,
            climatology_residual=True,
            condition_channels=7,
        )
        batch = _field_batch()
        batch["wet_mask"] = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]
        )
        batch["climatology"] = torch.full((1, 2, 2, 2), 4.0)
        batch["y_valid_mask"] = batch["wet_mask"].clone()
        wet, background = model._prepare_depth_context(batch, batch["x"])
        self.assertTrue(torch.equal(wet, batch["wet_mask"].bool()))
        self.assertTrue(torch.equal(background[~wet], torch.zeros(4)))
        condition = model._prepare_condition_for_model(
            batch["x"], batch["x_valid_mask"], wet_mask=wet, background=background
        )
        self.assertEqual(tuple(condition.shape), (1, 7, 2, 2))

    def test_prediction_residual_background_is_added_once(self) -> None:
        model = _model(
            condition_channels=5,
            climatology_residual=True,
            clamp_known_pixels=False,
        )
        batch = _field_batch()
        background = torch.full_like(batch["x"], 0.25)
        batch["climatology"] = background
        residual = torch.full_like(batch["x"], 0.5)
        with patch.object(model.model, "forward", return_value=residual):
            result = model.predict_step(batch, batch_idx=0)
        expected = temperature_normalize(mode="denorm", tensor=residual + background)
        self.assertTrue(torch.allclose(result["y_hat_denorm"], expected))

    def test_training_residual_target_is_normalized_absolute_error(self) -> None:
        model = _model(condition_channels=5, climatology_residual=True)
        batch = _field_batch()
        background = torch.full_like(batch["x"], 0.25)
        batch["climatology"] = background
        captured: dict[str, torch.Tensor] = {}

        def fake_p_loss(
            target: torch.Tensor, condition: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            _ = condition, kwargs
            captured["target"] = target.detach()
            return target.square().mean()

        with (
            patch.object(model.model, "p_loss", side_effect=fake_p_loss),
            patch.object(model, "log", return_value=None),
        ):
            loss = model.training_step(batch, batch_idx=0)
        expected_target = batch["y"] - background
        self.assertTrue(torch.equal(captured["target"], expected_target))
        self.assertTrue(torch.equal(loss, expected_target.square().mean()))

    def test_auxiliary_loss_context_restores_absolute_background_once(self) -> None:
        model = _model(condition_channels=5, climatology_residual=True)
        batch = _field_batch()
        background = torch.full_like(batch["x"], 0.25)
        batch["climatology"] = background
        residual = torch.full_like(batch["x"], 0.5)
        capture = _CaptureOceanLoss()
        model.ocean_loss = capture

        def fake_p_loss(
            target: torch.Tensor, condition: torch.Tensor, **kwargs: object
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            _ = condition, kwargs
            return target.square().mean(), {"x0_pred": residual.clone()}

        with (
            patch.object(model.model, "p_loss", side_effect=fake_p_loss),
            patch.object(model, "log", return_value=None),
        ):
            model.training_step(batch, batch_idx=0)
        assert capture.x0_pred is not None
        self.assertTrue(torch.equal(capture.x0_pred, residual + background))

    def test_joint_output_stacks_temperature_then_salinity(self) -> None:
        model = _model(
            generated_channels=4,
            condition_channels=5,
            output_fields=("temperature", "salinity"),
            condition_mask_channels=1,
        )
        batch = _field_batch(channels=2)
        batch["x_salinity"] = torch.full_like(batch["x"], 9.0)
        stacked = model._stack_output_tensor(
            batch, temperature_key="x", salinity_key="x_salinity"
        )
        self.assertTrue(torch.equal(stacked[:, :2], batch["x"]))
        self.assertTrue(torch.equal(stacked[:, 2:], batch["x_salinity"]))

    def test_validation_cache_retains_depth_context_and_none_is_legacy_compatible(
        self,
    ) -> None:
        model = _model()
        batch = _field_batch()
        batch["wet_mask"] = torch.ones_like(batch["x"])
        batch["climatology"] = torch.zeros_like(batch["x"])
        model._cache_validation_batch(batch, n_cache=1)
        self.assertTrue(
            torch.equal(model._cached_val_example["wet_mask"], batch["wet_mask"])
        )
        self.assertTrue(
            torch.equal(model._cached_val_example["climatology"], batch["climatology"])
        )

        condition_without_context = model._prepare_condition_for_model(
            batch["x"], batch["x_valid_mask"]
        )
        self.assertEqual(tuple(condition_without_context.shape), (1, 3, 2, 2))

    def test_valid_target_outside_wet_domain_fails_fast(self) -> None:
        model = _model(condition_use_wet_mask=True)
        batch = _field_batch()
        batch["wet_mask"] = torch.zeros_like(batch["x"])
        with self.assertRaisesRegex(
            ValueError, "outside the configured static wet domain"
        ):
            model._prepare_depth_context(batch, batch["x"])


if __name__ == "__main__":
    unittest.main()
