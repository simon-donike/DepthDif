from __future__ import annotations

import os
import unittest

import matplotlib
import numpy as np
import torch
from torch import nn

from models.difFF.DenoisingDiffusionProcess.DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
)
from models.difFF.DenoisingDiffusionProcess.samplers import DDIM_Sampler
from models.difFF.DenoisingDiffusionProcess.beta_schedules import (
    cosine_beta_schedule,
    get_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)
from models.difFF.PixelDiffusion import PixelDiffusionConditional

matplotlib.use("Agg")
os.environ.setdefault("WANDB_MODE", "disabled")


class _FakeForward(nn.Module):
    def __init__(self, noisy_offset: float, noise: torch.Tensor) -> None:
        super().__init__()
        self.noisy_offset = float(noisy_offset)
        self.noise = noise
        self.num_timesteps = 5

    def forward(
        self,
        output: torch.Tensor,
        t: torch.Tensor,
        return_noise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _ = t
        noisy = output + self.noisy_offset
        if return_noise:
            return noisy, self.noise.to(device=output.device, dtype=output.dtype)
        return noisy


class _CapturingPredictor(nn.Module):
    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = prediction
        self.last_model_input: torch.Tensor | None = None

    def forward(
        self,
        model_input: torch.Tensor,
        t: torch.Tensor,
        coord_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (t, coord_emb)
        self.last_model_input = model_input.detach().clone()
        return self.prediction.to(device=model_input.device, dtype=model_input.dtype)


def _make_conditional_process(
    *, parameterization: str = "x0"
) -> DenoisingDiffusionConditionalProcess:
    return DenoisingDiffusionConditionalProcess(
        generated_channels=2,
        condition_channels=2,
        parameterization=parameterization,
        num_timesteps=4,
        schedule="linear",
        unet_dim=8,
        unet_dim_mults=(1,),
    )


def _make_pixel_model(**overrides: object) -> PixelDiffusionConditional:
    kwargs = dict(
        generated_channels=2,
        condition_channels=3,
        condition_mask_channels=1,
        condition_include_eo=False,
        condition_use_valid_mask=True,
        mask_loss_with_valid_pixels=True,
        parameterization="x0",
        num_timesteps=2,
        noise_schedule="linear",
        unet_dim=8,
        unet_dim_mults=(1,),
        wandb_verbose=False,
        log_intermediates=False,
        val_inference_sampler="ddim",
        val_ddim_num_timesteps=2,
        max_full_reconstruction_samples=1,
    )
    kwargs.update(overrides)
    return PixelDiffusionConditional(**kwargs)


class TestDiffusionMath(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

    def test_beta_schedule_variants_match_their_implementations(self) -> None:
        timesteps = 8
        beta_start = 1.0e-4
        beta_end = 2.0e-2

        linear = get_beta_schedule(
            "linear", timesteps, beta_start=beta_start, beta_end=beta_end
        )
        quadratic = get_beta_schedule(
            "quadratic", timesteps, beta_start=beta_start, beta_end=beta_end
        )
        sigmoid = get_beta_schedule(
            "sigmoid", timesteps, beta_start=beta_start, beta_end=beta_end
        )
        cosine = get_beta_schedule(
            "cosine", timesteps, beta_start=beta_start, beta_end=beta_end
        )

        self.assertTrue(
            torch.allclose(
                linear, linear_beta_schedule(timesteps, beta_start, beta_end)
            )
        )
        self.assertTrue(
            torch.allclose(
                quadratic,
                quadratic_beta_schedule(timesteps, beta_start, beta_end),
            )
        )
        self.assertTrue(
            torch.allclose(
                sigmoid, sigmoid_beta_schedule(timesteps, beta_start, beta_end)
            )
        )
        # Cosine uses its own clipping defaults and intentionally ignores the passed linear range.
        self.assertTrue(
            torch.allclose(
                cosine,
                cosine_beta_schedule(timesteps, beta_start=None, beta_end=None),
            )
        )
        self.assertEqual(linear.shape, (timesteps,))
        self.assertTrue(torch.all(linear[1:] >= linear[:-1]))
        self.assertTrue(torch.all(quadratic[1:] >= quadratic[:-1]))
        self.assertTrue(torch.all(sigmoid[1:] >= sigmoid[:-1]))
        self.assertTrue(torch.all(cosine > 0.0))
        self.assertTrue(torch.all(cosine < 1.0))

    def test_beta_schedule_rejects_invalid_ranges_and_unknown_variants(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid beta range"):
            get_beta_schedule("linear", 4, beta_start=0.1, beta_end=0.05)

        with self.assertRaisesRegex(ValueError, "Unknown beta schedule"):
            get_beta_schedule("unknown", 4)

    def test_ddim_temperature_scales_initial_sampling_noise(self) -> None:
        process = DenoisingDiffusionConditionalProcess(
            generated_channels=1,
            condition_channels=1,
            parameterization="x0",
            num_timesteps=4,
            schedule="linear",
            unet_dim=8,
            unet_dim_mults=(1,),
        )
        prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        capturing_model = _CapturingPredictor(prediction)
        process.model = capturing_model
        condition = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        sampler = DDIM_Sampler(
            num_timesteps=1,
            train_timesteps=4,
            betas=process.forward_process.betas.detach().clone(),
            parameterization="x0",
            clip_sample=False,
            eta=0.0,
            temperature=0.0,
        )

        _ = process(condition, sampler=sampler)

        assert capturing_model.last_model_input is not None
        noisy_branch = capturing_model.last_model_input[:, :1]
        self.assertTrue(torch.equal(noisy_branch, torch.zeros_like(noisy_branch)))

    def test_ddim_temperature_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature must be >= 0.0"):
            DDIM_Sampler(temperature=-0.1)

    def test_build_valid_mask_aligns_channels_and_inverts_missing_mode(self) -> None:
        reference = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
        mask = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )

        observed = DenoisingDiffusionConditionalProcess._build_valid_mask(
            mask,
            reference,
            mode="observed",
        )
        missing = DenoisingDiffusionConditionalProcess._build_valid_mask(
            mask,
            reference,
            mode="missing",
        )

        expected = mask.unsqueeze(1).expand_as(reference)
        self.assertTrue(torch.equal(observed, expected))
        self.assertTrue(torch.equal(missing, 1.0 - expected))

    def test_build_ambient_further_valid_mask_keeps_subset_and_respects_minimum_pixels(
        self,
    ) -> None:
        model = _make_pixel_model(
            ambient_occlusion_enabled=True,
            ambient_further_drop_prob=1.0,
            ambient_shared_spatial_mask=True,
            ambient_min_kept_observed_pixels=2,
        )
        valid_mask = torch.ones((1, 2, 3, 3), dtype=torch.float32)
        reference = torch.zeros((1, 2, 3, 3), dtype=torch.float32)

        further = model._build_ambient_further_valid_mask(
            valid_mask, reference=reference
        )

        self.assertIsNotNone(further)
        assert further is not None
        self.assertEqual(further.shape, valid_mask.shape)
        self.assertTrue(torch.all(further <= valid_mask))
        # The min-kept safeguard is enforced over the full flattened observed support.
        self.assertGreaterEqual(int(further.sum().item()), 2)

    def test_build_task_supervision_mask_switches_between_standard_and_ambient_targets(
        self,
    ) -> None:
        reference = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        x_valid_mask = torch.tensor(
            [[[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]
        )
        y_valid_mask = torch.tensor(
            [[[[1.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]]]
        )

        standard_model = _make_pixel_model(ambient_occlusion_enabled=False)
        ambient_model = _make_pixel_model(ambient_occlusion_enabled=True)

        standard_mask = standard_model._build_task_supervision_mask(
            reference=reference,
            x_valid_mask=x_valid_mask,
            y_valid_mask=y_valid_mask,
        )
        ambient_mask = ambient_model._build_task_supervision_mask(
            reference=reference,
            x_valid_mask=x_valid_mask,
            y_valid_mask=y_valid_mask,
        )

        self.assertTrue(torch.equal(standard_mask, (y_valid_mask > 0.5).float()))
        self.assertTrue(
            torch.equal(
                ambient_mask, ((x_valid_mask > 0.5) & (y_valid_mask > 0.5)).float()
            )
        )

    def test_p_loss_averages_only_over_supervised_ocean_pixels(self) -> None:
        process = _make_conditional_process(parameterization="x0")
        output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        condition = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        prediction = output - 1.0
        loss_mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
        land_mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])

        process.forward_process = _FakeForward(
            noisy_offset=0.25, noise=torch.zeros_like(output)
        )
        process.model = _CapturingPredictor(prediction)

        loss = process.p_loss(
            output,
            condition,
            loss_mask=loss_mask,
            land_mask=land_mask,
            mask_loss=True,
        )

        expected_mask = loss_mask.unsqueeze(1).expand_as(output) * land_mask.unsqueeze(
            1
        ).expand_as(output)
        expected = (
            ((output - prediction) ** 2) * expected_mask
        ).sum() / expected_mask.sum()
        self.assertTrue(torch.isclose(loss, expected))

    def test_p_loss_returns_zero_when_mask_selects_nothing(self) -> None:
        process = _make_conditional_process(parameterization="x0")
        output = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        condition = torch.zeros((1, 2, 2, 2), dtype=torch.float32)

        process.forward_process = _FakeForward(
            noisy_offset=0.0, noise=torch.zeros_like(output)
        )
        process.model = _CapturingPredictor(torch.zeros_like(output))

        loss = process.p_loss(
            output,
            condition,
            loss_mask=torch.zeros((1, 2, 2), dtype=torch.float32),
            land_mask=torch.zeros((1, 2, 2), dtype=torch.float32),
            mask_loss=True,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_p_loss_applies_ambient_further_mask_to_the_noisy_branch(self) -> None:
        process = _make_conditional_process(parameterization="x0")
        output = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        condition = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        further_valid_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        process.forward_process = _FakeForward(
            noisy_offset=4.0,
            noise=torch.zeros_like(output),
        )
        capturing_model = _CapturingPredictor(torch.zeros_like(output))
        process.model = capturing_model

        _ = process.p_loss(
            output,
            condition,
            further_valid_mask=further_valid_mask,
            apply_further_corruption_to_noisy_branch=True,
        )

        assert capturing_model.last_model_input is not None
        noisy_branch = capturing_model.last_model_input[:, : output.size(1)]
        expected_mask = further_valid_mask.unsqueeze(1).expand_as(output)
        self.assertTrue(torch.equal(noisy_branch, (output + 4.0) * expected_mask))


if __name__ == "__main__":
    unittest.main()
