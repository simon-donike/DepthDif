from __future__ import annotations

import unittest

import torch
from torch import nn

from depth_recon.models.diffusion.DenoisingDiffusionProcess.DenoisingDiffusionProcess import (
    DenoisingDiffusionConditionalProcess,
)


class _ConstantPredictor(nn.Module):
    """Return a fixed prediction while recording every denoiser input."""

    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = prediction
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        model_input: torch.Tensor,
        timestep: torch.Tensor,
        coord_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = timestep, coord_emb
        self.inputs.append(model_input.detach().clone())
        return self.prediction.expand(model_input.size(0), -1, -1, -1)


class _IdentitySampler(nn.Module):
    """Sampler that exposes whether each reverse state was masked."""

    num_timesteps = 2
    temperature = 0.0

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        _ = timestep
        return x_t + prediction


class _CapturingForward(nn.Module):
    """Deterministic forward process used to inspect p_loss support."""

    num_timesteps = 1

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        output: torch.Tensor,
        timestep: torch.Tensor,
        return_noise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _ = timestep
        self.inputs.append(output.detach().clone())
        noise = torch.ones_like(output)
        if return_noise:
            return output + noise, noise
        return output + noise


def _make_process() -> DenoisingDiffusionConditionalProcess:
    return DenoisingDiffusionConditionalProcess(
        generated_channels=1,
        condition_channels=1,
        parameterization="epsilon",
        num_timesteps=2,
        unet_dim=8,
        unet_dim_mults=(1,),
    )


class TestDiffusionWetDomain(unittest.TestCase):
    def test_sampling_masks_initial_and_reverse_states_and_known_values(self) -> None:
        process = _make_process()
        predictor = _ConstantPredictor(torch.full((1, 1, 2, 2), 7.0))
        process.model = predictor
        wet_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        known_mask = torch.ones(1, 1, 2, 2)
        known_values = torch.full((1, 1, 2, 2), 3.0)
        known_values[:, :, 0, 1] = float("nan")

        output, intermediates = process(
            torch.zeros(1, 1, 2, 2),
            sampler=_IdentitySampler(),
            known_mask=known_mask,
            known_values=known_values,
            wet_mask=wet_mask,
            return_intermediates=True,
            intermediate_step_indices=[0, 1, 2],
        )

        self.assertTrue(torch.isfinite(output).all())
        output_band = output[:, 0]
        self.assertTrue(torch.equal(output_band[wet_mask == 0], torch.zeros(2)))
        self.assertTrue(torch.equal(output_band[wet_mask == 1], torch.full((2,), 3.0)))
        self.assertTrue(all(torch.isfinite(value).all() for _, value in intermediates))
        self.assertTrue(
            all(
                torch.equal(value[:, 0][wet_mask == 0], torch.zeros(2))
                for _, value in intermediates
            )
        )
        # The predictor can emit arbitrary dry values, but dry generated state is zeroed.
        self.assertTrue(
            all(
                torch.equal(inp[:, :1][..., 0, 1], torch.zeros(1, 1))
                for inp in predictor.inputs
            )
        )

    def test_training_masks_nonfinite_dry_values_before_denoiser(self) -> None:
        process = _make_process()
        predictor = _ConstantPredictor(torch.zeros(1, 1, 2, 2))
        process.model = predictor
        forward = _CapturingForward()
        process.forward_process = forward
        output = torch.tensor([[[[2.0, float("nan")], [4.0, float("nan")]]]])
        condition = torch.zeros_like(output)
        wet_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

        loss, context = process.p_loss(
            output, condition, wet_mask=wet_mask, return_context=True
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            torch.equal(forward.inputs[0][:, 0][wet_mask == 0], torch.zeros(2))
        )
        self.assertTrue(
            torch.equal(context["x_t"][:, 0][wet_mask == 0], torch.zeros(2))
        )
        self.assertTrue(
            torch.equal(context["target"][:, 0][wet_mask == 0], torch.zeros(2))
        )

    def test_all_wet_mask_matches_legacy_sampling_and_loss(self) -> None:
        process = _make_process()
        predictor = _ConstantPredictor(torch.zeros(1, 1, 2, 2))
        process.model = predictor
        condition = torch.zeros(1, 1, 2, 2)
        torch.manual_seed(11)
        legacy = process(condition, sampler=_IdentitySampler())
        torch.manual_seed(11)
        masked = process(
            condition,
            sampler=_IdentitySampler(),
            wet_mask=torch.ones(1, 2, 2),
        )
        self.assertTrue(torch.equal(legacy, masked))

        output = torch.ones(1, 1, 2, 2)
        torch.manual_seed(12)
        legacy_loss = process.p_loss(output, condition)
        torch.manual_seed(12)
        masked_loss = process.p_loss(output, condition, wet_mask=torch.ones(1, 2, 2))
        self.assertTrue(torch.equal(legacy_loss, masked_loss))

    def test_wet_mask_rejects_invalid_shapes_and_channels(self) -> None:
        process = _make_process()
        condition = torch.zeros(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            process(condition, sampler=_IdentitySampler(), wet_mask=torch.ones(2, 2))
        with self.assertRaises(ValueError):
            process(
                condition,
                sampler=_IdentitySampler(),
                wet_mask=torch.ones(2, 2, 2),
            )
        with self.assertRaises(ValueError):
            process(
                condition,
                sampler=_IdentitySampler(),
                wet_mask=torch.ones(1, 2, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
