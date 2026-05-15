from __future__ import annotations

import unittest

import torch

from depth_recon.utils.normalizations import salinity_normalize


class TestNormalizations(unittest.TestCase):
    def test_salinity_normalize_round_trips_psu_values(self) -> None:
        values = torch.tensor([30.0, 34.54260282159372, 40.0], dtype=torch.float32)

        normalized = salinity_normalize(mode="norm", tensor=values)
        denormalized = salinity_normalize(mode="denorm", tensor=normalized)

        self.assertTrue(torch.allclose(denormalized, values, atol=1.0e-5))

    def test_salinity_normalize_rejects_invalid_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be 'norm' or 'denorm'"):
            salinity_normalize(mode="bad", tensor=torch.tensor([35.0]))


if __name__ == "__main__":
    unittest.main()
