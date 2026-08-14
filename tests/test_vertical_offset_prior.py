from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from depth_recon.data.synthetic_dataset_creation.vertical_offset_prior import (
    VerticalOffsetAccumulator,
    VerticalOffsetPrior,
)


def write_prior_artifact(
    path: Path,
    *,
    depth_axis_m: np.ndarray = np.asarray([0.0, 10.0], dtype=np.float32),
    temperature_offset_c: np.ndarray | None = None,
    salinity_offset_psu: np.ndarray | None = None,
    supervision_weight: np.ndarray | None = None,
    max_supervised_depth_m: float = 1000.0,
) -> Path:
    """Write a compact deterministic prior fixture."""
    depth_axis = np.asarray(depth_axis_m, dtype=np.float32)
    if temperature_offset_c is None:
        temperature_offset_c = -0.5 * np.arange(len(depth_axis), dtype=np.float32)
    if salinity_offset_psu is None:
        salinity_offset_psu = 0.1 * np.arange(len(depth_axis), dtype=np.float32)
    if supervision_weight is None:
        supervision_weight = np.ones((len(depth_axis), 2), dtype=np.float32)
    prior = VerticalOffsetPrior(
        depth_axis_m=depth_axis,
        temperature_offset_c=temperature_offset_c,
        salinity_offset_psu=salinity_offset_psu,
        supervision_weight=supervision_weight,
        max_supervised_depth_m=max_supervised_depth_m,
        fit_years=(2010, 2011),
        excluded_years=(2018,),
        provenance={"source": "unit-test"},
    )
    return prior.to_npz(path)


class TestVerticalOffsetPrior(unittest.TestCase):
    def test_surface_spatial_pattern_is_identical_at_every_depth(self) -> None:
        """Depth targets must differ from the surface only by one scalar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prior = VerticalOffsetPrior.from_npz(
                write_prior_artifact(Path(tmpdir) / "prior.npz")
            )
            sst = np.asarray([[280.0, 281.0], [282.0, 284.0]], dtype=np.float32)
            sss = np.asarray([[34.0, 34.5], [35.0, 36.0]], dtype=np.float32)
            sample = prior.sample(
                {"sst": sst, "sss": sss},
                depth_valid_mask=np.ones((2, 2, 2), dtype=bool),
            )
            np.testing.assert_allclose(sample.temperature_k[0], sst)
            np.testing.assert_allclose(sample.temperature_k[1], sst - 0.5)
            np.testing.assert_allclose(sample.salinity_psu[0], sss)
            np.testing.assert_allclose(sample.salinity_psu[1], sss + 0.1)
            np.testing.assert_allclose(
                sample.temperature_k[1] - sample.temperature_k[1, 0, 0],
                sst - sst[0, 0],
            )

    def test_masks_cutoff_and_sparse_anchors_are_applied(self) -> None:
        """Bathymetry, supervision depth, and real observations remain authoritative."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prior = VerticalOffsetPrior.from_npz(
                write_prior_artifact(
                    Path(tmpdir) / "prior.npz", max_supervised_depth_m=5.0
                )
            )
            valid = np.ones((2, 2, 2), dtype=bool)
            valid[1, 0, 0] = False
            anchors = np.full((2, 2, 2), np.nan, dtype=np.float32)
            anchors[1, 1, 1] = 275.0
            sample = prior.sample(
                {
                    "sst": np.full((2, 2), 280.0, dtype=np.float32),
                    "sss": np.full((2, 2), 35.0, dtype=np.float32),
                },
                depth_valid_mask=valid,
                temperature_anchors=anchors,
            )
            self.assertEqual(float(sample.temperature_k[1, 1, 1]), 275.0)
            self.assertEqual(float(sample.temperature_k[1, 0, 0]), 0.0)
            np.testing.assert_array_equal(sample.temperature_supervision_weight[1], 0.0)

    def test_accumulator_fits_mean_depth_minus_surface_offsets(self) -> None:
        """Fitting must average GLORYS deltas rather than absolute fields."""
        accumulator = VerticalOffsetAccumulator(
            depth_axis_m=np.asarray([0.0, 50.0, 100.0]),
            excluded_years=(2018,),
        )
        surface_t = np.asarray([[20.0, 22.0], [24.0, 26.0]])
        surface_s = np.asarray([[34.0, 35.0], [36.0, 37.0]])
        for date, scale in ((20170115, 1.0), (20190115, 2.0)):
            accumulator.update(
                temperature_c=np.stack(
                    (surface_t, surface_t - 2.0 * scale, surface_t - 4.0 * scale)
                ),
                salinity_psu=np.stack(
                    (surface_s, surface_s + 0.1 * scale, surface_s + 0.2 * scale)
                ),
                date=date,
            )
        prior = accumulator.finalize(max_supervised_depth_m=100.0)
        np.testing.assert_allclose(prior.temperature_offset_c, (0.0, -3.0, -6.0))
        np.testing.assert_allclose(prior.salinity_offset_psu, (0.0, 0.15, 0.3))
        self.assertEqual(prior.fit_years, (2017, 2019))

    def test_round_trip_preserves_artifact_contract(self) -> None:
        """The pickle-free artifact must preserve offsets and provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_prior_artifact(Path(tmpdir) / "prior.npz")
            loaded = VerticalOffsetPrior.from_npz(
                path, expected_depth_axis_m=np.asarray([0.0, 10.0])
            )
            np.testing.assert_allclose(loaded.temperature_offset_c, (0.0, -0.5))
            self.assertEqual(loaded.excluded_years, (2018,))
            self.assertEqual(loaded.provenance, {"source": "unit-test"})


if __name__ == "__main__":
    unittest.main()
