"""Deterministic dense targets formed by depth-wise offsets from surface maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PriorSample:
    """One deterministic physical-unit target and its supervision support."""

    temperature_k: np.ndarray
    salinity_psu: np.ndarray
    valid_mask: np.ndarray
    temperature_supervision_weight: np.ndarray
    salinity_supervision_weight: np.ndarray


class VerticalOffsetPrior:
    """Copy surface structure to depth and add one fitted scalar per level."""

    def __init__(
        self,
        *,
        depth_axis_m: np.ndarray,
        temperature_offset_c: np.ndarray,
        salinity_offset_psu: np.ndarray,
        supervision_weight: np.ndarray,
        max_supervised_depth_m: float,
        fit_years: Sequence[int] = (),
        excluded_years: Sequence[int] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate a fitted depth-offset artifact."""
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        self.temperature_offset_c = np.asarray(
            temperature_offset_c, dtype=np.float32
        ).reshape(-1)
        self.salinity_offset_psu = np.asarray(
            salinity_offset_psu, dtype=np.float32
        ).reshape(-1)
        self.supervision_weight = np.asarray(supervision_weight, dtype=np.float32)
        self.max_supervised_depth_m = float(max_supervised_depth_m)
        self.fit_years = tuple(int(value) for value in fit_years)
        self.excluded_years = tuple(int(value) for value in excluded_years)
        self.provenance = dict(provenance or {})

        depth_count = int(self.depth_axis_m.size)
        if depth_count == 0 or np.any(~np.isfinite(self.depth_axis_m)):
            raise ValueError("depth_axis_m must be a non-empty finite vector.")
        if np.any(np.diff(self.depth_axis_m) < 0.0):
            raise ValueError("depth_axis_m must be monotonic increasing.")
        for name in ("temperature_offset_c", "salinity_offset_psu"):
            values = getattr(self, name)
            if values.shape != (depth_count,) or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must be a finite depth vector.")
        if self.supervision_weight.shape != (depth_count, 2):
            raise ValueError("supervision_weight must be shaped (depth,2).")
        if np.any(~np.isfinite(self.supervision_weight)):
            raise ValueError("supervision_weight must be finite.")

        # The first band is the observed surface itself, not an estimated offset.
        self.temperature_offset_c[0] = 0.0
        self.salinity_offset_psu[0] = 0.0

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_depth_axis_m: np.ndarray | None = None,
    ) -> "VerticalOffsetPrior":
        """Load and validate a pickle-free depth-offset artifact."""
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Vertical-offset prior not found: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["schema_version"]).item())
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported vertical-offset schema {version}; "
                    f"expected {SCHEMA_VERSION}."
                )
            prior = cls(
                depth_axis_m=payload["depth_axis_m"],
                temperature_offset_c=payload["temperature_offset_c"],
                salinity_offset_psu=payload["salinity_offset_psu"],
                supervision_weight=payload["supervision_weight"],
                max_supervised_depth_m=float(
                    np.asarray(payload["max_supervised_depth_m"]).item()
                ),
                fit_years=payload.get("fit_years", np.asarray([], dtype=np.int16)),
                excluded_years=payload.get(
                    "excluded_years", np.asarray([], dtype=np.int16)
                ),
                provenance=json.loads(
                    str(np.asarray(payload.get("provenance_json", "{}")).item())
                ),
            )
        if expected_depth_axis_m is not None:
            expected = np.asarray(expected_depth_axis_m).reshape(-1)
            if expected.shape != prior.depth_axis_m.shape or not np.allclose(
                expected, prior.depth_axis_m, atol=1.0e-3, rtol=0.0
            ):
                raise ValueError("Prior depth axis does not match the dataset.")
        return prior

    def to_npz(self, path: str | Path) -> Path:
        """Write the fitted offsets and provenance to a compressed artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
            depth_axis_m=self.depth_axis_m,
            temperature_offset_c=self.temperature_offset_c,
            salinity_offset_psu=self.salinity_offset_psu,
            supervision_weight=self.supervision_weight,
            max_supervised_depth_m=np.asarray(
                self.max_supervised_depth_m, dtype=np.float32
            ),
            fit_years=np.asarray(self.fit_years, dtype=np.int16),
            excluded_years=np.asarray(self.excluded_years, dtype=np.int16),
            provenance_json=np.asarray(json.dumps(self.provenance, sort_keys=True)),
        )
        return output_path

    save = to_npz

    def sample(
        self,
        surface_fields: Mapping[str, np.ndarray],
        *,
        depth_valid_mask: np.ndarray,
        temperature_anchors: np.ndarray | None = None,
        salinity_anchors: np.ndarray | None = None,
        **_: Any,
    ) -> PriorSample:
        """Duplicate SST/SSS at every depth and add the fitted depth offsets."""
        if "sst" not in surface_fields or "sss" not in surface_fields:
            raise ValueError("Vertical-offset targets require surface SST and SSS.")
        sst = np.asarray(surface_fields["sst"], dtype=np.float32)
        sss = np.asarray(surface_fields["sss"], dtype=np.float32)
        if sst.ndim != 2 or sss.shape != sst.shape:
            raise ValueError("Surface SST and SSS must share shape (H,W).")
        depth_count = int(self.depth_axis_m.size)
        valid_mask = np.asarray(depth_valid_mask, dtype=bool)
        if valid_mask.shape != (depth_count, *sst.shape):
            raise ValueError(
                f"depth_valid_mask has shape {valid_mask.shape}; expected "
                f"{(depth_count, *sst.shape)}."
            )

        temperature = sst[None, ...] + self.temperature_offset_c[:, None, None]
        salinity = sss[None, ...] + self.salinity_offset_psu[:, None, None]
        for values, anchors in (
            (temperature, temperature_anchors),
            (salinity, salinity_anchors),
        ):
            if anchors is None:
                continue
            anchor_values = np.asarray(anchors, dtype=np.float32)
            if anchor_values.shape != valid_mask.shape:
                raise ValueError("Sparse anchors must be shaped (D,H,W).")
            anchor_mask = np.isfinite(anchor_values) & valid_mask
            values[anchor_mask] = anchor_values[anchor_mask]

        # Invalid payload values are zeroed; their masks and weights remain zero.
        surface_support = np.isfinite(sst) & np.isfinite(sss)
        target_valid = valid_mask & surface_support[None, ...]
        temperature = np.where(target_valid, temperature, 0.0)
        salinity = np.where(target_valid, salinity, 0.0)
        weights = np.broadcast_to(
            self.supervision_weight.T[:, :, None, None],
            (2, depth_count, *sst.shape),
        ).copy()
        weights[:, ~target_valid] = 0.0
        weights[:, self.depth_axis_m > self.max_supervised_depth_m] = 0.0
        return PriorSample(
            temperature_k=temperature.astype(np.float32),
            salinity_psu=salinity.astype(np.float32),
            valid_mask=target_valid,
            temperature_supervision_weight=weights[0].astype(np.float32),
            salinity_supervision_weight=weights[1].astype(np.float32),
        )


class VerticalOffsetAccumulator:
    """Stream GLORYS surface-relative depth offsets without retaining fields."""

    def __init__(
        self,
        *,
        depth_axis_m: np.ndarray,
        excluded_years: Sequence[int] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Allocate sufficient statistics for temperature and salinity offsets."""
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        depth_count = int(self.depth_axis_m.size)
        self.excluded_years = tuple(int(value) for value in excluded_years)
        self.provenance = dict(provenance or {})
        self.delta_sum = np.zeros((depth_count, 2), dtype=np.float64)
        self.delta_count = np.zeros((depth_count, 2), dtype=np.int64)
        self.fit_years: set[int] = set()
        self.patch_count = 0

    def update(
        self,
        *,
        temperature_c: np.ndarray,
        salinity_psu: np.ndarray,
        date: int,
    ) -> None:
        """Accumulate one GLORYS patch's depth-minus-surface differences."""
        year = int(date) // 10000
        if year in self.excluded_years:
            raise ValueError(f"Refusing to fit offsets with excluded year {year}.")
        temperature = np.asarray(temperature_c, dtype=np.float64)
        salinity = np.asarray(salinity_psu, dtype=np.float64)
        expected_depths = int(self.depth_axis_m.size)
        if temperature.shape != salinity.shape or temperature.ndim != 3:
            raise ValueError("GLORYS temperature and salinity must share (D,H,W).")
        if temperature.shape[0] != expected_depths:
            raise ValueError("GLORYS depth axis does not match the accumulator.")

        for field_index, field in enumerate((temperature, salinity)):
            surface = field[0]
            for depth_index in range(expected_depths):
                valid = np.isfinite(surface) & np.isfinite(field[depth_index])
                if not np.any(valid):
                    continue
                delta = field[depth_index][valid] - surface[valid]
                self.delta_sum[depth_index, field_index] += float(delta.sum())
                self.delta_count[depth_index, field_index] += int(delta.size)
        self.fit_years.add(year)
        self.patch_count += 1

    def finalize(self, *, max_supervised_depth_m: float) -> VerticalOffsetPrior:
        """Compute the per-depth mean offsets and coverage confidence."""
        if self.patch_count == 0:
            raise RuntimeError("No patches were accumulated for vertical offsets.")
        offsets = np.divide(
            self.delta_sum,
            self.delta_count,
            out=np.full_like(self.delta_sum, np.nan),
            where=self.delta_count > 0,
        )
        for field_index in range(2):
            valid = np.isfinite(offsets[:, field_index])
            if not np.any(valid):
                raise RuntimeError(
                    f"No valid GLORYS offsets for field index {field_index}."
                )
            # Unsampled abyssal levels are outside the supervised target range;
            # interpolate or edge-hold only to keep all 50 payload bands finite.
            offsets[:, field_index] = np.interp(
                self.depth_axis_m,
                self.depth_axis_m[valid],
                offsets[valid, field_index],
            )
        offsets[0] = 0.0
        # Relative valid-pixel coverage is a conservative depth loss weight.
        coverage = self.delta_count / np.maximum(self.delta_count[0:1], 1)
        return VerticalOffsetPrior(
            depth_axis_m=self.depth_axis_m,
            temperature_offset_c=offsets[:, 0],
            salinity_offset_psu=offsets[:, 1],
            supervision_weight=np.clip(coverage, 0.0, 1.0),
            max_supervised_depth_m=float(max_supervised_depth_m),
            fit_years=sorted(self.fit_years),
            excluded_years=self.excluded_years,
            provenance={**self.provenance, "selected_patch_count": self.patch_count},
        )
