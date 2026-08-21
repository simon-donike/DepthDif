"""Deterministic EO-surface targets with offline GLORYS depth deltas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
MONTH_COUNT = 12


@dataclass(frozen=True)
class PriorSample:
    """One deterministic physical-unit target and its supervision support."""

    temperature_k: np.ndarray
    salinity_psu: np.ndarray
    valid_mask: np.ndarray
    temperature_supervision_weight: np.ndarray
    salinity_supervision_weight: np.ndarray


class VerticalOffsetPrior:
    """Add global or smooth monthly GLORYS deltas to EO SST and SSS."""

    def __init__(
        self,
        *,
        depth_axis_m: np.ndarray,
        temperature_offset_c: np.ndarray,
        salinity_offset_psu: np.ndarray,
        supervision_weight: np.ndarray,
        max_supervised_depth_m: float,
        latitude_bin_centers_deg: np.ndarray | None = None,
        longitude_bin_centers_deg: np.ndarray | None = None,
        fit_years: Sequence[int] = (),
        excluded_years: Sequence[int] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate a legacy global or v2 monthly spatial prior artifact."""
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        self.temperature_offset_c = np.asarray(temperature_offset_c, dtype=np.float32)
        self.salinity_offset_psu = np.asarray(salinity_offset_psu, dtype=np.float32)
        self.supervision_weight = np.asarray(supervision_weight, dtype=np.float32)
        self.max_supervised_depth_m = float(max_supervised_depth_m)
        self.fit_years = tuple(int(value) for value in fit_years)
        self.excluded_years = tuple(int(value) for value in excluded_years)
        self.provenance = dict(provenance or {})
        depth_count = len(self.depth_axis_m)
        if depth_count == 0 or np.any(~np.isfinite(self.depth_axis_m)):
            raise ValueError("depth_axis_m must be a non-empty finite vector.")
        if np.any(np.diff(self.depth_axis_m) < 0.0):
            raise ValueError("depth_axis_m must be monotonic increasing.")
        if self.temperature_offset_c.shape != self.salinity_offset_psu.shape:
            raise ValueError("Temperature and salinity offsets must share a shape.")
        self.is_spatial = self.temperature_offset_c.ndim == 4
        self.latitude_bin_centers_deg: np.ndarray | None = None
        self.longitude_bin_centers_deg: np.ndarray | None = None
        if self.is_spatial:
            if latitude_bin_centers_deg is None or longitude_bin_centers_deg is None:
                raise ValueError("Spatial offsets require latitude and longitude bins.")
            self.latitude_bin_centers_deg = np.asarray(
                latitude_bin_centers_deg, dtype=np.float32
            ).reshape(-1)
            self.longitude_bin_centers_deg = np.asarray(
                longitude_bin_centers_deg, dtype=np.float32
            ).reshape(-1)
            shape = (
                MONTH_COUNT,
                len(self.latitude_bin_centers_deg),
                len(self.longitude_bin_centers_deg),
                depth_count,
            )
            if (
                self.temperature_offset_c.shape != shape
                or self.supervision_weight.shape != (*shape, 2)
            ):
                raise ValueError(
                    "Spatial offsets must be (month,lat,lon,depth) with matching weights."
                )
        elif self.temperature_offset_c.shape != (
            depth_count,
        ) or self.supervision_weight.shape != (depth_count, 2):
            raise ValueError(
                "Global offsets and weights must be shaped (depth,) and (depth,2)."
            )
        if (
            np.any(~np.isfinite(self.temperature_offset_c))
            or np.any(~np.isfinite(self.salinity_offset_psu))
            or np.any(~np.isfinite(self.supervision_weight))
        ):
            raise ValueError("Prior offsets and weights must be finite.")
        # EO supplies the surface, so no climatological offset is applied there.
        self.temperature_offset_c[..., 0] = 0.0
        self.salinity_offset_psu[..., 0] = 0.0

    @classmethod
    def from_npz(
        cls, path: str | Path, *, expected_depth_axis_m: np.ndarray | None = None
    ) -> "VerticalOffsetPrior":
        """Load a pickle-free v1 global or v2 spatial prior artifact."""
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Vertical-offset prior not found: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["schema_version"]).item())
            if version not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
                raise ValueError(f"Unsupported vertical-offset schema {version}.")
            prior = cls(
                depth_axis_m=payload["depth_axis_m"],
                temperature_offset_c=payload["temperature_offset_c"],
                salinity_offset_psu=payload["salinity_offset_psu"],
                supervision_weight=payload["supervision_weight"],
                max_supervised_depth_m=float(
                    np.asarray(payload["max_supervised_depth_m"]).item()
                ),
                latitude_bin_centers_deg=payload.get("latitude_bin_centers_deg"),
                longitude_bin_centers_deg=payload.get("longitude_bin_centers_deg"),
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
        """Write the prior as a compressed v2 artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "depth_axis_m": self.depth_axis_m,
            "temperature_offset_c": self.temperature_offset_c,
            "salinity_offset_psu": self.salinity_offset_psu,
            "supervision_weight": self.supervision_weight,
            "max_supervised_depth_m": np.asarray(
                self.max_supervised_depth_m, dtype=np.float32
            ),
            "fit_years": np.asarray(self.fit_years, dtype=np.int16),
            "excluded_years": np.asarray(self.excluded_years, dtype=np.int16),
            "provenance_json": np.asarray(json.dumps(self.provenance, sort_keys=True)),
        }
        if self.is_spatial:
            payload["latitude_bin_centers_deg"] = self.latitude_bin_centers_deg
            payload["longitude_bin_centers_deg"] = self.longitude_bin_centers_deg
        np.savez_compressed(output_path, **payload)
        return output_path

    save = to_npz

    def _offsets(
        self,
        *,
        date: int | None,
        latitude_deg: np.ndarray | None,
        longitude_deg: np.ndarray | None,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return global or dateline-continuous bilinear spatial offsets."""
        depth_count = len(self.depth_axis_m)
        if not self.is_spatial:
            return (
                np.broadcast_to(
                    self.temperature_offset_c[:, None, None], (depth_count, *shape)
                ),
                np.broadcast_to(
                    self.salinity_offset_psu[:, None, None], (depth_count, *shape)
                ),
                np.broadcast_to(
                    self.supervision_weight.T[:, :, None, None],
                    (2, depth_count, *shape),
                ),
            )
        if date is None or latitude_deg is None or longitude_deg is None:
            raise ValueError("Spatial offsets require date and per-pixel coordinates.")
        latitude = np.asarray(latitude_deg, dtype=np.float32)
        longitude = np.asarray(longitude_deg, dtype=np.float32)
        if latitude.shape != shape or longitude.shape != shape:
            raise ValueError("Spatial coordinates must share the EO field shape.")
        month = (int(date) // 100 % 100) - 1
        if not 0 <= month < MONTH_COUNT:
            raise ValueError(f"date must contain a valid month: {date}")
        lat = self.latitude_bin_centers_deg
        lon = self.longitude_bin_centers_deg
        assert lat is not None and lon is not None
        py = np.clip((latitude - lat[0]) / float(lat[1] - lat[0]), 0.0, len(lat) - 1.0)
        px = np.mod((longitude - lon[0]) / float(lon[1] - lon[0]), len(lon))
        y0 = np.floor(py).astype(np.intp)
        y1 = np.minimum(y0 + 1, len(lat) - 1)
        x0 = np.floor(px).astype(np.intp) % len(lon)
        x1 = (x0 + 1) % len(lon)

        def interpolate(field: np.ndarray) -> np.ndarray:
            """Interpolate an (lat, lon, depth[, variable]) grid."""
            extra = (None,) * (field.ndim - 2)
            fy = (py - y0)[(...,) + extra]
            fx = (px - np.floor(px))[(...,) + extra]
            low = (1.0 - fx) * field[y0, x0] + fx * field[y0, x1]
            high = (1.0 - fx) * field[y1, x0] + fx * field[y1, x1]
            return (1.0 - fy) * low + fy * high

        temperature = interpolate(self.temperature_offset_c[month]).transpose(2, 0, 1)
        salinity = interpolate(self.salinity_offset_psu[month]).transpose(2, 0, 1)
        weights = interpolate(self.supervision_weight[month]).transpose(3, 2, 0, 1)
        return temperature, salinity, weights

    def sample(
        self,
        surface_fields: Mapping[str, np.ndarray],
        *,
        depth_valid_mask: np.ndarray,
        date: int | None = None,
        latitude_deg: np.ndarray | None = None,
        longitude_deg: np.ndarray | None = None,
        temperature_anchors: np.ndarray | None = None,
        salinity_anchors: np.ndarray | None = None,
        **_: Any,
    ) -> PriorSample:
        """Build EO surface-plus-GLORYS-delta targets with sparse ARGO anchors."""
        if "sst" not in surface_fields or "sss" not in surface_fields:
            raise ValueError("Vertical-offset targets require surface SST and SSS.")
        sst = np.asarray(surface_fields["sst"], dtype=np.float32)
        sss = np.asarray(surface_fields["sss"], dtype=np.float32)
        valid_mask = np.asarray(depth_valid_mask, dtype=bool)
        if (
            sst.ndim != 2
            or sss.shape != sst.shape
            or valid_mask.shape != (len(self.depth_axis_m), *sst.shape)
        ):
            raise ValueError(
                "Surface fields and depth_valid_mask have incompatible shapes."
            )
        temperature_delta, salinity_delta, weights = self._offsets(
            date=date,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            shape=sst.shape,
        )
        temperature = sst[None] + temperature_delta
        salinity = sss[None] + salinity_delta
        for values, anchors in (
            (temperature, temperature_anchors),
            (salinity, salinity_anchors),
        ):
            if anchors is not None:
                anchor_values = np.asarray(anchors, dtype=np.float32)
                if anchor_values.shape != valid_mask.shape:
                    raise ValueError(
                        "Sparse anchors must be shaped (depth,height,width)."
                    )
                anchor_mask = np.isfinite(anchor_values) & valid_mask
                values[anchor_mask] = anchor_values[anchor_mask]
        target_valid = valid_mask & (np.isfinite(sst) & np.isfinite(sss))[None]
        weights = np.asarray(weights, dtype=np.float32).copy()
        weights[:, ~target_valid] = 0.0
        weights[:, self.depth_axis_m > self.max_supervised_depth_m] = 0.0
        return PriorSample(
            temperature_k=np.where(target_valid, temperature, 0.0).astype(np.float32),
            salinity_psu=np.where(target_valid, salinity, 0.0).astype(np.float32),
            valid_mask=target_valid,
            temperature_supervision_weight=weights[0],
            salinity_supervision_weight=weights[1],
        )


class VerticalOffsetAccumulator:
    """Stream monthly 10-degree GLORYS depth-minus-surface statistics."""

    def __init__(
        self,
        *,
        depth_axis_m: np.ndarray,
        spatial_bin_size_deg: float = 10.0,
        smoothing_sigma_cells: float = 1.0,
        shrinkage_pixels: float = 4096.0,
        extrapolation_half_life_m: float = 1000.0,
        excluded_years: Sequence[int] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Allocate global and regional sufficient statistics."""
        self.depth_axis_m = np.asarray(depth_axis_m, dtype=np.float32).reshape(-1)
        if (
            spatial_bin_size_deg <= 0.0
            or 360.0 % spatial_bin_size_deg
            or 180.0 % spatial_bin_size_deg
        ):
            raise ValueError("spatial_bin_size_deg must divide 180 and 360 degrees.")
        if (
            smoothing_sigma_cells < 0.0
            or shrinkage_pixels <= 0.0
            or extrapolation_half_life_m <= 0.0
        ):
            raise ValueError(
                "Smoothing, shrinkage, and extrapolation parameters must be positive."
            )
        self.spatial_bin_size_deg = float(spatial_bin_size_deg)
        self.smoothing_sigma_cells = float(smoothing_sigma_cells)
        self.shrinkage_pixels = float(shrinkage_pixels)
        self.extrapolation_half_life_m = float(extrapolation_half_life_m)
        self.latitude_bin_centers_deg = np.arange(
            -90.0 + spatial_bin_size_deg / 2,
            90.0,
            spatial_bin_size_deg,
            dtype=np.float32,
        )
        self.longitude_bin_centers_deg = np.arange(
            -180.0 + spatial_bin_size_deg / 2,
            180.0,
            spatial_bin_size_deg,
            dtype=np.float32,
        )
        shape = (
            MONTH_COUNT,
            len(self.latitude_bin_centers_deg),
            len(self.longitude_bin_centers_deg),
            len(self.depth_axis_m),
            2,
        )
        self.delta_sum = np.zeros(shape, dtype=np.float64)
        self.delta_count = np.zeros(shape, dtype=np.int64)
        self.global_delta_sum = np.zeros(
            (MONTH_COUNT, len(self.depth_axis_m), 2), dtype=np.float64
        )
        self.global_delta_count = np.zeros(
            (MONTH_COUNT, len(self.depth_axis_m), 2), dtype=np.int64
        )
        self.excluded_years = tuple(int(value) for value in excluded_years)
        self.provenance = dict(provenance or {})
        self.fit_years: set[int] = set()
        self.patch_count = 0
        self._has_spatial_samples = False

    def update(
        self,
        *,
        temperature_c: np.ndarray,
        salinity_psu: np.ndarray,
        date: int,
        latitude_deg: np.ndarray | None = None,
        longitude_deg: np.ndarray | None = None,
    ) -> None:
        """Accumulate one GLORYS patch into global and optional spatial bins."""
        year = int(date) // 10000
        month = (int(date) // 100 % 100) - 1
        temperature = np.asarray(temperature_c, dtype=np.float64)
        salinity = np.asarray(salinity_psu, dtype=np.float64)
        if year in self.excluded_years:
            raise ValueError(f"Refusing to fit offsets with excluded year {year}.")
        if (
            not 0 <= month < MONTH_COUNT
            or temperature.shape != salinity.shape
            or temperature.ndim != 3
            or temperature.shape[0] != len(self.depth_axis_m)
        ):
            raise ValueError(
                "GLORYS fields/date do not match the accumulator contract."
            )
        cells = None
        if latitude_deg is not None or longitude_deg is not None:
            if latitude_deg is None or longitude_deg is None:
                raise ValueError("Latitude and longitude must be supplied together.")
            latitude, longitude = np.asarray(latitude_deg), np.asarray(longitude_deg)
            if (
                latitude.shape != temperature.shape[1:]
                or longitude.shape != temperature.shape[1:]
            ):
                raise ValueError("Coordinates must match the GLORYS patch shape.")
            y = np.clip(
                np.floor((latitude + 90.0) / self.spatial_bin_size_deg).astype(np.intp),
                0,
                len(self.latitude_bin_centers_deg) - 1,
            )
            x = np.mod(
                np.floor((longitude + 180.0) / self.spatial_bin_size_deg).astype(
                    np.intp
                ),
                len(self.longitude_bin_centers_deg),
            )
            cells = (y * len(self.longitude_bin_centers_deg) + x).reshape(-1)
            self._has_spatial_samples = True
        cell_count = len(self.latitude_bin_centers_deg) * len(
            self.longitude_bin_centers_deg
        )
        for field_index, field in enumerate((temperature, salinity)):
            for depth in range(len(self.depth_axis_m)):
                valid = np.isfinite(field[0]) & np.isfinite(field[depth])
                if not np.any(valid):
                    continue
                delta = field[depth][valid] - field[0][valid]
                self.global_delta_sum[month, depth, field_index] += float(delta.sum())
                self.global_delta_count[month, depth, field_index] += int(delta.size)
                if cells is not None:
                    index = cells[valid.reshape(-1)]
                    self.delta_sum[month, :, :, depth, field_index] += np.bincount(
                        index, weights=delta, minlength=cell_count
                    ).reshape(
                        len(self.latitude_bin_centers_deg),
                        len(self.longitude_bin_centers_deg),
                    )
                    self.delta_count[month, :, :, depth, field_index] += np.bincount(
                        index, minlength=cell_count
                    ).reshape(
                        len(self.latitude_bin_centers_deg),
                        len(self.longitude_bin_centers_deg),
                    )
        self.fit_years.add(year)
        self.patch_count += 1

    def _global_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute monthly profiles and decay confidence after missing deep levels."""
        offsets = np.divide(
            self.global_delta_sum,
            self.global_delta_count,
            out=np.full_like(self.global_delta_sum, np.nan),
            where=self.global_delta_count > 0,
        )
        weights = np.zeros_like(offsets)
        total_sum, total_count = self.global_delta_sum.sum(
            axis=0
        ), self.global_delta_count.sum(axis=0)
        for month in range(MONTH_COUNT):
            for field in range(2):
                counts = self.global_delta_count[month, :, field].copy()
                values = offsets[month, :, field]
                if not np.any(counts):
                    counts = total_count[:, field].copy()
                    values = np.divide(
                        total_sum[:, field],
                        counts,
                        out=np.full_like(values, np.nan),
                        where=counts > 0,
                    )
                if not np.any(counts):
                    raise RuntimeError("No valid GLORYS offsets were accumulated.")
                values[0], counts[0], last = 0.0, max(1, int(counts[0])), 0
                for depth in range(len(self.depth_axis_m)):
                    if counts[depth] > 0:
                        last = depth
                        weights[month, depth, field] = min(
                            1.0, counts[depth] / counts[0]
                        )
                    else:
                        values[depth] = values[last]
                        # Unsupported abyssal levels inherit the last delta with less confidence.
                        weights[month, depth, field] = weights[
                            month, last, field
                        ] * 0.5 ** (
                            (self.depth_axis_m[depth] - self.depth_axis_m[last])
                            / self.extrapolation_half_life_m
                        )
                offsets[month, :, field] = values
        offsets[:, 0, :] = 0.0
        return offsets, weights

    def finalize(self, *, max_supervised_depth_m: float) -> VerticalOffsetPrior:
        """Build a smooth monthly spatial prior, or global compatibility prior."""
        if self.patch_count == 0:
            raise RuntimeError("No patches were accumulated for vertical offsets.")
        fallback, global_weights = self._global_profiles()
        if not self._has_spatial_samples:
            sums, counts = self.global_delta_sum.sum(
                axis=0
            ), self.global_delta_count.sum(axis=0)
            offsets = np.divide(
                sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
            )
            for field in range(2):
                valid = np.isfinite(offsets[:, field])
                offsets[:, field] = np.interp(
                    self.depth_axis_m, self.depth_axis_m[valid], offsets[valid, field]
                )
            offsets[0] = 0.0
            return VerticalOffsetPrior(
                depth_axis_m=self.depth_axis_m,
                temperature_offset_c=offsets[:, 0],
                salinity_offset_psu=offsets[:, 1],
                supervision_weight=np.clip(
                    counts / np.maximum(counts[0:1], 1), 0.0, 1.0
                ),
                max_supervised_depth_m=max_supervised_depth_m,
                fit_years=sorted(self.fit_years),
                excluded_years=self.excluded_years,
                provenance={
                    **self.provenance,
                    "selected_patch_count": self.patch_count,
                },
            )
        count = self.delta_count.astype(np.float64)
        local = np.divide(
            self.delta_sum, count, out=np.zeros_like(self.delta_sum), where=count > 0
        )
        blended = (local * count + fallback[:, None, None] * self.shrinkage_pixels) / (
            count + self.shrinkage_pixels
        )
        offsets, weights = np.empty_like(blended), np.empty_like(blended)
        for month in range(MONTH_COUNT):
            for depth in range(len(self.depth_axis_m)):
                for field in range(2):
                    support = count[month, :, :, depth, field] + self.shrinkage_pixels
                    raw = blended[month, :, :, depth, field]
                    if self.smoothing_sigma_cells > 0.0:
                        mode = ("reflect", "wrap")
                        raw = gaussian_filter(
                            raw * support,
                            sigma=(
                                self.smoothing_sigma_cells,
                                self.smoothing_sigma_cells,
                            ),
                            mode=mode,
                        ) / np.maximum(
                            gaussian_filter(
                                support,
                                sigma=(
                                    self.smoothing_sigma_cells,
                                    self.smoothing_sigma_cells,
                                ),
                                mode=mode,
                            ),
                            1.0e-12,
                        )
                    offsets[month, :, :, depth, field] = raw
                    local_confidence = count[month, :, :, depth, field] / support
                    weights[month, :, :, depth, field] = gaussian_filter(
                        global_weights[month, depth, field]
                        * (0.5 + 0.5 * local_confidence),
                        sigma=(self.smoothing_sigma_cells, self.smoothing_sigma_cells),
                        mode=("reflect", "wrap"),
                    )
        offsets[..., 0, :] = 0.0
        return VerticalOffsetPrior(
            depth_axis_m=self.depth_axis_m,
            temperature_offset_c=offsets[..., 0],
            salinity_offset_psu=offsets[..., 1],
            supervision_weight=np.clip(weights, 0.0, 1.0),
            max_supervised_depth_m=max_supervised_depth_m,
            latitude_bin_centers_deg=self.latitude_bin_centers_deg,
            longitude_bin_centers_deg=self.longitude_bin_centers_deg,
            fit_years=sorted(self.fit_years),
            excluded_years=self.excluded_years,
            provenance={
                **self.provenance,
                "selected_patch_count": self.patch_count,
                "spatial_bin_size_deg": self.spatial_bin_size_deg,
                "smoothing_sigma_cells": self.smoothing_sigma_cells,
                "shrinkage_pixels": self.shrinkage_pixels,
                "extrapolation_half_life_m": self.extrapolation_half_life_m,
            },
        )
