"""Frozen depth-wise EN4-to-GLORYS target bias artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class TargetBias:
    """Load and validate an additive, depth-wise target bias artifact."""

    def __init__(
        self, path: str | Path, *, expected_depth_axis_m: np.ndarray | None = None
    ) -> None:
        """Open ``path`` and verify physical units, provenance, and depth alignment."""
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            required = {
                "bias",
                "count",
                "variance",
                "supported",
                "depth_axis_m",
                "metadata",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Target bias artifact is missing {missing}.")
            if data["supported"].dtype != np.dtype(bool):
                raise ValueError("Target bias supported mask must have boolean dtype.")
            if not np.issubdtype(data["count"].dtype, np.integer):
                raise ValueError("Target bias count must have an integer dtype.")
            self.bias = np.asarray(data["bias"], dtype=np.float32).reshape(-1)
            self.count = np.asarray(data["count"], dtype=np.int64).reshape(-1)
            self.variance = np.asarray(data["variance"], dtype=np.float32).reshape(-1)
            self.supported = np.asarray(data["supported"], dtype=bool).reshape(-1)
            self.depth_axis_m = np.asarray(
                data["depth_axis_m"], dtype=np.float32
            ).reshape(-1)
            raw_metadata = data["metadata"]
            metadata = json.loads(
                str(raw_metadata.item() if raw_metadata.ndim == 0 else raw_metadata)
            )
        if not isinstance(metadata, dict):
            raise ValueError("Target bias metadata must be a JSON object.")
        self.metadata: dict[str, Any] = metadata
        if metadata.get("kind") != "en4_minus_glorys_target_bias":
            raise ValueError("Target bias artifact has an unsupported kind.")
        if metadata.get("units") != "degree_C":
            raise ValueError("Target bias artifact must declare degree_C units.")
        if metadata.get("sign_convention") != "EN4_minus_GLORYS; target=GLORYS+bias":
            raise ValueError("Target bias artifact has an unsupported sign convention.")
        size = self.depth_axis_m.size
        arrays = (self.bias, self.count, self.variance, self.supported)
        if size == 0 or any(values.size != size for values in arrays):
            raise ValueError("Target bias arrays must match the depth axis.")
        if not np.isfinite(self.depth_axis_m).all() or np.any(
            np.diff(self.depth_axis_m) <= 0
        ):
            raise ValueError("Target bias depth axis must be finite and increasing.")
        metadata_depth = np.asarray(
            metadata.get("depth_axis_m", ()), dtype=np.float32
        ).reshape(-1)
        if not np.array_equal(metadata_depth, self.depth_axis_m):
            raise ValueError(
                "Target bias metadata depth axis must match artifact arrays."
            )
        if (
            not np.isfinite(self.bias).all()
            or not np.isfinite(self.variance).all()
            or np.any(self.count < 0)
            or np.any(self.variance < 0)
        ):
            raise ValueError(
                "Target bias values must be finite and nonnegative where applicable."
            )
        if np.any(~self.supported & (self.bias != 0)):
            raise ValueError("Unsupported target bias bands must have zero offset.")
        if (
            metadata.get("source_temperature")
            != "EN4 potential temperature raw compact profile"
        ):
            raise ValueError(
                "Target bias artifact must declare raw EN4 potential temperature provenance."
            )
        if metadata.get("source_temperature_variable") != "POTM_CORRECTED":
            raise ValueError(
                "Target bias artifact must declare POTM_CORRECTED provenance."
            )
        if (
            metadata.get("source_temperature_standard_name")
            != "sea_water_potential_temperature"
        ):
            raise ValueError(
                "Target bias artifact must declare potential-temperature provenance."
            )
        excluded_years = metadata.get("excluded_years")
        if not isinstance(excluded_years, list) or not excluded_years:
            raise ValueError("Target bias artifact must declare excluded years.")
        min_profiles = metadata.get("min_profiles")
        if not isinstance(min_profiles, int) or min_profiles < 1:
            raise ValueError(
                "Target bias artifact must declare a positive min_profiles value."
            )
        if np.any(self.supported & (self.count < min_profiles)):
            raise ValueError("Supported target bias bands do not meet min_profiles.")
        input_hashes = metadata.get("input_hashes")
        if (
            not isinstance(input_hashes, dict)
            or not input_hashes
            or not input_hashes.get("manifest.yaml")
        ):
            raise ValueError("Target bias artifact must include input hashes.")
        grid = metadata.get("grid")
        if not isinstance(grid, dict):
            raise ValueError("Target bias artifact is missing grid provenance.")
        if expected_depth_axis_m is not None and not np.array_equal(
            self.depth_axis_m, np.asarray(expected_depth_axis_m, dtype=np.float32)
        ):
            raise ValueError(
                "Target bias depth axis does not match the dataset manifest."
            )
        self.grid = grid

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Add supported offsets to finite cells of a depth-first target array."""
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 1 or array.shape[0] != self.bias.size:
            raise ValueError("Target bias values must have depth as their first axis.")
        out = array.copy()
        offsets = self.bias.reshape((self.bias.size,) + (1,) * (array.ndim - 1))
        supported = self.supported.reshape((self.bias.size,) + (1,) * (array.ndim - 1))
        valid = np.isfinite(out) & supported
        out[valid] += np.broadcast_to(offsets, out.shape)[valid]
        return out


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for provenance metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()
