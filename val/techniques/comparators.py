"""Import externally run published/AI comparators on the exact frozen point IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import save_table


def run(context: Any, output_dir: Path, options: dict | None = None) -> dict:
    """Register deterministic predictions or sampled ensembles with provenance.

    A prediction CSV must contain every evaluation point_id exactly once. Optional
    NPZ members use the CSV row order and a ``members`` key. Input hashes are
    mandatory, so updating external results changes the frozen specification.
    """
    options = options or {}
    points, arrays = context.evaluation.points, context.evaluation.members
    rows, artifacts = [], []
    for name, metadata in context.evaluation.metadata.get("baselines", {}).items():
        rows.append(
            {
                "method": name,
                "status": "available",
                "information": "same masked model batches",
                "provenance": json.dumps(metadata, sort_keys=True),
            }
        )
    for spec in options.get("predictions", []):
        name = str(spec["name"])
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            or name in arrays
            or f"baseline_{name}" in points
        ):
            raise ValueError(f"Invalid or duplicate external comparator name: {name}")
        provenance = spec.get("provenance", {})
        if not all(
            provenance.get(key)
            for key in ("method", "reference", "inputs", "training_data", "checkpoint")
        ):
            raise ValueError(
                "External comparators require method, reference, inputs, training_data and checkpoint provenance"
            )
        for key in ("points", "members"):
            if not spec.get(key):
                continue
            path = Path(spec[key])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != spec.get("sha256", {}).get(key):
                raise ValueError(
                    f"External comparator {name} {key} SHA256 mismatch or missing digest"
                )
        predictions = pd.read_csv(spec["points"])
        if predictions.point_id.isna().any() or predictions.point_id.duplicated().any():
            raise ValueError("External comparator point identities must be unique")
        lookup = predictions.reset_index().set_index("point_id")
        missing = set(points.point_id) - set(lookup.index)
        if missing:
            raise ValueError(
                f"External comparator {name} lacks {len(missing)} evaluation points"
            )
        selected = lookup.loc[points.point_id]
        # Verify supplied truth/location metadata rather than silently accepting
        # an ID collision with a different cast or observation operator.
        for column in ("variable", "profile_id"):
            if column in predictions and not np.array_equal(
                selected[column].astype(str), points[column].astype(str)
            ):
                raise ValueError(
                    f"External comparator {column} conflicts with frozen evaluation"
                )
        for column in ("observation", "depth_m", "latitude", "longitude"):
            if column in predictions and not np.allclose(
                selected[column], points[column], rtol=0, atol=1e-7
            ):
                raise ValueError(
                    f"External comparator {column} conflicts with frozen evaluation"
                )
        if spec.get("members"):
            if not spec.get("iid_members", False):
                raise ValueError(
                    "External ensembles require iid sampled members; deterministic quantile heads need a separately named scoring adapter"
                )
            with np.load(spec["members"], allow_pickle=False) as saved:
                values = saved["members"]
            if (
                values.ndim != 2
                or values.shape[0] != len(predictions)
                or values.shape[1] < 2
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "External members must be finite row-aligned iid sample matrices"
                )
            arrays[name] = values[selected["index"].to_numpy(int)]
        else:
            values = selected.prediction.to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("External deterministic predictions must be finite")
            points[f"baseline_{name}"] = values
        rows.append(
            {
                "method": name,
                "status": "available",
                "information": provenance["inputs"],
                "provenance": json.dumps(
                    {**spec, "verified_rows": len(points)}, sort_keys=True
                ),
            }
        )
    if not rows:
        rows = [
            {
                "method": "published_and_toy_AI",
                "status": "blocked",
                "information": "No matching trained AI checkpoints or external predictions configured",
                "provenance": "Set baselines in the cohort config or paper.comparators.predictions",
            }
        ]
    artifacts += save_table(
        pd.DataFrame(rows),
        output_dir,
        "comparator_availability",
        "AI comparators, matched support and declared information differences",
    )
    return {
        "status": (
            "complete"
            if any(row["status"] == "available" for row in rows)
            else "blocked"
        ),
        "summary": (
            "Verified matching AI/published comparator predictions"
            if any(row["status"] == "available" for row in rows)
            else rows[0]["information"]
        ),
        "artifacts": artifacts,
        "details": {
            "comparators": rows,
            "fair_comparison": "Different declared inputs/domains remain explicit; alignment alone does not make them equivalent",
        },
    }
