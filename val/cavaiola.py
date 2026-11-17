# PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.cavaiola --points val/results/en4/bundle/points.csv --repository /tmp/cavaiola2026 --output val/results/cavaiola2026 --source cds
"""Run Cavaiola's released model and score its upper-ocean diagnostic separately."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from val.metrics import _cluster_bootstrap, _profile_losses
from val.paper_common import method_arrays, point_crps, profile_mean, save_table

REFERENCE = "https://doi.org/10.1016/j.apor.2026.105121"
UPSTREAM = "https://github.com/Mattiads1/AI-Sea-Temperature-Profile-reconstruction"
LABEL = "Released Cavaiola 2026: training-era diagnostic, not independent validation"


def _file(path: Path) -> dict:
    """Record the absolute path and SHA256 of a local input or prediction."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _checked(spec: dict) -> Path:
    """Reject missing or modified artifacts before loading any predictions."""
    path = Path(spec["path"])
    if _file(path)["sha256"] != spec.get("sha256"):
        raise ValueError(f"Cavaiola artifact SHA256 mismatch: {path}")
    return path


def _support(points: pd.DataFrame) -> pd.DataFrame:
    """Select the published temperature support without extrapolating."""
    selected = points.loc[
        points.variable.eq("temperature") & points.depth_m.between(5, 200)
    ].copy()
    if selected.profile_id.isna().any():
        raise ValueError("Cavaiola comparison requires populated profile identifiers")
    return selected


def _prediction(depths: np.ndarray, members: np.ndarray) -> None:
    """Require the released model's finite 40-level, 100-head output."""
    if (
        not np.array_equal(depths, np.arange(5, 201, 5))
        or members.shape != (40, 100)
        or not np.isfinite(members).all()
    ):
        raise ValueError(
            "Expected finite Cavaiola predictions on (40 depths, 100 heads)"
        )


def predict(points_path: Path, repository: Path, output: Path, source: str) -> dict:
    """Call the author's preparation/inference functions without fetching new truth.

    Run this in a separate CLI process: the upstream code uses TensorFlow and
    imports modules named ``model`` and ``infer`` from its own checkout.
    """
    points = _support(pd.read_csv(points_path, dtype={"profile_id": str}))
    if points.empty:
        raise ValueError("No temperature points in the published 5-200 m range")
    repository = repository.resolve()
    artifacts = {
        name: _file(repository / name)
        for name in (
            "prepare_input_v2.py",
            "infer.py",
            "utils/model.py",
            "utils/saved_model/scalers.pkl",
            "utils/saved_model/model.weights.h5",
        )
    }
    sys.path.insert(0, str(repository))
    prepare, inference = import_module("prepare_input_v2"), import_module("infer")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profiles = []
    for index, (profile, frame) in enumerate(points.groupby("profile_id", sort=False)):
        keys = ["observation_time", "latitude", "longitude"]
        if len(frame[keys].drop_duplicates()) != 1:
            raise ValueError(f"Inconsistent cast metadata for {profile}")
        row = frame.iloc[0]
        timestamp = pd.to_datetime(row.observation_time, utc=True)
        payload_path = output / f"profile_{index:06d}_input.pkl"
        # Keep the upstream pickle local; scoring reads only numeric NPZ outputs.
        prepare.prepare_input(
            float(row.latitude),
            float(row.longitude),
            timestamp.tz_localize(None).to_pydatetime(),
            payload_path,
            include_glorys=False,
            include_argo=False,
            source=source,
        )
        result = inference.infer(
            input_path=payload_path,
            scalers_path=artifacts["utils/saved_model/scalers.pkl"]["path"],
            weights_path=artifacts["utils/saved_model/model.weights.h5"]["path"],
            scripts_dir=repository / "utils",
            output_path=None,
        )
        depths, members = np.asarray(result["depth_grid"]), np.asarray(
            result["ensemble"]
        )
        _prediction(depths, members)
        prediction_path = output / f"profile_{index:06d}.npz"
        np.savez_compressed(prediction_path, depth_grid=depths, members=members)
        profiles.append(
            {
                "profile_id": str(profile),
                "observation_time": timestamp.isoformat(),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "input": _file(payload_path),
                "prediction": _file(prediction_path),
            }
        )
    manifest = {
        "method": "cavaiola2026",
        "reference": REFERENCE,
        "upstream": UPSTREAM,
        "interpretation": LABEL,
        "temperature_kind": "in_situ",
        "distribution": "deterministic_crps_heads",
        "training_years": [2004, 2020],
        "source_points": _file(points_path),
        "implementation": artifacts,
        "era5_source": source,
        "profiles": profiles,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return _file(path)


def _aligned(points: pd.DataFrame, manifest: dict, options: dict) -> np.ndarray:
    """Match complete cast identities, interpolate heads, and convert to theta0."""
    records = manifest["profiles"]
    lookup = {record["profile_id"]: record for record in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate Cavaiola profile identifiers")
    values = np.full((len(points), 100), np.nan)
    for profile, frame in points.groupby("profile_id", sort=False):
        if str(profile) not in lookup:
            raise ValueError(f"Missing Cavaiola prediction for {profile}")
        record = lookup[str(profile)]
        times = pd.to_datetime(frame.observation_time, utc=True, format="mixed")
        if (
            not times.eq(pd.to_datetime(record["observation_time"], utc=True)).all()
            or not np.allclose(frame.latitude, record["latitude"], rtol=0, atol=1e-7)
            or not np.allclose(frame.longitude, record["longitude"], rtol=0, atol=1e-7)
        ):
            raise ValueError(f"Cavaiola cast time/coordinates differ for {profile}")
        _checked(record["input"])
        with np.load(_checked(record["prediction"]), allow_pickle=False) as saved:
            depths, members = saved["depth_grid"], saved["members"]
        _prediction(depths, members)
        for member in range(100):
            values[frame.index, member] = np.interp(
                frame.depth_m, depths, members[:, member]
            )
    if options.get("reference_salinity"):
        salinity = pd.read_csv(_checked(options["reference_salinity"]))
        if salinity.point_id.isna().any() or salinity.point_id.duplicated().any():
            raise ValueError(
                "Reference salinity point IDs must be populated and unique"
            )
        salinity = (
            salinity.set_index("point_id").reindex(points.point_id).reference_salinity
        )
    elif "reference_salinity" in points:
        salinity = points.reference_salinity
    else:
        raise ValueError(
            "Cavaiola in-situ conversion requires per-point reference_salinity"
        )
    salinity = np.asarray(salinity, dtype=float)
    if not np.isfinite(salinity).all() or np.any((salinity < 0) | (salinity > 50)):
        raise ValueError(
            "Reference salinity must be finite practical salinity in [0, 50]"
        )
    try:
        import gsw
    except ImportError as error:
        raise ValueError(
            "Cavaiola temperature conversion requires optional gsw"
        ) from error
    pressure = gsw.p_from_z(
        -points.depth_m.to_numpy(float), points.latitude.to_numpy(float)
    )
    absolute = gsw.SA_from_SP(
        salinity,
        pressure,
        points.longitude.to_numpy(float),
        points.latitude.to_numpy(float),
    )
    # Argo TEMP targets are in-situ; our frozen benchmark uses potential temperature.
    converted = gsw.pt0_from_t(absolute[:, None], values, pressure[:, None])
    if not np.isfinite(converted).all():
        raise ValueError("Nonfinite converted Cavaiola predictions")
    return converted


def run(context: Any, output: Path, options: dict) -> dict:
    """Score released weights as a diagnostic on fixed, shared upper-ocean points."""
    manifest = json.loads(
        _checked(
            {"path": options["manifest"], "sha256": options["manifest_sha256"]}
        ).read_text()
    )
    if (
        manifest.get("method") != "cavaiola2026"
        or manifest.get("temperature_kind") != "in_situ"
        or manifest.get("distribution") != "deterministic_crps_heads"
        or manifest.get("training_years") != [2004, 2020]
    ):
        raise ValueError("Manifest does not describe the released Cavaiola model")
    for spec in manifest["implementation"].values():
        _checked(spec)
    points = _support(context.evaluation.points)
    if points.empty:
        return {
            "status": "blocked",
            "summary": "No shared 5-200 m temperature support",
            "artifacts": [],
            "details": {"interpretation": LABEL},
        }
    if context.evaluation.metadata.get("temperature_kind_compared") != "potential":
        raise ValueError("Cavaiola adapter requires a potential-temperature benchmark")
    indices = points.index.to_numpy()
    points = points.reset_index(drop=True)
    arrays = {
        "cavaiola2026": _aligned(points, manifest, options),
        "depthdif": context.evaluation.members["residual"][indices],
        "glorys_packaged": points.glorys.to_numpy(float)[:, None],
    }
    for name, values in method_arrays(
        context.evaluation.points, context.evaluation.members
    ).items():
        if name not in {"residual", "glorys"}:
            arrays[name] = values[indices]
    rows, paired = [], []
    truth = points.observation.to_numpy(float)
    if not np.isfinite(truth).all() or any(
        not np.isfinite(values).all() for values in arrays.values()
    ):
        raise ValueError("Every method must be finite on the complete shared support")
    losses = {}
    for name, members in arrays.items():
        error = members.mean(axis=1) - truth
        losses[name] = {
            "mae": np.abs(error),
            "rmse": error**2,
            "crps": point_crps(members, truth),
        }
        rows.append(
            {
                "method": name,
                "profiles": points.profile_id.nunique(),
                "points": len(points),
                "mae": profile_mean(points, np.abs(error)),
                "rmse": np.sqrt(profile_mean(points, error**2)),
                "bias_prediction_minus_observation": profile_mean(points, error),
                "empirical_crps": profile_mean(points, losses[name]["crps"]),
                "interpretation": LABEL,
            }
        )
    settings = context.config.get("statistics", {})
    for name in arrays:
        if name == "depthdif":
            continue
        for metric in ("mae", "rmse", "crps"):
            estimate = _cluster_bootstrap(
                _profile_losses(
                    points, losses["depthdif"][metric], losses[name][metric]
                ),
                int(settings.get("bootstrap_replicates", 2000)),
                context.seed,
                int(settings.get("min_profiles", 30)),
                int(settings.get("min_clusters", 20)),
                np.sqrt if metric == "rmse" else lambda value: value,
            )
            paired.append(
                {
                    "reference": name,
                    "metric": metric,
                    **estimate,
                    "confirmation_eligible": False,
                    "interpretation": LABEL,
                }
            )
    artifacts = save_table(pd.DataFrame(rows), output, "cavaiola_scores", LABEL)
    artifacts += save_table(
        pd.DataFrame(paired),
        output,
        "cavaiola_paired_comparisons",
        "DepthDif minus reference on identical 5-200 m temperature points; diagnostic only",
    )
    points["cavaiola_mean"] = arrays["cavaiola2026"].mean(axis=1)
    artifacts += save_table(
        points,
        output,
        "cavaiola_matched_points",
        "Shared support; observations retain their benchmark depths",
    )
    np.savez_compressed(output / "cavaiola_members.npz", **arrays)
    artifacts.append(str(output / "cavaiola_members.npz"))
    return {
        "status": "complete",
        "summary": LABEL,
        "artifacts": artifacts,
        "details": {
            "manifest": options,
            "provenance": manifest,
            "confirmation_eligible": False,
            "fair_crps_applied": False,
            "points_excluded_outside_published_support": len(context.evaluation.points)
            - len(points),
            "temperature_operator": "in_situ_to_potential_teos10; practical reference salinity",
            "timing": "ERA5 at observation timestamp; DepthDif/GLORYS retain their frozen target times",
        },
    }


def main() -> None:
    """Prepare ERA5 inputs and predictions using a local upstream checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("cds", "gcs"), default="cds")
    args = parser.parse_args()
    manifest = predict(args.points, args.repository, args.output, args.source)
    print(
        json.dumps(
            {
                "paper": {
                    "comparators": {
                        "cavaiola2026": {
                            "manifest": manifest["path"],
                            "manifest_sha256": manifest["sha256"],
                        }
                    }
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
