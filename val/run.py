# Examples (from the repository root, each command shows all its options):
# /work/envs/depth/bin/python -m val.run preflight --config val/config.yaml --output val/results/preflight
# /work/envs/depth/bin/python -m val.run evaluate --config val/config.yaml --output val/results/en4
# /work/envs/depth/bin/python -m val.run demo --output val/results/demo
# /work/envs/depth/bin/python -m val.run score --bundle val/results/en4/bundle --output val/results/rescore
"""Validate frozen residual models without modifying the training application."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from val.metrics import score_bundle

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "val"


def _sha256(path: Path) -> str:
    """Hash large inputs by streaming rather than loading their contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """Persist machine-readable provenance, rejecting nonstandard NaN JSON."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def _output(path: Path) -> Path:
    """Constrain every workflow artifact to val and preserve existing results."""
    resolved = path.resolve()
    if VAL not in resolved.parents:
        raise ValueError("Output must be a new directory below the repository's val/.")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _config(path: Path) -> dict:
    """Load a serializable validation configuration relative to the repo root."""
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Validation configuration must be a YAML mapping.")
    variables = config.get("sampling", {}).get("variables", [])
    if not variables or set(variables) - {"temperature", "salinity"}:
        raise ValueError("sampling.variables requires temperature and/or salinity.")
    if config.get("cohort", {}).get("split_role") not in {"discovery", "confirmation"}:
        raise ValueError("cohort.split_role must be discovery or confirmation.")
    if int(config.get("sampling", {}).get("members", 0)) < 2:
        raise ValueError("sampling.members must be at least two.")
    reference_manifest = config.get("cohort", {}).get("reference_manifest")
    if reference_manifest and Path(reference_manifest).is_file():
        manifest = json.loads(Path(reference_manifest).read_text())
        for key in (
            "temperature_observation_operator",
            "reference_salinity_kind",
            "reference_salinity_provenance",
        ):
            config["cohort"][key] = manifest["config_snippet"][key]
    config["validation_config"] = str(path.resolve())
    return config


def _raster_entries(node):
    """Yield date/path records from either flat or variable-nested manifests."""
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "path" in item:
                yield item
    elif isinstance(node, dict):
        for value in node.values():
            yield from _raster_entries(value)


def preflight(config: dict) -> dict:
    """Inventory required inputs and field capability without constructing a model."""
    checks, issues = {}, []
    model = config["model"]
    for key in ("config", "checkpoint"):
        raw = model.get(key)
        path = Path(raw) if raw else None
        checks[key] = {"path": raw, "exists": bool(path and path.is_file())}
        if not checks[key]["exists"]:
            issues.append(
                f"Missing model {key}: {raw}; supply matching trained weights/config."
            )
    source = {}
    if checks["config"]["exists"]:
        source = yaml.safe_load(Path(model["config"]).read_text())
        scenario = source.get(
            "scenario", source.get("model", {}).get("scenario", "temperature")
        )
        fields = {"joint": ["temperature", "salinity"]}.get(scenario, [scenario])
        checks["checkpoint_fields"] = fields
        unavailable = set(config["sampling"]["variables"]) - set(fields)
        if unavailable:
            issues.append(f"Configured checkpoint has no {sorted(unavailable)} output.")
    if checks["checkpoint"]["exists"]:
        actual = _sha256(Path(model["checkpoint"]))
        checks["checkpoint"]["sha256"] = actual
        if not model.get("checkpoint_sha256"):
            issues.append("Pin model.checkpoint_sha256 before scientific inference.")
        elif actual != model["checkpoint_sha256"]:
            issues.append("Checkpoint contents differ from the pinned SHA-256.")
    root = Path(config["data_root"])
    manifest_path = root / "manifest.yaml"
    checks["manifest"] = {"path": str(manifest_path), "exists": manifest_path.is_file()}
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text())
        checks["manifest"]["sha256"] = _sha256(manifest_path)
        variables = config["sampling"]["variables"]
        surface_sources = (
            source.get("data", {})
            .get("dataset", {})
            .get("surface_conditioning", {})
            .get("sources", ["sst"])
        )
        required = {
            "glorys": ["thetao" if v == "temperature" else "so" for v in variables]
        }
        for surface in surface_sources:
            section = {"sst": "ostia", "sss": "sss", "ssh": "sealevel"}.get(
                surface, surface
            )
            required[section] = None
        raster_checks = {}
        year = int(config["cohort"]["year"])
        for section, fields in required.items():
            node = manifest.get("rasters", {}).get(section, {})
            entries = list(
                _raster_entries(
                    node if fields is None else {v: node.get(v, []) for v in fields}
                )
            )
            entries = [e for e in entries if int(e.get("date", 0)) // 10000 == year]
            missing = [
                str(root / e["path"])
                for e in entries
                if not (root / e["path"]).is_file()
            ]
            raster_checks[section] = {
                "expected": len(entries),
                "missing": len(missing),
                "first_missing": missing[:1],
            }
            if not entries or missing:
                issues.append(
                    f"{section}: {len(missing)}/{len(entries)} required {year} rasters missing; restore dataset."
                )
        checks["rasters"] = raster_checks
        for name, relative in (
            ("profiles", manifest.get("argo", {}).get("path")),
            ("land_mask", manifest.get("grid", {}).get("source")),
        ):
            path = root / relative if relative else None
            exists = bool(path and path.exists())
            checks[name] = {"path": str(path), "exists": exists}
            if not exists:
                issues.append(f"Missing dataset {name}: {path}")
    else:
        issues.append(f"Dataset not restored: {manifest_path}")
    climate = (
        source.get("data", {}).get("dataset", {}).get("climatology", {}).get("path")
    )
    if climate:
        path = Path(climate)
        if not path.is_absolute():
            path = path if path.exists() else root / path
        checks["climatology"] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            issues.append(f"Missing frozen climatology: {path}")
    cohort = config["cohort"]
    key = "audit_path" if cohort["kind"] == "en4_archive" else "profiles"
    path = Path(cohort[key]) if cohort.get(key) else None
    checks["cohort"] = {"path": str(path), "exists": bool(path and path.is_file())}
    if not checks["cohort"]["exists"]:
        issues.append(f"Missing cohort {key}: {path}")
    if (
        cohort.get("temperature_kind") == "in_situ"
        and "temperature" in config["sampling"]["variables"]
    ):
        if (
            cohort.get("temperature_observation_operator")
            != "in_situ_to_potential_teos10"
        ):
            issues.append(
                "XBT temperature is in-situ; configure the documented TEOS-10 observation operator before comparison with thetao."
            )
        else:
            import importlib.util

            if importlib.util.find_spec("gsw") is None:
                issues.append(
                    "TEOS-10 conversion needs the optional gsw dependency; see val/requirements-optional.txt."
                )
            if not cohort.get("reference_salinity_provenance"):
                issues.append(
                    "TEOS-10 conversion needs documented per-row reference salinity and its provenance."
                )
    return {
        "ready": not issues,
        "issues": issues,
        "checks": checks,
        "salinity_note": "QC fine-tuning run 1419568d is temperature-only. Use config_salinity.yaml with salinity/joint weights for S.",
    }


def _bundle(config: dict, output: Path):
    """Resolve evaluation-only overrides into val without changing source configs."""
    from depth_recon.configs.config_resolver_pixel import load_pixel_training_config

    overrides = {
        "data.dataset.core.geotiff_root_dir": str(Path(config["data_root"]).resolve()),
        "data.dataset.core.metadata_cache_dir": str(output / "dataset_cache"),
        "data.split.val_year": int(config["cohort"]["year"]),
        "data.dataset.selection.filter_bad_argo_quality": True,
        "training.training.validation_sampling.sampler": config["sampling"]["sampler"],
        "training.training.validation_sampling.ddim_num_timesteps": int(
            config["sampling"]["steps"]
        ),
    }
    return load_pixel_training_config(
        config_path_value=config["model"]["config"],
        runtime_config_dir=output / "effective",
        overrides=[
            f"{key}={yaml.safe_dump(value, default_flow_style=True).splitlines()[0]}"
            for key, value in overrides.items()
        ],
        write_snapshots=True,
    )


def _en4_cohort(dataset, config: dict, output: Path):
    """Adapt the existing QC/archive holdout selector to a standalone point cohort."""
    from depth_recon.inference.export_paper_metrics import (
        load_dataset_context,
        load_en4_candidate_profiles,
    )
    from depth_recon.utils.en4_archive_holdout import EN4ArchiveHoldoutValidation
    from val.observations import _region_label

    cfg = config["cohort"]
    audit_path = Path(cfg["audit_path"])
    candidates = load_en4_candidate_profiles(
        context=load_dataset_context(Path(config["data_root"])),
        date_year=int(cfg["year"]),
        candidate_profiles_path=audit_path,
        profile_store=dataset.argo_store,
        audit_status="no_spatiotemporal_candidate",
        require_quality=True,
    )
    selector = EN4ArchiveHoldoutValidation(
        dataset=dataset,
        candidate_df=candidates,
        audit_path=audit_path,
        output_dir=output,
        sample_count=int(cfg["max_patches"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(config["sampling"]["seed"]),
        holdout_fraction=float(cfg["holdout_fraction"]),
        min_input_profiles=int(cfg["min_input_profiles"]),
        max_profiles_to_plot=0,
    )
    fields = tuple(config["sampling"]["variables"])
    selector._select_patches(fields)
    records = []
    # Match raw timestamps by source identity; weekly product dates are not observation times.
    audit = pd.read_parquet(
        audit_path,
        columns=["profile_source_file", "source_profile_idx", "datetime_utc"],
    )
    audit = audit.drop_duplicates(
        ["profile_source_file", "source_profile_idx"]
    ).set_index(["profile_source_file", "source_profile_idx"])
    for position, item in selector.assignments.iterrows():
        profile = f"en4:{item.profile_source_file}:{int(item.source_profile_idx)}"
        target_time = pd.Timestamp(str(int(item.date)), tz="UTC")
        observation_time = pd.to_datetime(
            audit.loc[
                (item.profile_source_file, int(item.source_profile_idx)), "datetime_utc"
            ],
            utc=True,
        )
        region = _region_label(
            item.lat, item.lon, float(cfg.get("region_size_deg", 20))
        )
        for field in fields:
            for depth_index, value in enumerate(selector.references[field][position]):
                if not np.isfinite(value):
                    continue
                records.append(
                    dict(
                        point_id=f"{profile}:{field}:{depth_index}",
                        profile_id=profile,
                        # Conservative space/time clusters are a fallback, not independently identified platforms.
                        cluster_id=f"en4:{_region_label(item.lat, item.lon, float(cfg.get('cluster_size_deg', 5)))}:{target_time.strftime('%Y-%m')}",
                        source="en4_archive",
                        region=region,
                        latitude=float(item.lat),
                        longitude=float(item.lon),
                        observation_time=observation_time.isoformat(),
                        target_time=target_time.isoformat(),
                        depth_m=float(selector.depth_axis_m[depth_index]),
                        variable=field,
                        observation=float(value),
                        independence="archive_screened",
                        split_role=cfg["split_role"],
                        patch_index=int(item.patch_index),
                        row=int(item.local_row),
                        col=int(item.local_col),
                        depth_index=depth_index,
                    )
                )
    assignments = pd.DataFrame(records)
    if assignments.empty:
        raise ValueError("No QC-supported EN4 observations survived selection.")
    _write_json(output / "en4_selection.json", selector.selection)
    eligible = (
        candidates.assign(
            region=[
                _region_label(row.lat, row.lon, float(cfg.get("region_size_deg", 20)))
                for row in candidates.itertuples(index=False)
            ]
        )
        .groupby("region")
        .size()
        .rename("eligible_profiles")
    )
    selected = (
        assignments.drop_duplicates("profile_id")
        .groupby("region")
        .size()
        .rename("selected_profiles")
    )
    coverage = eligible.to_frame().join(selected).fillna(0).astype(int).reset_index()
    return SimpleNamespace(
        assignments=assignments,
        patch_indices=selector.patch_indices,
        masked_sample=selector._masked_sample,
        holdouts=selector.holdouts,
        exclusions=pd.DataFrame(columns=["profile_id", "reason"]),
        coverage=coverage,
        metadata={
            "independence": "archive_screened",
            "cluster_method": "5-degree tile x month fallback (configurable); platform dependence may extend across blocks",
            "reference_representation": "QC-filtered compact EN4 profiles on the 50 model levels; quantized by the dataset export, not native raw NetCDF profiles",
            "audit_sha256": _sha256(audit_path),
            "conditioning_exclusion": "all selected cells, both fields, every overlapping patch; not entire platforms",
        },
    )


def _batches(dataset, cohort, config: dict, exclusions: list):
    """Yield masked matched batches, establishing support before any prediction."""
    import torch
    from torch.utils.data import default_collate
    from depth_recon.utils.normalizations import (
        temperature_normalize,
        salinity_normalize,
    )
    from val.inference import _prediction_points

    size = int(config["cohort"].get("batch_size", 2))
    if size < 1:
        raise ValueError("cohort.batch_size must be positive.")
    for start in range(0, len(cohort.patch_indices), size):
        indices = cohort.patch_indices[start : start + size]
        samples = [cohort.masked_sample(index) for index in indices]
        points = cohort.assignments.loc[
            cohort.assignments.patch_index.isin(indices)
        ].copy()
        points["batch_index"] = points.patch_index.map(
            {index: i for i, index in enumerate(indices)}
        )
        batch = default_collate(samples)
        location_support = None
        for key in ("x_valid_mask", "x_salinity_valid_mask"):
            if key in batch:
                locations = batch[key].bool().any(dim=1)
                location_support = (
                    locations
                    if location_support is None
                    else location_support | locations
                )
        count = (
            location_support.sum(dim=(1, 2)).cpu()
            if location_support is not None
            else torch.zeros(len(samples), dtype=torch.long)
        )
        supported = points.batch_index.map(
            lambda i: int(count[int(i)])
            >= int(config["cohort"].get("min_input_profiles", 1))
        )
        if {"observation_time", "target_time"}.issubset(points):
            observation_time = pd.to_datetime(
                points.observation_time, utc=True, errors="raise", format="mixed"
            )
            target_time = pd.to_datetime(
                points.target_time, utc=True, errors="raise", format="mixed"
            )
            points["time_delta_days"] = (
                observation_time - target_time
            ).dt.total_seconds() / 86400.0
            supported &= points.time_delta_days.abs().le(
                float(config["cohort"].get("max_time_delta_days", 3.5))
            )
        for variable, normalize, ykey, ckey in (
            ("temperature", temperature_normalize, "y", "climatology"),
            ("salinity", salinity_normalize, "y_salinity", "climatology_salinity"),
        ):
            selected = points.variable.eq(variable)
            if not selected.any():
                continue
            target = ykey + "_glorys" if ykey + "_glorys" in batch else ykey
            if ckey not in batch:
                raise ValueError(
                    f"No frozen {variable} climatology in evaluation batch."
                )
            valid = batch[target + "_valid_mask"].bool() & batch["land_mask"].bool()
            field = normalize("denorm", batch[target]).masked_fill(~valid, float("nan"))
            climate = normalize("denorm", batch[ckey]).masked_fill(~valid, float("nan"))
            locations = points.loc[selected]
            for name, values in (("glorys", field), ("climatology", climate)):
                points.loc[selected, name] = _prediction_points(
                    values, locations, batch_offset=0
                ).prediction.to_numpy()
        # The common support depends only on observations/references, never model output.
        finite = np.isfinite(
            points[["observation", "glorys", "climatology"]].to_numpy(float)
        ).all(axis=1)
        for point_id in points.loc[~supported | ~finite, "point_id"]:
            exclusions.append(
                {
                    "point_id": point_id,
                    "reason": "reference_support_or_minimum_conditioning",
                }
            )
        points = points.loc[supported & finite].copy()
        if not points.empty:
            yield batch, points


def evaluate(config: dict, output: Path) -> dict:
    """Run the fixed model on one declared cohort and produce its scorecard."""
    from depth_recon.inference.core import build_dataset
    from val.inference import run_model
    from val.observations import prepare_external

    readiness = preflight(config)
    _write_json(output / "preflight.json", readiness)
    if not readiness["ready"]:
        raise ValueError("Preflight failed:\n" + "\n".join(readiness["issues"]))
    bundle = _bundle(config, output)
    dataset = build_dataset(
        bundle.effective_data_config_path, bundle.data_cfg["dataset"], split="val"
    )
    if config["cohort"]["kind"] == "en4_archive":
        cohort = _en4_cohort(dataset, config, output)
    else:
        points = pd.read_csv(config["cohort"]["profiles"])
        cohort = prepare_external(dataset, points, config["cohort"])
        cohort.assignments["split_role"] = config["cohort"]["split_role"]
        cohort.assignments["target_time"] = pd.to_datetime(
            cohort.assignments.target_date.astype(str), utc=True
        ).astype(str)
    cohort.assignments.to_csv(output / "cohort.csv", index=False)
    _write_json(
        output / "holdouts.json",
        {
            "locations": [list(key) for key in sorted(cohort.holdouts)],
            "metadata": cohort.metadata,
        },
    )
    exclusions = cohort.exclusions.to_dict(orient="records")
    batches = iter(_batches(dataset, cohort, config, exclusions))
    try:
        first = next(batches, None)
        if first is None:
            raise ValueError(
                "No eligible observations survive the common support checks; see bundle/exclusions.csv."
            )
        metadata = run_model(
            config,
            output / "bundle",
            chain([first], batches),
            bundle=bundle,
            dataset=dataset,
        )
    finally:
        (output / "bundle").mkdir(exist_ok=True)
        pd.DataFrame(exclusions, columns=["profile_id", "point_id", "reason"]).to_csv(
            output / "bundle" / "exclusions.csv", index=False
        )
    metadata.update(cohort=cohort.metadata, checkpoint_verified=True, demo=False)
    # Incomplete platform exclusion must not become a confirmatory result through a flag.
    metadata["conditioning_exclusion_verified"] = bool(
        cohort.metadata.get("conditioning_exclusion_verified", False)
    )
    evidence = config.get("evidence", {})
    metadata["confirmatory_endpoints"] = evidence.get("confirmatory_endpoints", [])
    if evidence.get("thresholds_external"):
        threshold_path = Path(evidence.get("thresholds_provenance", ""))
        if not threshold_path.is_file():
            raise ValueError("External extremes thresholds require a provenance file.")
        metadata["thresholds_external"] = True
        metadata["thresholds_provenance"] = {
            "path": str(threshold_path),
            "sha256": _sha256(threshold_path),
        }
    for flag, pathkey in (
        ("training_split_verified", "training_split_evidence"),
        ("confirmation_untouched", "confirmation_evidence"),
    ):
        evidence_path = Path(evidence[pathkey]) if evidence.get(pathkey) else None
        metadata[flag] = bool(
            evidence.get(flag) and evidence_path and evidence_path.is_file()
        )
        if metadata[flag]:
            metadata[pathkey] = {
                "path": str(evidence_path),
                "sha256": _sha256(evidence_path),
            }
    metadata["validation_config_sha256"] = _sha256(Path(config["validation_config"]))
    metadata["cohort_inputs"] = {
        key: {
            "path": config["cohort"][key],
            "sha256": _sha256(Path(config["cohort"][key])),
        }
        for key in ("profiles", "audit_path", "reference_manifest")
        if config["cohort"].get(key)
    }
    metadata["data_manifest_sha256"] = readiness["checks"]["manifest"]["sha256"]
    climate_path = readiness["checks"].get("climatology", {}).get("path")
    if climate_path:
        metadata["climatology_sha256"] = _sha256(Path(climate_path))
    metadata["bundle_sha256"] = {
        name: _sha256(output / "bundle" / name)
        for name in ("points.csv", "members.npz")
    }
    metadata["workflow_sha256"] = {p.name: _sha256(p) for p in VAL.glob("*.py")}
    metadata["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    _write_json(output / "bundle" / "metadata.json", metadata)
    result = _score(output / "bundle", output / "report", config.get("statistics", {}))
    if hasattr(cohort, "coverage"):
        scored = (
            pd.read_csv(output / "bundle" / "points.csv")
            .drop_duplicates("profile_id")
            .groupby("region")
            .size()
            .rename("scored_profiles")
        )
        coverage = (
            cohort.coverage.set_index("region")
            .join(scored)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        coverage.to_csv(output / "report" / "coverage.csv", index=False)
        report_path = output / "report" / "report.html"
        panel = (
            "<h2>Eligible and scored regional coverage</h2><p>Regions with no scored observations supply no evidence of skill. This is observation-sampled coverage, not ocean-area coverage.</p>"
            + coverage.to_html(index=False)
        )
        report_path.write_text(
            report_path.read_text().replace("</body>", panel + "</body>")
        )
    return result


def _score(bundle: Path, output: Path, settings: dict | None = None) -> dict:
    """Score persisted ensemble values without rerunning model inference."""
    metadata = json.loads((bundle / "metadata.json").read_text())
    for name, digest in metadata.get("bundle_sha256", {}).items():
        if (
            name not in {"points.csv", "members.npz"}
            or _sha256(bundle / name) != digest
        ):
            raise ValueError(
                "Saved observations/members differ from their recorded hashes."
            )
    points = pd.read_csv(bundle / "points.csv")
    with np.load(bundle / "members.npz", allow_pickle=False) as arrays:
        members = arrays["residual"]
    return score_bundle(points, members, output, metadata=metadata, **(settings or {}))


def demo(output: Path) -> dict:
    """Exercise both physical variables with conspicuously synthetic data."""
    rng = np.random.default_rng(20261103)
    rows, ensembles = [], []
    for variable, scale in (("temperature", 1.0), ("salinity", 0.1)):
        for region, model_error in (("example_gain", 0.25), ("example_loss", 1.5)):
            for profile in range(60):
                cluster = f"{region}:cruise_{profile // 2}"
                for depth in (10.0, 75.0, 300.0):
                    observed = float(rng.normal() * scale)
                    identifier = f"demo:{variable}:{region}:{profile}"
                    rows.append(
                        dict(
                            point_id=f"{identifier}:{depth}",
                            profile_id=identifier,
                            cluster_id=cluster,
                            source="synthetic_demo",
                            region=region,
                            latitude=10.0,
                            longitude=20.0,
                            observation_time="2026-01-01T00:00:00Z",
                            target_time="2026-01-01T00:00:00Z",
                            depth_m=depth,
                            variable=variable,
                            observation=observed,
                            glorys=observed + scale,
                            climatology=observed + 0.8 * scale,
                            independence="unknown",
                            split_role="discovery",
                        )
                    )
                    ensembles.append(
                        observed + model_error * scale + rng.normal(0, 0.2 * scale, 20)
                    )
    bundle = output / "bundle"
    bundle.mkdir()
    pd.DataFrame(rows).to_csv(bundle / "points.csv", index=False)
    np.savez_compressed(bundle / "members.npz", residual=np.asarray(ensembles))
    _write_json(
        bundle / "metadata.json",
        {"demo": True, "purpose": "software demonstration, never scientific evidence"},
    )
    return _score(bundle, output / "report", {"bootstrap_replicates": 300})


def main() -> None:
    """Expose preflight, evaluation, reproducible scoring, and a data-free demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "evaluate", "demo", "score"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name in {"preflight", "evaluate"}:
            command.add_argument("--config", type=Path, required=True)
        if name == "score":
            command.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    output = _output(args.output)
    status = {
        "command": args.command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        if args.command == "demo":
            result = demo(output)
        elif args.command == "score":
            result = _score(args.bundle, output / "report")
        else:
            config = _config(args.config)
            (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            if args.command == "preflight":
                result = preflight(config)
                _write_json(output / "preflight.json", result)
                for issue in result["issues"]:
                    print(issue)
                print("Ready:", result["ready"])
            else:
                result = evaluate(config, output)
        status.update(status="complete", result=result)
        print(f"Outputs: {output}")
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
