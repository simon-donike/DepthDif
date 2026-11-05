"""Resolve the shared, frozen baseline registry without changing training files."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import yaml


def check_artifact(path: str | Path, expected: str | None = None) -> str:
    """Require a baseline input and verify its optional frozen SHA256."""
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and expected != digest:
        raise ValueError(f"Baseline SHA256 mismatch: {path}")
    return digest


def registered_baselines(config: dict) -> list[dict]:
    """Share one baseline roster across standalone evaluations and campaigns."""
    result = deepcopy(config.get("baselines", []))
    registry_path = config.get("baseline_registry")
    if registry_path:
        registry = yaml.safe_load(Path(registry_path).read_text())
        fields = config["sampling"]["variables"]
        scenario = fields[0] if len(fields) == 1 else "joint"
        if scenario not in registry:
            raise ValueError(f"Baseline registry has no {scenario} checkpoints")
        result = deepcopy(registry[scenario]) + result
        for spec in result:
            if spec.get("inherit_model_config"):
                spec["config"] = config["model"]["config"]
                spec["scenario"] = scenario
    unique = {}
    for spec in result:
        name = spec["name"]
        if name in unique and unique[name] != spec:
            raise ValueError("Baseline names must be unique across registry and config")
        # Resolved campaign snapshots contain both the registry path and its
        # expanded roster. Reading that snapshot must preserve the same methods.
        unique[name] = spec
    return list(unique.values())


def materialize_config(spec: dict, output: Path) -> Path:
    """Combine archived effective snapshots, preserving their resolved architecture."""
    sources = spec.get("effective_configs")
    if not sources:
        return Path(spec["config"])
    sections = {}
    for name in ("data", "model", "training"):
        source = sources[name]
        check_artifact(source["path"], source["sha256"])
        value = yaml.safe_load(Path(source["path"]).read_text())
        sections[name] = value["model"] if name == "model" else value
    # Old super-config snapshots can contain the scratch diffusion defaults;
    # the effective snapshots record the actual trained baseline and scenario.
    sections["scenario"] = sections["model"]["scenario"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "baseline_super_config.yaml"
    path.write_text(yaml.safe_dump(sections, sort_keys=False))
    return path
