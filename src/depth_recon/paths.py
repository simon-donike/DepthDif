"""Package-relative paths for bundled DepthDif resources."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"


def config_path(*parts: str) -> Path:
    """Return a path inside the packaged configs directory."""
    return CONFIGS_DIR.joinpath(*parts)


def resolve_package_path(path: str | Path) -> Path:
    """Resolve source-tree package paths to this installed package directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.is_file():
        return candidate

    parts = candidate.parts
    if len(parts) >= 2 and parts[:2] == ("src", "depth_recon"):
        return PACKAGE_ROOT.joinpath(*parts[2:])
    if len(parts) >= 1 and parts[0] == "depth_recon":
        return PACKAGE_ROOT.joinpath(*parts[1:])
    return candidate


def resolve_config_path(path: str | Path) -> Path:
    """Resolve legacy config paths to the packaged configs directory."""
    candidate = resolve_package_path(path)
    if candidate.is_absolute() or candidate.is_file():
        return candidate

    parts = candidate.parts
    if len(parts) >= 3 and parts[:3] == ("src", "depth_recon", "configs"):
        return config_path(*parts[3:])
    if len(parts) >= 1 and parts[0] == "configs":
        return config_path(*parts[1:])
    return candidate
