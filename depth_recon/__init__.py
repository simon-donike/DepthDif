"""Public DepthDif inference API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("depth-recon")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__", "resolve_hf_assets", "run_week_inference"]


def __getattr__(name: str):
    """Lazily expose public inference helpers without importing the full stack."""
    if name in {"resolve_hf_assets", "run_week_inference"}:
        from inference.api import resolve_hf_assets, run_week_inference

        exports = {
            "resolve_hf_assets": resolve_hf_assets,
            "run_week_inference": run_week_inference,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
