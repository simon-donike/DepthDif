"""Dataset assembly and baseline models for ocean depth reconstruction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ocean-depth-reconstruction-code")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
