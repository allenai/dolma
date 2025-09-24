"""
Runtime package version for dolma.

Resolves to the installed distribution version when available; falls back to
"0.0.0" in editable/dev contexts where the package metadata is not present.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version


def _get_version() -> str:
    try:
        return _dist_version("dolma")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _get_version()

__all__ = ["__version__"]
