from ..core.errors import DolmaFatalError


def raise_warc_dependency_error(package: str):
    """Raise an error indicating that a package is required to run this processor."""
    raise DolmaFatalError(
        f"Package {package} is required to run this processor. "
        "Please install all dependencies with "
        "`pip install dolma[resilparse]` or `pip install dolma[trafilatura]`."
    )
