from ..errors import DolmaFatalError


def raise_dependency_error(package: str):
    """Raise an error indicating that a package is required to run this processor."""
    raise DolmaFatalError(
        f"Package {package} is required to run this processor. "
        f"Please install it with `pip install dolma[warc]`."
    )
