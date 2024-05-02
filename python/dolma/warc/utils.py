import re
from typing import TYPE_CHECKING

from necessary import necessary

from ..core.errors import DolmaFatalError

with necessary("w3lib", soft=True) as W3LIB_AVAILABLE:
    if W3LIB_AVAILABLE or TYPE_CHECKING:
        from w3lib.url import canonicalize_url  # noqa: F401

with necessary("url_normalize", soft=True) as URL_NORMALIZE_AVAILABLE:
    if URL_NORMALIZE_AVAILABLE or TYPE_CHECKING:
        from url_normalize import url_normalize  # noqa: F401


def raise_warc_dependency_error(package: str):
    """Raise an error indicating that a package is required to run this processor."""
    raise DolmaFatalError(
        f"Package {package} is required to run this processor. "
        "Please install all dependencies with "
        "`pip install dolma[resilparse]` or `pip install dolma[trafilatura]`."
    )


class UrlNormalizer:
    def __init__(self):
        assert URL_NORMALIZE_AVAILABLE, raise_warc_dependency_error("url-normalize")
        assert W3LIB_AVAILABLE, raise_warc_dependency_error("w3lib")
        self.www_subdomain_regex = re.compile(r"(^(www\d*\.))|(/+$)", re.IGNORECASE)

    def __call__(self, url: str) -> str:
        # remove leading '<' or quotes and trailing '>', quotes, or slashes
        clean_url = re.sub(r"(^['\"<]+)|([/'\">]+$)", "", url)

        # canonicalize the URL
        canonical = canonicalize_url(clean_url)
        normalized = str(url_normalize(canonical))

        # remove the protocol
        _, normalized = normalized.split("://", 1)

        # remove the www subdomain
        normalized = self.www_subdomain_regex.sub("", normalized)

        return normalized
