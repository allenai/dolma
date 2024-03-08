from pathlib import Path
from typing import List, Optional, Union

import smart_open
import urllib3.util

from .. import dolma as _dolma  # type: ignore   # noqa: E402


class UrlBlocker:
    """
    A class that provides URL blocking functionality based on a set of rules.

    Args:
        rules (List[str]): A list of rules to be used for blocking URLs.

    Attributes:
        engine: The underlying engine used for URL blocking.

    Methods:
        from_adblockplus_filepath: Create an instance of UrlBlocker from an AdBlock Plus file.
        check_network_urls: Check if a given URL should be blocked based on the rules.

    """

    def __init__(
        self,
        rules: List[str],
    ) -> None:
        """
        Initialize the UrlBlocker instance.

        Args:
            rules (List[str]): A list of rules to be used for blocking URLs.

        """
        self.engine = _dolma.UrlBlocker(rules=rules)

    @classmethod
    def from_adb_paths(
        cls,
        *file_paths: Union[str, Path],
    ) -> "UrlBlocker":
        """
        Create an instance of UrlBlocker from one or more AdBlock Plus files.

        Args:
            file_paths (Union[str, Path]): The filepath of the AdBlock Plus file.

        Returns:
            UrlBlocker: An instance of UrlBlocker created from the AdBlock Plus file.

        """
        rules = []
        for fp in file_paths:
            with smart_open.open(fp, "rt") as adb_file:
                rules.extend([ln.strip() for ln in adb_file if not ln.startswith("!")])
        return cls(sorted(set(rules)))

    def check_network_urls(
        self,
        url: str,
        source_url: Optional[str] = None,
        request_type: str = "",
    ) -> bool:
        """
        Check if a given URL should be blocked based on the rules.

        Args:
            url (str): The URL to be checked.
            source_url (str): The source URL of the request. If not provided, the host from the URL will be used.
            request_type (str): The type of the request. For a list of valid request types, see the adblockplus
                documentation: https://help.adblockplus.org/hc/en-us/articles/360062733293-How-to-write-filters

        Returns:
            bool: True if the URL should be blocked, False otherwise.

        """
        parsed = urllib3.util.parse_url(url)
        if parsed.scheme is None:
            # if the URL does not have a scheme, we assume it is an HTTP URL
            url = f"http://{url}"

        if source_url is None:
            # if the source URL is not provided, we use the host from the URL
            source_url = ""

        return self.engine.check_network_urls(
            url=str(url),
            source_url=str(source_url),
            request_type=request_type,
        )
