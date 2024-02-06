import json
import warnings
from pathlib import Path
from typing import List, Optional, Union

import smart_open
import urllib3.util

# warning raised by pkg_resources used in a lot of google packages
warnings.filterwarnings("ignore", message=r".*declare_namespace\(\'.*google.*", category=DeprecationWarning)
# base warning raised when warning above are raised
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=DeprecationWarning)

# must import taggers to register them
# we import the rust extension here and wrap it in a python module
from . import dolma as _dolma  # type: ignore   # noqa: E402
from .core import TaggerRegistry  # noqa: E402
from .core.errors import DolmaRustPipelineError  # noqa: E402
from .core.taggers import BaseTagger  # noqa: E402
from .taggers import *  # noqa: E402

__all__ = [
    "add_tagger",
    "BaseTagger",
]

# we create a shortcut to easily add taggers to the registry
add_tagger = TaggerRegistry.add


def deduper(config: dict):
    """
    Run the deduper with the given configuration.

    Args:
        config (dict): The configuration for the deduper.

    Raises:
        DolmaRustPipelineError: If there is an error running the deduper.

    """
    try:
        _dolma.deduper_entrypoint(json.dumps(config))
    except RuntimeError as e:
        raise DolmaRustPipelineError(f"Error running deduper: {e}") from e


def mixer(config: dict):
    """
    Run the mixer with the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the mixer.

    Raises:
        DolmaRustPipelineError: If an error occurs while running the mixer.
    """
    try:
        _dolma.mixer_entrypoint(json.dumps(config))
    except RuntimeError as e:
        raise DolmaRustPipelineError(f"Error running mixer: {e}") from e


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
    def from_adblockplus_filepath(
        cls,
        adblockplus_filepath: Union[str, Path],
    ) -> "UrlBlocker":
        """
        Create an instance of UrlBlocker from an AdBlock Plus file.

        Args:
            adblockplus_filepath (str): The filepath of the AdBlock Plus file.

        Returns:
            UrlBlocker: An instance of UrlBlocker created from the AdBlock Plus file.

        """
        with smart_open.open(adblockplus_filepath, "rt") as adb_file:
            rules = [ln.strip() for ln in adb_file if not ln.startswith("!")]
        return cls(rules)

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
            source_url = parsed.host

        return self.engine.check_network_urls(
            url=url,
            source_url=source_url,
            request_type=request_type,
        )
