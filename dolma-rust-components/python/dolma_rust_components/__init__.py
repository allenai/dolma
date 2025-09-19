"""
dolma-rust-components - Rust components for Dolma

This package provides the low-level Rust implementations used by the Dolma toolkit.
"""

import json
from . import dolma_rust_components as _rust  # type: ignore


# expose the rust version to be used by the python package
__version__ = _rust.get_version()


class DolmaRustPipelineError(Exception):
    """Exception raised for errors in Dolma Rust pipeline operations."""
    pass


def deduper(config: dict):
    """
    Run the deduper with the given configuration.

    Args:
        config (dict): The configuration for the deduper.

    Raises:
        DolmaRustPipelineError: If there is an error running the deduper.
    """
    try:
        _rust.deduper_entrypoint(json.dumps(config))
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
        _rust.mixer_entrypoint(json.dumps(config))
    except RuntimeError as e:
        raise DolmaRustPipelineError(f"Error running mixer: {e}") from e


class UrlBlocker:
    """
    URL blocker using adblock rules.

    This is a wrapper around the Rust UrlBlocker class.
    """
    def __init__(self, rules: list[str]):
        """
        Initialize the URL blocker with the given rules.

        Args:
            rules: List of adblock rules as strings.
        """
        self._blocker = _rust.UrlBlocker(rules)

    def check_network_urls(self, url: str, source_url: str, request_type: str) -> bool:
        """
        Check whether a network request should be blocked.

        Args:
            url: The URL being requested.
            source_url: The URL of the page making the request.
            request_type: The type of request (e.g., 'image', 'script', etc.).

        Returns:
            True if the request should be blocked, False otherwise.
        """
        return self._blocker.check_network_urls(url, source_url, request_type)


__all__ = [
    "deduper",
    "mixer",
    "UrlBlocker",
    "DolmaRustPipelineError",
]
