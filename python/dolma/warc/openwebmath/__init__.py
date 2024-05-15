from functools import partial
from typing import Optional

from necessary import necessary

from .utils import Config

__all__ = ["Extractor", "Config"]


class Extractor:
    def __init__(self, config: Optional[Config] = None):
        necessary("tabulate", message="{module_name} not available; please install dolma[openwebmath]")
        necessary("py_asciimath", message="{module_name} not available; please install dolma[openwebmath]")
        necessary("lxml", message="{module_name} not available; please install dolma[openwebmath]")
        from .extract import extract_text

        # create a config, merge it with empty dictionary to make sure it is casted to a python dict
        config = config or Config()

        self._extract_fn = partial(extract_text, config=config.sample(), fast=False)

    def extract_text(self, html: str) -> str:
        out = self._extract_fn(html)
        if isinstance(out, tuple):
            return str(out[0])

        return ""
