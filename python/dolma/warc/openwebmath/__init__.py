from functools import partial
from typing import Optional

from necessary import necessary
from omegaconf import OmegaConf as om

from .config import OpenWebMathConfig

__all__ = ["Extractor", "OpenWebMathConfig"]


class Extractor:
    def __init__(self, config: Optional[OpenWebMathConfig] = None):
        necessary("tabulate", message="{module_name} not available; please install dolma[openwebmath]")
        necessary("py_asciimath", message="{module_name} not available; please install dolma[openwebmath]")
        necessary("lxml", message="{module_name} not available; please install dolma[openwebmath]")
        from .extract import extract_text

        parsed_config = om.to_object(config or om.structured(OpenWebMathConfig))
        self._extract_fn = partial(extract_text, config=parsed_config, fast=False)

    def extract_text(self, html: str) -> str:
        out = self._extract_fn(html)
        if isinstance(out, tuple):
            return str(out[0])

        return ""
