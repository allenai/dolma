import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Type, Union

from necessary import necessary

from .utils import raise_dependency_error

with necessary("justext", soft=True) as JUSTEXT_AVAILABLE:
    if JUSTEXT_AVAILABLE or TYPE_CHECKING:
        from justext import justext

with necessary("trafilatura", soft=True) as TRAFILATURA_AVAILABLE:
    if TRAFILATURA_AVAILABLE or TYPE_CHECKING:
        import trafilatura
        import trafilatura.meta

with necessary("goose3", soft=True) as GOOSE3_AVAILABLE:
    if GOOSE3_AVAILABLE or TYPE_CHECKING:
        from goose3 import Goose


class BaseHtmlExtractor:
    """A base class for extractors. Turns HTML into text."""

    @abstractmethod
    def __call__(self, content: Union[str, bytes]) -> str:
        pass


class TrafilaturaHtmlExtractor(BaseHtmlExtractor):
    """An HTML extractor that uses trafilatura."""

    def __init__(
        self,
        include_comments: bool = False,
        include_links: bool = False,
        include_tables: bool = False,
        no_fallback: bool = False,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_formatting: bool = False,
        flush_after: int = 10_000,
    ) -> None:
        # make the trafilatura logger quiet
        logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

        # make sure trafilatura is available
        assert TRAFILATURA_AVAILABLE, raise_dependency_error("trafilatura")

        self.include_comments = include_comments
        self.include_links = include_links
        self.include_tables = include_tables
        self.no_fallback = no_fallback
        self.favor_precision = favor_precision
        self.favor_recall = favor_recall
        self.include_formatting = include_formatting

        # we reset caches every now and then to prevent memory leaks
        # see https://trafilatura.readthedocs.io/en/latest/usage-python.html#memory-use
        self.flush_after = flush_after
        self.counter = 0

    def _flush(self):
        if self.counter >= self.flush_after:
            trafilatura.meta.reset_caches()
            self.counter = 0
        self.counter += 1

    def __call__(self, content: Union[str, bytes]) -> str:
        output = trafilatura.extract(
            filecontent=content,
            output_format="txt",
            include_comments=self.include_comments,
            include_links=self.include_links,
            include_tables=self.include_tables,
            no_fallback=self.no_fallback,
            favor_precision=self.favor_precision,
            favor_recall=self.favor_recall,
            include_formatting=self.include_formatting,
        )
        self._flush()
        return output or ""


HTML_EXTRACTORS: Dict[str, Type[BaseHtmlExtractor]] = {
    "trafilatura": TrafilaturaHtmlExtractor,
}
