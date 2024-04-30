import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union

from necessary import necessary

from ..core.registry import BaseRegistry
from .utils import raise_warc_dependency_error

with necessary("trafilatura", soft=True) as TRAFILATURA_AVAILABLE:
    if TRAFILATURA_AVAILABLE or TYPE_CHECKING:
        import trafilatura  # noqa: F401
        import trafilatura.meta  # noqa: F401

with necessary("resiliparse", soft=True) as RESILIPARSE_AVAILABLE:
    if RESILIPARSE_AVAILABLE or TYPE_CHECKING:
        from resiliparse.extract.html2text import extract_plain_text  # noqa: F401
        from resiliparse.parse.encoding import detect_encoding  # noqa: F401
        from resiliparse.parse.html import HTMLTree  # noqa: F401


class BaseLinearizer:
    """A base class for linearizers, i.e. tools to turn HTML into text."""

    @abstractmethod
    def linearize(self, content: Union[str, bytes]) -> str:
        pass


class LinearizerRegistry(BaseRegistry[Type[BaseLinearizer]]):
    pass


@LinearizerRegistry.add("resiliparse")
class ResiliparseHtmlExtractor(BaseLinearizer):
    def __init__(
        self,
        preserve_formatting: bool = True,
        main_content: bool = True,
        list_bullets: bool = True,
        alt_texts: bool = False,
        links: bool = False,
        form_fields: bool = False,
        noscript: bool = False,
        comments: bool = True,
        skip_elements: Optional[Iterable[str]] = None,
    ) -> None:
        assert RESILIPARSE_AVAILABLE, raise_warc_dependency_error("resiliparse")

        self.preserve_formatting = preserve_formatting
        self.main_content = main_content
        self.list_bullets = list_bullets
        self.alt_texts = alt_texts
        self.links = links
        self.form_fields = form_fields
        self.noscript = noscript
        self.comments = comments
        self.skip_elements = skip_elements

    def linearize(self, content: Union[str, bytes]) -> str:
        #         html (HTMLTree or str) – HTML as DOM tree or Unicode string
        # preserve_formatting (bool) – preserve basic block-level formatting
        # main_content (bool) – apply simple heuristics for extracting only “main-content” elements
        # list_bullets (bool) – insert bullets / numbers for list items
        # alt_texts (bool) – preserve alternative text descriptions
        # links (bool) – extract link target URLs
        # form_fields (bool) – extract form fields and their values
        # noscript (bool) – extract contents of <noscript> elements
        # comments (bool) – treat comment sections as main content
        # skip_elements (t.Iterable[str] or None) – list of CSS selectors for elements to skip
        if isinstance(content, bytes):
            encoding = detect_encoding(content)
            html = HTMLTree.parse_from_bytes(content, encoding)
        else:
            html = HTMLTree.parse(content)

        text = extract_plain_text(
            html=html,
            preserve_formatting=self.preserve_formatting,
            main_content=self.main_content,
            list_bullets=self.list_bullets,
            alt_texts=self.alt_texts,
            links=self.links,
            form_fields=self.form_fields,
            noscript=self.noscript,
            comments=self.comments,
            skip_elements=self.skip_elements,
        )
        return text


@LinearizerRegistry.add("trafilatura")
class TrafilaturaHtmlExtractor(BaseLinearizer):
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
        assert TRAFILATURA_AVAILABLE, raise_warc_dependency_error("trafilatura")

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

    def linearize(self, content: Union[str, bytes]) -> str:
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
