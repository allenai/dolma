"""

Code-related taggers.

@akshitab

"""

import logging
import re
from typing import List

import numpy as np
from necessary import necessary

from ...core.data_types import DocResult, Document, DocumentWithMetadata, Span
from ...core.registry import TaggerRegistry
from ...core.taggers import BaseTagger, BaseTaggerWithMetadata

with necessary(["detect_secrets", "bs4", "regex", "pygments"], soft=True) as CODE_DEPENDENCIES_AVAILABLE:
    if CODE_DEPENDENCIES_AVAILABLE:
        from .starcoder import get_nl_ratio
        from .utils import (
            filter_html,
            get_ext_to_lang_mapping,
            get_secrets,
            get_whitespace_regex,
        )


logger = logging.getLogger(__name__)


def check_code_dependencies() -> None:
    """Check if code dependencies are available."""
    if not CODE_DEPENDENCIES_AVAILABLE:
        raise RuntimeError("Code dependencies are not available; please run `pip install dolma[code]`")


@TaggerRegistry.add("code_secrets_v1")
class CodeSecretsTagger(BaseTagger):
    def __init__(self) -> None:
        check_code_dependencies()
        super().__init__()

    @classmethod
    def _extract_code_secrets(cls, text: str) -> List[Span]:
        secrets_spans: List[Span] = []

        text_lines = text.splitlines()
        secrets = get_secrets(text)
        for _, secret in secrets:
            line_number = secret.line_number - 1
            span = secret.secret_value
            span_line = text_lines[line_number]
            line_start = text.find(span_line)
            start = line_start + span_line.find(span or "")
            end = start + len(span or "")
            assert text[start:end] == span
            secret_type = secret.type.replace(" ", "_")
            secrets_spans.append(Span(start=start, end=end, type=f"SECRET_{secret_type}"))  # , span])

        return secrets_spans

    def predict(self, doc: Document) -> DocResult:
        """Main runner."""
        spans = self._extract_code_secrets(doc.text)

        # document-level score
        score = self._score(text=doc.text, secrets_spans=spans)
        spans.append(Span(start=0, end=len(doc.text), type="doc", score=score))
        return DocResult(doc=doc, spans=spans)

    def _score(self, text: str, secrets_spans: List[Span]) -> float:
        try:
            score = len(secrets_spans) * 1.0 / len(text.split())
        except ZeroDivisionError:
            score = -1.0
        return score


@TaggerRegistry.add("code_copyright_comments_v1")
class CodeCopyrightTagger(BaseTagger):
    """
    Based on RedPajama code filtering.
    """

    def __init__(self):
        check_code_dependencies()
        self.cpat = re.compile("copyright", re.IGNORECASE)
        self.pat = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")

    def _extract_copyright_spans(self, text: str) -> List[Span]:
        copyright_spans: List[Span] = []

        reg = self.pat.search(text)

        if reg:
            # found one, now see if it contains "copyright", if so strip it
            span = reg.span()
            sub = text[span[0] : span[1]]
            if self.cpat.search(sub):
                copyright_spans.append(Span(start=span[0], end=span[1], type="copyright_notice", score=1.0))
            return copyright_spans

        lines = text.split("\n")
        skip = 0
        # Greedy replace any file that begins with comment block, most
        # are copyright headers
        end = 0
        for line in lines:
            if line.startswith("//") or line.startswith("#") or line.startswith("--") or not line:
                skip = skip + 1
                if not line:
                    end += 1
                else:
                    end += len(line)
            else:
                break

        if skip:
            copyright_spans.append(Span(start=0, end=end, type="comment_block", score=1.0))
        return copyright_spans

    def predict(self, doc: Document) -> DocResult:
        """Main runner."""
        spans = self._extract_copyright_spans(doc.text)

        # document-level score
        score = self._score(text=doc.text, copyright_spans=spans)
        spans.append(Span(start=0, end=len(doc.text), type="doc", score=score))
        return DocResult(doc=doc, spans=spans)

    def _score(self, text: str, copyright_spans: List[Span]) -> float:
        try:
            if len(copyright_spans) == 0:
                score = 0.0
            else:
                span = copyright_spans[0]
                # percentage of content affected
                score = (span.end - span.start + 1) * 1.0 / len(text)
        except ZeroDivisionError:
            score = -1.0
        return score


@TaggerRegistry.add("code_redpajama_taggers_v1")
class CodeRedPajamaTaggers(BaseTagger):
    """
    Based on RedPajama code filtering.
    """

    def __init__(self):
        check_code_dependencies()
        self.whitespace_regex = get_whitespace_regex()
        super().__init__()

    def _get_num_tokens(self, text: str) -> int:
        return len(self.whitespace_regex.split(text))

    def predict(self, doc: Document) -> DocResult:
        """Main runner."""

        spans: List[Span] = []

        doc_length = len(doc.text)

        line_lengths = list(map(len, doc.text.splitlines()))

        max_line_length = max(line_lengths, default=0.0)
        avg_line_length = float(np.mean(line_lengths) if len(line_lengths) > 0 else 0.0)

        alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, doc.text))
        alnum_prop = (alnum_count / doc_length) if doc_length > 0 else 0.0

        num_tokens = self._get_num_tokens(doc.text)
        num_alpha = len([c for c in doc.text if c.isalpha()])
        alpha_token_prop = float((num_alpha / num_tokens) if num_tokens > 0 else 0.0)

        # document-level scores
        spans.append(Span(start=0, end=doc_length, type="max_line_length_doc", score=max_line_length))
        spans.append(Span(start=0, end=doc_length, type="avg_line_length_doc", score=avg_line_length))
        spans.append(Span(start=0, end=doc_length, type="alnum_prop_doc", score=alnum_prop))
        spans.append(Span(start=0, end=doc_length, type="alpha_token_prop_doc", score=alpha_token_prop))

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("code_starcoder_taggers_v1")
class CodeStarCoderTaggers(BaseTaggerWithMetadata):
    """
    Based on StarCoder filtering.
    """

    def __init__(self) -> None:
        check_code_dependencies()
        self.ext_to_lang_mapping = get_ext_to_lang_mapping()
        super().__init__()

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        spans: List[Span] = []
        doc_length = len(doc.text)

        has_xml_template = 1.0 if "<?xml version=" in doc.text[:100] else 0.0
        num_github_stars = doc.metadata.get("max_stars_count", 0) or 0

        try:
            lang = self.ext_to_lang_mapping[doc.metadata.get("ext", "-no-lang")]
            nl_ratio = get_nl_ratio(doc.text, lang)

            if lang == "html":
                code_to_text_ratio = filter_html(doc.text)
            else:
                # Not relevant for non-html code
                code_to_text_ratio = 1.0
        except:  # pylint: disable=bare-except   # noqa: E722
            nl_ratio = -1.0
            code_to_text_ratio = -1.0

        # document-level scores
        spans.append(Span(start=0, end=doc_length, type="has_xml_template_doc", score=has_xml_template))
        spans.append(Span(start=0, end=doc_length, type="num_github_stars_doc", score=float(num_github_stars)))
        spans.append(Span(start=0, end=doc_length, type="nl_ratio_doc", score=nl_ratio))
        spans.append(Span(start=0, end=doc_length, type="code_to_text_ratio_html_doc", score=code_to_text_ratio))

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("code_starcoder_taggers_v2")
class CodeStarCoderTaggers2(BaseTaggerWithMetadata):
    """
    Based on StarCoder filtering.
    """

    def __init__(self) -> None:
        check_code_dependencies()
        self.ext_to_lang_mapping = get_ext_to_lang_mapping()
        super().__init__()

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        spans: List[Span] = []
        doc_length = len(doc.text)

        has_xml_template = 1.0 if "<?xml version=" in doc.text[:100] else 0.0
        num_github_stars = doc.metadata.get("max_stars_count", 0) or 0

        try:
            lang = self.ext_to_lang_mapping[doc.metadata.get("ext", "-no-lang")]
        except KeyError:
            lang = "-no-lang"

        if lang in ["python", "java", "javascript"]:
            code_to_comment_ratio = get_nl_ratio(doc.text, lang)
        else:
            code_to_comment_ratio = 0.5  # We use an upper and lower bound of filters; this is in the middle.

        if lang == "html":
            try:
                code_to_text_ratio = filter_html(doc.text)
            except:  # pylint: disable=bare-except   # noqa: E722
                code_to_text_ratio = -1.0
        else:
            code_to_text_ratio = 1.0

        # document-level scores
        spans.append(Span(start=0, end=doc_length, type="has_xml_template_doc", score=has_xml_template))
        spans.append(Span(start=0, end=doc_length, type="num_github_stars_doc", score=float(num_github_stars)))
        spans.append(Span(start=0, end=doc_length, type="code_to_comment_ratio_doc", score=code_to_comment_ratio))
        spans.append(Span(start=0, end=doc_length, type="code_to_text_ratio_html_doc", score=code_to_text_ratio))

        return DocResult(doc=doc, spans=spans)
