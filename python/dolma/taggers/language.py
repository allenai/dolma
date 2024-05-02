"""
Language Detection

@kylel, @soldni
"""

from typing import TYPE_CHECKING, List, Tuple

import necessary
import regex
from anyascii import anyascii

from ..core.data_types import DocResult, Document, Span
from ..core.ft_tagger import BaseFastTextTagger
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs

with necessary.necessary("cld3", soft=True) as CLD3_AVAILABLE:
    if CLD3_AVAILABLE or TYPE_CHECKING:
        import cld3  # pyright:ignore pylint:disable=import-error

with necessary.necessary("pycld2", soft=True) as CLD2_AVAILABLE:
    if CLD2_AVAILABLE or TYPE_CHECKING:
        import pycld2 as cld2  # pyright:ignore pylint:disable=import-error


with necessary.necessary("langdetect", soft=True) as LANGDETECT_AVAILABLE:
    if LANGDETECT_AVAILABLE or TYPE_CHECKING:
        from langdetect import PROFILES_DIRECTORY, DetectorFactory, LangDetectException


with necessary.necessary("lingua", soft=True) as LINGUA_AVAILABLE:
    if LINGUA_AVAILABLE or TYPE_CHECKING:
        from lingua import Language, LanguageDetectorBuilder


class BaseLanguageTagger(BaseTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        return []

    def make_negative(self, spans: List[Span]) -> List[Span]:
        return [
            Span(start=span.start, end=span.end, type=f"not_{span.type}", score=1.0 - span.score) for span in spans
        ]

    def predict_doc(self, doc: Document) -> DocResult:
        spans = [
            Span(start=0, end=len(doc.text), type=str(lang), score=score)
            for lang, score in self.predict_text(doc.text)
        ]
        return DocResult(doc=doc, spans=spans)

    def predict_paragraph(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            spans.extend(
                Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
                for lang, score in self.predict_text(paragraph.text)
            )
        return DocResult(doc=doc, spans=spans)

    def predict(self, doc: Document) -> DocResult:
        doc_result = self.predict_paragraph(doc) if self.PREDICT_ON_PARAGRAPHS else self.predict_doc(doc)
        if self.INCLUDE_NEGATIVE:
            doc_result.spans.extend(self.make_negative(doc_result.spans))
        return doc_result


@TaggerRegistry.add("cld3_en_doc_v2")
class Cld3LanguageTagger(BaseLanguageTagger):
    def __init__(self) -> None:
        super().__init__()
        if not CLD3_AVAILABLE:
            raise ImportError(f"cld3 is not installed, cannot instantiate {self.__class__.__name__}")

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        pred = cld3.get_language(text)  # pyright: ignore
        score = pred.probability if pred.language == "en" else 0.0
        return [("en", score)]


@TaggerRegistry.add("cld3_en_paragraph_v2")
class Cld3LanguageTaggerParagraph(Cld3LanguageTagger):
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("cld2_doc_v2")
class Cld2LanguageTagger(BaseLanguageTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False
    RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

    def __init__(self) -> None:
        super().__init__()
        if not CLD2_AVAILABLE:
            raise ImportError("pycld2 is not installed, please run `pip install dolma[lang]`.")

    def _sanitize_input(self, text: str) -> str:
        return self.RE_BAD_CHARS.sub("", text)

    def _to_ascii_input(self, text: str) -> str:
        return anyascii(text)

    def _identity_fn(self, text: str) -> str:
        return text

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        details = []
        is_reliable = False
        for fn in (self._identity_fn, self._to_ascii_input, self._sanitize_input):
            try:
                is_reliable, _, details = cld2.detect(fn(text))
                break
            except cld2.error:
                ...
        return [(d[0][:2].lower(), d[2] / 100.0) for d in details if d[0] != "UNKNOWN_LANGUAGE" and is_reliable]


@TaggerRegistry.add("cld2_paragraph_v2")
class Cld2LanguageTaggerParagraph(Cld2LanguageTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("cld2_en_doc_v2")
class Cld2EnglishLanguageTagger(Cld2LanguageTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        pred = super().predict_text(text)
        filtered_preds = [(lang, score) for lang, score in pred if lang == "en"] or [("en", 0.0)]
        return filtered_preds  # pyright: ignore


@TaggerRegistry.add("cld2_en_paragraph_v2")
class Cld2EnglishLanguageParagraphTagger(Cld2EnglishLanguageTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("ft_lang_id_doc_v1")
class FastTextAllLanguagesDocumentTagger(BaseLanguageTagger, BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False

    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        preds = self.classifier.predict(text.lower().replace("\n", " ").strip(), k=-1)
        return [(label.replace("__label__", ""), float(score)) for label, score in zip(*preds)]


@TaggerRegistry.add("ft_lang_id_1e2")
class FastTextAllLanguagesDocumentMinScoreTagger(FastTextAllLanguagesDocumentTagger):
    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        out = super().predict_text(text)
        return [(lang, round(score, 2)) for lang, score in out if score > 0.01]


@TaggerRegistry.add("ft_lang_id_paragraph_v1")
class FastTextAllLanguageParagraphTagger(FastTextAllLanguagesDocumentTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("ft_lang_id_en_doc_v2")
class FastTextEnglishLanguageDocumentTagger(FastTextAllLanguagesDocumentTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        preds = super().predict_text(text)
        filtered_preds = [(lang, score) for lang, score in preds if lang == "en"] or [("en", 0.0)]
        return filtered_preds  # pyright: ignore


@TaggerRegistry.add("ft_lang_id_en_only_v2")
class FastTextEnglishOnlyLanguageDocumentTagger(FastTextEnglishLanguageDocumentTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False


@TaggerRegistry.add("ft_lang_id_en_paragraph_v2")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("langdetect_doc_v1")
class LangdetectTagger(BaseLanguageTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False

    def __init__(self) -> None:
        if not LANGDETECT_AVAILABLE:
            raise ImportError("langdetect is not installed, please run `pip install dolma[lang]`.")

        (factory := DetectorFactory()).load_profile(PROFILES_DIRECTORY)
        factory.set_seed(0)
        self.detector = factory.create()
        super().__init__()

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        try:
            self.detector.append(text)
            langs = self.detector.get_probabilities()
            output = [(str(r.lang.strip().lower()), float(r.prob)) for r in langs]
        except LangDetectException:
            output = []
        finally:
            self.detector.text = ""
            self.detector.langprob = None
        return output


@TaggerRegistry.add("langdetect_doc_en_v1")
class LangdetectEnglishTagger(LangdetectTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        langs = super().predict_text(text)
        filtered_langs = [(lang, score) for lang, score in langs if lang == "en"] or [("en", 0.0)]
        return filtered_langs  # pyright: ignore


@TaggerRegistry.add("langdetect_paragraph_v1")
class LangdetectTaggerParagraph(LangdetectTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("langdetect_en_paragraph_v1")
class LangdetectEnglishTaggerParagraph(LangdetectEnglishTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = True


@TaggerRegistry.add("lingua_doc_v1")
class LinguaTagger(BaseLanguageTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False

    def __init__(self) -> None:
        super().__init__()
        if not LANGDETECT_AVAILABLE:
            raise ImportError("langdetect is not installed, please run `pip install dolma[lang]`.")
        self.detector = LanguageDetectorBuilder.from_languages(*Language.all()).build()

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        langs_conf = self.detector.compute_language_confidence_values(text) or []
        return [(lang.language.iso_code_639_1.name.lower(), float(lang.value)) for lang in langs_conf]


@TaggerRegistry.add("lingua_1e2")
class LinguaMinScoreTagger(LinguaTagger):
    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        out = super().predict_text(text)
        return [(lang, round(score, 2)) for lang, score in out if score > 0.01]


@TaggerRegistry.add("lingua_doc_en_v1")
class LinguaEnglishTagger(LinguaTagger):
    INCLUDE_NEGATIVE = True
    PREDICT_ON_PARAGRAPHS = False

    def predict_text(self, text: str) -> List[Tuple[str, float]]:
        pred = super().predict_text(text)
        filtered_langs = [(lang, score) for lang, score in pred if lang == "en"] or [("en", 0.0)]
        return filtered_langs  # pyright: ignore


@TaggerRegistry.add("lingua_en_only_v1")
class LinguaEnglishOnlyTagger(LinguaEnglishTagger):
    INCLUDE_NEGATIVE = False
    PREDICT_ON_PARAGRAPHS = False


@TaggerRegistry.add("lingua_par_v1")
class LinguaTaggerParagraph(LinguaTagger):
    PREDICT_ON_PARAGRAPHS = True
    INCLUDE_NEGATIVE = False


@TaggerRegistry.add("lingua_en_par_v1")
class LinguaEnglishTaggerParagraph(LinguaEnglishTagger):
    PREDICT_ON_PARAGRAPHS = True
    INCLUDE_NEGATIVE = True


def add_global_language_score_from_slice_score(result: DocResult) -> DocResult:
    # the total document score is # of characters in each "english" span multiplied by the likelihood
    # of said span being english
    try:
        doc_en_score = sum((s.end - s.start) * s.score for s in result.spans if s.type == "en") / len(
            result.doc.text
        )
        doc_not_en_score = 1 - doc_en_score
    except ZeroDivisionError:
        doc_en_score = doc_not_en_score = 0.0

    doc_level = (
        Span(start=0, end=len(result.doc.text), type="doc_en", score=doc_en_score),
        Span(start=0, end=len(result.doc.text), type="doc_not_en", score=doc_not_en_score),
    )
    result.spans.extend(doc_level)
    return result


@TaggerRegistry.add("cld2_en_paragraph_with_doc_score_v2")
class Cld2LanguageFilterParagraphWithDocScoreTagger(Cld2EnglishLanguageParagraphTagger):
    def predict(self, doc: Document) -> DocResult:
        doc_result = super().predict(doc)
        doc_result = add_global_language_score_from_slice_score(doc_result)
        return doc_result


@TaggerRegistry.add("cld3_en_paragraph_with_doc_score_v2")
class Cld3LanguageFilterParagraphWithDocScoreTagger(Cld3LanguageTaggerParagraph):
    def predict(self, doc: Document) -> DocResult:
        doc_result = super().predict(doc)
        doc_result = add_global_language_score_from_slice_score(doc_result)
        return doc_result


@TaggerRegistry.add("ft_lang_id_en_paragraph_with_doc_score_v2")
class FastTextEnglishLanguageParagraphWithDocScoreTagger(FastTextEnglishLanguageParagraphTagger):
    def predict(self, doc: Document) -> DocResult:
        doc_result = super().predict(doc)
        doc_result = add_global_language_score_from_slice_score(doc_result)
        return doc_result
