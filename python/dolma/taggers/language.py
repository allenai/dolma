"""
Language filters

@kylel, @soldni

"""
from typing import TYPE_CHECKING, Iterable, List, NamedTuple

import pycld2 as cld2
import regex
from anyascii import anyascii
from necessary import necessary

from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs

with necessary("cld3", soft=True) as CLD3_AVAILABLE:
    if CLD3_AVAILABLE or TYPE_CHECKING:
        import cld3  # pyright:ignore pylint:disable=import-error

with necessary("resiliparse", soft=True) as RESILIPARSE_AVAILABLE:
    if RESILIPARSE_AVAILABLE or TYPE_CHECKING:
        from resiliparse.parse.lang import (  # pyright:ignore pylint:disable=import-error
            detect_fast,
            supported_langs,
        )


class LanguagePrediction(NamedTuple):
    code: str
    conf: float


class BaseLanguageTagger(BaseTagger):
    ADD_NEGATIVE_SPANS: bool = False

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        raise NotImplementedError

    def _build_spans(self, text: str, matches: Iterable[LanguagePrediction], offset: int = 0) -> List[Span]:
        spans = [Span(start=offset, end=len(text) + offset, type=lang, score=score) for lang, score in matches]
        if self.ADD_NEGATIVE_SPANS:
            negs = [Span(start=s.start, end=s.end, type=f"not_{s.type}", score=1 - s.score) for s in spans]
            spans.extend(negs)
        return spans

    def predict(self, doc: Document) -> DocResult:
        matches = self.predict_text(doc.text)
        spans = self._build_spans(doc.text, matches)
        return DocResult(doc=doc, spans=spans)


class BaseParagraphLanguageTagger(BaseLanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            matches = self.predict_text(paragraph.text)
            paragraph_spans = self._build_spans(paragraph.text, matches, offset=paragraph.start)
            spans.extend(paragraph_spans)
        return DocResult(doc=doc, spans=spans)


class NullLanguageTagger(BaseLanguageTagger):
    """A language tagger that does nothing. Useful when you want to disable language tagging.
    Currently used for the `null` option in WARC the pipeline."""

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        return []


@TaggerRegistry.add("resiliparse_v1")
class ResiliparseLangIdTagger(BaseLanguageTagger):
    # from docs: max 101 languages detected
    top_k: int = len(supported_langs())

    # from docs: "The lower the value, the more accurate the prediction probably is.
    #             Values above 1200 are most likely false results."
    max_score: int = 1200

    def __init__(self) -> None:
        if not RESILIPARSE_AVAILABLE:
            raise ImportError(f"resiliparse is not install, cannot instantiate {self.__class__.__name__}")

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        pred = detect_fast(text, n_results=self.top_k, cutoff=self.max_score)
        return [
            LanguagePrediction(code=lang, conf=1 - (min(score, self.max_score) / self.max_score))
            for lang, score in pred
        ]


@TaggerRegistry.add("resiliparse_paragraph_v1")
class ResiliparseLangIdParagraphTagger(ResiliparseLangIdTagger, BaseParagraphLanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        return BaseParagraphLanguageTagger.predict(self, doc)


@TaggerRegistry.add("cld3_en_v2")
class Cld3LanguageTagger(BaseLanguageTagger):
    ADD_NEGATIVE_SPANS = True

    def __init__(self) -> None:
        if not CLD3_AVAILABLE:
            raise ImportError(f"cld3 is not install, cannot instantiate {self.__class__.__name__}")

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        raw_preds = cld3.get_language(text)
        return [
            LanguagePrediction(code=pred.language, conf=pred.probability) for pred in raw_preds if pred.is_reliable
        ]


@TaggerRegistry.add("cld3_en_doc_v2")
class Cld3EnglishLanguageTagger(Cld3LanguageTagger):
    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        preds = super().predict_text(text)
        en_pred = next((p for p in preds if p.code == "en"), LanguagePrediction(code="en", conf=0.0))
        return [en_pred]


@TaggerRegistry.add("cld3_en_paragraph_v2")
class Cld3EnglishLanguageParagraphTagger(Cld3EnglishLanguageTagger, BaseParagraphLanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        return BaseParagraphLanguageTagger.predict(self, doc)


@TaggerRegistry.add("cld2_en_v2")
class Cld2LanguageTagger(BaseLanguageTagger):
    ADD_NEGATIVE_SPANS = True
    RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

    def _sanitize_input(self, text: str) -> str:
        return self.RE_BAD_CHARS.sub("", text)

    def _to_ascii_input(self, text: str) -> str:
        return anyascii(text)

    def _identity_fn(self, text: str) -> str:
        return text

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        isReliable, textBytesFound, details = False, 0, []
        for fn in (self._identity_fn, self._to_ascii_input, self._sanitize_input):
            try:
                isReliable, textBytesFound, details = cld2.detect(fn(text))
                break
            except cld2.error:
                ...

        if not isReliable:
            return []

        return [
            LanguagePrediction(code=languageCode, conf=percent / 100.0)
            for languageName, languageCode, percent, score in details
        ]


@TaggerRegistry.add("cld2_en_doc_v2")
class Cld2EnglishLanguageTagger(Cld2LanguageTagger):
    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        preds = super().predict_text(text)
        en_pred = next((p for p in preds if p.code == "en"), LanguagePrediction(code="en", conf=0.0))
        return [en_pred]


@TaggerRegistry.add("cld2_en_paragraph_v2")
class Cld2EnglishLanguageParagraphTagger(Cld2EnglishLanguageTagger, BaseParagraphLanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        return BaseParagraphLanguageTagger.predict(self, doc)


@TaggerRegistry.add("ft_lang_id_doc_v1")
class FastTextLangIdTagger(BaseFastTextTagger, BaseLanguageTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        preds = self.classifier.predict(text.lower().replace("\n", " ").strip(), k=-1)
        return [
            LanguagePrediction(code=label.replace("__label__", ""), conf=score) for label, score in zip(*preds)
        ]

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        preds = self.predict_text(text_slice.text)
        return [Prediction(label=pred.code, score=pred.conf) for pred in preds]


@TaggerRegistry.add("ft_lang_id_paragraph_v1")
class FastTextLangIdParagraphTagger(FastTextLangIdTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)


@TaggerRegistry.add("ft_lang_id_en_doc_v2")
class FastTextEnglishLanguageDocumentTagger(FastTextLangIdTagger):
    ADD_NEGATIVE_SPANS = True

    def predict_text(self, text: str) -> Iterable[LanguagePrediction]:
        preds = super().predict_text(text)
        en_pred = next((p for p in preds if p.code == "en"), LanguagePrediction(code="en", conf=0.0))
        if self.ADD_NEGATIVE_SPANS:
            neg_pred = LanguagePrediction(code=f"not_{en_pred.code}", conf=1 - en_pred.conf)
            return [en_pred, neg_pred]
        return [en_pred]


@TaggerRegistry.add("ft_lang_id_en_paragraph_v2")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger, FastTextLangIdParagraphTagger):
    def __init__(self):
        FastTextLangIdParagraphTagger.__init__(self)


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
class Cld3LanguageFilterParagraphWithDocScoreTagger(Cld2EnglishLanguageParagraphTagger):
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


# @TaggerRegistry.add("langdetect_en_doc_v1")
# class LangdetectTagger(BaseTagger):
#     def __init__(self) -> None:
#         (factory := DetectorFactory()).load_profile(PROFILES_DIRECTORY)
#         factory.set_seed(0)
#         self.detector = factory.create()
#         super().__init__()

#     # document-level, english / not english
#     def _predict_text(self, text: str) -> Tuple[str, float]:
#         try:
#             self.detector.append(text)
#             langs = self.detector.get_probabilities()
#             score, *_ = ([lang.prob for lang in langs if lang.lang == "en"] or [0.0])
#         except LangDetectException:
#             score = 0.0
#         finally:
#             self.detector.text = ""
#             self.detector.langprob = None

#         return "en", score

#     def predict(self, doc: Document) -> DocResult:
#         lang, score = self._predict_text(doc.text)
#         positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
#         negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
#         return DocResult(doc=doc, spans=[positive_span, negative_span])
