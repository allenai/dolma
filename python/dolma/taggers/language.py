"""

Filters.

@kylel, @soldni

"""

from typing import TYPE_CHECKING, Iterable, List, Tuple

import necessary
import pycld2 as cld2
import regex
from anyascii import anyascii

from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs

with necessary.necessary("cld3", soft=True) as CLD3_AVAILABLE:
    if CLD3_AVAILABLE or TYPE_CHECKING:
        import cld3  # pyright:ignore pylint:disable=import-error


@TaggerRegistry.add("cld3_en_doc_v2")
class Cld3LanguageTagger(BaseTagger):
    def __init__(self) -> None:
        if not CLD3_AVAILABLE:
            raise ImportError(f"cld3 is not install, cannot instantiate {self.__class__.__name__}")

    def _predict_text(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)  # pyright: ignore
        score = pred.probability if pred.language == "en" else 0.0
        return "en", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
        negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
        return DocResult(doc=doc, spans=[positive_span, negative_span])


@TaggerRegistry.add("cld3_en_paragraph_v2")
class Cld3LanguageTaggerParagraph(Cld3LanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("cld2_en_doc_v2")
class Cld2LanguageFilter(BaseTagger):
    RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

    def _sanitize_input(self, text: str) -> str:
        return self.RE_BAD_CHARS.sub("", text)

    def _to_ascii_input(self, text: str) -> str:
        return anyascii(text)

    def _identity_fn(self, text: str) -> str:
        return text

    def _predict_text(self, text: str) -> Tuple[str, float]:
        details = []
        is_reliable = False
        for fn in (self._identity_fn, self._to_ascii_input, self._sanitize_input):
            try:
                is_reliable, _, details = cld2.detect(fn(text))
                break
            except cld2.error:
                ...

        score = max([d[2] for d in details if d[0] == "ENGLISH" and is_reliable] or [0])
        return "en", score / 100.0

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
        negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
        return DocResult(doc=doc, spans=[positive_span, negative_span])


@TaggerRegistry.add("cld2_en_paragraph_v2")
class Cld2LanguageFilterParagraph(Cld2LanguageFilter):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("ft_lang_id_doc_v1")
class FastTextAllLanguagesDocumentTagger(BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        preds = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        return [
            Prediction(label=label.replace("__label__", ""), score=score)
            for label, score in sorted(zip(*preds), key=lambda x: x[1], reverse=True)
        ]


@TaggerRegistry.add("ft_lang_id_paragraph_v1")
class FastTextAllLanguageParagraphTagger(FastTextAllLanguagesDocumentTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)


@TaggerRegistry.add("ft_lang_id_en_doc_v2")
class FastTextEnglishLanguageDocumentTagger(BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        for label, score in zip(*pred):
            if label == "__label__en":
                return Prediction(label="en", score=score), Prediction(label="not_en", score=1.0 - score)
        return Prediction(label="en", score=0.0), Prediction(label="not_en", score=1.0)


@TaggerRegistry.add("ft_lang_id_en_paragraph_v2")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)


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
class Cld2LanguageFilterParagraphWithDocScoreTagger(Cld2LanguageFilterParagraph):
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
