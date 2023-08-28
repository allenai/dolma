"""

Filters.

@kylel, @soldni, @lucy3

"""
from typing import Iterable, List, Tuple

try:
    import cld3

    CLD3_AVAILABLE = True
except ImportError:
    CLD3_AVAILABLE = False

import pycld2 as cld2
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
import regex
from anyascii import anyascii
import random

from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs, split_sentences

'''
Langdetect
- Document, paragraph, and sentence level taggers
- Taggers that return score for English or score for most likely language
'''

@TaggerRegistry.add("langdetect_en_doc_v2")
class LangdetectTagger(BaseTagger):
    # document-level, english / not english
    def _predict_text(self, text: str) -> Tuple[str, float]:
        random.seed(0)
        try:
            langs = detect_langs(text)
        except LangDetectException as e:
            return "en", 0.0
        if not langs: return "en", 0.0
        score = max([lang.prob for lang in langs if lang.lang == 'en'] or [0.0])
        return "en", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
        negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
        return DocResult(doc=doc, spans=[positive_span, negative_span])

@TaggerRegistry.add("langdetect_multi_doc_v2")
class LangdetectMultiTagger(LangdetectTagger):
    # doc-level, return most likely language / not that language
    def _predict_text(self, text: str) -> Tuple[str, float]:
        random.seed(0)
        try:
            langs = detect_langs(text)
        except LangDetectException as e:
            return "none", 0.0
        if not langs: return "none", 0.0
        return langs[0].lang, langs[0].prob

@TaggerRegistry.add("langdetect_en_paragraph_v2")
class LangdetectTaggerParagraph(LangdetectTagger):
    # paragraph-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("langdetect_multi_paragraph_v2")
class LangdetectMultiTaggerParagraph(LangdetectTaggerParagraph, LangdetectMultiTagger):
    # paragraph-level, return most likely language / not that language
    pass

@TaggerRegistry.add("langdetect_en_sent_v2")
class LangdetectTaggerSentence(LangdetectTagger):
    # sentence-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        sentences = split_sentences(doc.text)
        spans: List[Span] = []
        for sent in sentences:
            lang, score = self._predict_text(sent.text)  # pyright: ignore
            positive_span = Span(start=sent.start, end=sent.end, type=lang, score=score)
            negative_span = Span(start=sent.start, end=sent.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("langdetect_multi_sent_v2")
class LangdetectMultiTaggerSentence(LangdetectTaggerSentence, LangdetectMultiTagger):
    # sentence-level, return most likely language / not that language
    pass

'''
CLD3 Language ID
- Document, paragraph, and sentence level taggers
- Taggers that return score for English or score for most likely language
'''

@TaggerRegistry.add("cld3_en_doc_v2")
class Cld3LanguageTagger(BaseTagger):
    # document-level, english / not english
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

@TaggerRegistry.add("cld3_multi_doc_v2")
class Cld3MultiLanguageTagger(Cld3LanguageTagger):
    # doc-level, return most likely language / not that language
    def _predict_text(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)
        return pred.language, pred.probability

@TaggerRegistry.add("cld3_en_paragraph_v2")
class Cld3LanguageTaggerParagraph(Cld3LanguageTagger):
    # paragraph-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("cld3_multi_paragraph_v2")
class Cld3MultiLanguageTaggerParagraph(Cld3MultiLanguageTagger, Cld3LanguageTaggerParagraph):
    # paragraph-level, return most likely language / not that language
    pass

@TaggerRegistry.add("cld3_en_sent_v2")
class Cld3LanguageTaggerSentence(Cld3LanguageTagger):
    # sentence-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        sentences = split_sentences(doc.text)
        spans: List[Span] = []
        for sent in sentences:
            lang, score = self._predict_text(sent.text)  # pyright: ignore
            positive_span = Span(start=sent.start, end=sent.end, type=lang, score=score)
            negative_span = Span(start=sent.start, end=sent.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("cld3_multi_sent_v2")
class Cld3MultiLanguageTaggerSentence(Cld3MultiLanguageTagger, Cld3LanguageTaggerSentence):
    # sentence-level, return most likely language / not that language
    pass

'''
CLD2 Language ID
- Document, paragraph, and sentence-level taggers
- Taggers that return score for English or for most likely language

Note: "Filter" in class names below is synonymous to "Tagger" in other class names.
'''

@TaggerRegistry.add("cld2_en_doc_v2")
class Cld2LanguageFilter(BaseTagger):
    # doc-level, english / not english
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

@TaggerRegistry.add("cld2_multi_doc_v2")
class Cld2MultiLanguageFilter(Cld2LanguageFilter):
    # doc-level, return score of most likely language / not that language
    def _predict_text(self, text: str) -> Tuple[str, float]:
        details = []
        is_reliable = False
        for fn in (self._identity_fn, self._to_ascii_input, self._sanitize_input):
            try:
                is_reliable, _, details = cld2.detect(fn(text))
                break
            except cld2.error:
                ...
        if not details:
            return 'none', 0.0
        score = details[0][2]
        lang = details[0][1]
        return lang, score / 100.0

@TaggerRegistry.add("cld2_en_paragraph_v2")
class Cld2LanguageFilterParagraph(Cld2LanguageFilter):
    # paragraph-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("cld2_multi_paragraph_v2")
class Cld2MultiLanguageFilterParagraph(Cld2LanguageFilterParagraph, Cld2MultiLanguageFilter):
    # paragraph-level, return score of most likely language / not that language
    pass

@TaggerRegistry.add("cld2_en_sent_v2")
class Cld2LanguageFilterSentence(Cld2LanguageFilter):
    # sentence-level, english / not english
    def predict(self, doc: Document) -> DocResult:
        sentences = split_sentences(doc.text)
        spans: List[Span] = []
        for sent in sentences:
            lang, score = self._predict_text(sent.text)  # pyright: ignore
            positive_span = Span(start=sent.start, end=sent.end, type=lang, score=score)
            negative_span = Span(start=sent.start, end=sent.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)

@TaggerRegistry.add("cld2_multi_sent_v2")
class Cld2MultiLanguageFilterSentence(Cld2MultiLanguageFilter, Cld2LanguageFilterSentence):
    # sentence-level, return most likely language / not that language
    pass

'''
FastText language ID
- document-level, paragraph-level, sentence-level
- taggers that return score for English or most likely language
'''

@TaggerRegistry.add("ft_lang_id_en_doc_v2")
class FastTextEnglishLanguageDocumentTagger(BaseFastTextTagger):
    # doc-level, return score for English / not English
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        for label, score in zip(*pred):
            if label == "__label__en":
                return Prediction(label="en", score=score), Prediction(label="not_en", score=1.0 - score)
        return Prediction(label="en", score=0.0), Prediction(label="not_en", score=1.0)

@TaggerRegistry.add("ft_lang_id_multi_doc_v2")
class FastTextMultiLanguageDocumentTagger(BaseFastTextTagger):
    # doc-level, return score of most likely language
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip())
        label = pred[0][0].split("__label__")[1]
        score = pred[1][0]
        return (Prediction(label=label, score=score),)

@TaggerRegistry.add("ft_lang_id_en_paragraph_v2")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger):
    # paragraph-level, English / not English
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)

@TaggerRegistry.add("ft_lang_id_multi_paragraph_v2")
class FastTextMultiLanguageParagraphTagger(FastTextMultiLanguageDocumentTagger):
    # paragraph-level, most likely language
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)

@TaggerRegistry.add("ft_lang_id_en_sent_v2")
class FastTextEnglishLanguageSentenceTagger(FastTextEnglishLanguageDocumentTagger):
    # sentence-level, English / not English
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.SENTENCE_LEVEL_TAGGER)

@TaggerRegistry.add("ft_lang_id_multi_sent_v2")
class FastTextMultiLanguageSentenceTagger(FastTextMultiLanguageDocumentTagger):
    # sentence-level, most likely language
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.SENTENCE_LEVEL_TAGGER)

'''
Additional taggers that aggregate scores from slices to document score
- CLD2, CLD3, fasttext
'''

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
