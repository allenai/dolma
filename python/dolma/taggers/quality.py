from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs

@TaggerRegistry.add("wikiwebbooks_doc")
class FastTextWikiWebBooksDocTagger(BaseFastTextTagger):
    MODEL_PATH = "/home/lucyl/llm_social_identities/data/filter_data/wikiwebbooks_cc.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        preds = []
        for label, score in zip(*pred):
            if label == '__label__wikiwebbooks':
                preds.append(Prediction(label='neg', score=score))
            if label == '__label__random_cc':
                preds.append(Prediction(label='pos', score=score))
        assert len(preds) == 2
        return preds[0], preds[1]
