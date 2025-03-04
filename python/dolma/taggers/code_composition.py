"""

Code Prose Composition Classifier.

This tagger classifies the composition of code and prose in a given text slice
at the document level. It uses a FastText model trained on code and prose
composition data.

Tags include information about the number of code-prose boundaries, the
composition of code and prose in the text, and the entropy of the predicted
labels.

@robertb

"""

import math
from typing import Dict, Iterable, List, Tuple

from ..core.data_types import TextSlice
from ..core.ft_tagger import BaseFastTextTagger, Prediction
from ..core.registry import TaggerRegistry


@TaggerRegistry.add("code_composition")
class CodeProseCompositionClassifier(BaseFastTextTagger):
    MODEL_PATH = "hf://techarb/code-prose-composition/code-comment-prose-model.bin"  # noqa: E501

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def calculate_entropy(self, distribution: List[float]) -> float:
        entropy = 0.0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def mean_entropy(self, list_of_distributions: List[List[float]]) -> float:
        if not list_of_distributions:
            return 0

        total_entropy = 0.0
        for dist in list_of_distributions:
            total_entropy += self.calculate_entropy(dist)
        return total_entropy / len(list_of_distributions)

    def line_label(self, line: str) -> Tuple[str, List[float]]:
        label = "other"
        probabilities = []
        if len(line) > 3:
            labels, probabilities = self.classifier.predict(line, k=-1)

            label = labels[0].lstrip("__label__")
        return label, probabilities

    def predictions(
        self,
        code_prose_boundaries: int,
        class_counts: Dict[str, int],
        prediction_distributions: Dict[str, List[List[float]]],
    ) -> Iterable[Prediction]:
        composition = {}
        for label, count in class_counts.items():
            composition[label] = round((count / sum(class_counts.values())), 2)

        out = [Prediction(label="boundaries", score=code_prose_boundaries)]

        for label in composition.keys():
            out.append(Prediction(label=f"{label}_pct", score=composition[label]))
            out.append(Prediction(label=f"{label}", score=class_counts.get(label, 0)))
            out.append(
                Prediction(
                    label=f"{label}_entropy", score=self.mean_entropy(prediction_distributions.get(label, []))
                )
            )

        return out

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        class_counts: Dict[str, int] = {}
        prediction_distributions: Dict[str, List[List[float]]] = {}
        active_class, code_prose_boundaries = None, 0

        for line in [line.strip() for line in text_slice.text.splitlines()]:
            if not line:
                continue

            label, probabilities = self.line_label(line)

            prediction_distributions.setdefault(label, []).append(probabilities)
            class_counts[label] = class_counts.get(label, 0) + 1

            if active_class in ["code", "prose"] and label in ["code", "prose"] and label != active_class:
                code_prose_boundaries += 1
            active_class = label

        return self.predictions(code_prose_boundaries, class_counts, prediction_distributions)
