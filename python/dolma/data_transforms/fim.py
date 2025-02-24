"""
Given a code document, perform FIM operations.
"""

from dataclasses import dataclass

from dolma.core.data_types import DocumentWithMetadata
from dolma.dolma import FillInMiddle as FillInMiddleImpl


@dataclass
class FIMConfig:
    fim_rate: float
    psm_spm_split: float
    file_separator_token: str
    fim_prefix_token: str
    fim_middle_token: str
    fim_suffix_token: str


class FillInMiddle:
    def __init__(self, fim_config: FIMConfig) -> None:
        self._impl = FillInMiddleImpl(
            fim_config.fim_rate,
            fim_config.psm_spm_split,
            fim_config.file_separator_token,
            fim_config.fim_prefix_token,
            fim_config.fim_middle_token,
            fim_config.fim_suffix_token,
        )

    def perform_on_document(self, doc: DocumentWithMetadata) -> DocumentWithMetadata:
        """Updates `text` field in-place"""
        doc.text = self.perform_on_document_text(doc.text)
        return doc

    def perform_on_document_text(self, document_text: str) -> str:
        return self._impl.perform_on_document_text(document_text)
