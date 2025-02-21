"""
Given a code document, perform FIM operations.
"""

import random
from dataclasses import dataclass

from dolma.core.data_types import DocumentWithMetadata


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
        self._config = fim_config

    def perform_on_document(self, doc: DocumentWithMetadata) -> DocumentWithMetadata:
        """Updates `text` field in-place"""
        doc.text = self.perform_on_document_text(doc.text)
        return doc

    def perform_on_document_text(self, document_text: str) -> str:
        files = document_text.split(self._config.file_separator_token)

        new_files = []

        for file in files:
            if random.random() < self._config.fim_rate:
                # Select two random character-level positions to break at
                # anywhere in the document except the first and last chars.
                position1 = random.randint(1, len(file) - 2)
                while True:
                    position2 = random.randint(1, len(file) - 2)
                    if position1 != position2:
                        break

                if position1 > position2:
                    swap = position1
                    position1 = position2
                    position2 = swap

                if random.random() < self._config.psm_spm_split:
                    # Place in Prefix-Suffix-Middle Order
                    file_parts = [
                        self._config.fim_prefix_token,
                        file[0:position1],
                        self._config.fim_suffix_token,
                        file[position2:],
                        self._config.fim_middle_token,
                        file[position1:position2],
                    ]
                else:
                    # Place in Suffix-Prefix-Middle order
                    file_parts = [
                        self._config.fim_suffix_token,
                        file[position2:],
                        self._config.fim_prefix_token,
                        file[0:position1],
                        self._config.fim_middle_token,
                        file[position1:position2],
                    ]

                new_files.append("".join(file_parts))

            else:
                new_files.append(file)

        return self._config.file_separator_token.join(new_files)
