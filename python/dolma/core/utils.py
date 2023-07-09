import re
import string
from typing import List

try:
    import blingfire

    BLINGFIRE_AVAILABLE = True
except Exception:
    BLINGFIRE_AVAILABLE = False

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


from .data_types import TextSlice

sent_tokenizer = PunktSentenceTokenizer()


def make_variable_name(name: str, remove_multiple_underscores: bool = False) -> str:
    # use underscores for any non-valid characters in variable name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    if remove_multiple_underscores:
        # replace multiple underscores with a single underscore
        name = re.sub(r"__+", "_", name)

    if name[0] in string.digits:
        raise ValueError(f"Invalid variable name {name}")

    return name


def split_paragraphs(text: str, remove_empty: bool = True) -> List[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.
    """
    text_slices = [
        TextSlice(doc=text, start=match.start(), end=match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    ]
    if remove_empty is True:
        text_slices = [text_slice for text_slice in text_slices if text_slice.text.strip()]
    return text_slices


def split_sentences(text: str, remove_empty: bool = True) -> List[TextSlice]:
    """
    Split a string into sentences.
    """
    if text and BLINGFIRE_AVAILABLE:
        _, offsets = blingfire.text_to_sentences_and_offsets(text)
    elif text:
        offsets = [(start, end) for start, end in sent_tokenizer.span_tokenize(text)]
    else:
        offsets = []

    if remove_empty is True:
        return [TextSlice(doc=text, start=start, end=end) for (start, end) in offsets]
    else:
        raise NotImplementedError("remove_empty=False is not implemented yet")
