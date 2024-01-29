import importlib
import os
import re
import string
import sys
from typing import List, Union

try:
    import blingfire

    BLINGFIRE_AVAILABLE = True
except Exception:
    BLINGFIRE_AVAILABLE = False

import nltk
import uniseg.wordbreak
from nltk.tokenize.punkt import PunktSentenceTokenizer

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


from .data_types import TextSlice
from .loggers import get_logger

sent_tokenizer = PunktSentenceTokenizer()
logger = get_logger(__name__)


def make_variable_name(name: str, remove_multiple_underscores: bool = False) -> str:
    # use underscores for any non-valid characters in variable name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    if remove_multiple_underscores:
        # replace multiple underscores with a single underscore
        name = re.sub(r"__+", "_", name)

    if name[0] in string.digits:
        raise ValueError(f"Invalid variable name {name}")

    return name


def split_words(text: str, remove_empty: bool = True) -> List[TextSlice]:
    """
    Split a string into words, as defined by the unicode standard.
    For more info, see https://www.unicode.org/reports/tr29/
    """
    text_slices: List[TextSlice] = []
    offset = 0
    for word in uniseg.wordbreak.words(text):
        if word.strip() or not remove_empty:
            text_slices.append(TextSlice(doc=text, start=offset, end=offset + len(word)))
        offset += len(word)
    return text_slices


def split_paragraphs(text: str, remove_empty: bool = True) -> List[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.

    Args:
        text (str): The text to split into paragraphs.
        remove_empty (bool): Whether to remove empty paragraphs. Defaults to True.
    """
    text_slices = [
        TextSlice(doc=text, start=match.start(), end=match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    ]
    if remove_empty:
        text_slices = [text_slice for text_slice in text_slices if text_slice.text.strip()]
    return text_slices


def split_sentences(text: str, remove_empty: bool = True) -> List[TextSlice]:
    """
    Split a string into sentences.
    """
    if text and BLINGFIRE_AVAILABLE:
        _, offsets = blingfire.text_to_sentences_and_offsets(text)  # pyright: ignore
    elif text:
        offsets = [(start, end) for start, end in sent_tokenizer.span_tokenize(text)]
    else:
        offsets = []

    if remove_empty is True:
        return [TextSlice(doc=text, start=start, end=end) for (start, end) in offsets]
    else:
        raise NotImplementedError("remove_empty=False is not implemented yet")


def import_modules(modules_path: Union[List[str], None]):
    """Import one or more user modules from either names or paths.
    Importing from path is modeled after fairseq's import_user_module function:
    github.com/facebookresearch/fairseq/blob/da8fb630880d529ab47e53381c30ddc8ad235216/fairseq/utils.py#L464

    Args:
        modules_path (Union[List[str], None]): List of module names or paths to import.
    """

    for module_path in modules_path or []:
        # try importing the module directly
        try:
            importlib.import_module(module_path)
            continue
        except ModuleNotFoundError:
            pass
        except Exception as exp:
            raise RuntimeError(f"Could not import module {module_path}: {exp}") from exp

        # if that fails, try importing the module as a path

        # check if this function has a memorization attribute; if not; create it
        # the memorization attribute is used to ensure that user modules are only imported once
        if (modules_memo := getattr(import_modules, "memo", None)) is None:
            modules_memo = set()
            import_modules.memo = modules_memo  # type: ignore

        # ensure that user modules are only imported once
        if module_path not in modules_memo:
            modules_memo.add(module_path)

            if not os.path.exists(module_path):
                raise FileNotFoundError(f"Could not find module {module_path}")

            # the format is `<module_parent>/<module_name>.py` or `<module_parent>/<module_name>`
            module_parent, module_name = os.path.split(module_path)
            module_name, _ = os.path.splitext(module_name)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            elif module_path in sys.modules[module_name].__path__:
                logger.info(f"{module_path} has already been imported.")
            else:
                raise ImportError(
                    f"Failed to import {module_path} because the corresponding module name "
                    f"({module_name}) is not globally unique. Please rename the directory to "
                    "something unique and try again."
                )
