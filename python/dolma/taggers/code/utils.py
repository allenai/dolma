"""

Utilities for code-related taggers.

@akshitab, @soldni

"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator

import regex
import smart_open
from bs4 import BeautifulSoup  # pylint: disable=import-error
from detect_secrets.core.potential_secret import PotentialSecret
from detect_secrets.core.scan import _process_line_based_plugins
from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings, get_plugins

logger = logging.getLogger(__name__)


def scan_code(code: str) -> Generator["PotentialSecret", None, None]:
    if not get_plugins():
        logger.error("No plugins to scan with!")
        return

    has_secret = False
    for lines in [code.splitlines()]:
        for secret in _process_line_based_plugins(
            lines=list(enumerate(lines, start=1)),
            filename="code_str.yml",
        ):
            has_secret = True
            yield secret

        if has_secret:
            break


class SecretsCollectionForStringInput(SecretsCollection):
    def scan_str(self, code_str: str):
        for secret in scan_code(code_str):
            self["code_str.yml"].add(secret)


def get_secrets(code: str):
    secrets = SecretsCollectionForStringInput()
    with default_settings():
        secrets.scan_str(code)

    return secrets


@dataclass
class StarCoderRegexFilterResults:
    longest_match: int
    proportion_match: float


def regex_match(regex_string: str, text: str) -> StarCoderRegexFilterResults:
    all_matches = re.findall(regex_string, text)

    match_lengths = [len(match) for match in all_matches]
    longest_match = max(match_lengths) if match_lengths else 0
    proportion_match = sum(match_lengths) / len(text)

    return StarCoderRegexFilterResults(longest_match=longest_match, proportion_match=proportion_match)


def b64_filter(text: str) -> StarCoderRegexFilterResults:
    """
    Taken from the StarCoder2 paper.
    """
    regex = r"[a-zA-Z0-9+/\n=]{64,}"
    return regex_match(regex, text)


def hexadecimal_filter(text: str) -> StarCoderRegexFilterResults:
    """
    Taken from StarCoder2 paper.
    The escaped literal case, e.g. "\\x48\\x31\\xc0\\x50\\x68\\x2f\\x2f\\x73\\x68",
    is a bit broken, because it'll always drop the first byte in the sequence due to
    how \b is interpreted in that context.
    """
    regex = r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}"
    return regex_match(regex, text)


def unicode_filter(text: str) -> StarCoderRegexFilterResults:
    """
    Taken from the StarCoder2 paper.
    """
    regex = r"(?:\\u[0-9a-fA-F]{4}){8,}"
    return regex_match(regex, text)


def get_proportion_alphabetic_chars(text: str) -> float:
    """Calculates the proportion of characters in passed text that are alphabetic"""
    nonalpha = re.sub(r"[^A-Za-z]", "", text)
    return len(nonalpha) / len(text)


@dataclass
class LineStats:
    total_count: int
    mean_length: float
    max_length: int


def get_line_stats(text: str) -> LineStats:
    """Finds some summary stats about the lines in the passed text"""

    lines = text.split("\n")
    line_lengths = [len(line) for line in lines]

    return LineStats(
        total_count=len(lines), mean_length=sum(line_lengths) / len(lines), max_length=max(line_lengths)
    )


def filter_html(html: str) -> float:
    """Filter HTML files based on displayed text VS code ratio"""
    try:
        soup = BeautifulSoup(html, features="html.parser")
    except (TypeError, UnboundLocalError):
        return False

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()
    ratio = len(text) / len(html)

    return (ratio) * (len(text) > 100)


def get_whitespace_regex() -> regex.Pattern:
    return regex.compile(r"\w+|[^\w\s]+")


def get_ext_to_lang_mapping() -> Dict[str, str]:
    path = Path(__file__).parent / "../../data/ext_to_lang_mapping.json"
    with smart_open.open(path, "r") as f:
        return json.load(f)


def special_text_file_filter(filepath: str, lang: str) -> bool:
    if lang == "text":  # TODO: include markdown as well?
        filename = Path(filepath).stem.lower()

        if "requirement" in filename:
            return True

        if filename in {"readme", "todo", "description", "cmakelists"}:
            return True

    return False
